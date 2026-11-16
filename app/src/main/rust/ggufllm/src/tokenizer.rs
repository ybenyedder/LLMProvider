//! Pure-Rust GGUF Tokenizer supporting GPT-2 BPE (Qwen2/Qwen2.5/Qwen3/Qwen3.5/Llama-3/SmolLM),
//! SentencePiece BPE/Unigram (Llama/Mistral/Gemma), and WordPiece (Nomic-BERT).

use crate::gguf::Gguf;
use anyhow::{Context, bail};
use fancy_regex::Regex;
use std::collections::HashMap;

const QWEN35_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const GLM4_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3_RE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const FLINT_RE: [&str; 3] = [
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokKind {
    Gpt2Bpe,
    SentencePiece,
    BertWordPiece,
}

pub struct Tokenizer {
    kind: TokKind,
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    token_types: Vec<u8>,
    scores: Vec<f32>,
    /// `(left_id, right_id) -> (rank, merged_id)`
    merges: HashMap<(u32, u32), (usize, u32)>,
    pre_regexes: Vec<Regex>,
    special_tokens: Vec<(String, u32)>, // sorted descending by byte length
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    eos: Vec<u32>,
    bos: Option<u32>,
    unk: Option<u32>,
    add_bos: bool,
    add_space_prefix: bool,
}

fn gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(256);
    let mut extra = 0u32;
    for b in 0..=255u8 {
        let is_direct = matches!(b, 33..=126 | 161..=172 | 174..=255);
        let ch = if is_direct {
            b as char
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        b2c[b as usize] = ch;
        c2b.insert(ch, b);
    }
    (b2c, c2b)
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let model = g
            .meta
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt2");
        let pre = g
            .meta
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let kind = match model {
            "gpt2" => TokKind::Gpt2Bpe,
            "llama" | "gemma" | "gemma4" => TokKind::SentencePiece,
            "bert" | "wordpiece" => TokKind::BertWordPiece,
            other => bail!("unsupported tokenizer model '{other}'"),
        };

        let strs = |k: &str| -> anyhow::Result<Vec<String>> {
            g.get(k)?
                .as_arr()
                .with_context(|| format!("{k}: not an array"))?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .with_context(|| format!("{k}: not a string"))
                })
                .collect()
        };

        let tokens = strs("tokenizer.ggml.tokens")?;
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.entry(t.clone()).or_insert(i as u32);
        }

        let token_types: Vec<u8> = g
            .meta
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_arr())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_u64().unwrap_or(1) as u8)
                    .collect()
            })
            .unwrap_or_else(|| vec![1u8; tokens.len()]);

        let scores: Vec<f32> = g
            .meta
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_arr())
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_f32().unwrap_or(0.0))
                    .collect()
            })
            .unwrap_or_default();

        let mut merges = HashMap::new();
        if let Some(arr) = g.meta.get("tokenizer.ggml.merges").and_then(|v| v.as_arr()) {
            merges.reserve(arr.len());
            for (rank, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str()
                    && let Some((a, b)) = s.split_once(' ')
                {
                    let merged = format!("{a}{b}");
                    if let (Some(&ia), Some(&ib), Some(&im)) = (
                        token_to_id.get(a),
                        token_to_id.get(b),
                        token_to_id.get(&merged),
                    ) {
                        merges.insert((ia, ib), (rank, im));
                    }
                }
            }
        }

        let patterns: &[&str] = if kind == TokKind::Gpt2Bpe {
            match pre {
                "qwen35" => &[QWEN35_RE],
                "qwen2" | "" => &[QWEN2_RE],
                "llama3" | "llama-bpe" | "smollm" => &[LLAMA3_RE],
                "glm4" | "glm5" | "chatglm-bpe" => &[GLM4_RE],
                "flint" => &FLINT_RE,
                _ => &[QWEN2_RE],
            }
        } else {
            &[]
        };

        let mut pre_regexes = Vec::with_capacity(patterns.len());
        for pat in patterns {
            pre_regexes.push(
                Regex::new(pat).with_context(|| format!("compile tokenizer regex {pat}"))?,
            );
        }

        let mut special_tokens: Vec<(String, u32)> = token_types
            .iter()
            .enumerate()
            .filter(|&(i, &ty)| {
                matches!(ty, 3 | 4)
                    || (tokens[i].starts_with("<|") && tokens[i].ends_with("|>"))
            })
            .map(|(i, _)| (tokens[i].clone(), i as u32))
            .collect();
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let (byte_to_char, char_to_byte) = gpt2_byte_tables();

        let mut eos: Vec<u32> = g
            .meta
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .into_iter()
            .collect();
        if let Some(a) = g
            .meta
            .get("tokenizer.ggml.eog_token_ids")
            .and_then(|v| v.as_arr())
        {
            for v in a {
                if let Some(id) = v.as_u64().map(|x| x as u32)
                    && !eos.contains(&id)
                {
                    eos.push(id);
                }
            }
        }
        for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "</s>"] {
            if let Some(&id) = token_to_id.get(t)
                && !eos.contains(&id)
            {
                eos.push(id);
            }
        }
        if pre == "flint" {
            for t in ["<|user|>", "<|observation|>"] {
                if let Some(&id) = token_to_id.get(t)
                    && !eos.contains(&id)
                {
                    eos.push(id);
                }
            }
        }

        let bos = g
            .meta
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| token_to_id.get("[CLS]").copied());
        let unk = g
            .meta
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .or_else(|| token_to_id.get("<unk>").copied())
            .or_else(|| token_to_id.get("[UNK]").copied());
        let add_bos = g
            .meta
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(matches!(kind, TokKind::SentencePiece | TokKind::BertWordPiece));
        let add_space_prefix = g
            .meta
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.as_bool())
            .unwrap_or(kind == TokKind::SentencePiece);

        Ok(Tokenizer {
            kind,
            tokens,
            token_to_id,
            token_types,
            scores,
            merges,
            pre_regexes,
            special_tokens,
            byte_to_char,
            char_to_byte,
            eos,
            bos,
            unk,
            add_bos,
            add_space_prefix,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn eos_ids(&self) -> Vec<u32> {
        self.eos.clone()
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos && let Some(bos) = self.bos {
            out.push(bos);
        }
        if !text.is_empty() {
            if parse_special && !self.special_tokens.is_empty() {
                self.encode_with_special(text, &mut out);
            } else {
                self.encode_plain(text, &mut out);
            }
        }
        if add_special
            && self.kind == TokKind::BertWordPiece
            && let Some(&sep) = self.token_to_id.get("[SEP]")
        {
            out.push(sep);
        }
        out
    }

    fn encode_with_special(&self, text: &str, out: &mut Vec<u32>) {
        let mut pos = 0;
        let mut plain_start = 0;
        while pos < text.len() {
            let rest = &text[pos..];
            let mut matched = None;
            for (sym, id) in &self.special_tokens {
                if rest.starts_with(sym.as_str()) {
                    matched = Some((sym.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                if pos > plain_start {
                    self.encode_plain(&text[plain_start..pos], out);
                }
                out.push(id);
                pos += len;
                plain_start = pos;
            } else {
                let ch = rest.chars().next().unwrap();
                pos += ch.len_utf8();
            }
        }
        if plain_start < text.len() {
            self.encode_plain(&text[plain_start..], out);
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        match self.kind {
            TokKind::Gpt2Bpe => {
                let chunks = self.pre_tokenize(text);
                let mut cache: HashMap<String, Vec<u32>> = HashMap::new();
                for chunk in chunks {
                    if let Some(cached) = cache.get(&chunk) {
                        out.extend_from_slice(cached);
                        continue;
                    }
                    let ids = self.bpe_gpt2_chunk(&chunk);
                    out.extend_from_slice(&ids);
                    if cache.len() < 4096 {
                        cache.insert(chunk, ids);
                    }
                }
            }
            TokKind::SentencePiece => {
                let s = if self.add_space_prefix && !text.starts_with(' ') {
                    format!("\u{2581}{}", text.replace(' ', "\u{2581}"))
                } else {
                    text.replace(' ', "\u{2581}")
                };
                self.encode_spm_segment(&s, out);
            }
            TokKind::BertWordPiece => {
                self.encode_wordpiece(text, out);
            }
        }
    }

    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = vec![text.to_string()];
        for re in &self.pre_regexes {
            let mut next = Vec::new();
            for p in pieces {
                let mut last = 0;
                for m in re.find_iter(&p).flatten() {
                    if m.start() > last {
                        next.push(p[last..m.start()].to_string());
                    }
                    next.push(m.as_str().to_string());
                    last = m.end();
                }
                if last < p.len() {
                    next.push(p[last..].to_string());
                }
            }
            pieces = next;
        }
        pieces
    }

    fn bpe_gpt2_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut word: Vec<u32> = Vec::with_capacity(chunk.len());
        let mut tmp = [0u8; 4];
        for &b in chunk.as_bytes() {
            let ch = self.byte_to_char[b as usize];
            let s = ch.encode_utf8(&mut tmp);
            if let Some(&id) = self.token_to_id.get(s) {
                word.push(id);
            } else if let Some(unk) = self.unk {
                word.push(unk);
            }
        }
        if word.len() <= 1 || self.merges.is_empty() {
            return word;
        }
        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = None;
            let mut best_merged = 0u32;
            for i in 0..word.len() - 1 {
                if let Some(&(rank, merged)) = self.merges.get(&(word[i], word[i + 1]))
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_idx = Some(i);
                    best_merged = merged;
                }
            }
            let Some(idx) = best_idx else {
                break;
            };
            word[idx] = best_merged;
            word.remove(idx + 1);
            if word.len() <= 1 {
                break;
            }
        }
        word
    }

    fn encode_spm_segment(&self, s: &str, out: &mut Vec<u32>) {
        // Greedy merge if BPE merges exist, otherwise DP max-score / longest-match
        if !self.merges.is_empty() {
            let mut syms: Vec<u32> = Vec::new();
            let mut buf = [0u8; 4];
            for ch in s.chars() {
                let cs = ch.encode_utf8(&mut buf);
                if let Some(&id) = self.token_to_id.get(cs) {
                    syms.push(id);
                } else {
                    for &b in cs.as_bytes() {
                        let hex = format!("<0x{b:02X}>");
                        if let Some(&id) = self.token_to_id.get(&hex) {
                            syms.push(id);
                        } else if let Some(unk) = self.unk {
                            syms.push(unk);
                        }
                    }
                }
            }
            while syms.len() > 1 {
                let mut best_rank = usize::MAX;
                let mut best_idx = None;
                let mut best_merged = 0u32;
                for i in 0..syms.len() - 1 {
                    if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1]))
                        && rank < best_rank
                    {
                        best_rank = rank;
                        best_idx = Some(i);
                        best_merged = merged;
                    }
                }
                let Some(idx) = best_idx else { break };
                syms[idx] = best_merged;
                syms.remove(idx + 1);
            }
            out.extend_from_slice(&syms);
            return;
        }

        // Score-based Viterbi / greedy longest match over UTF-8 char boundaries
        let bytes = s.as_bytes();
        let n = bytes.len();
        let mut dp = vec![(f32::NEG_INFINITY, 0usize, 0u32); n + 1];
        dp[0] = (0.0, 0, 0);
        for i in 0..n {
            if !s.is_char_boundary(i) || dp[i].0 == f32::NEG_INFINITY {
                continue;
            }
            let max_end = (i + 48).min(n);
            for j in (i + 1)..=max_end {
                if !s.is_char_boundary(j) {
                    continue;
                }
                if let Some(&id) = self.token_to_id.get(&s[i..j]) {
                    let sc = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    let cand = dp[i].0 + sc;
                    if cand > dp[j].0 {
                        dp[j] = (cand, i, id);
                    }
                }
            }
            // Byte fallback for one char if no token matched at all
            let ch = s[i..].chars().next().unwrap();
            let next_i = i + ch.len_utf8();
            if dp[next_i].0 == f32::NEG_INFINITY {
                let fallback = self.unk.unwrap_or(0);
                dp[next_i] = (dp[i].0 - 100.0, i, fallback);
            }
        }
        let mut rev = Vec::new();
        let mut cur = n;
        while cur > 0 {
            let (_, prev, id) = dp[cur];
            if prev == cur {
                break;
            }
            rev.push(id);
            cur = prev;
        }
        rev.reverse();
        out.extend_from_slice(&rev);
    }

    fn encode_wordpiece(&self, text: &str, out: &mut Vec<u32>) {
        let lower = text.to_lowercase();
        let mut words = Vec::new();
        let mut cur = String::new();
        for ch in lower.chars() {
            if ch.is_whitespace() {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            } else if ch.is_ascii_punctuation() {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
                words.push(ch.to_string());
            } else {
                cur.push(ch);
            }
        }
        if !cur.is_empty() {
            words.push(cur);
        }
        for w in words {
            let mut start = 0;
            let mut sub_tokens = Vec::new();
            let mut bad = false;
            while start < w.len() {
                let mut end = w.len();
                let mut found = None;
                while start < end {
                    let sub = if start == 0 {
                        w[start..end].to_string()
                    } else {
                        format!("##{}", &w[start..end])
                    };
                    if let Some(&id) = self.token_to_id.get(&sub) {
                        found = Some((end, id));
                        break;
                    }
                    let last_ch = w[start..end].chars().next_back().unwrap();
                    end -= last_ch.len_utf8();
                }
                if let Some((next_start, id)) = found {
                    sub_tokens.push(id);
                    start = next_start;
                } else {
                    bad = true;
                    break;
                }
            }
            if bad {
                if let Some(unk) = self.unk {
                    out.push(unk);
                }
            } else {
                out.extend_from_slice(&sub_tokens);
            }
        }
    }

    /// Raw bytes for one token ID (preserving split UTF-8 sequences during streaming).
    pub fn token_bytes(&self, id: u32, special: bool) -> Vec<u8> {
        let idx = id as usize;
        if idx >= self.tokens.len() {
            return Vec::new();
        }
        let ty = self.token_types.get(idx).copied().unwrap_or(1);
        let tok = &self.tokens[idx];
        if ty == 3 || ty == 4 {
            return if special {
                tok.as_bytes().to_vec()
            } else {
                Vec::new()
            };
        }
        match self.kind {
            TokKind::Gpt2Bpe => {
                let mut bytes = Vec::with_capacity(tok.len());
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
                bytes
            }
            TokKind::SentencePiece => {
                if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                    if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                        return vec![b];
                    }
                }
                tok.replace('\u{2581}', " ").into_bytes()
            }
            TokKind::BertWordPiece => {
                if let Some(rest) = tok.strip_prefix("##") {
                    rest.as_bytes().to_vec()
                } else {
                    format!(" {tok}").into_bytes()
                }
            }
        }
    }

    pub fn decode_tokens(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes(id, !skip_special));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Incremental UTF-8 byte accumulator so multi-byte characters (accents, emojis) split across
/// BPE tokens are emitted only once complete, never producing replacement chars or JNI crashes.
#[derive(Default)]
pub struct Utf8StreamDecoder {
    buf: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> Option<String> {
        if bytes.is_empty() {
            return None;
        }
        self.buf.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(unsafe { std::str::from_utf8_unchecked(&self.buf[..valid]) });
                    }
                    if let Some(err_len) = e.error_len() {
                        out.push('\u{FFFD}');
                        self.buf.drain(..valid + err_len);
                    } else {
                        // Incomplete UTF-8 sequence at end; keep trailing bytes for next token.
                        if valid > 0 {
                            self.buf.drain(..valid);
                        }
                        break;
                    }
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    pub fn flush(&mut self) -> Option<String> {
        if self.buf.is_empty() {
            return None;
        }
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        Some(s)
    }
}
