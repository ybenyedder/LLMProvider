//! Context-lookup speculative decoding (`ngram-simple` + `ngram-cache` from RustingStrata).

use std::collections::HashMap;

pub const LLAMA_TOKEN_NULL: i32 = -1;
pub const LLAMA_NGRAM_MIN: usize = 1;
pub const LLAMA_NGRAM_MAX: usize = 4;

const DRAFT_MIN_SAMPLE_SIZE_LAX: [i32; LLAMA_NGRAM_MAX] = [2, 2, 1, 1];
const DRAFT_MIN_PERCENT_LAX: [i32; LLAMA_NGRAM_MAX] = [66, 50, 50, 50];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NgramKey(pub [i32; LLAMA_NGRAM_MAX]);

pub type NgramCachePart = HashMap<u32, i32>;
pub type NgramCacheMap = HashMap<NgramKey, NgramCachePart>;

pub fn ngram_simple_draft_eos(
    tokens: &[u32],
    sampled: u32,
    size_n: usize,
    size_m: usize,
    draft_max: usize,
    eos: &[u32],
) -> Vec<u32> {
    let cur_len = tokens.len();
    let n = size_n.max(1);
    let m = size_m.max(1);
    if cur_len <= n + 1 || draft_max == 0 {
        return Vec::new();
    }
    let pat_prefix = &tokens[cur_len + 1 - n..cur_len];
    let first_tok = if n == 1 { sampled } else { pat_prefix[0] };

    let Some(start_j) = cur_len.checked_sub(n + 1) else {
        return Vec::new();
    };
    let mut match_pos = 0usize;
    let mut fallback_pos = 0usize;
    for j in (1..=start_j).rev() {
        if tokens[j + n - 1] == sampled
            && tokens[j] == first_tok
            && tokens[j..j + n - 1] == *pat_prefix
        {
            let avail = cur_len.saturating_sub(j + n);
            let take = m.min(avail).min(draft_max);
            if !eos.is_empty()
                && tokens[j + n..j + n + take]
                    .iter()
                    .any(|t| eos.contains(t))
            {
                if fallback_pos == 0 {
                    fallback_pos = j;
                }
                continue;
            }
            match_pos = j;
            break;
        }
    }
    if match_pos == 0 {
        match_pos = fallback_pos;
    }
    if match_pos == 0 {
        return Vec::new();
    }
    let avail = cur_len.saturating_sub(match_pos + n);
    let copy_max = m.min(avail);
    if copy_max < n.min(draft_max) {
        return Vec::new();
    }
    let take = copy_max.min(draft_max);
    tokens[match_pos + n..match_pos + n + take].to_vec()
}

pub fn ngram_cache_update(
    cache: &mut NgramCacheMap,
    nmin: usize,
    nmax: usize,
    inp: &[u32],
    nnew: usize,
) {
    let nmin = nmin.clamp(LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX);
    let nmax = nmax.clamp(nmin, LLAMA_NGRAM_MAX);
    if inp.len() < 2 || nnew == 0 {
        return;
    }
    let start_i = inp.len().saturating_sub(nnew);
    for ngram_size in nmin..=nmax {
        let mut key = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
        for i in start_i..inp.len() {
            if i < ngram_size {
                continue;
            }
            for k in 0..ngram_size {
                key[k] = inp[i - ngram_size + k] as i32;
            }
            let part = cache.entry(NgramKey(key)).or_default();
            let cnt = part.entry(inp[i]).or_insert(0);
            if *cnt < 16_380 {
                *cnt += 1;
            }
        }
    }
}

fn try_draft_primary(
    inp: &[u32],
    sampled: u32,
    draft_len: usize,
    nc_context: &NgramCacheMap,
) -> Option<u32> {
    let mut tokens_for_key = Vec::with_capacity(LLAMA_NGRAM_MAX);
    let keep = inp.len().min(LLAMA_NGRAM_MAX - 1);
    tokens_for_key.extend_from_slice(&inp[inp.len() - keep..]);
    tokens_for_key.push(sampled);

    for ngram_size in (LLAMA_NGRAM_MIN..=LLAMA_NGRAM_MAX.min(tokens_for_key.len())).rev() {
        let mut key = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
        let start = tokens_for_key.len() - ngram_size;
        for (k, &tok) in tokens_for_key[start..].iter().enumerate() {
            key[k] = tok as i32;
        }
        if let Some(part) = nc_context.get(&NgramKey(key)) {
            let mut n_total = 0i32;
            let mut max_tok = None;
            let mut max_cnt = 0i32;
            for (&tok, &cnt) in part {
                n_total += cnt;
                if cnt > max_cnt {
                    max_cnt = cnt;
                    max_tok = Some(tok);
                }
            }
            let idx = (ngram_size + draft_len - 1).min(LLAMA_NGRAM_MAX - 1);
            if n_total >= DRAFT_MIN_SAMPLE_SIZE_LAX[idx]
                && max_cnt * 100 >= n_total * DRAFT_MIN_PERCENT_LAX[idx]
            {
                return max_tok;
            }
        }
    }
    None
}

pub fn ngram_cache_draft(
    inp: &[u32],
    sampled: u32,
    draft_max: usize,
    nc_context: &NgramCacheMap,
) -> Vec<u32> {
    if inp.is_empty() || draft_max == 0 {
        return Vec::new();
    }
    let mut history = inp.to_vec();
    let mut cur_sampled = sampled;
    let mut drafted = Vec::with_capacity(draft_max);
    for d in 0..draft_max {
        if let Some(tok) = try_draft_primary(&history, cur_sampled, d, nc_context) {
            drafted.push(tok);
            history.push(cur_sampled);
            cur_sampled = tok;
        } else {
            break;
        }
    }
    drafted
}

pub struct NgramDrafter {
    pub draft_max: usize,
    pub history: Vec<u32>,
    pub eos: Vec<u32>,
    pub n_vocab: usize,
    pub cache_size: usize,
    pub nc_context: NgramCacheMap,
    pub short_cooldown: usize,
    pub last_draft_len: usize,
}

impl NgramDrafter {
    pub fn new(draft_max: usize, eos: Vec<u32>, n_vocab: usize) -> Self {
        Self {
            draft_max,
            history: Vec::new(),
            eos,
            n_vocab,
            cache_size: 0,
            nc_context: NgramCacheMap::new(),
            short_cooldown: 0,
            last_draft_len: 0,
        }
    }

    pub fn begin(&mut self, prompt: &[u32]) {
        self.history.clear();
        self.history.extend_from_slice(prompt);
        self.nc_context.clear();
        self.cache_size = 0;
        self.short_cooldown = 0;
        self.last_draft_len = 0;
    }

    pub fn draft(&mut self, sampled: u32) -> Vec<u32> {
        if self.draft_max == 0 {
            return Vec::new();
        }
        let mut r = ngram_simple_draft_eos(
            &self.history,
            sampled,
            8,
            48,
            self.draft_max,
            &self.eos,
        );
        if r.is_empty() && self.short_cooldown == 0 {
            r = ngram_simple_draft_eos(
                &self.history,
                sampled,
                4,
                48,
                self.draft_max,
                &self.eos,
            );
        }
        if r.is_empty() {
            let mut inp = self.history.clone();
            inp.push(sampled);
            if self.cache_size < inp.len() {
                let nnew = inp.len() - self.cache_size;
                ngram_cache_update(
                    &mut self.nc_context,
                    LLAMA_NGRAM_MIN,
                    LLAMA_NGRAM_MAX,
                    &inp,
                    nnew,
                );
                self.cache_size = inp.len();
            }
            if self.short_cooldown == 0 {
                r = ngram_cache_draft(&self.history, sampled, self.draft_max.min(4), &self.nc_context);
            }
        }
        if let Some(pos) = r.iter().position(|t| self.eos.contains(t)) {
            r.truncate(pos + 1);
        }
        if self.n_vocab > 0
            && let Some(pos) = r.iter().position(|&t| (t as usize) >= self.n_vocab)
        {
            r.truncate(pos);
        }
        self.last_draft_len = r.len();
        r
    }

    pub fn accept(&mut self, accepted_tokens: &[u32]) {
        let n_accepted_drafts = accepted_tokens.len().saturating_sub(1);
        if self.short_cooldown > 0 {
            self.short_cooldown -= 1;
        }
        if self.last_draft_len > 0 && n_accepted_drafts == 0 {
            self.short_cooldown = 8;
        }
        self.history.extend_from_slice(accepted_tokens);
    }
}
