pub mod config;
pub mod gguf;
pub mod jni_bridge;
pub mod jni_sys;
pub mod model;
pub mod ngram;
pub mod ops;
pub mod quant;
pub mod simd;
pub mod tokenizer;
pub mod weights;

#[cfg(test)]
mod tests {
    use crate::jni_bridge::{LlmEngine, prefill_embeddings, prefill_tokens, run_decode_loop};
    use crate::tokenizer::Utf8StreamDecoder;
    use std::path::PathBuf;

    fn w_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn add_u32(b: &mut Vec<u8>, nkv: &mut u64, k: &str, v: u32) {
        w_str(b, k);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&v.to_le_bytes());
        *nkv += 1;
    }

    fn add_f32(b: &mut Vec<u8>, nkv: &mut u64, k: &str, v: f32) {
        w_str(b, k);
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&v.to_le_bytes());
        *nkv += 1;
    }

    fn add_str(b: &mut Vec<u8>, nkv: &mut u64, k: &str, v: &str) {
        w_str(b, k);
        b.extend_from_slice(&8u32.to_le_bytes());
        w_str(b, v);
        *nkv += 1;
    }

    fn add_str_arr(b: &mut Vec<u8>, nkv: &mut u64, k: &str, items: &[&str]) {
        w_str(b, k);
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for s in items {
            w_str(b, s);
        }
        *nkv += 1;
    }

    fn add_u32_arr(b: &mut Vec<u8>, nkv: &mut u64, k: &str, items: &[u32]) {
        w_str(b, k);
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for &v in items {
            b.extend_from_slice(&v.to_le_bytes());
        }
        *nkv += 1;
    }

    fn push_f32_tensor(
        hdr: &mut Vec<u8>,
        body: &mut Vec<u8>,
        off: &mut u64,
        name: &str,
        dims: &[u64],
        vals: &[f32],
    ) {
        w_str(hdr, name);
        hdr.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for &d in dims {
            hdr.extend_from_slice(&d.to_le_bytes());
        }
        hdr.extend_from_slice(&0u32.to_le_bytes()); // F32
        hdr.extend_from_slice(&(*off).to_le_bytes());
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        body.extend_from_slice(&data);
        let pad = (32 - data.len() % 32) % 32;
        body.extend(std::iter::repeat_n(0u8, pad));
        *off += (data.len() + pad) as u64;
    }

    fn build_synthetic_qwen2_gguf(tag: &str) -> PathBuf {
        // Tiny deterministic Qwen2 GGUF: n_embd=8, n_head=2, n_head_kv=1, head_dim=4, n_ff=16, 1 layer, vocab=12.
        let vocab_tokens = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "user",
            "assistant",
            "\n",
            "Hello",
            "Ġworld",
            "!",
            "H",
            "e",
            "l",
        ];
        let token_types = [3u32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let merges = ["H e", "He l"];

        let mut kvs = Vec::new();
        let mut nkv = 0u64;
        add_str(&mut kvs, &mut nkv, "general.architecture", "qwen2");
        add_u32(&mut kvs, &mut nkv, "qwen2.block_count", 1);
        add_u32(&mut kvs, &mut nkv, "qwen2.context_length", 512);
        add_u32(&mut kvs, &mut nkv, "qwen2.embedding_length", 8);
        add_u32(&mut kvs, &mut nkv, "qwen2.feed_forward_length", 16);
        add_u32(&mut kvs, &mut nkv, "qwen2.attention.head_count", 2);
        add_u32(&mut kvs, &mut nkv, "qwen2.attention.head_count_kv", 1);
        add_f32(&mut kvs, &mut nkv, "qwen2.attention.layer_norm_rms_epsilon", 1e-6);
        add_f32(&mut kvs, &mut nkv, "qwen2.rope.freq_base", 1_000_000.0);
        add_str(&mut kvs, &mut nkv, "tokenizer.ggml.model", "gpt2");
        add_str(&mut kvs, &mut nkv, "tokenizer.ggml.pre", "qwen2");
        add_str_arr(&mut kvs, &mut nkv, "tokenizer.ggml.tokens", &vocab_tokens);
        add_u32_arr(&mut kvs, &mut nkv, "tokenizer.ggml.token_type", &token_types);
        add_str_arr(&mut kvs, &mut nkv, "tokenizer.ggml.merges", &merges);
        add_u32(&mut kvs, &mut nkv, "tokenizer.ggml.eos_token_id", 2);

        let det = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| (((i * 17 + seed * 31) % 97) as f32 - 48.0) * 0.02)
                .collect()
        };
        let ones = |n: usize| -> Vec<f32> { vec![1.0f32; n] };
        let zeros = |n: usize| -> Vec<f32> { vec![0.0f32; n] };

        let mut hdr = Vec::new();
        let mut body = Vec::new();
        let mut off = 0u64;
        let mut nt = 0u64;

        let mut add_t = |name: &str, dims: &[u64], vals: Vec<f32>| {
            push_f32_tensor(&mut hdr, &mut body, &mut off, name, dims, &vals);
            nt += 1;
        };

        add_t("token_embd.weight", &[8, 12], det(8 * 12, 1));
        add_t("output_norm.weight", &[8], ones(8));
        add_t("blk.0.attn_norm.weight", &[8], ones(8));
        add_t("blk.0.attn_q.weight", &[8, 8], det(8 * 8, 2));
        add_t("blk.0.attn_q.bias", &[8], zeros(8));
        add_t("blk.0.attn_k.weight", &[8, 4], det(8 * 4, 3));
        add_t("blk.0.attn_k.bias", &[4], zeros(4));
        add_t("blk.0.attn_v.weight", &[8, 4], det(8 * 4, 4));
        add_t("blk.0.attn_v.bias", &[4], zeros(4));
        add_t("blk.0.attn_output.weight", &[8, 8], det(8 * 8, 5));
        add_t("blk.0.ffn_norm.weight", &[8], ones(8));
        add_t("blk.0.ffn_gate.weight", &[8, 16], det(8 * 16, 6));
        add_t("blk.0.ffn_up.weight", &[8, 16], det(8 * 16, 7));
        add_t("blk.0.ffn_down.weight", &[16, 8], det(16 * 8, 8));

        let mut gguf = Vec::new();
        gguf.extend_from_slice(b"GGUF");
        gguf.extend_from_slice(&3u32.to_le_bytes());
        gguf.extend_from_slice(&nt.to_le_bytes());
        gguf.extend_from_slice(&nkv.to_le_bytes());
        gguf.extend_from_slice(&kvs);
        gguf.extend_from_slice(&hdr);
        while gguf.len() % 32 != 0 {
            gguf.push(0);
        }
        gguf.extend_from_slice(&body);

        let p = std::env::temp_dir().join(format!("ggufllm-test-{}-{tag}.gguf", std::process::id()));
        std::fs::write(&p, &gguf).unwrap();
        p
    }

    #[test]
    fn utf8_stream_decoder_handles_split_emoji() {
        let emoji = "🌲";
        let bytes = emoji.as_bytes();
        let mut dec = Utf8StreamDecoder::default();
        assert_eq!(dec.push(&bytes[..2]), None);
        assert_eq!(dec.push(&bytes[2..]), Some("🌲".to_string()));
    }

    #[test]
    fn qwen2_text_vs_embeddings_and_spec_and_prefix_parity() {
        let p = build_synthetic_qwen2_gguf("parity");
        let engine = LlmEngine::open(p.to_str().unwrap(), 128, 2).unwrap();
        let _ = std::fs::remove_file(&p);

        assert_eq!(engine.n_embd, 8);
        assert_eq!(engine.n_ctx_reported, 128);

        let prompt = "<|im_start|>user\nHello world!<|im_end|>\n<|im_start|>assistant\n";
        let tokens;
        let text_out_spec = {
            let mut g = engine.inner.lock().unwrap();
            tokens = g.model.tok.encode(prompt, true, true);
            assert!(tokens.len() >= 5, "unexpected token count: {tokens:?}");
            g.runtime.spec_decoding = true;
            g.runtime.prefix_cache = true;
            let lg = prefill_tokens(&engine, &mut g, &tokens).unwrap();
            let mut out = String::new();
            run_decode_loop(&engine, &mut g, lg, 12, 0.0, 40, 0.95, &tokens, |piece| {
                out.push_str(piece);
                true
            })
            .unwrap();
            out
        };

        // Run again with prefix_cache hitting + speculative decoding off -> MUST match bit-for-bit!
        let text_out_plain = {
            let mut g = engine.inner.lock().unwrap();
            g.runtime.spec_decoding = false;
            let lg = prefill_tokens(&engine, &mut g, &tokens).unwrap();
            let mut out = String::new();
            run_decode_loop(&engine, &mut g, lg, 12, 0.0, 40, 0.95, &tokens, |piece| {
                out.push_str(piece);
                true
            })
            .unwrap();
            out
        };
        assert_eq!(text_out_spec, text_out_plain);

        // Run via embedTokenRows + generateFromEmbeddings -> MUST match text path bit-for-bit!
        let emb_out = {
            let mut g = engine.inner.lock().unwrap();
            let flat = g.model.embed_tokens_flat(&tokens);
            let lg = prefill_embeddings(&engine, &mut g, &flat, tokens.len(), &[]).unwrap();
            let mut out = String::new();
            run_decode_loop(&engine, &mut g, lg, 12, 0.0, 40, 0.95, &[], |piece| {
                out.push_str(piece);
                true
            })
            .unwrap();
            out
        };
        assert_eq!(text_out_plain, emb_out);
    }
}
