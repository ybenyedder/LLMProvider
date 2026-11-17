//! Pure-Rust inference engine (derived from RustingStrata) with batched Q8_K prefill,
//! prefix KV-cache reuse, N-gram speculative verification, and multi-architecture support.

use crate::config::{LayerKind, ModelConfig};
use crate::gguf::Gguf;
use crate::ops;
use crate::quant::GgmlType;
use crate::simd;
use crate::tokenizer::Tokenizer;
use crate::weights::{Tensor, Weights};
use anyhow::Context;

pub enum LayerState {
    Recurrent { conv: Vec<f32>, s: Vec<f32> },
    Attn { k: Vec<f32>, v: Vec<f32> },
}

#[derive(Default)]
pub(crate) struct DecodeScratch {
    h: Vec<f32>,
    xq: Vec<simd::Q8K>,
    q: Vec<f32>,
    q_normed: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qkv: Vec<f32>,
    attn_out: Vec<f32>,
    proj: Vec<f32>,
    ffn_act: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    sc: Vec<f32>,
}

pub struct State {
    pub layers: Vec<LayerState>,
    pub pos: usize,
    pub rope_pos: [i32; 4],
    pub cached_tokens: Vec<u32>,
    pub(crate) scratch: DecodeScratch,
}

impl State {
    pub fn new(c: &ModelConfig) -> State {
        let channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
        let layers = c
            .layers
            .iter()
            .map(|k| match k {
                LayerKind::Delta => LayerState::Recurrent {
                    conv: vec![0.0; c.ssm_conv.saturating_sub(1) * channels],
                    s: vec![0.0; c.ssm_dt_rank * c.ssm_dv() * c.ssm_d_state],
                },
                LayerKind::Attn => LayerState::Attn {
                    k: Vec::new(),
                    v: Vec::new(),
                },
            })
            .collect();
        State {
            layers,
            pos: 0,
            rope_pos: [0; 4],
            cached_tokens: Vec::new(),
            scratch: DecodeScratch::default(),
        }
    }

    pub fn clear(&mut self) {
        self.pos = 0;
        self.rope_pos = [0; 4];
        self.cached_tokens.clear();
        for ls in &mut self.layers {
            match ls {
                LayerState::Recurrent { conv, s } => {
                    conv.fill(0.0);
                    s.fill(0.0);
                }
                LayerState::Attn { k, v } => {
                    k.clear();
                    v.clear();
                }
            }
        }
    }

    /// Truncates attention KV caches back to `new_pos` (used for prefix reuse & speculative rollback).
    pub fn truncate_attn(&mut self, new_pos: usize, n_head_kv: usize, head_dim: usize) {
        let stride = n_head_kv * head_dim;
        let keep = new_pos * stride;
        for ls in &mut self.layers {
            if let LayerState::Attn { k, v } = ls {
                if k.len() > keep {
                    k.truncate(keep);
                }
                if v.len() > keep {
                    v.truncate(keep);
                }
            }
        }
        self.pos = new_pos;
        self.rope_pos = [new_pos as i32; 4];
        if self.cached_tokens.len() > new_pos {
            self.cached_tokens.truncate(new_pos);
        }
    }
}

struct LayerNormCache {
    attn_norm: Vec<f32>,
    attn_norm_bias: Option<Vec<f32>>,
    post_attn_norm: Vec<f32>,
    post_attn_norm_bias: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    o_bias: Option<Vec<f32>>,
    qkv_bias: Option<Vec<f32>>,
    attn_sinks: Option<Vec<f32>>,
    ssm_norm: Option<Vec<f32>>,
    ssm_dt_bias: Option<Vec<f32>>,
    ssm_a: Option<Vec<f32>>,
    ssm_conv1d: Option<Vec<f32>>,
}

pub struct Model<'a> {
    pub cfg: ModelConfig,
    pub w: Weights<'a>,
    pub tok: Tokenizer,
    pub n_ctx: usize,
    pub has_delta: bool,
    pub primary_quant: &'static str,
    layer_cache: Vec<LayerNormCache>,
    output_norm: Vec<f32>,
    output_norm_bias: Option<Vec<f32>>,
    pub pool: rayon::ThreadPool,
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf, n_ctx_req: usize, n_threads: usize) -> anyhow::Result<Model<'a>> {
        let cfg = ModelConfig::from_gguf(g)?;
        let w = Weights::new(g)?;
        let tok = Tokenizer::from_gguf(g)?;
        let _ = w.validate(cfg.n_layer)?;

        // Pre-touch token_embd.weight pages to avoid page faults during decode.
        let emb_t = w.get("token_embd.weight")?;
        let mut _touch = 0u8;
        for off in (0..emb_t.data.len()).step_by(4096) {
            _touch ^= unsafe { std::ptr::read_volatile(&emb_t.data[off]) };
        }
        std::hint::black_box(_touch);

        let primary_quant = w
            .layer(0, "ffn_down.weight")
            .or_else(|_| w.layer(0, "ffn_down_exps.weight"))
            .map(|t| t.ty.name())
            .unwrap_or_else(|_| emb_t.ty.name());

        let opt_f32 = |name: &str| -> Option<Vec<f32>> {
            w.get(name).ok().map(|t| t.to_f32())
        };
        let layer_opt = |il: usize, suffix: &str| -> Option<Vec<f32>> {
            w.layer(il, suffix).ok().map(|t| t.to_f32())
        };

        let mut layer_cache = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let attn_norm = layer_opt(il, "attn_norm.weight")
                .or_else(|| layer_opt(il, "attn_output_norm.weight"))
                .with_context(|| format!("blk.{il}: missing attn_norm.weight"))?;
            let post_attn_norm = layer_opt(il, "post_attention_norm.weight")
                .or_else(|| layer_opt(il, "layer_output_norm.weight"))
                .unwrap_or_else(|| vec![1.0; cfg.n_embd]);
            layer_cache.push(LayerNormCache {
                attn_norm,
                attn_norm_bias: layer_opt(il, "attn_norm.bias")
                    .or_else(|| layer_opt(il, "attn_output_norm.bias")),
                post_attn_norm,
                post_attn_norm_bias: layer_opt(il, "post_attention_norm.bias")
                    .or_else(|| layer_opt(il, "ffn_norm.bias"))
                    .or_else(|| layer_opt(il, "layer_output_norm.bias")),
                q_norm: layer_opt(il, "attn_q_norm.weight"),
                k_norm: layer_opt(il, "attn_k_norm.weight"),
                q_bias: layer_opt(il, "attn_q.bias"),
                k_bias: layer_opt(il, "attn_k.bias"),
                v_bias: layer_opt(il, "attn_v.bias"),
                o_bias: layer_opt(il, "attn_output.bias"),
                qkv_bias: layer_opt(il, "attn_qkv.bias"),
                attn_sinks: if cfg.attn_sinks {
                    layer_opt(il, "attn_sinks.weight")
                } else {
                    None
                },
                ssm_norm: layer_opt(il, "ssm_norm.weight"),
                ssm_dt_bias: layer_opt(il, "ssm_dt.bias"),
                ssm_a: layer_opt(il, "ssm_a"),
                ssm_conv1d: layer_opt(il, "ssm_conv1d.weight"),
            });
        }

        let output_norm = opt_f32("output_norm.weight")
            .unwrap_or_else(|| vec![1.0; cfg.n_embd]);
        let output_norm_bias = opt_f32("output_norm.bias");

        let has_delta = cfg.layers.contains(&LayerKind::Delta);
        let n_ctx = if n_ctx_req > 0 {
            n_ctx_req.min(cfg.n_ctx_train.max(512))
        } else {
            cfg.n_ctx_train.min(4096)
        };

        let threads = n_threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("failed to build rayon thread pool")?;

        Ok(Model {
            cfg,
            w,
            tok,
            n_ctx,
            has_delta,
            primary_quant,
            layer_cache,
            output_norm,
            output_norm_bias,
            pool,
        })
    }

    pub fn context_length_reported(&self) -> usize {
        if self.cfg.n_ctx_train > 0 {
            self.cfg.n_ctx_train.min(self.n_ctx)
        } else {
            self.n_ctx
        }
    }

    pub fn token_embd_tensor(&self) -> anyhow::Result<Tensor<'a>> {
        self.w.get("token_embd.weight")
    }

    pub fn token_embd_type(&self) -> Option<GgmlType> {
        self.w.get("token_embd.weight").ok().map(|t| t.ty)
    }

    pub fn embed_token(&self, token: u32) -> Vec<f32> {
        let mut x = vec![0f32; self.cfg.n_embd];
        let idx = (token as usize).min(self.cfg.n_vocab.saturating_sub(1));
        self.w.get("token_embd.weight").unwrap().row(idx, &mut x);
        if self.cfg.arch == "gemma" || self.cfg.arch == "gemma2" {
            let s = (self.cfg.n_embd as f32).sqrt();
            for v in &mut x {
                *v *= s;
            }
        }
        x
    }

    pub fn embed_tokens_flat(&self, tokens: &[u32]) -> Vec<f32> {
        let d = self.cfg.n_embd;
        let t = self.w.get("token_embd.weight").unwrap();
        let mut out = vec![0f32; tokens.len() * d];
        let max_id = self.cfg.n_vocab.saturating_sub(1);
        for (i, &tok) in tokens.iter().enumerate() {
            let idx = (tok as usize).min(max_id);
            t.row(idx, &mut out[i * d..(i + 1) * d]);
        }
        out
    }

    fn t(&self, il: usize, suffix: &str) -> Tensor<'a> {
        self.w
            .layer(il, suffix)
            .unwrap_or_else(|e| panic!("{e:#}"))
    }

    #[inline(always)]
    fn norm_vec(&self, x: &[f32], w: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
        if self.cfg.uses_layer_norm {
            let mean = x.iter().sum::<f32>() / x.len() as f32;
            let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (var + self.cfg.eps).sqrt();
            match bias {
                Some(b) => {
                    for (((o, &v), &wv), &bv) in out.iter_mut().zip(x).zip(w).zip(b) {
                        *o = (v - mean) * inv * wv + bv;
                    }
                }
                None => {
                    for ((o, &v), &wv) in out.iter_mut().zip(x).zip(w) {
                        *o = (v - mean) * inv * wv;
                    }
                }
            }
        } else {
            ops::rms_norm(x, w, self.cfg.eps, out);
        }
    }

    #[inline(always)]
    fn rope_head(&self, x: &mut [f32], pos: usize, rope_pos: [i32; 4]) {
        match self.cfg.mrope_sections {
            Some(sections) => {
                ops::rope_mrope(x, self.cfg.n_rot, rope_pos, sections, self.cfg.rope_base)
            }
            None if self.cfg.rope_neox => {
                ops::rope_neox(x, self.cfg.n_rot, pos, self.cfg.rope_base)
            }
            None => ops::rope_norm(x, self.cfg.n_rot, pos, self.cfg.rope_base),
        }
    }

    /// Runs a batch of embeddings `[nt * n_embd]` through the model.
    ///
    /// - If `want_all_logits == true`, returns logits for every token in the batch (used by speculative
    ///   verification); otherwise returns logits only for the final token `[last_logits]`.
    /// - Also returns the normalized final hidden states for every token in the batch (used by `embedText` mode 1).
    pub fn step_embed_batch(
        &self,
        st: &mut State,
        xs_in: Vec<Vec<f32>>,
        want_all_logits: bool,
        compute_logits: bool,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let nt = xs_in.len();
        if nt == 0 {
            return (Vec::new(), Vec::new());
        }
        if nt == 1 && !self.has_delta && !self.cfg.uses_layer_norm {
            return self.step_one_fast(st, xs_in.into_iter().next().unwrap(), compute_logits);
        }
        let mut xs = xs_in;
        self.pool.install(|| {
            let pos0 = st.pos;
            for il in 0..self.cfg.n_layer {
                self.block_batch(il, &mut xs, &mut st.layers[il], pos0);
            }
            st.pos += nt;
            st.rope_pos = [(pos0 + nt) as i32; 4];

            let mut normed_hids = Vec::with_capacity(nt);
            for x in &xs {
                let mut h = vec![0f32; self.cfg.n_embd];
                self.norm_vec(x, &self.output_norm, self.output_norm_bias.as_deref(), &mut h);
                normed_hids.push(h);
            }

            let logits = if !compute_logits {
                Vec::new()
            } else if let Ok(head) = self
                .w
                .get("output.weight")
                .or_else(|_| self.w.get("token_embd.weight"))
            {
                if want_all_logits {
                    let refs: Vec<&[f32]> = normed_hids.iter().map(Vec::as_slice).collect();
                    simd::rows_multi(&head, &refs)
                } else {
                    vec![simd::matvec_f32(&head, normed_hids.last().unwrap())]
                }
            } else {
                Vec::new()
            };
            (logits, normed_hids)
        })
    }

    /// Zero-allocation single-token decode fast path (`nt == 1`, non-Delta, Pre-RMSNorm).
    fn step_one_fast(
        &self,
        st: &mut State,
        mut x: Vec<f32>,
        compute_logits: bool,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let c = &self.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let qs_stride = if c.q_gate { 2 * hd } else { hd };
        let q_dim = nh * qs_stride;
        let kv_dim = nkv * hd;
        let pos = st.pos;
        let rope_pos = [pos as i32; 4];
        let scale = 1.0 / (hd as f32).sqrt();
        let g_div = nh / nkv.max(1);
        let is_gemma = matches!(c.arch.as_str(), "gemma" | "gemma2" | "gemma4");

        let s = &mut st.scratch;
        if s.h.len() != c.n_embd {
            s.h.resize(c.n_embd, 0.0);
        }
        if s.q.len() != q_dim {
            s.q.resize(q_dim, 0.0);
        }
        if s.q_normed.len() != nh * hd {
            s.q_normed.resize(nh * hd, 0.0);
        }
        if s.k.len() != kv_dim {
            s.k.resize(kv_dim, 0.0);
        }
        if s.v.len() != kv_dim {
            s.v.resize(kv_dim, 0.0);
        }
        if s.attn_out.len() != nh * hd {
            s.attn_out.resize(nh * hd, 0.0);
        }
        if s.proj.len() != c.n_embd {
            s.proj.resize(c.n_embd, 0.0);
        }

        self.pool.install(|| {
            for il in 0..c.n_layer {
                let lc = &self.layer_cache[il];
                let use_rope = c.rope_layer.get(il).copied().unwrap_or(true);
                let swa_active = c.swa.get(il).copied().unwrap_or(false);

                ops::rms_norm(&x, &lc.attn_norm, c.eps, &mut s.h);
                simd::quantize_q8k_into(&s.h, &mut s.xq);

                if let Ok(qkv_w) = self.w.layer(il, "attn_qkv.weight") {
                    if s.qkv.len() != qkv_w.rows {
                        s.qkv.resize(qkv_w.rows, 0.0);
                    }
                    simd::matvec_with_q8k(&qkv_w, &s.xq, &s.h, &mut s.qkv);
                    if let Some(bias) = &lc.qkv_bias {
                        for (a, &b) in s.qkv.iter_mut().zip(bias) {
                            *a += b;
                        }
                    }
                    s.q.copy_from_slice(&s.qkv[..q_dim]);
                    s.k.copy_from_slice(&s.qkv[q_dim..q_dim + kv_dim]);
                    s.v.copy_from_slice(&s.qkv[q_dim + kv_dim..q_dim + 2 * kv_dim]);
                } else {
                    simd::matvec_qkv_with_q8k(
                        &self.t(il, "attn_q.weight"),
                        &self.t(il, "attn_k.weight"),
                        &self.t(il, "attn_v.weight"),
                        &s.xq,
                        &s.h,
                        &mut s.q,
                        &mut s.k,
                        &mut s.v,
                    );
                    if let Some(b) = &lc.q_bias {
                        for (a, &bb) in s.q.iter_mut().zip(b) {
                            *a += bb;
                        }
                    }
                    if let Some(b) = &lc.k_bias {
                        for (a, &bb) in s.k.iter_mut().zip(b) {
                            *a += bb;
                        }
                    }
                    if let Some(b) = &lc.v_bias {
                        for (a, &bb) in s.v.iter_mut().zip(b) {
                            *a += bb;
                        }
                    }
                }

                for i in 0..nh {
                    let q_src = &s.q[i * qs_stride..i * qs_stride + hd];
                    let qh = &mut s.q_normed[i * hd..(i + 1) * hd];
                    if let Some(qn) = &lc.q_norm {
                        ops::rms_norm(q_src, qn, c.eps, qh);
                    } else {
                        qh.copy_from_slice(q_src);
                    }
                    if use_rope {
                        self.rope_head(qh, pos, rope_pos);
                    }
                }

                let LayerState::Attn { k: kc, v: vc } = &mut st.layers[il] else {
                    unreachable!()
                };
                let kc_base = kc.len();
                kc.resize(kc_base + kv_dim, 0.0);
                for i in 0..nkv {
                    let kh = &mut kc[kc_base + i * hd..kc_base + (i + 1) * hd];
                    if let Some(kn) = &lc.k_norm {
                        ops::rms_norm(&s.k[i * hd..(i + 1) * hd], kn, c.eps, kh);
                    } else {
                        kh.copy_from_slice(&s.k[i * hd..(i + 1) * hd]);
                    }
                    if use_rope {
                        self.rope_head(kh, pos, rope_pos);
                    }
                }
                vc.extend_from_slice(&s.v);

                let start_t = if !c.is_causal {
                    0
                } else if swa_active && c.n_swa > 0 {
                    (pos + 1).saturating_sub(c.n_swa)
                } else {
                    0
                };
                let end_t = pos + 1;
                let span = end_t - start_t;
                s.sc.resize(span, 0.0);
                let sinks = lc.attn_sinks.as_deref();

                for i in 0..nh {
                    let g = i / g_div;
                    let qh = &s.q_normed[i * hd..(i + 1) * hd];
                    for (si, kv_pos) in (start_t..end_t).enumerate() {
                        let kt = &kc[(kv_pos * nkv + g) * hd..(kv_pos * nkv + g + 1) * hd];
                        s.sc[si] = simd::dot_f32_slice(qh, kt) * scale;
                    }
                    ops::softmax_sink(&mut s.sc[..span], sinks.and_then(|sk| sk.get(i).copied()));
                    let oh = &mut s.attn_out[i * hd..(i + 1) * hd];
                    oh.fill(0.0);
                    for (si, kv_pos) in (start_t..end_t).enumerate() {
                        let p = s.sc[si];
                        let vt = &vc[(kv_pos * nkv + g) * hd..(kv_pos * nkv + g + 1) * hd];
                        simd::fma_f32_slice(oh, vt, p);
                    }
                    if c.q_gate {
                        let gate = &s.q[i * qs_stride + hd..(i + 1) * qs_stride];
                        for (o, &gg) in oh.iter_mut().zip(gate) {
                            *o *= ops::sigmoid(gg);
                        }
                    }
                }

                simd::quantize_q8k_into(&s.attn_out, &mut s.xq);
                simd::matvec_with_q8k(
                    &self.t(il, "attn_output.weight"),
                    &s.xq,
                    &s.attn_out,
                    &mut s.proj,
                );
                if let Some(b) = &lc.o_bias {
                    for (a, &bb) in s.proj.iter_mut().zip(b) {
                        *a += bb;
                    }
                }
                for (a, &p) in x.iter_mut().zip(s.proj.iter()) {
                    *a += p;
                }

                ops::rms_norm(&x, &lc.post_attn_norm, c.eps, &mut s.h);
                if c.is_moe_layer(il) {
                    let ffns = self.moe_batch(il, &[&s.h]);
                    for (a, &f) in x.iter_mut().zip(ffns[0].iter()) {
                        *a += f;
                    }
                } else {
                    simd::quantize_q8k_into(&s.h, &mut s.xq);
                    let up_t = self.t(il, "ffn_up.weight");
                    if s.ffn_act.len() != up_t.rows {
                        s.ffn_act.resize(up_t.rows, 0.0);
                    }
                    if let Ok(gate_t) = self.w.layer(il, "ffn_gate.weight") {
                        simd::fused_gate_up_act(
                            &gate_t,
                            &up_t,
                            &s.xq,
                            &s.h,
                            &mut s.ffn_act,
                            &mut s.ffn_gate,
                            &mut s.ffn_up,
                            is_gemma,
                        );
                    } else {
                        simd::matvec_with_q8k(&up_t, &s.xq, &s.h, &mut s.ffn_act);
                        for v in &mut s.ffn_act {
                            *v = ops::gelu_tanh(*v);
                        }
                    }
                    simd::quantize_q8k_into(&s.ffn_act, &mut s.xq);
                    simd::matvec_with_q8k(
                        &self.t(il, "ffn_down.weight"),
                        &s.xq,
                        &s.ffn_act,
                        &mut s.proj,
                    );
                    for (a, &p) in x.iter_mut().zip(s.proj.iter()) {
                        *a += p;
                    }
                }
            }

            st.pos += 1;
            st.rope_pos = [(pos + 1) as i32; 4];

            ops::rms_norm(&x, &self.output_norm, c.eps, &mut s.h);
            let logits = if !compute_logits {
                Vec::new()
            } else if let Ok(head) = self
                .w
                .get("output.weight")
                .or_else(|_| self.w.get("token_embd.weight"))
            {
                simd::quantize_q8k_into(&s.h, &mut s.xq);
                let mut lg = vec![0f32; head.rows];
                simd::matvec_with_q8k(&head, &s.xq, &s.h, &mut lg);
                vec![lg]
            } else {
                Vec::new()
            };
            (logits, vec![s.h.clone()])
        })
    }

    fn block_batch(&self, il: usize, xs: &mut [Vec<f32>], ls: &mut LayerState, pos0: usize) {
        let c = &self.cfg;
        let nt = xs.len();
        let lc = &self.layer_cache[il];

        if c.uses_layer_norm {
            // Post-LN (BERT): attn(x) -> x += attn -> LN(x) -> ffn(x) -> x += ffn -> LN(x)
            let refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
            let mixs = match ls {
                LayerState::Attn { k, v } => self.attention_batch(il, &refs, k, v, pos0),
                LayerState::Recurrent { .. } => unreachable!(),
            };
            for (x, m) in xs.iter_mut().zip(mixs) {
                for (a, b) in x.iter_mut().zip(m) {
                    *a += b;
                }
                let mut tmp = vec![0f32; c.n_embd];
                self.norm_vec(x, &lc.attn_norm, lc.attn_norm_bias.as_deref(), &mut tmp);
                *x = tmp;
            }
            let refs: Vec<&[f32]> = xs.iter().map(Vec::as_slice).collect();
            let ffns = self.dense_ffn_batch(il, &refs);
            for (x, f) in xs.iter_mut().zip(ffns) {
                for (a, b) in x.iter_mut().zip(f) {
                    *a += b;
                }
                let mut tmp = vec![0f32; c.n_embd];
                self.norm_vec(
                    x,
                    &lc.post_attn_norm,
                    lc.post_attn_norm_bias.as_deref(),
                    &mut tmp,
                );
                *x = tmp;
            }
            return;
        }

        // Standard Pre-RMSNorm Transformer block
        let mut hs_norm = vec![vec![0f32; c.n_embd]; nt];
        for (x, h) in xs.iter().zip(hs_norm.iter_mut()) {
            ops::rms_norm(x, &lc.attn_norm, c.eps, h);
        }
        let h_refs: Vec<&[f32]> = hs_norm.iter().map(Vec::as_slice).collect();
        let mixs = match ls {
            LayerState::Recurrent { conv, s } => self.deltanet_batch(il, &h_refs, conv, s),
            LayerState::Attn { k, v } => self.attention_batch(il, &h_refs, k, v, pos0),
        };
        for (x, m) in xs.iter_mut().zip(mixs) {
            for (a, b) in x.iter_mut().zip(m) {
                *a += b;
            }
        }

        for (x, h) in xs.iter().zip(hs_norm.iter_mut()) {
            ops::rms_norm(x, &lc.post_attn_norm, c.eps, h);
        }
        let h_refs: Vec<&[f32]> = hs_norm.iter().map(Vec::as_slice).collect();
        let ffns = if c.is_moe_layer(il) {
            self.moe_batch(il, &h_refs)
        } else {
            self.dense_ffn_batch(il, &h_refs)
        };
        for (x, f) in xs.iter_mut().zip(ffns) {
            for (a, b) in x.iter_mut().zip(f) {
                *a += b;
            }
        }
    }

    fn attention_batch(
        &self,
        il: usize,
        hs: &[&[f32]],
        kc: &mut Vec<f32>,
        vc: &mut Vec<f32>,
        pos0: usize,
    ) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let lc = &self.layer_cache[il];
        let nt = hs.len();
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let qs_stride = if c.q_gate { 2 * hd } else { hd };
        let use_rope = c.rope_layer.get(il).copied().unwrap_or(true);

        let (mut qg_all, mut k_all, mut v_all) = if let Ok(qkv_w) = self.w.layer(il, "attn_qkv.weight") {
            let mut qkv = simd::rows_multi(&qkv_w, hs);
            if let Some(bias) = &lc.qkv_bias {
                for row in &mut qkv {
                    for (a, b) in row.iter_mut().zip(bias) {
                        *a += b;
                    }
                }
            }
            let q_dim = nh * qs_stride;
            let kv_dim = nkv * hd;
            let mut qg = Vec::with_capacity(nt);
            let mut kk = Vec::with_capacity(nt);
            let mut vv = Vec::with_capacity(nt);
            for row in qkv {
                qg.push(row[..q_dim].to_vec());
                kk.push(row[q_dim..q_dim + kv_dim].to_vec());
                vv.push(row[q_dim + kv_dim..q_dim + 2 * kv_dim].to_vec());
            }
            (qg, kk, vv)
        } else {
            let mut qg = simd::rows_multi(&self.t(il, "attn_q.weight"), hs);
            let mut kk = simd::rows_multi(&self.t(il, "attn_k.weight"), hs);
            let mut vv = simd::rows_multi(&self.t(il, "attn_v.weight"), hs);
            if let Some(b) = &lc.q_bias {
                for row in &mut qg {
                    for (x, bb) in row.iter_mut().zip(b) {
                        *x += bb;
                    }
                }
            }
            if let Some(b) = &lc.k_bias {
                for row in &mut kk {
                    for (x, bb) in row.iter_mut().zip(b) {
                        *x += bb;
                    }
                }
            }
            if let Some(b) = &lc.v_bias {
                for row in &mut vv {
                    for (x, bb) in row.iter_mut().zip(b) {
                        *x += bb;
                    }
                }
            }
            (qg, kk, vv)
        };

        let mut q_normed = vec![vec![0f32; nh * hd]; nt];
        for t in 0..nt {
            let pos = pos0 + t;
            let rope_pos = [pos as i32; 4];
            let qg = &qg_all[t];
            let q_out = &mut q_normed[t];
            for i in 0..nh {
                let q_src = &qg[i * qs_stride..i * qs_stride + hd];
                let qh = &mut q_out[i * hd..(i + 1) * hd];
                if let Some(qn) = &lc.q_norm {
                    ops::rms_norm(q_src, qn, c.eps, qh);
                } else {
                    qh.copy_from_slice(q_src);
                }
                if use_rope {
                    self.rope_head(qh, pos, rope_pos);
                }
            }

            let k_src = &mut k_all[t];
            let mut k_normed = vec![0f32; nkv * hd];
            for i in 0..nkv {
                let kh = &mut k_normed[i * hd..(i + 1) * hd];
                if let Some(kn) = &lc.k_norm {
                    ops::rms_norm(&k_src[i * hd..(i + 1) * hd], kn, c.eps, kh);
                } else {
                    kh.copy_from_slice(&k_src[i * hd..(i + 1) * hd]);
                }
                if use_rope {
                    self.rope_head(kh, pos, rope_pos);
                }
            }
            kc.extend_from_slice(&k_normed);
            vc.extend_from_slice(&v_all[t]);
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let swa_active = c.swa.get(il).copied().unwrap_or(false);
        let sinks = lc.attn_sinks.as_deref();
        let total_kv_len = pos0 + nt;
        let g_div = nh / nkv.max(1);
        let mut outs = vec![vec![0f32; nh * hd]; nt];
        let mut sc = Vec::new();

        for t in 0..nt {
            let pos = pos0 + t;
            let start_t = if !c.is_causal {
                0
            } else if swa_active && c.n_swa > 0 {
                (pos + 1).saturating_sub(c.n_swa)
            } else {
                0
            };
            let end_t = if c.is_causal { pos + 1 } else { total_kv_len };
            let span = end_t - start_t;
            sc.resize(span, 0.0);

            let q_t = &q_normed[t];
            let qg_t = &qg_all[t];
            let out_t = &mut outs[t];

            for i in 0..nh {
                let g = i / g_div;
                let qh = &q_t[i * hd..(i + 1) * hd];
                for (si, kv_pos) in (start_t..end_t).enumerate() {
                    let kt = &kc[(kv_pos * nkv + g) * hd..(kv_pos * nkv + g + 1) * hd];
                    sc[si] = simd::dot_f32_slice(qh, kt) * scale;
                }
                ops::softmax_sink(&mut sc[..span], sinks.and_then(|s| s.get(i).copied()));
                let oh = &mut out_t[i * hd..(i + 1) * hd];
                for (si, kv_pos) in (start_t..end_t).enumerate() {
                    let p = sc[si];
                    let vt = &vc[(kv_pos * nkv + g) * hd..(kv_pos * nkv + g + 1) * hd];
                    simd::fma_f32_slice(oh, vt, p);
                }
                if c.q_gate {
                    let gate = &qg_t[i * qs_stride + hd..(i + 1) * qs_stride];
                    for (o, &gg) in oh.iter_mut().zip(gate) {
                        *o *= ops::sigmoid(gg);
                    }
                }
            }
        }
        let _ = &mut qg_all;
        let _ = &mut v_all;

        let out_refs: Vec<&[f32]> = outs.iter().map(Vec::as_slice).collect();
        let mut proj = simd::rows_multi(&self.t(il, "attn_output.weight"), &out_refs);
        if let Some(b) = &lc.o_bias {
            for row in &mut proj {
                for (x, bb) in row.iter_mut().zip(b) {
                    *x += bb;
                }
            }
        }
        proj
    }

    fn deltanet_batch(
        &self,
        il: usize,
        hs: &[&[f32]],
        conv: &mut [f32],
        s: &mut [f32],
    ) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let lc = &self.layer_cache[il];
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let dv = c.ssm_dv();
        let channels = c.ssm_d_inner + 2 * nk * dk;
        let kw = c.ssm_conv;

        let qkv_all = simd::rows_multi(&self.t(il, "attn_qkv.weight"), hs);
        let z_all = simd::rows_multi(&self.t(il, "attn_gate.weight"), hs);
        let beta_all = simd::rows_multi(&self.t(il, "ssm_beta.weight"), hs);
        let alpha_all = simd::rows_multi(&self.t(il, "ssm_alpha.weight"), hs);

        let dt_bias = lc.ssm_dt_bias.as_deref().unwrap();
        let a_vec = lc.ssm_a.as_deref().unwrap();
        let norm_w = lc.ssm_norm.as_deref().unwrap();
        let conv_w = lc.ssm_conv1d.as_deref().unwrap();
        let scale = 1.0 / (dk as f32).sqrt();

        let mut outs = Vec::with_capacity(hs.len());
        let mut o = vec![0f32; dv];
        for t in 0..hs.len() {
            let qkv = &qkv_all[t];
            let z = &z_all[t];
            let beta: Vec<f32> = beta_all[t].iter().copied().map(ops::sigmoid).collect();
            let g: Vec<f32> = alpha_all[t]
                .iter()
                .enumerate()
                .map(|(i, &al)| ops::softplus(al + dt_bias[i]) * a_vec[i])
                .collect();

            let mut y = vec![0f32; channels];
            for ch in 0..channels {
                let mut acc = 0.0;
                for k in 0..kw - 1 {
                    acc += conv[k * channels + ch] * conv_w[ch * kw + k];
                }
                acc += qkv[ch] * conv_w[ch * kw + kw - 1];
                y[ch] = ops::silu(acc);
            }
            conv.copy_within(channels.., 0);
            conv[(kw - 2) * channels..].copy_from_slice(qkv);

            let (q, rest) = y.split_at_mut(nk * dk);
            let (k, v) = rest.split_at_mut(nk * dk);
            for hd in q.chunks_exact_mut(dk).chain(k.chunks_exact_mut(dk)) {
                ops::l2_norm(hd, c.eps);
            }

            let mut out = vec![0f32; nv * dv];
            for vh in 0..nv {
                let kh = if c.k_interleave {
                    vh / (nv / nk)
                } else {
                    vh % nk
                };
                let (qh, khv, vv) = (
                    &q[kh * dk..(kh + 1) * dk],
                    &k[kh * dk..(kh + 1) * dk],
                    &v[vh * dv..(vh + 1) * dv],
                );
                let sm = &mut s[vh * dv * dk..(vh + 1) * dv * dk];
                let decay = g[vh].exp();
                for x in sm.iter_mut() {
                    *x *= decay;
                }
                for j in 0..dv {
                    let row = &mut sm[j * dk..(j + 1) * dk];
                    let sk: f32 = simd::dot_f32_slice(row, khv);
                    let d = beta[vh] * (vv[j] - sk);
                    for (r, &kk) in row.iter_mut().zip(khv) {
                        *r += d * kk;
                    }
                    o[j] = simd::dot_f32_slice(row, qh) * scale;
                }
                let oh = &mut out[vh * dv..(vh + 1) * dv];
                ops::rms_norm(&o, norm_w, c.eps, oh);
                let gate_fn = if c.arch == "qwen4exp" {
                    ops::sigmoid
                } else {
                    ops::silu
                };
                for (x, &zz) in oh.iter_mut().zip(&z[vh * dv..(vh + 1) * dv]) {
                    *x *= gate_fn(zz);
                }
            }
            outs.push(out);
        }
        let out_refs: Vec<&[f32]> = outs.iter().map(Vec::as_slice).collect();
        simd::rows_multi(&self.t(il, "ssm_out.weight"), &out_refs)
    }

    fn dense_ffn_batch(&self, il: usize, hs: &[&[f32]]) -> Vec<Vec<f32>> {
        let up = simd::rows_multi(&self.t(il, "ffn_up.weight"), hs);
        let acts: Vec<Vec<f32>> = if let Ok(gate_t) = self.w.layer(il, "ffn_gate.weight") {
            let gate = simd::rows_multi(&gate_t, hs);
            let is_gemma = matches!(self.cfg.arch.as_str(), "gemma" | "gemma2" | "gemma4");
            gate.into_iter()
                .zip(up)
                .map(|(mut g, u)| {
                    for (x, uu) in g.iter_mut().zip(u) {
                        *x = if is_gemma {
                            ops::gelu_tanh(*x) * uu
                        } else {
                            ops::silu(*x) * uu
                        };
                    }
                    g
                })
                .collect()
        } else {
            // Encoder (e.g., BERT) without gate projection: GELU on ffn_up
            up.into_iter()
                .map(|mut u| {
                    for x in &mut u {
                        *x = ops::gelu_tanh(*x);
                    }
                    u
                })
                .collect()
        };
        let act_refs: Vec<&[f32]> = acts.iter().map(Vec::as_slice).collect();
        simd::rows_multi(&self.t(il, "ffn_down.weight"), &act_refs)
    }

    fn moe_batch(&self, il: usize, hs: &[&[f32]]) -> Vec<Vec<f32>> {
        let c = &self.cfg;
        let nt = hs.len();
        let router_scores = simd::rows_multi(&self.t(il, "ffn_gate_inp.weight"), hs);
        let mut sels = Vec::with_capacity(nt);
        for mut p in router_scores {
            if c.router_sigmoid {
                for v in &mut p {
                    *v = ops::sigmoid(*v);
                }
            } else {
                ops::softmax(&mut p);
            }
            let top = ops::top_k(&p, c.n_expert_used);
            let sum = top.iter().map(|&e| p[e]).sum::<f32>().max(6.103515625e-5);
            let mut scale = 1.0 / sum;
            if c.expert_weights_scale != 0.0 {
                scale *= c.expert_weights_scale;
            }
            sels.push(
                top.into_iter()
                    .map(|e| (e, p[e] * scale))
                    .collect::<Vec<_>>(),
            );
        }

        let (gate_all, up_all, down_all) = (
            self.t(il, "ffn_gate_exps.weight"),
            self.t(il, "ffn_up_exps.weight"),
            self.t(il, "ffn_down_exps.weight"),
        );
        let (n_exp, ff) = (c.n_expert, c.n_ff_exp);
        let mut ys = vec![vec![0f32; c.n_embd]; nt];
        if nt == 1 {
            let sel = &sels[0];
            if !sel.is_empty() {
                let k = sel.len();
                let gu: Vec<Tensor> = sel
                    .iter()
                    .flat_map(|&(e, _)| [gate_all.expert(e, n_exp), up_all.expert(e, n_exp)])
                    .collect();
                let mut a = vec![0f32; 2 * k * ff];
                if gate_all.ty == up_all.ty {
                    simd::batch_rows(&gu, &[hs[0]], &mut a);
                } else {
                    for (i, t) in gu.iter().enumerate() {
                        simd::matvec(t, hs[0], &mut a[i * ff..(i + 1) * ff]);
                    }
                }
                let acts: Vec<Vec<f32>> = a
                    .chunks_exact(2 * ff)
                    .map(|g| (0..ff).map(|i| ops::silu(g[i]) * g[ff + i]).collect())
                    .collect();
                let ds: Vec<Tensor> = sel.iter().map(|&(e, _)| down_all.expert(e, n_exp)).collect();
                let act_refs: Vec<&[f32]> = acts.iter().map(Vec::as_slice).collect();
                let mut o = vec![0f32; k * c.n_embd];
                simd::batch_rows(&ds, &act_refs, &mut o);
                for (oj, &(_, w)) in o.chunks_exact(c.n_embd).zip(sel) {
                    for (yy, &oo) in ys[0].iter_mut().zip(oj) {
                        *yy += w * oo;
                    }
                }
            }
        } else {
            use std::collections::BTreeMap;
            let mut uses: BTreeMap<usize, Vec<(usize, f32)>> = BTreeMap::new();
            for (ti, sel) in sels.iter().enumerate() {
                for &(e, w) in sel {
                    uses.entry(e).or_default().push((ti, w));
                }
            }
            for (&e, list) in &uses {
                let xs: Vec<&[f32]> = list.iter().map(|&(ti, _)| hs[ti]).collect();
                let g = simd::rows_multi(&gate_all.expert(e, n_exp), &xs);
                let u = simd::rows_multi(&up_all.expert(e, n_exp), &xs);
                let acts: Vec<Vec<f32>> = g
                    .into_iter()
                    .zip(u)
                    .map(|(mut gv, uv)| {
                        for (x, uu) in gv.iter_mut().zip(uv) {
                            *x = ops::silu(*x) * uu;
                        }
                        gv
                    })
                    .collect();
                let act_refs: Vec<&[f32]> = acts.iter().map(Vec::as_slice).collect();
                let outs = simd::rows_multi(&down_all.expert(e, n_exp), &act_refs);
                for (&(ti, w), o) in list.iter().zip(outs) {
                    for (yy, &oo) in ys[ti].iter_mut().zip(&o) {
                        *yy += w * oo;
                    }
                }
            }
        }

        if let (Ok(gs), Ok(us), Ok(ds)) = (
            self.w.layer(il, "ffn_gate_shexp.weight"),
            self.w.layer(il, "ffn_up_shexp.weight"),
            self.w.layer(il, "ffn_down_shexp.weight"),
        ) {
            let sg_opt = self
                .w
                .layer(il, "ffn_gate_inp_shexp.weight")
                .ok()
                .map(|sg| simd::rows_multi(&sg, hs));
            let g_sh = simd::rows_multi(&gs, hs);
            let u_sh = simd::rows_multi(&us, hs);
            let acts_sh: Vec<Vec<f32>> = g_sh
                .into_iter()
                .zip(u_sh)
                .map(|(mut g, u)| {
                    for (x, uu) in g.iter_mut().zip(u) {
                        *x = ops::silu(*x) * uu;
                    }
                    g
                })
                .collect();
            let act_refs: Vec<&[f32]> = acts_sh.iter().map(Vec::as_slice).collect();
            let o_sh = simd::rows_multi(&ds, &act_refs);
            for t in 0..nt {
                let s = sg_opt
                    .as_ref()
                    .map_or(1.0, |v| ops::sigmoid(v[t][0]));
                for (yy, &oo) in ys[t].iter_mut().zip(&o_sh[t]) {
                    *yy += s * oo;
                }
            }
        }
        ys
    }
}
