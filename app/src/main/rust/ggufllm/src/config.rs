//! Model hyperparameters read from GGUF metadata, plus the per-architecture table.

use crate::gguf::Gguf;
use anyhow::{Context, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Delta,
    Attn,
}

#[derive(Clone, Debug)]
pub struct Eagle3Config {
    pub extract_layers: [usize; 3],
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub eps: f32,
    pub draft_vocab: usize,
    pub norm_before_residual: bool,
    pub rope_neox: bool,
    pub d2t: Vec<i32>,
}

struct Arch {
    hybrid: bool,
    q_gate: bool,
    k_interleave: bool,
    rope_neox: bool,
    is_causal: bool,
    uses_layer_norm: bool,
    aliases: &'static [(&'static str, &'static str)],
}

fn arch(name: &str) -> Option<Arch> {
    const FFN_NORM: &[(&str, &str)] = &[("post_attention_norm.weight", "ffn_norm.weight")];
    Some(match name {
        "qwen35" | "qwen35moe" => Arch {
            hybrid: true,
            q_gate: true,
            k_interleave: false,
            rope_neox: true,
            is_causal: true,
            uses_layer_norm: false,
            aliases: &[],
        },
        "qwen4exp" => Arch {
            hybrid: true,
            q_gate: true,
            k_interleave: false,
            rope_neox: true,
            is_causal: true,
            uses_layer_norm: false,
            aliases: &[],
        },
        "qwen3next" => Arch {
            hybrid: true,
            q_gate: true,
            k_interleave: true,
            rope_neox: true,
            is_causal: true,
            uses_layer_norm: false,
            aliases: &[],
        },
        "flint" | "qwen3" | "qwen3moe" | "qwen2" | "qwen2moe" | "gemma" | "gemma2" | "gemma4"
        | "phi3" => Arch {
            hybrid: false,
            q_gate: false,
            k_interleave: false,
            rope_neox: true,
            is_causal: true,
            uses_layer_norm: false,
            aliases: FFN_NORM,
        },
        "llama" | "mistral" | "smollm" => Arch {
            hybrid: false,
            q_gate: false,
            k_interleave: false,
            rope_neox: false,
            is_causal: true,
            uses_layer_norm: false,
            aliases: FFN_NORM,
        },
        "nomic-bert" => Arch {
            hybrid: false,
            q_gate: false,
            k_interleave: false,
            rope_neox: true,
            is_causal: false,
            uses_layer_norm: true,
            aliases: FFN_NORM,
        },
        _ => return None,
    })
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layer: usize,
    pub n_nextn: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub n_ctx_train: usize,
    pub mrope_sections: Option<[usize; 4]>,
    pub rope_base: f32,
    pub rope_neox: bool,
    pub eps: f32,
    pub n_vocab: usize,
    pub n_ff: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub n_ff_shexp: usize,
    pub n_dense: usize,
    pub expert_weights_scale: f32,
    pub router_sigmoid: bool,
    pub n_swa: usize,
    pub swa: Vec<bool>,
    pub rope_layer: Vec<bool>,
    pub attn_sinks: bool,
    pub ssm_d_inner: usize,
    pub ssm_d_state: usize,
    pub ssm_n_group: usize,
    pub ssm_dt_rank: usize,
    pub ssm_conv: usize,
    pub hc_n: usize,
    pub hc_rank: usize,
    pub n_embd_ple: usize,
    pub idx_n_head: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    pub layers: Vec<LayerKind>,
    pub q_gate: bool,
    pub k_interleave: bool,
    pub is_causal: bool,
    pub uses_layer_norm: bool,
    pub aliases: &'static [(&'static str, &'static str)],
    pub eagle3: Option<Eagle3Config>,
}

impl ModelConfig {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<ModelConfig> {
        let arch = g
            .get("general.architecture")?
            .as_str()
            .context("general.architecture")?
            .to_string();
        let Some(a) = self::arch(&arch) else {
            bail!("unsupported architecture '{arch}'")
        };
        let opt = |k: &str| {
            g.meta
                .get(&format!("{arch}.{k}"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize
        };
        let req = |k: &str| -> anyhow::Result<usize> {
            let key = format!("{arch}.{k}");
            Ok(g.get(&key)?
                .as_u64()
                .with_context(|| format!("{key}: not an integer"))? as usize)
        };
        let f = |k: &str| {
            g.meta
                .get(&format!("{arch}.{k}"))
                .and_then(|v| v.as_f32())
        };
        let n_nextn = opt("nextn_predict_layers");
        let n_layer = req("block_count")?.saturating_sub(n_nextn);
        let layers: Vec<LayerKind> = if a.hybrid {
            let every = match opt("full_attention_interval") {
                0 => 4,
                v => v,
            };
            (0..n_layer)
                .map(|il| {
                    if (il + 1) % every == 0 {
                        LayerKind::Attn
                    } else {
                        LayerKind::Delta
                    }
                })
                .collect()
        } else {
            vec![LayerKind::Attn; n_layer]
        };
        let has_delta = layers.contains(&LayerKind::Delta);
        let ssm = |k: &str| if has_delta { req(k) } else { Ok(opt(k)) };
        let (n_embd, n_head) = (req("embedding_length")?, req("attention.head_count")?);
        let n_head_kv = match opt("attention.head_count_kv") {
            0 => n_head,
            v => v,
        };
        let head_dim = match opt("attention.key_length") {
            0 => n_embd / n_head.max(1),
            v => v,
        };
        let n_ctx_train = match opt("context_length") {
            0 => 2048,
            v => v,
        };
        let n_expert = opt("expert_count");
        anyhow::ensure!(
            n_expert <= 512,
            "{n_expert} experts (at most 512 supported)"
        );
        let n_dense = opt("leading_dense_block_count");
        anyhow::ensure!(
            n_dense <= n_layer,
            "{n_dense} leading dense blocks for {n_layer} layers"
        );
        let flags = |k: &str| -> anyhow::Result<Option<Vec<bool>>> {
            let key = format!("{arch}.{k}");
            let Some(v) = g.meta.get(&key) else {
                return Ok(None);
            };
            let arr = v
                .as_arr()
                .with_context(|| format!("{key}: not an array"))?;
            anyhow::ensure!(
                arr.len() >= n_layer,
                "{key} has {} entries for {n_layer} layers",
                arr.len()
            );
            Ok(Some(
                arr[..n_layer]
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) != 0)
                    .collect(),
            ))
        };
        let swa = flags("attention.sliding_window_pattern")?
            .unwrap_or_else(|| vec![false; n_layer]);
        let rope_layer = flags("attention.rope_pattern")?
            .unwrap_or_else(|| vec![true; n_layer]);
        let n_swa = opt("attention.sliding_window");
        anyhow::ensure!(
            !swa.contains(&true) || n_swa > 0,
            "sliding-window layers without attention.sliding_window"
        );
        let emb = g
            .tensors
            .get("token_embd.weight")
            .context("missing tensor token_embd.weight")?;
        let eps = f("attention.layer_norm_rms_epsilon")
            .or_else(|| f("attention.layer_norm_epsilon"))
            .unwrap_or(1e-6);
        let c = ModelConfig {
            n_layer,
            n_nextn,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_rot: match opt("rope.dimension_count") {
                0 => head_dim,
                v => v,
            },
            n_ctx_train,
            mrope_sections: g
                .meta
                .get(&format!("{arch}.rope.dimension_sections"))
                .and_then(|v| v.as_arr())
                .and_then(|arr| {
                    if arr.len() < 4 {
                        return None;
                    }
                    let mut s = [0usize; 4];
                    for (o, v) in s.iter_mut().zip(arr) {
                        *o = v.as_u64()? as usize;
                    }
                    Some(s)
                }),
            rope_base: f("rope.freq_base").unwrap_or(10000.0),
            rope_neox: a.rope_neox,
            eps,
            n_vocab: emb.dims.get(1).copied().unwrap_or(0) as usize,
            n_ff: opt("feed_forward_length"),
            n_expert,
            n_expert_used: opt("expert_used_count"),
            n_ff_exp: opt("expert_feed_forward_length"),
            n_ff_shexp: opt("expert_shared_feed_forward_length"),
            n_dense,
            expert_weights_scale: f("expert_weights_scale").unwrap_or(0.0),
            router_sigmoid: arch == "flint" && opt("expert_gating_func") == 2,
            n_swa,
            swa,
            rope_layer,
            attn_sinks: g.tensors.contains_key("blk.0.attn_sinks.weight"),
            ssm_d_inner: ssm("ssm.inner_size")?,
            ssm_d_state: ssm("ssm.state_size")?,
            ssm_n_group: ssm("ssm.group_count")?,
            ssm_dt_rank: ssm("ssm.time_step_rank")?,
            ssm_conv: ssm("ssm.conv_kernel")?,
            hc_n: opt("hyper_connection.count"),
            hc_rank: opt("hyper_connection.low_rank"),
            n_embd_ple: opt("embedding_length_per_layer_input"),
            idx_n_head: opt("attention.indexer.head_count"),
            idx_head_dim: opt("attention.indexer.key_length"),
            idx_top_k: opt("attention.indexer.top_k"),
            layers,
            q_gate: a.q_gate,
            k_interleave: a.k_interleave,
            is_causal: a.is_causal,
            uses_layer_norm: a.uses_layer_norm,
            aliases: a.aliases,
            arch,
            eagle3: None,
        };
        let (nk, nv) = (c.ssm_n_group, c.ssm_dt_rank);
        anyhow::ensure!(
            !has_delta || (nk > 0 && nv > 0 && nv % nk == 0 && c.ssm_d_inner % nv == 0),
            "bad ssm dims: group_count {nk}, time_step_rank {nv}, inner_size {}",
            c.ssm_d_inner
        );
        Ok(c)
    }

    pub fn is_moe(&self) -> bool {
        self.n_expert > 0
    }

    pub fn is_moe_layer(&self, il: usize) -> bool {
        self.n_expert > 0 && il >= self.n_dense
    }

    pub fn ssm_dv(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank.max(1)
    }
}
