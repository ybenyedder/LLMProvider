//! Named tensor views over the mmapped GGUF.

use crate::config::ModelConfig;
use crate::gguf::Gguf;
use crate::quant::{GgmlType, dequant_row};
use anyhow::Context;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    pub ty: GgmlType,
    /// Elements per row (GGUF dims[0]).
    pub cols: usize,
    /// Product of the remaining dims; for 3-D expert tensors this is out_rows * n_expert.
    pub rows: usize,
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    #[inline(always)]
    pub fn row_bytes(&self) -> usize {
        self.ty.row_bytes(self.cols as u64)
    }

    #[inline(always)]
    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// Rows belonging to expert `e` of an [cols, out, n_expert] tensor.
    pub fn expert(&self, e: usize, n_expert: usize) -> Tensor<'a> {
        let rows = self.rows / n_expert;
        let rb = self.row_bytes();
        Tensor {
            rows,
            data: &self.data[e * rows * rb..(e + 1) * rows * rb],
            ..*self
        }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for (r, out) in v.chunks_exact_mut(self.cols).enumerate() {
            self.row(r, out);
        }
        v
    }
}

pub struct Weights<'a> {
    pub g: &'a Gguf,
    pub aux_g: Option<&'a Gguf>,
    aliases: &'static [(&'static str, &'static str)],
    pub built: HashMap<String, Tensor<'a>>,
    _owned_ssm: Vec<Box<[u8]>>,
}

impl<'a> Weights<'a> {
    pub fn new(g: &'a Gguf) -> anyhow::Result<Weights<'a>> {
        Self::new_with_aux(g, None)
    }

    pub fn new_with_aux(g: &'a Gguf, aux_g: Option<&'a Gguf>) -> anyhow::Result<Weights<'a>> {
        let c = ModelConfig::from_gguf(g)?;
        let mut w = Weights {
            g,
            aux_g,
            aliases: c.aliases,
            built: HashMap::new(),
            _owned_ssm: Vec::new(),
        };
        for il in 0..c.n_layer {
            let Ok(ba) = w.layer(il, "ssm_ba.weight") else {
                continue;
            };
            let (nk, nv, rb) = (c.ssm_n_group, c.ssm_dt_rank, ba.row_bytes());
            anyhow::ensure!(
                nk > 0
                    && nv > 0
                    && nv % nk == 0
                    && ba.rows == 2 * nv
                    && ba.data.len() == ba.rows * rb,
                "blk.{il}.ssm_ba: {} rows, {} bytes for group_count {nk}, time_step_rank {nv}",
                ba.rows,
                ba.data.len()
            );
            let r = nv / nk;
            let (mut b, mut a) = (
                Vec::with_capacity(ba.data.len() / 2),
                Vec::with_capacity(ba.data.len() / 2),
            );
            for grp in ba.data.chunks_exact(2 * r * rb) {
                b.extend_from_slice(&grp[..r * rb]);
                a.extend_from_slice(&grp[r * rb..]);
            }
            let half = Tensor {
                rows: ba.rows / 2,
                ..ba
            };
            let b_box = b.into_boxed_slice();
            let a_box = a.into_boxed_slice();
            let b_slice: &'a [u8] =
                unsafe { std::slice::from_raw_parts(b_box.as_ptr(), b_box.len()) };
            let a_slice: &'a [u8] =
                unsafe { std::slice::from_raw_parts(a_box.as_ptr(), a_box.len()) };
            w._owned_ssm.push(b_box);
            w._owned_ssm.push(a_box);
            w.built
                .insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: b_slice, ..half });
            w.built
                .insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: a_slice, ..half });
        }
        Ok(w)
    }

    pub fn get(&self, name: &str) -> anyhow::Result<Tensor<'a>> {
        if let Some(t) = self.built.get(name) {
            return Ok(*t);
        }
        let find_t = |g: &'a Gguf| -> Option<(&'a crate::gguf::TensorInfo, &'a Gguf)> {
            if let Some(t) = g.tensors.get(name) {
                return Some((t, g));
            }
            if let Some(t) = self.aliases.iter().find_map(|&(c, a)| {
                g.tensors
                    .get(&format!("{}{a}", name.strip_suffix(c)?))
            }) {
                return Some((t, g));
            }
            None
        };
        let (t, owner_g) = find_t(self.g)
            .or_else(|| self.aux_g.and_then(find_t))
            .with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        Ok(Tensor {
            ty,
            cols: t.dims[0] as usize,
            rows: t.dims[1..].iter().product::<u64>() as usize,
            data: owner_g.bytes(t),
        })
    }

    pub fn layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'a>> {
        self.get(&format!("blk.{il}.{suffix}"))
    }

    pub fn validate(&self, n_layer: usize) -> anyhow::Result<bool> {
        let mut nextn = true;
        let iter = self
            .g
            .tensors
            .values()
            .chain(self.aux_g.into_iter().flat_map(|ag| ag.tensors.values()));
        for t in iter {
            if matches!(t.name.as_str(), "d2t" | "t2d") {
                continue;
            }
            if let Err(e) = GgmlType::from_u32(t.ty) {
                let il = t
                    .name
                    .strip_prefix("blk.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|i| i.parse::<usize>().ok());
                if il.is_some_and(|il| il >= n_layer) {
                    nextn = false;
                    continue;
                }
                return Err(e).with_context(|| format!("tensor {}", t.name));
            }
        }
        Ok(nextn)
    }
}
