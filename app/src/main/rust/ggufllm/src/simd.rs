//! Quantized `Q8_K` integer dot products + ARM64 NEON / x86_64 AVX2 accelerated matvec & batched matmul.
//! Derived from RustingStrata (`cpu_moe.rs`).

use crate::quant::{GgmlType, dequant_row, tables::*};
use crate::weights::Tensor;
use half::{bf16, f16};
use rayon::prelude::*;

pub struct Q8K {
    pub d: f32,
    pub qs: [i8; 256],
    pub bs: [i16; 16],
}

static SIGNS: [[i8; 8]; 128] = {
    let mut t = [[1i8; 8]; 128];
    let mut i = 0;
    while i < 128 {
        let mut j = 0;
        while j < 8 {
            if KSIGNS_IQ2XS[i] & (1 << j) != 0 {
                t[i][j] = -1;
            }
            j += 1;
        }
        i += 1;
    }
    t
};

/// Returns the active SIMD backend label for diagnostics.
pub fn simd_backend_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return "ARM64-NEON-DOTPROD";
        }
        return "ARM64-NEON";
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return "x86_64-AVX2-FMA";
        }
        return "x86_64-SSE2";
    }
    #[allow(unreachable_code)]
    "Scalar-Q8K"
}

pub fn quantize_q8k_into(x: &[f32], out: &mut Vec<Q8K>) {
    out.clear();
    let n_blocks = x.len().div_ceil(256);
    if out.capacity() < n_blocks {
        out.reserve(n_blocks - out.capacity());
    }
    for b in x.chunks(256) {
        let mut pad = [0f32; 256];
        let b = if b.len() == 256 {
            b
        } else {
            pad[..b.len()].copy_from_slice(b);
            &pad[..]
        };
        let amax = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            out.push(Q8K {
                d: 0.0,
                qs: [0; 256],
                bs: [0; 16],
            });
            continue;
        }
        let max = *b.iter().find(|v| v.abs() == amax).unwrap();
        let is = -127.0 / max;
        let mut qs = [0i8; 256];

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe { avx2::quant_block(is, b, &mut qs) };
            } else {
                for (q, &v) in qs.iter_mut().zip(b) {
                    *q = ((is * v).round_ties_even() as i32).clamp(-127, 127) as i8;
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { neon::quant_block(is, b, &mut qs) };
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for (q, &v) in qs.iter_mut().zip(b) {
                *q = ((is * v).round_ties_even() as i32).clamp(-127, 127) as i8;
            }
        }

        let mut bs = [0i16; 16];
        for (s, c) in bs.iter_mut().zip(qs.chunks_exact(16)) {
            *s = c.iter().map(|&v| v as i16).sum();
        }
        out.push(Q8K { d: 1.0 / is, qs, bs });
    }
}

pub fn quantize_q8k(x: &[f32]) -> Vec<Q8K> {
    let mut out = Vec::with_capacity(x.len().div_ceil(256));
    quantize_q8k_into(x, &mut out);
    out
}

#[inline(always)]
fn d16(b: &[u8]) -> f32 {
    f16::from_le_bytes([b[0], b[1]]).to_f32()
}

#[inline(always)]
fn dot8(g: [u8; 8], s: &[i8; 8], q: &[i8]) -> i32 {
    (0..8)
        .map(|j| g[j] as i32 * s[j] as i32 * q[j] as i32)
        .sum()
}

fn dot_iq2xs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(74).zip(xq) {
        let sc = &b[66..74];
        let mut bsum = 0i32;
        for ib in 0..8 {
            let mut s = [0i32; 2];
            for l in 0..4 {
                let o = 2 + 2 * (4 * ib + l);
                let q = u16::from_le_bytes([b[o], b[o + 1]]);
                s[l / 2] += dot8(
                    IQ2XS_GRID[(q & 511) as usize].to_le_bytes(),
                    &SIGNS[(q >> 9) as usize],
                    &y.qs[32 * ib + 8 * l..],
                );
            }
            bsum += s[0] * (2 * (sc[ib] & 0xF) as i32 + 1)
                + s[1] * (2 * (sc[ib] >> 4) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.125 * sumf
}

fn dot_iq2xxs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(66).zip(xq) {
        let mut bsum = 0i32;
        for ib in 0..8 {
            let aux = u32::from_le_bytes(b[6 + 8 * ib..10 + 8 * ib].try_into().unwrap());
            let s: i32 = (0..4)
                .map(|l| {
                    dot8(
                        IQ2XXS_GRID[b[2 + 8 * ib + l] as usize].to_le_bytes(),
                        &SIGNS[((aux >> (7 * l)) & 127) as usize],
                        &y.qs[32 * ib + 8 * l..],
                    )
                })
                .sum();
            bsum += s * (2 * (aux >> 28) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.125 * sumf
}

fn dot_iq2s(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(82).zip(xq) {
        let (qs, signs, qh, sc) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
        let mut bsum = 0i32;
        for ib in 0..8 {
            let s: [i32; 2] = std::array::from_fn(|h| {
                (2 * h..2 * h + 2)
                    .map(|l| {
                        let g = IQ2S_GRID
                            [qs[4 * ib + l] as usize | ((qh[ib] as usize) << (8 - 2 * l) & 0x300)]
                            .to_le_bytes();
                        let sg: [i8; 8] = std::array::from_fn(|j| {
                            1 - 2 * ((signs[4 * ib + l] >> j) & 1) as i8
                        });
                        dot8(g, &sg, &y.qs[32 * ib + 8 * l..])
                    })
                    .sum()
            });
            bsum += s[0] * (2 * (sc[ib] & 0xF) as i32 + 1)
                + s[1] * (2 * (sc[ib] >> 4) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.125 * sumf
}

fn dot1(g: u64, q: &[i8]) -> i32 {
    g.to_le_bytes()
        .iter()
        .zip(&q[..8])
        .map(|(&a, &b)| a as i8 as i32 * b as i32)
        .sum()
}

fn dot_iq1s(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(50).zip(xq) {
        let (mut s, mut s1) = (0i32, 0i32);
        for ib in 0..8 {
            let qh = u16::from_le_bytes([b[34 + 2 * ib], b[35 + 2 * ib]]) as usize;
            let ls = 2 * ((qh >> 12) & 7) as i32 + 1;
            s += ls
                * (0..4)
                    .map(|l| {
                        dot1(
                            IQ1S_GRID[b[2 + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8)],
                            &y.qs[32 * ib + 8 * l..],
                        )
                    })
                    .sum::<i32>();
            s1 += ls
                * if qh & 0x8000 != 0 { -1 } else { 1 }
                * (y.bs[2 * ib] + y.bs[2 * ib + 1]) as i32;
        }
        sumf += d16(b) * y.d * (s as f32 + 0.125 * s1 as f32);
    }
    sumf
}

fn iq1m_d(b: &[u8]) -> f32 {
    let sc = |i: usize| u16::from_le_bytes([b[48 + 2 * i], b[49 + 2 * i]]);
    f16::from_bits(
        (sc(0) >> 12) | ((sc(1) >> 8) & 0xF0) | ((sc(2) >> 4) & 0xF00) | (sc(3) & 0xF000),
    )
    .to_f32()
}

fn iq1m_ls(b: &[u8], ib: usize, h: usize) -> i32 {
    2 * ((u16::from_le_bytes([b[48 + ib / 2 * 2], b[49 + ib / 2 * 2]])
        >> (6 * (ib % 2) + 3 * h))
        & 7) as i32
        + 1
}

fn dot_iq1m(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(56).zip(xq) {
        let (mut s1, mut s2) = (0i32, 0i32);
        for ib in 0..8 {
            for l in 0..4 {
                let qh = b[32 + 2 * ib + l / 2] as usize >> (4 * (l % 2));
                let q = &y.qs[32 * ib + 8 * l..32 * ib + 8 * l + 8];
                let ls = iq1m_ls(b, ib, l / 2);
                s1 += ls * dot1(IQ1S_GRID[b[4 * ib + l] as usize | ((qh & 7) << 8)], q);
                s2 += ls
                    * if qh & 8 != 0 { -1 } else { 1 }
                    * q.iter().map(|&v| v as i32).sum::<i32>();
            }
        }
        sumf += iq1m_d(b) * y.d * (s1 as f32 + 0.125 * s2 as f32);
    }
    sumf
}

fn dot_iq3xxs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(98).zip(xq) {
        let (qs, gas) = (&b[2..66], &b[66..98]);
        let mut bsum = 0i32;
        for ib in 0..8 {
            let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
            let mut s = 0i32;
            for l in 0..4 {
                let g1 = IQ3XXS_GRID[qs[8 * ib + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[qs[8 * ib + 2 * l + 1] as usize].to_le_bytes();
                let g = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                s += dot8(
                    g,
                    &SIGNS[((aux >> (7 * l)) & 127) as usize],
                    &y.qs[32 * ib + 8 * l..],
                );
            }
            bsum += s * (2 * (aux >> 28) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    0.25 * sumf
}

fn dot_iq3s(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(110).zip(xq) {
        let (qs, qh, signs, sc) = (&b[2..66], &b[66..74], &b[74..106], &b[106..110]);
        let mut bsum = 0i32;
        for ib in 0..8 {
            let mut s = 0i32;
            for l in 0..4 {
                let g = |k: usize, sh: usize| {
                    IQ3S_GRID[(qs[8 * ib + 2 * l + k] as usize) | ((qh[ib] as usize) << sh & 256)]
                        .to_le_bytes()
                };
                let (g1, g2) = (g(0, 8 - 2 * l), g(1, 7 - 2 * l));
                let g = [g1[0], g1[1], g1[2], g1[3], g2[0], g2[1], g2[2], g2[3]];
                let sg: [i8; 8] =
                    std::array::from_fn(|j| if (signs[4 * ib + l] >> j) & 1 != 0 { -1 } else { 1 });
                s += dot8(g, &sg, &y.qs[32 * ib + 8 * l..]);
            }
            bsum += s * (2 * ((sc[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32 + 1);
        }
        sumf += d16(b) * y.d * bsum as f32;
    }
    sumf
}

fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(136).zip(xq) {
        let dd = d16(b) * y.d;
        let sh = u16::from_le_bytes([b[2], b[3]]);
        let (sl, qs) = (&b[4..8], &b[8..136]);
        for ib in 0..8 {
            let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                | ((((sh >> (2 * ib)) & 3) as i32) << 4);
            let q8 = &y.qs[32 * ib..32 * ib + 32];
            let s: i32 = (0..16)
                .map(|j| {
                    let q = qs[16 * ib + j];
                    q8[j] as i32 * KVALUES_IQ4NL[(q & 0xF) as usize] as i32
                        + q8[j + 16] as i32 * KVALUES_IQ4NL[(q >> 4) as usize] as i32
                })
                .sum();
            sumf += dd * (ls - 32) as f32 * s as f32;
        }
    }
    sumf
}

fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(18)
        .enumerate()
        .map(|(i, b)| {
            let (y, o) = (&xq[i / 8], (i % 8) * 32);
            let s: i32 = (0..16)
                .map(|j| {
                    let q = b[2 + j];
                    y.qs[o + j] as i32 * KVALUES_IQ4NL[(q & 0xF) as usize] as i32
                        + y.qs[o + j + 16] as i32 * KVALUES_IQ4NL[(q >> 4) as usize] as i32
                })
                .sum();
            d16(b) * y.d * s as f32
        })
        .sum()
}

#[inline(always)]
fn grouped(a: &[i8; 256], q: &[i8; 256], sc: impl Fn(usize) -> i32) -> i32 {
    (0..16)
        .map(|j| {
            sc(j)
                * (16 * j..16 * j + 16)
                    .map(|i| a[i] as i32 * q[i] as i32)
                    .sum::<i32>()
        })
        .sum()
}

fn dot_q2k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(84).zip(xq) {
        let sc = &b[0..16];
        let mut a = [0i8; 256];
        for i in 0..256 {
            a[i] = ((b[16 + 32 * (i / 128) + i % 32] >> (2 * (i / 32 % 4))) & 3) as i8;
        }
        let s = grouped(&a, &y.qs, |j| (sc[j] & 0xF) as i32);
        let m: i32 = (0..16)
            .map(|j| (sc[j] >> 4) as i32 * y.bs[j] as i32)
            .sum();
        sumf += y.d * d16(&b[80..]) * s as f32 - y.d * d16(&b[82..]) * m as f32;
    }
    sumf
}

fn dot_q3k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(110).zip(xq) {
        let (hm, sc) = (&b[0..32], crate::quant::q3k_scales(b));
        let mut a = [0i8; 256];
        for i in 0..256 {
            let k = i / 32 % 4 + 4 * (i / 128);
            a[i] = ((b[32 + 32 * (i / 128) + i % 32] >> (2 * (k % 4))) & 3) as i8
                - if (hm[i % 32] >> k) & 1 != 0 { 0 } else { 4 };
        }
        sumf += d16(&b[108..]) * y.d * grouped(&a, &y.qs, |j| sc[j] as i32 - 32) as f32;
    }
    sumf
}

fn dot_q2_0(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(18)
        .enumerate()
        .map(|(i, b)| {
            let (y, o) = (&xq[i / 4], (i % 4) * 64);
            let s: i32 = (0..64)
                .map(|j| ((b[2 + j / 4] >> (2 * (j % 4))) & 3) as i32 * y.qs[o + j] as i32)
                .sum();
            let m: i32 = y.bs[o / 16..o / 16 + 4].iter().map(|&v| v as i32).sum();
            d16(b) * y.d * (s - m) as f32
        })
        .sum()
}

fn dot_tq1_0(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(54).zip(xq) {
        let trit = |v: u8, n: usize| {
            const POW3: [u8; 5] = [1, 3, 9, 27, 81];
            ((v.wrapping_mul(POW3[n]) as u16 * 3) >> 8) as i32 - 1
        };
        let mut s = 0i32;
        for n in 0..5 {
            for m in 0..32 {
                s += trit(b[m], n) * y.qs[32 * n + m] as i32;
            }
        }
        for n in 0..5 {
            for m in 0..16 {
                s += trit(b[32 + m], n) * y.qs[160 + 16 * n + m] as i32;
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                s += trit(b[48 + j], n) * y.qs[240 + 4 * n + j] as i32;
            }
        }
        sumf += d16(&b[52..]) * y.d * s as f32;
    }
    sumf
}

fn dot_tq2_0(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(66).zip(xq) {
        let mut s = 0i32;
        for j in 0..2 {
            for l in 0..4 {
                for m in 0..32 {
                    let code = (b[32 * j + m] >> (2 * l)) & 3;
                    s += (code as i32 - 1) * y.qs[128 * j + 32 * l + m] as i32;
                }
            }
        }
        sumf += d16(&b[64..]) * y.d * s as f32;
    }
    sumf
}

#[inline(always)]
const fn q4_layout(t: u8) -> (usize, usize, usize) {
    match t {
        2 => (18, 0, 2),
        3 => (20, 0, 4),
        6 => (22, 2, 6),
        _ => (24, 4, 8),
    }
}

#[inline(always)]
fn q4_min(t: u8, b: &[u8]) -> f32 {
    match t {
        3 | 7 => d16(&b[2..]),
        2 => -8.0 * d16(b),
        _ => -16.0 * d16(b),
    }
}

fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
    let (bb, oh, oq) = q4_layout(T);
    row.chunks_exact(bb)
        .enumerate()
        .map(|(i, b)| {
            let (y, o) = (&xq[i / 8], (i % 8) * 32);
            let (qh, qs) = if oh > 0 {
                (
                    u32::from_le_bytes(b[oh..oh + 4].try_into().unwrap()),
                    &b[oq..oq + 16],
                )
            } else {
                (0, &b[oq..oq + 16])
            };
            let s: i32 = (0..16)
                .map(|j| {
                    let (l, h) = (
                        (qs[j] & 0xF) as i32 | (((qh >> j) & 1) as i32 * 16),
                        (qs[j] >> 4) as i32 | (((qh >> (j + 16)) & 1) as i32 * 16),
                    );
                    l * y.qs[o + j] as i32 + h * y.qs[o + j + 16] as i32
                })
                .sum();
            y.d * (d16(b) * s as f32
                + q4_min(T, b) * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32)
        })
        .sum()
}

fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
    row.chunks_exact(34)
        .enumerate()
        .map(|(i, b)| {
            let (y, o) = (&xq[i / 8], (i % 8) * 32);
            let s: i32 = (0..32)
                .map(|j| b[2 + j] as i8 as i32 * y.qs[o + j] as i32)
                .sum();
            d16(b) * y.d * s as f32
        })
        .sum()
}

fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(144).zip(xq) {
        let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
        let mut a = [0i8; 256];
        for i in 0..256 {
            a[i] = ((b[16 + 32 * (i / 64) + i % 32] >> (4 * (i / 32 % 2))) & 0xF) as i8;
        }
        let s = grouped(&a, &y.qs, |j| sm[j / 2].0 as i32);
        let m: i32 = (0..8)
            .map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32)
            .sum();
        sumf += d16(b) * y.d * s as f32 - d16(&b[2..]) * y.d * m as f32;
    }
    sumf
}

fn dot_q5k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(176).zip(xq) {
        let sm: [(u8, u8); 8] = std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
        let mut a = [0i8; 256];
        for j in 0..4 {
            for l in 0..32 {
                let (q, h) = (b[48 + 32 * j + l], b[16 + l] >> (2 * j));
                a[64 * j + l] = ((q & 0xF) | ((h & 1) << 4)) as i8;
                a[64 * j + 32 + l] = ((q >> 4) | ((h & 2) << 3)) as i8;
            }
        }
        let s = grouped(&a, &y.qs, |j| sm[j / 2].0 as i32);
        let m: i32 = (0..8)
            .map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32)
            .sum();
        sumf += d16(b) * y.d * s as f32 - d16(&b[2..]) * y.d * m as f32;
    }
    sumf
}

fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
    let mut sumf = 0.0;
    for (b, y) in row.chunks_exact(210).zip(xq) {
        let mut a = [0i8; 256];
        for n in 0..2 {
            let (ql, qh, a) = (&b[64 * n..], &b[128 + 32 * n..], &mut a[128 * n..]);
            for l in 0..32 {
                a[l] = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                a[l + 32] = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                a[l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                a[l + 96] = ((ql[l + 32] >> 4) | ((qh[l] >> 6) << 4)) as i8 - 32;
            }
        }
        let s: i32 = (0..16)
            .map(|j| {
                b[192 + j] as i8 as i32
                    * (16 * j..16 * j + 16)
                        .map(|i| a[i] as i32 * y.qs[i] as i32)
                        .sum::<i32>()
            })
            .sum();
        sumf += d16(&b[208..]) * y.d * s as f32;
    }
    sumf
}

// ============================================================================
// ARM64 NEON Kernels (ARMv8.0-A baseline — safe on all 64-bit Android SoCs)
// ============================================================================
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Q8K, d16, q4_layout, q4_min};
    use crate::quant::tables::KVALUES_IQ4NL;
    use std::arch::aarch64::*;

    #[inline(always)]
    unsafe fn dot_s8_16(a: int8x16_t, b: int8x16_t) -> i32 {
        unsafe {
            let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
            let p1 = vmull_high_s8(a, b);
            let sum16 = vaddq_s16(p0, p1);
            let sum32 = vpaddlq_s16(sum16);
            vaddvq_s32(sum32)
        }
    }

    pub unsafe fn quant_block(is: f32, b: &[f32], qs: &mut [i8; 256]) {
        unsafe {
            let vs = vdupq_n_f32(is);
            let min_v = vdupq_n_s32(-127);
            let max_v = vdupq_n_s32(127);
            for j in (0..256).step_by(16) {
                let mut acc = [0i8; 16];
                for k in 0..4 {
                    let f = vld1q_f32(b.as_ptr().add(j + k * 4));
                    let scaled = vmulq_f32(f, vs);
                    let mut i32v = vcvtnq_s32_f32(scaled);
                    i32v = vmaxq_s32(min_v, vminq_s32(max_v, i32v));
                    let i16v = vmovn_s32(i32v);
                    let i8v = vmovn_s16(vcombine_s16(i16v, i16v));
                    let bytes = vget_lane_u32(vreinterpret_u32_s8(i8v), 0)
                        .to_le_bytes()
                        .map(|b| b as i8);
                    acc[k * 4..k * 4 + 4].copy_from_slice(&bytes);
                }
                std::ptr::copy_nonoverlapping(acc.as_ptr(), qs.as_mut_ptr().add(j), 16);
            }
        }
    }

    pub fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut sumf = 0.0f32;
            for (i, b) in row.chunks_exact(34).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let w0 = vld1q_s8(b.as_ptr().add(2) as *const i8);
                let w1 = vld1q_s8(b.as_ptr().add(18) as *const i8);
                let q0 = vld1q_s8(y.qs.as_ptr().add(o));
                let q1 = vld1q_s8(y.qs.as_ptr().add(o + 16));
                let p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(q0));
                let p1 = vmull_high_s8(w0, q0);
                let p2 = vmull_s8(vget_low_s8(w1), vget_low_s8(q1));
                let p3 = vmull_high_s8(w1, q1);
                let acc16 = vaddq_s16(vaddq_s16(p0, p1), vaddq_s16(p2, p3));
                let s = vaddvq_s32(vpaddlq_s16(acc16));
                sumf += d16(b) * y.d * s as f32;
            }
            sumf
        }
    }

    pub fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        let (bb, oh, oq) = q4_layout(T);
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let qs = vld1q_u8(b.as_ptr().add(oq));
                let mut w_lo = vreinterpretq_s8_u8(vandq_u8(qs, m4));
                let mut w_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs));
                if oh > 0 {
                    let qh = u32::from_le_bytes(b[oh..oh + 4].try_into().unwrap());
                    let mut hi0 = [0i8; 16];
                    let mut hi1 = [0i8; 16];
                    for j in 0..16 {
                        hi0[j] = (((qh >> j) & 1) << 4) as i8;
                        hi1[j] = (((qh >> (j + 16)) & 1) << 4) as i8;
                    }
                    w_lo = vaddq_s8(w_lo, vld1q_s8(hi0.as_ptr()));
                    w_hi = vaddq_s8(w_hi, vld1q_s8(hi1.as_ptr()));
                }
                let q0 = vld1q_s8(y.qs.as_ptr().add(o));
                let q1 = vld1q_s8(y.qs.as_ptr().add(o + 16));
                let s = dot_s8_16(w_lo, q0) + dot_s8_16(w_hi, q1);
                sumf += y.d
                    * (d16(b) * s as f32
                        + q4_min(T, b) * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32);
            }
            sumf
        }
    }

    pub fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(144).zip(xq) {
                let sm: [(u8, u8); 8] =
                    std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let mut sumi = 0i32;
                for j in 0..4 {
                    let b0 = vld1q_u8(b.as_ptr().add(16 + 32 * j));
                    let b1 = vld1q_u8(b.as_ptr().add(16 + 32 * j + 16));
                    let w0_lo = vreinterpretq_s8_u8(vandq_u8(b0, m4));
                    let w1_lo = vreinterpretq_s8_u8(vandq_u8(b1, m4));
                    let w0_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0));
                    let w1_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1));

                    let q0_lo = vld1q_s8(y.qs.as_ptr().add(64 * j));
                    let q1_lo = vld1q_s8(y.qs.as_ptr().add(64 * j + 16));
                    let q0_hi = vld1q_s8(y.qs.as_ptr().add(64 * j + 32));
                    let q1_hi = vld1q_s8(y.qs.as_ptr().add(64 * j + 48));

                    let s0 = dot_s8_16(w0_lo, q0_lo) + dot_s8_16(w1_lo, q1_lo);
                    let s1 = dot_s8_16(w0_hi, q0_hi) + dot_s8_16(w1_hi, q1_hi);
                    sumi += s0 * sm[2 * j].0 as i32 + s1 * sm[2 * j + 1].0 as i32;
                }
                let m: i32 = (0..8)
                    .map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32)
                    .sum();
                sumf += d16(b) * y.d * sumi as f32 - d16(&b[2..]) * y.d * m as f32;
            }
            sumf
        }
    }

    pub fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let m2 = vdupq_n_u8(0x03);
            let off32 = vdupq_n_s8(32);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(210).zip(xq) {
                let mut sumi = 0i32;
                for n in 0..2 {
                    let ql = b.as_ptr().add(64 * n);
                    let qh = b.as_ptr().add(128 + 32 * n);
                    let sc = &b[192 + 8 * n..];
                    let yp = y.qs.as_ptr().add(128 * n);
                    for half in 0..2 {
                        let off = 16 * half;
                        let la = vld1q_u8(ql.add(off));
                        let lb = vld1q_u8(ql.add(32 + off));
                        let h = vld1q_u8(qh.add(off));
                        let w1 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(la, m4), vshlq_n_u8::<4>(vandq_u8(h, m2)))),
                            off32,
                        );
                        let w2 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(lb, m4),
                                vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h), m2)),
                            )),
                            off32,
                        );
                        let w3 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8::<4>(la),
                                vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h), m2)),
                            )),
                            off32,
                        );
                        let w4 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8::<4>(lb),
                                vshlq_n_u8::<4>(vshrq_n_u8::<6>(h)),
                            )),
                            off32,
                        );
                        let q1 = vld1q_s8(yp.add(off));
                        let q2 = vld1q_s8(yp.add(32 + off));
                        let q3 = vld1q_s8(yp.add(64 + off));
                        let q4 = vld1q_s8(yp.add(96 + off));
                        sumi += (sc[half] as i8 as i32) * dot_s8_16(w1, q1)
                            + (sc[2 + half] as i8 as i32) * dot_s8_16(w2, q2)
                            + (sc[4 + half] as i8 as i32) * dot_s8_16(w3, q3)
                            + (sc[6 + half] as i8 as i32) * dot_s8_16(w4, q4);
                    }
                }
                sumf += d16(&b[208..]) * y.d * sumi as f32;
            }
            sumf
        }
    }

    pub fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let tbl = vld1q_s8(KVALUES_IQ4NL.as_ptr());
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (i, b) in row.chunks_exact(18).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let qs = vld1q_u8(b.as_ptr().add(2));
                let w0 = vqtbl1q_s8(tbl, vandq_u8(qs, m4));
                let w1 = vqtbl1q_s8(tbl, vshrq_n_u8::<4>(qs));
                let q0 = vld1q_s8(y.qs.as_ptr().add(o));
                let q1 = vld1q_s8(y.qs.as_ptr().add(o + 16));
                let s = dot_s8_16(w0, q0) + dot_s8_16(w1, q1);
                sumf += d16(b) * y.d * s as f32;
            }
            sumf
        }
    }

    pub fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let tbl = vld1q_s8(KVALUES_IQ4NL.as_ptr());
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(136).zip(xq) {
                let dd = d16(b) * y.d;
                let sh = u16::from_le_bytes([b[2], b[3]]);
                let (sl, qs) = (&b[4..8], &b[8..136]);
                for ib in 0..8 {
                    let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                        | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                    let raw = vld1q_u8(qs.as_ptr().add(16 * ib));
                    let w0 = vqtbl1q_s8(tbl, vandq_u8(raw, m4));
                    let w1 = vqtbl1q_s8(tbl, vshrq_n_u8::<4>(raw));
                    let q0 = vld1q_s8(y.qs.as_ptr().add(32 * ib));
                    let q1 = vld1q_s8(y.qs.as_ptr().add(32 * ib + 16));
                    let s = dot_s8_16(w0, q0) + dot_s8_16(w1, q1);
                    sumf += dd * (ls - 32) as f32 * s as f32;
                }
            }
            sumf
        }
    }

    #[inline(always)]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let mut i = 0;
            while i + 16 <= n {
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
                acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
                acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
                i += 16;
            }
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            while i + 4 <= n {
                acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
                i += 4;
            }
            let mut s = vaddvq_f32(acc0);
            while i < n {
                s += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            s
        }
    }

    #[inline(always)]
    pub fn fma_f32(out: &mut [f32], v: &[f32], scale: f32) {
        unsafe {
            let n = out.len();
            let po = out.as_mut_ptr();
            let pv = v.as_ptr();
            let vs = vdupq_n_f32(scale);
            let mut i = 0;
            while i + 16 <= n {
                vst1q_f32(po.add(i), vfmaq_f32(vld1q_f32(po.add(i)), vld1q_f32(pv.add(i)), vs));
                vst1q_f32(
                    po.add(i + 4),
                    vfmaq_f32(vld1q_f32(po.add(i + 4)), vld1q_f32(pv.add(i + 4)), vs),
                );
                vst1q_f32(
                    po.add(i + 8),
                    vfmaq_f32(vld1q_f32(po.add(i + 8)), vld1q_f32(pv.add(i + 8)), vs),
                );
                vst1q_f32(
                    po.add(i + 12),
                    vfmaq_f32(vld1q_f32(po.add(i + 12)), vld1q_f32(pv.add(i + 12)), vs),
                );
                i += 16;
            }
            while i + 4 <= n {
                vst1q_f32(po.add(i), vfmaq_f32(vld1q_f32(po.add(i)), vld1q_f32(pv.add(i)), vs));
                i += 4;
            }
            while i < n {
                *po.add(i) += scale * *pv.add(i);
                i += 1;
            }
        }
    }
}

// ============================================================================
// ARMv8.2-A DOTPROD (`vdotq_s32`) Runtime-Dispatched Kernels
// ============================================================================
#[cfg(target_arch = "aarch64")]
mod neon_dotprod {
    use super::{Q8K, d16, q4_layout, q4_min};
    use crate::quant::tables::KVALUES_IQ4NL;
    use std::arch::aarch64::*;

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q8_0_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut sumf = 0.0f32;
            for (i, b) in row.chunks_exact(34).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let w0 = vld1q_s8(b.as_ptr().add(2) as *const i8);
                let w1 = vld1q_s8(b.as_ptr().add(18) as *const i8);
                let q0 = vld1q_s8(y.qs.as_ptr().add(o));
                let q1 = vld1q_s8(y.qs.as_ptr().add(o + 16));
                let mut acc = vdotq_s32(vdupq_n_s32(0), w0, q0);
                acc = vdotq_s32(acc, w1, q1);
                sumf += d16(b) * y.d * vaddvq_s32(acc) as f32;
            }
            sumf
        }
    }
    pub fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q8_0_impl(row, xq) }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q4_impl<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        let (bb, oh, oq) = q4_layout(T);
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let qs = vld1q_u8(b.as_ptr().add(oq));
                let mut w_lo = vreinterpretq_s8_u8(vandq_u8(qs, m4));
                let mut w_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(qs));
                if oh > 0 {
                    let qh = u32::from_le_bytes(b[oh..oh + 4].try_into().unwrap());
                    let mut hi0 = [0i8; 16];
                    let mut hi1 = [0i8; 16];
                    for j in 0..16 {
                        hi0[j] = (((qh >> j) & 1) << 4) as i8;
                        hi1[j] = (((qh >> (j + 16)) & 1) << 4) as i8;
                    }
                    w_lo = vaddq_s8(w_lo, vld1q_s8(hi0.as_ptr()));
                    w_hi = vaddq_s8(w_hi, vld1q_s8(hi1.as_ptr()));
                }
                let q0 = vld1q_s8(y.qs.as_ptr().add(o));
                let q1 = vld1q_s8(y.qs.as_ptr().add(o + 16));
                let mut acc = vdotq_s32(vdupq_n_s32(0), w_lo, q0);
                acc = vdotq_s32(acc, w_hi, q1);
                let s = vaddvq_s32(acc);
                sumf += y.d
                    * (d16(b) * s as f32
                        + q4_min(T, b) * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32);
            }
            sumf
        }
    }
    pub fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q4_impl::<T>(row, xq) }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q4k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(144).zip(xq) {
                let sm: [(u8, u8); 8] =
                    std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let mut sumi = 0i32;
                let mut m_sum = 0i32;
                let bp = b.as_ptr();
                let qp = y.qs.as_ptr();
                for j in 0..4 {
                    let b0 = vld1q_u8(bp.add(16 + 32 * j));
                    let b1 = vld1q_u8(bp.add(32 + 32 * j));
                    let w0_lo = vreinterpretq_s8_u8(vandq_u8(b0, m4));
                    let w1_lo = vreinterpretq_s8_u8(vandq_u8(b1, m4));
                    let w0_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b0));
                    let w1_hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(b1));

                    let q0_lo = vld1q_s8(qp.add(64 * j));
                    let q1_lo = vld1q_s8(qp.add(64 * j + 16));
                    let q0_hi = vld1q_s8(qp.add(64 * j + 32));
                    let q1_hi = vld1q_s8(qp.add(64 * j + 48));

                    let acc0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0_lo, q0_lo), w1_lo, q1_lo);
                    let acc1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0_hi, q0_hi), w1_hi, q1_hi);
                    sumi += vaddvq_s32(acc0) * sm[2 * j].0 as i32
                        + vaddvq_s32(acc1) * sm[2 * j + 1].0 as i32;
                    m_sum += sm[2 * j].1 as i32 * (y.bs[4 * j] + y.bs[4 * j + 1]) as i32
                        + sm[2 * j + 1].1 as i32 * (y.bs[4 * j + 2] + y.bs[4 * j + 3]) as i32;
                }
                sumf += d16(b) * y.d * sumi as f32 - d16(&b[2..]) * y.d * m_sum as f32;
            }
            sumf
        }
    }
    pub fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q4k_impl(row, xq) }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q5k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let m1 = vdupq_n_u8(0x01);
            let m2 = vdupq_n_u8(0x02);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(176).zip(xq) {
                let sm: [(u8, u8); 8] =
                    std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let mut sumi = 0i32;
                let mut m_sum = 0i32;
                let bp = b.as_ptr();
                let qp = y.qs.as_ptr();
                let mut h0 = vld1q_u8(bp.add(16));
                let mut h1 = vld1q_u8(bp.add(32));
                for j in 0..4 {
                    let b0 = vld1q_u8(bp.add(48 + 32 * j));
                    let b1 = vld1q_u8(bp.add(64 + 32 * j));

                    let w0_lo = vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(b0, m4),
                        vshlq_n_u8::<4>(vandq_u8(h0, m1)),
                    ));
                    let w1_lo = vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(b1, m4),
                        vshlq_n_u8::<4>(vandq_u8(h1, m1)),
                    ));
                    let w0_hi = vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8::<4>(b0),
                        vshlq_n_u8::<3>(vandq_u8(h0, m2)),
                    ));
                    let w1_hi = vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8::<4>(b1),
                        vshlq_n_u8::<3>(vandq_u8(h1, m2)),
                    ));
                    h0 = vshrq_n_u8::<2>(h0);
                    h1 = vshrq_n_u8::<2>(h1);

                    let q0_lo = vld1q_s8(qp.add(64 * j));
                    let q1_lo = vld1q_s8(qp.add(64 * j + 16));
                    let q0_hi = vld1q_s8(qp.add(64 * j + 32));
                    let q1_hi = vld1q_s8(qp.add(64 * j + 48));

                    let acc0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0_lo, q0_lo), w1_lo, q1_lo);
                    let acc1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0_hi, q0_hi), w1_hi, q1_hi);
                    sumi += vaddvq_s32(acc0) * sm[2 * j].0 as i32
                        + vaddvq_s32(acc1) * sm[2 * j + 1].0 as i32;
                    m_sum += sm[2 * j].1 as i32 * (y.bs[4 * j] + y.bs[4 * j + 1]) as i32
                        + sm[2 * j + 1].1 as i32 * (y.bs[4 * j + 2] + y.bs[4 * j + 3]) as i32;
                }
                sumf += d16(b) * y.d * sumi as f32 - d16(&b[2..]) * y.d * m_sum as f32;
            }
            sumf
        }
    }
    pub fn dot_q5k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q5k_impl(row, xq) }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_q6k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = vdupq_n_u8(0x0F);
            let m2 = vdupq_n_u8(0x03);
            let off32 = vdupq_n_s8(32);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(210).zip(xq) {
                let mut sumi = 0i32;
                for n in 0..2 {
                    let ql = b.as_ptr().add(64 * n);
                    let qh = b.as_ptr().add(128 + 32 * n);
                    let sc = &b[192 + 8 * n..];
                    let yp = y.qs.as_ptr().add(128 * n);
                    for half in 0..2 {
                        let off = 16 * half;
                        let la = vld1q_u8(ql.add(off));
                        let lb = vld1q_u8(ql.add(32 + off));
                        let h = vld1q_u8(qh.add(off));
                        let w1 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(la, m4),
                                vshlq_n_u8::<4>(vandq_u8(h, m2)),
                            )),
                            off32,
                        );
                        let w2 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vandq_u8(lb, m4),
                                vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(h), m2)),
                            )),
                            off32,
                        );
                        let w3 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8::<4>(la),
                                vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(h), m2)),
                            )),
                            off32,
                        );
                        let w4 = vsubq_s8(
                            vreinterpretq_s8_u8(vorrq_u8(
                                vshrq_n_u8::<4>(lb),
                                vshlq_n_u8::<4>(vshrq_n_u8::<6>(h)),
                            )),
                            off32,
                        );
                        let q1 = vld1q_s8(yp.add(off));
                        let q2 = vld1q_s8(yp.add(32 + off));
                        let q3 = vld1q_s8(yp.add(64 + off));
                        let q4 = vld1q_s8(yp.add(96 + off));
                        let d1 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w1, q1));
                        let d2 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w2, q2));
                        let d3 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w3, q3));
                        let d4 = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), w4, q4));
                        sumi += (sc[half] as i8 as i32) * d1
                            + (sc[2 + half] as i8 as i32) * d2
                            + (sc[4 + half] as i8 as i32) * d3
                            + (sc[6 + half] as i8 as i32) * d4;
                    }
                }
                sumf += d16(&b[208..]) * y.d * sumi as f32;
            }
            sumf
        }
    }
    pub fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q6k_impl(row, xq) }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_iq4nl_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let tbl = vld1q_s8(KVALUES_IQ4NL.as_ptr());
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (i, b) in row.chunks_exact(18).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let qs = vld1q_u8(b.as_ptr().add(2));
                let w0 = vqtbl1q_s8(tbl, vandq_u8(qs, m4));
                let w1 = vqtbl1q_s8(tbl, vshrq_n_u8::<4>(qs));
                let q0 = vld1q_s8(y.qs.as_ptr().add(o));
                let q1 = vld1q_s8(y.qs.as_ptr().add(o + 16));
                let acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, q0), w1, q1);
                sumf += d16(b) * y.d * vaddvq_s32(acc) as f32;
            }
            sumf
        }
    }
    pub fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq4nl_impl(row, xq) }
    }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_iq4xs_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let tbl = vld1q_s8(KVALUES_IQ4NL.as_ptr());
            let m4 = vdupq_n_u8(0x0F);
            let mut sumf = 0.0f32;
            for (b, y) in row.chunks_exact(136).zip(xq) {
                let dd = d16(b) * y.d;
                let sh = u16::from_le_bytes([b[2], b[3]]);
                let (sl, qs) = (&b[4..8], &b[8..136]);
                for ib in 0..8 {
                    let ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                        | ((((sh >> (2 * ib)) & 3) as i32) << 4);
                    let raw = vld1q_u8(qs.as_ptr().add(16 * ib));
                    let w0 = vqtbl1q_s8(tbl, vandq_u8(raw, m4));
                    let w1 = vqtbl1q_s8(tbl, vshrq_n_u8::<4>(raw));
                    let q0 = vld1q_s8(y.qs.as_ptr().add(32 * ib));
                    let q1 = vld1q_s8(y.qs.as_ptr().add(32 * ib + 16));
                    let acc = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0, q0), w1, q1);
                    sumf += dd * (ls - 32) as f32 * vaddvq_s32(acc) as f32;
                }
            }
            sumf
        }
    }
    pub fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq4xs_impl(row, xq) }
    }
}

// ============================================================================
// x86_64 AVX2 + FMA Kernels (runtime-dispatched)
// ============================================================================
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Q8K, SIGNS, d16, iq1m_d};
    use crate::quant::tables::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        unsafe {
            let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn quant_block(is: f32, b: &[f32], qs: &mut [i8; 256]) {
        unsafe {
            let s = _mm256_set1_ps(is);
            let q = |i: usize| {
                _mm256_cvtps_epi32(_mm256_round_ps(
                    _mm256_mul_ps(s, _mm256_loadu_ps(b[i..i + 8].as_ptr())),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ))
            };
            let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            for j in (0..256).step_by(32) {
                let p = _mm256_packs_epi16(
                    _mm256_packs_epi32(q(j), q(j + 8)),
                    _mm256_packs_epi32(q(j + 16), q(j + 24)),
                );
                _mm256_storeu_si256(
                    qs[j..j + 32].as_mut_ptr() as *mut __m256i,
                    _mm256_permutevar8x32_epi32(p, perm),
                );
            }
        }
    }

    #[inline(always)]
    fn sg(i: usize) -> i64 {
        unsafe { std::mem::transmute::<[i8; 8], i64>(SIGNS[i]) }
    }
    #[inline(always)]
    unsafe fn udot_scale(p: __m256i, lo: i32, hi: i32) -> __m256i {
        unsafe {
            _mm256_madd_epi16(
                p,
                _mm256_set_m128i(_mm_set1_epi16(hi as i16), _mm_set1_epi16(lo as i16)),
            )
        }
    }
    #[inline(always)]
    unsafe fn udot32(w: __m256i, q8: __m256i, lo: i32, hi: i32) -> __m256i {
        unsafe { udot_scale(_mm256_maddubs_epi16(w, q8), lo, hi) }
    }
    #[inline(always)]
    unsafe fn q8(y: &Q8K, i: usize) -> __m256i {
        unsafe { _mm256_loadu_si256(y.qs.as_ptr().add(i) as *const __m256i) }
    }
    #[inline(always)]
    unsafe fn ld(b: &[u8], i: usize) -> __m256i {
        unsafe { _mm256_loadu_si256(b[i..i + 32].as_ptr() as *const __m256i) }
    }
    #[inline(always)]
    unsafe fn crumb(v: __m256i, k: usize) -> __m256i {
        unsafe {
            _mm256_and_si256(
                _mm256_srl_epi16(v, _mm_cvtsi32_si128(2 * k as i32)),
                _mm256_set1_epi8(3),
            )
        }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq2xs_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(74).zip(xq) {
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let v = u64::from_le_bytes(b[2 + 8 * ib..10 + 8 * ib].try_into().unwrap());
                    let (q0, q1, q2, q3) =
                        (v as u16, (v >> 16) as u16, (v >> 32) as u16, (v >> 48) as u16);
                    let g = |q: u16| IQ2XS_GRID[(q & 511) as usize] as i64;
                    let grid = _mm256_set_epi64x(g(q3), g(q2), g(q1), g(q0));
                    let signs = _mm256_set_epi64x(
                        sg((q3 >> 9) as usize),
                        sg((q2 >> 9) as usize),
                        sg((q1 >> 9) as usize),
                        sg((q0 >> 9) as usize),
                    );
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8(y, 32 * ib), signs));
                    let sc = b[66 + ib];
                    let ls = _mm256_set_m128i(
                        _mm_set1_epi16(2 * (sc >> 4) as i16 + 1),
                        _mm_set1_epi16(2 * (sc & 0xF) as i16 + 1),
                    );
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, ls));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }
    pub fn dot_iq2xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq2xs_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq3xxs_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(98).zip(xq) {
                let (qs, gas) = (&b[2..66], &b[66..98]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let v = u64::from_le_bytes(qs[8 * ib..8 * ib + 8].try_into().unwrap());
                    let g = |i: usize| IQ3XXS_GRID[(v >> (8 * i)) as u8 as usize] as i32;
                    let grid = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
                    let aux = u32::from_le_bytes(gas[4 * ib..4 * ib + 4].try_into().unwrap());
                    let s = |l: u32| sg(((aux >> (7 * l)) & 127) as usize);
                    let signs = _mm256_set_epi64x(s(3), s(2), s(1), s(0));
                    let dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8(y, 32 * ib), signs));
                    sumi = _mm256_add_epi32(
                        sumi,
                        _mm256_madd_epi16(dot, _mm256_set1_epi16(2 * (aux >> 28) as i16 + 1)),
                    );
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.25 * hsum(acc)
        }
    }
    pub fn dot_iq3xxs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq3xxs_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq3s_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m1 = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
            let m2 = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
            let shl = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(110).zip(xq) {
                let (qs, qh) = (&b[2..66], &b[66..74]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
                        qs[8 * ib..].as_ptr() as *const __m128i
                    ));
                    let hi = _mm256_and_si256(
                        _mm256_sllv_epi32(_mm256_set1_epi32(qh[ib] as i32), shl),
                        _mm256_set1_epi32(256),
                    );
                    let grid = _mm256_i32gather_epi32(
                        IQ3S_GRID.as_ptr() as *const i32,
                        _mm256_or_si256(lo, hi),
                        4,
                    );
                    let sb = i32::from_le_bytes(b[74 + 4 * ib..78 + 4 * ib].try_into().unwrap());
                    let s = _mm256_cmpeq_epi8(
                        _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2),
                        m2,
                    );
                    let dot = _mm256_maddubs_epi16(
                        grid,
                        _mm256_sub_epi8(_mm256_xor_si256(s, q8(y, 32 * ib)), s),
                    );
                    let ls = 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i16 + 1;
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16(ls)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum(acc)
        }
    }
    pub fn dot_iq3s(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq3s_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq4xs_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
            let m4 = _mm_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(136).zip(xq) {
                let sh = u16::from_le_bytes([b[2], b[3]]);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let bits = _mm_loadu_si128(b.as_ptr().add(8 + 16 * ib) as *const __m128i);
                    let w = _mm256_set_m128i(
                        _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                        _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)),
                    );
                    let p16 = _mm256_maddubs_epi16(
                        _mm256_sign_epi8(w, w),
                        _mm256_sign_epi8(q8(y, 32 * ib), w),
                    );
                    let ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) as i16
                        | ((((sh >> (2 * ib)) & 3) as i16) << 4);
                    sumi =
                        _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, _mm256_set1_epi16(ls - 32)));
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            hsum(acc)
        }
    }
    pub fn dot_iq4xs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq4xs_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq4nl_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let values = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
            let m4 = _mm_set1_epi8(0xF);
            let ones = _mm256_set1_epi16(1);
            let mut acc = _mm256_setzero_ps();
            for (i, b) in row.chunks_exact(18).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let bits = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
                let w = _mm256_set_m128i(
                    _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                    _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)),
                );
                let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8(y, o), w));
                let s = _mm256_madd_epi16(p, ones);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(s), acc);
            }
            hsum(acc)
        }
    }
    pub fn dot_iq4nl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq4nl_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q4_impl<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        let (bb, oh, oq) = super::q4_layout(T);
        unsafe {
            let (m4, ones, mut acc, mut sumf) = (
                _mm_set1_epi8(0xF),
                _mm256_set1_epi16(1),
                _mm256_setzero_ps(),
                0f32,
            );
            for (i, b) in row.chunks_exact(bb).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let q = _mm_loadu_si128(b[oq..oq + 16].as_ptr() as *const __m128i);
                let mut w = _mm256_set_m128i(
                    _mm_and_si128(_mm_srli_epi16(q, 4), m4),
                    _mm_and_si128(q, m4),
                );
                if oh > 0 {
                    let qh = i32::from_le_bytes(b[oh..oh + 4].try_into().unwrap());
                    let sh = _mm256_shuffle_epi8(
                        _mm256_set1_epi32(qh),
                        _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0),
                    );
                    let set = _mm256_cmpeq_epi8(
                        _mm256_or_si256(sh, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe)),
                        _mm256_set1_epi64x(-1),
                    );
                    w = _mm256_or_si256(w, _mm256_and_si256(set, _mm256_set1_epi8(16)));
                }
                let s = _mm256_madd_epi16(_mm256_maddubs_epi16(w, q8(y, o)), ones);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(s), acc);
                sumf += super::q4_min(T, b)
                    * y.d
                    * (y.bs[o / 16] as i32 + y.bs[o / 16 + 1] as i32) as f32;
            }
            hsum(acc) + sumf
        }
    }
    pub fn dot_q4<const T: u8>(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q4_impl::<T>(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q8_0_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (ones, mut acc) = (_mm256_set1_epi16(1), _mm256_setzero_ps());
            for (i, b) in row.chunks_exact(34).enumerate() {
                let (y, o) = (&xq[i / 8], (i % 8) * 32);
                let w = _mm256_loadu_si256(b[2..34].as_ptr() as *const __m256i);
                let p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8(y, o), w));
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(d16(b) * y.d),
                    _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)),
                    acc,
                );
            }
            hsum(acc)
        }
    }
    pub fn dot_q8_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q8_0_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q2k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(84).zip(xq) {
                let sc = &b[0..16];
                let mut sumi = _mm256_setzero_si256();
                for n in 0..2 {
                    let bits = ld(b, 16 + 32 * n);
                    for k in 0..4 {
                        let g = 8 * n + 2 * k;
                        sumi = _mm256_add_epi32(
                            sumi,
                            udot32(
                                crumb(bits, k),
                                q8(y, 32 * (4 * n + k)),
                                (sc[g] & 0xF) as i32,
                                (sc[g + 1] & 0xF) as i32,
                            ),
                        );
                    }
                }
                let mins = _mm256_cvtepu8_epi16(_mm_and_si128(
                    _mm_srli_epi16(_mm_loadu_si128(sc.as_ptr() as *const __m128i), 4),
                    _mm_set1_epi8(0xF),
                ));
                let m = _mm256_madd_epi16(mins, _mm256_loadu_si256(y.bs.as_ptr() as *const __m256i));
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(y.d * d16(&b[80..])),
                    _mm256_cvtepi32_ps(sumi),
                    acc,
                );
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(-y.d * d16(&b[82..])),
                    _mm256_cvtepi32_ps(m),
                    acc,
                );
            }
            hsum(acc)
        }
    }
    pub fn dot_q2k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q2k_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q3k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(110).zip(xq) {
                let (hm, sc) = (ld(b, 0), crate::quant::q3k_scales(b));
                let mut sumi = _mm256_setzero_si256();
                for n in 0..2 {
                    let bits = ld(b, 32 + 32 * n);
                    for k in 0..4 {
                        let kk = 4 * n + k;
                        let mask = _mm256_set1_epi8(1 << kk);
                        let low = _mm256_cmpeq_epi8(_mm256_and_si256(hm, mask), _mm256_setzero_si256());
                        let w = _mm256_sub_epi8(
                            crumb(bits, k),
                            _mm256_and_si256(low, _mm256_set1_epi8(4)),
                        );
                        let p = _mm256_maddubs_epi16(
                            _mm256_sign_epi8(w, w),
                            _mm256_sign_epi8(q8(y, 32 * kk), w),
                        );
                        let s = _mm256_set_m128i(
                            _mm_set1_epi16(sc[2 * kk + 1] as i16 - 32),
                            _mm_set1_epi16(sc[2 * kk] as i16 - 32),
                        );
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, s));
                    }
                }
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(y.d * d16(&b[108..])),
                    _mm256_cvtepi32_ps(sumi),
                    acc,
                );
            }
            hsum(acc)
        }
    }
    pub fn dot_q3k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q3k_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q4k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(144).zip(xq) {
                let sm: [(u8, u8); 8] =
                    std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    let bits = ld(b, 16 + 32 * j);
                    let (s0, s1) = (sm[2 * j].0 as i32, sm[2 * j + 1].0 as i32);
                    sumi = _mm256_add_epi32(
                        sumi,
                        udot32(_mm256_and_si256(bits, m4), q8(y, 64 * j), s0, s0),
                    );
                    sumi = _mm256_add_epi32(
                        sumi,
                        udot32(
                            _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4),
                            q8(y, 64 * j + 32),
                            s1,
                            s1,
                        ),
                    );
                }
                let m: i32 = (0..8)
                    .map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32)
                    .sum();
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b)), _mm256_cvtepi32_ps(sumi), acc);
                sumf -= y.d * d16(&b[2..]) * m as f32;
            }
            hsum(acc) + sumf
        }
    }
    pub fn dot_q4k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q4k_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q5k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (m4, one) = (_mm256_set1_epi8(0xF), _mm256_set1_epi8(1));
            let hb = |qh: __m256i, k: i32| {
                _mm256_slli_epi16(
                    _mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(k)), one),
                    4,
                )
            };
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(176).zip(xq) {
                let sm: [(u8, u8); 8] =
                    std::array::from_fn(|j| crate::quant::scale_min_k4(j, &b[4..16]));
                let qh = ld(b, 16);
                let mut sumi = _mm256_setzero_si256();
                for j in 0..4 {
                    let bits = ld(b, 48 + 32 * j);
                    let (s0, s1) = (sm[2 * j].0 as i32, sm[2 * j + 1].0 as i32);
                    let w0 = _mm256_or_si256(_mm256_and_si256(bits, m4), hb(qh, 2 * j as i32));
                    let w1 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4),
                        hb(qh, 2 * j as i32 + 1),
                    );
                    sumi = _mm256_add_epi32(sumi, udot32(w0, q8(y, 64 * j), s0, s0));
                    sumi = _mm256_add_epi32(sumi, udot32(w1, q8(y, 64 * j + 32), s1, s1));
                }
                let m: i32 = (0..8)
                    .map(|j| sm[j].1 as i32 * (y.bs[2 * j] + y.bs[2 * j + 1]) as i32)
                    .sum();
                acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b)), _mm256_cvtepi32_ps(sumi), acc);
                sumf -= y.d * d16(&b[2..]) * m as f32;
            }
            hsum(acc) + sumf
        }
    }
    pub fn dot_q5k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q5k_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q6k_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let m4 = _mm256_set1_epi8(0xF);
            let mut acc = _mm256_setzero_ps();
            let mut sumf = 0.0;
            for (b, y) in row.chunks_exact(210).zip(xq) {
                let sc = |g: usize| b[192 + g] as i8 as i32;
                let mut sumi = _mm256_setzero_si256();
                for n in 0..2 {
                    let (la, lb, qh) = (ld(b, 64 * n), ld(b, 64 * n + 32), ld(b, 128 + 32 * n));
                    let hi = |k: usize| _mm256_slli_epi16(crumb(qh, k), 4);
                    let w = [
                        _mm256_or_si256(_mm256_and_si256(la, m4), hi(0)),
                        _mm256_or_si256(_mm256_and_si256(lb, m4), hi(1)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(la, 4), m4), hi(2)),
                        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m4), hi(3)),
                    ];
                    for (k, w) in w.into_iter().enumerate() {
                        let g = 8 * n + 2 * k;
                        sumi = _mm256_add_epi32(
                            sumi,
                            udot32(w, q8(y, 128 * n + 32 * k), sc(g), sc(g + 1)),
                        );
                    }
                }
                let m: i32 = (0..16).map(|g| sc(g) * y.bs[g] as i32).sum();
                let d = y.d * d16(&b[208..]);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
                sumf -= 32.0 * d * m as f32;
            }
            hsum(acc) + sumf
        }
    }
    pub fn dot_q6k(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q6k_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_f32_impl(a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let n = a.len();
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= n {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                acc = _mm256_fmadd_ps(va, vb, acc);
                i += 8;
            }
            let mut s = hsum(acc);
            while i < n {
                s += a[i] * b[i];
                i += 1;
            }
            s
        }
    }
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        unsafe { dot_f32_impl(a, b) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq2xxs_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(66).zip(xq) {
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let g = |l: usize| IQ2XXS_GRID[b[2 + 8 * ib + l] as usize] as i64;
                    let aux = u32::from_le_bytes(b[6 + 8 * ib..10 + 8 * ib].try_into().unwrap());
                    let s = |l: u32| sg(((aux >> (7 * l)) & 127) as usize);
                    let grid = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
                    let dot = _mm256_maddubs_epi16(
                        grid,
                        _mm256_sign_epi8(
                            q8(y, 32 * ib),
                            _mm256_set_epi64x(s(3), s(2), s(1), s(0)),
                        ),
                    );
                    sumi = _mm256_add_epi32(
                        sumi,
                        _mm256_madd_epi16(dot, _mm256_set1_epi16(2 * (aux >> 28) as i16 + 1)),
                    );
                }
                acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
            }
            0.125 * hsum(acc)
        }
    }
    pub fn dot_iq2xxs(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq2xxs_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_iq1m_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let nine = _mm256_set1_epi8(9);
            let mut acc = _mm256_setzero_ps();
            for (b, y) in row.chunks_exact(56).zip(xq) {
                let sc = u64::from_le_bytes(b[48..56].try_into().unwrap());
                let ls: [i16; 16] =
                    std::array::from_fn(|i| 2 * ((sc >> (16 * (i / 4) + 3 * (i % 4))) & 7) as i16 + 1);
                let mut sumi = _mm256_setzero_si256();
                for ib in 0..8 {
                    let qh = u16::from_le_bytes([b[32 + 2 * ib], b[33 + 2 * ib]]) as usize;
                    let g = |l: usize| {
                        let idx = b[4 * ib + l] as usize | (((qh >> (4 * l)) & 15) << 8);
                        let delta = if idx & 2048 != 0 { 8 } else { 10 };
                        i64::from_le_bytes(
                            IQ1S_GRID[idx & 2047]
                                .to_le_bytes()
                                .map(|gg| (8 * gg as i8 + delta) as u8),
                        )
                    };
                    let q = q8(y, 32 * ib);
                    let p = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), q),
                        _mm256_maddubs_epi16(nine, q),
                    );
                    sumi = _mm256_add_epi32(
                        sumi,
                        udot_scale(p, ls[2 * ib] as i32, ls[2 * ib + 1] as i32),
                    );
                }
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(iq1m_d(b) * y.d),
                    _mm256_cvtepi32_ps(sumi),
                    acc,
                );
            }
            0.125 * hsum(acc)
        }
    }
    pub fn dot_iq1m(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_iq1m_impl(row, xq) }
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_q2_0_impl(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe {
            let (mut acc, mut sumf) = (_mm256_setzero_ps(), 0f32);
            let ones = _mm256_set1_epi16(1);
            for (sb, y) in row.chunks(72).zip(xq) {
                let bs = _mm256_madd_epi16(
                    _mm256_loadu_si256(y.bs.as_ptr() as *const __m256i),
                    ones,
                );
                let mut m = [0f32; 8];
                _mm256_storeu_ps(m.as_mut_ptr(), _mm256_cvtepi32_ps(_mm256_hadd_epi32(bs, bs)));
                for (j, b) in sb.chunks_exact(18).enumerate() {
                    let v = _mm_loadu_si128(b[2..18].as_ptr() as *const __m128i);
                    let c = |k: i32| {
                        _mm_and_si128(_mm_srl_epi16(v, _mm_cvtsi32_si128(2 * k)), _mm_set1_epi8(3))
                    };
                    let (c0, c1, c2, c3) = (c(0), c(1), c(2), c(3));
                    let (lo01, lo23) = (_mm_unpacklo_epi8(c0, c1), _mm_unpacklo_epi8(c2, c3));
                    let (hi01, hi23) = (_mm_unpackhi_epi8(c0, c1), _mm_unpackhi_epi8(c2, c3));
                    let w0 = _mm256_set_m128i(
                        _mm_unpackhi_epi16(lo01, lo23),
                        _mm_unpacklo_epi16(lo01, lo23),
                    );
                    let w1 = _mm256_set_m128i(
                        _mm_unpackhi_epi16(hi01, hi23),
                        _mm_unpacklo_epi16(hi01, hi23),
                    );
                    let p = _mm256_add_epi16(
                        _mm256_maddubs_epi16(w0, q8(y, 64 * j)),
                        _mm256_maddubs_epi16(w1, q8(y, 64 * j + 32)),
                    );
                    let d = d16(b) * y.d;
                    acc = _mm256_fmadd_ps(
                        _mm256_set1_ps(d),
                        _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)),
                        acc,
                    );
                    sumf -= d * m[j + j / 2 * 2];
                }
            }
            hsum(acc) + sumf
        }
    }
    pub fn dot_q2_0(row: &[u8], xq: &[Q8K]) -> f32 {
        unsafe { dot_q2_0_impl(row, xq) }
    }
}

pub fn int_dot(ty: GgmlType) -> Option<fn(&[u8], &[Q8K]) -> f32> {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return match ty {
                GgmlType::Q4_0 => Some(neon_dotprod::dot_q4::<2>),
                GgmlType::Q4_1 => Some(neon_dotprod::dot_q4::<3>),
                GgmlType::Q5_0 => Some(neon_dotprod::dot_q4::<6>),
                GgmlType::Q5_1 => Some(neon_dotprod::dot_q4::<7>),
                GgmlType::Q8_0 => Some(neon_dotprod::dot_q8_0),
                GgmlType::Q4K => Some(neon_dotprod::dot_q4k),
                GgmlType::Q5K => Some(neon_dotprod::dot_q5k),
                GgmlType::Q6K => Some(neon_dotprod::dot_q6k),
                GgmlType::IQ4NL => Some(neon_dotprod::dot_iq4nl),
                GgmlType::IQ4XS => Some(neon_dotprod::dot_iq4xs),
                GgmlType::Q2K => Some(dot_q2k),
                GgmlType::Q3K => Some(dot_q3k),
                GgmlType::Q2_0 => Some(dot_q2_0),
                GgmlType::TQ1_0 => Some(dot_tq1_0),
                GgmlType::TQ2_0 => Some(dot_tq2_0),
                GgmlType::IQ1S => Some(dot_iq1s),
                GgmlType::IQ1M => Some(dot_iq1m),
                GgmlType::IQ2XXS => Some(dot_iq2xxs),
                GgmlType::IQ2XS => Some(dot_iq2xs),
                GgmlType::IQ2S => Some(dot_iq2s),
                GgmlType::IQ3XXS => Some(dot_iq3xxs),
                GgmlType::IQ3S => Some(dot_iq3s),
                _ => None,
            };
        }
        return match ty {
            GgmlType::Q4_0 => Some(neon::dot_q4::<2>),
            GgmlType::Q4_1 => Some(neon::dot_q4::<3>),
            GgmlType::Q5_0 => Some(neon::dot_q4::<6>),
            GgmlType::Q5_1 => Some(neon::dot_q4::<7>),
            GgmlType::Q8_0 => Some(neon::dot_q8_0),
            GgmlType::Q4K => Some(neon::dot_q4k),
            GgmlType::Q6K => Some(neon::dot_q6k),
            GgmlType::IQ4NL => Some(neon::dot_iq4nl),
            GgmlType::IQ4XS => Some(neon::dot_iq4xs),
            GgmlType::Q2K => Some(dot_q2k),
            GgmlType::Q3K => Some(dot_q3k),
            GgmlType::Q5K => Some(dot_q5k),
            GgmlType::Q2_0 => Some(dot_q2_0),
            GgmlType::TQ1_0 => Some(dot_tq1_0),
            GgmlType::TQ2_0 => Some(dot_tq2_0),
            GgmlType::IQ1S => Some(dot_iq1s),
            GgmlType::IQ1M => Some(dot_iq1m),
            GgmlType::IQ2XXS => Some(dot_iq2xxs),
            GgmlType::IQ2XS => Some(dot_iq2xs),
            GgmlType::IQ2S => Some(dot_iq2s),
            GgmlType::IQ3XXS => Some(dot_iq3xxs),
            GgmlType::IQ3S => Some(dot_iq3s),
            _ => None,
        };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return match ty {
                GgmlType::IQ2XS => Some(avx2::dot_iq2xs),
                GgmlType::IQ3XXS => Some(avx2::dot_iq3xxs),
                GgmlType::IQ4XS => Some(avx2::dot_iq4xs),
                GgmlType::IQ4NL => Some(avx2::dot_iq4nl),
                GgmlType::IQ3S => Some(avx2::dot_iq3s),
                GgmlType::Q6K => Some(avx2::dot_q6k),
                GgmlType::Q5K => Some(avx2::dot_q5k),
                GgmlType::Q8_0 => Some(avx2::dot_q8_0),
                GgmlType::IQ2XXS => Some(avx2::dot_iq2xxs),
                GgmlType::IQ2S => Some(dot_iq2s),
                GgmlType::IQ1S => Some(dot_iq1s),
                GgmlType::IQ1M => Some(avx2::dot_iq1m),
                GgmlType::Q2K => Some(avx2::dot_q2k),
                GgmlType::Q3K => Some(avx2::dot_q3k),
                GgmlType::Q4K => Some(avx2::dot_q4k),
                GgmlType::Q2_0 => Some(avx2::dot_q2_0),
                GgmlType::TQ1_0 => Some(dot_tq1_0),
                GgmlType::TQ2_0 => Some(dot_tq2_0),
                GgmlType::Q4_0 => Some(avx2::dot_q4::<2>),
                GgmlType::Q4_1 => Some(avx2::dot_q4::<3>),
                GgmlType::Q5_0 => Some(avx2::dot_q4::<6>),
                GgmlType::Q5_1 => Some(avx2::dot_q4::<7>),
                _ => None,
            };
        }
    }
    #[allow(unreachable_code)]
    match ty {
        GgmlType::IQ2XS => Some(dot_iq2xs),
        GgmlType::IQ3XXS => Some(dot_iq3xxs),
        GgmlType::IQ4XS => Some(dot_iq4xs),
        GgmlType::IQ4NL => Some(dot_iq4nl),
        GgmlType::IQ3S => Some(dot_iq3s),
        GgmlType::Q6K => Some(dot_q6k),
        GgmlType::Q5K => Some(dot_q5k),
        GgmlType::Q8_0 => Some(dot_q8_0),
        GgmlType::IQ2XXS => Some(dot_iq2xxs),
        GgmlType::IQ2S => Some(dot_iq2s),
        GgmlType::IQ1S => Some(dot_iq1s),
        GgmlType::IQ1M => Some(dot_iq1m),
        GgmlType::Q2K => Some(dot_q2k),
        GgmlType::Q3K => Some(dot_q3k),
        GgmlType::Q4K => Some(dot_q4k),
        GgmlType::Q2_0 => Some(dot_q2_0),
        GgmlType::TQ1_0 => Some(dot_tq1_0),
        GgmlType::TQ2_0 => Some(dot_tq2_0),
        GgmlType::Q4_0 => Some(dot_q4::<2>),
        GgmlType::Q4_1 => Some(dot_q4::<3>),
        GgmlType::Q5_0 => Some(dot_q4::<6>),
        GgmlType::Q5_1 => Some(dot_q4::<7>),
        _ => None,
    }
}

#[inline(always)]
pub fn dot_f32_slice(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return neon::dot_f32(a, b);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return avx2::dot_f32(a, b);
        }
    }
    #[allow(unreachable_code)]
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline(always)]
pub fn fma_f32_slice(out: &mut [f32], v: &[f32], scale: f32) {
    #[cfg(target_arch = "aarch64")]
    {
        neon::fma_f32(out, v, scale);
        return;
    }
    #[allow(unreachable_code)]
    for (o, &vv) in out.iter_mut().zip(v) {
        *o += scale * vv;
    }
}

#[inline(always)]
fn dot_float_row(ty: GgmlType, row_bytes: &[u8], x: &[f32], buf: &mut [f32]) -> f32 {
    match ty {
        GgmlType::F32 => {
            // Unaligned-safe F32 read in chunks or direct slice when aligned
            if (row_bytes.as_ptr() as usize) % 4 == 0 {
                let s = unsafe {
                    std::slice::from_raw_parts(row_bytes.as_ptr() as *const f32, x.len())
                };
                dot_f32_slice(s, x)
            } else {
                for (o, c) in buf.iter_mut().zip(row_bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes(c.try_into().unwrap());
                }
                dot_f32_slice(buf, x)
            }
        }
        GgmlType::F16 => {
            for (o, c) in buf.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
            dot_f32_slice(buf, x)
        }
        GgmlType::BF16 => {
            for (o, c) in buf.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *o = bf16::from_le_bytes([c[0], c[1]]).to_f32();
            }
            dot_f32_slice(buf, x)
        }
        _ => {
            dequant_row(ty, row_bytes, buf);
            dot_f32_slice(buf, x)
        }
    }
}

/// Fast quantized or float matrix-vector product `y = t · x` using a pre-quantized `Q8K` input.
pub fn matvec_with_q8k(t: &Tensor, xq: &[Q8K], x: &[f32], y: &mut [f32]) {
    assert_eq!((x.len(), y.len()), (t.cols, t.rows));
    if t.rows == 0 || t.cols == 0 {
        return;
    }
    let rb = t.row_bytes();
    if let Some(dot) = int_dot(t.ty) {
        if t.rows * t.cols >= 16_384 && rayon::current_num_threads() > 1 {
            let chunk = if t.rows >= 4096 {
                64
            } else if t.rows >= 1024 {
                32
            } else {
                16
            };
            y.par_chunks_mut(chunk).enumerate().for_each(|(c, ys)| {
                let base = c * chunk;
                for (k, v) in ys.iter_mut().enumerate() {
                    let r = base + k;
                    *v = dot(&t.data[r * rb..(r + 1) * rb], xq);
                }
            });
        } else {
            for (r, v) in y.iter_mut().enumerate() {
                *v = dot(&t.data[r * rb..(r + 1) * rb], xq);
            }
        }
    } else if t.rows * t.cols >= 16_384 && rayon::current_num_threads() > 1 {
        let chunk = if t.rows >= 4096 { 32 } else { 16 };
        y.par_chunks_mut(chunk).enumerate().for_each_init(
            || vec![0f32; t.cols],
            |buf, (c, ys)| {
                let base = c * chunk;
                for (k, v) in ys.iter_mut().enumerate() {
                    let r = base + k;
                    *v = dot_float_row(t.ty, &t.data[r * rb..(r + 1) * rb], x, buf);
                }
            },
        );
    } else {
        let mut buf = vec![0f32; t.cols];
        for (r, v) in y.iter_mut().enumerate() {
            *v = dot_float_row(t.ty, &t.data[r * rb..(r + 1) * rb], x, &mut buf);
        }
    }
}

/// Fused SwiGLU / GeGLU MLP first stage: computes `act[r] = act_fn(gate[r] · x) * (up[r] · x)`
/// in a single parallel pass without intermediate vectors or duplicate `Q8K` quantization.
#[allow(clippy::too_many_arguments)]
pub fn fused_gate_up_act(
    gate: &Tensor,
    up: &Tensor,
    xq: &[Q8K],
    x: &[f32],
    act: &mut [f32],
    gate_scratch: &mut Vec<f32>,
    up_scratch: &mut Vec<f32>,
    is_gemma: bool,
) {
    let rows = up.rows;
    if let (Some(dg), Some(du)) = (int_dot(gate.ty), int_dot(up.ty)) {
        let rbg = gate.row_bytes();
        let rbu = up.row_bytes();
        let chunk = if rows >= 4096 { 64 } else if rows >= 1024 { 32 } else { 16 };
        if rows * up.cols >= 16_384 && rayon::current_num_threads() > 1 {
            act.par_chunks_mut(chunk).enumerate().for_each(|(c, out)| {
                let base = c * chunk;
                for (k, dst) in out.iter_mut().enumerate() {
                    let r = base + k;
                    let g = dg(&gate.data[r * rbg..(r + 1) * rbg], xq);
                    let u = du(&up.data[r * rbu..(r + 1) * rbu], xq);
                    *dst = if is_gemma {
                        crate::ops::gelu_tanh(g) * u
                    } else {
                        crate::ops::silu(g) * u
                    };
                }
            });
        } else {
            for (r, dst) in act.iter_mut().enumerate() {
                let g = dg(&gate.data[r * rbg..(r + 1) * rbg], xq);
                let u = du(&up.data[r * rbu..(r + 1) * rbu], xq);
                *dst = if is_gemma {
                    crate::ops::gelu_tanh(g) * u
                } else {
                    crate::ops::silu(g) * u
                };
            }
        }
    } else {
        gate_scratch.resize(rows, 0.0);
        up_scratch.resize(rows, 0.0);
        matvec_with_q8k(gate, xq, x, gate_scratch);
        matvec_with_q8k(up, xq, x, up_scratch);
        for ((dst, &g), &u) in act.iter_mut().zip(gate_scratch.iter()).zip(up_scratch.iter()) {
            *dst = if is_gemma {
                crate::ops::gelu_tanh(g) * u
            } else {
                crate::ops::silu(g) * u
            };
        }
    }
}

/// Fast quantized or float matrix-vector product `y = t · x`.
pub fn matvec(t: &Tensor, x: &[f32], y: &mut [f32]) {
    assert_eq!((x.len(), y.len()), (t.cols, t.rows));
    if t.rows == 0 || t.cols == 0 {
        return;
    }
    if int_dot(t.ty).is_some() {
        let xq = quantize_q8k(x);
        matvec_with_q8k(t, &xq, x, y);
    } else {
        matvec_with_q8k(t, &[], x, y);
    }
}

/// Fast batched matrix-vector product: reads every weight row once and dots it with all inputs `xs`.
pub fn rows_multi(t: &Tensor, xs: &[&[f32]]) -> Vec<Vec<f32>> {
    let nt = xs.len();
    if nt == 0 {
        return Vec::new();
    }
    if nt == 1 {
        let mut y = vec![0f32; t.rows];
        matvec(t, xs[0], &mut y);
        return vec![y];
    }
    let (rows, cols) = (t.rows, t.cols);
    let rb = t.row_bytes();
    let mut tmp = vec![0f32; rows * nt];
    if let Some(dot) = int_dot(t.ty) {
        let xq: Vec<Vec<Q8K>> = xs.iter().map(|x| quantize_q8k(x)).collect();
        if rows * cols >= 16_384 && rayon::current_num_threads() > 1 {
            tmp.par_chunks_mut(16 * nt).enumerate().for_each(|(ci, chunk)| {
                let n_rows_chunk = chunk.len() / nt;
                for r_off in 0..n_rows_chunk {
                    let row = ci * 16 + r_off;
                    let data = &t.data[row * rb..(row + 1) * rb];
                    for j in 0..nt {
                        chunk[r_off * nt + j] = dot(data, &xq[j]);
                    }
                }
            });
        } else {
            for r in 0..rows {
                let data = &t.data[r * rb..(r + 1) * rb];
                for j in 0..nt {
                    tmp[r * nt + j] = dot(data, &xq[j]);
                }
            }
        }
    } else if rows * cols >= 16_384 && rayon::current_num_threads() > 1 {
        tmp.par_chunks_mut(16 * nt).enumerate().for_each_init(
            || vec![0f32; cols],
            |buf, (ci, chunk)| {
                let n_rows_chunk = chunk.len() / nt;
                for r_off in 0..n_rows_chunk {
                    let row = ci * 16 + r_off;
                    let data = &t.data[row * rb..(row + 1) * rb];
                    if t.ty == GgmlType::F32 && (data.as_ptr() as usize) % 4 == 0 {
                        let s = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, cols)
                        };
                        for j in 0..nt {
                            chunk[r_off * nt + j] = dot_f32_slice(s, xs[j]);
                        }
                    } else {
                        dequant_row(t.ty, data, buf);
                        for j in 0..nt {
                            chunk[r_off * nt + j] = dot_f32_slice(buf, xs[j]);
                        }
                    }
                }
            },
        );
    } else {
        let mut buf = vec![0f32; cols];
        for r in 0..rows {
            let data = &t.data[r * rb..(r + 1) * rb];
            dequant_row(t.ty, data, &mut buf);
            for j in 0..nt {
                tmp[r * nt + j] = dot_f32_slice(&buf, xs[j]);
            }
        }
    }
    (0..nt)
        .map(|j| (0..rows).map(|r| tmp[r * nt + j]).collect())
        .collect()
}

pub fn batch_rows(mats: &[Tensor], xs: &[&[f32]], out: &mut [f32]) {
    if mats.is_empty() {
        return;
    }
    let rows = out.len() / mats.len();
    let dot = int_dot(mats[0].ty);
    let xq: Vec<Vec<Q8K>> = if dot.is_some() {
        xs.iter().map(|x| quantize_q8k(x)).collect()
    } else {
        vec![]
    };
    let batch = 16;
    out.par_chunks_mut(batch).enumerate().for_each_init(
        || {
            if dot.is_none() {
                vec![0f32; mats[0].cols]
            } else {
                Vec::new()
            }
        },
        |buf, (c, ys)| {
            for (k, v) in ys.iter_mut().enumerate() {
                let i = c * batch + k;
                let (t, r, x) = (&mats[i / rows], i % rows, (i / rows) * xs.len() / mats.len());
                let rb = t.row_bytes();
                *v = match dot {
                    Some(d) => d(&t.data[r * rb..(r + 1) * rb], &xq[x]),
                    None => dot_float_row(t.ty, &t.data[r * rb..(r + 1) * rb], xs[x], buf),
                };
            }
        },
    );
}

pub fn matvec_f32(t: &Tensor, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; t.rows];
    matvec(t, x, &mut y);
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_dots_match_dequant() {
        let mut r = 0x9E3779B97F4A7C15u64;
        let mut rnd = || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r as u8
        };
        let x: Vec<f32> = (0..1024)
            .map(|i| ((i as f32 * 0.7548777).fract() - 0.5) * 2.0)
            .collect();
        for ty in [
            GgmlType::Q2_0,
            GgmlType::TQ1_0,
            GgmlType::TQ2_0,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::IQ4NL,
            GgmlType::IQ4XS,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::IQ1S,
            GgmlType::IQ1M,
            GgmlType::IQ2XXS,
            GgmlType::IQ2XS,
            GgmlType::IQ2S,
            GgmlType::IQ3XXS,
            GgmlType::IQ3S,
        ] {
            let n = if ty == GgmlType::Q2_0 { 640 } else { 1024 };
            let xq = quantize_q8k(&x[..n]);
            let mut raw: Vec<u8> = (0..(n / ty.block_elems()) * ty.block_bytes())
                .map(|_| rnd())
                .collect();
            let fs: &[usize] = match ty {
                GgmlType::Q6K => &[208],
                GgmlType::Q2K => &[80, 82],
                GgmlType::Q3K => &[108],
                GgmlType::TQ1_0 => &[52],
                GgmlType::TQ2_0 => &[64],
                GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_1 | GgmlType::Q5_1 => &[0, 2],
                GgmlType::IQ1M => &[],
                _ => &[0],
            };
            for b in raw.chunks_exact_mut(ty.block_bytes()) {
                for &o in fs {
                    b[o..o + 2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes());
                }
            }
            if ty == GgmlType::IQ1M {
                let d = f16::from_f32(0.01).to_bits();
                for b in raw.chunks_exact_mut(56) {
                    for i in 0..4 {
                        b[49 + 2 * i] = (b[49 + 2 * i] & 0xF) | (((d >> (4 * i)) as u8 & 0xF) << 4);
                    }
                }
            }
            let mut w = vec![0f32; n];
            dequant_row(ty, &raw, &mut w);
            let want: f64 = w
                .iter()
                .enumerate()
                .map(|(i, &v)| v as f64 * (xq[i / 256].d * xq[i / 256].qs[i % 256] as f32) as f64)
                .sum();
            let got = int_dot(ty).unwrap()(&raw, &xq);
            assert!(
                ((got as f64 - want) / want.abs().max(1e-3)).abs() < 1e-5,
                "{}: {got} vs {want}",
                ty.name()
            );
        }
    }

    #[test]
    fn rows_multi_matches_matvec() {
        let cols = 512usize;
        let rows = 64usize;
        let mut raw = vec![0u8; rows * 4 * cols];
        for (i, ch) in raw.chunks_exact_mut(4).enumerate() {
            let v = ((i % 73) as f32 - 36.0) * 0.05;
            ch.copy_from_slice(&v.to_le_bytes());
        }
        let t = Tensor {
            ty: GgmlType::F32,
            cols,
            rows,
            data: &raw,
        };
        let x0: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.1).sin()).collect();
        let x1: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.2).cos()).collect();
        let y0 = matvec_f32(&t, &x0);
        let y1 = matvec_f32(&t, &x1);
        let ym = rows_multi(&t, &[&x0, &x1]);
        assert_eq!(ym[0], y0);
        assert_eq!(ym[1], y1);
    }

    #[test]
    fn fused_gate_up_and_fma_match_reference() {
        let cols = 256usize;
        let rows = 32usize;
        let mut raw_g = vec![0u8; rows * 34 * (cols / 32)];
        let mut raw_u = vec![0u8; rows * 34 * (cols / 32)];
        for (i, b) in raw_g.chunks_exact_mut(34).enumerate() {
            b[0..2].copy_from_slice(&f16::from_f32(0.05).to_le_bytes());
            for j in 0..32 {
                b[2 + j] = (((i * 7 + j * 3) % 21) as i8 - 10) as u8;
            }
        }
        for (i, b) in raw_u.chunks_exact_mut(34).enumerate() {
            b[0..2].copy_from_slice(&f16::from_f32(0.04).to_le_bytes());
            for j in 0..32 {
                b[2 + j] = (((i * 5 + j * 11) % 19) as i8 - 9) as u8;
            }
        }
        let tg = Tensor { ty: GgmlType::Q8_0, cols, rows, data: &raw_g };
        let tu = Tensor { ty: GgmlType::Q8_0, cols, rows, data: &raw_u };
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.07).sin()).collect();

        let mut g_ref = vec![0f32; rows];
        let mut u_ref = vec![0f32; rows];
        matvec(&tg, &x, &mut g_ref);
        matvec(&tu, &x, &mut u_ref);

        let mut xq = Vec::new();
        quantize_q8k_into(&x, &mut xq);
        let mut act = vec![0f32; rows];
        let (mut sg, mut su) = (Vec::new(), Vec::new());
        fused_gate_up_act(&tg, &tu, &xq, &x, &mut act, &mut sg, &mut su, false);

        for r in 0..rows {
            let want = crate::ops::silu(g_ref[r]) * u_ref[r];
            assert!((act[r] - want).abs() < 1e-5, "row {r}: {} vs {want}", act[r]);
        }

        let mut acc = vec![1.0f32; 37];
        let v: Vec<f32> = (0..37).map(|i| i as f32 * 0.25).collect();
        fma_f32_slice(&mut acc, &v, 0.5);
        for (i, &val) in acc.iter().enumerate() {
            assert!((val - (1.0 + 0.5 * v[i])).abs() < 1e-6);
        }
    }
}
