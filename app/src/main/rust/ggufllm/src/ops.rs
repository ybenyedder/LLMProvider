//! Scalar & SIMD-friendly f32 reference math and token sampler.

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(w) {
        *o = v * s * w;
    }
}

pub fn rms_norm_unit(x: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ms + eps).sqrt();
    for (o, v) in out.iter_mut().zip(x) {
        *o = v * s;
    }
}

pub fn l2_norm(x: &mut [f32], eps: f32) {
    let s = 1.0 / x.iter().map(|v| v * v).sum::<f32>().sqrt().max(eps);
    for v in x {
        *v *= s;
    }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[inline(always)]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const A: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + A * x * x)).tanh())
}

pub fn softmax(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if s > 0.0 {
        let inv = 1.0 / s;
        for v in x {
            *v *= inv;
        }
    }
}

pub fn softmax_sink(x: &mut [f32], sink: Option<f32>) {
    let Some(sink) = sink else {
        return softmax(x);
    };
    let m = x.iter().cloned().fold(sink, f32::max);
    let mut s = (sink - m).exp();
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    if s > 0.0 {
        let inv = 1.0 / s;
        for v in x {
            *v *= inv;
        }
    }
}

/// NEOX-style rotary embedding on the first `n_rot` dims: pairs `(i, i + n_rot/2)`.
pub fn rope_neox(x: &mut [f32], n_rot: usize, pos: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f64 * (base as f64).powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = (theta.sin() as f32, theta.cos() as f32);
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

/// Standard interleaved rotary embedding (Llama/Mistral): pairs `(2*i, 2*i + 1)`.
pub fn rope_norm(x: &mut [f32], n_rot: usize, pos: usize, base: f32) {
    let half = n_rot / 2;
    for i in 0..half {
        let theta = pos as f64 * (base as f64).powf(-2.0 * i as f64 / n_rot as f64);
        let (s, c) = (theta.sin() as f32, theta.cos() as f32);
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// Text-model M-RoPE: pairs `(i, i + n_rot/2)`.
pub fn rope_mrope(
    x: &mut [f32],
    n_rot: usize,
    pos: [i32; 4],
    sections: [usize; 4],
    base: f32,
) {
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let mut ends = [0usize; 4];
    let mut acc = 0;
    for i in 0..4 {
        acc += sections[i];
        ends[i] = acc;
    }
    for i in 0..half {
        let sec = if i < ends[0] {
            0
        } else if i < ends[1] {
            1
        } else if i < ends[2] {
            2
        } else {
            3
        };
        let theta = pos[sec] as f32 * theta_scale.powi(i as i32);
        let (s, c) = (theta.sin(), theta.cos());
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

pub fn top_k(p: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

pub fn argmax(x: &[f32]) -> usize {
    (1..x.len()).fold(0, |b, i| if x[i].total_cmp(&x[b]).is_gt() { i } else { b })
}

const NON_FINITE: &str = "the forward pass produced NaN or infinite logits";

#[derive(Clone, Debug)]
pub struct Sampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: usize,
    pub rng: u64,
}

impl Sampler {
    pub fn greedy() -> Self {
        Sampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.10,
            penalty_last_n: 64,
            rng: 0x9E3779B97F4A7C15,
        }
    }

    pub fn apply_repeat_penalty(&self, logits: &mut [f32], recent: &[u32]) {
        if self.repeat_penalty <= 1.0 || recent.is_empty() {
            return;
        }
        let n = recent.len().min(self.penalty_last_n);
        let tail = &recent[recent.len() - n..];
        let mut seen = Vec::with_capacity(n);
        for &t in tail {
            let idx = t as usize;
            if idx < logits.len() && !seen.contains(&idx) {
                seen.push(idx);
                if logits[idx] <= 0.0 {
                    logits[idx] *= self.repeat_penalty;
                } else {
                    logits[idx] /= self.repeat_penalty;
                }
            }
        }
    }

    pub fn pick(&mut self, logits: &mut [f32], recent: &[u32]) -> anyhow::Result<usize> {
        self.apply_repeat_penalty(logits, recent);
        if self.temp <= 0.0 {
            let i = argmax(logits);
            anyhow::ensure!(logits[i].is_finite(), NON_FINITE);
            return Ok(i);
        }
        let c = self.cands(logits);
        self.draw(&c)
    }

    fn cands(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let ch = logits.chunks_exact(16);
        let tail = ch
            .remainder()
            .iter()
            .fold(f32::NEG_INFINITY, |a, &x| a.max(x));
        let m = ch
            .fold([f32::NEG_INFINITY; 16], |mut a, c| {
                for j in 0..16 {
                    a[j] = a[j].max(c[j]);
                }
                a
            })
            .into_iter()
            .fold(tail, f32::max);
        let lo = m - 21.0 * self.temp;
        let mut c: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter(|x| *x.1 >= lo)
            .map(|(i, &l)| (i, (l - m) / self.temp))
            .filter(|x| x.1 > -20.0)
            .map(|(i, x)| (i, x.exp()))
            .collect();
        c.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if self.top_k > 0 {
            c.truncate(self.top_k);
        }
        let total: f32 = c.iter().map(|x| x.1).sum();
        let (mut acc, mut n) = (0.0, c.len());
        for (j, x) in c.iter().enumerate() {
            acc += x.1;
            if acc >= self.top_p * total {
                n = j + 1;
                break;
            }
        }
        c.truncate(n);
        c
    }

    fn draw(&mut self, c: &[(usize, f32)]) -> anyhow::Result<usize> {
        anyhow::ensure!(!c.is_empty(), NON_FINITE);
        let mut r = self.rand() * c.iter().map(|x| x.1).sum::<f32>();
        for &(i, p) in c {
            r -= p;
            if r < 0.0 {
                return Ok(i);
            }
        }
        Ok(c[c.len() - 1].0)
    }

    fn rand(&mut self) -> f32 {
        if self.rng == 0 {
            self.rng = 0x9E3779B97F4A7C15;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }
}
