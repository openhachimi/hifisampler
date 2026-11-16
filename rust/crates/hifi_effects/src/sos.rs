//! Butterworth SOS filter design, Direct-Form II Transposed `sosfilt`,
//! steady-state step initialization `sosfilt_zi`, and zero-phase `sosfiltfilt`.
//!
//! Clean-room implementation matching `scipy.signal.butter(..., output="sos")`,
//! `scipy.signal.sosfilt`, `scipy.signal.sosfilt_zi`, and `scipy.signal.sosfiltfilt`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{LazyLock, Mutex};

/// One Second-Order Section: `[b0, b1, b2, a0, a1, a2]`.
pub type SosSection = [f64; 6];

/// Butterworth filter passband type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ButterBand {
    Lowpass,
    Highpass,
}

#[derive(Debug, Clone, Copy)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denom,
            (self.im * rhs.re - self.re * rhs.im) / denom,
        )
    }

    #[inline]
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    #[inline]
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }

    #[inline]
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    #[inline]
    fn is_real(self) -> bool {
        self.im.abs() <= 1e-12
    }
}

type ButterSosKey = (usize, u64, ButterBand);
type ButterSosCache = Mutex<HashMap<ButterSosKey, Vec<SosSection>>>;

static BUTTER_SOS_CACHE: LazyLock<ButterSosCache> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// Returns cached digital Butterworth SOS sections for `(order, norm_freq, band)`.
///
/// `norm_freq` is normalized to Nyquist (`0.0 < norm_freq < 1.0`, i.e., `cutoff_hz / (sr / 2)`).
pub fn cached_butter_sos(order: usize, norm_freq: f64, band: ButterBand) -> Vec<SosSection> {
    let clamped = norm_freq.clamp(1e-9, 1.0 - 1e-9);
    let key = (order, clamped.to_bits(), band);
    {
        let cache = BUTTER_SOS_CACHE.lock().unwrap();
        if let Some(sos) = cache.get(&key) {
            return sos.clone();
        }
    }
    let sos = butter_sos(order, clamped, band);
    let mut cache = BUTTER_SOS_CACHE.lock().unwrap();
    if cache.len() >= 128 {
        cache.clear();
    }
    cache.entry(key).or_insert(sos).clone()
}

/// Designs an `order`-th order digital Butterworth filter in SOS form.
///
/// Matches `scipy.signal.butter(order, norm_freq, btype=..., output="sos")`.
pub fn butter_sos(order: usize, norm_freq: f64, band: ButterBand) -> Vec<SosSection> {
    if order == 0 {
        return vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]];
    }
    let wn = norm_freq.clamp(1e-9, 1.0 - 1e-9);

    // 1. Analog prototype (buttap)
    let mut proto_poles = Vec::with_capacity(order);
    for idx in 0..order {
        let m = -(order as isize) + 1 + 2 * (idx as isize);
        let theta = PI * (m as f64) / (2.0 * order as f64);
        proto_poles.push(C64::new(-theta.cos(), -theta.sin()));
    }

    // 2. Frequency pre-warp (fs = 2.0)
    let fs = 2.0_f64;
    let warped = 2.0 * fs * (PI * wn / fs).tan();

    // 3. Analog lowpass -> lowpass / highpass ZPK
    let mut z_analog = Vec::new();
    let mut p_analog = Vec::with_capacity(order);

    let k_analog: f64 = match band {
        ButterBand::Lowpass => {
            for &p in &proto_poles {
                p_analog.push(p.scale(warped));
            }
            warped.powi(order as i32)
        }
        ButterBand::Highpass => {
            z_analog.resize(order, C64::new(0.0, 0.0));
            let mut prod_neg_p = C64::new(1.0, 0.0);
            for &p in &proto_poles {
                p_analog.push(C64::new(warped, 0.0).div(p));
                prod_neg_p = prod_neg_p.mul(p.neg());
            }
            1.0 / prod_neg_p.re
        }
    };

    // 4. Bilinear transform (fs = 2.0 => fs2 = 4.0)
    let fs2 = 2.0 * fs;
    let four = C64::new(fs2, 0.0);
    let degree = p_analog.len().saturating_sub(z_analog.len());

    let mut z_dig = Vec::with_capacity(order);
    let mut num_k = C64::new(1.0, 0.0);
    for &z in &z_analog {
        z_dig.push(four.add(z).div(four.sub(z)));
        num_k = num_k.mul(four.sub(z));
    }
    for _ in 0..degree {
        z_dig.push(C64::new(-1.0, 0.0));
    }

    let mut p_dig = Vec::with_capacity(order);
    let mut den_k = C64::new(1.0, 0.0);
    for &p in &p_analog {
        p_dig.push(four.add(p).div(four.sub(p)));
        den_k = den_k.mul(four.sub(p));
    }

    let k_dig = k_analog * num_k.div(den_k).re;

    // 5. ZPK to SOS (pairing = "nearest")
    zpk2sos(&z_dig, &p_dig, k_dig)
}

/// Converts discrete zeros, poles, and gain to cascaded Second-Order Sections (`pairing = "nearest"`).
fn zpk2sos(zeros: &[C64], poles: &[C64], gain: f64) -> Vec<SosSection> {
    if zeros.is_empty() && poles.is_empty() {
        return vec![[gain, 0.0, 0.0, 1.0, 0.0, 0.0]];
    }

    let mut p_work = poles.to_vec();
    let mut z_work = zeros.to_vec();
    while p_work.len() < z_work.len() {
        p_work.push(C64::new(0.0, 0.0));
    }
    while z_work.len() < p_work.len() {
        z_work.push(C64::new(0.0, 0.0));
    }
    if p_work.len() % 2 == 1 {
        p_work.push(C64::new(0.0, 0.0));
        z_work.push(C64::new(0.0, 0.0));
    }

    let n_sections = p_work.len() / 2;
    let mut p_reps = collapse_conjugates(&p_work);
    let mut z_reps = collapse_conjugates(&z_work);

    let mut sos = vec![[0.0; 6]; n_sections];

    for si in (0..n_sections).rev() {
        let p1_idx = worst_pole_idx(&p_reps);
        let p1 = p_reps.remove(p1_idx);

        let p2 = if p1.is_real() {
            // Find next real pole closest to the unit circle
            let mut best_idx = None;
            let mut best_dist = f64::INFINITY;
            for (idx, &p) in p_reps.iter().enumerate() {
                if p.is_real() {
                    let d = (1.0 - p.abs()).abs();
                    if d < best_dist {
                        best_dist = d;
                        best_idx = Some(idx);
                    }
                }
            }
            if let Some(idx) = best_idx {
                p_reps.remove(idx)
            } else {
                C64::new(0.0, 0.0)
            }
        } else {
            p1.conj()
        };

        let (z1, z2) = if !z_reps.is_empty() {
            let z1_idx = nearest_zero_idx(&z_reps, p1, false);
            let z1 = z_reps.remove(z1_idx);
            if !z1.is_real() {
                (z1, z1.conj())
            } else if !z_reps.is_empty() {
                let z2_idx = nearest_zero_idx(&z_reps, p1, true);
                let z2 = z_reps.remove(z2_idx);
                (z1, z2)
            } else {
                (z1, C64::new(0.0, 0.0))
            }
        } else {
            (C64::new(0.0, 0.0), C64::new(0.0, 0.0))
        };

        sos[si] = pair_to_sos(z1, z2, p1, p2);
    }

    sos[0][0] *= gain;
    sos[0][1] *= gain;
    sos[0][2] *= gain;
    sos
}

/// Keeps one representative per complex conjugate pair (with `im > 0`) plus all real roots.
fn collapse_conjugates(roots: &[C64]) -> Vec<C64> {
    let mut used = vec![false; roots.len()];
    let mut out = Vec::new();
    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        let r = roots[i];
        if r.is_real() {
            used[i] = true;
            out.push(C64::new(r.re, 0.0));
        } else {
            // Find matching conjugate
            let target = r.conj();
            let mut match_idx = None;
            let mut best = 1e-8;
            for j in (i + 1)..roots.len() {
                if !used[j] && roots[j].sub(target).abs() < best {
                    best = roots[j].sub(target).abs();
                    match_idx = Some(j);
                }
            }
            used[i] = true;
            if let Some(j) = match_idx {
                used[j] = true;
            }
            out.push(if r.im >= 0.0 { r } else { r.conj() });
        }
    }
    out
}

fn worst_pole_idx(poles: &[C64]) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (idx, &p) in poles.iter().enumerate() {
        let d = (1.0 - p.abs()).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = idx;
        }
    }
    best_idx
}

fn nearest_zero_idx(zeros: &[C64], pole: C64, real_only: bool) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (idx, &z) in zeros.iter().enumerate() {
        if real_only && !z.is_real() {
            continue;
        }
        let d = z.sub(pole).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = idx;
        }
    }
    best_idx
}

#[inline]
fn pair_to_sos(z1: C64, z2: C64, p1: C64, p2: C64) -> SosSection {
    let b0 = 1.0;
    let b1 = -(z1.re + z2.re);
    let b2 = z1.mul(z2).re;
    let a0 = 1.0;
    let a1 = -(p1.re + p2.re);
    let a2 = p1.mul(p2).re;
    [b0, b1, b2, a0, a1, a2]
}

/// Computes step-response steady-state initial filter states `zi` for `sosfilt`,
/// matching `scipy.signal.sosfilt_zi(sos)`.
pub fn sosfilt_zi(sos: &[SosSection]) -> Vec<[f64; 2]> {
    let mut zi = Vec::with_capacity(sos.len());
    let mut scale = 1.0_f64;
    for sec in sos {
        let a0 = sec[3];
        let b = [sec[0] / a0, sec[1] / a0, sec[2] / a0];
        let a = [1.0, sec[4] / a0, sec[5] / a0];
        let sum_b = b[0] + b[1] + b[2];
        let sum_a = a[0] + a[1] + a[2];
        let y_inf = if sum_a.abs() > 1e-18 {
            sum_b / sum_a
        } else {
            0.0
        };
        let c1 = b[1] - y_inf * a[1];
        let c2 = b[2] - y_inf * a[2];
        zi.push([scale * (c1 + c2), scale * c2]);
        scale *= y_inf;
    }
    zi
}

/// Filters `x` using cascaded Second-Order Sections in Direct-Form II Transposed structure.
///
/// Matches `scipy.signal.sosfilt(sos, x, zi=zi)` in `f64` precision.
pub fn sosfilt(sos: &[SosSection], x: &[f64], zi: Option<&[[f64; 2]]>) -> Vec<f64> {
    let mut state = match zi {
        Some(init) => init.to_vec(),
        None => vec![[0.0, 0.0]; sos.len()],
    };
    let mut out = Vec::with_capacity(x.len());
    for &xi in x {
        let mut cur = xi;
        for (s, sec) in sos.iter().enumerate() {
            let a0 = sec[3];
            let (b0, b1, b2, a1, a2) = if a0 == 1.0 {
                (sec[0], sec[1], sec[2], sec[4], sec[5])
            } else {
                (
                    sec[0] / a0,
                    sec[1] / a0,
                    sec[2] / a0,
                    sec[4] / a0,
                    sec[5] / a0,
                )
            };
            let y = b0 * cur + state[s][0];
            state[s][0] = b1 * cur - a1 * y + state[s][1];
            state[s][1] = b2 * cur - a2 * y;
            cur = y;
        }
        out.push(cur);
    }
    out
}

/// Applies forward-backward zero-phase SOS filtering with odd boundary extension,
/// matching `scipy.signal.sosfiltfilt(sos, x)`.
///
/// Returns `None` if `x` is too short for the default `padlen` (`x.len() <= edge`).
pub fn sosfiltfilt(sos: &[SosSection], x: &[f64]) -> Option<Vec<f64>> {
    if sos.is_empty() || x.is_empty() {
        return Some(x.to_vec());
    }
    let n_sections = sos.len();
    let z_b2 = sos.iter().filter(|s| s[2] == 0.0).count();
    let z_a2 = sos.iter().filter(|s| s[5] == 0.0).count();
    let ntaps = 2 * n_sections + 1 - z_b2.min(z_a2);
    let edge = ntaps * 3;
    if x.len() <= edge {
        return None;
    }

    // Odd extension at both boundaries
    let n = x.len();
    let mut ext = Vec::with_capacity(n + 2 * edge);
    let left_end = x[0];
    for k in (1..=edge).rev() {
        ext.push(2.0 * left_end - x[k]);
    }
    ext.extend_from_slice(x);
    let right_end = x[n - 1];
    for k in 1..=edge {
        ext.push(2.0 * right_end - x[n - 1 - k]);
    }

    let base_zi = sosfilt_zi(sos);
    let mut zi_fwd = base_zi.clone();
    let x0 = ext[0];
    for z in &mut zi_fwd {
        z[0] *= x0;
        z[1] *= x0;
    }
    let mut y = sosfilt(sos, &ext, Some(&zi_fwd));
    y.reverse();

    let mut zi_bwd = base_zi;
    let y0 = y[0];
    for z in &mut zi_bwd {
        z[0] *= y0;
        z[1] *= y0;
    }
    let mut y2 = sosfilt(sos, &y, Some(&zi_bwd));
    y2.reverse();

    Some(y2[edge..(edge + n)].to_vec())
}
