//! Ht v3 continuous-time Fant LF glottal tension engine, Ho/Hr Klatt 2nd-order
//! resonators, minimum-phase cepstral folding, psychoacoustic middle-ear loudness
//! compensation, and voiced-only aspiration noise coupling.
//!
//! Clean-room port of `util/audio.py::pre_emphasis_base_tension` and
//! `util/audio.py::couple_aperiodic_to_tension`.

use std::f64::consts::PI;
use std::sync::OnceLock;

use ndarray::Array2;
use num_complex::{Complex32, Complex64};

use crate::hnsep_stft::{
    constant_pad_1d, irfft_real_1d, istft_1d, periodic_hann_window, rfft_1d, stft_1d, PadMode,
};
use crate::DspError;

/// Configuration for the Ht v3 / Ho / Hr tension & aspiration coupling pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensionConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
}

impl Default for TensionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            n_fft: 2048,
            hop_size: 512,
            win_size: 2048,
        }
    }
}

impl TensionConfig {
    fn validate(&self) -> Result<(), DspError> {
        if self.sample_rate == 0 || self.n_fft == 0 || self.hop_size == 0 || self.win_size == 0 {
            return Err(DspError::InvalidParameter {
                name: "tension_config",
                reason: "sample_rate, n_fft, hop_size, and win_size must be > 0".to_string(),
            });
        }
        if self.win_size > self.n_fft || self.n_fft < 4 {
            return Err(DspError::InvalidParameter {
                name: "tension_config",
                reason: format!(
                    "require n_fft >= 4 and win_size ({}) <= n_fft ({})",
                    self.win_size, self.n_fft
                ),
            });
        }
        Ok(())
    }
}

struct LfGlottalTable {
    rd_grid: Vec<f32>,
    log_table: Array2<f32>,       // [96, 256]
    power_templates: Array2<f32>, // [96, 80]
    psi_grid: Vec<f32>,           // [96]
}

static LF_GLOTTAL_TABLE: OnceLock<LfGlottalTable> = OnceLock::new();

/// Solves normalized continuous-time Fant (1995) LF glottal pulse parameters on unit period
/// `T0 = 1.0` for `Rd in [0.06, 3.00]`, returning `(te, ta, wg, eps, alpha)`.
pub fn solve_lf_unit_period_params(rd: f64) -> (f64, f64, f64, f64, f64) {
    let rd = rd.clamp(0.06, 3.00);
    let tau = 0.035_f64;
    let x_s = (rd - 0.2083333) / tau;
    let softplus_val = tau * (x_s.max(0.0) + (-x_s.abs()).exp().ln_1p());
    let ra = 0.0008 + 0.048 * softplus_val;
    let rk = 0.224 + 0.118 * rd;
    let denom = (0.11 * rd) / (0.5 + 1.2 * rk) - ra;
    let rg = (0.25 * rk) / denom.max(1e-4);

    let tp = 1.0 / (2.0 * rg);
    let ta = ra;
    let te = (tp * (1.0 + rk)).min(1.0 - ta - 1e-4);
    let wg = PI / tp;
    let tc = (1.0 - te).max(1e-5);

    // 1. Return-phase decay rate eps: 1 - eps * ta - exp(-eps * tc) = 0
    let eps = if tc <= ta + 1e-7 {
        1.0 / ta.max(1e-6)
    } else {
        let mut lo = 1e-4_f64;
        let mut hi = 1.0 / ta.max(1e-6);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if 1.0 - mid * ta - (-mid * tc).exp() > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };

    let exp_ret = (-eps * tc).exp();
    let i_ret = (1.0 - exp_ret) / (eps * eps * ta) - (tc * exp_ret) / (eps * ta);

    // 2. Opening-phase growth rate alpha: strictly decreasing A_open(alpha) == i_ret
    let sin_wte = (wg * te).sin();
    let cos_wte = (wg * te).cos();

    let a_open = |a: f64| -> f64 {
        let exp_neg = if a >= -50.0 {
            (-a * te).exp()
        } else {
            (-a * te).min(60.0).exp()
        };
        (-wg * exp_neg - a * sin_wte + wg * cos_wte) / (sin_wte * (a * a + wg * wg))
    };

    let mut a_lo = -1.0_f64;
    while a_open(a_lo) < i_ret && a_lo > -500.0 {
        a_lo *= 2.0;
    }
    let mut a_hi = 1.0_f64;
    while a_open(a_hi) > i_ret && a_hi < 1e6 {
        a_hi *= 2.0;
    }
    for _ in 0..52 {
        let a_mid = 0.5 * (a_lo + a_hi);
        if a_open(a_mid) > i_ret {
            a_lo = a_mid;
        } else {
            a_hi = a_mid;
        }
    }
    let alpha = 0.5 * (a_lo + a_hi);
    (te, ta, wg, eps, alpha)
}

fn get_lf_glottal_table() -> &'static LfGlottalTable {
    LF_GLOTTAL_TABLE.get_or_init(|| {
        const N_RD: usize = 96;
        const K_MAX: usize = 256;
        let mut rd_grid_f64 = Vec::with_capacity(N_RD);
        let step = (3.00_f64 - 0.07_f64) / ((N_RD - 1) as f64);
        for i in 0..N_RD {
            rd_grid_f64.push(if i == N_RD - 1 {
                3.00
            } else {
                0.07 + (i as f64) * step
            });
        }

        let mut log_table_f64 = Array2::<f64>::zeros((N_RD, K_MAX));
        for (i, &rd) in rd_grid_f64.iter().enumerate() {
            let (te, ta, wg, eps, alpha) = solve_lf_unit_period_params(rd);
            let tc = (1.0 - te).max(1e-5);
            let sin_wte = (wg * te).sin();
            let cos_wte = (wg * te).cos();
            let neg_inv_sin = -1.0 / sin_wte;
            let exp_neg_ate = if alpha >= -50.0 {
                (-alpha * te).exp()
            } else {
                (-alpha * te).min(60.0).exp()
            };
            let exp_ret = (-eps * tc).exp();

            let mut g_mag = Vec::with_capacity(K_MAX);
            for k_idx in 0..K_MAX {
                let k = (k_idx + 1) as f64;
                let w = 2.0 * PI * k;
                let jw = Complex64::new(0.0, w);
                let e_j_wte = Complex64::from_polar(1.0, -w * te);
                let e_pos_jwte = Complex64::from_polar(1.0, w * te);

                let a_minus_jw = Complex64::new(alpha, -w);
                let g_open = (Complex64::new(wg * exp_neg_ate, 0.0)
                    + e_j_wte * (a_minus_jw * sin_wte - Complex64::new(wg * cos_wte, 0.0)))
                    * neg_inv_sin
                    / (a_minus_jw * a_minus_jw + wg * wg);

                let term1 =
                    (Complex64::new(1.0, 0.0) - e_pos_jwte * exp_ret) / Complex64::new(eps, w);
                let term2 = ((Complex64::new(1.0, 0.0) - e_pos_jwte) * exp_ret) / jw;
                let g_ret = (e_j_wte / (eps * ta)) * (term1 - term2);

                g_mag.push((g_open + g_ret).norm() / k);
            }

            let g1 = g_mag[0].max(1e-12);
            log_table_f64[[i, 0]] = 0.0;
            for k_idx in 1..K_MAX {
                let g_tilde = (g_mag[k_idx] / g1).max(1e-6);
                log_table_f64[[i, k_idx]] = g_tilde.ln();
            }
        }

        // Arc-length equalized perceptual tilt coordinate psi_grid in [0, 1] over harmonics k=2..24
        let mut psi_cum = Vec::with_capacity(N_RD);
        psi_cum.push(0.0_f64);
        let mut acc = 0.0_f64;
        for i in 0..(N_RD - 1) {
            let mut sum_sq = 0.0_f64;
            for k_idx in 1..24 {
                let d = log_table_f64[[i + 1, k_idx]] - log_table_f64[[i, k_idx]];
                sum_sq += d * d;
            }
            let step_norm = (sum_sq / 23.0).sqrt() + 1e-5;
            acc += step_norm;
            psi_cum.push(acc);
        }
        let psi_max = *psi_cum.last().expect("non-empty");
        let psi_grid: Vec<f32> = psi_cum.iter().map(|&v| (v / psi_max) as f32).collect();

        let mut log_table = Array2::<f32>::zeros((N_RD, K_MAX));
        let mut power_templates = Array2::<f32>::zeros((N_RD, 80));
        for i in 0..N_RD {
            for k_idx in 0..K_MAX {
                let l = log_table_f64[[i, k_idx]];
                log_table[[i, k_idx]] = l as f32;
                if k_idx < 80 {
                    power_templates[[i, k_idx]] = (2.0 * l).exp() as f32;
                }
            }
        }
        let rd_grid: Vec<f32> = rd_grid_f64.iter().map(|&v| v as f32).collect();

        LfGlottalTable {
            rd_grid,
            log_table,
            power_templates,
            psi_grid,
        }
    })
}

/// Psychoacoustic middle-ear / equal-loudness frequency weighting `W(f)`.
pub fn middle_ear_loudness_weight(freqs: &[f32]) -> Vec<f32> {
    freqs
        .iter()
        .map(|&f| {
            let fc = (f as f64).max(300.0);
            let fc2 = fc * fc;
            let num = (fc2 * fc2) * (fc2 + 1200.0 * 1200.0);
            let den = (fc2 + 400.0 * 400.0).powi(2) * (fc2 + 3100.0 * 3100.0);
            (num / den.max(1e-20)).powf(0.33) as f32
        })
        .collect()
}

/// Estimates source fundamental frequency `f0_src` clamped in `[95.0, 580.0]` Hz
/// from the voiced-frame mean power spectrum via 1D Wiener-Khinchin autocorrelation.
pub fn estimate_harmonic_f0(mean_power: &[f32], sr: u32, n_fft: usize) -> Result<f64, DspError> {
    let total_power: f64 = mean_power.iter().map(|&x| x as f64).sum();
    if total_power < 1e-10 {
        return Ok(240.0);
    }

    let autocorr = irfft_real_1d(mean_power, n_fft)?;
    let r0 = autocorr[0] as f64;
    if r0 <= 1e-10 {
        return Ok(240.0);
    }

    let lag_min = ((sr as f64 / 580.0).floor() as usize).max(2);
    let lag_max = ((sr as f64 / 95.0).floor() as usize).min(n_fft / 2 - 2);
    if lag_max <= lag_min + 2 {
        return Ok(240.0);
    }

    let mut max_val = f64::NEG_INFINITY;
    let mut argmax = lag_min;
    for (lag, &ac) in autocorr.iter().enumerate().take(lag_max + 1).skip(lag_min) {
        let v = ac as f64;
        if v > max_val {
            max_val = v;
            argmax = lag;
        }
    }
    if max_val < 0.25 * r0 {
        return Ok(240.0);
    }

    let threshold = 0.82 * max_val;
    let mut tau = argmax;
    for lag in lag_min..=lag_max {
        let v = autocorr[lag] as f64;
        if v >= autocorr[lag - 1] as f64 && v >= autocorr[lag + 1] as f64 && v >= threshold {
            tau = lag;
            break;
        }
    }

    let tau_interp = if tau >= 1 && tau < n_fft / 2 - 1 {
        let y_prev = autocorr[tau - 1] as f64;
        let y_curr = autocorr[tau] as f64;
        let y_next = autocorr[tau + 1] as f64;
        let denom = y_prev - 2.0 * y_curr + y_next;
        if denom.abs() > 1e-12 {
            let delta = 0.5 * (y_prev - y_next) / denom;
            tau as f64 + delta.clamp(-0.5, 0.5)
        } else {
            tau as f64
        }
    } else {
        tau as f64
    };

    Ok((sr as f64 / tau_interp.max(1.0)).clamp(95.0, 580.0))
}

/// Extracts zero-macroscopic-tilt vocal-tract formant log-envelope (quefrency `q in [3, sr/1150)`).
pub fn extract_cepstral_formant_envelope(
    mean_power: &[f32],
    sr: u32,
    n_fft: usize,
) -> Result<Vec<f32>, DspError> {
    let total: f64 = mean_power.iter().map(|&x| x as f64).sum();
    if total < 1e-10 {
        return Ok(vec![0.0_f32; mean_power.len()]);
    }

    let log_spec_mean: Vec<f32> = mean_power
        .iter()
        .map(|&p| 0.5 * p.max(1e-12).ln())
        .collect();
    let cep = irfft_real_1d(&log_spec_mean, n_fft)?;

    let q_low = 3usize;
    let q_high = ((sr as f64 / 1150.0).floor() as usize)
        .max(12)
        .min(n_fft / 2 - 1);
    let mut cep_bp = vec![0.0_f32; n_fft];
    if q_high > q_low {
        cep_bp[q_low..q_high].copy_from_slice(&cep[q_low..q_high]);
        // Symmetric negative quefrencies [-q_high + 1 : -q_low + 1]
        let neg_start = n_fft - q_high + 1;
        let neg_end = n_fft - q_low + 1;
        cep_bp[neg_start..neg_end].copy_from_slice(&cep[neg_start..neg_end]);
    }

    let spec = rfft_1d(&cep_bp)?;
    Ok(spec.into_iter().map(|c| c.re).collect())
}

/// Estimates baseline glottal shape parameter `Rd0 in [0.45, 1.35]` via broadband presence balance
/// and Itakura-Saito harmonic template matching over harmonics `k >= 2`.
pub fn estimate_baseline_rd(
    mean_power: &[f32],
    freqs: &[f32],
    f0_src: f64,
    sr: u32,
    n_fft: usize,
) -> f64 {
    let total_p: f64 = mean_power.iter().map(|&x| x as f64).sum();
    if total_p < 1e-10 {
        return 1.0;
    }

    let low_lo = (0.80 * f0_src).max(110.0) as f32;
    let low_hi = (3.2 * f0_src).max(850.0) as f32;
    let mut e_low = 0.0_f64;
    let mut e_pres = 0.0_f64;
    for (&f, &p) in freqs.iter().zip(mean_power.iter()) {
        if f >= low_lo && f <= low_hi {
            e_low += p as f64;
        }
        if (1500.0..=6500.0).contains(&f) {
            e_pres += p as f64;
        }
    }
    e_low += 1e-12;
    e_pres += 1e-12;
    let pres_ratio_db = 10.0 * (e_pres / e_low).log10();

    let k_est = ((8000.0 / f0_src.max(95.0)).floor() as usize).min(56);
    let mut rd_is = 1.0_f64;
    if k_est >= 5 {
        let bin_hz = sr as f64 / n_fft as f64;
        let half_bw_bins = ((0.32 * f0_src / bin_hz).round_ties_even() as isize).max(1);
        let n_bins = mean_power.len() as isize;
        let mut q_obs = Vec::with_capacity(k_est - 1);
        let mut q_sum = 0.0_f64;

        for k in 2..=k_est {
            let fk = (k as f64) * f0_src;
            let x_lip2 = (fk.max(350.0) / 4250.0).powi(2);
            let lip_p = (2.0765167 * x_lip2) / (2.0765167 + x_lip2);
            let c_bin = (fk / bin_hz).round_ties_even() as isize;
            let lo_b = (c_bin - half_bw_bins).max(1) as usize;
            let hi_b = (c_bin + half_bw_bins + 1).min(n_bins) as usize;
            let peak_p = if hi_b > lo_b {
                mean_power[lo_b..hi_b]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64
            } else {
                1e-10
            };
            let q = (peak_p / lip_p.max(1e-4)).max(1e-10);
            q_sum += q;
            q_obs.push(q);
        }

        if q_sum > 1e-10 {
            let q_norm: Vec<f64> = q_obs.iter().map(|&q| (q / q_sum).max(1e-8)).collect();
            let table = get_lf_glottal_table();
            let mut best_idx = 0usize;
            let mut best_div = f64::INFINITY;

            for i in 0..table.rd_grid.len() {
                let mut row_sum = 0.0_f64;
                for k in 1..k_est {
                    row_sum += table.power_templates[[i, k]] as f64;
                }
                let denom = row_sum.max(1e-12);
                let mut div_sum = 0.0_f64;
                for (m, &qn) in q_norm.iter().enumerate() {
                    let t_norm = (table.power_templates[[i, m + 1]] as f64 / denom).max(1e-8);
                    let r = qn / t_norm;
                    div_sum += r - r.ln() - 1.0;
                }
                let is_div = div_sum / (q_norm.len() as f64);
                if is_div < best_div {
                    best_div = is_div;
                    best_idx = i;
                }
            }
            rd_is = table.rd_grid[best_idx] as f64;
        }
    }

    let u_bright = ((pres_ratio_db + 24.0) / 32.0).clamp(0.0, 1.0);
    let rd_bright = 1.32 - 0.87 * u_bright;
    let rd_blend = 0.65 * rd_bright + 0.35 * rd_is.clamp(0.40, 1.60);
    rd_blend.clamp(0.45, 1.35)
}

fn apply_formant_plateau_protection(
    log_gain: &[f32],
    formant_prominence: &[f32],
    freqs: &[f32],
    sr: u32,
    n_fft: usize,
    alpha_prot: f32,
) -> Vec<f32> {
    let n = log_gain.len();
    let mut pos_prom = vec![0.0_f32; n];
    let mut max_prom = 0.0_f32;
    for i in 0..n {
        if freqs[i] >= 220.0 && freqs[i] <= 3600.0 {
            let v = formant_prominence[i].max(0.0);
            pos_prom[i] = v;
            if v > max_prom {
                max_prom = v;
            }
        }
    }
    if max_prom < 1e-5 {
        return log_gain.to_vec();
    }

    let formant_mask: Vec<f32> = pos_prom
        .iter()
        .map(|&v| (v / (max_prom + 1e-6)).clamp(0.0, 1.0))
        .collect();

    let mut dg = Vec::with_capacity(n - 1);
    let mut mask_mid = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        dg.push(log_gain[i + 1] - log_gain[i]);
        mask_mid.push(0.5 * (formant_mask[i + 1] + formant_mask[i]));
    }

    let low_idx = ((200.0 * n_fft as f64 / sr as f64).floor() as usize)
        .max(1)
        .min(n - 2);
    let ref_idx = ((4500.0 * n_fft as f64 / sr as f64).floor() as usize)
        .max(low_idx + 2)
        .min(n - 1);

    let mut dg_damped = dg.clone();
    let mut orig_span = 0.0_f64;
    let mut damped_span = 0.0_f64;
    for i in low_idx..ref_idx {
        orig_span += dg[i] as f64;
        let d = dg[i] * (1.0 - alpha_prot * mask_mid[i]);
        dg_damped[i] = d;
        damped_span += d as f64;
    }

    if damped_span.abs() > 1e-5 {
        let span_scale = (orig_span / damped_span).clamp(0.5, 2.0) as f32;
        for d in &mut dg_damped[low_idx..ref_idx] {
            *d *= span_scale;
        }
    }

    let mut out = vec![0.0_f32; n];
    out[0] = log_gain[0];
    let mut acc = log_gain[0];
    for i in 0..(n - 1) {
        acc += dg_damped[i];
        out[i + 1] = acc;
    }
    out
}

#[inline]
fn interp_1d_linear(xq: f64, xp: &[f32], fp: &[f32]) -> f64 {
    let n = xp.len();
    if xq <= xp[0] as f64 {
        return fp[0] as f64;
    }
    if xq >= xp[n - 1] as f64 {
        return fp[n - 1] as f64;
    }
    let idx = xp.partition_point(|&v| (v as f64) <= xq).saturating_sub(1);
    let x0 = xp[idx] as f64;
    let x1 = xp[idx + 1] as f64;
    let t = (xq - x0) / (x1 - x0);
    (fp[idx] as f64) * (1.0 - t) + (fp[idx + 1] as f64) * t
}

fn interp_lf_log_spectrum(rd_val: f64, f0_src: f64, freqs: &[f32]) -> Vec<f32> {
    let table = get_lf_glottal_table();
    let n_rd = table.rd_grid.len();
    let rd_step = (table.rd_grid[1] - table.rd_grid[0]) as f64;
    let pos = ((rd_val - table.rd_grid[0] as f64) / rd_step).clamp(0.0, (n_rd as f64) - 1.0001);
    let i0 = pos.floor() as usize;
    let frac = (pos - i0 as f64) as f32;

    let k_max = table.log_table.ncols();
    let mut row = Vec::with_capacity(k_max);
    for k in 0..k_max {
        row.push((1.0 - frac) * table.log_table[[i0, k]] + frac * table.log_table[[i0 + 1, k]]);
    }

    let f0_safe = f0_src.max(1.0);
    freqs
        .iter()
        .map(|&f| {
            let nu = (f as f64 / f0_safe).max(1.0);
            if nu <= 1.0 {
                row[0]
            } else if nu >= k_max as f64 {
                row[k_max - 1]
            } else {
                let k0 = (nu.floor() as usize).saturating_sub(1).min(k_max - 2);
                let u = (nu - (k0 + 1) as f64) as f32;
                row[k0] * (1.0 - u) + row[k0 + 1] * u
            }
        })
        .collect()
}

/// Unit-peak Klatt real-pole second-order resonator magnitude response in `(0, 1]`,
/// strictly equal to `1.0` at `freq == center_hz`.
pub fn klatt_resonator_response(
    freqs: &[f32],
    center_hz: f64,
    bandwidth_hz: f64,
    sr: u32,
) -> Vec<f32> {
    let r = (-PI * bandwidth_hz / sr as f64).exp();
    let num = (1.0 - r) * (1.0 - r);
    let r2 = r * r;
    freqs
        .iter()
        .map(|&f| {
            let den = 1.0 - 2.0 * r * (2.0 * PI * (f as f64 - center_hz) / sr as f64).cos() + r2;
            (num / den.max(1e-12)) as f32
        })
        .collect()
}

/// Applies Ht v3 LF glottal tension (`b = -Ht / 50`), Ho openness, and Hr singer's formant
/// filtering with default 44.1 kHz / 2048-FFT parameters.
pub fn pre_emphasis_base_tension(
    wave: &[f32],
    b: f64,
    ho: f64,
    hr: f64,
) -> Result<Vec<f32>, DspError> {
    pre_emphasis_base_tension_with_config(wave, b, ho, hr, &TensionConfig::default())
}

/// Applies Ht v3 LF glottal tension, Ho openness, and Hr singer's formant filtering
/// using explicit STFT configuration.
pub fn pre_emphasis_base_tension_with_config(
    wave: &[f32],
    b: f64,
    ho: f64,
    hr: f64,
    config: &TensionConfig,
) -> Result<Vec<f32>, DspError> {
    config.validate()?;
    if b.abs() < 1e-6 && ho.abs() < 1e-6 && hr.abs() < 1e-6 {
        return Ok(wave.to_vec());
    }
    let original_length = wave.len();
    if original_length == 0 {
        return Ok(Vec::new());
    }

    let sr = config.sample_rate;
    let n_fft = config.n_fft;
    let hop_size = config.hop_size;
    let win_size = config.win_size;

    let mut pad_length = (hop_size - (original_length % hop_size)) % hop_size;
    let min_stft_len = n_fft / 2 + 1;
    if original_length + pad_length < min_stft_len {
        pad_length = min_stft_len - original_length;
    }
    let wave_padded = constant_pad_1d(wave, 0, pad_length, 0.0);

    let window = periodic_hann_window(win_size);
    let mut spec = stft_1d(
        &wave_padded,
        n_fft,
        hop_size,
        &window,
        true,
        PadMode::Reflect,
    )?;

    let (n_freq, n_frames) = spec.dim();
    let df = sr as f64 / n_fft as f64;
    let freqs: Vec<f32> = (0..n_freq).map(|k| (k as f64 * df) as f32).collect();
    let ear_weight = middle_ear_loudness_weight(&freqs);

    let b_clamped = b.clamp(-2.0, 2.0);
    let ho_clamped = ho.clamp(-100.0, 100.0);
    let hr_clamped = hr.clamp(-100.0, 100.0);

    // 1. Voiced-energy-weighted mean power spectrum and adaptive f0_src
    let mut frame_energy = vec![0.0_f64; n_frames];
    let mut total_energy = 0.0_f64;
    for t in 0..n_frames {
        let mut fe = 0.0_f64;
        for f in 0..n_freq {
            let c = spec[[f, t]];
            fe += (c.re * c.re + c.im * c.im) as f64;
        }
        frame_energy[t] = fe;
        total_energy += fe;
    }
    if total_energy <= 1e-12 {
        return Ok(vec![0.0_f32; original_length]);
    }

    let mut mean_power = vec![0.0_f32; n_freq];
    let denom_total = total_energy + 1e-12;
    for f in 0..n_freq {
        let mut acc = 0.0_f64;
        for t in 0..n_frames {
            let c = spec[[f, t]];
            let p = (c.re * c.re + c.im * c.im) as f64;
            acc += p * frame_energy[t];
        }
        mean_power[f] = (acc / denom_total) as f32;
    }

    let f0_src = estimate_harmonic_f0(&mean_power, sr, n_fft)?;

    let mut log_gain = if b_clamped.abs() >= 1e-6 {
        let formant_prominence = extract_cepstral_formant_envelope(&mean_power, sr, n_fft)?;
        let rd0 = estimate_baseline_rd(&mean_power, &freqs, f0_src, sr, n_fft);
        let table = get_lf_glottal_table();

        let (rd_base, psi_target, tilt_scale) = if b_clamped < 0.0 {
            let rd_base = rd0;
            let psi0 = interp_1d_linear(rd_base, &table.rd_grid, &table.psi_grid);
            let u = -b_clamped / 2.0;
            let psi_floor = psi0.min(0.042 + 0.030 * (0.55 - psi0).max(0.0));
            let psi_target =
                psi_floor + (psi0 - psi_floor) * (1.0 - (1.75 * u).tanh() / 1.75_f64.tanh());
            (rd_base, psi_target, 1.28_f32)
        } else {
            let rd_base = rd0.min(0.92);
            let psi0 = interp_1d_linear(rd_base, &table.rd_grid, &table.psi_grid);
            let drive = (b_clamped / 2.0) * 1.65;
            let psi_target = psi0 + (1.0 - psi0) * drive.tanh();
            (rd_base, psi_target, 1.48_f32)
        };

        let rd_target = interp_1d_linear(psi_target, &table.psi_grid, &table.rd_grid);
        let log_g0 = interp_lf_log_spectrum(rd_base, f0_src, &freqs);
        let log_gt = interp_lf_log_spectrum(rd_target, f0_src, &freqs);

        let mut lg: Vec<f32> = log_g0
            .iter()
            .zip(log_gt.iter())
            .enumerate()
            .map(|(k, (&g0, &gt))| {
                let raw = tilt_scale * (gt - g0);
                let hf_ramp = ((freqs[k] - 8200.0) / 3300.0).clamp(0.0, 1.0);
                let hf_counter = (-hf_ramp * raw).clamp(-0.80, 0.80);
                raw + hf_counter
            })
            .collect();

        lg = apply_formant_plateau_protection(&lg, &formant_prominence, &freqs, sr, n_fft, 0.50);
        lg
    } else {
        vec![0.0_f32; n_freq]
    };

    // 5b. Openness (Ho): F0-anchored Klatt real-pole resonator tilt
    if ho_clamped.abs() >= 1e-6 {
        let f_eff: Vec<f32> = freqs.iter().map(|&f| f.max(f0_src as f32)).collect();
        let h_res_o = klatt_resonator_response(&f_eff, f0_src, 300.0, sr);
        let ho_scale = -(ho_clamped / 300.0) as f32;
        for k in 0..n_freq {
            let lg_ho = ho_scale * h_res_o[k].max(1e-7).ln();
            let hf_ramp = ((freqs[k] - 8200.0) / 3300.0).clamp(0.0, 1.0);
            let counter = (-hf_ramp * lg_ho).clamp(-0.80, 0.80);
            log_gain[k] += lg_ho + counter;
        }
    }

    // 5c. Singer's Formant (Hr): 3200 Hz (B = 1000 Hz) additive-magnitude bell/notch
    if hr_clamped.abs() >= 1e-6 {
        let h_res_r = klatt_resonator_response(&freqs, 3200.0, 1000.0, sr);
        let beta_hr = if hr_clamped > 0.0 {
            0.1 * hr_clamped
        } else {
            hr_clamped / 101.0
        };
        for k in 0..n_freq {
            log_gain[k] += (1.0 + beta_hr * h_res_r[k] as f64).max(1e-4).ln() as f32;
        }
    }

    // 6. Minimum-phase complex transfer function via causal real cepstrum folding
    let cepstrum = irfft_real_1d(&log_gain, n_fft)?;
    let half_n = n_fft / 2;
    let mut cep_min = vec![0.0_f32; n_fft];
    cep_min[0] = cepstrum[0];
    for i in 1..half_n {
        cep_min[i] = 2.0 * cepstrum[i];
    }
    cep_min[half_n] = cepstrum[half_n];

    let rfft_cep = rfft_1d(&cep_min)?;
    let gain: Vec<Complex32> = rfft_cep
        .into_iter()
        .map(|z| {
            let mag = z.re.exp();
            Complex32::new(mag * z.im.cos(), mag * z.im.sin())
        })
        .collect();

    for f in 0..n_freq {
        let gf = gain[f];
        for t in 0..n_frames {
            spec[[f, t]] *= gf;
        }
    }

    let mut filtered_wave = istft_1d(
        spec.view(),
        n_fft,
        hop_size,
        &window,
        true,
        Some(wave_padded.len()),
    )?;

    // 7. Psychoacoustic middle-ear loudness preservation W(f)
    let mut l_orig = 0.0_f64;
    let mut l_mod = 0.0_f64;
    for f in 0..n_freq {
        let mp = mean_power[f] as f64;
        let ew = ear_weight[f] as f64;
        let mag_sq = (2.0 * (log_gain[f].clamp(-8.0, 8.0) as f64)).exp();
        l_orig += mp * ew;
        l_mod += mp * mag_sq * ew;
    }

    let orig_rms = (wave_padded
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        / (wave_padded.len() as f64))
        .sqrt();
    let filtered_rms = (filtered_wave
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        / (filtered_wave.len() as f64))
        .sqrt();

    if filtered_rms > 1e-8 && l_mod > 1e-18 && l_orig > 1e-18 {
        let g_loud_w = (l_orig / l_mod).sqrt();
        let g_rms = orig_rms / filtered_rms;
        let (g_loud, expressive_trim) = if b_clamped < 0.0 {
            (
                (0.78 * g_loud_w + 0.22 * g_rms).min(1.18 * g_rms),
                10.0_f64.powf(-0.30 * b_clamped / 20.0),
            )
        } else {
            (
                1.85_f64
                    .min(0.72 * g_loud_w + 0.28 * g_rms)
                    .min(1.22 * g_rms),
                1.0_f64,
            )
        };
        let scale_out = (g_loud * expressive_trim) as f32;
        for s in filtered_wave.iter_mut().take(original_length) {
            *s *= scale_out;
        }
        filtered_wave.truncate(original_length);
        Ok(filtered_wave)
    } else {
        Ok(vec![0.0_f32; original_length])
    }
}

/// Voiced-only high-frequency (> 1.6 kHz) aspiration noise coupling with 5-frame boundary
/// anchor stabilization, keeping unvoiced consonants and silence bit-identical.
pub fn couple_aperiodic_to_tension(
    aperiodic_wave: &[f32],
    orig_voiced_wave: &[f32],
    mod_voiced_wave: &[f32],
    b: f64,
    ho: f64,
    hr: f64,
) -> Result<Vec<f32>, DspError> {
    couple_aperiodic_to_tension_with_config(
        aperiodic_wave,
        orig_voiced_wave,
        mod_voiced_wave,
        b,
        ho,
        hr,
        &TensionConfig::default(),
    )
}

/// Voiced-only aspiration coupling with explicit STFT configuration.
pub fn couple_aperiodic_to_tension_with_config(
    aperiodic_wave: &[f32],
    orig_voiced_wave: &[f32],
    mod_voiced_wave: &[f32],
    b: f64,
    ho: f64,
    hr: f64,
    config: &TensionConfig,
) -> Result<Vec<f32>, DspError> {
    config.validate()?;
    let n_samples = aperiodic_wave.len();
    if orig_voiced_wave.len() != n_samples || mod_voiced_wave.len() != n_samples {
        return Err(DspError::ShapeMismatch {
            expected: format!("all waveforms len == {n_samples}"),
            actual: format!(
                "orig={}, mod={}",
                orig_voiced_wave.len(),
                mod_voiced_wave.len()
            ),
        });
    }
    if n_samples == 0 || (b.abs() < 1e-6 && ho.abs() < 1e-6 && hr.abs() < 1e-6) {
        return Ok(aperiodic_wave.to_vec());
    }

    let sr = config.sample_rate;
    let n_fft = config.n_fft;
    let hop_size = config.hop_size;
    let win_size = config.win_size;

    let min_stft_len = n_fft / 2 + 1;
    let mut pad_len = (hop_size - (n_samples % hop_size)) % hop_size;
    if n_samples + pad_len < min_stft_len {
        pad_len = min_stft_len - n_samples;
    }

    let ap_pad = constant_pad_1d(aperiodic_wave, 0, pad_len, 0.0);
    let vo_pad = constant_pad_1d(orig_voiced_wave, 0, pad_len, 0.0);
    let vm_pad = constant_pad_1d(mod_voiced_wave, 0, pad_len, 0.0);

    let window = periodic_hann_window(win_size);
    let mut spec_ap = stft_1d(&ap_pad, n_fft, hop_size, &window, true, PadMode::Reflect)?;
    let spec_vo = stft_1d(&vo_pad, n_fft, hop_size, &window, true, PadMode::Reflect)?;
    let spec_vm = stft_1d(&vm_pad, n_fft, hop_size, &window, true, PadMode::Reflect)?;

    let (n_freq, n_frames) = spec_ap.dim();
    let df = sr as f64 / n_fft as f64;
    let freqs: Vec<f32> = (0..n_freq).map(|k| (k as f64 * df) as f32).collect();

    let mut p_ap = vec![0.0_f64; n_frames];
    let mut p_vo = vec![0.0_f64; n_frames];
    let mut eh_orig = vec![0.0_f64; n_frames];
    let mut eh_mod = vec![0.0_f64; n_frames];
    let mut max_vo = 0.0_f64;

    for t in 0..n_frames {
        let mut s_ap = 0.0_f64;
        let mut s_vo = 0.0_f64;
        let mut hf_o = 0.0_f64;
        let mut hf_m = 0.0_f64;
        for f in 0..n_freq {
            let ca = spec_ap[[f, t]];
            let co = spec_vo[[f, t]];
            let cm = spec_vm[[f, t]];
            let pa = (ca.re * ca.re + ca.im * ca.im) as f64;
            let po = (co.re * co.re + co.im * co.im) as f64;
            let pm = (cm.re * cm.re + cm.im * cm.im) as f64;
            s_ap += pa;
            s_vo += po;
            if freqs[f] >= 1600.0 && freqs[f] <= 9000.0 {
                hf_o += po;
                hf_m += pm;
            }
        }
        p_ap[t] = s_ap;
        p_vo[t] = s_vo;
        eh_orig[t] = hf_o;
        eh_mod[t] = hf_m;
        if s_vo > max_vo {
            max_vo = s_vo;
        }
    }

    if max_vo < 1e-10 {
        return Ok(aperiodic_wave.to_vec());
    }

    let mut voiced_bool = vec![false; n_frames];
    let mut any_voiced = false;
    for t in 0..n_frames {
        let harm_ratio = p_vo[t] / (p_vo[t] + p_ap[t] + 1e-12);
        let rel_level = p_vo[t] / (max_vo + 1e-12);
        if rel_level > 3e-4 && harm_ratio > 0.55 {
            voiced_bool[t] = true;
            any_voiced = true;
        }
    }
    if !any_voiced {
        return Ok(aperiodic_wave.to_vec());
    }

    let delta_l_frames: Vec<f32> = (0..n_frames)
        .map(|t| (0.5 * ((eh_mod[t] + 1e-12) / (eh_orig[t] + 1e-12)).ln()) as f32)
        .collect();

    // 5-frame onset/offset boundary anchor stabilization on contiguous voiced intervals
    let mut stabilized_delta = vec![0.0_f32; n_frames];
    let mut voiced_weight = vec![0.0_f32; n_frames];
    let mut idx = 0usize;
    while idx < n_frames {
        if !voiced_bool[idx] {
            idx += 1;
            continue;
        }
        let mut end_idx = idx + 1;
        while end_idx < n_frames && voiced_bool[end_idx] {
            end_idx += 1;
        }
        let seg_len = end_idx - idx;
        let margin = if seg_len < 4 {
            0
        } else if seg_len <= 10 {
            seg_len / 3
        } else {
            5
        };
        let n_l = idx + margin;
        let n_r = end_idx - 1 - margin;
        stabilized_delta[idx..end_idx].copy_from_slice(&delta_l_frames[idx..end_idx]);
        if margin > 0 && n_l <= n_r {
            let v_l = delta_l_frames[n_l];
            let v_r = delta_l_frames[n_r];
            for item in &mut stabilized_delta[idx..n_l] {
                *item = v_l;
            }
            for item in &mut stabilized_delta[n_r + 1..end_idx] {
                *item = v_r;
            }
        }
        for item in &mut voiced_weight[idx..end_idx] {
            *item = 1.0;
        }
        idx = end_idx;
    }

    if n_frames >= 3 {
        let mut w_smooth = vec![0.0_f32; n_frames];
        for t in 0..n_frames {
            let left = if t > 0 { voiced_weight[t - 1] } else { 0.0 };
            let mid = voiced_weight[t];
            let right = if t + 1 < n_frames {
                voiced_weight[t + 1]
            } else {
                0.0
            };
            w_smooth[t] = 0.25 * left + 0.50 * mid + 0.25 * right;
        }
        for t in 0..n_frames {
            voiced_weight[t] = if voiced_bool[t] { w_smooth[t] } else { 0.0 };
        }
    }

    let coupled_log: Vec<f32> = (0..n_frames)
        .map(|t| (0.48 * stabilized_delta[t]).clamp(-0.95, 0.62) * voiced_weight[t])
        .collect();

    let freq_profile: Vec<f32> = freqs
        .iter()
        .map(|&f| {
            let r = f / 1600.0;
            let r2 = r * r;
            let s_hf = r2 / (1.0 + r2);
            0.22 + 0.78 * s_hf
        })
        .collect();

    for f in 0..n_freq {
        let fp = freq_profile[f];
        for t in 0..n_frames {
            let g = (fp * coupled_log[t]).exp();
            spec_ap[[f, t]] *= g;
        }
    }

    let ap_filtered = istft_1d(
        spec_ap.view(),
        n_fft,
        hop_size,
        &window,
        true,
        Some(ap_pad.len()),
    )?;

    // Interpolate voiced_weight to sample level (left=0, right=0) so unvoiced consonants are bit-identical
    let mut ap_out = Vec::with_capacity(n_samples);
    let max_frame_sample = (n_frames - 1) * hop_size;
    for s in 0..n_samples {
        let mask = if n_frames == 1 {
            if s == 0 {
                voiced_weight[0]
            } else {
                0.0
            }
        } else if s > max_frame_sample {
            0.0
        } else {
            let i0 = (s / hop_size).min(n_frames - 2);
            let frac = ((s - i0 * hop_size) as f64 / hop_size as f64) as f32;
            voiced_weight[i0] * (1.0 - frac) + voiced_weight[i0 + 1] * frac
        };

        if mask == 0.0 {
            ap_out.push(aperiodic_wave[s]);
        } else {
            ap_out.push(aperiodic_wave[s] + mask * (ap_filtered[s] - aperiodic_wave[s]));
        }
    }

    Ok(ap_out)
}
