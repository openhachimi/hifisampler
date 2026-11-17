//! Leap-safe relative-pit LF glottal vocal fry ("气泡音" / Creaky Voice) synthesis.
//!
//! Clean-room port of `util/vocal_fry.py`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{LazyLock, Mutex};

use ndarray::{Array2, ArrayView2};

use crate::akima::Akima1DInterpolator;
use crate::modulation::{gradient_1d_uniform, interp_1d_clamped};
use crate::sos::{cached_butter_sos, sosfiltfilt, ButterBand};

// Strict detection thresholds (cents below current note):
// Normal vibrato (+/-150c), glides (-600c), and 1-octave portamentos (-1200c) stay 100% untouched.
pub const FRY_ONSET_CENTS: f64 = -1250.0;
pub const FRY_FULL_CENTS: f64 = -1680.0;

// Wide morphing transition bounds inside confirmed vocal-fry basins.
pub const BASIN_MORPH_START_CENTS: f64 = -560.0;
pub const BASIN_MORPH_FULL_CENTS: f64 = -1700.0;

// Portamento slew-rate gate bounds (cents / second) for detection pass.
pub const SLEW_PASS_CPS: f64 = 2400.0;
pub const SLEW_STOP_CPS: f64 = 5600.0;

// Minimum contiguous stationary dwell durations (seconds) inside vowel region.
pub const MIN_DWELL_INTERIOR_SEC: f64 = 0.085;
pub const MIN_DWELL_BOUNDARY_SEC: f64 = 0.115;

/// Input parameters for [`prepare_vocal_fry_excitation`].
#[derive(Debug, Clone)]
pub struct VocalFryInput<'a> {
    /// Relative pitchbend curve in cents (UTAU `pit` array).
    pub pitchbend: &'a [f64],
    /// Base MIDI note number (e.g., `64.0` for E4).
    pub base_pitch_midi: f64,
    /// UTAU `t` flag value (tenths of a semitone).
    pub t_flag_val: f64,
    /// Note tempo in BPM.
    pub tempo: f64,
    /// Render window start time in seconds.
    pub new_start: f64,
    /// Render window end time in seconds.
    pub new_end: f64,
    /// Mel frame timestamps in seconds (length `T`).
    pub t_mel: &'a [f64],
    /// Base synthesized F0 curve in Hz (length `T`).
    pub f0_render: &'a [f32],
    /// Mel spectrogram log-magnitude array of shape `[n_mels, T]`.
    pub mel_render: ArrayView2<'a, f32>,
    /// Sample rate in Hz (e.g. `44100`).
    pub sr: u32,
    /// FFT size used for Mel filterbank (e.g. `2048`).
    pub n_fft: usize,
    /// Minimum Mel frequency in Hz (e.g. `40.0`).
    pub mel_fmin: f64,
    /// Maximum Mel frequency in Hz (e.g. `16000.0`).
    pub mel_fmax: f64,
    /// Optional vowel onset timestamp (`new_end - length_req`) in seconds.
    pub new_con: Option<f64>,
}

/// Output of [`prepare_vocal_fry_excitation`] passed into vocoder synthesis and [`apply_lf_vocal_fry`].
#[derive(Debug, Clone)]
pub struct FryExcitationResult {
    /// Tilted Mel spectrogram (`[n_mels, T]`) ready for vocoder inference (`mel_render` in Python).
    pub mel_mod: Array2<f32>,
    /// Modulated F0 curve in Hz (`f0_render` in Python, `f32`).
    pub f0_mod: Vec<f32>,
    /// Smooth per-frame vocal fry weight envelope in `[0.0, 1.0]` (`w_fry` in Python).
    pub fry_env: Vec<f64>,
    /// Effective F0 curve in Hz (`f0_eff` in Python, `f64`).
    pub f0_eff: Vec<f64>,
    /// Mel frame timestamps in seconds.
    pub t_mel: Vec<f64>,
    /// Render window start time in seconds.
    pub new_start: f64,
    /// Render window end time in seconds.
    pub new_end: f64,
    /// Deterministic per-note RNG seed.
    pub seed: u32,
}

#[derive(Clone)]
struct MelFryProfile {
    low_voiced_bins: Vec<bool>,
    high_fric_bins: Vec<bool>,
    lf_fry_mel_tilt: Vec<f32>,
}

type MelProfileKey = (u32, usize, usize, u64, u64);

static MEL_PROFILE_CACHE: LazyLock<Mutex<HashMap<MelProfileKey, MelFryProfile>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static HANNING_KERNEL_CACHE: LazyLock<Mutex<HashMap<usize, Vec<f64>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Converts MIDI note number to frequency in Hz (`A4 = 440 Hz`).
pub use hifi_dsp::midi_to_hz;

/// Computes a Slaney-normalized Mel filterbank matrix of shape `[n_mels, n_fft / 2 + 1]`
/// identical to `util/utils.py::slaney_mel_filterbank`, delegating to [`hifi_dsp::slaney_mel_filterbank`].
pub fn slaney_mel_filterbank(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Array2<f32> {
    if n_mels == 0 || n_fft < 2 || sr == 0 {
        return Array2::zeros((n_mels, n_fft / 2 + 1));
    }
    hifi_dsp::slaney_mel_filterbank(sr as f64, n_fft, n_mels, fmin, Some(fmax))
}

fn get_hanning_kernel(win: usize) -> Vec<f64> {
    {
        let cache = HANNING_KERNEL_CACHE.lock().unwrap();
        if let Some(k) = cache.get(&win) {
            return k.clone();
        }
    }
    let m = win + 2;
    let denom = (m - 1) as f64;
    let mut kern = Vec::with_capacity(win);
    let mut sum = 0.0_f64;
    for k in 1..=win {
        let v = 0.5 - 0.5 * (2.0 * PI * (k as f64) / denom).cos();
        kern.push(v);
        sum += v;
    }
    if sum > 0.0 {
        for v in &mut kern {
            *v /= sum;
        }
    }
    let mut cache = HANNING_KERNEL_CACHE.lock().unwrap();
    if cache.len() >= 64 {
        cache.clear();
    }
    cache.entry(win).or_insert(kern).clone()
}

fn get_mel_fry_profiles(
    sr: u32,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> MelFryProfile {
    let key = (sr, n_fft, n_mels, fmin.to_bits(), fmax.to_bits());
    {
        let cache = MEL_PROFILE_CACHE.lock().unwrap();
        if let Some(p) = cache.get(&key) {
            return p.clone();
        }
    }

    let fb = slaney_mel_filterbank(sr, n_fft, n_mels, fmin, fmax);
    let n_bins = n_fft / 2 + 1;
    let nyq = 0.5 * sr as f64;
    let mut fft_freqs = Vec::with_capacity(n_bins);
    if n_bins == 1 {
        fft_freqs.push(0.0);
    } else {
        for j in 0..n_bins {
            fft_freqs.push(nyq * (j as f64) / ((n_bins - 1) as f64));
        }
    }

    let mut low_voiced_bins = Vec::with_capacity(n_mels);
    let mut high_fric_bins = Vec::with_capacity(n_mels);
    let mut lf_fry_mel_tilt = Vec::with_capacity(n_mels);
    let db_to_neper = 10.0_f64.ln() / 20.0;

    for i in 0..n_mels {
        let mut fb_sum = 0.0_f64;
        let mut weighted_sum = 0.0_f64;
        for j in 0..n_bins {
            let w = fb[[i, j]] as f64;
            fb_sum += w;
            weighted_sum += w * fft_freqs[j];
        }
        let center = weighted_sum / fb_sum.max(1e-12);
        low_voiced_bins.push((80.0..=1050.0).contains(&center));
        high_fric_bins.push((4200.0..=14500.0).contains(&center));

        let f_hz = center.max(35.0);
        let r1 = f_hz / 165.0;
        let low_cut_db = -2.8 / (1.0 + r1 * r1);
        let r2 = f_hz / 650.0;
        let r2_sq = r2 * r2;
        let r3 = f_hz / 5200.0;
        let mid_snap_db = 3.6 * (r2_sq / (1.0 + r2_sq)) / (1.0 + r3 * r3);
        lf_fry_mel_tilt.push(((low_cut_db + mid_snap_db) * db_to_neper) as f32);
    }

    let profile = MelFryProfile {
        low_voiced_bins,
        high_fric_bins,
        lf_fry_mel_tilt,
    };
    let mut cache = MEL_PROFILE_CACHE.lock().unwrap();
    if cache.len() >= 32 {
        cache.clear();
    }
    cache.entry(key).or_insert(profile).clone()
}

#[inline]
fn smoothstep01_scalar(u: f64) -> f64 {
    let uc = u.clamp(0.0, 1.0);
    uc * uc * (3.0 - 2.0 * uc)
}

fn compute_voiced_mel_gate(
    mel_render: ArrayView2<'_, f32>,
    sr: u32,
    n_fft: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f64> {
    let (n_mels, n_frames) = mel_render.dim();
    if n_frames == 0 {
        return Vec::new();
    }

    let profile = get_mel_fry_profiles(sr, n_fft, n_mels, fmin, fmax);
    let any_low = profile.low_voiced_bins.iter().any(|&b| b);
    let any_fric = profile.high_fric_bins.iter().any(|&b| b);

    let mut e_low = vec![0.0_f64; n_frames];
    let mut e_fric = vec![0.0_f64; n_frames];

    for t in 0..n_frames {
        let mut sum_low = 0.0_f64;
        let mut sum_fric = 0.0_f64;
        for m in 0..n_mels {
            let is_low = !any_low || profile.low_voiced_bins[m];
            let is_fric = any_fric && profile.high_fric_bins[m];
            if is_low || is_fric {
                let v = (mel_render[[m, t]] as f64).clamp(-12.0, 6.0);
                let p = (2.0 * v).exp();
                if is_low {
                    sum_low += p;
                }
                if is_fric {
                    sum_fric += p;
                }
            }
        }
        e_low[t] = sum_low;
        e_fric[t] = sum_fric;
    }

    let peak_low = e_low.iter().copied().fold(0.0_f64, f64::max);
    if peak_low <= 1e-9 || !peak_low.is_finite() {
        return vec![0.0_f64; n_frames];
    }

    let mut raw_gate = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let el = e_low[t].max(1e-18);
        let ef = e_fric[t].max(1e-18);
        let rel_db = 10.0 * (el / peak_low).log10();
        let gate_energy = ((rel_db + 28.0) / 10.0).clamp(0.0, 1.0);
        let vh_db = 10.0 * (el / ef).log10();
        let gate_vh = ((vh_db + 4.5) / 8.0).clamp(0.0, 1.0);
        raw_gate.push(gate_energy * gate_vh);
    }

    if n_frames >= 3 {
        let mut smoothed = vec![0.0_f64; n_frames];
        smoothed[0] = 0.75 * raw_gate[0] + 0.25 * raw_gate[1];
        for t in 1..(n_frames - 1) {
            smoothed[t] = 0.25 * raw_gate[t - 1] + 0.5 * raw_gate[t] + 0.25 * raw_gate[t + 1];
        }
        smoothed[n_frames - 1] = 0.25 * raw_gate[n_frames - 2] + 0.75 * raw_gate[n_frames - 1];
        smoothed
    } else {
        raw_gate
    }
}

/// Convolves `data` with symmetric or arbitrary `kernel` using edge-replicated padding (`mode="edge"`).
fn convolve_edge_valid(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = data.len();
    let w = kernel.len();
    if n == 0 || w == 0 {
        return Vec::new();
    }
    let pad = w / 2;
    let out_len = n + 2 * pad - w + 1;
    let max_idx = (n - 1) as isize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let mut acc = 0.0_f64;
        for k in 0..w {
            let idx = (i as isize + k as isize - pad as isize).clamp(0, max_idx) as usize;
            acc += data[idx] * kernel[w - 1 - k];
        }
        out.push(acc);
    }
    out
}

/// Convolves `data` with `kernel` using `np.pad(..., mode="reflect")` padding.
fn convolve_reflect_valid(data: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = data.len();
    let w = kernel.len();
    if n == 0 || w == 0 {
        return Vec::new();
    }
    let pad = w / 2;
    let out_len = n + 2 * pad - w + 1;
    let max_idx = (n - 1) as isize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let mut acc = 0.0_f64;
        for k in 0..w {
            let mut idx = i as isize + k as isize - pad as isize;
            if n == 1 {
                idx = 0;
            } else {
                while idx < 0 || idx > max_idx {
                    if idx < 0 {
                        idx = -idx;
                    }
                    if idx > max_idx {
                        idx = 2 * max_idx - idx;
                    }
                }
            }
            acc += data[idx as usize] * kernel[w - 1 - k];
        }
        out.push(acc);
    }
    out
}

struct BasinInputs<'a> {
    w_cand: &'a [f64],
    ramp_region: &'a [f64],
    voiced_gate: &'a [f64],
    rel_cents: &'a [f64],
    t_mel: &'a [f64],
    new_end: f64,
    t_vowel_start: f64,
    thop: f64,
}

fn confirm_and_build_fry_basins(inputs: &BasinInputs<'_>) -> Option<Vec<f64>> {
    let w_cand = inputs.w_cand;
    let ramp_region = inputs.ramp_region;
    let voiced_gate = inputs.voiced_gate;
    let rel_cents = inputs.rel_cents;
    let t_mel = inputs.t_mel;
    let new_end = inputs.new_end;
    let t_vowel_start = inputs.t_vowel_start;
    let thop = inputs.thop;
    let n_frames = w_cand.len();
    if n_frames == 0 {
        return None;
    }

    // Collect contiguous islands where w_cand > 0.05
    let mut islands: Vec<(usize, usize)> = Vec::new();
    let mut in_island = false;
    let mut start = 0;
    for (i, &w) in w_cand.iter().enumerate() {
        let active = w > 0.05;
        if active && !in_island {
            in_island = true;
            start = i;
        } else if !active && in_island {
            in_island = false;
            islands.push((start, i));
        }
    }
    if in_island {
        islands.push((start, n_frames));
    }
    if islands.is_empty() {
        return None;
    }

    // Precompute contiguous runs of rel_cents < -180.0
    let mut run_starts = Vec::new();
    let mut run_ends = Vec::new();
    let mut in_run = false;
    let mut r_start = 0;
    for (i, &rc) in rel_cents.iter().enumerate() {
        let below = rc < -180.0;
        if below && !in_run {
            in_run = true;
            r_start = i;
        } else if !below && in_run {
            in_run = false;
            run_starts.push(r_start);
            run_ends.push(i);
        }
    }
    if in_run {
        run_starts.push(r_start);
        run_ends.push(n_frames);
    }

    let mut basin_mask = vec![0.0_f64; n_frames];
    let left_entered_low = rel_cents[0] <= -850.0;
    let vowel_start_idx = t_mel.partition_point(|&t| t < t_vowel_start);
    let right_tol = (2.0 * thop).max(0.025);

    let mut confirmed_any = false;
    for &(i0, i1) in &islands {
        let dwell_sec = (i1 - i0) as f64 * thop;
        let touches_right = t_mel[(i1 - 1).min(n_frames - 1)] >= new_end - right_tol;
        let touches_left_vowel = i0 <= vowel_start_idx + 2;

        // 1. Left-boundary melodic upward leap check ('nian C3 -> xiang F4')
        if touches_left_vowel && left_entered_low {
            let post_max = if i1 < n_frames {
                rel_cents[i1..]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                rel_cents[n_frames - 1]
            };
            let island_min = rel_cents[i0..i1]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let dipped_locally = island_min < (rel_cents[0] - 180.0);
            if post_max >= -350.0 && !dipped_locally && dwell_sec < MIN_DWELL_BOUNDARY_SEC {
                continue;
            }
        }

        // 2. Minimum stationary dwell check
        let min_req = if touches_right {
            MIN_DWELL_BOUNDARY_SEC
        } else {
            MIN_DWELL_INTERIOR_SEC
        };
        if dwell_sec < min_req {
            continue;
        }

        confirmed_any = true;
        let min_lo = if left_entered_low { vowel_start_idx } else { 0 };
        let r_pos = run_starts.partition_point(|&rs| rs <= i0);
        let (b_lo, b_hi) = if r_pos > 0 {
            let r = r_pos - 1;
            if run_starts[r] <= i0 && i0 < run_ends[r] {
                let lo = if i0 <= min_lo {
                    i0
                } else {
                    run_starts[r].max(min_lo)
                };
                let hi = i1.max(run_ends[r]);
                (lo, hi)
            } else {
                (i0, i1)
            }
        } else {
            (i0, i1)
        };
        for m in &mut basin_mask[b_lo..b_hi] {
            *m = 1.0;
        }
    }

    if !confirmed_any {
        return None;
    }

    let win_b = ((0.065 / thop.max(1e-4)).round() as usize).max(3) | 1;
    let kern_b = if n_frames >= win_b {
        Some(get_hanning_kernel(win_b))
    } else {
        None
    };

    let mut m_basin = match &kern_b {
        Some(kern) => {
            let mut conv = convolve_edge_valid(&basin_mask, kern);
            conv.truncate(n_frames);
            conv
        }
        None => basin_mask,
    };

    for i in 0..n_frames {
        if left_entered_low {
            m_basin[i] *= ramp_region[i];
        }
        m_basin[i] = m_basin[i].clamp(0.0, 1.0);
    }

    let morph_span = (BASIN_MORPH_START_CENTS - BASIN_MORPH_FULL_CENTS).max(1.0);
    let mut w_raw = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let w_wide = smoothstep01_scalar((BASIN_MORPH_START_CENTS - rel_cents[i]) / morph_span);
        w_raw.push(m_basin[i] * w_wide * ramp_region[i] * voiced_gate[i]);
    }

    let mut w_out = match &kern_b {
        Some(kern) => {
            let mut conv = convolve_edge_valid(&w_raw, kern);
            conv.truncate(n_frames);
            conv
        }
        None => w_raw,
    };

    for i in 0..n_frames {
        w_out[i] = (w_out[i] * ramp_region[i]).clamp(0.0, 1.0);
    }
    Some(w_out)
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn u64_to_unit_open_left(u: u64) -> f64 {
    (((u >> 11) as f64) + 0.5) * (1.0 / 9_007_199_254_740_992.0)
}

/// Generates `n` deterministic independent `N(0, 1)` Gaussian samples via SplitMix64 + Box-Muller.
fn seeded_standard_normal(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed;
    // Warm up one step to decorrelate sequential integer seeds
    let _ = splitmix64(&mut state);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = u64_to_unit_open_left(splitmix64(&mut state));
        let u2 = u64_to_unit_open_left(splitmix64(&mut state));
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        out.push(r * theta.cos());
        if out.len() < n {
            out.push(r * theta.sin());
        }
    }
    out
}

fn bandlimited_fry_jitter(n_frames: usize, thop: f64, seed: u32) -> Vec<f64> {
    if n_frames < 4 {
        return vec![0.0_f64; n_frames];
    }
    let raw = seeded_standard_normal(seed as u64, n_frames);
    let win = ((0.038 / thop.max(1e-4)).round() as usize).max(3) | 1;
    let mut sm = if n_frames >= win {
        let kern = get_hanning_kernel(win);
        let mut conv = convolve_reflect_valid(&raw, &kern);
        conv.truncate(n_frames);
        conv
    } else {
        raw
    };

    let mean = sm.iter().sum::<f64>() / (n_frames as f64);
    let mut var = 0.0_f64;
    for v in &mut sm {
        *v -= mean;
        var += (*v) * (*v);
    }
    let std = (var / (n_frames as f64)).sqrt();
    if std > 1e-8 {
        for v in &mut sm {
            *v = (*v / std).clamp(-2.2, 2.2);
        }
    } else {
        for v in &mut sm {
            *v = v.clamp(-2.2, 2.2);
        }
    }
    sm
}

pub(crate) struct DeterministicFryPrep {
    pub w_fry: Vec<f64>,
    pub rel_cents: Vec<f64>,
    pub f0_base: Vec<f64>,
    pub lf_mel_tilt: Vec<f32>,
    pub seed: u32,
}

pub(crate) fn prepare_vocal_fry_deterministic(
    input: &VocalFryInput<'_>,
) -> Option<DeterministicFryPrep> {
    let pb_in = input.pitchbend;
    if pb_in.len() < 2 {
        return None;
    }
    let (n_mels, n_frames) = input.mel_render.dim();
    if input.t_mel.is_empty()
        || n_mels == 0
        || n_frames == 0
        || input.t_mel.len() != n_frames
        || input.f0_render.len() != n_frames
    {
        return None;
    }

    // O(1)-guarded early-exit: normal notes, vibratos, and <=1-octave portamentos stay untouched.
    let mut min_pb = f64::INFINITY;
    for &v in pb_in {
        if v < min_pb {
            min_pb = v;
        }
    }
    if min_pb >= FRY_ONSET_CENTS || !min_pb.is_finite() {
        return None;
    }

    let thop = if n_frames >= 2 {
        input.t_mel[1] - input.t_mel[0]
    } else {
        512.0 / (input.sr as f64).max(1.0)
    };

    let fix_tail = pb_in.len() >= 2
        && pb_in[pb_in.len() - 1] == 0.0
        && pb_in[pb_in.len() - 2] <= FRY_ONSET_CENTS;
    let pb: Vec<f64> = if fix_tail {
        let mut copy = pb_in.to_vec();
        let last = copy.len() - 1;
        copy[last] = copy[last - 1];
        copy
    } else {
        pb_in.to_vec()
    };

    let safe_tempo = input.tempo.max(1e-3);
    let dt_pitch = 60.0 / (safe_tempo * 96.0);
    let max_pb_idx = (pb.len() - 1) as f64;
    let mut rel_cents = Vec::with_capacity(n_frames);
    for &tm in input.t_mel {
        let u = ((tm - input.new_start) / dt_pitch).clamp(0.0, max_pb_idx);
        let i0 = u as usize;
        let frac = u - (i0 as f64);
        let i1 = (i0 + 1).min(pb.len() - 1);
        rel_cents.push(pb[i0] + frac * (pb[i1] - pb[i0]));
    }

    // Strict detection depth [-1250c .. -1680c]
    let det_span = (FRY_ONSET_CENTS - FRY_FULL_CENTS).max(1.0);
    let mut w_det_depth = Vec::with_capacity(n_frames);
    let mut max_det = 0.0_f64;
    for &rc in &rel_cents {
        let w = smoothstep01_scalar((FRY_ONSET_CENTS - rc) / det_span);
        if w > max_det {
            max_det = w;
        }
        w_det_depth.push(w);
    }
    if max_det <= 1e-4 {
        return None;
    }

    // Pre-vowel / VCV preutterance exclusion [new_start .. t_vowel_start]
    let note_span = (input.new_end - input.new_start).max(thop);
    let raw_con_dur = input
        .new_con
        .map(|nc| (nc - input.new_start).max(0.0))
        .unwrap_or(0.0);
    let min_left_guard = if rel_cents[0] <= -850.0 { 0.030 } else { 0.0 };
    let guard_dur = raw_con_dur.max(min_left_guard).min(0.55 * note_span);
    let t_vowel_start = input.new_start + guard_dur;
    let ramp_denom = (2.0 * thop).max(0.025);
    let ramp_region: Vec<f64> = input
        .t_mel
        .iter()
        .map(|&tm| smoothstep01_scalar((tm - t_vowel_start) / ramp_denom))
        .collect();

    // Slew-rate gate for detection pass (rejects >1-octave melodic leaps)
    let abs_slope_cps: Vec<f64> = if n_frames >= 3 {
        let sm_w = ((0.025 / thop.max(1e-4)).round() as usize).max(3) | 1;
        let rc_smooth = if n_frames >= sm_w {
            let k_sm = vec![1.0 / (sm_w as f64); sm_w];
            let mut conv = convolve_edge_valid(&rel_cents, &k_sm);
            conv.truncate(n_frames);
            conv
        } else {
            rel_cents.clone()
        };
        gradient_1d_uniform(&rc_smooth, thop.max(1e-5))
            .into_iter()
            .map(f64::abs)
            .collect()
    } else {
        vec![0.0_f64; n_frames]
    };

    let slew_span = (SLEW_STOP_CPS - SLEW_PASS_CPS).max(1.0);
    let g_slew: Vec<f64> = abs_slope_cps
        .iter()
        .map(|&s| smoothstep01_scalar((SLEW_STOP_CPS - s) / slew_span))
        .collect();

    let voiced_gate = compute_voiced_mel_gate(
        input.mel_render,
        input.sr,
        input.n_fft,
        input.mel_fmin,
        input.mel_fmax,
    );

    let mut w_cand = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        w_cand.push(w_det_depth[i] * ramp_region[i] * g_slew[i] * voiced_gate[i]);
    }

    let w_fry = confirm_and_build_fry_basins(&BasinInputs {
        w_cand: &w_cand,
        ramp_region: &ramp_region,
        voiced_gate: &voiced_gate,
        rel_cents: &rel_cents,
        t_mel: input.t_mel,
        new_end: input.new_end,
        t_vowel_start,
        thop,
    })?;

    let max_fry = w_fry.iter().copied().fold(0.0_f64, f64::max);
    if max_fry <= 1e-3 {
        return None;
    }

    let f0_base: Vec<f64> = if fix_tail {
        let t_offset = input.t_flag_val / 10.0;
        let pitch_full: Vec<f64> = pb
            .iter()
            .map(|&p| p / 100.0 + input.base_pitch_midi + t_offset)
            .collect();
        let all_equal = pitch_full.iter().all(|&p| p == pitch_full[0]);
        if all_equal {
            vec![midi_to_hz(pitch_full[0]); n_frames]
        } else {
            let mut t_pitch = Vec::with_capacity(pb.len());
            for k in 0..pb.len() {
                t_pitch.push(dt_pitch * (k as f64) + input.new_start);
            }
            if let Ok(akima) = Akima1DInterpolator::new(&t_pitch, &pitch_full) {
                akima
                    .evaluate_clamped(input.t_mel)
                    .into_iter()
                    .map(midi_to_hz)
                    .collect()
            } else {
                input.f0_render.iter().map(|&f| f as f64).collect()
            }
        }
    } else {
        input.f0_render.iter().map(|&f| f as f64).collect()
    };

    let seed = ((input.base_pitch_midi * 100.0).round() as i64)
        .wrapping_mul(1_000_003)
        .wrapping_add((pb_in.len() as i64).wrapping_mul(9176)) as u32;

    let profile = get_mel_fry_profiles(
        input.sr,
        input.n_fft,
        n_mels,
        input.mel_fmin,
        input.mel_fmax,
    );

    Some(DeterministicFryPrep {
        w_fry,
        rel_cents,
        f0_base,
        lf_mel_tilt: profile.lf_fry_mel_tilt,
        seed,
    })
}

pub(crate) fn finalize_vocal_fry_excitation(
    input: &VocalFryInput<'_>,
    prep: DeterministicFryPrep,
    use_jitter: bool,
) -> FryExcitationResult {
    let n_frames = prep.w_fry.len();
    let thop = if n_frames >= 2 {
        input.t_mel[1] - input.t_mel[0]
    } else {
        512.0 / (input.sr as f64).max(1.0)
    };

    let jit_proc = if use_jitter {
        bandlimited_fry_jitter(n_frames, thop, prep.seed)
    } else {
        vec![0.0_f64; n_frames]
    };

    let mut f0_mod = Vec::with_capacity(n_frames);
    let mut f0_eff = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let f_fry_target =
            (34.5 * ((prep.rel_cents[i] + 1750.0) / 520.0).exp2()).clamp(27.0, 135.0);
        let slow_fry_w = prep.w_fry[i] * smoothstep01_scalar((62.0 - f_fry_target) / 26.0);
        let jitter_mod = (0.115 * slow_fry_w * jit_proc[i]).exp();
        let f_fry_inst = prep.f0_base[i].min(f_fry_target) * jitter_mod;

        let log_f0_eff = (1.0 - prep.w_fry[i]) * prep.f0_base[i].max(1.0).ln()
            + prep.w_fry[i] * f_fry_inst.max(1.0).ln();
        let f32_val = log_f0_eff.exp() as f32;
        f0_mod.push(f32_val);
        f0_eff.push(f32_val as f64);
    }

    let (n_mels, _) = input.mel_render.dim();
    let mut mel_mod = input.mel_render.to_owned();
    for m in 0..n_mels {
        let tilt = prep.lf_mel_tilt[m];
        for t in 0..n_frames {
            mel_mod[[m, t]] += tilt * (prep.w_fry[t] as f32);
        }
    }

    FryExcitationResult {
        mel_mod,
        f0_mod,
        fry_env: prep.w_fry,
        f0_eff,
        t_mel: input.t_mel.to_vec(),
        new_start: input.new_start,
        new_end: input.new_end,
        seed: prep.seed,
    }
}

/// O(1)-guarded leap-safe vocal fry detector & continuous single-pass F0/mel excitation shaper.
///
/// Returns `None` whenever vocal fry is not triggered (100% bit-identical to normal synthesis).
pub fn prepare_vocal_fry_excitation(input: &VocalFryInput<'_>) -> Option<FryExcitationResult> {
    let prep = prepare_vocal_fry_deterministic(input)?;
    Some(finalize_vocal_fry_excitation(input, prep, true))
}

#[inline]
pub(crate) fn pulse_rms_integral(tau_att: f64, tau_dec: f64) -> f64 {
    let r1 = 2.0 / tau_dec;
    let r2 = (1.0 / tau_att) + (2.0 / tau_dec);
    let r3 = (2.0 / tau_att) + (2.0 / tau_dec);
    let val = (1.0 / r1) - (2.0 / r2) + (1.0 / r3);
    val.max(1e-8)
}

fn interp_control_curves(
    t_mel: &[f64],
    new_start: f64,
    new_end: f64,
    n_samples: usize,
    w_fry: &[f64],
    f0_eff: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_mel = t_mel.len();
    if n_mel == 0 || w_fry.len() != n_mel || f0_eff.len() != n_mel || n_samples == 0 {
        return None;
    }

    if n_mel == 1 {
        let w0 = w_fry[0];
        if w0 <= 1e-4 {
            return None;
        }
        let w_s = vec![w0; n_samples];
        let f0_s = vec![f0_eff[0].clamp(20.0, 1200.0); n_samples];
        return Some((w_s, f0_s));
    }

    let t0 = t_mel[0];
    let dt_mel = (t_mel[n_mel - 1] - t0) / ((n_mel - 1) as f64);
    let mut is_uniform = dt_mel > 0.0;
    if is_uniform {
        let tol = 1e-9 * dt_mel.max(1.0);
        for i in 0..(n_mel - 1) {
            if ((t_mel[i + 1] - t_mel[i]) - dt_mel).abs() > tol {
                is_uniform = false;
                break;
            }
        }
    }

    if is_uniform {
        let u_start = (new_start - t0) / dt_mel;
        let u_end = (new_end - t0) / dt_mel;
        let u_step = (u_end - u_start) / (n_samples as f64);
        let max_u = (n_mel - 1) as f64;

        let mut w_s = Vec::with_capacity(n_samples);
        let mut max_w = 0.0_f64;
        for k in 0..n_samples {
            let u = (u_start + (k as f64) * u_step).clamp(0.0, max_u);
            let i0 = u as usize;
            let frac = u - (i0 as f64);
            let i1 = (i0 + 1).min(n_mel - 1);
            let w = w_fry[i0] + frac * (w_fry[i1] - w_fry[i0]);
            if w > max_w {
                max_w = w;
            }
            w_s.push(w);
        }
        if max_w <= 1e-4 {
            return None;
        }

        let mut f0_s = Vec::with_capacity(n_samples);
        for k in 0..n_samples {
            let u = (u_start + (k as f64) * u_step).clamp(0.0, max_u);
            let i0 = u as usize;
            let frac = u - (i0 as f64);
            let i1 = (i0 + 1).min(n_mel - 1);
            let f0 = f0_eff[i0] + frac * (f0_eff[i1] - f0_eff[i0]);
            f0_s.push(f0.clamp(20.0, 1200.0));
        }
        return Some((w_s, f0_s));
    }

    // General non-uniform t_mel fallback
    let t_step = (new_end - new_start) / (n_samples as f64);
    let mut w_s = Vec::with_capacity(n_samples);
    let mut max_w = 0.0_f64;
    for k in 0..n_samples {
        let t_a = new_start + (k as f64) * t_step;
        let w = interp_1d_clamped(t_mel, w_fry, t_a);
        if w > max_w {
            max_w = w;
        }
        w_s.push(w);
    }
    if max_w <= 1e-4 {
        return None;
    }

    let mut f0_s = Vec::with_capacity(n_samples);
    for k in 0..n_samples {
        let t_a = new_start + (k as f64) * t_step;
        let f0 = interp_1d_clamped(t_mel, f0_eff, t_a);
        f0_s.push(f0.clamp(20.0, 1200.0));
    }
    Some((w_s, f0_s))
}

fn unwrap_phase(angles: &[f64]) -> Vec<f64> {
    if angles.is_empty() {
        return Vec::new();
    }
    let two_pi = 2.0 * PI;
    let mut out = Vec::with_capacity(angles.len());
    out.push(angles[0]);
    let mut offset = 0.0_f64;
    for i in 1..angles.len() {
        let dd = angles[i] - angles[i - 1];
        let mut ddmod = (dd + PI).rem_euclid(two_pi) - PI;
        if ddmod == -PI && dd > 0.0 {
            ddmod = PI;
        }
        let ph_correct = if dd.abs() < PI { 0.0 } else { ddmod - dd };
        offset += ph_correct;
        out.push(angles[i] + offset);
    }
    out
}

/// O(N) monotonic two-pointer cursor interpolating block phase `phi_blk` at sample indices
/// `0..n_samples` and applying the slew-limited epoch offset (`|dphi| <= 0.18 * dphase`).
fn align_epoch_phase_monotonic(
    blk_t: &[f64],
    phi_blk: &[f64],
    dphase: &[f64],
    phase: &[f64],
) -> Vec<f64> {
    let n_blk = blk_t.len();
    let n_samples = phase.len();
    let mut aligned = Vec::with_capacity(n_samples);
    if n_blk == 0 || n_samples == 0 {
        return aligned;
    }

    let mut prev_phi_raw = phi_blk[0];
    let mut running_phi_peak = prev_phi_raw;
    let mut seg = 0_usize;
    let last_seg = n_blk.saturating_sub(2);
    for i in 0..n_samples {
        let x = i as f64;
        let phi_raw = if n_blk <= 1 || x <= blk_t[0] {
            phi_blk[0]
        } else if x >= blk_t[n_blk - 1] {
            phi_blk[n_blk - 1]
        } else {
            while seg < last_seg && x >= blk_t[seg + 1] {
                seg += 1;
            }
            let span = blk_t[seg + 1] - blk_t[seg];
            if span <= 0.0 {
                phi_blk[seg]
            } else {
                let frac = (x - blk_t[seg]) / span;
                phi_blk[seg] + frac * (phi_blk[seg + 1] - phi_blk[seg])
            }
        };
        let dphi_raw = phi_raw - prev_phi_raw;
        prev_phi_raw = phi_raw;
        let max_d = 0.18 * dphase[i];
        let dphi = dphi_raw.clamp(-max_d, max_d);
        running_phi_peak += dphi;
        aligned.push(phase[i] - running_phi_peak + 0.050);
    }
    aligned
}

pub(crate) fn apply_lf_vocal_fry_impl(
    render: &[f32],
    fry_ctx: Option<&FryExcitationResult>,
    sr: u32,
    use_shimmer: bool,
) -> Vec<f32> {
    let Some(ctx) = fry_ctx else {
        return render.to_vec();
    };
    if render.len() < 32 || sr == 0 {
        return render.to_vec();
    }

    // O(1) early exit when fry_env is all-zero
    let max_env = ctx.fry_env.iter().copied().fold(0.0_f64, f64::max);
    if max_env <= 1e-4 {
        return render.to_vec();
    }

    let n_samples = render.len();
    let Some((w_s, f0_s)) = interp_control_curves(
        &ctx.t_mel,
        ctx.new_start,
        ctx.new_end,
        n_samples,
        &ctx.fry_env,
        &ctx.f0_eff,
    ) else {
        return render.to_vec();
    };

    // Exact 3-band zero-phase complementary crossover: sub + mid + high == x64
    let x64: Vec<f64> = render.iter().map(|&v| v as f64).collect();
    let (sub_band, mid_band, high_band) = if n_samples > 64 {
        let nyq = 0.5 * (sr as f64);
        let sos_mid = cached_butter_sos(2, (155.0 / nyq).clamp(0.003, 0.95), ButterBand::Highpass);
        let sos_hi = cached_butter_sos(2, (900.0 / nyq).clamp(0.003, 0.95), ButterBand::Highpass);
        let above_sub = sosfiltfilt(&sos_mid, &x64).unwrap_or_else(|| x64.clone());
        let high = sosfiltfilt(&sos_hi, &above_sub).unwrap_or_else(|| above_sub.clone());
        let mut sub = Vec::with_capacity(n_samples);
        let mut mid = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            sub.push(x64[i] - above_sub[i]);
            mid.push(above_sub[i] - high[i]);
        }
        (sub, mid, high)
    } else {
        let sub: Vec<f64> = x64.iter().map(|&v| 0.2 * v).collect();
        let mid: Vec<f64> = x64.iter().map(|&v| 0.4 * v).collect();
        let high = mid.clone();
        (sub, mid, high)
    };

    // Slew-limited sliding complex Fourier epoch tracker
    let mut period_s = Vec::with_capacity(n_samples);
    let mut dphase = Vec::with_capacity(n_samples);
    let mut phase = Vec::with_capacity(n_samples);
    let mut ph_acc = 0.0_f64;
    let sr_f = sr as f64;
    for &f0 in &f0_s {
        period_s.push(1.0 / f0);
        let dp = f0 / sr_f;
        dphase.push(dp);
        ph_acc += dp;
        phase.push(ph_acc);
    }

    let step = 1024_usize;
    let n_blk = n_samples.div_ceil(step).max(1);
    let inv_step = 1.0 / (step as f64);
    let two_pi_neg = -2.0 * PI;

    let mut c_blk_re = vec![0.0_f64; n_blk];
    let mut c_blk_im = vec![0.0_f64; n_blk];
    let mut z_glob_re = 0.0_f64;
    let mut z_glob_im = 0.0_f64;

    for b in 0..n_blk {
        let s0 = b * step;
        let s1 = (s0 + step).min(n_samples);
        let mut sum_re = 0.0_f64;
        let mut sum_im = 0.0_f64;
        for i in s0..s1 {
            let e =
                (0.35 * mid_band[i] * mid_band[i] + 0.65 * high_band[i] * high_band[i]) * w_s[i];
            let theta = two_pi_neg * (phase[i] % 1.0);
            sum_re += e * theta.cos();
            sum_im += e * theta.sin();
        }
        let br = sum_re * inv_step;
        let bi = sum_im * inv_step;
        c_blk_re[b] = br;
        c_blk_im[b] = bi;
        z_glob_re += sum_re;
        z_glob_im += sum_im;
    }

    let z_glob_abs = z_glob_re.hypot(z_glob_im);
    let aligned = if z_glob_abs > 1e-16 {
        let z_fb_re = z_glob_re / z_glob_abs;
        let z_fb_im = z_glob_im / z_glob_abs;
        let win_l = ((n_blk / 2) * 2 + 1).clamp(3, 11);
        let kern = get_hanning_kernel(win_l);
        let c_re = convolve_edge_valid(&c_blk_re, &kern);
        let c_im = convolve_edge_valid(&c_blk_im, &kern);

        let mut max_c_abs = 0.0_f64;
        for b in 0..n_blk {
            let a = c_blk_re[b].hypot(c_blk_im[b]);
            if a > max_c_abs {
                max_c_abs = a;
            }
        }
        let bias = 0.08 * (max_c_abs + 1e-12);

        let mut raw_angles = Vec::with_capacity(n_blk);
        let mut blk_t = Vec::with_capacity(n_blk);
        for b in 0..n_blk {
            let sm_re = c_re[b] + bias * z_fb_re;
            let sm_im = c_im[b] + bias * z_fb_im;
            raw_angles.push(-sm_im.atan2(sm_re));
            blk_t.push((b as f64 + 0.5) * (step as f64));
        }

        let phi_blk: Vec<f64> = unwrap_phase(&raw_angles)
            .into_iter()
            .map(|a| a / (2.0 * PI))
            .collect();

        align_epoch_phase_monotonic(&blk_t, &phi_blk, &dphase, &phase)
    } else {
        phase.iter().map(|&ph| ph + 0.050).collect()
    };

    // Stochastic + dichrotic per-bubble amplitude shimmer
    let min_al = aligned.iter().copied().fold(f64::INFINITY, f64::min);
    let max_al = aligned.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let c_min = (min_al.floor() as i64) - 2;
    let c_max = (max_al.floor() as i64) + 2;
    let n_cyc = ((c_max - c_min + 1).max(4)) as usize;

    let shim_table = if use_shimmer {
        seeded_standard_normal((ctx.seed ^ 0x005A_3C91) as u64, n_cyc)
    } else {
        vec![0.0_f64; n_cyc]
    };

    let mut amp_table = Vec::with_capacity(n_cyc);
    for (k, &shim) in shim_table.iter().enumerate().take(n_cyc) {
        let cyc_id = c_min + (k as i64);
        let parity = (cyc_id & 1) as f64;
        let amp = (1.0 - 0.34 * parity + 0.11 * shim).clamp(0.45, 1.18);
        amp_table.push(amp);
    }

    let tau_a_mid = 0.00075_f64;
    let tau_d_mid = 0.0044_f64;
    let tau_a_hi = 0.00040_f64;
    let tau_d_hi = 0.00235_f64;
    let inv_tau_a_mid = 1.0 / tau_a_mid;
    let inv_tau_d_mid = 1.0 / tau_d_mid;
    let inv_tau_a_hi = 1.0 / tau_a_hi;
    let inv_tau_d_hi = 1.0 / tau_d_hi;
    let amp_rms = 0.86_f64;
    let int_mid = pulse_rms_integral(tau_a_mid, tau_d_mid);
    let int_hi = pulse_rms_integral(tau_a_hi, tau_d_hi);

    let orig_peak = x64.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let ceil_peak = (0.96 * orig_peak).max(0.12);
    let inv_ceil = 1.0 / ceil_peak;
    let max_amp_idx = (n_cyc - 1) as i64;

    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let al = aligned[i];
        let per = period_s[i];
        let cyc_f = al.floor();
        let cyc_i = cyc_f as i64;
        let tau = (al - cyc_f) * per;
        let tau_prev = tau + per;

        let idx_cur = (cyc_i - c_min).clamp(0, max_amp_idx) as usize;
        let idx_prev = (cyc_i - 1 - c_min).clamp(0, max_amp_idx) as usize;
        let amp_cur = amp_table[idx_cur];
        let amp_prev = amp_table[idx_prev];

        let env_mid = amp_cur * (1.0 - (-tau * inv_tau_a_mid).exp()) * (-tau * inv_tau_d_mid).exp()
            + amp_prev
                * (1.0 - (-tau_prev * inv_tau_a_mid).exp())
                * (-tau_prev * inv_tau_d_mid).exp();

        let env_hi = amp_cur * (1.0 - (-tau * inv_tau_a_hi).exp()) * (-tau * inv_tau_d_hi).exp()
            + amp_prev
                * (1.0 - (-tau_prev * inv_tau_a_hi).exp())
                * (-tau_prev * inv_tau_d_hi).exp();

        let rms_mid = ((int_mid / per).sqrt() * amp_rms).max(1e-4);
        let rms_hi = ((int_hi / per).sqrt() * amp_rms).max(1e-4);

        let g_sub = (0.85 / rms_mid).min(1.9) * env_mid * 0.42;
        let g_mid = (1.28 / rms_mid).min(3.10) * env_mid;
        let g_hi = (1.52 / rms_hi).min(3.65) * env_hi;

        let fry_val = sub_band[i] * g_sub + mid_band[i] * g_mid + high_band[i] * g_hi;
        let limited = ceil_peak * (fry_val * inv_ceil).tanh();
        let blended = x64[i] + w_s[i] * (limited - x64[i]);
        out.push(blended as f32);
    }

    out
}

/// Post-vocoder sliding-phase-locked 3-band LF glottal closure shaper.
///
/// Performs O(1) early return when `fry_ctx` is `None`, `render.len() < 32`,
/// or `fry_ctx.fry_env` is all-zero (`max <= 1e-4`).
pub fn apply_lf_vocal_fry(
    render: &[f32],
    fry_ctx: Option<&FryExcitationResult>,
    sr: u32,
) -> Vec<f32> {
    apply_lf_vocal_fry_impl(render, fry_ctx, sr, true)
}

/// In-place variant of [`apply_lf_vocal_fry`] that performs zero allocations on the O(1) early-exit path.
pub fn apply_lf_vocal_fry_in_place(
    render: &mut [f32],
    fry_ctx: Option<&FryExcitationResult>,
    sr: u32,
) {
    let Some(ctx) = fry_ctx else {
        return;
    };
    if render.len() < 32 || sr == 0 {
        return;
    }
    let max_env = ctx.fry_env.iter().copied().fold(0.0_f64, f64::max);
    if max_env <= 1e-4 {
        return;
    }
    let shaped = apply_lf_vocal_fry_impl(render, Some(ctx), sr, true);
    render.copy_from_slice(&shaped);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_deterministic_apply_lf_vocal_fry_matches_python() {
        let sr = 44100_u32;
        let n_samples = 2048_usize;
        let mut carrier = Vec::with_capacity(n_samples);
        for n in 0..n_samples {
            let t = (n as f64) / (sr as f64);
            let s = 0.25 * (2.0 * PI * 620.0 * t).cos() + 0.18 * (2.0 * PI * 1750.0 * t).cos();
            carrier.push(s as f32);
        }

        let ctx_det = FryExcitationResult {
            mel_mod: Array2::zeros((128, 4)),
            f0_mod: vec![220.0, 31.0, 35.0, 180.0],
            fry_env: vec![0.0, 0.9, 0.85, 0.1],
            f0_eff: vec![220.0, 31.0, 35.0, 180.0],
            t_mel: vec![0.0, 0.005, 0.02, 0.046],
            new_start: 0.0,
            new_end: 2048.0 / (sr as f64),
            seed: 99,
        };

        let out = apply_lf_vocal_fry_impl(&carrier, Some(&ctx_det), sr, false);
        assert_eq!(out.len(), n_samples);

        let expected_500 = [
            0.042139135_f32,
            0.045705944,
            0.04805446,
            0.062254902,
            0.094455786,
            0.11926165,
            0.13386275,
            0.13679814,
        ];
        for (i, &exp) in expected_500.iter().enumerate() {
            assert_abs_diff_eq!(out[500 + i], exp, epsilon = 1e-5);
        }

        let expected_1200 = [
            0.03077781_f32,
            0.051310126,
            0.07353166,
            0.09643237,
            0.11891787,
            0.13987441,
            0.15823565,
            0.17304665,
        ];
        for (i, &exp) in expected_1200.iter().enumerate() {
            assert_abs_diff_eq!(out[1200 + i], exp, epsilon = 1e-5);
        }

        let rms =
            (out.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n_samples as f64).sqrt();
        assert_abs_diff_eq!(rms, 0.15936643876530274, epsilon = 1e-5);
    }

    #[test]
    fn test_deterministic_prepare_no_jitter_is_finite_and_bounded() {
        let sr = 44100_u32;
        let thop = 512.0 / (sr as f64);
        let n_frames = 80_usize;
        let t_mel: Vec<f64> = (0..n_frames).map(|i| (i as f64) * thop).collect();
        let f0_flat = vec![329.63_f32; n_frames];
        let mut mel = Array2::<f32>::from_elem((128, n_frames), -3.0);
        for m in 0..35 {
            for t in 0..n_frames {
                mel[[m, t]] = 0.8;
            }
        }
        let mut pb = vec![0.0_f64; 20];
        for k in 0..16 {
            let u = (k as f64) / 15.0;
            let s = -1900.0 * (3.0 * u * u - 2.0 * u * u * u);
            pb.push((s as i32) as f64);
        }
        pb.extend(std::iter::repeat_n(-1900.0_f64, 90));
        pb.push(0.0);

        let input = VocalFryInput {
            pitchbend: &pb,
            base_pitch_midi: 64.0,
            t_flag_val: 0.0,
            tempo: 120.0,
            new_start: 0.0,
            new_end: (n_frames as f64) * thop,
            t_mel: &t_mel,
            f0_render: &f0_flat,
            mel_render: mel.view(),
            sr,
            n_fft: 2048,
            mel_fmin: 40.0,
            mel_fmax: 16000.0,
            new_con: Some(0.03),
        };
        let prep = prepare_vocal_fry_deterministic(&input).expect("should detect basin");
        let res = finalize_vocal_fry_excitation(&input, prep, false);
        assert_eq!(res.seed, 2106217256);
        let min_f0 = res.f0_mod.iter().copied().fold(f32::INFINITY, f32::min);
        assert!((27.0..=36.0).contains(&min_f0), "min_f0={min_f0}");
    }

    #[test]
    fn test_align_epoch_phase_monotonic_matches_binary_search() {
        for &n_samples in &[500_usize, 1536, 8193] {
            let step = 1024_usize;
            let n_blk = n_samples.div_ceil(step).max(1);
            let blk_t: Vec<f64> = (0..n_blk)
                .map(|b| (b as f64 + 0.5) * (step as f64))
                .collect();
            let phi_blk: Vec<f64> = (0..n_blk)
                .map(|b| 0.15 * ((b as f64) * 0.7).sin())
                .collect();
            let dphase: Vec<f64> = (0..n_samples)
                .map(|i| (32.0 + 10.0 * ((i as f64) * 0.001).cos()) / 44100.0)
                .collect();
            let mut phase = Vec::with_capacity(n_samples);
            let mut acc = 0.0_f64;
            for &dp in &dphase {
                acc += dp;
                phase.push(acc);
            }

            let fast = align_epoch_phase_monotonic(&blk_t, &phi_blk, &dphase, &phase);

            let mut prev_phi_raw = interp_1d_clamped(&blk_t, &phi_blk, 0.0);
            let mut running_phi_peak = prev_phi_raw;
            let mut ref_aligned = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let phi_raw = interp_1d_clamped(&blk_t, &phi_blk, i as f64);
                let dphi_raw = phi_raw - prev_phi_raw;
                prev_phi_raw = phi_raw;
                let max_d = 0.18 * dphase[i];
                running_phi_peak += dphi_raw.clamp(-max_d, max_d);
                ref_aligned.push(phase[i] - running_phi_peak + 0.050);
            }

            assert_eq!(fast.len(), ref_aligned.len());
            for (a, b) in fast.iter().zip(ref_aligned.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 0.0);
            }
        }
    }
}
