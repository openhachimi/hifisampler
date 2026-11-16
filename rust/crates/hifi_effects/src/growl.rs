//! Growl (`HG` flag) DSP effect — clean-room port of `util/growl.py`.

use crate::sos::{cached_butter_sos, sosfilt, ButterBand};

const MAX_VIBRATO_CENTS: f64 = 100.0;
const HP_CUTOFF_HZ: f64 = 20.0;
const MIN_NYQ_FRAC: f64 = 0.01;

/// Default growl square LFO modulation frequency in Hz (`80.0`).
pub const DEFAULT_GROWL_FREQ_HZ: f64 = 80.0;

/// Default growl highpass crossover frequency in Hz (`400.0`).
pub const DEFAULT_GROWL_CUTOFF_HZ: f64 = 400.0;

#[inline]
fn round_6(val: f64) -> f64 {
    (val * 1_000_000.0).round() / 1_000_000.0
}

fn highpass_split(audio: &[f32], sr: u32, cutoff: f64) -> (Vec<f32>, Vec<f32>) {
    let nyq = sr as f64 / 2.0;
    let norm = round_6((cutoff / nyq).clamp(MIN_NYQ_FRAC, 0.99));
    let sos = cached_butter_sos(4, norm, ButterBand::Highpass);

    let audio_f64: Vec<f64> = audio.iter().map(|&v| v as f64).collect();
    let filtered = sosfilt(&sos, &audio_f64, None);

    let mut high = Vec::with_capacity(audio.len());
    let mut complement = Vec::with_capacity(audio.len());
    for (&x, &h) in audio.iter().zip(filtered.iter()) {
        let hf = h as f32;
        high.push(hf);
        complement.push(x - hf);
    }
    (high, complement)
}

#[inline]
fn rms_f64(x: &[f32]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0_f64;
    for &v in x {
        let vf = v as f64;
        acc += vf * vf;
    }
    (acc / (n as f64)).sqrt()
}

fn square_lfo(num_samples: usize, sr: u32, freq: f64) -> Vec<f32> {
    let step = (2.0 * freq / sr as f64) as f32;
    let mut lfo = Vec::with_capacity(num_samples);
    for n in 0..num_samples {
        let half_cycles = ((n as f32) * step) as i32;
        let bit = half_cycles & 1;
        lfo.push(1.0_f32 - 2.0_f32 * (bit as f32));
    }
    lfo
}

fn apply_pitch_modulation(band: &[f32], sr: u32, lfo: &[f32], strength: f64) -> Vec<f32> {
    let n_samples = band.len();
    if n_samples == 0 {
        return Vec::new();
    }

    let exp_arg = strength * MAX_VIBRATO_CENTS / 1200.0;
    let r_pos = exp_arg.exp2();
    let r_neg = (-exp_arg).exp2();
    let half_diff = 0.5 * (r_pos - r_neg);

    let lfo0 = lfo[0] as f64;
    let sum_lfo: f64 = lfo.iter().map(|&v| v as f64).sum();
    let mean_lfo = sum_lfo / (n_samples as f64);

    let mut drift = Vec::with_capacity(n_samples);
    let mut cumsum = 0.0_f64;
    for (n, &lv) in lfo.iter().enumerate() {
        cumsum += lv as f64;
        let mut d = cumsum - lfo0;
        if mean_lfo != 0.0 {
            d -= (n as f64) * mean_lfo;
        }
        drift.push(d * half_diff);
    }

    if n_samples > 100 {
        let hp_norm = round_6((HP_CUTOFF_HZ / (sr as f64 / 2.0)).clamp(1e-6, 0.99));
        let sos2 = cached_butter_sos(2, hp_norm, ButterBand::Highpass);
        drift = sosfilt(&sos2, &drift, None);
    }

    let max_idx = (n_samples - 1) as f64;
    let mut modulated = Vec::with_capacity(n_samples);
    for (n, &d) in drift.iter().enumerate() {
        let idx = (n as f64 + d).clamp(0.0, max_idx);
        let i0 = idx as usize;
        let frac = (idx - i0 as f64) as f32;
        let i1 = (i0 + 1).min(n_samples - 1);
        let b0 = band[i0];
        let b1 = band[i1];
        modulated.push(b0 + (b1 - b0) * frac);
    }

    let rms_orig = rms_f64(band);
    let rms_new = rms_f64(&modulated);
    if rms_new > 1e-10 {
        let scale = (rms_orig / rms_new) as f32;
        for m in &mut modulated {
            *m *= scale;
        }
    }
    modulated
}

/// Applies growl pitch-vibrato modulation with full parameter control,
/// matching `util/growl.py::growl`.
///
/// - `frequency`: square LFO rate in Hz (default `80.0`)
/// - `strength`: normalized vibrato strength (`HG / 100.0`, e.g., `0.5` for `HG50`)
/// - `freq_low`: 4th-order Butterworth highpass split frequency in Hz (default `400.0`)
pub fn growl_with_params(
    audio: &[f32],
    sample_rate: u32,
    frequency: f64,
    strength: f64,
    freq_low: f64,
) -> Vec<f32> {
    if audio.is_empty()
        || sample_rate == 0
        || strength == 0.0
        || !strength.is_finite()
        || frequency <= 0.0
        || !frequency.is_finite()
    {
        return audio.to_vec();
    }

    let (band, mut complement) = highpass_split(audio, sample_rate, freq_low);
    let lfo = square_lfo(audio.len(), sample_rate, frequency);
    let modulated = apply_pitch_modulation(&band, sample_rate, &lfo, strength);

    for (c, &m) in complement.iter_mut().zip(modulated.iter()) {
        *c += m;
    }
    complement
}

/// Applies growl effect with default `80.0` Hz LFO and `400.0` Hz crossover split.
///
/// `strength` matches `util/growl.py`'s normalized parameter (`0.0..=1.0`, e.g. `hg as f64 / 100.0`).
pub fn growl(audio: &[f32], sample_rate: u32, strength: f64) -> Vec<f32> {
    growl_with_params(
        audio,
        sample_rate,
        DEFAULT_GROWL_FREQ_HZ,
        strength,
        DEFAULT_GROWL_CUTOFF_HZ,
    )
}

/// Applies UTAU `HG`-flag growl (`hg` clamped to `0..=100`, mapped to `strength = hg / 100.0`).
///
/// Matches `backend/resampler.py` lines 608-613. Returns `audio.to_vec()` untouched if `hg <= 0`.
pub fn apply_hg_growl(audio: &[f32], sample_rate: u32, hg: i32) -> Vec<f32> {
    if hg <= 0 || audio.is_empty() {
        return audio.to_vec();
    }
    let hg_clamped = hg.clamp(0, 100) as f64;
    growl(audio, sample_rate, hg_clamped / 100.0)
}
