//! ITU-R BS.1770-4 integrated LUFS measurement and envelope-aware loudness normalization.

use crate::biquad::{accumulate_k_weighted_power, k_weighting_sos};
use crate::LufsError;

/// ITU-R BS.1770-4 channel weighting factors `[L, R, C, Ls, Rs]`.
const BS1770_CHANNEL_WEIGHTS: [f64; 5] = [1.0, 1.0, 1.0, 1.41, 1.41];

/// Absolute gating threshold in LUFS (`-70.0` LUFS).
pub const ABS_GATE_LUFS: f64 = -70.0;

/// Relative gating offset below the absolute-gated mean (`-10.0` LU).
pub const REL_GATE_LU: f64 = -10.0;

/// Default BS.1770-4 sliding gating block duration in seconds (400 ms).
pub const DEFAULT_BLOCK_SIZE_SEC: f64 = 0.400;

/// Default silence trimming threshold in dBFS (`-52.0` dB).
pub const DEFAULT_SILENCE_THRESHOLD_DB: f64 = -52.0;

/// Linear weighted power corresponding to `-70.0` LUFS.
#[inline]
fn abs_gate_power() -> f64 {
    10.0_f64.powf((ABS_GATE_LUFS + 0.691) / 10.0)
}

/// Relative gating linear power multiplier (`10^(-10/10) = 0.1`).
#[inline]
fn rel_gate_ratio() -> f64 {
    10.0_f64.powf(REL_GATE_LU / 10.0)
}

#[inline]
fn channel_weight(channel_idx: usize) -> f64 {
    BS1770_CHANNEL_WEIGHTS
        .get(channel_idx)
        .copied()
        .unwrap_or(1.0)
}

/// Tail protection strategy used by [`apply_envelope_aware_gain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TailMode {
    /// Retains a 45% floor ratio of the target gain change in quiet tails (`8 dB..24 dB` below sustained level).
    #[default]
    PreserveRelative,
    /// Legacy/clip tail attenuation (`10 dB..20 dB` below sustained level, 0% floor).
    Clip,
}

impl TailMode {
    /// Parses a tail mode string (`"preserve_relative"`, `"clip"`, `"legacy_protect"`).
    pub fn parse(s: &str) -> Result<Self, LufsError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "preserve_relative" => Ok(Self::PreserveRelative),
            "clip" | "legacy_protect" => Ok(Self::Clip),
            _ => Err(LufsError::UnsupportedTailMode(s.to_string())),
        }
    }
}

/// Options controlling [`apply_envelope_aware_gain`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvelopeGainOptions {
    /// Threshold (dBFS) below which low-level breath tails are gain-protected (default `-40.0`).
    pub low_level_protect_db: f64,
    /// Hard peak limit (dBFS) applied via smooth gain clamping inside protected low-level tails (default `-6.0`).
    pub tail_peak_limit_dbfs: f64,
    /// Tail transition curve mode (default [`TailMode::PreserveRelative`]).
    pub tail_mode: TailMode,
    /// Optional cap on positive boost in dB (default `Some(18.0)`).
    pub max_boost_db: Option<f64>,
}

impl Default for EnvelopeGainOptions {
    fn default() -> Self {
        Self {
            low_level_protect_db: -40.0,
            tail_peak_limit_dbfs: -6.0,
            tail_mode: TailMode::PreserveRelative,
            max_boost_db: Some(18.0),
        }
    }
}

/// Options controlling [`loudness_norm_fast`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessNormOptions {
    /// Target peak ceiling in dBFS (retained for signature compatibility with `util.audio.loudness_norm`).
    pub peak_db: f64,
    /// Target integrated loudness in LUFS (default `-23.0`; resampler uses `-16.0`).
    pub target_loudness: f64,
    /// Gating block duration in seconds (default `0.400`).
    pub block_size: f64,
    /// Normalization strength in `[0.0, 100.0]` (default `100.0`).
    pub strength: f64,
    /// Whether to trim leading/trailing silence before measuring and normalizing (default `true`).
    pub trim_silence: bool,
    /// Frame RMS threshold in dB used to locate voiced span (default `-52.0`).
    pub silence_threshold: f64,
    /// Optional cap on positive boost in dB (default `Some(18.0)`).
    pub max_boost_db: Option<f64>,
    /// Low-level tail protection threshold in dBFS (default `-40.0`).
    pub low_level_protect_db: f64,
    /// Peak limit in dBFS for protected low-level tails (default `-6.0`).
    pub tail_peak_limit_dbfs: f64,
    /// Tail protection mode (default [`TailMode::PreserveRelative`]).
    pub tail_mode: TailMode,
}

impl Default for LoudnessNormOptions {
    fn default() -> Self {
        Self {
            peak_db: -1.0,
            target_loudness: -23.0,
            block_size: DEFAULT_BLOCK_SIZE_SEC,
            strength: 100.0,
            trim_silence: true,
            silence_threshold: DEFAULT_SILENCE_THRESHOLD_DB,
            max_boost_db: Some(18.0),
            low_level_protect_db: -40.0,
            tail_peak_limit_dbfs: -6.0,
            tail_mode: TailMode::PreserveRelative,
        }
    }
}

impl LoudnessNormOptions {
    /// Returns options matching the default resampler configuration (`target_loudness = -16.0`).
    pub fn resampler_default() -> Self {
        Self {
            target_loudness: -16.0,
            ..Self::default()
        }
    }
}

/// Measures integrated gated loudness in LUFS for mono `f32` audio using the standard 400 ms block size.
///
/// Returns `f64::NEG_INFINITY` for empty, silent, or fully gated signals.
pub fn measure_lufs(audio: &[f32], sample_rate: u32) -> Result<f64, LufsError> {
    measure_lufs_multichannel(audio, sample_rate, 1, DEFAULT_BLOCK_SIZE_SEC)
}

/// Measures integrated gated loudness in LUFS for mono `f32` audio with a configurable block duration.
pub fn measure_lufs_with_block_size(
    audio: &[f32],
    sample_rate: u32,
    block_size: f64,
) -> Result<f64, LufsError> {
    measure_lufs_multichannel(audio, sample_rate, 1, block_size)
}

/// Measures integrated gated loudness in LUFS for interleaved multichannel `f32` audio
/// according to ITU-R BS.1770-4.
pub fn measure_lufs_multichannel(
    interleaved: &[f32],
    sample_rate: u32,
    channels: usize,
    block_size: f64,
) -> Result<f64, LufsError> {
    if sample_rate == 0 {
        return Err(LufsError::InvalidSampleRate(sample_rate));
    }
    if channels == 0 {
        return Err(LufsError::InvalidChannelCount(channels));
    }
    if !block_size.is_finite() || block_size <= 0.0 {
        return Err(LufsError::InvalidBlockSize(block_size));
    }
    if !interleaved.len().is_multiple_of(channels) {
        return Err(LufsError::ShapeMismatch {
            len: interleaved.len(),
            channels,
        });
    }
    if interleaved.is_empty() {
        return Ok(f64::NEG_INFINITY);
    }

    let sos = k_weighting_sos(sample_rate)?;
    let num_frames = interleaved.len() / channels;
    let mut sq_weighted = vec![0.0_f64; num_frames];

    for ch in 0..channels {
        accumulate_k_weighted_power(
            interleaved,
            channels,
            ch,
            channel_weight(ch),
            &sos,
            &mut sq_weighted,
        );
    }

    let block_samples = ((sample_rate as f64 * block_size).round_ties_even() as usize).max(1);
    let hop_samples = (block_samples / 4).max(1); // 75% overlap (100 ms hop for 400 ms block)

    if num_frames < block_samples {
        let mean_sq = sq_weighted.iter().sum::<f64>() / (num_frames as f64);
        if mean_sq < 1e-20 {
            return Ok(f64::NEG_INFINITY);
        }
        return Ok(-0.691 + 10.0 * mean_sq.log10());
    }

    let mut cs = Vec::with_capacity(num_frames + 1);
    cs.push(0.0_f64);
    let mut running = 0.0_f64;
    for &p in &sq_weighted {
        running += p;
        cs.push(running);
    }

    let abs_gate = abs_gate_power();
    let inv_block = 1.0 / (block_samples as f64);
    let num_blocks = (num_frames - block_samples) / hop_samples + 1;
    let mut block_powers = Vec::with_capacity(num_blocks);

    let mut abs_sum = 0.0_f64;
    let mut abs_count = 0usize;
    let mut start = 0usize;
    while start + block_samples <= num_frames {
        let p = (cs[start + block_samples] - cs[start]) * inv_block;
        block_powers.push(p);
        if p >= abs_gate {
            abs_sum += p;
            abs_count += 1;
        }
        start += hop_samples;
    }

    if abs_count == 0 {
        return Ok(f64::NEG_INFINITY);
    }

    let mean_abs = abs_sum / (abs_count as f64);
    let rel_gate = mean_abs * rel_gate_ratio();

    let mut rel_sum = 0.0_f64;
    let mut rel_count = 0usize;
    for &p in &block_powers {
        if p >= abs_gate && p > rel_gate {
            rel_sum += p;
            rel_count += 1;
        }
    }

    if rel_count == 0 {
        return Ok(f64::NEG_INFINITY);
    }

    let mean_rel = rel_sum / (rel_count as f64);
    if mean_rel <= 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(-0.691 + 10.0 * mean_rel.log10())
}

/// Applies gain with envelope-aware tail protection to prevent boosting low-level breath tails.
///
/// Uses 20 ms frame RMS (10 ms hop), 90th percentile of non-silent frames (`>-60 dB`) for
/// sustained-level estimation, 50 ms centered moving-average smoothing, and smooth gain limiting
/// in protected low-level regions.
pub fn apply_envelope_aware_gain(
    audio: &[f32],
    target_gain: f64,
    sample_rate: u32,
    opts: EnvelopeGainOptions,
) -> Result<Vec<f32>, LufsError> {
    if sample_rate == 0 {
        return Err(LufsError::InvalidSampleRate(sample_rate));
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let mut gain = target_gain.max(0.0);
    if let Some(max_b) = opts.max_boost_db {
        gain = gain.min(10.0_f64.powf(max_b / 20.0));
    }

    if (gain - 1.0).abs() < 0.01 {
        return Ok(audio.to_vec());
    }

    let (fade_start_db, fade_end_db, tail_gain_floor_ratio) = match opts.tail_mode {
        TailMode::PreserveRelative => (8.0_f64, 24.0_f64, 0.45_f64),
        TailMode::Clip => (10.0_f64, 20.0_f64, 0.0_f64),
    };

    let n = audio.len();
    let frame_len = (sample_rate as f64 * 0.02) as usize;
    let hop_len = (sample_rate as f64 * 0.01) as usize;
    if frame_len == 0 || hop_len == 0 || n < frame_len {
        let g = gain as f32;
        return Ok(audio.iter().map(|&s| s * g).collect());
    }

    // Frame-level RMS in dB via prefix sums.
    let mut cs = Vec::with_capacity(n + 1);
    cs.push(0.0_f64);
    let mut running = 0.0_f64;
    for &s in audio {
        let v = s as f64;
        running += v * v;
        cs.push(running);
    }

    let num_frames = (n - frame_len) / hop_len + 1;
    if num_frames == 0 {
        let g = gain as f32;
        return Ok(audio.iter().map(|&s| s * g).collect());
    }

    let inv_frame_len = 1.0 / (frame_len as f64);
    let mut frame_rms = Vec::with_capacity(num_frames);
    let mut non_silent = Vec::new();

    for f in 0..num_frames {
        let start = f * hop_len;
        let mean_sq = ((cs[start + frame_len] - cs[start]) * inv_frame_len).max(0.0);
        let rms = mean_sq.sqrt();
        let db = if rms < 1e-10 {
            -100.0
        } else {
            20.0 * rms.max(1e-10).log10()
        };
        frame_rms.push(db);
        if db > -60.0 {
            non_silent.push(db);
        }
    }

    if non_silent.is_empty() {
        let g = gain as f32;
        return Ok(audio.iter().map(|&s| s * g).collect());
    }

    non_silent.sort_by(f64::total_cmp);
    let p90_idx = ((non_silent.len() as f64 * 0.9) as usize).min(non_silent.len() - 1);
    let sustained_db = non_silent[p90_idx];

    let threshold_high = sustained_db - fade_start_db;
    let threshold_low = sustained_db - fade_end_db;
    let db_range = (threshold_high - threshold_low).max(1e-12);
    let protected_gain = 1.0 + (gain - 1.0) * tail_gain_floor_ratio;

    let mut frame_gains = Vec::with_capacity(num_frames);
    for &db in &frame_rms {
        let relative = ((db - threshold_low) / db_range).clamp(0.0, 1.0);
        let blend = tail_gain_floor_ratio + (1.0 - tail_gain_floor_ratio) * relative;
        let mut fg = 1.0 + (gain - 1.0) * blend;
        if gain > 1.0 && db <= opts.low_level_protect_db {
            fg = fg.min(protected_gain);
        }
        frame_gains.push(fg);
    }

    // 50 ms centered moving-average smoothing over frame gains.
    let smooth_frames =
        (((sample_rate as f64 * 0.05) / (hop_len as f64)).round_ties_even() as usize).max(1);
    let smoothed_gains = if smooth_frames > 1 && frame_gains.len() > smooth_frames {
        let half = smooth_frames / 2;
        let mut fg_cs = Vec::with_capacity(num_frames + 1);
        fg_cs.push(0.0_f64);
        let mut fg_run = 0.0_f64;
        for &g in &frame_gains {
            fg_run += g;
            fg_cs.push(fg_run);
        }
        let mut sm = Vec::with_capacity(num_frames);
        for i in 0..num_frames {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(num_frames);
            sm.push((fg_cs[hi] - fg_cs[lo]) / ((hi - lo) as f64));
        }
        sm
    } else {
        frame_gains
    };

    let tail_peak_limit = if gain > 1.0 {
        10.0_f64.powf(opts.tail_peak_limit_dbfs / 20.0)
    } else {
        0.0
    };

    // Linearly interpolate frame gains to sample level and apply smooth tail peak limit.
    let last_frame = smoothed_gains.len() - 1;
    let inv_hop = 1.0 / (hop_len as f64);
    let mut output = Vec::with_capacity(n);

    for (i, &sample) in audio.iter().enumerate() {
        let sample_pos = (i as f64) * inv_hop;
        let frame_idx = (sample_pos as usize).min(last_frame);
        let next_idx = (frame_idx + 1).min(last_frame);
        let frac = if frame_idx < last_frame {
            sample_pos - (frame_idx as f64)
        } else {
            0.0
        };

        let mut sample_gain =
            smoothed_gains[frame_idx] * (1.0 - frac) + smoothed_gains[next_idx] * frac;
        let orig_f64 = sample as f64;

        if gain > 1.0 && tail_peak_limit > 0.0 {
            let sample_db = frame_rms[frame_idx] * (1.0 - frac) + frame_rms[next_idx] * frac;
            if sample_db <= opts.low_level_protect_db {
                let abs_orig = orig_f64.abs();
                if abs_orig * sample_gain > tail_peak_limit {
                    sample_gain = tail_peak_limit / abs_orig.max(1e-12);
                }
            }
        }

        output.push((orig_f64 * sample_gain) as f32);
    }

    Ok(output)
}

/// Returns `(start_sample, end_sample)` of voiced audio with 100 ms tail padding,
/// or `None` if no frames exceed `silence_threshold`.
fn find_voiced_span(audio: &[f32], rate: u32, silence_threshold: f64) -> Option<(usize, usize)> {
    let frame_length = (rate as f64 * 0.02) as usize;
    let hop_length = (rate as f64 * 0.01) as usize;
    if frame_length == 0 || hop_length == 0 || audio.len() < frame_length {
        return Some((0, audio.len()));
    }

    let num_frames = (audio.len() - frame_length) / hop_length + 1;
    if num_frames == 0 {
        return Some((0, audio.len()));
    }

    let mut cs = Vec::with_capacity(audio.len() + 1);
    cs.push(0.0_f64);
    let mut running = 0.0_f64;
    for &s in audio {
        let v = s as f64;
        running += v * v;
        cs.push(running);
    }

    let thresh_sq = 10.0_f64.powf(silence_threshold / 10.0).max(1e-20);
    let inv_frame_len = 1.0 / (frame_length as f64);

    let mut first_voiced: Option<usize> = None;
    let mut last_voiced: Option<usize> = None;

    for f in 0..num_frames {
        let start = f * hop_length;
        let mean_sq = (cs[start + frame_length] - cs[start]) * inv_frame_len;
        if mean_sq > thresh_sq {
            if first_voiced.is_none() {
                first_voiced = Some(f);
            }
            last_voiced = Some(f);
        }
    }

    let (first, last) = match (first_voiced, last_voiced) {
        (Some(a), Some(b)) => (a, b),
        _ => return None,
    };

    let padding_frames = ((rate as f64 * 0.1) as usize) / hop_length;
    let start_sample = first * hop_length;
    let end_sample = audio
        .len()
        .min((last + 1 + padding_frames) * hop_length + frame_length);
    Some((start_sample, end_sample))
}

/// Pads `work_audio` on the right up to `min_samples` using `numpy.pad(..., mode="reflect")`
/// semantics (`reflect` when `len > 1`, `edge` when `len == 1`).
fn pad_short_for_measurement(work_audio: &[f32], min_samples: usize) -> Vec<f32> {
    let len = work_audio.len();
    if len >= min_samples || len == 0 {
        return work_audio.to_vec();
    }
    let mut out = Vec::with_capacity(min_samples);
    out.extend_from_slice(work_audio);
    if len == 1 {
        out.resize(min_samples, work_audio[0]);
        return out;
    }

    let period = 2 * (len - 1);
    for idx in len..min_samples {
        let r = idx % period;
        let src_idx = if r < len { r } else { period - r };
        out.push(work_audio[src_idx]);
    }
    out
}

/// Performs fast ITU-R BS.1770-4 loudness normalization with optional silence trimming,
/// 5 ms crossfade lead-in / up to 200 ms tail crossfade, and envelope-aware tail protection.
///
/// Mirrors `util/lufs.py::loudness_norm_fast`.
pub fn loudness_norm_fast(
    audio: &[f32],
    rate: u32,
    opts: LoudnessNormOptions,
) -> Result<Vec<f32>, LufsError> {
    if rate == 0 {
        return Err(LufsError::InvalidSampleRate(rate));
    }
    if !opts.block_size.is_finite() || opts.block_size <= 0.0 {
        return Err(LufsError::InvalidBlockSize(opts.block_size));
    }

    let original_length = audio.len();
    if original_length == 0 {
        return Ok(Vec::new());
    }

    let original: Vec<f32> = audio
        .iter()
        .map(|&s| if s.is_finite() { s } else { 0.0 })
        .collect();

    let peak_abs = original.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
    if peak_abs < 1e-8 || opts.strength < 0.01 {
        return Ok(original);
    }

    let mut span: Option<(usize, usize)> = None;
    let work_audio: &[f32] = if opts.trim_silence {
        match find_voiced_span(&original, rate, opts.silence_threshold) {
            Some((start, end)) => {
                span = Some((start, end));
                let slice = &original[start..end];
                let slice_peak = slice.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
                if slice.is_empty() || slice_peak < 1e-8 {
                    return Ok(original);
                }
                slice
            }
            None => return Ok(original),
        }
    } else {
        &original
    };

    let min_samples = ((rate as f64 * opts.block_size) as usize).max(1);
    let current_lufs = if work_audio.len() < min_samples {
        let padded = pad_short_for_measurement(work_audio, min_samples);
        measure_lufs_multichannel(&padded, rate, 1, opts.block_size)?
    } else {
        measure_lufs_multichannel(work_audio, rate, 1, opts.block_size)?
    };

    if !current_lufs.is_finite() {
        return Ok(original);
    }

    let final_loudness =
        current_lufs + (opts.target_loudness - current_lufs) * opts.strength / 100.0;
    let mut gain_db = final_loudness - current_lufs;
    if let Some(max_b) = opts.max_boost_db {
        gain_db = gain_db.min(max_b);
    }
    let target_gain = 10.0_f64.powf(gain_db / 20.0);

    let normalized = apply_envelope_aware_gain(
        work_audio,
        target_gain,
        rate,
        EnvelopeGainOptions {
            low_level_protect_db: opts.low_level_protect_db,
            tail_peak_limit_dbfs: opts.tail_peak_limit_dbfs,
            tail_mode: opts.tail_mode,
            max_boost_db: opts.max_boost_db,
        },
    )?;

    if opts.trim_silence {
        if let Some((start_sample, _)) = span {
            let mut output = original;
            let available_length = normalized.len().min(original_length - start_sample);
            let fade_length = ((rate as f64 * 0.2) as usize).min(available_length / 4);

            if available_length > 0 {
                let lead_xfade = ((rate as f64 * 0.005) as usize)
                    .min(start_sample)
                    .min(available_length / 4);

                let do_tail_fade =
                    fade_length > 0 && (start_sample + available_length) < original_length;

                for i in 0..available_length {
                    let mut w = 1.0_f32;
                    if lead_xfade > 0 && i < lead_xfade {
                        w = if lead_xfade == 1 {
                            0.0
                        } else {
                            i as f32 / ((lead_xfade - 1) as f32)
                        };
                    }
                    if do_tail_fade && i >= available_length - fade_length {
                        let tail_pos = i - (available_length - fade_length);
                        let tail_w = if fade_length == 1 {
                            1.0
                        } else {
                            1.0 - (tail_pos as f32 / ((fade_length - 1) as f32))
                        };
                        w = tail_w;
                    }
                    let orig_v = output[start_sample + i];
                    output[start_sample + i] = normalized[i] * w + orig_v * (1.0 - w);
                }
            }
            return Ok(output);
        }
    }

    Ok(normalized.into_iter().take(original_length).collect())
}

/// Convenience wrapper matching `util/audio.py::loudness_norm`.
pub fn loudness_norm(
    audio: &[f32],
    rate: u32,
    target_loudness: f64,
    strength: f64,
    trim_silence: bool,
    silence_threshold: f64,
) -> Result<Vec<f32>, LufsError> {
    loudness_norm_fast(
        audio,
        rate,
        LoudnessNormOptions {
            target_loudness,
            strength,
            trim_silence,
            silence_threshold,
            ..LoudnessNormOptions::default()
        },
    )
}

/// Normalizes audio so its peak absolute amplitude reaches `peak_dbfs` (e.g. `-1.0` dBFS).
/// Silent signals (`peak < 1e-8`) are returned unchanged.
pub fn peak_normalize(audio: &[f32], peak_dbfs: f32) -> Vec<f32> {
    let peak = audio
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    if peak < 1e-8 || !peak_dbfs.is_finite() {
        return audio
            .iter()
            .map(|&v| if v.is_finite() { v } else { 0.0 })
            .collect();
    }
    let target_amp = 10.0_f32.powf(peak_dbfs / 20.0);
    let scale = target_amp / peak;
    audio
        .iter()
        .map(|&v| if v.is_finite() { v * scale } else { 0.0 })
        .collect()
}

/// Uniformly scales `audio` in-place if its peak absolute amplitude exceeds `peak_limit`.
pub fn apply_peak_limit(audio: &mut [f32], peak_limit: f32) {
    if !peak_limit.is_finite() || peak_limit <= 0.0 {
        return;
    }
    let peak = audio
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    if peak > peak_limit && peak > 1e-8 {
        let gain = peak_limit / peak;
        for sample in audio.iter_mut() {
            *sample = if sample.is_finite() {
                *sample * gain
            } else {
                0.0
            };
        }
    }
}
