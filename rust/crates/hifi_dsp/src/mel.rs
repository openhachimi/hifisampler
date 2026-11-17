//! Pitch- and speed-adjustable Slaney Mel spectrogram and natural-log dynamic range compression.
//!
//! Clean-room port of `util/onnx_dsp.py` (`NumpyMelSpectrogram`, `dynamic_range_compression_np`)
//! and `util/utils.py` (`slaney_mel_filterbank`).

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use ndarray::{s, Array2, Array3, ArrayView2};

use crate::hnsep_stft::{
    cached_forward_rfft_plan, pad_window_to_nfft, periodic_hann_window, reflect_pad_1d,
};
use crate::DspError;

/// Maximum number of cached Hann windows per `MelSpectrogram` instance.
pub const MAX_WINDOW_CACHE_SIZE: usize = 32;

/// Default minimum floor before taking natural logarithm in `dynamic_range_compression`,
/// matching `util/onnx_dsp.py` and `util/audio.py` (`clip_val = 1e-9`).
pub const DEFAULT_DRC_CLIP_VAL: f32 = 1e-9;

/// Configuration for [`MelSpectrogram`], defaulting to PC-NSF-HiFiGAN 44.1 kHz parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub win_size: usize,
    pub hop_length: usize,
    pub f_min: f64,
    pub f_max: f64,
    pub n_mels: usize,
    pub center: bool,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            n_fft: 2048,
            win_size: 2048,
            hop_length: 512,
            f_min: 40.0,
            f_max: 16_000.0,
            n_mels: 128,
            center: false,
        }
    }
}

/// Computes a Slaney-normalized triangular Mel filterbank matrix `[n_mels, n_fft / 2 + 1]`
/// matching `util/utils.py::slaney_mel_filterbank` (librosa Slaney normalization).
pub fn slaney_mel_filterbank(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
) -> Array2<f32> {
    let fmax = fmax.unwrap_or(sr / 2.0);
    let n_freqs = n_fft / 2 + 1;
    let mut weights = Array2::<f32>::zeros((n_mels, n_freqs));
    if n_mels == 0 || n_fft == 0 || sr <= 0.0 {
        return weights;
    }

    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4_f64.ln() / 27.0;

    let hz_to_mel = |f: f64| -> f64 {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_to_hz = |m: f64| -> f64 {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    };

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let n_points = n_mels + 2;
    let step = (max_mel - min_mel) / ((n_points - 1) as f64);

    let mut mel_f = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let m = if i == n_points - 1 {
            max_mel
        } else {
            min_mel + (i as f64) * step
        };
        mel_f.push(mel_to_hz(m));
    }

    let df = 1.0_f64 / ((n_fft as f64) * (1.0_f64 / sr));
    let fftfreqs: Vec<f64> = (0..n_freqs).map(|k| (k as f64) * df).collect();

    for m in 0..n_mels {
        let f_lo = mel_f[m];
        let f_mid = mel_f[m + 1];
        let f_hi = mel_f[m + 2];
        let d_lo = f_mid - f_lo;
        let d_hi = f_hi - f_mid;
        let enorm = 2.0 / (f_hi - f_lo);

        for (k, &fk) in fftfreqs.iter().enumerate() {
            let lower = -(f_lo - fk) / d_lo;
            let upper = (f_hi - fk) / d_hi;
            let tri = lower.min(upper).max(0.0) as f32;
            weights[[m, k]] = ((tri as f64) * enorm) as f32;
        }
    }

    weights
}

/// Applies natural-log dynamic range compression `ln(max(x, clip_val) * c)`.
pub fn dynamic_range_compression(x: ArrayView2<f32>, c: f32, clip_val: f32) -> Array2<f32> {
    x.mapv(|v| (v.max(clip_val) * c).ln())
}

/// In-place natural-log dynamic range compression `ln(max(x, clip_val) * c)`.
pub fn dynamic_range_compression_inplace(x: &mut Array2<f32>, c: f32, clip_val: f32) {
    x.mapv_inplace(|v| (v.max(clip_val) * c).ln());
}

/// Applies natural-log dynamic range compression with default parameters (`c = 1.0`, `clip_val = 1e-9`).
pub fn dynamic_range_compression_default(x: ArrayView2<f32>) -> Array2<f32> {
    dynamic_range_compression(x, 1.0, DEFAULT_DRC_CLIP_VAL)
}

#[derive(Debug)]
struct WindowLru {
    entries: Vec<(usize, Arc<Vec<f32>>)>,
    capacity: usize,
}

impl WindowLru {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn get_or_insert(&mut self, win_size: usize) -> Arc<Vec<f32>> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| *k == win_size) {
            let entry = self.entries.remove(pos);
            let arc = Arc::clone(&entry.1);
            self.entries.push(entry);
            return arc;
        }
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        let win = Arc::new(periodic_hann_window(win_size));
        self.entries.push((win_size, Arc::clone(&win)));
        win
    }
}

/// Thread-safe pitch- and speed-adjustable Slaney Mel spectrogram analyzer.
#[derive(Debug)]
pub struct MelSpectrogram {
    config: MelConfig,
    mel_basis: Array2<f32>,
    /// Precomputed sparse non-zero filterbank intervals `(k_start, contiguous_weights)` per Mel row.
    sparse_mel_rows: Vec<(usize, Vec<f32>)>,
    hann_cache: Mutex<WindowLru>,
}

impl MelSpectrogram {
    /// Creates a new `MelSpectrogram` with precomputed Slaney filterbank and thread-safe Hann cache.
    pub fn new(config: MelConfig) -> Result<Self, DspError> {
        if config.sample_rate == 0 {
            return Err(DspError::InvalidParameter {
                name: "sample_rate",
                reason: "must be > 0".to_string(),
            });
        }
        if config.n_fft == 0 || config.win_size == 0 || config.hop_length == 0 || config.n_mels == 0
        {
            return Err(DspError::InvalidParameter {
                name: "mel_config",
                reason: "n_fft, win_size, hop_length, and n_mels must be > 0".to_string(),
            });
        }
        if config.win_size > config.n_fft {
            return Err(DspError::InvalidParameter {
                name: "win_size",
                reason: format!(
                    "win_size ({}) must be <= n_fft ({})",
                    config.win_size, config.n_fft
                ),
            });
        }
        if config.f_max <= config.f_min || config.f_min < 0.0 {
            return Err(DspError::InvalidParameter {
                name: "f_min/f_max",
                reason: format!(
                    "require 0 <= f_min ({}) < f_max ({})",
                    config.f_min, config.f_max
                ),
            });
        }

        let mel_basis = slaney_mel_filterbank(
            config.sample_rate as f64,
            config.n_fft,
            config.n_mels,
            config.f_min,
            Some(config.f_max),
        );

        let n_freqs = mel_basis.ncols();
        let mut sparse_mel_rows = Vec::with_capacity(config.n_mels);
        for m in 0..config.n_mels {
            let row = mel_basis.row(m);
            let first_nz = (0..n_freqs).find(|&k| row[k] != 0.0);
            let last_nz = (0..n_freqs).rfind(|&k| row[k] != 0.0);
            if let (Some(k0), Some(k1)) = (first_nz, last_nz) {
                sparse_mel_rows.push((k0, row.slice(s![k0..=k1]).to_vec()));
            } else {
                sparse_mel_rows.push((0, Vec::new()));
            }
        }

        Ok(Self {
            config,
            mel_basis,
            sparse_mel_rows,
            hann_cache: Mutex::new(WindowLru::new(MAX_WINDOW_CACHE_SIZE)),
        })
    }

    /// Creates a `MelSpectrogram` with default 44.1 kHz / 128-mel configuration.
    pub fn with_defaults() -> Self {
        Self::new(MelConfig::default()).expect("default MelConfig is valid")
    }

    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    pub fn mel_basis(&self) -> ArrayView2<'_, f32> {
        self.mel_basis.view()
    }

    /// Returns the number of cached Hann windows (bounded by [`MAX_WINDOW_CACHE_SIZE`]).
    pub fn hann_cache_len(&self) -> usize {
        self.hann_cache
            .lock()
            .expect("hann cache lock")
            .entries
            .len()
    }

    fn get_window(&self, win_size_new: usize) -> Arc<Vec<f32>> {
        let mut cache = self.hann_cache.lock().expect("hann cache lock");
        cache.get_or_insert(win_size_new)
    }

    /// Computes a `[n_mels, n_frames]` Mel spectrogram for a mono waveform via fused
    /// per-frame RFFT + sparse filterbank dot-product (no intermediate STFT matrix).
    pub fn compute(
        &self,
        audio: &[f32],
        key_shift: f64,
        speed: f64,
    ) -> Result<Array2<f32>, DspError> {
        if !key_shift.is_finite() || !speed.is_finite() || speed <= 0.0 {
            return Err(DspError::InvalidParameter {
                name: "key_shift/speed",
                reason: format!("invalid key_shift={key_shift}, speed={speed}"),
            });
        }

        let factor = 2.0_f64.powf(key_shift / 12.0);
        let n_fft_new = (self.config.n_fft as f64 * factor).round_ties_even() as isize;
        let win_size_new = (self.config.win_size as f64 * factor).round_ties_even() as isize;
        let hop_length = (self.config.hop_length as f64 * speed).round_ties_even() as isize;

        if n_fft_new <= 0 || win_size_new <= 0 || hop_length <= 0 {
            return Err(DspError::InvalidParameter {
                name: "stft_sizes",
                reason: format!(
                    "n_fft_new={n_fft_new}, win_size_new={win_size_new}, hop_length={hop_length} must be > 0"
                ),
            });
        }

        let n_fft_u = n_fft_new as usize;
        let win_size_u = win_size_new as usize;
        let hop_u = hop_length as usize;
        let window = self.get_window(win_size_u);
        let window_padded: Cow<'_, [f32]> = if window.len() == n_fft_u {
            Cow::Borrowed(&window[..])
        } else {
            Cow::Owned(pad_window_to_nfft(&window, n_fft_u)?)
        };

        let pad_left = (win_size_new - hop_length).div_euclid(2);
        let pad_right = (win_size_new - hop_length + 1).div_euclid(2);
        let mut y_padded = reflect_pad_1d(audio, pad_left, pad_right);
        if self.config.center {
            let pad_c = (n_fft_u / 2) as isize;
            y_padded = reflect_pad_1d(&y_padded, pad_c, pad_c);
        }

        if y_padded.len() < n_fft_u {
            return Err(DspError::SignalTooShort {
                actual: y_padded.len(),
                minimum: n_fft_u,
            });
        }

        let n_frames = (y_padded.len() - n_fft_u) / hop_u + 1;
        let target_bins = self.config.n_fft / 2 + 1;
        let src_bins = n_fft_u / 2 + 1;
        let copy_bins = src_bins.min(target_bins);

        let plan = cached_forward_rfft_plan(n_fft_u)?;
        let (mut time_buf, mut out_spec, mut scratch, mut complex_buf) = plan.new_scratch(n_fft_u);
        let mut mag_buf = vec![0.0_f32; target_bins];

        let scale = if key_shift != 0.0 {
            (self.config.win_size as f64 / win_size_u as f64) as f32
        } else {
            1.0
        };

        let mut mel = Array2::<f32>::zeros((self.config.n_mels, n_frames));
        let mel_slice = mel.as_slice_mut().expect("fresh Array2 is contiguous");

        for t in 0..n_frames {
            let start = t * hop_u;
            let frame_samples = &y_padded[start..start + n_fft_u];
            for ((dst, &s), &w) in time_buf
                .iter_mut()
                .zip(frame_samples.iter())
                .zip(window_padded.iter())
            {
                *dst = s * w;
            }
            plan.process(&mut time_buf, &mut out_spec, &mut scratch, &mut complex_buf)?;

            if key_shift != 0.0 {
                for (dst, c) in mag_buf[..copy_bins].iter_mut().zip(&out_spec[..copy_bins]) {
                    *dst = (c.re * c.re + c.im * c.im).sqrt() * scale;
                }
            } else {
                for (dst, c) in mag_buf[..copy_bins].iter_mut().zip(&out_spec[..copy_bins]) {
                    *dst = (c.re * c.re + c.im * c.im).sqrt();
                }
            }

            for (m, (k_start, weights)) in self.sparse_mel_rows.iter().enumerate() {
                if weights.is_empty() {
                    continue;
                }
                let mag_slice = &mag_buf[*k_start..(*k_start + weights.len())];
                let mut acc = 0.0_f32;
                for (&w, &mag) in weights.iter().zip(mag_slice.iter()) {
                    acc += w * mag;
                }
                mel_slice[m * n_frames + t] = acc;
            }
        }

        Ok(mel)
    }

    /// Computes a `[n_mels, n_frames]` Mel spectrogram using the UTAU `g` (gender) flag
    /// clamped to `[-100, 100]` (`key_shift = -gender * 0.06`).
    pub fn compute_with_gender(
        &self,
        audio: &[f32],
        gender: f64,
        speed: f64,
    ) -> Result<Array2<f32>, DspError> {
        let g = gender.clamp(-100.0, 100.0);
        self.compute(audio, -g * 0.06, speed)
    }

    /// Computes a gender-adjusted `[n_mels, n_frames]` Mel spectrogram and applies natural-log
    /// dynamic range compression in-place (`ln(max(x, 1e-9))`).
    pub fn compute_compressed_with_gender(
        &self,
        audio: &[f32],
        gender: f64,
        speed: f64,
    ) -> Result<Array2<f32>, DspError> {
        let mut mel = self.compute_with_gender(audio, gender, speed)?;
        dynamic_range_compression_inplace(&mut mel, 1.0, DEFAULT_DRC_CLIP_VAL);
        Ok(mel)
    }

    /// Computes a `[batch, n_mels, n_frames]` Mel spectrogram for a 2D batch `[batch, samples]`.
    pub fn compute_batch(
        &self,
        batch: ArrayView2<f32>,
        key_shift: f64,
        speed: f64,
    ) -> Result<Array3<f32>, DspError> {
        let n_batch = batch.nrows();
        if n_batch == 0 {
            return Err(DspError::InvalidParameter {
                name: "batch",
                reason: "batch size must be > 0".to_string(),
            });
        }
        let first_row = batch.row(0);
        let first = if let Some(s) = first_row.as_slice() {
            self.compute(s, key_shift, speed)?
        } else {
            let owned: Vec<f32> = first_row.iter().copied().collect();
            self.compute(&owned, key_shift, speed)?
        };
        let (n_mels, n_frames) = first.dim();
        let mut out = Array3::<f32>::zeros((n_batch, n_mels, n_frames));
        out.slice_mut(s![0, .., ..]).assign(&first);

        for b in 1..n_batch {
            let row = batch.row(b);
            let mel = if let Some(s) = row.as_slice() {
                self.compute(s, key_shift, speed)?
            } else {
                let owned: Vec<f32> = row.iter().copied().collect();
                self.compute(&owned, key_shift, speed)?
            };
            out.slice_mut(s![b, .., ..]).assign(&mel);
        }
        Ok(out)
    }
}
