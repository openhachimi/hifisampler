//! Centered STFT/ISTFT and HN-SEP complex spectrogram framing & reconstruction.
//!
//! Clean-room port of `_stft_np`, `_istft_ola_np`, `_reflect_pad_1d_batch`, and
//! `NumpyHnsepModel` framing/masking from `util/onnx_dsp.py`.

use std::f64::consts::PI;
use std::sync::Arc;

use ndarray::{Array2, Array4, ArrayView2, ArrayView4};
use num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::{Fft, FftPlanner};

use crate::DspError;

/// Padding mode used when `center = true` in STFT.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    /// NumPy/PyTorch `"reflect"` padding without repeating edge samples.
    Reflect,
    /// Constant-value padding (e.g. `0.0`).
    Constant(f32),
}

/// Computes a periodic Hann window of length `win_len` matching
/// `scipy.signal.windows.hann(win_len, sym=False)` and `torch.hann_window(win_len, periodic=True)`.
pub fn periodic_hann_window(win_len: usize) -> Vec<f32> {
    if win_len == 0 {
        return Vec::new();
    }
    if win_len == 1 {
        return vec![1.0];
    }
    let denom = win_len as f64;
    (0..win_len)
        .map(|n| (0.5 * (1.0 - (2.0 * PI * n as f64 / denom).cos())) as f32)
        .collect()
}

/// Maps an out-of-bounds signed index onto `[0, len)` via non-repeating reflection,
/// matching `numpy.pad(..., mode="reflect")` for arbitrary pad widths.
#[inline]
pub fn reflect_index(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let k = idx.rem_euclid(period) as usize;
    if k < len {
        k
    } else {
        (period as usize) - k
    }
}

/// Applies 1D reflect padding (or cropping when `pad_left < 0` / `pad_right < 0`),
/// matching `_reflect_pad_1d_batch` in `util/onnx_dsp.py`.
///
/// If the cropped signal length is `<= max(pad_left, pad_right)`, it is first
/// zero-padded on the right to `max_pad + 1` before applying reflect padding.
pub fn reflect_pad_1d(y: &[f32], mut pad_left: isize, mut pad_right: isize) -> Vec<f32> {
    let mut slice = y;
    if pad_left < 0 {
        let drop_left = (-pad_left) as usize;
        slice = if drop_left >= slice.len() {
            &[]
        } else {
            &slice[drop_left..]
        };
        pad_left = 0;
    }
    if pad_right < 0 {
        let drop_right = (-pad_right) as usize;
        let keep = slice.len().saturating_sub(drop_right);
        slice = &slice[..keep];
        pad_right = 0;
    }

    if pad_left == 0 && pad_right == 0 {
        return slice.to_vec();
    }

    let pl = pad_left as usize;
    let pr = pad_right as usize;
    let max_pad = pl.max(pr);

    let owned_pre;
    let src: &[f32] = if slice.len() <= max_pad {
        let target_len = max_pad + 1;
        let mut tmp = vec![0.0_f32; target_len];
        tmp[..slice.len()].copy_from_slice(slice);
        owned_pre = tmp;
        &owned_pre
    } else {
        slice
    };

    let src_len = src.len();
    let total_len = pl + src_len + pr;
    let mut out = Vec::with_capacity(total_len);
    for i in 0..pl {
        let idx = reflect_index(i as isize - pl as isize, src_len);
        out.push(src[idx]);
    }
    out.extend_from_slice(src);
    for j in 0..pr {
        let idx = reflect_index((src_len + j) as isize, src_len);
        out.push(src[idx]);
    }
    out
}

/// Applies 1D constant-value padding on both ends of `y`.
pub fn constant_pad_1d(y: &[f32], pad_left: usize, pad_right: usize, value: f32) -> Vec<f32> {
    if pad_left == 0 && pad_right == 0 {
        return y.to_vec();
    }
    let mut out = Vec::with_capacity(pad_left + y.len() + pad_right);
    out.resize(pad_left, value);
    out.extend_from_slice(y);
    out.resize(out.len() + pad_right, value);
    out
}

/// Internal FFT dispatch supporting both even sizes (`realfft`) and odd sizes (`rustfft`).
pub(crate) enum ForwardRfftPlan {
    Real(Arc<dyn RealToComplex<f32>>),
    Complex(Arc<dyn Fft<f32>>),
    TrivialOne,
}

impl ForwardRfftPlan {
    pub(crate) fn new(n_fft: usize) -> Result<Self, DspError> {
        if n_fft == 0 {
            return Err(DspError::InvalidParameter {
                name: "n_fft",
                reason: "must be > 0".to_string(),
            });
        }
        if n_fft == 1 {
            return Ok(Self::TrivialOne);
        }
        if n_fft.is_multiple_of(2) {
            let mut planner = RealFftPlanner::<f32>::new();
            Ok(Self::Real(planner.plan_fft_forward(n_fft)))
        } else {
            let mut planner = FftPlanner::<f32>::new();
            Ok(Self::Complex(planner.plan_fft_forward(n_fft)))
        }
    }

    pub(crate) fn new_scratch(
        &self,
        n_fft: usize,
    ) -> (Vec<f32>, Vec<Complex32>, Vec<Complex32>, Vec<Complex32>) {
        let n_freq = n_fft / 2 + 1;
        match self {
            Self::Real(plan) => (
                vec![0.0; n_fft],
                vec![Complex32::default(); n_freq],
                vec![Complex32::default(); plan.get_scratch_len()],
                Vec::new(),
            ),
            Self::Complex(plan) => (
                vec![0.0; n_fft],
                vec![Complex32::default(); n_freq],
                vec![Complex32::default(); plan.get_inplace_scratch_len()],
                vec![Complex32::default(); n_fft],
            ),
            Self::TrivialOne => (
                vec![0.0; 1],
                vec![Complex32::default(); 1],
                Vec::new(),
                Vec::new(),
            ),
        }
    }

    pub(crate) fn process(
        &self,
        time_buf: &mut [f32],
        out_spec: &mut [Complex32],
        scratch: &mut [Complex32],
        complex_buf: &mut [Complex32],
    ) -> Result<(), DspError> {
        match self {
            Self::Real(plan) => {
                plan.process_with_scratch(time_buf, out_spec, scratch)
                    .map_err(|e| DspError::FftError(e.to_string()))?;
            }
            Self::Complex(plan) => {
                for (dst, &src) in complex_buf.iter_mut().zip(time_buf.iter()) {
                    *dst = Complex32::new(src, 0.0);
                }
                plan.process_with_scratch(complex_buf, scratch);
                out_spec.copy_from_slice(&complex_buf[..out_spec.len()]);
            }
            Self::TrivialOne => {
                out_spec[0] = Complex32::new(time_buf[0], 0.0);
            }
        }
        Ok(())
    }
}

pub(crate) enum InverseRfftPlan {
    Real(Arc<dyn ComplexToReal<f32>>),
    Complex(Arc<dyn Fft<f32>>),
    TrivialOne,
}

impl InverseRfftPlan {
    pub(crate) fn new(n_fft: usize) -> Result<Self, DspError> {
        if n_fft == 0 {
            return Err(DspError::InvalidParameter {
                name: "n_fft",
                reason: "must be > 0".to_string(),
            });
        }
        if n_fft == 1 {
            return Ok(Self::TrivialOne);
        }
        if n_fft.is_multiple_of(2) {
            let mut planner = RealFftPlanner::<f32>::new();
            Ok(Self::Real(planner.plan_fft_inverse(n_fft)))
        } else {
            let mut planner = FftPlanner::<f32>::new();
            Ok(Self::Complex(planner.plan_fft_inverse(n_fft)))
        }
    }

    pub(crate) fn new_scratch(
        &self,
        n_fft: usize,
    ) -> (Vec<Complex32>, Vec<f32>, Vec<Complex32>, Vec<Complex32>) {
        let n_freq = n_fft / 2 + 1;
        match self {
            Self::Real(plan) => (
                vec![Complex32::default(); n_freq],
                vec![0.0; n_fft],
                vec![Complex32::default(); plan.get_scratch_len()],
                Vec::new(),
            ),
            Self::Complex(plan) => (
                vec![Complex32::default(); n_freq],
                vec![0.0; n_fft],
                vec![Complex32::default(); plan.get_inplace_scratch_len()],
                vec![Complex32::default(); n_fft],
            ),
            Self::TrivialOne => (
                vec![Complex32::default(); 1],
                vec![0.0; 1],
                Vec::new(),
                Vec::new(),
            ),
        }
    }

    /// Computes normalized `irfft` (dividing by `n_fft`), matching `scipy.fft.irfft` and `torch.fft.irfft`.
    pub(crate) fn process(
        &self,
        in_spec: &mut [Complex32],
        out_time: &mut [f32],
        scratch: &mut [Complex32],
        complex_buf: &mut [Complex32],
    ) -> Result<(), DspError> {
        let n_fft = out_time.len();
        let inv_n = 1.0_f32 / n_fft as f32;
        match self {
            Self::Real(plan) => {
                // realfft requires strictly zero imaginary components at DC and Nyquist.
                in_spec[0].im = 0.0;
                if let Some(last) = in_spec.last_mut() {
                    last.im = 0.0;
                }
                plan.process_with_scratch(in_spec, out_time, scratch)
                    .map_err(|e| DspError::FftError(e.to_string()))?;
                for sample in out_time.iter_mut() {
                    *sample *= inv_n;
                }
            }
            Self::Complex(plan) => {
                complex_buf[0] = Complex32::new(in_spec[0].re, 0.0);
                for k in 1..in_spec.len() {
                    complex_buf[k] = in_spec[k];
                    complex_buf[n_fft - k] = in_spec[k].conj();
                }
                if n_fft.is_multiple_of(2) {
                    let nyq = n_fft / 2;
                    complex_buf[nyq] = Complex32::new(in_spec[nyq].re, 0.0);
                }
                plan.process_with_scratch(complex_buf, scratch);
                for (dst, src) in out_time.iter_mut().zip(complex_buf.iter()) {
                    *dst = src.re * inv_n;
                }
            }
            Self::TrivialOne => {
                out_time[0] = in_spec[0].re;
            }
        }
        Ok(())
    }
}

/// Computes a single 1D real-to-complex FFT returning `n_fft / 2 + 1` bins.
pub(crate) fn rfft_1d(input: &[f32]) -> Result<Vec<Complex32>, DspError> {
    let n_fft = input.len();
    let plan = ForwardRfftPlan::new(n_fft)?;
    let (mut time_buf, mut out_spec, mut scratch, mut complex_buf) = plan.new_scratch(n_fft);
    time_buf.copy_from_slice(input);
    plan.process(&mut time_buf, &mut out_spec, &mut scratch, &mut complex_buf)?;
    Ok(out_spec)
}

/// Computes a single 1D normalized complex-to-real inverse FFT of length `n_fft`.
pub(crate) fn irfft_1d(spec: &[Complex32], n_fft: usize) -> Result<Vec<f32>, DspError> {
    let expected = n_fft / 2 + 1;
    if spec.len() != expected {
        return Err(DspError::ShapeMismatch {
            expected: format!("len == {expected} for n_fft={n_fft}"),
            actual: format!("len == {}", spec.len()),
        });
    }
    let plan = InverseRfftPlan::new(n_fft)?;
    let (mut in_spec, mut out_time, mut scratch, mut complex_buf) = plan.new_scratch(n_fft);
    in_spec.copy_from_slice(spec);
    plan.process(&mut in_spec, &mut out_time, &mut scratch, &mut complex_buf)?;
    Ok(out_time)
}

/// Computes `irfft` of a purely real onesided spectrum (zero imaginary parts).
pub(crate) fn irfft_real_1d(spec_real: &[f32], n_fft: usize) -> Result<Vec<f32>, DspError> {
    let expected = n_fft / 2 + 1;
    if spec_real.len() != expected {
        return Err(DspError::ShapeMismatch {
            expected: format!("len == {expected} for n_fft={n_fft}"),
            actual: format!("len == {}", spec_real.len()),
        });
    }
    let complex_spec: Vec<Complex32> = spec_real
        .iter()
        .map(|&re| Complex32::new(re, 0.0))
        .collect();
    irfft_1d(&complex_spec, n_fft)
}

fn pad_window_to_nfft(window: &[f32], n_fft: usize) -> Result<Vec<f32>, DspError> {
    if window.is_empty() {
        return Err(DspError::InvalidParameter {
            name: "window",
            reason: "cannot be empty".to_string(),
        });
    }
    if window.len() > n_fft {
        return Err(DspError::InvalidParameter {
            name: "window",
            reason: format!("win_length ({}) must be <= n_fft ({n_fft})", window.len()),
        });
    }
    if window.len() == n_fft {
        return Ok(window.to_vec());
    }
    let pad_l = (n_fft - window.len()) / 2;
    let mut padded = vec![0.0_f32; n_fft];
    padded[pad_l..pad_l + window.len()].copy_from_slice(window);
    Ok(padded)
}

/// Computes onesided complex STFT matching `_stft_np` in `util/onnx_dsp.py` and
/// `torch.stft(..., normalized=False, onesided=True)`.
///
/// Output shape is `[n_fft / 2 + 1, n_frames]`.
pub fn stft_1d(
    x: &[f32],
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
    center: bool,
    pad_mode: PadMode,
) -> Result<Array2<Complex32>, DspError> {
    if n_fft == 0 {
        return Err(DspError::InvalidParameter {
            name: "n_fft",
            reason: "must be > 0".to_string(),
        });
    }
    if hop_length == 0 {
        return Err(DspError::InvalidParameter {
            name: "hop_length",
            reason: "must be > 0".to_string(),
        });
    }
    let window_padded = pad_window_to_nfft(window, n_fft)?;

    let owned_padded;
    let x_padded: &[f32] = if center {
        let pad_c = n_fft / 2;
        owned_padded = match pad_mode {
            PadMode::Reflect => reflect_pad_1d(x, pad_c as isize, pad_c as isize),
            PadMode::Constant(val) => constant_pad_1d(x, pad_c, pad_c, val),
        };
        &owned_padded
    } else {
        x
    };

    if x_padded.len() < n_fft {
        return Err(DspError::SignalTooShort {
            actual: x_padded.len(),
            minimum: n_fft,
        });
    }

    let n_frames = (x_padded.len() - n_fft) / hop_length + 1;
    let n_freq = n_fft / 2 + 1;
    let mut spec = Array2::<Complex32>::zeros((n_freq, n_frames));

    let plan = ForwardRfftPlan::new(n_fft)?;
    let (mut time_buf, mut out_spec, mut scratch, mut complex_buf) = plan.new_scratch(n_fft);

    for t in 0..n_frames {
        let start = t * hop_length;
        let frame = &x_padded[start..start + n_fft];
        for ((dst, &s), &w) in time_buf
            .iter_mut()
            .zip(frame.iter())
            .zip(window_padded.iter())
        {
            *dst = s * w;
        }
        plan.process(&mut time_buf, &mut out_spec, &mut scratch, &mut complex_buf)?;
        for (f, &val) in out_spec.iter().enumerate() {
            spec[[f, t]] = val;
        }
    }

    Ok(spec)
}

/// Computes overlap-add inverse STFT matching `_istft_ola_np` in `util/onnx_dsp.py` and
/// `torch.istft(..., normalized=False, onesided=True)`.
///
/// Expects `spec` shape `[n_fft / 2 + 1, n_frames]`.
/// When `length` is `Some(target_len)` (used to match `torch.istft(..., length=...)`),
/// trims or zero-pads the centered/uncropped output to `target_len`.
pub fn istft_1d(
    spec: ArrayView2<Complex32>,
    n_fft: usize,
    hop_length: usize,
    window: &[f32],
    center: bool,
    length: Option<usize>,
) -> Result<Vec<f32>, DspError> {
    if n_fft == 0 {
        return Err(DspError::InvalidParameter {
            name: "n_fft",
            reason: "must be > 0".to_string(),
        });
    }
    if hop_length == 0 {
        return Err(DspError::InvalidParameter {
            name: "hop_length",
            reason: "must be > 0".to_string(),
        });
    }
    let n_freq = n_fft / 2 + 1;
    if spec.nrows() != n_freq {
        return Err(DspError::ShapeMismatch {
            expected: format!("[{n_freq}, n_frames]"),
            actual: format!("[{}, {}]", spec.nrows(), spec.ncols()),
        });
    }
    let n_frames = spec.ncols();
    if n_frames == 0 {
        return Err(DspError::InvalidParameter {
            name: "spec",
            reason: "n_frames must be > 0".to_string(),
        });
    }

    let window_padded = pad_window_to_nfft(window, n_fft)?;
    let out_len = (n_frames - 1) * hop_length + n_fft;
    let mut out = vec![0.0_f32; out_len];
    let mut win_sum = vec![0.0_f32; out_len];
    let win_sq: Vec<f32> = window_padded.iter().map(|&w| w * w).collect();

    let plan = InverseRfftPlan::new(n_fft)?;
    let (mut in_spec, mut out_time, mut scratch, mut complex_buf) = plan.new_scratch(n_fft);

    for t in 0..n_frames {
        for f in 0..n_freq {
            in_spec[f] = spec[[f, t]];
        }
        plan.process(&mut in_spec, &mut out_time, &mut scratch, &mut complex_buf)?;

        let start = t * hop_length;
        for k in 0..n_fft {
            out[start + k] += out_time[k] * window_padded[k];
            win_sum[start + k] += win_sq[k];
        }
    }

    let start_idx = if center { n_fft / 2 } else { 0 };
    let default_end = if center {
        out_len.saturating_sub(n_fft / 2)
    } else {
        out_len
    };

    match length {
        Some(target_len) => {
            let mut result = vec![0.0_f32; target_len];
            let avail = out_len.saturating_sub(start_idx).min(target_len);
            for (i, dst) in result.iter_mut().enumerate().take(avail) {
                let idx = start_idx + i;
                *dst = out[idx] / win_sum[idx].max(1e-11);
            }
            Ok(result)
        }
        None => {
            if default_end <= start_idx {
                return Ok(Vec::new());
            }
            let mut result = Vec::with_capacity(default_end - start_idx);
            for idx in start_idx..default_end {
                result.push(out[idx] / win_sum[idx].max(1e-11));
            }
            Ok(result)
        }
    }
}

/// Applies the HN-SEP complex bounded mask activation `tanh(|m|) * m / (|m| + eps)` in-place.
///
/// Note: The exported `hnsep/vr/model.onnx` already incorporates this activation inside its graph,
/// matching `NumpyHnsepModel.forward`. Call this helper only on unbounded raw mask tensors.
pub fn apply_bounded_mask(mask: &mut Array2<Complex32>, eps: f32) {
    for c in mask.iter_mut() {
        let mag = (c.re * c.re + c.im * c.im).sqrt();
        let scale = mag.tanh() / (mag + eps);
        c.re *= scale;
        c.im *= scale;
    }
}

/// Applies the HN-SEP complex bounded mask activation to a `[B, 2, F, T]` real/imag tensor.
pub fn apply_bounded_mask_tensor(mask: &mut Array4<f32>, eps: f32) -> Result<(), DspError> {
    let shape = mask.dim();
    if shape.1 != 2 {
        return Err(DspError::ShapeMismatch {
            expected: "[B, 2, F, T]".to_string(),
            actual: format!("[{}, {}, {}, {}]", shape.0, shape.1, shape.2, shape.3),
        });
    }
    let (b, _, n_freq, n_frames) = shape;
    for bi in 0..b {
        for f in 0..n_freq {
            for t in 0..n_frames {
                let re = mask[[bi, 0, f, t]];
                let im = mask[[bi, 1, f, t]];
                let mag = (re * re + im * im).sqrt();
                let scale = mag.tanh() / (mag + eps);
                mask[[bi, 0, f, t]] = re * scale;
                mask[[bi, 1, f, t]] = im * scale;
            }
        }
    }
    Ok(())
}

/// Multiplies a complex STFT spectrogram `[n_freq, n_frames]` by a complex mask tensor
/// of shape `[1, 2, F_mask, n_frames]` (accepting both `n_freq` = 1025 and `n_freq - 1` = 1024,
/// edge-replicating the Nyquist bin when `F_mask == n_freq - 1`).
pub fn hnsep_bounded_mask_multiply(
    spec: ArrayView2<Complex32>,
    mask_tensor: ArrayView4<f32>,
) -> Result<Array2<Complex32>, DspError> {
    let (n_freq, n_frames) = spec.dim();
    let (mb, mc, mf, mt) = mask_tensor.dim();
    if mb != 1 || mc != 2 || mt != n_frames || (mf != n_freq && mf + 1 != n_freq) {
        return Err(DspError::ShapeMismatch {
            expected: format!(
                "[1, 2, {n_freq} (or {}), {n_frames}]",
                n_freq.saturating_sub(1)
            ),
            actual: format!("[{mb}, {mc}, {mf}, {mt}]"),
        });
    }

    let mut out = Array2::<Complex32>::zeros((n_freq, n_frames));
    for f in 0..n_freq {
        let mf_idx = f.min(mf - 1);
        for t in 0..n_frames {
            let s = spec[[f, t]];
            let m = Complex32::new(
                mask_tensor[[0, 0, mf_idx, t]],
                mask_tensor[[0, 1, mf_idx, t]],
            );
            out[[f, t]] = s * m;
        }
    }
    Ok(out)
}

/// Prepared HN-SEP STFT spectrogram and ONNX input tensor `[1, 2, 1025, T]`.
#[derive(Debug, Clone)]
pub struct HnsepPrepared {
    /// Complex STFT spectrogram `[n_fft / 2 + 1, n_frames]` (e.g. `[1025, T]`).
    pub spec: Array2<Complex32>,
    /// Contiguous `[1, 2, n_fft / 2 + 1, n_frames]` float32 tensor (channel 0 = real, 1 = imag).
    pub tensor: Array4<f32>,
    /// Original unpadded audio sample count.
    pub orig_len: usize,
    /// Left zero-padding length applied before STFT (used to slice reconstructed audio).
    pub tl_pad: usize,
}

/// Clean-room HN-SEP STFT/ISTFT processor matching `util/onnx_dsp.py::NumpyHnsepModel`.
#[derive(Debug, Clone)]
pub struct HnsepStft {
    n_fft: usize,
    hop_length: usize,
    window: Vec<f32>,
}

impl Default for HnsepStft {
    fn default() -> Self {
        Self::new(2048, 512).expect("default HnsepStft parameters (2048, 512) are valid")
    }
}

impl HnsepStft {
    /// Creates an `HnsepStft` configured with `n_fft` and `hop_length` (typically 2048 and 512).
    pub fn new(n_fft: usize, hop_length: usize) -> Result<Self, DspError> {
        if n_fft == 0 {
            return Err(DspError::InvalidParameter {
                name: "n_fft",
                reason: "must be > 0".to_string(),
            });
        }
        if hop_length == 0 {
            return Err(DspError::InvalidParameter {
                name: "hop_length",
                reason: "must be > 0".to_string(),
            });
        }
        let window = periodic_hann_window(n_fft);
        Ok(Self {
            n_fft,
            hop_length,
            window,
        })
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Pads mono audio to a `32 * hop_length` segment boundary, runs centered constant-padded STFT,
    /// and packs `[1, 2, n_fft / 2 + 1, T]` real/imag tensor for ONNX HN-SEP inference.
    pub fn prepare_input(&self, wave: &[f32]) -> Result<HnsepPrepared, DspError> {
        let orig_len = wave.len();
        let t1 = orig_len + self.hop_length;
        let seg_length = 32 * self.hop_length;
        let t_pad = seg_length * ((t1 - 1) / seg_length + 1) - t1;
        let nl_pad = (t_pad / 2) / self.hop_length;
        let tl_pad = nl_pad * self.hop_length;
        let tr_pad = t_pad - tl_pad;

        let x_padded = constant_pad_1d(wave, tl_pad, tr_pad, 0.0);
        let spec = stft_1d(
            &x_padded,
            self.n_fft,
            self.hop_length,
            &self.window,
            true,
            PadMode::Constant(0.0),
        )?;

        let (n_freq, n_frames) = spec.dim();
        let mut tensor = Array4::<f32>::zeros((1, 2, n_freq, n_frames));
        for f in 0..n_freq {
            for t in 0..n_frames {
                let c = spec[[f, t]];
                tensor[[0, 0, f, t]] = c.re;
                tensor[[0, 1, f, t]] = c.im;
            }
        }

        Ok(HnsepPrepared {
            spec,
            tensor,
            orig_len,
            tl_pad,
        })
    }

    /// Applies an ONNX `[1, 2, F, T]` complex mask tensor and synthesizes the separated harmonic
    /// waveform cropped back to `prepared.orig_len`.
    pub fn reconstruct_from_mask(
        &self,
        prepared: &HnsepPrepared,
        mask_tensor: ArrayView4<f32>,
    ) -> Result<Vec<f32>, DspError> {
        if prepared.orig_len == 0 {
            return Ok(Vec::new());
        }
        let spec_pred = hnsep_bounded_mask_multiply(prepared.spec.view(), mask_tensor)?;
        let x_pred = istft_1d(
            spec_pred.view(),
            self.n_fft,
            self.hop_length,
            &self.window,
            true,
            None,
        )?;
        let end = prepared.tl_pad + prepared.orig_len;
        if end > x_pred.len() {
            return Err(DspError::ShapeMismatch {
                expected: format!("ISTFT output len >= {end}"),
                actual: format!("{}", x_pred.len()),
            });
        }
        Ok(x_pred[prepared.tl_pad..end].to_vec())
    }

    /// Applies a complex mask `[n_freq, n_frames]` and synthesizes the separated harmonic waveform.
    pub fn reconstruct_from_complex_mask(
        &self,
        prepared: &HnsepPrepared,
        mask: ArrayView2<Complex32>,
    ) -> Result<Vec<f32>, DspError> {
        if prepared.orig_len == 0 {
            return Ok(Vec::new());
        }
        if mask.dim() != prepared.spec.dim() {
            return Err(DspError::ShapeMismatch {
                expected: format!("{:?}", prepared.spec.dim()),
                actual: format!("{:?}", mask.dim()),
            });
        }
        let spec_pred = &prepared.spec * &mask;
        let x_pred = istft_1d(
            spec_pred.view(),
            self.n_fft,
            self.hop_length,
            &self.window,
            true,
            None,
        )?;
        let end = prepared.tl_pad + prepared.orig_len;
        if end > x_pred.len() {
            return Err(DspError::ShapeMismatch {
                expected: format!("ISTFT output len >= {end}"),
                actual: format!("{}", x_pred.len()),
            });
        }
        Ok(x_pred[prepared.tl_pad..end].to_vec())
    }
}
