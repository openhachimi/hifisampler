//! Akima 1D cubic spline interpolator, time-axis/velocity stretching, loop-mode reflection,
//! Mel-frame linear interpolation, and F0 curve synthesis.
//!
//! Clean-room port of `scipy.interpolate.Akima1DInterpolator` (1970 Akima weights &
//! extrapolated endpoint slopes) and `backend/resampler.py` time/pitch pipeline.

use ndarray::{Array2, ArrayView2};

use crate::hnsep_stft::reflect_index;
use crate::DspError;

/// Converts MIDI note number (may be fractional) to Hertz using A4 = 440 Hz equal temperament.
#[inline]
pub fn midi_to_hz(midi: f64) -> f64 {
    440.0 * ((midi - 69.0) / 12.0).exp2()
}

/// Piecewise cubic Hermite spline using Hiroshi Akima's (1970) local slope weighting,
/// matching `scipy.interpolate.Akima1DInterpolator(..., method="akima")`.
#[derive(Debug, Clone)]
pub struct Akima1DInterpolator {
    x: Vec<f64>,
    /// Polynomial coefficients `[c0, c1, c2, c3]` per interval `i` for
    /// `p(x) = ((c0 * u + c1) * u + c2) * u + c3` where `u = x - self.x[i]`.
    coeffs: Vec<[f64; 4]>,
    extrapolate: bool,
}

impl Akima1DInterpolator {
    /// Constructs an Akima 1D interpolator with `extrapolate = false` (matching SciPy default).
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, DspError> {
        Self::with_extrapolate(x, y, false)
    }

    /// Constructs an Akima 1D interpolator with explicit out-of-bounds extrapolation flag.
    pub fn with_extrapolate(x: &[f64], y: &[f64], extrapolate: bool) -> Result<Self, DspError> {
        let n = x.len();
        if n != y.len() {
            return Err(DspError::ShapeMismatch {
                expected: format!("x.len() == y.len() ({n})"),
                actual: format!("y.len() == {}", y.len()),
            });
        }
        if n < 2 {
            return Err(DspError::InvalidParameter {
                name: "x/y",
                reason: format!("Akima1DInterpolator requires at least 2 points, got {n}"),
            });
        }
        for i in 0..n {
            if !x[i].is_finite() || !y[i].is_finite() {
                return Err(DspError::InvalidParameter {
                    name: "x/y",
                    reason: "inputs must be finite".to_string(),
                });
            }
            if i > 0 && x[i] <= x[i - 1] {
                return Err(DspError::InvalidParameter {
                    name: "x",
                    reason: format!(
                        "x must be strictly increasing (x[{}]={} <= x[{}]={})",
                        i,
                        x[i],
                        i - 1,
                        x[i - 1]
                    ),
                });
            }
        }

        let t: Vec<f64> = if n == 2 {
            let mk = (y[1] - y[0]) / (x[1] - x[0]);
            vec![mk, mk]
        } else {
            // Slopes between breakpoints plus 2 extrapolated slopes on each end.
            let mut m = vec![0.0_f64; n + 3];
            for i in 0..(n - 1) {
                m[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
            }
            // Left extrapolated slopes
            m[1] = 2.0 * m[2] - m[3];
            m[0] = 2.0 * m[1] - m[2];
            // Right extrapolated slopes
            m[n + 1] = 2.0 * m[n] - m[n - 1];
            m[n + 2] = 2.0 * m[n + 1] - m[n];

            // Default fill when f1 + f2 == 0
            let mut t_slopes = vec![0.0_f64; n];
            for i in 0..n {
                t_slopes[i] = 0.5 * (m[i + 3] + m[i]);
            }

            let mut dm = vec![0.0_f64; n + 2];
            for j in 0..(n + 2) {
                dm[j] = (m[j + 1] - m[j]).abs();
            }

            let mut f12 = vec![0.0_f64; n];
            let mut mmax = f64::NEG_INFINITY;
            for i in 0..n {
                let f1 = dm[i + 2];
                let f2 = dm[i];
                let s = f1 + f2;
                f12[i] = s;
                if s > mmax {
                    mmax = s;
                }
            }

            let break_mult = 1.0e-9_f64;
            let cutoff = break_mult * mmax;
            for i in 0..n {
                if f12[i] > cutoff {
                    let f2 = dm[i];
                    t_slopes[i] = m[i + 1] + (f2 / f12[i]) * (m[i + 2] - m[i + 1]);
                }
            }
            t_slopes
        };

        // Build CubicHermiteSpline coefficients per segment [x[i], x[i+1]]
        let mut coeffs = Vec::with_capacity(n - 1);
        for i in 0..(n - 1) {
            let dx = x[i + 1] - x[i];
            let slope = (y[i + 1] - y[i]) / dx;
            let tt = (t[i] + t[i + 1] - 2.0 * slope) / dx;
            let c0 = tt / dx;
            let c1 = (slope - t[i]) / dx - tt;
            let c2 = t[i];
            let c3 = y[i];
            coeffs.push([c0, c1, c2, c3]);
        }

        Ok(Self {
            x: x.to_vec(),
            coeffs,
            extrapolate,
        })
    }

    /// Evaluates the Akima spline at query point `xq`.
    /// Returns `f64::NAN` if out of bounds and `extrapolate == false`.
    pub fn evaluate(&self, xq: f64) -> f64 {
        if !xq.is_finite() {
            return f64::NAN;
        }
        let n = self.x.len();
        if !self.extrapolate && (xq < self.x[0] || xq > self.x[n - 1]) {
            return f64::NAN;
        }
        let seg = if xq <= self.x[0] {
            0
        } else if xq >= self.x[n - 1] {
            n - 2
        } else {
            let idx = self.x.partition_point(|&xi| xi <= xq);
            idx.saturating_sub(1).min(n - 2)
        };
        let u = xq - self.x[seg];
        let [c0, c1, c2, c3] = self.coeffs[seg];
        ((c0 * u + c1) * u + c2) * u + c3
    }

    /// Evaluates the Akima spline at all points in `xs`.
    pub fn evaluate_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&xq| self.evaluate(xq)).collect()
    }

    /// Evaluates the Akima spline at points clamped to `[x[0], x[n - 1]]`.
    pub fn evaluate_clamped(&self, xs: &[f64]) -> Vec<f64> {
        let lo = self.x[0];
        let hi = self.x[self.x.len() - 1];
        xs.iter()
            .map(|&xq| self.evaluate(xq.clamp(lo, hi)))
            .collect()
    }
}

/// Parameters controlling time-axis stretching, loop-mode reflection, and F0 rendering,
/// matching `backend/resampler.py::Resampler::resample`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResampleTimeParams {
    pub sample_rate: u32,
    pub hop_size: usize,
    pub origin_hop_size: usize,
    pub fill: usize,
    pub loop_mode: bool,
    pub base_pitch_midi: f64,
    pub velocity: f64,
    pub offset_ms: f64,
    pub length_ms: f64,
    pub consonant_ms: f64,
    pub cutoff_ms: f64,
    pub t_flag: f64,
    pub tempo: f64,
    pub modulation: f64,
}

impl Default for ResampleTimeParams {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            hop_size: 512,
            origin_hop_size: 128,
            fill: 6,
            loop_mode: true,
            base_pitch_midi: 60.0,
            velocity: 100.0,
            offset_ms: 0.0,
            length_ms: 1000.0,
            consonant_ms: 0.0,
            cutoff_ms: 0.0,
            t_flag: 0.0,
            tempo: 100.0,
            modulation: 0.0,
        }
    }
}

/// Resampled Mel spectrogram, time axis, F0 curve, and vocoder trim boundaries.
#[derive(Debug, Clone)]
pub struct ResampledFeatures {
    /// Time-stretched Mel spectrogram `[n_mels, n_frames]` ready for vocoder or vocal-fry pass.
    pub mel_render: Array2<f32>,
    /// Fundamental frequency curve in Hz `[n_frames]`.
    pub f0_render: Vec<f32>,
    /// Pitch curve in MIDI note units `[n_frames]` (used by A-flag modulation).
    pub pitch_render: Vec<f64>,
    /// Mel frame timestamps in seconds (`[0, thop, 2*thop, ...]`).
    pub t_mel: Vec<f64>,
    /// Start timestamp in seconds within the synthesized waveform (`new_start`).
    pub new_start: f64,
    /// End timestamp in seconds within the synthesized waveform (`new_end`).
    pub new_end: f64,
    /// Consonant boundary timestamp (`new_end - length_req`).
    pub new_con: f64,
    /// Sample start index for cropping the synthesized waveform.
    pub cut_l: usize,
    /// Sample end index for cropping the synthesized waveform.
    pub cut_r: usize,
}

/// Applies velocity/consonant/stretch time-axis mapping, optional `loop_mode` reflection,
/// `fill` margin framing, Mel-frame linear interpolation, and Akima F0 generation.
pub fn resample_mel_and_pitch(
    mel_origin: ArrayView2<f32>,
    pitchbend_cents: &[i32],
    params: &ResampleTimeParams,
) -> Result<ResampledFeatures, DspError> {
    if params.sample_rate == 0 || params.hop_size == 0 || params.origin_hop_size == 0 {
        return Err(DspError::InvalidParameter {
            name: "sample_rate/hop_size/origin_hop_size",
            reason: "must all be > 0".to_string(),
        });
    }
    let n_mels = mel_origin.nrows();
    let n_origin_init = mel_origin.ncols();
    if n_mels == 0 || n_origin_init == 0 {
        return Err(DspError::InvalidParameter {
            name: "mel_origin",
            reason: "must have non-zero mels and frames".to_string(),
        });
    }

    let sr = params.sample_rate as f64;
    let thop_origin = params.origin_hop_size as f64 / sr;
    let thop = params.hop_size as f64 / sr;

    let mut total_time = (n_origin_init as f64 - 1.0) * thop_origin + thop_origin;
    let vel = (1.0 - params.velocity / 100.0).exp2();
    let offset = params.offset_ms / 1000.0;
    let cutoff = params.cutoff_ms / 1000.0;
    let start = offset;

    let end = if params.cutoff_ms < 0.0 {
        start - cutoff
    } else {
        total_time - cutoff
    };
    let con = start + params.consonant_ms / 1000.0;

    let length_rounded = params.length_ms.round_ties_even().max(1.0);
    let length_req = length_rounded / 1000.0;
    let mut stretch_length = end - con;

    // Optional loop_mode reflection (when CONFIG.loop_mode or He != 0)
    let owned_looped: Option<Array2<f32>>;
    let mel_src: ArrayView2<f32> = if params.loop_mode {
        let con_frame = (((con + thop_origin / 2.0) / thop_origin).floor() as isize)
            .clamp(0, (n_origin_init as isize) - 1) as usize;
        let end_frame = (((end + thop_origin / 2.0) / thop_origin).floor() as isize)
            .clamp(con_frame as isize + 1, n_origin_init as isize) as usize;
        let loop_len = end_frame - con_frame;

        if loop_len > 0 {
            let pad_loop_size = ((length_req / thop_origin).floor() as isize + 1).max(1) as usize;
            let new_cols = con_frame + loop_len + pad_loop_size;
            let mut looped = Array2::<f32>::zeros((n_mels, new_cols));

            for m in 0..n_mels {
                for j in 0..con_frame {
                    looped[[m, j]] = mel_origin[[m, j]];
                }
                for j in 0..(loop_len + pad_loop_size) {
                    let src_rel = reflect_index(j as isize, loop_len);
                    looped[[m, con_frame + j]] = mel_origin[[m, con_frame + src_rel]];
                }
            }
            stretch_length = pad_loop_size as f64 * thop_origin;
            total_time = new_cols as f64 * thop_origin;
            owned_looped = Some(looped);
            owned_looped.as_ref().expect("just set").view()
        } else {
            mel_origin
        }
    } else {
        mel_origin
    };

    let n_origin = mel_src.ncols();
    let t_origin_first = thop_origin / 2.0;
    let t_origin_last = (n_origin as f64 - 1.0) * thop_origin + thop_origin / 2.0;

    let scaling_ratio = if stretch_length <= 1e-6 {
        1.0
    } else if stretch_length < length_req {
        length_req / stretch_length
    } else {
        1.0
    };

    let stretched_n_frames =
        (((con * vel + (total_time - con).max(0.0) * scaling_ratio) / thop).floor() as isize + 1)
            .max(1) as usize;

    let stretched_t_mel: Vec<f64> = (0..stretched_n_frames)
        .map(|i| (i as f64) * thop + thop / 2.0)
        .collect();

    let start_left_mel_frames = ((start * vel + thop / 2.0) / thop).floor();
    let mut cut_left_mel_frames = if start_left_mel_frames > params.fill as f64 {
        start_left_mel_frames - params.fill as f64
    } else {
        0.0
    };

    let end_right_mel_frames =
        stretched_n_frames as f64 - ((length_req + con * vel + thop / 2.0) / thop).floor();
    let cut_right_mel_frames = if end_right_mel_frames > params.fill as f64 {
        end_right_mel_frames - params.fill as f64
    } else {
        0.0
    };

    let slice_start = (cut_left_mel_frames.trunc() as isize).max(0) as usize;
    let slice_end_raw = (stretched_n_frames as isize - cut_right_mel_frames.trunc() as isize)
        .max(slice_start as isize + 1) as usize;
    let active_t_mel: &[f64] = if slice_start < stretched_t_mel.len() {
        let slice_end = slice_end_raw.min(stretched_t_mel.len());
        &stretched_t_mel[slice_start..slice_end]
    } else {
        cut_left_mel_frames = 0.0;
        &stretched_t_mel
    };

    let n_render = active_t_mel.len();
    let stretch_t_mel: Vec<f64> = active_t_mel
        .iter()
        .map(|&tm| {
            let s = if tm < vel * con {
                tm / vel
            } else {
                con + (tm - vel * con) / scaling_ratio
            };
            s.clamp(t_origin_first, t_origin_last)
        })
        .collect();

    let new_start = start * vel - cut_left_mel_frames * thop;
    let new_end = (length_req + con * vel) - cut_left_mel_frames * thop;
    let new_con = new_end - length_req;

    // Linear interpolation on the uniform origin mel grid
    let mut mel_render = Array2::<f32>::zeros((n_mels, n_render));
    if n_origin <= 1 {
        for m in 0..n_mels {
            let val = mel_src[[m, 0]];
            for j in 0..n_render {
                mel_render[[m, j]] = val;
            }
        }
    } else {
        let max_idx = (n_origin - 1) as f64;
        for (j, &ts) in stretch_t_mel.iter().enumerate() {
            let pos = ((ts - t_origin_first) / thop_origin).clamp(0.0, max_idx);
            let idx0 = pos.floor() as usize;
            let idx1 = (idx0 + 1).min(n_origin - 1);
            let frac = (pos - idx0 as f64) as f32;
            let w0 = 1.0 - frac;
            for m in 0..n_mels {
                mel_render[[m, j]] = mel_src[[m, idx0]] * w0 + mel_src[[m, idx1]] * frac;
            }
        }
    }

    let t_mel: Vec<f64> = (0..n_render).map(|j| j as f64 * thop).collect();

    // Pitch curve construction (MIDI base pitch + pitchbend cents + t flag)
    let t_shift = if params.t_flag != 0.0 && params.t_flag.is_finite() {
        params.t_flag.clamp(-120.0, 120.0) / 10.0
    } else {
        0.0
    };
    let tempo = params.tempo.max(1.0);

    let pitch_pts: Vec<f64> = if pitchbend_cents.is_empty() {
        vec![params.base_pitch_midi + t_shift]
    } else {
        pitchbend_cents
            .iter()
            .map(|&c| (c as f64) / 100.0 + params.base_pitch_midi + t_shift)
            .collect()
    };

    let all_equal =
        pitch_pts.len() < 2 || pitch_pts.iter().all(|&p| (p - pitch_pts[0]).abs() == 0.0);

    let (pitch_render, f0_render) = if all_equal {
        let const_pitch = pitch_pts[0];
        let const_f0 = midi_to_hz(const_pitch) as f32;
        (vec![const_pitch; n_render], vec![const_f0; n_render])
    } else {
        let tick_dt = 60.0 / (tempo * 96.0);
        let t_pitch: Vec<f64> = (0..pitch_pts.len())
            .map(|i| (i as f64) * tick_dt + new_start)
            .collect();
        let akima = Akima1DInterpolator::new(&t_pitch, &pitch_pts)?;
        let lo = new_start;
        let hi = t_pitch[t_pitch.len() - 1];
        let mut p_out = Vec::with_capacity(n_render);
        let mut f_out = Vec::with_capacity(n_render);
        for &tm in &t_mel {
            let p = akima.evaluate(tm.clamp(lo, hi));
            p_out.push(p);
            f_out.push(midi_to_hz(p) as f32);
        }
        (p_out, f_out)
    };

    let cut_l = if new_start <= 0.0 {
        0
    } else {
        (new_start * sr).trunc() as usize
    };
    let cut_r_raw = if new_end <= 0.0 {
        0isize
    } else {
        (new_end * sr).trunc() as isize
    };
    let cut_r = cut_r_raw.max(cut_l as isize + 1) as usize;

    Ok(ResampledFeatures {
        mel_render,
        f0_render,
        pitch_render,
        t_mel,
        new_start,
        new_end,
        new_con,
        cut_l,
        cut_r,
    })
}
