//! UTAU `A`-flag pitch-slope amplitude modulation, feature scale restoration,
//! and output volume scaling with peak limiting (from `backend/resampler.py`).

/// Computes NumPy-compatible 1-D discrete gradient `np.gradient(y, dx)` using
/// first-order forward/backward differences at boundaries and second-order central
/// differences in the interior.
pub fn gradient_1d_uniform(y: &[f64], dx: f64) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 || dx == 0.0 || !dx.is_finite() {
        return vec![0.0; n];
    }
    let mut out = vec![0.0_f64; n];
    out[0] = (y[1] - y[0]) / dx;
    let two_dx = 2.0 * dx;
    for i in 1..(n - 1) {
        out[i] = (y[i + 1] - y[i - 1]) / two_dx;
    }
    out[n - 1] = (y[n - 1] - y[n - 2]) / dx;
    out
}

/// Evaluates piecewise-linear interpolation at `x` over sorted breakpoints `xp`
/// and values `fp`, clamping outside `[xp[0], xp[last]]` to boundary values
/// (`np.interp(x, xp, fp, left=fp[0], right=fp[-1])`).
#[inline]
pub fn interp_1d_clamped(xp: &[f64], fp: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let pos = xp.partition_point(|&v| v <= x);
    let i0 = pos.saturating_sub(1).min(n - 2);
    let i1 = i0 + 1;
    let span = xp[i1] - xp[i0];
    if span <= 0.0 {
        return fp[i0];
    }
    let frac = (x - xp[i0]) / span;
    fp[i0] + frac * (fp[i1] - fp[i0])
}

/// Applies UTAU `A`-flag pitch-slope amplitude modulation in-place.
///
/// Matches `backend/resampler.py`:
/// - `A` is clamped to `[-100, 100]` (`a_flag == 0` is an O(1) no-op).
/// - `pitch_render` is in MIDI semitones sampled on frame times `t_mel` with step `thop`.
/// - Frame gain is `5.0_f64.powf(1e-4 * A_clamped * gradient(pitch_render, thop))`.
/// - Gain is linearly interpolated over `linspace(new_start, new_end, num=render.len(), endpoint=False)`.
pub fn apply_amplitude_modulation(
    render: &mut [f32],
    a_flag: i32,
    pitch_render: &[f64],
    t_mel: &[f64],
    thop: f64,
    new_start: f64,
    new_end: f64,
) {
    if a_flag == 0
        || render.is_empty()
        || pitch_render.len() <= 1
        || t_mel.len() <= 1
        || pitch_render.len() != t_mel.len()
        || thop <= 0.0
        || !thop.is_finite()
    {
        return;
    }

    let a_clamped = (a_flag as f64).clamp(-100.0, 100.0);
    let pitch_derivative = gradient_1d_uniform(pitch_render, thop);

    let gain_at_mel_frames: Vec<f64> = pitch_derivative
        .iter()
        .map(|&dp| 5.0_f64.powf(1e-4 * a_clamped * dp))
        .collect();

    let num_samples = render.len();
    let step = (new_end - new_start) / (num_samples as f64);
    for (n, sample) in render.iter_mut().enumerate() {
        let t_audio = new_start + (n as f64) * step;
        let g = interp_1d_clamped(t_mel, &gain_at_mel_frames, t_audio) as f32;
        *sample *= g;
    }
}

/// Restores pre-normalized feature scale (`render *= 1.0 / scale`) when `scale != 1.0`.
pub fn apply_scale_inverse(render: &mut [f32], scale: f64) {
    if scale != 1.0 && scale != 0.0 && scale.is_finite() {
        let inv = (1.0 / scale) as f32;
        for sample in render.iter_mut() {
            *sample *= inv;
        }
    }
}

/// Computes `max(|x|)` across `render` (0.0 if empty).
#[inline]
pub fn peak_abs(render: &[f32]) -> f64 {
    let mut peak = 0.0_f64;
    for &v in render {
        let a = (v as f64).abs();
        if a > peak {
            peak = a;
        }
    }
    peak
}

/// Applies UTAU `volume` scaling (`volume / 100.0`) with hard peak ceiling at `peak_limit`.
///
/// Matches `backend/resampler.py` lines 630-638.
pub fn apply_volume_and_peak_limit(render: &mut [f32], volume: f64, peak_limit: f64) {
    if render.is_empty() {
        return;
    }
    let volume_scale = volume / 100.0;
    let raw_peak = peak_abs(render);
    let scaled_peak = raw_peak * volume_scale.abs();
    let final_gain = if scaled_peak > peak_limit && scaled_peak > 1e-8 {
        (volume_scale * peak_limit) / scaled_peak
    } else {
        volume_scale
    };
    if final_gain != 1.0 {
        let g = final_gain as f32;
        for sample in render.iter_mut() {
            *sample *= g;
        }
    }
}
