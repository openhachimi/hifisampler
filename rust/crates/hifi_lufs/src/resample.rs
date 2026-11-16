//! Rational polyphase FIR resampling matching `scipy.signal.resample_poly`
//! with Kaiser window (`beta = 5.0`) and `padtype = "line"`.

use crate::LufsError;

/// Kaiser window shape parameter used by `scipy.signal.resample_poly`.
const KAISER_BETA: f64 = 5.0;
/// Half-length multiplier in `scipy.signal.resample_poly` (`half_len = 10 * max(up, down)`).
const HALF_LEN_FACTOR: usize = 10;
/// Maximum rational factor before falling back to `limit_denominator(512)`.
const MAX_POLY_FACTOR: usize = 1000;
const FALLBACK_MAX_DENOMINATOR: u64 = 512;

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Exact port of Python's `fractions.Fraction(num, den).limit_denominator(max_den)`.
pub(crate) fn limit_denominator(num: u64, den: u64, max_den: u64) -> (usize, usize) {
    let g = gcd_u64(num, den);
    let orig_n = num / g;
    let orig_d = den / g;
    if orig_d <= max_den {
        return (orig_n as usize, orig_d as usize);
    }

    let mut n = orig_n;
    let mut d = orig_d;
    let (mut p0, mut q0, mut p1, mut q1) = (0_u64, 1_u64, 1_u64, 0_u64);

    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_den {
            break;
        }
        let p2 = p0 + a * p1;
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
        let rem = n % d;
        n = d;
        d = rem;
    }

    let k = (max_den - q0) / q1;
    let bound1_n = p0 + k * p1;
    let bound1_d = q0 + k * q1;

    // Compare |orig - bound1| <= |orig - p1/q1| via cross-multiplication in u128.
    let lhs = (orig_n as u128 * bound1_d as u128).abs_diff(bound1_n as u128 * orig_d as u128)
        * (q1 as u128);
    let rhs =
        (orig_n as u128 * q1 as u128).abs_diff(p1 as u128 * orig_d as u128) * (bound1_d as u128);

    if lhs <= rhs {
        (bound1_n as usize, bound1_d as usize)
    } else {
        (p1 as usize, q1 as usize)
    }
}

/// Evaluates the modified zeroth-order Bessel function of the first kind `I_0(x)`
/// via power series (converges to full IEEE-754 `f64` precision for `|x| <= 5`).
#[inline]
fn bessel_i0(x: f64) -> f64 {
    let half_x = 0.5 * x;
    let half_x2 = half_x * half_x;
    let mut term = 1.0_f64;
    let mut sum = 1.0_f64;
    for k in 1..=32 {
        let kf = k as f64;
        term *= half_x2 / (kf * kf);
        sum += term;
        if term <= 1e-17 * sum {
            break;
        }
    }
    sum
}

/// Designs the zero-phase low-pass FIR filter matching
/// `scipy.signal.firwin(2 * half_len + 1, 1.0 / max_rate, window=('kaiser', 5.0)).astype(float32) * up`,
/// partitioned into `up` polyphase branches `branches[phase][j] = h[phase + j * up]`.
fn design_polyphase_branches(up: usize, down: usize) -> (usize, Vec<Vec<f32>>) {
    let max_rate = up.max(down);
    let f_c = 1.0_f64 / (max_rate as f64);
    let half_len = HALF_LEN_FACTOR * max_rate;
    let numtaps = 2 * half_len + 1;
    let alpha = half_len as f64;
    let inv_i0_beta = 1.0 / bessel_i0(KAISER_BETA);

    let mut taps = Vec::with_capacity(numtaps);
    let mut sum = 0.0_f64;

    for i in 0..numtaps {
        let m = (i as f64) - alpha;
        let sinc = if m == 0.0 {
            1.0
        } else {
            let arg = std::f64::consts::PI * f_c * m;
            arg.sin() / arg
        };
        let ideal = f_c * sinc;

        let t = m / alpha;
        let rad = (1.0 - t * t).max(0.0).sqrt();
        let win = bessel_i0(KAISER_BETA * rad) * inv_i0_beta;

        let v = ideal * win;
        taps.push(v);
        sum += v;
    }

    let inv_sum = 1.0 / sum;
    let up_f32 = up as f32;
    let mut branches = Vec::with_capacity(up);
    for phase in 0..up {
        let branch_len = (numtaps - phase).div_ceil(up);
        let mut branch = Vec::with_capacity(branch_len);
        let mut idx = phase;
        while idx < numtaps {
            branch.push(((taps[idx] * inv_sum) as f32) * up_f32);
            idx += up;
        }
        branches.push(branch);
    }

    (half_len, branches)
}

/// High-quality rational polyphase FIR resampling returning `Vec<f32>` with exact expected length
/// `round(x.len() * target_sr / orig_sr)`.
///
/// Mirrors `util/lufs.py::resample_poly_fast` (`scipy.signal.resample_poly` with
/// Kaiser window `beta = 5.0` and `padtype = "line"`).
pub fn resample_poly_fast(x: &[f32], orig_sr: u32, target_sr: u32) -> Result<Vec<f32>, LufsError> {
    if orig_sr == 0 {
        return Err(LufsError::InvalidSampleRate(orig_sr));
    }
    if target_sr == 0 {
        return Err(LufsError::InvalidSampleRate(target_sr));
    }
    if x.is_empty() || orig_sr == target_sr {
        return Ok(x.to_vec());
    }

    let n_in = x.len();
    let expected_len = ((n_in as f64 * target_sr as f64) / (orig_sr as f64))
        .round_ties_even()
        .max(0.0) as usize;
    if expected_len == 0 {
        return Ok(Vec::new());
    }

    let g = gcd_u64(orig_sr as u64, target_sr as u64);
    let mut up = (target_sr as u64 / g) as usize;
    let mut down = (orig_sr as u64 / g) as usize;

    if up.max(down) > MAX_POLY_FACTOR {
        let (approx_up, approx_down) =
            limit_denominator(target_sr as u64, orig_sr as u64, FALLBACK_MAX_DENOMINATOR);
        let g2 = gcd_u64(approx_up as u64, approx_down as u64);
        up = (approx_up as u64 / g2) as usize;
        down = (approx_down as u64 / g2) as usize;
    }

    if up == 1 && down == 1 {
        // Identity ratio after limit_denominator: copy and adjust length to expected_len.
        let mut out = Vec::with_capacity(expected_len);
        let copy_len = n_in.min(expected_len);
        out.extend_from_slice(&x[..copy_len]);
        if out.len() < expected_len {
            out.resize(expected_len, 0.0);
        }
        return Ok(out);
    }

    let (half_len, branches) = design_polyphase_branches(up, down);
    let n_out = (n_in * up).div_ceil(down);
    let compute_len = n_out.min(expected_len);

    // Linear trend slope for `padtype = "line"`. For len == 1, avoid NaN by extending constantly.
    let slope = if n_in > 1 {
        ((x[n_in - 1] as f64) - (x[0] as f64)) / ((n_in - 1) as f64)
    } else {
        0.0
    };
    let x_first = x[0] as f64;
    let x_last = x[n_in - 1] as f64;
    let n_in_isize = n_in as isize;

    let mut out = Vec::with_capacity(expected_len);

    for k in 0..compute_len {
        let t = k * down + half_len;
        let phase = t % up;
        let branch = &branches[phase];
        let p_start = ((t - phase) / up) as isize;
        let branch_len = branch.len() as isize;
        let p_end = p_start - branch_len + 1;

        if p_start < n_in_isize && p_end >= 0 {
            // Fast interior path: all accessed indices lie within x[0..n_in].
            let mut acc = 0.0_f64;
            let mut p = p_start as usize;
            for &coeff in branch {
                acc += (coeff as f64) * (x[p] as f64);
                p = p.saturating_sub(1);
            }
            out.push(acc as f32);
        } else {
            // Boundary path with `padtype = "line"` linear extension.
            let mut acc = 0.0_f64;
            let mut p = p_start;
            for &coeff in branch {
                let xv = if p < 0 {
                    x_first + (p as f64) * slope
                } else if p >= n_in_isize {
                    x_last + ((p - (n_in_isize - 1)) as f64) * slope
                } else {
                    x[p as usize] as f64
                };
                acc += (coeff as f64) * xv;
                p -= 1;
            }
            out.push(acc as f32);
        }
    }

    if out.len() < expected_len {
        out.resize(expected_len, 0.0);
    }

    Ok(out)
}
