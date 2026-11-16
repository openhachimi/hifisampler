//! ITU-R BS.1770-4 K-weighting SOS biquad coefficient calculation and Direct Form II Transposed filtering.

use crate::LufsError;

/// Head-related high-shelf filter prototype corner frequency (Hz).
#[allow(clippy::excessive_precision)]
const HS_FC: f64 = 1681.9744509555319;
/// Head-related high-shelf filter gain (dB).
const HS_GAIN_DB: f64 = 3.99984385397;
/// Head-related high-shelf filter quality factor.
const HS_Q: f64 = 0.7071752369554193;

/// RLB high-pass filter prototype corner frequency (Hz).
const HP_FC: f64 = 38.13547087613982;
/// RLB high-pass filter quality factor.
const HP_Q: f64 = 0.5003270373253953;

/// Precomputed exact 44,100 Hz ITU-R BS.1770-4 K-weighting SOS matrix:
/// `[b0, b1, b2, a0, a1, a2]` for each of the two cascaded biquad stages.
pub const K_WEIGHT_SOS_44100: [[f64; 6]; 2] = [
    [
        1.5308412300498355,
        -2.6509799951536985,
        1.1690790799210682,
        1.0,
        -1.6636551132560204,
        0.7125954280732254,
    ],
    [1.0, -2.0, 1.0, 1.0, -1.9891696736297957, 0.9891990357870394],
];

/// Computes the 2x6 ITU-R BS.1770-4 K-weighting Second-Order Sections (SOS) coefficients
/// `[b0, b1, b2, a0, a1, a2]` for an arbitrary positive sample rate.
///
/// Uses exact prewarped bilinear transform matching `util/lufs.py::_k_weighting_sos`.
pub fn k_weighting_sos(sample_rate: u32) -> Result<[[f64; 6]; 2], LufsError> {
    if sample_rate == 0 {
        return Err(LufsError::InvalidSampleRate(sample_rate));
    }
    if sample_rate == 44100 {
        return Ok(K_WEIGHT_SOS_44100);
    }
    let min_sr = (2.0 * HS_FC) as u32;
    if sample_rate <= min_sr {
        return Err(LufsError::SampleRateTooLowForKWeighting {
            sample_rate,
            min_hz: 2.0 * HS_FC,
        });
    }

    let sr = sample_rate as f64;

    // Stage 1: head-related high-shelf via exact prewarped bilinear transform.
    let k_hs = (std::f64::consts::PI * HS_FC / sr).tan();
    let vh = 10.0_f64.powf(HS_GAIN_DB / 20.0);
    let vb = vh.powf(0.499666774155);
    let k_hs2 = k_hs * k_hs;
    let a0_hs = 1.0 + k_hs / HS_Q + k_hs2;
    let b0_hs = (vh + vb * k_hs / HS_Q + k_hs2) / a0_hs;
    let b1_hs = 2.0 * (k_hs2 - vh) / a0_hs;
    let b2_hs = (vh - vb * k_hs / HS_Q + k_hs2) / a0_hs;
    let a1_hs = 2.0 * (k_hs2 - 1.0) / a0_hs;
    let a2_hs = (1.0 - k_hs / HS_Q + k_hs2) / a0_hs;

    // Stage 2: RLB high-pass via exact prewarped bilinear transform.
    let k_hp = (std::f64::consts::PI * HP_FC / sr).tan();
    let k_hp2 = k_hp * k_hp;
    let a0_hp = 1.0 + k_hp / HP_Q + k_hp2;
    let a1_hp = 2.0 * (k_hp2 - 1.0) / a0_hp;
    let a2_hp = (1.0 - k_hp / HP_Q + k_hp2) / a0_hp;

    Ok([
        [b0_hs, b1_hs, b2_hs, 1.0, a1_hs, a2_hs],
        [1.0, -2.0, 1.0, 1.0, a1_hp, a2_hp],
    ])
}

/// Filters a 1D `f32` signal through cascaded Second-Order Sections (`[b0, b1, b2, a0, a1, a2]`)
/// using Direct Form II Transposed with zero initial state, matching `scipy.signal.sosfilt`.
pub fn apply_sos_filter(samples: &[f32], sos: &[[f64; 6]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(samples.len());
    let mut states = vec![[0.0_f64; 2]; sos.len()];
    let coeffs: Vec<[f64; 5]> = sos
        .iter()
        .map(|s| {
            let inv_a0 = if s[3] == 1.0 { 1.0 } else { 1.0 / s[3] };
            [
                s[0] * inv_a0,
                s[1] * inv_a0,
                s[2] * inv_a0,
                s[4] * inv_a0,
                s[5] * inv_a0,
            ]
        })
        .collect();

    for &raw in samples {
        let mut x = if raw.is_finite() { raw as f64 } else { 0.0 };
        for (sec, st) in coeffs.iter().zip(states.iter_mut()) {
            let y = sec[0] * x + st[0];
            st[0] = sec[1] * x - sec[3] * y + st[1];
            st[1] = sec[2] * x - sec[4] * y;
            x = y;
        }
        out.push(x);
    }
    out
}

/// Filters a single 2-stage K-weighted channel directly into instantaneous squared power
/// accumulated with channel weight `weight`, avoiding intermediate per-channel buffers.
pub(crate) fn accumulate_k_weighted_power(
    interleaved: &[f32],
    channels: usize,
    channel_idx: usize,
    weight: f64,
    sos: &[[f64; 6]; 2],
    sq_weighted: &mut [f64],
) {
    let [b0_0, b1_0, b2_0, a0_0, a1_0, a2_0] = sos[0];
    let inv0 = if a0_0 == 1.0 { 1.0 } else { 1.0 / a0_0 };
    let (b0_0, b1_0, b2_0, a1_0, a2_0) = (
        b0_0 * inv0,
        b1_0 * inv0,
        b2_0 * inv0,
        a1_0 * inv0,
        a2_0 * inv0,
    );

    let [b0_1, b1_1, b2_1, a0_1, a1_1, a2_1] = sos[1];
    let inv1 = if a0_1 == 1.0 { 1.0 } else { 1.0 / a0_1 };
    let (b0_1, b1_1, b2_1, a1_1, a2_1) = (
        b0_1 * inv1,
        b1_1 * inv1,
        b2_1 * inv1,
        a1_1 * inv1,
        a2_1 * inv1,
    );

    let mut z0_0 = 0.0_f64;
    let mut z1_0 = 0.0_f64;
    let mut z0_1 = 0.0_f64;
    let mut z1_1 = 0.0_f64;

    for (frame_idx, dst) in sq_weighted.iter_mut().enumerate() {
        let raw = interleaved[frame_idx * channels + channel_idx];
        let x0 = if raw.is_finite() { raw as f64 } else { 0.0 };

        let y0 = b0_0 * x0 + z0_0;
        z0_0 = b1_0 * x0 - a1_0 * y0 + z1_0;
        z1_0 = b2_0 * x0 - a2_0 * y0;

        let y1 = b0_1 * y0 + z0_1;
        z0_1 = b1_1 * y0 - a1_1 * y1 + z1_1;
        z1_1 = b2_1 * y0 - a2_1 * y1;

        *dst += weight * y1 * y1;
    }
}
