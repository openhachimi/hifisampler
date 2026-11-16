//! `hifi_lufs`: Clean-room ITU-R BS.1770-4 LUFS loudness measurement, envelope-aware
//! loudness normalization, and rational polyphase FIR resampling for `hifisampler-rs`.

pub mod biquad;
pub mod loudness;
pub mod resample;

use thiserror::Error;

pub use biquad::{apply_sos_filter, k_weighting_sos, K_WEIGHT_SOS_44100};
pub use loudness::{
    apply_envelope_aware_gain, apply_peak_limit, loudness_norm, loudness_norm_fast, measure_lufs,
    measure_lufs_multichannel, measure_lufs_with_block_size, peak_normalize, EnvelopeGainOptions,
    LoudnessNormOptions, TailMode, ABS_GATE_LUFS, DEFAULT_BLOCK_SIZE_SEC,
    DEFAULT_SILENCE_THRESHOLD_DB, REL_GATE_LU,
};
pub use resample::resample_poly_fast;

/// Errors returned by `hifi_lufs` measurement, normalization, and resampling functions.
#[derive(Debug, Error)]
pub enum LufsError {
    #[error("sample_rate must be positive, got {0}")]
    InvalidSampleRate(u32),

    #[error("sample_rate={sample_rate} Hz is too low for K-weighting high-shelf (requires > {min_hz:.1} Hz)")]
    SampleRateTooLowForKWeighting { sample_rate: u32, min_hz: f64 },

    #[error("block_size must be positive and finite, got {0}")]
    InvalidBlockSize(f64),

    #[error("channels must be >= 1, got {0}")]
    InvalidChannelCount(usize),

    #[error("interleaved buffer length {len} is not divisible by {channels} channels")]
    ShapeMismatch { len: usize, channels: usize },

    #[error("unsupported tail_mode '{0}'; expected 'preserve_relative' or 'clip'")]
    UnsupportedTailMode(String),
}
