//! `hifi_dsp`: Clean-room spectral DSP, STFT/ISTFT, Mel-spectrogram, Akima pitch/time-axis
//! resampling, and Ht v3 LF glottal tension / Klatt resonator engine for `hifisampler-rs`.

use thiserror::Error;

pub mod hnsep_stft;
pub mod mel;
pub mod phrase;
pub mod pitch;
pub mod tension;

pub use hnsep_stft::{
    apply_bounded_mask, apply_bounded_mask_tensor, constant_pad_1d, hnsep_bounded_mask_multiply,
    istft_1d, periodic_hann_window, reflect_index, reflect_pad_1d, HnsepPrepared, HnsepStft,
    PadMode,
};
pub use mel::{
    dynamic_range_compression, dynamic_range_compression_default,
    dynamic_range_compression_inplace, slaney_mel_filterbank, MelConfig, MelSpectrogram,
    DEFAULT_DRC_CLIP_VAL,
};
pub use phrase::{
    build_spliced_phrase_mel_and_f0, layout_phrase_timeline, mel_note_unscaled_gain, PhraseLayout,
    PhraseNoteMelSlice, PhraseSpan, PhraseTimelineItem, SplicedPhraseExcitation, PHRASE_PAD_FRAMES,
};
pub use pitch::{
    midi_to_hz, resample_mel_and_pitch, Akima1DInterpolator, ResampleTimeParams, ResampledFeatures,
};
pub use tension::{
    apply_tension_and_coupling, apply_tension_and_coupling_with_config,
    couple_aperiodic_to_tension, couple_aperiodic_to_tension_with_config,
    pre_emphasis_base_tension, pre_emphasis_base_tension_with_config, TensionConfig,
};

/// Errors produced by `hifi_dsp` spectral, interpolation, and tension routines.
#[derive(Debug, Error)]
pub enum DspError {
    #[error("Invalid DSP parameter '{name}': {reason}")]
    InvalidParameter { name: &'static str, reason: String },

    #[error("Signal length ({actual}) is smaller than required minimum ({minimum})")]
    SignalTooShort { actual: usize, minimum: usize },

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("FFT execution error: {0}")]
    FftError(String),
}
