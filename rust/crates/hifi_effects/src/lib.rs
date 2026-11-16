//! `hifi_effects`: Clean-room vocal DSP effects for `hifisampler-rs`.
//!
//! Includes:
//! - [`vocal_fry`]: Leap-safe relative-pit LF glottal vocal fry ("气泡音") excitation and 3-band shaper.
//! - [`growl`]: `HG`-flag 4th-order Butterworth highpass split + 80 Hz square LFO pitch modulation.
//! - [`modulation`]: `A`-flag pitch-slope amplitude modulation, scale restoration, and volume/peak limiting.
//! - [`sos`]: Butterworth SOS design, Direct-Form II Transposed `sosfilt`, and zero-phase `sosfiltfilt`.
//! - [`akima`]: Exact 1-D Akima cubic Hermite interpolator.

pub mod akima;
pub mod growl;
pub mod modulation;
pub mod sos;
pub mod vocal_fry;

pub use akima::Akima1DInterpolator;
pub use growl::{
    apply_hg_growl, growl, growl_with_params, DEFAULT_GROWL_CUTOFF_HZ, DEFAULT_GROWL_FREQ_HZ,
};
pub use modulation::{
    apply_amplitude_modulation, apply_scale_inverse, apply_volume_and_peak_limit,
    gradient_1d_uniform, interp_1d_clamped, peak_abs,
};
pub use sos::{
    butter_sos, cached_butter_sos, sosfilt, sosfilt_zi, sosfiltfilt, ButterBand, SosSection,
};
pub use vocal_fry::{
    apply_lf_vocal_fry, apply_lf_vocal_fry_in_place, midi_to_hz, prepare_vocal_fry_excitation,
    slaney_mel_filterbank, FryExcitationResult, VocalFryInput, BASIN_MORPH_FULL_CENTS,
    BASIN_MORPH_START_CENTS, FRY_FULL_CENTS, FRY_ONSET_CENTS, MIN_DWELL_BOUNDARY_SEC,
    MIN_DWELL_INTERIOR_SEC, SLEW_PASS_CPS, SLEW_STOP_CPS,
};
