//! `hifi_core`: Core audio I/O, HIFIBIN1 feature cache, and UTAU argument/pitch/flag parsing
//! for `hifisampler-rs`.

pub mod audio;
pub mod cache;
pub mod utau;
pub mod wavtool;

pub use audio::{
    is_null_device, read_wav, read_wav_raw, read_wav_with_target_rate, resolve_audio_path,
    write_wav, AudioBuffer, AudioError, DEFAULT_SAMPLE_RATE,
};
pub use cache::{
    cleanup_orphan_tmp_files, cleanup_stale_tmp_files, load_cache, load_cache_with_options,
    rename_with_retry, resolve_feature_cache_paths, save_cache, save_cache_with_options, stat_sig,
    CacheError, CacheManager, CachedFeatures, HifiBinHeader, StatSig, HEADER_SIZE, HIFIBIN_MAGIC,
    HIFIBIN_VERSION, MAX_FEATURES_RAM_ITEMS, MAX_WAV_RAM_ITEMS,
};
pub use utau::{
    find_flags, midi_slice_to_hz, midi_to_hz, note_to_midi, parse_flags, parse_flags_map,
    parse_tempo, pitch_string_to_cents, to_int12, to_int12_stream, to_uint6, UtauError, UtauFlags,
    ALL_FLAGS, MAX_PITCH_RLE_COUNT,
};
pub use wavtool::{
    parse_wavtool_cli_args, parse_wavtool_duration_ms, WavtoolCliArgs, WavtoolEnvelope,
};
