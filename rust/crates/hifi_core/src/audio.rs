//! Audio I/O implementation for reading and writing WAV files with bit-exact
//! parity to soundfile/libsndfile.

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

/// Default audio sample rate required by UTAU and hifisampler (44.1 kHz).
pub const DEFAULT_SAMPLE_RATE: u32 = 44100;

/// In-memory audio buffer containing normalized float32 samples.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    /// Normalized audio samples in range `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hertz (e.g. 44100).
    pub sample_rate: u32,
    /// Number of audio channels (typically 1 for mono).
    pub channels: u16,
}

impl AudioBuffer {
    /// Creates a new `AudioBuffer`.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Returns audio duration in seconds.
    pub fn duration_seconds(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            0.0
        } else {
            (self.samples.len() / self.channels as usize) as f32 / self.sample_rate as f32
        }
    }
}

/// Errors occurring during audio file operations.
#[derive(Debug, Error)]
pub enum AudioError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hound WAV error: {0}")]
    Hound(#[from] hound::Error),

    #[error("Audio file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Unsupported audio format: {format:?} with {bits} bits per sample")]
    UnsupportedFormat { format: SampleFormat, bits: u16 },

    #[error("Invalid channel count: {0}")]
    InvalidChannelCount(u16),

    #[error("Invalid sample rate: {0}")]
    InvalidSampleRate(u32),

    #[error("Resampling error: {0}")]
    Resample(#[from] hifi_lufs::LufsError),

    #[error("Atomic rename failed for '{tmp}' -> '{dest}': {source}")]
    AtomicRenameFailed {
        tmp: PathBuf,
        dest: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Helper to determine if a destination path points to a platform null device.
pub fn is_null_device(path: &Path) -> bool {
    let s = path.to_string_lossy().trim().to_lowercase();
    s == "nul" || s == "/dev/null"
}

/// Resolves an audio file path, supporting case-insensitive file matching and
/// extension fallbacks (e.g. `.wav`, `.WAV`).
pub fn resolve_audio_path(path: &Path) -> Result<PathBuf, AudioError> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    // Try uppercase and lowercase extension variants
    for ext in &["wav", "WAV", "Wav"] {
        let candidate = path.with_extension(ext);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    // Search parent directory for case-insensitive stem and extension match
    if let Some(parent) = path.parent() {
        if parent.exists() {
            if let Some(target_stem) = path.file_stem().and_then(|s| s.to_str()) {
                let target_stem_lower = target_stem.to_lowercase();
                if let Ok(entries) = fs::read_dir(parent) {
                    for entry in entries.flatten() {
                        let candidate_path = entry.path();
                        if let Some(cand_stem) = candidate_path.file_stem().and_then(|s| s.to_str())
                        {
                            if cand_stem.to_lowercase() == target_stem_lower {
                                if let Some(cand_ext) =
                                    candidate_path.extension().and_then(|e| e.to_str())
                                {
                                    if cand_ext.eq_ignore_ascii_case("wav") {
                                        return Ok(candidate_path);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Err(AudioError::FileNotFound(path.to_path_buf()))
}

/// Reads a WAV audio file into an `AudioBuffer`, normalizing samples to `[-1.0, 1.0]`,
/// downmixing multi-channel audio to mono via arithmetic mean, and resampling to
/// `DEFAULT_SAMPLE_RATE` (44,100 Hz) via rational polyphase FIR filtering when needed.
///
/// Mirrors `util/audio.py::read_wav`.
pub fn read_wav(path: &Path) -> Result<AudioBuffer, AudioError> {
    read_wav_with_target_rate(path, Some(DEFAULT_SAMPLE_RATE))
}

/// Reads a WAV audio file without resampling, preserving the file's native sample rate.
pub fn read_wav_raw(path: &Path) -> Result<AudioBuffer, AudioError> {
    read_wav_with_target_rate(path, None)
}

/// Reads a WAV audio file into an `AudioBuffer`, optionally resampling to `target_sample_rate`.
pub fn read_wav_with_target_rate(
    path: &Path,
    target_sample_rate: Option<u32>,
) -> Result<AudioBuffer, AudioError> {
    let resolved = resolve_audio_path(path)?;
    let mut reader = WavReader::open(&resolved)?;
    let spec = reader.spec();

    if spec.channels == 0 {
        return Err(AudioError::InvalidChannelCount(0));
    }
    if spec.sample_rate == 0 {
        return Err(AudioError::InvalidSampleRate(0));
    }

    let channels = spec.channels as usize;
    let raw_samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<Vec<f32>, _>>()?,
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<Result<Vec<f32>, _>>()?,
        (SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8388608.0))
            .collect::<Result<Vec<f32>, _>>()?,
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.map(|v| (v as f64 / 2147483648.0) as f32))
            .collect::<Result<Vec<f32>, _>>()?,
        (SampleFormat::Int, 8) => reader
            .samples::<i8>()
            .map(|s| s.map(|v| v as f32 / 128.0))
            .collect::<Result<Vec<f32>, _>>()?,
        (format, bits) => {
            return Err(AudioError::UnsupportedFormat { format, bits });
        }
    };

    // Arithmetic mean downmixing across channels
    let mono_samples = if channels == 1 {
        raw_samples
    } else {
        let frame_count = raw_samples.len() / channels;
        let mut mono = Vec::with_capacity(frame_count);
        for chunk in raw_samples.chunks_exact(channels) {
            let sum: f32 = chunk.iter().sum();
            mono.push(sum / channels as f32);
        }
        mono
    };

    if let Some(target_sr) = target_sample_rate {
        if target_sr == 0 {
            return Err(AudioError::InvalidSampleRate(0));
        }
        if spec.sample_rate != target_sr {
            let resampled =
                hifi_lufs::resample_poly_fast(&mono_samples, spec.sample_rate, target_sr)?;
            return Ok(AudioBuffer {
                samples: resampled,
                sample_rate: target_sr,
                channels: 1,
            });
        }
    }

    Ok(AudioBuffer {
        samples: mono_samples,
        sample_rate: spec.sample_rate,
        channels: 1,
    })
}

/// Atomically writes an `AudioBuffer` as a 16-bit PCM WAV file.
/// If `path` is a null device ("nul" or "/dev/null"), the operation succeeds as a no-op.
pub fn write_wav(path: &Path, audio: &AudioBuffer) -> Result<(), AudioError> {
    if is_null_device(path) {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file_stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio");
    let tmp_file_name = format!(".{}.{}.tmp", file_stem, Uuid::new_v4().simple());
    let tmp_path = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(&tmp_file_name),
        _ => PathBuf::from(&tmp_file_name),
    };

    // RAII temporary file cleanup on unexpected errors
    struct TmpCleanup<'a>(&'a Path);
    impl<'a> Drop for TmpCleanup<'a> {
        fn drop(&mut self) {
            if self.0.exists() {
                let _ = fs::remove_file(self.0);
            }
        }
    }
    let cleanup = TmpCleanup(&tmp_path);

    let spec = WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    {
        let file = File::create(&tmp_path)?;
        let buf_writer = BufWriter::new(file);
        let mut writer = WavWriter::new(buf_writer, spec)?;

        for &sample in &audio.samples {
            let sample_f64 = if sample.is_nan() { 0.0 } else { sample as f64 };
            // Exact libsndfile 1.2.2 PCM_16 quantization formula:
            // floor(sample * 32768.0).clamp(-32768.0, 32767.0)
            let scaled = sample_f64 * 32768.0;
            let floored = scaled.floor();
            let clamped = floored.clamp(-32768.0, 32767.0);
            writer.write_sample(clamped as i16)?;
        }

        writer.finalize()?;
    }

    fs::rename(&tmp_path, path).map_err(|e| AudioError::AtomicRenameFailed {
        tmp: tmp_path.clone(),
        dest: path.to_path_buf(),
        source: e,
    })?;

    // Disarm cleanup guard on successful rename
    std::mem::forget(cleanup);

    Ok(())
}
