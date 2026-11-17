use crate::phrase_mgr::{PhraseBatchRequest, WavtoolStepRequest};
use crate::stats::StatsCollector;
use hifi_core::utau::{note_to_midi, parse_flags, parse_tempo, UtauFlags};
use ndarray::{Array4, ArrayView2};
use std::path::{Path, PathBuf};

/// Errors produced during note synthesis.
#[derive(Debug)]
pub enum ResampleError {
    InputNotFound(String),
    BadInput(String),
    Internal(String),
}

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResampleError::InputNotFound(msg) => write!(f, "{msg}"),
            ResampleError::BadInput(msg) => write!(f, "{msg}"),
            ResampleError::Internal(msg) => write!(f, "{msg}"),
        }
    }
}

/// Build the exact `.hifi.bin` feature cache path matching `backend/resampler.py:99-122`.
pub fn build_feature_cache_path(in_file: &Path, flags: &UtauFlags) -> PathBuf {
    let stem = in_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    in_file.with_file_name(flags.feature_cache_filename(stem))
}

/// Unpacked synthesis arguments passed into `Resampler`.
#[derive(Debug, Clone)]
pub struct ResampleJob {
    pub in_file: PathBuf,
    pub out_file: String,
    pub pitch_midi: f64,
    pub velocity: f64,
    pub flags: UtauFlags,
    pub offset_ms: f64,
    pub length_ms: usize,
    pub consonant_ms: f64,
    pub cutoff_ms: f64,
    pub volume: f64,
    pub modulation: f64,
    pub tempo: f64,
    pub pitch_string: String,
}

impl ResampleJob {
    pub fn from_raw_args(args: &[String]) -> Result<Self, ResampleError> {
        if args.len() < 4 {
            return Err(ResampleError::BadInput(format!(
                "Expected at least 4 UTAU arguments, got {}",
                args.len()
            )));
        }

        fn arg_or<'a>(args: &'a [String], i: usize, def: &'a str) -> &'a str {
            args.get(i).map(String::as_str).unwrap_or(def)
        }

        let pitch_midi =
            note_to_midi(&args[2]).map_err(|e| ResampleError::BadInput(e.to_string()))? as f64;
        let velocity: f64 = args[3]
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput(format!("Invalid velocity: {}", args[3])))?;
        let flags = parse_flags(arg_or(args, 4, ""));
        let offset_ms: f64 = arg_or(args, 5, "0")
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput("Invalid offset".into()))?;
        let length_raw: f64 = arg_or(args, 6, "1000")
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput("Invalid length".into()))?;
        let length_ms = (length_raw.round() as isize).max(1) as usize;
        let consonant_ms: f64 = arg_or(args, 7, "0")
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput("Invalid consonant".into()))?;
        let cutoff_ms: f64 = arg_or(args, 8, "0")
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput("Invalid cutoff".into()))?;
        let volume: f64 = arg_or(args, 9, "100")
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput("Invalid volume".into()))?;
        let modulation: f64 = arg_or(args, 10, "0")
            .trim()
            .parse()
            .map_err(|_| ResampleError::BadInput("Invalid modulation".into()))?;
        let tempo = parse_tempo(arg_or(args, 11, "!100"))
            .map_err(|e| ResampleError::BadInput(e.to_string()))?;
        let pitch_string = arg_or(args, 12, "AA").to_string();

        Ok(Self {
            in_file: PathBuf::from(&args[0]),
            out_file: args[1].clone(),
            pitch_midi,
            velocity,
            flags,
            offset_ms,
            length_ms,
            consonant_ms,
            cutoff_ms,
            volume,
            modulation,
            tempo,
            pitch_string,
        })
    }
}

/// Abstract ONNX inference sessions for HN-SEP and PC-NSF-HiFiGAN vocoder.
///
/// Implementers MUST guarantee internal thread-safety and GPU serialization (for example via
/// `OrtRuntime::lock_gpu()`), allowing multiple HTTP workers to invoke DSP and ONNX overlap
/// concurrently without an outer mutex in `RealResampleEngine`.
pub trait OnnxRunner: Send + Sync {
    fn hnsep_uses_gpu(&self) -> bool;
    fn vocoder_uses_gpu(&self) -> bool;
    fn run_hnsep_mask(&self, stft_tensor: &Array4<f32>) -> Result<Array4<f32>, ResampleError>;
    fn run_vocoder(
        &self,
        mel_n_mels_by_t: ArrayView2<f32>,
        f0: &[f32],
    ) -> Result<Vec<f32>, ResampleError>;
}

/// Thread-safe server resampler engine matching `backend/resampler.py`.
pub trait ResampleEngine: Send + Sync {
    fn resample(&self, raw_args: &[String], stats: &StatsCollector) -> Result<(), ResampleError>;

    /// Prepares acoustic features (Mel + F0 + fry + P-gain) without running the GPU vocoder,
    /// persisting a `MELOSNT1` sidecar and writing a minimal valid WAV stub for `melos`.
    fn prepare_phrase_note(
        &self,
        _raw_args: &[String],
        _stats: &StatsCollector,
    ) -> Result<(), ResampleError> {
        Err(ResampleError::Internal(
            "Phrase preparation is not supported by this engine".into(),
        ))
    }

    /// Accumulates or executes continuous phrase synthesis for a `melos-wavtool` step.
    fn process_wavtool_step(
        &self,
        _req: WavtoolStepRequest,
        _stats: &StatsCollector,
    ) -> Result<(), ResampleError> {
        Err(ResampleError::Internal(
            "Phrase wavtool synthesis is not supported by this engine".into(),
        ))
    }

    /// One-shot batch continuous phrase synthesis endpoint.
    fn render_phrase_batch(
        &self,
        _req: PhraseBatchRequest,
        _stats: &StatsCollector,
    ) -> Result<(), ResampleError> {
        Err(ResampleError::Internal(
            "Phrase batch rendering is not supported by this engine".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_filename_matches_python_resampler_exactly() {
        let wav = Path::new("/tmp/sample.wav");

        let f_default = parse_flags("");
        assert_eq!(
            build_feature_cache_path(wav, &f_default)
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "sample_Hb100_Ht0_Hv100_g0.hifi.bin"
        );

        let f_ht = parse_flags("Ht50Ho40Hr-30g-10");
        assert_eq!(
            build_feature_cache_path(wav, &f_ht)
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "sample_Hb100_Ho40_Hr-30_Ht50v3_Hv100_g-10.hifi.bin"
        );
    }

    #[test]
    fn job_unpacking_defaults_and_clamping() {
        let job = ResampleJob::from_raw_args(&[
            "in.wav".into(),
            "out.wav".into(),
            "A4".into(),
            "100".into(),
            "G0He0A".into(),
            "0".into(),
            "240.6".into(),
            "0".into(),
            "0".into(),
            "100".into(),
            "0".into(),
            "!0".into(),
            "AA".into(),
        ])
        .unwrap();
        assert_eq!(job.pitch_midi, 69.0);
        assert_eq!(job.length_ms, 241);
        assert_eq!(job.tempo, 1.0);
        assert!(!job.flags.force_generate());
        assert!(!job.flags.he_loop());
    }
}
