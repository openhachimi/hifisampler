use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{ArrayView2, ArrayView3};

use crate::device::{DevicePreference, ExecutionProvider, SessionConfig};
use crate::error::OnnxError;
use crate::runtime::OrtRuntime;
use crate::session::OrtSession;

/// Default relative path for the PC-NSF-HiFiGAN ONNX vocoder model.
pub const DEFAULT_VOCODER_ONNX_PATH: &str = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx";

/// Resolve a configured vocoder path to an `.onnx` file (no `.ckpt` PyTorch loading).
///
/// If a `.ckpt` path is supplied, this checks for a sibling `.onnx` file (e.g. `model.onnx`)
/// before falling back to [`DEFAULT_VOCODER_ONNX_PATH`].
pub fn resolve_vocoder_path(configured_path: impl AsRef<Path>) -> Result<PathBuf, OnnxError> {
    let raw = configured_path.as_ref();
    let mut checked = Vec::new();

    if raw
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("onnx"))
    {
        checked.push(raw.to_path_buf());
        if raw.is_file() {
            return Ok(raw.to_path_buf());
        }
    } else {
        // Pure-ONNX engine: map `.ckpt` or extensionless path to sibling `.onnx`.
        let sibling_onnx = raw.with_extension("onnx");
        checked.push(sibling_onnx.clone());
        if sibling_onnx.is_file() {
            return Ok(sibling_onnx);
        }
        if let Some(parent) = raw.parent() {
            let model_onnx = parent.join("model.onnx");
            checked.push(model_onnx.clone());
            if model_onnx.is_file() {
                return Ok(model_onnx);
            }
        }
    }

    let default_rel = PathBuf::from(DEFAULT_VOCODER_ONNX_PATH);
    checked.push(default_rel.clone());
    if default_rel.is_file() {
        log::warn!(
            "Configured vocoder path '{}' not found; using default '{}'",
            raw.display(),
            default_rel.display()
        );
        return Ok(default_rel);
    }

    // Also search ancestor directories (useful when running inside `rust/` or test binaries).
    for base in [
        std::env::current_dir().ok(),
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf)),
    ]
    .into_iter()
    .flatten()
    {
        let mut cur = Some(base.as_path());
        for _ in 0..=6 {
            let Some(dir) = cur else { break };
            let candidate = dir.join(DEFAULT_VOCODER_ONNX_PATH);
            if !checked.contains(&candidate) {
                checked.push(candidate.clone());
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
            cur = dir.parent();
        }
    }

    Err(OnnxError::ModelNotFound {
        path: raw.to_path_buf(),
        checked,
    })
}

/// ONNX Runtime inference session for the PC-NSF-HiFiGAN vocoder (`mel` + `f0` -> `waveform`).
pub struct VocoderSession {
    inner: OrtSession,
    mel_input_name: String,
    f0_input_name: String,
    output_name: String,
    n_mels: usize,
}

impl VocoderSession {
    /// Load a vocoder ONNX model with [`DevicePreference`].
    pub fn load(path: impl AsRef<Path>, device: DevicePreference) -> Result<Self, OnnxError> {
        Self::load_with_config(path, &SessionConfig::with_device(device))
    }

    /// Load a vocoder ONNX model with full [`SessionConfig`].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let resolved = resolve_vocoder_path(path)?;
        let inner = OrtSession::from_file(resolved, config)?;
        Self::from_session(inner)
    }

    /// Load a vocoder ONNX model using an explicit [`OrtRuntime`].
    pub fn load_with_runtime(
        runtime: Arc<OrtRuntime>,
        path: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let resolved = resolve_vocoder_path(path)?;
        let inner = OrtSession::from_file_with_runtime(runtime, resolved, config)?;
        Self::from_session(inner)
    }

    fn from_session(inner: OrtSession) -> Result<Self, OnnxError> {
        let inputs = inner.inputs();
        let outputs = inner.outputs();

        let mel_meta = inputs
            .iter()
            .find(|m| m.name == "mel")
            .or_else(|| inputs.first())
            .ok_or_else(|| OnnxError::InvalidModel {
                path: inner.model_path().to_path_buf(),
                message: "vocoder ONNX model has no inputs".to_string(),
            })?;

        let f0_meta = inputs
            .iter()
            .find(|m| m.name == "f0")
            .or_else(|| inputs.get(1))
            .ok_or_else(|| OnnxError::InvalidModel {
                path: inner.model_path().to_path_buf(),
                message: "vocoder ONNX model requires 2 inputs (mel, f0)".to_string(),
            })?;

        let out_meta = outputs
            .iter()
            .find(|m| m.name == "waveform")
            .or_else(|| outputs.first())
            .ok_or_else(|| OnnxError::InvalidModel {
                path: inner.model_path().to_path_buf(),
                message: "vocoder ONNX model has no outputs".to_string(),
            })?;

        let n_mels = mel_meta
            .dims
            .get(2)
            .copied()
            .filter(|&d| d > 0)
            .unwrap_or(128) as usize;

        Ok(Self {
            mel_input_name: mel_meta.name.clone(),
            f0_input_name: f0_meta.name.clone(),
            output_name: out_meta.name.clone(),
            n_mels,
            inner,
        })
    }

    /// Run vocoder inference on `mel` `[B, T, n_mels]` (`[1, T, 128]`) and `f0` `[B, T]` (`[1, T]`).
    pub fn synthesize(
        &self,
        mel: ArrayView3<'_, f32>,
        f0: ArrayView2<'_, f32>,
    ) -> Result<Vec<f32>, OnnxError> {
        let (batch, frames, mel_bins) = mel.dim();
        let (f0_batch, f0_frames) = f0.dim();

        if batch == 0 || frames == 0 {
            return Err(OnnxError::ShapeMismatch {
                expected: format!("[1, T>0, {}]", self.n_mels),
                actual: format!("[{batch}, {frames}, {mel_bins}]"),
            });
        }
        if mel_bins != self.n_mels {
            return Err(OnnxError::ShapeMismatch {
                expected: format!("[{batch}, {frames}, {}]", self.n_mels),
                actual: format!("[{batch}, {frames}, {mel_bins}]"),
            });
        }
        if f0_batch != batch || f0_frames != frames {
            return Err(OnnxError::ShapeMismatch {
                expected: format!("f0 shape [{batch}, {frames}]"),
                actual: format!("f0 shape [{f0_batch}, {f0_frames}]"),
            });
        }

        let mel_std = mel.as_standard_layout();
        let f0_std = f0.as_standard_layout();
        let mel_slice = mel_std.as_slice().ok_or_else(|| OnnxError::InvalidInput {
            message: "mel array could not be viewed as contiguous slice".to_string(),
        })?;
        let f0_slice = f0_std.as_slice().ok_or_else(|| OnnxError::InvalidInput {
            message: "f0 array could not be viewed as contiguous slice".to_string(),
        })?;

        let mel_dims = [batch as i64, frames as i64, mel_bins as i64];
        let f0_dims = [f0_batch as i64, f0_frames as i64];

        let mut outputs = self.inner.run_f32(
            &[
                (&self.mel_input_name, &mel_dims, mel_slice),
                (&self.f0_input_name, &f0_dims, f0_slice),
            ],
            &[&self.output_name],
        )?;

        let (_, waveform) = outputs.pop().ok_or_else(|| OnnxError::InvalidInput {
            message: "vocoder returned empty output vector".to_string(),
        })?;
        Ok(waveform)
    }

    /// Convenience wrapper taking `[T, 128]` Mel frames and `[T]` F0 slice.
    pub fn synthesize_frames(
        &self,
        mel_frames_by_bins: ArrayView2<'_, f32>,
        f0: &[f32],
    ) -> Result<Vec<f32>, OnnxError> {
        let mel_3d = mel_frames_by_bins.insert_axis(ndarray::Axis(0));
        let f0_2d =
            ArrayView2::from_shape((1, f0.len()), f0).map_err(|e| OnnxError::InvalidInput {
                message: format!("failed to reshape f0 slice: {e}"),
            })?;
        self.synthesize(mel_3d, f0_2d)
    }

    /// Convenience wrapper taking `[128, T]` (`[n_mels, n_frames]`, matching `hifi_dsp`) and `[T]` F0.
    pub fn synthesize_mels_by_frames(
        &self,
        mel_bins_by_frames: ArrayView2<'_, f32>,
        f0: &[f32],
    ) -> Result<Vec<f32>, OnnxError> {
        self.synthesize_frames(mel_bins_by_frames.t(), f0)
    }

    /// Run a short dummy forward pass to warm up execution provider kernels (matching `backend/models.py`).
    pub fn warmup(&self) -> Result<(), OnnxError> {
        let dummy_mel = ndarray::Array3::<f32>::zeros((1, 20, self.n_mels));
        let dummy_f0 = ndarray::Array2::<f32>::from_elem((1, 20), 440.0);
        let _ = self.synthesize(dummy_mel.view(), dummy_f0.view())?;
        Ok(())
    }

    pub fn execution_provider(&self) -> &ExecutionProvider {
        self.inner.execution_provider()
    }

    pub fn is_gpu_ep(&self) -> bool {
        self.inner.is_gpu_ep()
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.inner.fallback_reason()
    }

    pub fn model_path(&self) -> &Path {
        self.inner.model_path()
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }
}
