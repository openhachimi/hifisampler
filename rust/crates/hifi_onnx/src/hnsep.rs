use std::path::{Path, PathBuf};
use std::sync::Arc;

use ndarray::{s, Array3, Array4, ArrayView3, ArrayView4};

use crate::device::{DevicePreference, ExecutionProvider, SessionConfig};
use crate::error::OnnxError;
use crate::runtime::OrtRuntime;
use crate::session::OrtSession;

/// Default relative path candidates for the HN-SEP ONNX model.
pub const DEFAULT_HNSEP_ONNX_PATHS: &[&str] = &["hnsep/vr/model.onnx", "hnsep/vr/model_fp16.onnx"];

/// Metadata parsed from sibling `config.yaml` and ONNX input tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnsepConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub expected_freq_bins: Option<usize>,
}

impl Default for HnsepConfig {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop_length: 512,
            expected_freq_bins: Some(1025),
        }
    }
}

/// Resolve an HN-SEP path (supporting `.pt` -> sibling `.onnx` resolution matching Python config).
pub fn resolve_hnsep_path(configured_path: impl AsRef<Path>) -> Result<PathBuf, OnnxError> {
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
        // Resolve `.pt` / `.pth` / `.ckpt` config path to sibling `.onnx` (e.g. `model.pt` -> `model.onnx`).
        let sibling_onnx = raw.with_extension("onnx");
        checked.push(sibling_onnx.clone());
        if sibling_onnx.is_file() {
            return Ok(sibling_onnx);
        }
    }

    if let Some(parent) = raw.parent() {
        for fallback_name in ["model.onnx", "model_fp16.onnx"] {
            let candidate = parent.join(fallback_name);
            if !checked.contains(&candidate) {
                checked.push(candidate.clone());
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
    }

    for rel in DEFAULT_HNSEP_ONNX_PATHS {
        let p = PathBuf::from(rel);
        if !checked.contains(&p) {
            checked.push(p.clone());
            if p.is_file() {
                log::warn!(
                    "Configured HN-SEP path '{}' not found; using default '{}'",
                    raw.display(),
                    p.display()
                );
                return Ok(p);
            }
        }
    }

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
            for rel in DEFAULT_HNSEP_ONNX_PATHS {
                let candidate = dir.join(rel);
                if !checked.contains(&candidate) {
                    checked.push(candidate.clone());
                    if candidate.is_file() {
                        return Ok(candidate);
                    }
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

/// ONNX Runtime inference session for the HN-SEP harmonic-noise separator.
///
/// Accepts `[B, 2, F, T]` real/imag stacked STFT tensors (`F == 1025` native for `hnsep/vr/model.onnx`,
/// or `F == 1024` which is automatically zero-padded to 1025 and sliced back to 1024).
pub struct HnsepSession {
    inner: OrtSession,
    config: HnsepConfig,
    input_name: String,
    output_name: String,
}

impl HnsepSession {
    /// Load an HN-SEP ONNX model (resolving `.pt` config paths to sibling `.onnx` if present).
    pub fn load(path: impl AsRef<Path>, device: DevicePreference) -> Result<Self, OnnxError> {
        Self::load_with_config(path, &SessionConfig::with_device(device))
    }

    /// Load an HN-SEP ONNX model with full [`SessionConfig`].
    pub fn load_with_config(
        path: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let resolved = resolve_hnsep_path(path)?;
        let inner = OrtSession::from_file(resolved, config)?;
        Self::from_session(inner)
    }

    /// Load an HN-SEP ONNX model using an explicit [`OrtRuntime`].
    pub fn load_with_runtime(
        runtime: Arc<OrtRuntime>,
        path: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let resolved = resolve_hnsep_path(path)?;
        let inner = OrtSession::from_file_with_runtime(runtime, resolved, config)?;
        Self::from_session(inner)
    }

    fn from_session(inner: OrtSession) -> Result<Self, OnnxError> {
        let inputs = inner.inputs();
        let outputs = inner.outputs();

        let in_meta = inputs
            .iter()
            .find(|m| m.name == "input")
            .or_else(|| inputs.first())
            .ok_or_else(|| OnnxError::InvalidModel {
                path: inner.model_path().to_path_buf(),
                message: "HN-SEP ONNX model has no inputs".to_string(),
            })?;

        let out_meta = outputs
            .iter()
            .find(|m| m.name == "output")
            .or_else(|| outputs.first())
            .ok_or_else(|| OnnxError::InvalidModel {
                path: inner.model_path().to_path_buf(),
                message: "HN-SEP ONNX model has no outputs".to_string(),
            })?;

        let expected_freq_bins = in_meta
            .dims
            .get(2)
            .copied()
            .filter(|&d| d > 0)
            .map(|d| d as usize);

        let mut cfg = read_sibling_config_yaml(inner.model_path());
        cfg.expected_freq_bins = expected_freq_bins;

        Ok(Self {
            input_name: in_meta.name.clone(),
            output_name: out_meta.name.clone(),
            config: cfg,
            inner,
        })
    }

    /// Run HN-SEP bounded complex mask inference on `[B, 2, F, T]` (`channel 0 = real`, `channel 1 = imag`).
    ///
    /// Supports both `[B, 2, 1025, T]` (exact `n_fft/2 + 1`) and `[B, 2, 1024, T]` transparently,
    /// returning an output mask tensor matching the caller's input `[B, 2, F, T]` dimensions.
    pub fn forward(&self, spec: ArrayView4<'_, f32>) -> Result<Array4<f32>, OnnxError> {
        let (batch, channels, freq_bins, time_frames) = spec.dim();
        if batch == 0 || channels != 2 || time_frames == 0 {
            return Err(OnnxError::ShapeMismatch {
                expected: "[B>=1, 2, F, T>0]".to_string(),
                actual: format!("[{batch}, {channels}, {freq_bins}, {time_frames}]"),
            });
        }

        let target_freq = self.config.expected_freq_bins.unwrap_or(freq_bins);
        if freq_bins != target_freq
            && !((freq_bins == 1024 && target_freq == 1025)
                || (freq_bins == 1025 && target_freq == 1024))
        {
            return Err(OnnxError::ShapeMismatch {
                expected: format!("[{batch}, 2, {target_freq}, {time_frames}]"),
                actual: format!("[{batch}, 2, {freq_bins}, {time_frames}]"),
            });
        }

        let owned_input: Array4<f32>;
        let input_view: ArrayView4<'_, f32> = if freq_bins == target_freq {
            spec
        } else if freq_bins == 1024 && target_freq == 1025 {
            let mut padded = Array4::<f32>::zeros((batch, 2, 1025, time_frames));
            padded.slice_mut(s![.., .., ..1024, ..]).assign(&spec);
            owned_input = padded;
            owned_input.view()
        } else {
            // freq_bins == 1025 && target_freq == 1024
            owned_input = spec.slice(s![.., .., ..1024, ..]).to_owned();
            owned_input.view()
        };

        let std_input = input_view.as_standard_layout();
        let input_slice = std_input
            .as_slice()
            .ok_or_else(|| OnnxError::InvalidInput {
                message: "HN-SEP input array could not be viewed as contiguous slice".to_string(),
            })?;
        let in_dims = [batch as i64, 2i64, target_freq as i64, time_frames as i64];

        let mut outputs = self.inner.run_f32(
            &[(&self.input_name, &in_dims, input_slice)],
            &[&self.output_name],
        )?;
        let (out_dims, out_vec) = outputs.pop().ok_or_else(|| OnnxError::InvalidInput {
            message: "HN-SEP returned empty output vector".to_string(),
        })?;

        if out_dims.len() != 4 {
            return Err(OnnxError::ShapeMismatch {
                expected: "4D tensor [B, 2, F, T]".to_string(),
                actual: format!("{out_dims:?}"),
            });
        }
        let ob = out_dims[0] as usize;
        let oc = out_dims[1] as usize;
        let of = out_dims[2] as usize;
        let ot = out_dims[3] as usize;

        let raw_arr = Array4::from_shape_vec((ob, oc, of, ot), out_vec).map_err(|e| {
            OnnxError::InvalidInput {
                message: format!("failed to build HN-SEP output array: {e}"),
            }
        })?;

        if of == freq_bins {
            Ok(raw_arr)
        } else if of == 1025 && freq_bins == 1024 {
            Ok(raw_arr.slice(s![.., .., ..1024, ..]).to_owned())
        } else if of == 1024 && freq_bins == 1025 {
            let mut padded = Array4::<f32>::zeros((ob, oc, 1025, ot));
            padded.slice_mut(s![.., .., ..1024, ..]).assign(&raw_arr);
            let last_bin = raw_arr.slice(s![.., .., 1023..1024, ..]);
            padded
                .slice_mut(s![.., .., 1024..1025, ..])
                .assign(&last_bin);
            Ok(padded)
        } else {
            Err(OnnxError::ShapeMismatch {
                expected: format!("[{batch}, 2, {freq_bins}, {time_frames}]"),
                actual: format!("[{ob}, {oc}, {of}, {ot}]"),
            })
        }
    }

    /// Convenience wrapper accepting separate `[B, F, T]` real & imaginary tensors.
    pub fn forward_complex(
        &self,
        real: ArrayView3<'_, f32>,
        imag: ArrayView3<'_, f32>,
    ) -> Result<(Array3<f32>, Array3<f32>), OnnxError> {
        if real.dim() != imag.dim() {
            return Err(OnnxError::ShapeMismatch {
                expected: format!("imag shape {:?}", real.dim()),
                actual: format!("imag shape {:?}", imag.dim()),
            });
        }
        let (b, f, t) = real.dim();
        let mut stacked = Array4::<f32>::zeros((b, 2, f, t));
        stacked.slice_mut(s![.., 0, .., ..]).assign(&real);
        stacked.slice_mut(s![.., 1, .., ..]).assign(&imag);
        let mask = self.forward(stacked.view())?;
        let mask_real = mask.slice(s![.., 0, .., ..]).to_owned();
        let mask_imag = mask.slice(s![.., 1, .., ..]).to_owned();
        Ok((mask_real, mask_imag))
    }

    /// Run a short dummy forward pass to warm up execution provider kernels.
    pub fn warmup(&self) -> Result<(), OnnxError> {
        let f = self.config.expected_freq_bins.unwrap_or(1025);
        let dummy = Array4::<f32>::zeros((1, 2, f, 32));
        let _ = self.forward(dummy.view())?;
        Ok(())
    }

    pub fn config(&self) -> &HnsepConfig {
        &self.config
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
}

fn read_sibling_config_yaml(model_path: &Path) -> HnsepConfig {
    let mut cfg = HnsepConfig::default();
    let Some(parent) = model_path.parent() else {
        return cfg;
    };
    let yaml_path = parent.join("config.yaml");
    let Ok(contents) = std::fs::read_to_string(&yaml_path) else {
        return cfg;
    };
    for line in contents.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((key, val)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let val = val.trim().trim_matches('"').trim_matches('\'');
        match key {
            "n_fft" => {
                if let Ok(n) = val.parse::<usize>() {
                    if n > 0 {
                        cfg.n_fft = n;
                    }
                }
            }
            "hop_length" => {
                if let Ok(h) = val.parse::<usize>() {
                    if h > 0 {
                        cfg.hop_length = h;
                    }
                }
            }
            _ => {}
        }
    }
    cfg
}
