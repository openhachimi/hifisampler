use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

/// User/configuration preference for the ONNX Runtime execution provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Rocm,
    DirectMl,
    CoreMl,
}

impl DevicePreference {
    /// Parse a device preference string (e.g. from `config.yaml` or CLI flags).
    pub fn from_config_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "rocm" | "hip" => Self::Rocm,
            "dml" | "directml" => Self::DirectMl,
            "coreml" | "mps" => Self::CoreMl,
            _ => Self::Auto,
        }
    }
}

impl FromStr for DevicePreference {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::from_config_str(s))
    }
}

impl fmt::Display for DevicePreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Rocm => "rocm",
            Self::DirectMl => "directml",
            Self::CoreMl => "coreml",
        };
        f.write_str(name)
    }
}

/// Active ONNX Runtime execution provider bound to a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    DirectMl,
    Cuda,
    Rocm,
    CoreMl,
    MigraphX,
    Other(String),
}

impl ExecutionProvider {
    /// Returns the canonical ONNX Runtime provider name string.
    pub fn provider_name(&self) -> &str {
        match self {
            Self::Cpu => "CPUExecutionProvider",
            Self::DirectMl => "DmlExecutionProvider",
            Self::Cuda => "CUDAExecutionProvider",
            Self::Rocm => "ROCMExecutionProvider",
            Self::CoreMl => "CoreMLExecutionProvider",
            Self::MigraphX => "MIGraphXExecutionProvider",
            Self::Other(name) => name.as_str(),
        }
    }

    /// Map an ONNX Runtime provider string into an [`ExecutionProvider`].
    pub fn from_ort_name(name: &str) -> Self {
        match name {
            "CPUExecutionProvider" => Self::Cpu,
            "DmlExecutionProvider" => Self::DirectMl,
            "CUDAExecutionProvider" => Self::Cuda,
            "ROCMExecutionProvider" => Self::Rocm,
            "CoreMLExecutionProvider" => Self::CoreMl,
            "MIGraphXExecutionProvider" => Self::MigraphX,
            other => Self::Other(other.to_string()),
        }
    }

    /// Returns `true` when the active provider executes on GPU/accelerator hardware.
    ///
    /// GPU inference passes serialize internally via [`crate::OrtRuntime::lock_gpu()`] inside
    /// [`crate::OrtSession::run_f32`] / [`crate::OrtSession::run_f32_timed`], satisfying the
    /// thread-safe `OnnxRunner` contract without external caller locking.
    pub fn is_gpu_ep(&self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

impl fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.provider_name())
    }
}

/// Configuration for creating an ONNX Runtime session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Preferred device / execution provider (`Auto` by default).
    pub device: DevicePreference,
    /// Number of intra-op threads; `None` selects `max(1, physical_cpus / workers_hint)`.
    pub intra_op_threads: Option<usize>,
    /// Number of inter-op threads (`1` by default, matching `util/model_loader.py`).
    pub inter_op_threads: usize,
    /// Expected HTTP/DSP worker concurrency hint used to divide intra-op threads (`2` by default).
    pub workers_hint: usize,
    /// GPU device index (`0` by default).
    pub device_id: i32,
    /// Disable DirectML metacommands to prevent dynamic-shape shader recompilation stalls (`true` by default).
    pub disable_dml_metacommands: bool,
    /// Override memory pattern setting (`None` enables it on CPU/CUDA/ROCm and disables it on DirectML).
    pub enable_mem_pattern: Option<bool>,
    /// Enable CPU memory arena (`false` by default to prevent unbounded arena growth).
    pub enable_cpu_mem_arena: bool,
    /// Explicit ONNX Runtime shared library path override.
    pub ort_library_path: Option<PathBuf>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            device: DevicePreference::Auto,
            intra_op_threads: None,
            inter_op_threads: 1,
            workers_hint: 2,
            device_id: 0,
            disable_dml_metacommands: true,
            enable_mem_pattern: None,
            enable_cpu_mem_arena: false,
            ort_library_path: None,
        }
    }
}

impl SessionConfig {
    /// Create a [`SessionConfig`] with the given [`DevicePreference`].
    pub fn with_device(device: DevicePreference) -> Self {
        Self {
            device,
            ..Self::default()
        }
    }

    /// Compute the effective intra-op thread count (`(physical / workers_hint).max(1)`)
    /// uniformly across all execution providers so concurrent HTTP workers do not oversubscribe CPUs.
    pub fn effective_intra_op_threads(&self, _provider: &ExecutionProvider) -> i32 {
        if let Some(explicit) = self.intra_op_threads {
            return explicit.max(1) as i32;
        }
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let physical = (logical / 2).max(1);
        let workers = self.workers_hint.max(1);
        (physical / workers).max(1) as i32
    }
}
