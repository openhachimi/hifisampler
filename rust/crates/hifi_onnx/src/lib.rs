//! `hifi_onnx`: Clean-room, pure-ONNX dynamic inference wrapper for `hifisampler-rs`.
//!
//! Dynamically loads ONNX Runtime (`onnxruntime.dll` / `libonnxruntime.so` / `libonnxruntime.dylib`)
//! via `OrtGetApiBase` without PyTorch or static ORT link dependencies, supporting cross-platform
//! Execution Providers (`DirectML`, `CUDA`, `ROCm`, `CoreML`, `MIGraphX`, `CPU`) with automatic
//! fallback to CPU on any provider initialization failure.
//!
//! GPU forward passes on `VocoderSession` and `HnsepSession` are automatically serialized across
//! worker threads via an internal runtime GPU mutex whenever `is_gpu_ep()` is `true`.

pub mod device;
pub mod error;
pub(crate) mod ffi;
pub mod hnsep;
pub mod loader;
pub mod runtime;
pub mod session;
pub mod vocoder;

pub use device::{DevicePreference, ExecutionProvider, SessionConfig};
pub use error::OnnxError;
pub use hnsep::{resolve_hnsep_path, HnsepConfig, HnsepSession, DEFAULT_HNSEP_ONNX_PATHS};
pub use loader::{resolve_ort_library_candidates, resolve_ort_library_path};
pub use runtime::OrtRuntime;
pub use session::{OrtSession, TensorMeta, TensorOutput};
pub use vocoder::{resolve_vocoder_path, VocoderSession, DEFAULT_VOCODER_ONNX_PATH};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{f16_bits_to_f32, f32_to_f16_bits};

    #[test]
    fn device_preference_parsing_matches_config_aliases() {
        assert_eq!(
            DevicePreference::from_config_str("auto"),
            DevicePreference::Auto
        );
        assert_eq!(
            DevicePreference::from_config_str(""),
            DevicePreference::Auto
        );
        assert_eq!(
            DevicePreference::from_config_str("CPU"),
            DevicePreference::Cpu
        );
        assert_eq!(
            DevicePreference::from_config_str("cuda"),
            DevicePreference::Cuda
        );
        assert_eq!(
            DevicePreference::from_config_str("rocm"),
            DevicePreference::Rocm
        );
        assert_eq!(
            DevicePreference::from_config_str("hip"),
            DevicePreference::Rocm
        );
        assert_eq!(
            DevicePreference::from_config_str("dml"),
            DevicePreference::DirectMl
        );
        assert_eq!(
            DevicePreference::from_config_str("DirectML"),
            DevicePreference::DirectMl
        );
        assert_eq!(
            DevicePreference::from_config_str("coreml"),
            DevicePreference::CoreMl
        );
        assert_eq!(
            DevicePreference::from_config_str("mps"),
            DevicePreference::CoreMl
        );
        assert_eq!(
            DevicePreference::from_config_str("unknown"),
            DevicePreference::Auto
        );
    }

    #[test]
    fn execution_provider_gpu_flags() {
        assert!(!ExecutionProvider::Cpu.is_gpu_ep());
        assert!(ExecutionProvider::DirectMl.is_gpu_ep());
        assert!(ExecutionProvider::Cuda.is_gpu_ep());
        assert!(ExecutionProvider::Rocm.is_gpu_ep());
        assert!(ExecutionProvider::CoreMl.is_gpu_ep());
        assert_eq!(
            ExecutionProvider::Rocm.provider_name(),
            "ROCMExecutionProvider"
        );
        assert_eq!(
            ExecutionProvider::DirectMl.provider_name(),
            "DmlExecutionProvider"
        );
    }

    #[test]
    fn f16_f32_conversion_roundtrip() {
        for &val in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 440.0, 1e-3, -12.25] {
            let bits = f32_to_f16_bits(val);
            let back = f16_bits_to_f32(bits);
            assert!(
                (val - back).abs() <= val.abs() * 1e-3 + 1e-6,
                "f16 roundtrip mismatch for {val}: got {back}"
            );
        }
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY)).is_infinite());
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }
}
