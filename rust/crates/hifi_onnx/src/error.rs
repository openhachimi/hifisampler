use std::path::PathBuf;
use thiserror::Error;

/// Errors returned by the `hifi_onnx` ONNX Runtime loader and inference wrappers.
#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("ONNX Runtime shared library not found; searched: {searched:?}")]
    LibraryNotFound { searched: Vec<PathBuf> },

    #[error("failed to load ONNX Runtime shared library '{path}': {reason}")]
    LibraryLoadFailed { path: PathBuf, reason: String },

    #[error("required ONNX Runtime symbol '{symbol}' not found: {reason}")]
    SymbolNotFound { symbol: String, reason: String },

    #[error(
        "unsupported ONNX Runtime C API version {requested} (library version '{version_string}')"
    )]
    UnsupportedApiVersion {
        requested: u32,
        version_string: String,
    },

    #[error("ONNX Runtime error (code {code}): {message}")]
    OrtStatus { code: i32, message: String },

    #[error("model file not found at '{path}' (checked: {checked:?})")]
    ModelNotFound {
        path: PathBuf,
        checked: Vec<PathBuf>,
    },

    #[error("unsupported or invalid model path '{path}': {message}")]
    InvalidModel { path: PathBuf, message: String },

    #[error("tensor shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("invalid inference input: {message}")]
    InvalidInput { message: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
