use std::ffi::{CStr, CString};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use once_cell::sync::OnceCell;

use crate::error::OnnxError;
use crate::ffi::{
    OrtApiBase, OrtApiTable, OrtEnv, OrtGetApiBaseFn, OrtStatus, ORT_LOGGING_LEVEL_WARNING,
};
use crate::loader::{resolve_ort_library_candidates, DynamicLibrary};

static GLOBAL_RUNTIME: OnceCell<Arc<OrtRuntime>> = OnceCell::new();

/// Process-wide or explicitly loaded ONNX Runtime dynamic library and `OrtEnv` handle.
pub struct OrtRuntime {
    lib: DynamicLibrary,
    version_string: String,
    api: OrtApiTable,
    env: *mut OrtEnv,
    available_providers: Vec<String>,
    gpu_mutex: Mutex<()>,
}

// Safety: `OrtEnv` is thread-safe in ONNX Runtime C API after initialization,
// `OrtApiTable` points to immutable `.rdata` function pointers, and `gpu_mutex`
// serializes GPU inference across worker threads.
unsafe impl Send for OrtRuntime {}
unsafe impl Sync for OrtRuntime {}

impl OrtRuntime {
    /// Get or initialize the shared global [`OrtRuntime`].
    pub fn global() -> Result<Arc<Self>, OnnxError> {
        GLOBAL_RUNTIME.get_or_try_init(|| Self::load(None)).cloned()
    }

    /// Load an independent [`OrtRuntime`] from an optional explicit shared library path.
    pub fn load(explicit_path: Option<&Path>) -> Result<Arc<Self>, OnnxError> {
        let candidates = resolve_ort_library_candidates(explicit_path);
        let mut last_err: Option<OnnxError> = None;

        // First try candidates that already exist on disk as files, then bare library names.
        let mut ordered = Vec::with_capacity(candidates.len());
        for c in &candidates {
            if c.is_file() {
                ordered.push(c.clone());
            }
        }
        for c in &candidates {
            if c.components().count() == 1 && !ordered.contains(c) {
                ordered.push(c.clone());
            }
        }

        if ordered.is_empty() {
            return Err(OnnxError::LibraryNotFound {
                searched: candidates,
            });
        }

        for candidate in ordered {
            match Self::load_from_candidate(&candidate) {
                Ok(rt) => return Ok(Arc::new(rt)),
                Err(err) => {
                    log::debug!(
                        "ONNX Runtime candidate '{}' failed to load: {}",
                        candidate.display(),
                        err
                    );
                    last_err = Some(err);
                }
            }
        }

        Err(last_err.unwrap_or(OnnxError::LibraryNotFound {
            searched: candidates,
        }))
    }

    fn load_from_candidate(path: &Path) -> Result<Self, OnnxError> {
        let lib = DynamicLibrary::open(path)?;
        let get_api_base_sym = lib.symbol("OrtGetApiBase")?;
        let get_api_base: OrtGetApiBaseFn = unsafe { std::mem::transmute(get_api_base_sym) };

        let base_ptr: *const OrtApiBase = unsafe { get_api_base() };
        if base_ptr.is_null() {
            return Err(OnnxError::LibraryLoadFailed {
                path: path.to_path_buf(),
                reason: "OrtGetApiBase() returned NULL".to_string(),
            });
        }

        let version_string = unsafe {
            let ver_ptr = ((*base_ptr).get_version_string)();
            if ver_ptr.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(ver_ptr).to_string_lossy().into_owned()
            }
        };

        let target_api_version = parse_ort_api_version(&version_string);
        let mut api_versions = Vec::new();
        if let Some(parsed) = target_api_version {
            api_versions.push(parsed.clamp(8, 24));
        }
        for v in [24u32, 22, 20, 18, 17, 16, 14, 12, 11, 10, 8] {
            if target_api_version.is_none_or(|p| v <= p) && !api_versions.contains(&v) {
                api_versions.push(v);
            }
        }

        let mut chosen_table = std::ptr::null();
        let mut chosen_version = 0u32;
        for v in api_versions {
            let table = unsafe { ((*base_ptr).get_api)(v) };
            if !table.is_null() {
                chosen_table = table;
                chosen_version = v;
                break;
            }
        }

        if chosen_table.is_null() {
            return Err(OnnxError::UnsupportedApiVersion {
                requested: target_api_version.unwrap_or(14),
                version_string,
            });
        }

        let api = unsafe { OrtApiTable::from_raw(chosen_table, chosen_version) };

        let mut env: *mut OrtEnv = std::ptr::null_mut();
        let log_id = CString::new("hifi_onnx").expect("static log_id has no NUL");
        let status =
            unsafe { api.create_env(ORT_LOGGING_LEVEL_WARNING, log_id.as_ptr(), &mut env) };
        if !status.is_null() {
            return Err(status_to_error(&api, status));
        }
        if env.is_null() {
            return Err(OnnxError::LibraryLoadFailed {
                path: path.to_path_buf(),
                reason: "OrtApi::CreateEnv returned NULL OrtEnv".to_string(),
            });
        }

        let available_providers = query_available_providers(&api);
        log::info!(
            "Loaded ONNX Runtime {} (API v{}) from '{}' with providers {:?}",
            version_string,
            chosen_version,
            lib.path().display(),
            available_providers
        );

        Ok(Self {
            lib,
            version_string,
            api,
            env,
            available_providers,
            gpu_mutex: Mutex::new(()),
        })
    }

    pub(crate) fn api(&self) -> &OrtApiTable {
        &self.api
    }

    pub(crate) fn env(&self) -> *mut OrtEnv {
        self.env
    }

    #[cfg_attr(
        not(any(
            feature = "directml",
            feature = "rocm",
            feature = "cuda",
            feature = "coreml"
        )),
        allow(dead_code)
    )]
    pub(crate) fn lib(&self) -> &DynamicLibrary {
        &self.lib
    }

    pub(crate) fn check_status(&self, status: *mut OrtStatus) -> Result<(), OnnxError> {
        if status.is_null() {
            Ok(())
        } else {
            Err(status_to_error(&self.api, status))
        }
    }

    /// Acquire the shared GPU inference mutex (used to serialize DirectML / CUDA / ROCm passes).
    pub fn lock_gpu(&self) -> MutexGuard<'_, ()> {
        match self.gpu_mutex.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Return the loaded ONNX Runtime library path.
    pub fn library_path(&self) -> &Path {
        self.lib.path()
    }

    /// Return the ONNX Runtime version string reported by `OrtGetApiBase`.
    pub fn version_string(&self) -> &str {
        &self.version_string
    }

    /// Return the negotiated ORT C API version.
    pub fn api_version(&self) -> u32 {
        self.api.api_version
    }

    /// Return the execution providers reported by `OrtApi::GetAvailableProviders`.
    pub fn available_providers(&self) -> &[String] {
        &self.available_providers
    }
}

impl Drop for OrtRuntime {
    fn drop(&mut self) {
        if !self.env.is_null() {
            unsafe {
                self.api.release_env(self.env);
            }
            self.env = std::ptr::null_mut();
        }
    }
}

fn parse_ort_api_version(version_string: &str) -> Option<u32> {
    let mut parts = version_string.trim().split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;
    if major == 1 {
        Some(minor)
    } else if major > 1 {
        Some(24)
    } else {
        None
    }
}

pub(crate) fn status_to_error(api: &OrtApiTable, status: *mut OrtStatus) -> OnnxError {
    unsafe {
        let code = api.get_error_code(status);
        let msg_ptr = api.get_error_message(status);
        let message = if msg_ptr.is_null() {
            "unknown ONNX Runtime error".to_string()
        } else {
            CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
        };
        api.release_status(status);
        OnnxError::OrtStatus { code, message }
    }
}

fn query_available_providers(api: &OrtApiTable) -> Vec<String> {
    let mut ptr: *mut *mut std::ffi::c_char = std::ptr::null_mut();
    let mut len: std::ffi::c_int = 0;
    let Some(status) = (unsafe { api.get_available_providers(&mut ptr, &mut len) }) else {
        return vec!["CPUExecutionProvider".to_string()];
    };
    if !status.is_null() {
        let err = status_to_error(api, status);
        log::warn!("GetAvailableProviders failed: {}", err);
        return vec!["CPUExecutionProvider".to_string()];
    }
    let mut providers = Vec::new();
    if !ptr.is_null() && len > 0 {
        unsafe {
            for i in 0..(len as usize) {
                let p = *ptr.add(i);
                if !p.is_null() {
                    providers.push(CStr::from_ptr(p).to_string_lossy().into_owned());
                }
            }
            api.release_available_providers(ptr, len);
        }
    }
    if providers.is_empty() {
        providers.push("CPUExecutionProvider".to_string());
    }
    providers
}
