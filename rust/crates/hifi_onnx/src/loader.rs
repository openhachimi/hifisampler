use std::env;
use std::ffi::{c_void, CString};
use std::path::{Path, PathBuf};

use crate::error::OnnxError;

#[cfg(target_os = "windows")]
const ORT_LIB_NAME: &str = "onnxruntime.dll";
#[cfg(target_os = "linux")]
const ORT_LIB_NAME: &str = "libonnxruntime.so";
#[cfg(target_os = "macos")]
const ORT_LIB_NAME: &str = "libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
const ORT_LIB_NAME: &str = "libonnxruntime.so";

/// Collect candidate ONNX Runtime shared library paths in priority order.
pub fn resolve_ort_library_candidates(explicit_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(p) = explicit_path {
        push_path_or_dir_candidates(&mut candidates, p);
    }

    if let Ok(env_val) = env::var("ORT_DYLIB_PATH") {
        let trimmed = env_val.trim();
        if !trimmed.is_empty() {
            let p = PathBuf::from(trimmed);
            if !p.exists() {
                log::warn!(
                    "ORT_DYLIB_PATH is set to '{}' which does not exist; continuing fallback search",
                    p.display()
                );
            }
            push_path_or_dir_candidates(&mut candidates, &p);
        }
    }

    // Executable directory and ancestor workspace/venv roots.
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            push_dir_ort_candidates(&mut candidates, exe_dir);
            let sub_capi = exe_dir.join("onnxruntime").join("capi");
            push_dir_ort_candidates(&mut candidates, &sub_capi);
            collect_venv_candidates_upwards(&mut candidates, exe_dir, 6);
        }
    }

    // Current working directory and ancestor workspace/venv roots.
    if let Ok(cwd) = env::current_dir() {
        push_dir_ort_candidates(&mut candidates, &cwd);
        collect_venv_candidates_upwards(&mut candidates, &cwd, 6);
    }

    // Active virtualenv / conda environment variables.
    for env_key in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Ok(root) = env::var(env_key) {
            let root_path = PathBuf::from(root.trim());
            collect_venv_root_candidates(&mut candidates, &root_path);
        }
    }

    // Platform-specific ROCm / Homebrew / system library paths.
    collect_system_candidates(&mut candidates);

    // Deduplicate while preserving priority order.
    let mut unique = Vec::with_capacity(candidates.len());
    for c in candidates {
        if !unique.contains(&c) {
            unique.push(c);
        }
    }
    unique
}

/// Resolve the highest-priority existing ONNX Runtime shared library path on disk,
/// falling back to a bare library name if no explicit file was found.
pub fn resolve_ort_library_path(explicit_path: Option<&Path>) -> Result<PathBuf, OnnxError> {
    let candidates = resolve_ort_library_candidates(explicit_path);
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    Err(OnnxError::LibraryNotFound {
        searched: candidates,
    })
}

fn push_path_or_dir_candidates(out: &mut Vec<PathBuf>, p: &Path) {
    if p.is_dir() {
        push_dir_ort_candidates(out, p);
    } else {
        out.push(p.to_path_buf());
    }
}

fn push_dir_ort_candidates(out: &mut Vec<PathBuf>, dir: &Path) {
    out.push(dir.join(ORT_LIB_NAME));
    #[cfg(target_os = "linux")]
    collect_versioned_so_files(out, dir);
    #[cfg(target_os = "macos")]
    collect_versioned_dylib_files(out, dir);
}

#[cfg(target_os = "linux")]
fn collect_versioned_so_files(out: &mut Vec<PathBuf>, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut matches: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libonnxruntime.so.") && p.is_file())
                .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.reverse();
    out.extend(matches);
}

#[cfg(target_os = "macos")]
fn collect_versioned_dylib_files(out: &mut Vec<PathBuf>, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut matches: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("libonnxruntime.") && n.ends_with(".dylib") && p.is_file())
                .unwrap_or(false)
        })
        .collect();
    matches.sort();
    matches.reverse();
    out.extend(matches);
}

fn collect_venv_candidates_upwards(out: &mut Vec<PathBuf>, start: &Path, max_levels: usize) {
    let mut cur = Some(start);
    for _ in 0..=max_levels {
        let Some(dir) = cur else {
            break;
        };
        for venv_name in [".venv", "venv"] {
            let venv_dir = dir.join(venv_name);
            if venv_dir.is_dir() {
                collect_venv_root_candidates(out, &venv_dir);
            }
        }
        cur = dir.parent();
    }
}

fn collect_venv_root_candidates(out: &mut Vec<PathBuf>, venv_root: &Path) {
    #[cfg(target_os = "windows")]
    {
        let capi_dir = venv_root
            .join("Lib")
            .join("site-packages")
            .join("onnxruntime")
            .join("capi");
        push_dir_ort_candidates(out, &capi_dir);
    }

    #[cfg(not(target_os = "windows"))]
    {
        for lib_dir_name in ["lib", "lib64"] {
            let lib_dir = venv_root.join(lib_dir_name);
            let Ok(entries) = std::fs::read_dir(&lib_dir) else {
                continue;
            };
            let mut py_dirs: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.starts_with("python"))
                            .unwrap_or(false)
                })
                .collect();
            py_dirs.sort();
            py_dirs.reverse();
            for py_dir in py_dirs {
                let capi_dir = py_dir
                    .join("site-packages")
                    .join("onnxruntime")
                    .join("capi");
                push_dir_ort_candidates(out, &capi_dir);
            }
        }
    }
}

fn collect_system_candidates(out: &mut Vec<PathBuf>) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(rocm_path) = env::var("ROCM_PATH") {
            let base = PathBuf::from(rocm_path.trim());
            push_dir_ort_candidates(out, &base.join("lib"));
            push_dir_ort_candidates(out, &base.join("onnxruntime").join("lib"));
        }
        for dir in [
            "/opt/rocm/lib",
            "/opt/rocm/onnxruntime/lib",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
        ] {
            push_dir_ort_candidates(out, Path::new(dir));
        }
        if let Ok(ld_path) = env::var("LD_LIBRARY_PATH") {
            for entry in env::split_paths(&ld_path) {
                if !entry.as_os_str().is_empty() {
                    push_dir_ort_candidates(out, &entry);
                }
            }
        }
        out.push(PathBuf::from("libonnxruntime.so"));
        out.push(PathBuf::from("libonnxruntime.so.1"));
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(dyld_path) = env::var("DYLD_LIBRARY_PATH") {
            for entry in env::split_paths(&dyld_path) {
                if !entry.as_os_str().is_empty() {
                    push_dir_ort_candidates(out, &entry);
                }
            }
        }
        for dir in ["/opt/homebrew/lib", "/usr/local/lib"] {
            push_dir_ort_candidates(out, Path::new(dir));
        }
        out.push(PathBuf::from("libonnxruntime.dylib"));
    }

    #[cfg(target_os = "windows")]
    {
        if let Ok(path_var) = env::var("PATH") {
            for entry in env::split_paths(&path_var) {
                if !entry.as_os_str().is_empty() {
                    let dll = entry.join(ORT_LIB_NAME);
                    if dll.is_file() {
                        out.push(dll);
                    }
                }
            }
        }
        out.push(PathBuf::from(ORT_LIB_NAME));
    }
}

/// Platform-native dynamically loaded shared library handle.
pub(crate) struct DynamicLibrary {
    #[cfg(target_os = "windows")]
    handle: windows_sys::Win32::Foundation::HMODULE,
    #[cfg(not(target_os = "windows"))]
    handle: *mut c_void,
    path: PathBuf,
}

// Safety: OS dynamic library handles (`HMODULE` / `dlopen` handles) are process-wide and safe to share across threads.
unsafe impl Send for DynamicLibrary {}
unsafe impl Sync for DynamicLibrary {}

impl DynamicLibrary {
    /// Load a shared library from `path`, preparing sibling DLL search paths on Windows.
    pub fn open(path: &Path) -> Result<Self, OnnxError> {
        #[cfg(target_os = "windows")]
        {
            Self::open_windows(path)
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self::open_unix(path)
        }
    }

    /// Look up an exported function/symbol pointer by exact C name.
    pub fn symbol(&self, name: &str) -> Result<*const c_void, OnnxError> {
        let cname = CString::new(name).map_err(|_| OnnxError::SymbolNotFound {
            symbol: name.to_string(),
            reason: "symbol name contains interior NUL byte".to_string(),
        })?;

        #[cfg(target_os = "windows")]
        {
            let proc = unsafe {
                windows_sys::Win32::System::LibraryLoader::GetProcAddress(
                    self.handle,
                    cname.as_ptr() as *const u8,
                )
            };
            match proc {
                Some(f) => Ok(f as *const c_void),
                None => Err(OnnxError::SymbolNotFound {
                    symbol: name.to_string(),
                    reason: format!("GetProcAddress returned NULL (error {})", unsafe {
                        windows_sys::Win32::Foundation::GetLastError()
                    }),
                }),
            }
        }

        #[cfg(not(target_os = "windows"))]
        {
            unsafe {
                // Clear any prior dlerror before calling dlsym.
                let _ = libc::dlerror();
                let sym = libc::dlsym(self.handle, cname.as_ptr());
                if sym.is_null() {
                    let err_ptr = libc::dlerror();
                    let reason = if err_ptr.is_null() {
                        "dlsym returned NULL".to_string()
                    } else {
                        std::ffi::CStr::from_ptr(err_ptr)
                            .to_string_lossy()
                            .into_owned()
                    };
                    return Err(OnnxError::SymbolNotFound {
                        symbol: name.to_string(),
                        reason,
                    });
                }
                Ok(sym as *const c_void)
            }
        }
    }

    /// Return the path used to open this library.
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(target_os = "windows")]
    fn open_windows(path: &Path) -> Result<Self, OnnxError> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Foundation::GetLastError;
        use windows_sys::Win32::System::LibraryLoader::{
            AddDllDirectory, LoadLibraryExW, LoadLibraryW, SetDllDirectoryW,
            LOAD_WITH_ALTERED_SEARCH_PATH,
        };

        // Ensure DirectML.dll and onnxruntime_providers_shared.dll next to onnxruntime.dll
        // are discoverable by the Windows loader before loading onnxruntime.dll.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && parent.is_dir() {
                let wide_dir: Vec<u16> = parent
                    .as_os_str()
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect();
                unsafe {
                    let _ = SetDllDirectoryW(wide_dir.as_ptr());
                    let _ = AddDllDirectory(wide_dir.as_ptr());
                }
            }
        }

        let wide_path: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        let mut handle = unsafe { LoadLibraryW(wide_path.as_ptr()) };
        if handle.is_null() {
            handle = unsafe {
                LoadLibraryExW(
                    wide_path.as_ptr(),
                    std::ptr::null_mut(),
                    LOAD_WITH_ALTERED_SEARCH_PATH,
                )
            };
        }
        if handle.is_null() {
            let err = unsafe { GetLastError() };
            return Err(OnnxError::LibraryLoadFailed {
                path: path.to_path_buf(),
                reason: format!("LoadLibraryW failed with Win32 error {err}"),
            });
        }

        Ok(Self {
            handle,
            path: path.to_path_buf(),
        })
    }

    #[cfg(not(target_os = "windows"))]
    fn open_unix(path: &Path) -> Result<Self, OnnxError> {
        use std::os::unix::ffi::OsStrExt;

        let cpath = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
            OnnxError::LibraryLoadFailed {
                path: path.to_path_buf(),
                reason: "library path contains interior NUL byte".to_string(),
            }
        })?;

        let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            let reason = unsafe {
                let err_ptr = libc::dlerror();
                if err_ptr.is_null() {
                    "dlopen returned NULL".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err_ptr)
                        .to_string_lossy()
                        .into_owned()
                }
            };
            return Err(OnnxError::LibraryLoadFailed {
                path: path.to_path_buf(),
                reason,
            });
        }

        Ok(Self {
            handle,
            path: path.to_path_buf(),
        })
    }
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            if !self.handle.is_null() {
                let _ = windows_sys::Win32::Foundation::FreeLibrary(self.handle);
            }
        }
        #[cfg(not(target_os = "windows"))]
        unsafe {
            if !self.handle.is_null() {
                let _ = libc::dlclose(self.handle);
            }
        }
    }
}
