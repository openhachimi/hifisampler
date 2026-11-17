use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::device::{DevicePreference, ExecutionProvider, SessionConfig};
use crate::error::OnnxError;
use crate::ffi::{
    f16_bits_to_f32, f32_to_f16_bits, OrtAllocator, OrtMemoryInfo, OrtSession as RawOrtSession,
    OrtSessionOptions, OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtValue,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    ORT_DEVICE_ALLOCATOR, ORT_ENABLE_ALL, ORT_MEM_TYPE_DEFAULT, ORT_SEQUENTIAL,
};
use crate::runtime::{status_to_error, OrtRuntime};

/// Metadata describing one input or output tensor of a loaded ONNX session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub name: String,
    pub elem_type: i32,
    pub dims: Vec<i64>,
}

/// Output tensor returned by [`OrtSession::run_f32`]: `(shape, flat_f32_data)`.
pub type TensorOutput = (Vec<i64>, Vec<f32>);

/// Thread-safe ONNX Runtime inference session with automatic EP selection & CPU fallback.
pub struct OrtSession {
    runtime: Arc<OrtRuntime>,
    session: *mut RawOrtSession,
    cpu_mem_info: *mut OrtMemoryInfo,
    execution_provider: ExecutionProvider,
    fallback_reason: Option<String>,
    inputs: Vec<TensorMeta>,
    outputs: Vec<TensorMeta>,
    model_path: PathBuf,
}

// Safety: ONNX Runtime `OrtSession::Run` is thread-safe for concurrent execution;
// GPU execution is additionally serialized by `OrtRuntime::lock_gpu()` when `is_gpu_ep()` is true.
unsafe impl Send for OrtSession {}
unsafe impl Sync for OrtSession {}

impl OrtSession {
    /// Load an ONNX model from `model_path` using the shared [`OrtRuntime`] (or explicit path in `config`).
    pub fn from_file(
        model_path: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let runtime = match config.ort_library_path.as_deref() {
            Some(p) => OrtRuntime::load(Some(p))?,
            None => OrtRuntime::global()?,
        };
        Self::from_file_with_runtime(runtime, model_path, config)
    }

    /// Load an ONNX model from `model_path` using a provided [`OrtRuntime`].
    pub fn from_file_with_runtime(
        runtime: Arc<OrtRuntime>,
        model_path: impl AsRef<Path>,
        config: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let path = model_path.as_ref();
        if !path.is_file() {
            return Err(OnnxError::ModelNotFound {
                path: path.to_path_buf(),
                checked: vec![path.to_path_buf()],
            });
        }

        let (candidates, initial_reason) =
            plan_provider_candidates(config, runtime.available_providers());
        let mut fallback_reasons: Vec<String> = Vec::new();
        if let Some(reason) = initial_reason {
            log::warn!("{}", reason);
            fallback_reasons.push(reason);
        }

        let mut last_err: Option<OnnxError> = None;
        for candidate in candidates {
            match try_create_session(&runtime, path, config, &candidate) {
                Ok(session) => {
                    let fallback_reason = if fallback_reasons.is_empty() {
                        None
                    } else {
                        Some(fallback_reasons.join("; "))
                    };
                    return finish_session(
                        runtime,
                        session,
                        path.to_path_buf(),
                        candidate,
                        fallback_reason,
                    );
                }
                Err(err) => {
                    let msg = format!(
                        "Execution provider '{}' failed for '{}': {}",
                        candidate.provider_name(),
                        path.display(),
                        err
                    );
                    if matches!(candidate, ExecutionProvider::Cpu) {
                        last_err = Some(err);
                    } else {
                        log::warn!("{}; falling back to next provider", msg);
                        fallback_reasons.push(msg);
                        last_err = Some(err);
                    }
                }
            }
        }

        Err(last_err.unwrap_or_else(|| OnnxError::InvalidInput {
            message: format!("no execution provider succeeded for '{}'", path.display()),
        }))
    }

    pub fn runtime(&self) -> &Arc<OrtRuntime> {
        &self.runtime
    }

    pub fn execution_provider(&self) -> &ExecutionProvider {
        &self.execution_provider
    }

    pub fn is_gpu_ep(&self) -> bool {
        self.execution_provider.is_gpu_ep()
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    pub fn inputs(&self) -> &[TensorMeta] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[TensorMeta] {
        &self.outputs
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Run `f32` inference by tensor name, converting `f32 <-> f16` transparently if a tensor
    /// expects/returns `FLOAT16`. Automatically acquires the GPU mutex when `is_gpu_ep()` is true.
    pub fn run_f32(
        &self,
        inputs: &[(&str, &[i64], &[f32])],
        output_names: &[&str],
    ) -> Result<Vec<TensorOutput>, OnnxError> {
        self.run_f32_timed(inputs, output_names)
            .map(|(outputs, _)| outputs)
    }

    /// Run `f32` inference like [`Self::run_f32`], also returning `gpu_wait_ms` spent waiting
    /// on the shared runtime GPU mutex (`0.0` on CPU).
    pub fn run_f32_timed(
        &self,
        inputs: &[(&str, &[i64], &[f32])],
        output_names: &[&str],
    ) -> Result<(Vec<TensorOutput>, f64), OnnxError> {
        if inputs.is_empty() {
            return Err(OnnxError::InvalidInput {
                message: "at least one input tensor is required".to_string(),
            });
        }
        if output_names.is_empty() {
            return Err(OnnxError::InvalidInput {
                message: "at least one output tensor name is required".to_string(),
            });
        }

        let api = self.runtime.api();

        let mut c_input_names = Vec::with_capacity(inputs.len());
        let mut f16_buffers: Vec<Vec<u16>> = Vec::new();
        let mut input_guards = Vec::with_capacity(inputs.len());

        for &(name, dims, data) in inputs {
            let expected_elems = checked_shape_product(dims)?;
            if data.len() != expected_elems {
                return Err(OnnxError::ShapeMismatch {
                    expected: format!("{dims:?} ({expected_elems} elements)"),
                    actual: format!("{} elements", data.len()),
                });
            }

            let cname = CString::new(name).map_err(|_| OnnxError::InvalidInput {
                message: format!("input name '{name}' contains interior NUL"),
            })?;
            c_input_names.push(cname);

            let meta_elem_type = self
                .inputs
                .iter()
                .find(|m| m.name == name)
                .map(|m| m.elem_type)
                .unwrap_or(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

            let (elem_type, byte_len, ptr) =
                if meta_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 {
                    let mut f16_vec: Vec<u16> = data.iter().copied().map(f32_to_f16_bits).collect();
                    let byte_len = f16_vec.len() * std::mem::size_of::<u16>();
                    let ptr = f16_vec.as_mut_ptr() as *mut c_void;
                    f16_buffers.push(f16_vec);
                    (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, byte_len, ptr)
                } else {
                    // Safe: `inputs` outlives `api.run()` and `input_guards` (dropped right after `run`).
                    let byte_len = std::mem::size_of_val(data);
                    let ptr = data.as_ptr() as *mut c_void;
                    (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, byte_len, ptr)
                };

            let mut ort_val: *mut OrtValue = std::ptr::null_mut();
            let st = unsafe {
                api.create_tensor_with_data_as_ort_value(
                    self.cpu_mem_info,
                    ptr,
                    byte_len,
                    dims.as_ptr(),
                    dims.len(),
                    elem_type,
                    &mut ort_val,
                )
            };
            self.runtime.check_status(st)?;
            input_guards.push(OrtValueGuard {
                api: *api,
                val: ort_val,
            });
        }

        let mut c_output_names = Vec::with_capacity(output_names.len());
        for &name in output_names {
            c_output_names.push(CString::new(name).map_err(|_| OnnxError::InvalidInput {
                message: format!("output name '{name}' contains interior NUL"),
            })?);
        }

        let input_name_ptrs: Vec<*const c_char> =
            c_input_names.iter().map(|s| s.as_ptr()).collect();
        let input_val_ptrs: Vec<*const OrtValue> = input_guards
            .iter()
            .map(|g| g.val as *const OrtValue)
            .collect();
        let output_name_ptrs: Vec<*const c_char> =
            c_output_names.iter().map(|s| s.as_ptr()).collect();
        let mut raw_outputs: Vec<*mut OrtValue> = vec![std::ptr::null_mut(); output_names.len()];

        let gpu_wait_ms = {
            let (gpu_wait_ms, _gpu_guard) = if self.execution_provider.is_gpu_ep() {
                let wait_start = std::time::Instant::now();
                let guard = self.runtime.lock_gpu();
                (wait_start.elapsed().as_secs_f64() * 1000.0, Some(guard))
            } else {
                (0.0, None)
            };
            let st = unsafe {
                api.run(
                    self.session,
                    std::ptr::null(),
                    input_name_ptrs.as_ptr(),
                    input_val_ptrs.as_ptr(),
                    input_val_ptrs.len(),
                    output_name_ptrs.as_ptr(),
                    output_name_ptrs.len(),
                    raw_outputs.as_mut_ptr(),
                )
            };
            // Ensure all output guards are registered even if run fails partially.
            if !st.is_null() {
                for val in raw_outputs {
                    if !val.is_null() {
                        unsafe { api.release_value(val) };
                    }
                }
                return Err(status_to_error(api, st));
            }
            gpu_wait_ms
        };
        // Release OrtValue wrappers before dropping temporary FP16 backing buffers.
        drop(input_guards);
        drop(f16_buffers);

        let output_guards: Vec<OrtValueGuard> = raw_outputs
            .into_iter()
            .map(|val| OrtValueGuard { api: *api, val })
            .collect();

        let mut results = Vec::with_capacity(output_guards.len());
        for guard in &output_guards {
            if guard.val.is_null() {
                return Err(OnnxError::InvalidInput {
                    message: "ONNX Runtime returned a NULL output OrtValue".to_string(),
                });
            }
            let mut is_tensor: c_int = 0;
            let st = unsafe { api.is_tensor(guard.val, &mut is_tensor) };
            self.runtime.check_status(st)?;
            if is_tensor == 0 {
                return Err(OnnxError::InvalidInput {
                    message: "ONNX Runtime output is not a tensor".to_string(),
                });
            }

            let mut shape_info: *mut OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
            let st = unsafe { api.get_tensor_type_and_shape(guard.val, &mut shape_info) };
            self.runtime.check_status(st)?;
            let shape_guard = TensorShapeInfoGuard {
                api: *api,
                info: shape_info,
            };

            let mut elem_type: c_int = 0;
            let st = unsafe { api.get_tensor_element_type(shape_guard.info, &mut elem_type) };
            self.runtime.check_status(st)?;

            let mut ndims: usize = 0;
            let st = unsafe { api.get_dimensions_count(shape_guard.info, &mut ndims) };
            self.runtime.check_status(st)?;

            let mut dims = vec![0i64; ndims];
            if ndims > 0 {
                let st = unsafe { api.get_dimensions(shape_guard.info, dims.as_mut_ptr(), ndims) };
                self.runtime.check_status(st)?;
            }

            let mut elem_count: usize = 0;
            let st =
                unsafe { api.get_tensor_shape_element_count(shape_guard.info, &mut elem_count) };
            self.runtime.check_status(st)?;

            let mut data_ptr: *mut c_void = std::ptr::null_mut();
            let st = unsafe { api.get_tensor_mutable_data(guard.val, &mut data_ptr) };
            self.runtime.check_status(st)?;

            let values = if elem_count == 0 {
                Vec::new()
            } else if data_ptr.is_null() {
                return Err(OnnxError::InvalidInput {
                    message: "GetTensorMutableData returned NULL for non-empty tensor".to_string(),
                });
            } else if elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT {
                let slice =
                    unsafe { std::slice::from_raw_parts(data_ptr as *const f32, elem_count) };
                slice.to_vec()
            } else if elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 {
                let slice =
                    unsafe { std::slice::from_raw_parts(data_ptr as *const u16, elem_count) };
                slice.iter().copied().map(f16_bits_to_f32).collect()
            } else {
                return Err(OnnxError::InvalidInput {
                    message: format!("unsupported output tensor element type {elem_type}"),
                });
            };

            results.push((dims, values));
        }

        Ok((results, gpu_wait_ms))
    }
}

impl Drop for OrtSession {
    fn drop(&mut self) {
        let api = self.runtime.api();
        unsafe {
            if !self.cpu_mem_info.is_null() {
                api.release_memory_info(self.cpu_mem_info);
                self.cpu_mem_info = std::ptr::null_mut();
            }
            if !self.session.is_null() {
                api.release_session(self.session);
                self.session = std::ptr::null_mut();
            }
        }
    }
}

fn finish_session(
    runtime: Arc<OrtRuntime>,
    session: *mut RawOrtSession,
    model_path: PathBuf,
    execution_provider: ExecutionProvider,
    fallback_reason: Option<String>,
) -> Result<OrtSession, OnnxError> {
    let api = runtime.api();
    let mut mem_info: *mut OrtMemoryInfo = std::ptr::null_mut();
    let st = unsafe {
        api.create_cpu_memory_info(ORT_DEVICE_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, &mut mem_info)
    };
    if !st.is_null() {
        unsafe { api.release_session(session) };
        return Err(status_to_error(api, st));
    }

    let mut out = OrtSession {
        runtime,
        session,
        cpu_mem_info: mem_info,
        execution_provider,
        fallback_reason,
        inputs: Vec::new(),
        outputs: Vec::new(),
        model_path,
    };

    out.inputs = inspect_io(&out.runtime, out.session, true)?;
    out.outputs = inspect_io(&out.runtime, out.session, false)?;

    log::info!(
        "Loaded ONNX model '{}' using {} (gpu_ep={})",
        out.model_path.display(),
        out.execution_provider,
        out.execution_provider.is_gpu_ep()
    );
    Ok(out)
}

fn inspect_io(
    runtime: &OrtRuntime,
    session: *mut RawOrtSession,
    is_input: bool,
) -> Result<Vec<TensorMeta>, OnnxError> {
    let api = runtime.api();
    let mut count: usize = 0;
    let st = unsafe {
        if is_input {
            api.session_get_input_count(session, &mut count)
        } else {
            api.session_get_output_count(session, &mut count)
        }
    };
    runtime.check_status(st)?;

    let mut allocator: *mut OrtAllocator = std::ptr::null_mut();
    let st = unsafe { api.get_allocator_with_default_options(&mut allocator) };
    runtime.check_status(st)?;

    let mut metas = Vec::with_capacity(count);
    for idx in 0..count {
        let mut name_ptr: *mut c_char = std::ptr::null_mut();
        let st = unsafe {
            if is_input {
                api.session_get_input_name(session, idx, allocator, &mut name_ptr)
            } else {
                api.session_get_output_name(session, idx, allocator, &mut name_ptr)
            }
        };
        runtime.check_status(st)?;
        let name = if name_ptr.is_null() {
            format!("{}_{idx}", if is_input { "input" } else { "output" })
        } else {
            let s = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();
            let free_st = unsafe { api.allocator_free(allocator, name_ptr as *mut c_void) };
            runtime.check_status(free_st)?;
            s
        };

        let mut type_info: *mut OrtTypeInfo = std::ptr::null_mut();
        let st = unsafe {
            if is_input {
                api.session_get_input_type_info(session, idx, &mut type_info)
            } else {
                api.session_get_output_type_info(session, idx, &mut type_info)
            }
        };
        runtime.check_status(st)?;
        let ti_guard = TypeInfoGuard {
            api: *api,
            info: type_info,
        };

        let mut tensor_info: *const OrtTensorTypeAndShapeInfo = std::ptr::null();
        let st = unsafe { api.cast_type_info_to_tensor_info(ti_guard.info, &mut tensor_info) };
        runtime.check_status(st)?;

        let mut elem_type: c_int = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        let mut dims = Vec::new();
        if !tensor_info.is_null() {
            let st = unsafe { api.get_tensor_element_type(tensor_info, &mut elem_type) };
            runtime.check_status(st)?;
            let mut ndims: usize = 0;
            let st = unsafe { api.get_dimensions_count(tensor_info, &mut ndims) };
            runtime.check_status(st)?;
            if ndims > 0 {
                dims.resize(ndims, -1);
                let st = unsafe { api.get_dimensions(tensor_info, dims.as_mut_ptr(), ndims) };
                runtime.check_status(st)?;
            }
        }

        metas.push(TensorMeta {
            name,
            elem_type,
            dims,
        });
    }

    Ok(metas)
}

fn plan_provider_candidates(
    config: &SessionConfig,
    available: &[String],
) -> (Vec<ExecutionProvider>, Option<String>) {
    let has = |name: &str| available.iter().any(|p| p == name);

    match config.device {
        DevicePreference::Cpu => (vec![ExecutionProvider::Cpu], None),
        DevicePreference::DirectMl => {
            if !cfg!(feature = "directml") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("DirectML support was not compiled in (feature `directml` disabled); falling back to CPU".to_string()),
                );
            }
            if !cfg!(target_os = "windows") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("DirectML is only supported on Windows; falling back to CPU".to_string()),
                );
            }
            if !has("DmlExecutionProvider") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some(format!(
                        "DmlExecutionProvider not available in loaded ORT ({available:?}); falling back to CPU"
                    )),
                );
            }
            (
                vec![ExecutionProvider::DirectMl, ExecutionProvider::Cpu],
                None,
            )
        }
        DevicePreference::Cuda => {
            if !cfg!(feature = "cuda") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("CUDA support was not compiled in (feature `cuda` disabled); falling back to CPU".to_string()),
                );
            }
            if !has("CUDAExecutionProvider") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some(format!(
                        "CUDAExecutionProvider not available in loaded ORT ({available:?}); falling back to CPU"
                    )),
                );
            }
            (vec![ExecutionProvider::Cuda, ExecutionProvider::Cpu], None)
        }
        DevicePreference::Rocm => {
            if !cfg!(feature = "rocm") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("ROCm support was not compiled in (feature `rocm` disabled); falling back to CPU".to_string()),
                );
            }
            if !cfg!(target_os = "linux") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("ROCm is only supported on Linux; falling back to CPU".to_string()),
                );
            }
            if !has("ROCMExecutionProvider") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some(format!(
                        "ROCMExecutionProvider not available in loaded ORT ({available:?}); falling back to CPU"
                    )),
                );
            }
            (vec![ExecutionProvider::Rocm, ExecutionProvider::Cpu], None)
        }
        DevicePreference::CoreMl => {
            if !cfg!(feature = "coreml") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("CoreML support was not compiled in (feature `coreml` disabled); falling back to CPU".to_string()),
                );
            }
            if !cfg!(target_os = "macos") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some("CoreML is only supported on macOS; falling back to CPU".to_string()),
                );
            }
            if !has("CoreMLExecutionProvider") {
                return (
                    vec![ExecutionProvider::Cpu],
                    Some(format!(
                        "CoreMLExecutionProvider not available in loaded ORT ({available:?}); falling back to CPU"
                    )),
                );
            }
            (
                vec![ExecutionProvider::CoreMl, ExecutionProvider::Cpu],
                None,
            )
        }
        #[allow(clippy::vec_init_then_push)]
        DevicePreference::Auto => {
            let mut list = Vec::new();
            #[cfg(all(target_os = "windows", feature = "cuda"))]
            if has("CUDAExecutionProvider") {
                list.push(ExecutionProvider::Cuda);
            }
            #[cfg(all(target_os = "windows", feature = "directml"))]
            if has("DmlExecutionProvider") {
                list.push(ExecutionProvider::DirectMl);
            }
            #[cfg(all(target_os = "linux", feature = "cuda"))]
            if has("CUDAExecutionProvider") {
                list.push(ExecutionProvider::Cuda);
            }
            #[cfg(all(target_os = "linux", feature = "rocm"))]
            if has("ROCMExecutionProvider") {
                list.push(ExecutionProvider::Rocm);
            }
            #[cfg(all(target_os = "linux", feature = "migraphx"))]
            if has("MIGraphXExecutionProvider") {
                list.push(ExecutionProvider::MigraphX);
            }
            #[cfg(all(target_os = "macos", feature = "coreml"))]
            if has("CoreMLExecutionProvider") {
                list.push(ExecutionProvider::CoreMl);
            }
            list.push(ExecutionProvider::Cpu);
            (list, None)
        }
    }
}

fn try_create_session(
    runtime: &OrtRuntime,
    model_path: &Path,
    config: &SessionConfig,
    provider: &ExecutionProvider,
) -> Result<*mut RawOrtSession, OnnxError> {
    let api = runtime.api();
    let mut so: *mut OrtSessionOptions = std::ptr::null_mut();
    let st = unsafe { api.create_session_options(&mut so) };
    runtime.check_status(st)?;
    let so_guard = SessionOptionsGuard {
        api: *api,
        opts: so,
    };

    let is_dml = matches!(provider, ExecutionProvider::DirectMl);
    let enable_mem_pattern = config.enable_mem_pattern.unwrap_or(!is_dml);
    let intra_threads = config.effective_intra_op_threads(provider);
    let inter_threads = if is_dml {
        1
    } else {
        config.inter_op_threads.max(1) as i32
    };

    unsafe {
        runtime.check_status(api.set_intra_op_num_threads(so_guard.opts, intra_threads))?;
        runtime.check_status(api.set_inter_op_num_threads(so_guard.opts, inter_threads))?;
        runtime.check_status(api.set_session_execution_mode(so_guard.opts, ORT_SEQUENTIAL))?;
        runtime.check_status(
            api.set_session_graph_optimization_level(so_guard.opts, ORT_ENABLE_ALL),
        )?;
        if enable_mem_pattern {
            runtime.check_status(api.enable_mem_pattern(so_guard.opts))?;
        } else {
            runtime.check_status(api.disable_mem_pattern(so_guard.opts))?;
        }
        if config.enable_cpu_mem_arena {
            runtime.check_status(api.enable_cpu_mem_arena(so_guard.opts))?;
        } else {
            runtime.check_status(api.disable_cpu_mem_arena(so_guard.opts))?;
        }
    }

    if let Some(st) = unsafe {
        api.add_session_config_entry(
            so_guard.opts,
            c"session.intra_op.allow_spinning".as_ptr(),
            c"0".as_ptr(),
        )
    } {
        if !st.is_null() {
            let err = status_to_error(api, st);
            log::debug!("session.intra_op.allow_spinning=0 ignored: {}", err);
        }
    }

    append_execution_provider(runtime, so_guard.opts, config, provider)?;

    create_session_checked(api, runtime, model_path, so_guard.opts)
}

fn create_session_checked(
    api: &crate::ffi::OrtApiTable,
    runtime: &OrtRuntime,
    model_path: &Path,
    opts: *mut OrtSessionOptions,
) -> Result<*mut RawOrtSession, OnnxError> {
    let mut session: *mut RawOrtSession = std::ptr::null_mut();

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = model_path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let st = unsafe { api.create_session(runtime.env(), wide.as_ptr(), opts, &mut session) };
        runtime.check_status(st)?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        use std::os::unix::ffi::OsStrExt;
        let cpath = CString::new(model_path.as_os_str().as_bytes()).map_err(|_| {
            OnnxError::InvalidModel {
                path: model_path.to_path_buf(),
                message: "model path contains interior NUL byte".to_string(),
            }
        })?;
        let st = unsafe { api.create_session(runtime.env(), cpath.as_ptr(), opts, &mut session) };
        runtime.check_status(st)?;
    }

    if session.is_null() {
        return Err(OnnxError::InvalidModel {
            path: model_path.to_path_buf(),
            message: "CreateSession returned NULL OrtSession".to_string(),
        });
    }
    Ok(session)
}

fn append_execution_provider(
    runtime: &OrtRuntime,
    so: *mut OrtSessionOptions,
    config: &SessionConfig,
    provider: &ExecutionProvider,
) -> Result<(), OnnxError> {
    let api = runtime.api();
    match provider {
        ExecutionProvider::Cpu => {
            let _ = config;
            Ok(())
        }
        ExecutionProvider::DirectMl => {
            #[cfg(feature = "directml")]
            {
                use crate::ffi::AppendProviderDeviceIdFn;

                let mut keys = Vec::new();
                let mut vals = Vec::new();
                if config.disable_dml_metacommands {
                    keys.push(c"disable_metacommands".as_ptr());
                    vals.push(c"true".as_ptr());
                }
                if let Some(st) = unsafe {
                    api.session_options_append_execution_provider(
                        so,
                        c"DML".as_ptr(),
                        keys.as_ptr(),
                        vals.as_ptr(),
                        keys.len(),
                    )
                } {
                    return runtime.check_status(st);
                }

                let sym = runtime
                    .lib()
                    .symbol("OrtSessionOptionsAppendExecutionProvider_DML")?;
                let append_dml: AppendProviderDeviceIdFn = unsafe { std::mem::transmute(sym) };
                let st = unsafe { append_dml(so, config.device_id) };
                runtime.check_status(st)
            }
            #[cfg(not(feature = "directml"))]
            {
                let _ = (so, config);
                Err(OnnxError::InvalidInput {
                    message: "feature `directml` is disabled".to_string(),
                })
            }
        }
        ExecutionProvider::Rocm => {
            #[cfg(feature = "rocm")]
            {
                use crate::ffi::{AppendProviderDeviceIdFn, OrtROCMProviderOptions};

                // 1. Explicit exported C symbol (`OrtSessionOptionsAppendExecutionProvider_ROCM`) when present.
                if let Ok(sym) = runtime
                    .lib()
                    .symbol("OrtSessionOptionsAppendExecutionProvider_ROCM")
                {
                    let append_rocm: AppendProviderDeviceIdFn = unsafe { std::mem::transmute(sym) };
                    let st = unsafe { append_rocm(so, config.device_id) };
                    if st.is_null() {
                        return Ok(());
                    }
                    let err = status_to_error(api, st);
                    log::debug!(
                        "OrtSessionOptionsAppendExecutionProvider_ROCM symbol failed: {}",
                        err
                    );
                }

                // 2. Unified `SessionOptionsAppendExecutionProvider` (`"ROCM"`).
                if let Some(st) = unsafe {
                    api.session_options_append_execution_provider(
                        so,
                        c"ROCM".as_ptr(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                    )
                } {
                    if st.is_null() {
                        return Ok(());
                    }
                    let err = status_to_error(api, st);
                    log::debug!(
                        "SessionOptionsAppendExecutionProvider(\"ROCM\") failed: {}",
                        err
                    );
                }

                // 3. `OrtApi::SessionOptionsAppendExecutionProvider_ROCM` (slot 153).
                let rocm_opts = OrtROCMProviderOptions {
                    device_id: config.device_id,
                    ..Default::default()
                };
                if let Some(st) =
                    unsafe { api.session_options_append_execution_provider_rocm(so, &rocm_opts) }
                {
                    return runtime.check_status(st);
                }

                Err(OnnxError::InvalidInput {
                    message: "ROCMExecutionProvider entrypoints unavailable".to_string(),
                })
            }
            #[cfg(not(feature = "rocm"))]
            {
                let _ = (so, config);
                Err(OnnxError::InvalidInput {
                    message: "feature `rocm` is disabled".to_string(),
                })
            }
        }
        ExecutionProvider::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use crate::ffi::{AppendProviderDeviceIdFn, OrtCUDAProviderOptions};

                // 1. Try V2 options API when available.
                let mut cuda_v2 = std::ptr::null_mut();
                if let Some(st) = unsafe { api.create_cuda_provider_options(&mut cuda_v2) } {
                    if st.is_null() && !cuda_v2.is_null() {
                        let _guard = CudaV2OptsGuard {
                            api: *api,
                            opts: cuda_v2,
                        };
                        if let Some(append_st) = unsafe {
                            api.session_options_append_execution_provider_cuda_v2(so, cuda_v2)
                        } {
                            if append_st.is_null() {
                                return Ok(());
                            }
                            let _ = status_to_error(api, append_st);
                        }
                    } else if !st.is_null() {
                        let _ = status_to_error(api, st);
                    }
                }

                // 2. Fallback to legacy `SessionOptionsAppendExecutionProvider_CUDA` (slot 152).
                let cuda_opts = OrtCUDAProviderOptions {
                    device_id: config.device_id,
                    ..Default::default()
                };
                if let Some(st) =
                    unsafe { api.session_options_append_execution_provider_cuda(so, &cuda_opts) }
                {
                    if st.is_null() {
                        return Ok(());
                    }
                    return runtime.check_status(st);
                }

                // 3. Direct symbol lookup fallback.
                let sym = runtime
                    .lib()
                    .symbol("OrtSessionOptionsAppendExecutionProvider_Cuda")?;
                let append_cuda: AppendProviderDeviceIdFn = unsafe { std::mem::transmute(sym) };
                let st = unsafe { append_cuda(so, config.device_id) };
                runtime.check_status(st)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (so, config);
                Err(OnnxError::InvalidInput {
                    message: "feature `cuda` is disabled".to_string(),
                })
            }
        }
        ExecutionProvider::CoreMl => {
            #[cfg(feature = "coreml")]
            {
                use crate::ffi::AppendProviderFlagsFn;

                if let Some(st) = unsafe {
                    api.session_options_append_execution_provider(
                        so,
                        c"CoreML".as_ptr(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                    )
                } {
                    if st.is_null() {
                        return Ok(());
                    }
                    let err = status_to_error(api, st);
                    log::debug!(
                        "SessionOptionsAppendExecutionProvider(\"CoreML\") failed: {}",
                        err
                    );
                }

                let sym = runtime
                    .lib()
                    .symbol("OrtSessionOptionsAppendExecutionProvider_CoreML")?;
                let append_coreml: AppendProviderFlagsFn = unsafe { std::mem::transmute(sym) };
                let st = unsafe { append_coreml(so, 0) };
                runtime.check_status(st)
            }
            #[cfg(not(feature = "coreml"))]
            {
                let _ = (so, config);
                Err(OnnxError::InvalidInput {
                    message: "feature `coreml` is disabled".to_string(),
                })
            }
        }
        ExecutionProvider::MigraphX => {
            #[cfg(feature = "migraphx")]
            {
                if let Some(st) = unsafe {
                    api.session_options_append_execution_provider(
                        so,
                        c"MIGraphX".as_ptr(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                    )
                } {
                    return runtime.check_status(st);
                }
                Err(OnnxError::InvalidInput {
                    message: "MIGraphXExecutionProvider requires ORT API v12+".to_string(),
                })
            }
            #[cfg(not(feature = "migraphx"))]
            {
                let _ = (so, config);
                Err(OnnxError::InvalidInput {
                    message: "feature `migraphx` is disabled".to_string(),
                })
            }
        }
        ExecutionProvider::Other(name) => {
            let cname = CString::new(name.as_str()).map_err(|_| OnnxError::InvalidInput {
                message: format!("provider name '{name}' contains interior NUL"),
            })?;
            if let Some(st) = unsafe {
                api.session_options_append_execution_provider(
                    so,
                    cname.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                )
            } {
                runtime.check_status(st)
            } else {
                Err(OnnxError::InvalidInput {
                    message: format!("cannot append custom provider '{name}' on ORT API < v12"),
                })
            }
        }
    }
}

fn checked_shape_product(dims: &[i64]) -> Result<usize, OnnxError> {
    let mut acc: usize = 1;
    for &d in dims {
        if d < 0 {
            return Err(OnnxError::ShapeMismatch {
                expected: "non-negative concrete dimensions".to_string(),
                actual: format!("{dims:?}"),
            });
        }
        acc = acc
            .checked_mul(d as usize)
            .ok_or_else(|| OnnxError::InvalidInput {
                message: format!("tensor shape {dims:?} overflows usize"),
            })?;
    }
    Ok(acc)
}

struct SessionOptionsGuard {
    api: crate::ffi::OrtApiTable,
    opts: *mut OrtSessionOptions,
}

impl Drop for SessionOptionsGuard {
    fn drop(&mut self) {
        if !self.opts.is_null() {
            unsafe { self.api.release_session_options(self.opts) };
        }
    }
}

struct OrtValueGuard {
    api: crate::ffi::OrtApiTable,
    val: *mut OrtValue,
}

impl Drop for OrtValueGuard {
    fn drop(&mut self) {
        if !self.val.is_null() {
            unsafe { self.api.release_value(self.val) };
        }
    }
}

struct TypeInfoGuard {
    api: crate::ffi::OrtApiTable,
    info: *mut OrtTypeInfo,
}

impl Drop for TypeInfoGuard {
    fn drop(&mut self) {
        if !self.info.is_null() {
            unsafe { self.api.release_type_info(self.info) };
        }
    }
}

struct TensorShapeInfoGuard {
    api: crate::ffi::OrtApiTable,
    info: *mut OrtTensorTypeAndShapeInfo,
}

impl Drop for TensorShapeInfoGuard {
    fn drop(&mut self) {
        if !self.info.is_null() {
            unsafe { self.api.release_tensor_type_and_shape_info(self.info) };
        }
    }
}

#[cfg(feature = "cuda")]
struct CudaV2OptsGuard {
    api: crate::ffi::OrtApiTable,
    opts: *mut crate::ffi::OrtCUDAProviderOptionsV2,
}

#[cfg(feature = "cuda")]
impl Drop for CudaV2OptsGuard {
    fn drop(&mut self) {
        if !self.opts.is_null() {
            unsafe { self.api.release_cuda_provider_options(self.opts) };
        }
    }
}
