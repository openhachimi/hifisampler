#![allow(non_camel_case_types, dead_code)]

use std::ffi::{c_char, c_int, c_void};

pub type OrtEnv = c_void;
pub type OrtStatus = c_void;
pub type OrtMemoryInfo = c_void;
pub type OrtSession = c_void;
pub type OrtSessionOptions = c_void;
pub type OrtValue = c_void;
pub type OrtRunOptions = c_void;
pub type OrtTypeInfo = c_void;
pub type OrtTensorTypeAndShapeInfo = c_void;
pub type OrtAllocator = c_void;
pub type OrtArenaCfg = c_void;
pub type OrtCUDAProviderOptionsV2 = c_void;

#[cfg(target_os = "windows")]
pub type OrtChar = u16;
#[cfg(not(target_os = "windows"))]
pub type OrtChar = c_char;

pub const ORT_LOGGING_LEVEL_VERBOSE: c_int = 0;
pub const ORT_LOGGING_LEVEL_INFO: c_int = 1;
pub const ORT_LOGGING_LEVEL_WARNING: c_int = 2;
pub const ORT_LOGGING_LEVEL_ERROR: c_int = 3;
pub const ORT_LOGGING_LEVEL_FATAL: c_int = 4;

pub const ORT_SEQUENTIAL: c_int = 0;
pub const ORT_PARALLEL: c_int = 1;

pub const ORT_ENABLE_ALL: c_int = 99;

pub const ORT_ARENA_ALLOCATOR: c_int = 1;
pub const ORT_MEM_TYPE_DEFAULT: c_int = 0;

pub const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: c_int = 1;
pub const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: c_int = 10;

/// Returned by `OrtGetApiBase()`.
#[repr(C)]
pub struct OrtApiBase {
    pub get_api: unsafe extern "C" fn(version: u32) -> *const *const c_void,
    pub get_version_string: unsafe extern "C" fn() -> *const c_char,
}

pub type OrtGetApiBaseFn = unsafe extern "C" fn() -> *const OrtApiBase;

/// Legacy CUDA provider options struct passed to `SessionOptionsAppendExecutionProvider_CUDA` (slot 152).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrtCUDAProviderOptions {
    pub device_id: c_int,
    pub cudnn_conv_algo_search: c_int,
    pub gpu_mem_limit: usize,
    pub arena_extend_strategy: c_int,
    pub do_copy_in_default_stream: c_int,
    pub has_user_compute_stream: c_int,
    pub user_compute_stream: *mut c_void,
    pub default_memory_arena_cfg: *mut OrtArenaCfg,
    pub tunable_op_enable: c_int,
    pub tunable_op_tuning_enable: c_int,
    pub tunable_op_max_tuning_duration_ms: c_int,
}

impl Default for OrtCUDAProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            cudnn_conv_algo_search: 0,
            gpu_mem_limit: usize::MAX,
            arena_extend_strategy: 0,
            do_copy_in_default_stream: 1,
            has_user_compute_stream: 0,
            user_compute_stream: std::ptr::null_mut(),
            default_memory_arena_cfg: std::ptr::null_mut(),
            tunable_op_enable: 0,
            tunable_op_tuning_enable: 0,
            tunable_op_max_tuning_duration_ms: 0,
        }
    }
}

/// Legacy ROCm provider options struct passed to `SessionOptionsAppendExecutionProvider_ROCM` (slot 153).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OrtROCMProviderOptions {
    pub device_id: c_int,
    pub miopen_conv_exhaustive_search: c_int,
    pub gpu_mem_limit: usize,
    pub arena_extend_strategy: c_int,
    pub do_copy_in_default_stream: c_int,
    pub has_user_compute_stream: c_int,
    pub user_compute_stream: *mut c_void,
    pub default_memory_arena_cfg: *mut OrtArenaCfg,
    pub tunable_op_enable: c_int,
    pub tunable_op_tuning_enable: c_int,
    pub tunable_op_max_tuning_duration_ms: c_int,
}

impl Default for OrtROCMProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            miopen_conv_exhaustive_search: 0,
            gpu_mem_limit: usize::MAX,
            arena_extend_strategy: 0,
            do_copy_in_default_stream: 1,
            has_user_compute_stream: 0,
            user_compute_stream: std::ptr::null_mut(),
            default_memory_arena_cfg: std::ptr::null_mut(),
            tunable_op_enable: 0,
            tunable_op_tuning_enable: 0,
            tunable_op_max_tuning_duration_ms: 0,
        }
    }
}

/// Direct C-symbol signature exported by ORT EP factories (`OrtSessionOptionsAppendExecutionProvider_{DML,ROCM,Cuda,CoreML}`).
pub type AppendProviderDeviceIdFn =
    unsafe extern "C" fn(options: *mut OrtSessionOptions, device_id: c_int) -> *mut OrtStatus;
pub type AppendProviderFlagsFn =
    unsafe extern "C" fn(options: *mut OrtSessionOptions, flags: u32) -> *mut OrtStatus;

/// Verified `OrtApi` function-pointer table wrapper.
///
/// Each slot index matches `onnxruntime_c_api.h`'s `struct OrtApi` layout (verified against ORT 1.24.4).
#[derive(Clone, Copy)]
pub struct OrtApiTable {
    table: *const *const c_void,
    pub api_version: u32,
}

// Safety: `OrtApi` table lives in `.rdata` of the loaded ORT library for its entire lifetime.
unsafe impl Send for OrtApiTable {}
unsafe impl Sync for OrtApiTable {}

impl OrtApiTable {
    pub(crate) unsafe fn from_raw(table: *const *const c_void, api_version: u32) -> Self {
        Self { table, api_version }
    }

    #[inline]
    unsafe fn slot<T: Copy>(&self, index: usize) -> T {
        debug_assert!(!self.table.is_null());
        let ptr = *self.table.add(index);
        std::mem::transmute_copy(&ptr)
    }

    #[inline]
    pub(crate) unsafe fn get_error_code(&self, status: *const OrtStatus) -> c_int {
        let f: unsafe extern "C" fn(*const OrtStatus) -> c_int = self.slot(1);
        f(status)
    }

    #[inline]
    pub(crate) unsafe fn get_error_message(&self, status: *const OrtStatus) -> *const c_char {
        let f: unsafe extern "C" fn(*const OrtStatus) -> *const c_char = self.slot(2);
        f(status)
    }

    #[inline]
    pub(crate) unsafe fn create_env(
        &self,
        log_severity: c_int,
        log_id: *const c_char,
        out: *mut *mut OrtEnv,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(c_int, *const c_char, *mut *mut OrtEnv) -> *mut OrtStatus =
            self.slot(3);
        f(log_severity, log_id, out)
    }

    #[inline]
    pub(crate) unsafe fn create_session(
        &self,
        env: *const OrtEnv,
        model_path: *const OrtChar,
        options: *const OrtSessionOptions,
        out: *mut *mut OrtSession,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtEnv,
            *const OrtChar,
            *const OrtSessionOptions,
            *mut *mut OrtSession,
        ) -> *mut OrtStatus = self.slot(7);
        f(env, model_path, options, out)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn run(
        &self,
        session: *mut OrtSession,
        run_options: *const OrtRunOptions,
        input_names: *const *const c_char,
        inputs: *const *const OrtValue,
        input_len: usize,
        output_names: *const *const c_char,
        output_names_len: usize,
        outputs: *mut *mut OrtValue,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *mut OrtSession,
            *const OrtRunOptions,
            *const *const c_char,
            *const *const OrtValue,
            usize,
            *const *const c_char,
            usize,
            *mut *mut OrtValue,
        ) -> *mut OrtStatus = self.slot(9);
        f(
            session,
            run_options,
            input_names,
            inputs,
            input_len,
            output_names,
            output_names_len,
            outputs,
        )
    }

    #[inline]
    pub(crate) unsafe fn create_session_options(
        &self,
        out: *mut *mut OrtSessionOptions,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut *mut OrtSessionOptions) -> *mut OrtStatus = self.slot(10);
        f(out)
    }

    #[inline]
    pub(crate) unsafe fn set_session_execution_mode(
        &self,
        options: *mut OrtSessionOptions,
        mode: c_int,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions, c_int) -> *mut OrtStatus =
            self.slot(13);
        f(options, mode)
    }

    #[inline]
    pub(crate) unsafe fn enable_mem_pattern(
        &self,
        options: *mut OrtSessionOptions,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions) -> *mut OrtStatus = self.slot(16);
        f(options)
    }

    #[inline]
    pub(crate) unsafe fn disable_mem_pattern(
        &self,
        options: *mut OrtSessionOptions,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions) -> *mut OrtStatus = self.slot(17);
        f(options)
    }

    #[inline]
    pub(crate) unsafe fn enable_cpu_mem_arena(
        &self,
        options: *mut OrtSessionOptions,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions) -> *mut OrtStatus = self.slot(18);
        f(options)
    }

    #[inline]
    pub(crate) unsafe fn disable_cpu_mem_arena(
        &self,
        options: *mut OrtSessionOptions,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions) -> *mut OrtStatus = self.slot(19);
        f(options)
    }

    #[inline]
    pub(crate) unsafe fn set_session_graph_optimization_level(
        &self,
        options: *mut OrtSessionOptions,
        level: c_int,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions, c_int) -> *mut OrtStatus =
            self.slot(23);
        f(options, level)
    }

    #[inline]
    pub(crate) unsafe fn set_intra_op_num_threads(
        &self,
        options: *mut OrtSessionOptions,
        threads: c_int,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions, c_int) -> *mut OrtStatus =
            self.slot(24);
        f(options, threads)
    }

    #[inline]
    pub(crate) unsafe fn set_inter_op_num_threads(
        &self,
        options: *mut OrtSessionOptions,
        threads: c_int,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions, c_int) -> *mut OrtStatus =
            self.slot(25);
        f(options, threads)
    }

    #[inline]
    pub(crate) unsafe fn session_get_input_count(
        &self,
        session: *const OrtSession,
        out: *mut usize,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*const OrtSession, *mut usize) -> *mut OrtStatus =
            self.slot(30);
        f(session, out)
    }

    #[inline]
    pub(crate) unsafe fn session_get_output_count(
        &self,
        session: *const OrtSession,
        out: *mut usize,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*const OrtSession, *mut usize) -> *mut OrtStatus =
            self.slot(31);
        f(session, out)
    }

    #[inline]
    pub(crate) unsafe fn session_get_input_type_info(
        &self,
        session: *const OrtSession,
        index: usize,
        out: *mut *mut OrtTypeInfo,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtSession,
            usize,
            *mut *mut OrtTypeInfo,
        ) -> *mut OrtStatus = self.slot(33);
        f(session, index, out)
    }

    #[inline]
    pub(crate) unsafe fn session_get_output_type_info(
        &self,
        session: *const OrtSession,
        index: usize,
        out: *mut *mut OrtTypeInfo,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtSession,
            usize,
            *mut *mut OrtTypeInfo,
        ) -> *mut OrtStatus = self.slot(34);
        f(session, index, out)
    }

    #[inline]
    pub(crate) unsafe fn session_get_input_name(
        &self,
        session: *const OrtSession,
        index: usize,
        allocator: *mut OrtAllocator,
        out: *mut *mut c_char,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtSession,
            usize,
            *mut OrtAllocator,
            *mut *mut c_char,
        ) -> *mut OrtStatus = self.slot(36);
        f(session, index, allocator, out)
    }

    #[inline]
    pub(crate) unsafe fn session_get_output_name(
        &self,
        session: *const OrtSession,
        index: usize,
        allocator: *mut OrtAllocator,
        out: *mut *mut c_char,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtSession,
            usize,
            *mut OrtAllocator,
            *mut *mut c_char,
        ) -> *mut OrtStatus = self.slot(37);
        f(session, index, allocator, out)
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn create_tensor_with_data_as_ort_value(
        &self,
        info: *const OrtMemoryInfo,
        p_data: *mut c_void,
        p_data_len: usize,
        shape: *const i64,
        shape_len: usize,
        elem_type: c_int,
        out: *mut *mut OrtValue,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtMemoryInfo,
            *mut c_void,
            usize,
            *const i64,
            usize,
            c_int,
            *mut *mut OrtValue,
        ) -> *mut OrtStatus = self.slot(49);
        f(info, p_data, p_data_len, shape, shape_len, elem_type, out)
    }

    #[inline]
    pub(crate) unsafe fn is_tensor(
        &self,
        value: *const OrtValue,
        out: *mut c_int,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*const OrtValue, *mut c_int) -> *mut OrtStatus = self.slot(50);
        f(value, out)
    }

    #[inline]
    pub(crate) unsafe fn get_tensor_mutable_data(
        &self,
        value: *mut OrtValue,
        out: *mut *mut c_void,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtValue, *mut *mut c_void) -> *mut OrtStatus =
            self.slot(51);
        f(value, out)
    }

    #[inline]
    pub(crate) unsafe fn cast_type_info_to_tensor_info(
        &self,
        type_info: *const OrtTypeInfo,
        out: *mut *const OrtTensorTypeAndShapeInfo,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtTypeInfo,
            *mut *const OrtTensorTypeAndShapeInfo,
        ) -> *mut OrtStatus = self.slot(55);
        f(type_info, out)
    }

    #[inline]
    pub(crate) unsafe fn get_tensor_element_type(
        &self,
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut c_int,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtTensorTypeAndShapeInfo,
            *mut c_int,
        ) -> *mut OrtStatus = self.slot(60);
        f(info, out)
    }

    #[inline]
    pub(crate) unsafe fn get_dimensions_count(
        &self,
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtTensorTypeAndShapeInfo,
            *mut usize,
        ) -> *mut OrtStatus = self.slot(61);
        f(info, out)
    }

    #[inline]
    pub(crate) unsafe fn get_dimensions(
        &self,
        info: *const OrtTensorTypeAndShapeInfo,
        dims: *mut i64,
        dims_len: usize,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtTensorTypeAndShapeInfo,
            *mut i64,
            usize,
        ) -> *mut OrtStatus = self.slot(62);
        f(info, dims, dims_len)
    }

    #[inline]
    pub(crate) unsafe fn get_tensor_shape_element_count(
        &self,
        info: *const OrtTensorTypeAndShapeInfo,
        out: *mut usize,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtTensorTypeAndShapeInfo,
            *mut usize,
        ) -> *mut OrtStatus = self.slot(64);
        f(info, out)
    }

    #[inline]
    pub(crate) unsafe fn get_tensor_type_and_shape(
        &self,
        value: *const OrtValue,
        out: *mut *mut OrtTensorTypeAndShapeInfo,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(
            *const OrtValue,
            *mut *mut OrtTensorTypeAndShapeInfo,
        ) -> *mut OrtStatus = self.slot(65);
        f(value, out)
    }

    #[inline]
    pub(crate) unsafe fn create_cpu_memory_info(
        &self,
        alloc_type: c_int,
        mem_type: c_int,
        out: *mut *mut OrtMemoryInfo,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(c_int, c_int, *mut *mut OrtMemoryInfo) -> *mut OrtStatus =
            self.slot(69);
        f(alloc_type, mem_type, out)
    }

    #[inline]
    pub(crate) unsafe fn allocator_free(
        &self,
        allocator: *mut OrtAllocator,
        p: *mut c_void,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut OrtAllocator, *mut c_void) -> *mut OrtStatus =
            self.slot(76);
        f(allocator, p)
    }

    #[inline]
    pub(crate) unsafe fn get_allocator_with_default_options(
        &self,
        out: *mut *mut OrtAllocator,
    ) -> *mut OrtStatus {
        let f: unsafe extern "C" fn(*mut *mut OrtAllocator) -> *mut OrtStatus = self.slot(78);
        f(out)
    }

    #[inline]
    pub(crate) unsafe fn release_env(&self, env: *mut OrtEnv) {
        let f: unsafe extern "C" fn(*mut OrtEnv) = self.slot(92);
        f(env);
    }

    #[inline]
    pub(crate) unsafe fn release_status(&self, status: *mut OrtStatus) {
        let f: unsafe extern "C" fn(*mut OrtStatus) = self.slot(93);
        f(status);
    }

    #[inline]
    pub(crate) unsafe fn release_memory_info(&self, info: *mut OrtMemoryInfo) {
        let f: unsafe extern "C" fn(*mut OrtMemoryInfo) = self.slot(94);
        f(info);
    }

    #[inline]
    pub(crate) unsafe fn release_session(&self, session: *mut OrtSession) {
        let f: unsafe extern "C" fn(*mut OrtSession) = self.slot(95);
        f(session);
    }

    #[inline]
    pub(crate) unsafe fn release_value(&self, value: *mut OrtValue) {
        let f: unsafe extern "C" fn(*mut OrtValue) = self.slot(96);
        f(value);
    }

    #[inline]
    pub(crate) unsafe fn release_type_info(&self, info: *mut OrtTypeInfo) {
        let f: unsafe extern "C" fn(*mut OrtTypeInfo) = self.slot(98);
        f(info);
    }

    #[inline]
    pub(crate) unsafe fn release_tensor_type_and_shape_info(
        &self,
        info: *mut OrtTensorTypeAndShapeInfo,
    ) {
        let f: unsafe extern "C" fn(*mut OrtTensorTypeAndShapeInfo) = self.slot(99);
        f(info);
    }

    #[inline]
    pub(crate) unsafe fn release_session_options(&self, options: *mut OrtSessionOptions) {
        let f: unsafe extern "C" fn(*mut OrtSessionOptions) = self.slot(100);
        f(options);
    }

    #[inline]
    pub(crate) unsafe fn get_available_providers(
        &self,
        out_ptr: *mut *mut *mut c_char,
        len: *mut c_int,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 6 {
            return None;
        }
        let ptr = *self.table.add(125);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(*mut *mut *mut c_char, *mut c_int) -> *mut OrtStatus =
            std::mem::transmute_copy(&ptr);
        Some(f(out_ptr, len))
    }

    #[inline]
    pub(crate) unsafe fn release_available_providers(&self, ptr: *mut *mut c_char, len: c_int) {
        if self.api_version < 6 || ptr.is_null() {
            return;
        }
        let slot_ptr = *self.table.add(126);
        if !slot_ptr.is_null() {
            let f: unsafe extern "C" fn(*mut *mut c_char, c_int) -> *mut OrtStatus =
                std::mem::transmute_copy(&slot_ptr);
            let st = f(ptr, len);
            if !st.is_null() {
                self.release_status(st);
            }
        }
    }

    #[inline]
    pub(crate) unsafe fn add_session_config_entry(
        &self,
        options: *mut OrtSessionOptions,
        key: *const c_char,
        value: *const c_char,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 6 {
            return None;
        }
        let ptr = *self.table.add(130);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(
            *mut OrtSessionOptions,
            *const c_char,
            *const c_char,
        ) -> *mut OrtStatus = std::mem::transmute_copy(&ptr);
        Some(f(options, key, value))
    }

    #[inline]
    pub(crate) unsafe fn session_options_append_execution_provider_cuda(
        &self,
        options: *mut OrtSessionOptions,
        cuda_opts: *const OrtCUDAProviderOptions,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 7 {
            return None;
        }
        let ptr = *self.table.add(152);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(
            *mut OrtSessionOptions,
            *const OrtCUDAProviderOptions,
        ) -> *mut OrtStatus = std::mem::transmute_copy(&ptr);
        Some(f(options, cuda_opts))
    }

    #[inline]
    pub(crate) unsafe fn session_options_append_execution_provider_rocm(
        &self,
        options: *mut OrtSessionOptions,
        rocm_opts: *const OrtROCMProviderOptions,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 7 {
            return None;
        }
        let ptr = *self.table.add(153);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(
            *mut OrtSessionOptions,
            *const OrtROCMProviderOptions,
        ) -> *mut OrtStatus = std::mem::transmute_copy(&ptr);
        Some(f(options, rocm_opts))
    }

    #[inline]
    pub(crate) unsafe fn session_options_append_execution_provider_cuda_v2(
        &self,
        options: *mut OrtSessionOptions,
        cuda_opts: *const OrtCUDAProviderOptionsV2,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 11 {
            return None;
        }
        let ptr = *self.table.add(204);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(
            *mut OrtSessionOptions,
            *const OrtCUDAProviderOptionsV2,
        ) -> *mut OrtStatus = std::mem::transmute_copy(&ptr);
        Some(f(options, cuda_opts))
    }

    #[inline]
    pub(crate) unsafe fn create_cuda_provider_options(
        &self,
        out: *mut *mut OrtCUDAProviderOptionsV2,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 11 {
            return None;
        }
        let ptr = *self.table.add(205);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(*mut *mut OrtCUDAProviderOptionsV2) -> *mut OrtStatus =
            std::mem::transmute_copy(&ptr);
        Some(f(out))
    }

    #[inline]
    pub(crate) unsafe fn release_cuda_provider_options(&self, opts: *mut OrtCUDAProviderOptionsV2) {
        if self.api_version < 11 || opts.is_null() {
            return;
        }
        let ptr = *self.table.add(208);
        if !ptr.is_null() {
            let f: unsafe extern "C" fn(*mut OrtCUDAProviderOptionsV2) =
                std::mem::transmute_copy(&ptr);
            f(opts);
        }
    }

    #[inline]
    pub(crate) unsafe fn session_options_append_execution_provider(
        &self,
        options: *mut OrtSessionOptions,
        provider_name: *const c_char,
        keys: *const *const c_char,
        values: *const *const c_char,
        num_keys: usize,
    ) -> Option<*mut OrtStatus> {
        if self.api_version < 12 {
            return None;
        }
        let ptr = *self.table.add(216);
        if ptr.is_null() {
            return None;
        }
        let f: unsafe extern "C" fn(
            *mut OrtSessionOptions,
            *const c_char,
            *const *const c_char,
            *const *const c_char,
            usize,
        ) -> *mut OrtStatus = std::mem::transmute_copy(&ptr);
        Some(f(options, provider_name, keys, values, num_keys))
    }
}

/// Convert IEEE-754 `f32` to `f16` bit representation (`u16`), round-to-nearest-even.
pub fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs_bits = bits & 0x7FFF_FFFF;

    if abs_bits >= 0x7F80_0000 {
        // Inf or NaN
        let mant = bits & 0x007F_FFFF;
        return if mant != 0 {
            sign | 0x7E00 | ((mant >> 13) as u16).max(1)
        } else {
            sign | 0x7C00
        };
    }

    if abs_bits < 0x3880_0000 {
        // Subnormal or zero in f16
        if abs_bits < 0x3300_0000 {
            return sign;
        }
        let exp = (abs_bits >> 23) as i32;
        let mant = (abs_bits & 0x007F_FFFF) | 0x0080_0000;
        let shift = (126 - exp) as u32;
        let round_bit = 1u32 << (shift - 1);
        let rounded = mant + round_bit - 1 + ((mant >> shift) & 1);
        return sign | ((rounded >> shift) as u16);
    }

    let exp = ((abs_bits >> 23) as i32) - 112;
    if exp >= 31 {
        return sign | 0x7C00;
    }
    let mant = abs_bits & 0x007F_FFFF;
    let rounded_mant = mant + 0x0000_0FFF + ((mant >> 13) & 1);
    let final_exp = exp + ((rounded_mant >> 23) as i32);
    if final_exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((final_exp as u16) << 10) | (((rounded_mant >> 13) & 0x03FF) as u16)
}

/// Convert IEEE-754 `f16` bit representation (`u16`) to `f32`.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    let out_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e = 113u32;
            let mut m = frac;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (frac << 13)
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(out_bits)
}
