use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use hifi_onnx::{
    resolve_hnsep_path, resolve_ort_library_candidates, resolve_ort_library_path,
    resolve_vocoder_path, DevicePreference, ExecutionProvider, HnsepSession, OrtRuntime,
    SessionConfig, VocoderSession,
};
use ndarray::{Array2, Array3, Array4};

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("crate lives under rust/crates/hifi_onnx")
        .to_path_buf()
}

fn real_vocoder_onnx_path() -> PathBuf {
    workspace_root()
        .join("pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
        .join("model.onnx")
}

fn real_hnsep_pt_path() -> PathBuf {
    workspace_root().join("hnsep").join("vr").join("model.pt")
}

fn real_hnsep_onnx_path() -> PathBuf {
    workspace_root().join("hnsep").join("vr").join("model.onnx")
}

#[test]
fn resolver_finds_real_ort_library() {
    let resolved = resolve_ort_library_path(None).expect("ORT shared library must be found");
    assert!(
        resolved.is_file(),
        "resolved ORT path does not exist: {}",
        resolved.display()
    );
    let rt = OrtRuntime::global().expect("OrtRuntime::global must succeed");
    assert!(rt.api_version() >= 12);
    assert!(rt
        .available_providers()
        .iter()
        .any(|p| p == "CPUExecutionProvider"));
}

#[test]
fn resolver_honors_explicit_path_first() {
    let tmp = tempfile::tempdir().unwrap();
    let fake_dll = tmp.path().join("custom_ort.dll");
    std::fs::write(&fake_dll, b"stub").unwrap();
    let candidates = resolve_ort_library_candidates(Some(&fake_dll));
    assert_eq!(candidates.first(), Some(&fake_dll));
}

#[test]
fn path_resolvers_map_ckpt_and_pt_to_sibling_onnx() {
    let tmp = tempfile::tempdir().unwrap();
    let ckpt = tmp.path().join("model.ckpt");
    let voc_onnx = tmp.path().join("model.onnx");
    std::fs::write(&ckpt, b"pt").unwrap();
    std::fs::write(&voc_onnx, b"onnx").unwrap();
    assert_eq!(resolve_vocoder_path(&ckpt).unwrap(), voc_onnx);

    let pt = tmp.path().join("sep.pt");
    let sep_onnx = tmp.path().join("sep.onnx");
    std::fs::write(&pt, b"pt").unwrap();
    std::fs::write(&sep_onnx, b"onnx").unwrap();
    assert_eq!(resolve_hnsep_path(&pt).unwrap(), sep_onnx);

    // Real workspace `.pt` resolves to sibling `hnsep/vr/model.onnx`.
    let real_resolved = resolve_hnsep_path(real_hnsep_pt_path()).unwrap();
    assert_eq!(
        std::fs::canonicalize(real_resolved).unwrap(),
        std::fs::canonicalize(real_hnsep_onnx_path()).unwrap()
    );
}

#[test]
fn vocoder_cpu_inference_and_dynamic_shapes() {
    let vocoder_path = real_vocoder_onnx_path();
    assert!(vocoder_path.is_file(), "missing {}", vocoder_path.display());

    let vocoder = VocoderSession::load(&vocoder_path, DevicePreference::Cpu)
        .expect("CPU VocoderSession load must succeed");
    assert_eq!(vocoder.execution_provider(), &ExecutionProvider::Cpu);
    assert!(!vocoder.is_gpu_ep());
    assert!(vocoder.fallback_reason().is_none());
    assert_eq!(vocoder.n_mels(), 128);

    vocoder.warmup().expect("vocoder warmup must succeed");

    for frames in [32usize, 47usize] {
        let mut mel = Array3::<f32>::from_elem((1, frames, 128), -6.0);
        for t in 0..frames {
            for m in 0..16 {
                mel[[0, t, m]] = -2.0 + 0.05 * (t as f32).sin();
            }
        }
        let f0 = Array2::<f32>::from_elem((1, frames), 220.0);
        let wav = vocoder
            .synthesize(mel.view(), f0.view())
            .expect("vocoder synthesize must succeed");
        assert_eq!(wav.len(), frames * 512);
        assert!(wav.iter().all(|x| x.is_finite()));
    }

    // Also verify `[128, T]` helper matching `hifi_dsp` layout and timed CPU zero wait.
    let mel_mels_by_frames = Array2::<f32>::from_elem((128, 24), -5.0);
    let f0_slice = vec![330.0f32; 24];
    let (wav2, wait_ms) = vocoder
        .synthesize_mels_by_frames_timed(mel_mels_by_frames.view(), &f0_slice)
        .expect("synthesize_mels_by_frames_timed must succeed");
    assert_eq!(wav2.len(), 24 * 512);
    assert!(wav2.iter().all(|x| x.is_finite()));
    assert_eq!(wait_ms, 0.0);

    // Ensure opt-in `enable_cpu_mem_arena = true` toggle remains functional.
    let mut arena_cfg = SessionConfig::with_device(DevicePreference::Cpu);
    arena_cfg.enable_cpu_mem_arena = true;
    let arena_vocoder = VocoderSession::load_with_config(&vocoder_path, &arena_cfg)
        .expect("Vocoder with explicit enable_cpu_mem_arena=true must succeed");
    let wav_arena = arena_vocoder
        .synthesize_mels_by_frames(mel_mels_by_frames.view(), &f0_slice)
        .unwrap();
    assert_eq!(wav_arena.len(), wav2.len());
}

#[test]
fn vocoder_unavailable_eps_fallback_cleanly_to_cpu() {
    let vocoder_path = real_vocoder_onnx_path();

    // Test providers not present on this Windows DML+CPU runtime: must fall back cleanly without panicking.
    for pref in [
        DevicePreference::Rocm,
        DevicePreference::CoreMl,
        DevicePreference::Cuda,
    ] {
        let runtime = OrtRuntime::global().unwrap();
        let expected_present = match pref {
            DevicePreference::Rocm => {
                cfg!(target_os = "linux")
                    && runtime
                        .available_providers()
                        .iter()
                        .any(|p| p == "ROCMExecutionProvider")
            }
            DevicePreference::CoreMl => {
                cfg!(target_os = "macos")
                    && runtime
                        .available_providers()
                        .iter()
                        .any(|p| p == "CoreMLExecutionProvider")
            }
            DevicePreference::Cuda => runtime
                .available_providers()
                .iter()
                .any(|p| p == "CUDAExecutionProvider"),
            _ => false,
        };

        let vocoder = VocoderSession::load(&vocoder_path, pref)
            .expect("Unavailable EP must fall back to CPU cleanly");
        if !expected_present {
            assert_eq!(vocoder.execution_provider(), &ExecutionProvider::Cpu);
            assert!(!vocoder.is_gpu_ep());
            assert!(
                vocoder.fallback_reason().is_some(),
                "expected fallback_reason when {pref} unavailable"
            );
        }

        let mel = Array3::<f32>::from_elem((1, 20, 128), -5.0);
        let f0 = Array2::<f32>::from_elem((1, 20), 440.0);
        let wav = vocoder.synthesize(mel.view(), f0.view()).unwrap();
        assert_eq!(wav.len(), 20 * 512);
        assert!(wav.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn vocoder_dml_or_auto_matches_cpu_and_handles_dynamic_shapes() {
    let vocoder_path = real_vocoder_onnx_path();
    let cpu = VocoderSession::load(&vocoder_path, DevicePreference::Cpu).unwrap();
    let auto = VocoderSession::load(&vocoder_path, DevicePreference::Auto).unwrap();

    if Path::new(vocoder_path.as_os_str()).is_file() {
        let rt = OrtRuntime::global().unwrap();
        if cfg!(all(target_os = "windows", feature = "directml"))
            && rt
                .available_providers()
                .iter()
                .any(|p| p == "DmlExecutionProvider")
            && auto.fallback_reason().is_none()
        {
            assert_eq!(auto.execution_provider(), &ExecutionProvider::DirectMl);
            assert!(auto.is_gpu_ep());
        }
    }

    for frames in [32usize, 47usize] {
        let mut mel = Array3::<f32>::from_elem((1, frames, 128), -5.5);
        for t in 0..frames {
            for m in 0..20 {
                mel[[0, t, m]] = -2.5 + 0.1 * ((t + m) as f32).cos();
            }
        }
        let f0 = Array2::<f32>::from_elem((1, frames), 261.63);
        let wav_cpu = cpu.synthesize(mel.view(), f0.view()).unwrap();
        let wav_auto = auto.synthesize(mel.view(), f0.view()).unwrap();
        assert_eq!(wav_cpu.len(), wav_auto.len());
        assert!(wav_auto.iter().all(|x| x.is_finite()));

        let max_diff = wav_cpu
            .iter()
            .zip(wav_auto.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-2,
            "CPU vs {} max_diff={} exceeded tolerance at T={}",
            auto.execution_provider(),
            max_diff,
            frames
        );
    }
}

#[test]
fn hnsep_resolves_pt_to_onnx_and_runs_1025_and_1024_tensors() {
    let pt_path = real_hnsep_pt_path();
    assert!(
        real_hnsep_onnx_path().is_file(),
        "missing real hnsep/vr/model.onnx"
    );

    let cpu = HnsepSession::load(&pt_path, DevicePreference::Cpu)
        .expect("HNSEP load via .pt config path must resolve sibling .onnx");
    assert_eq!(cpu.execution_provider(), &ExecutionProvider::Cpu);
    assert_eq!(cpu.config().n_fft, 2048);
    assert_eq!(cpu.config().hop_length, 512);
    assert_eq!(cpu.config().expected_freq_bins, Some(1025));

    cpu.warmup().expect("hnsep warmup must succeed");

    // Native [1, 2, 1025, T] tensor (matching `util/onnx_dsp.py` & `hifi_dsp`).
    let mut spec_1025 = Array4::<f32>::zeros((1, 2, 1025, 32));
    for t in 0..32 {
        spec_1025[[0, 0, 10, t]] = 0.8 * (t as f32 * 0.2).sin();
        spec_1025[[0, 1, 10, t]] = 0.4 * (t as f32 * 0.2).cos();
    }
    let out_1025 = cpu.forward(spec_1025.view()).unwrap();
    assert_eq!(out_1025.dim(), (1, 2, 1025, 32));
    assert!(out_1025.iter().all(|x| x.is_finite()));
    for t in 0..32 {
        for f in 0..1025 {
            let r = out_1025[[0, 0, f, t]];
            let im = out_1025[[0, 1, f, t]];
            let mag = (r * r + im * im).sqrt();
            assert!(mag <= 1.0001, "bounded mask exceeded 1.0: {mag}");
        }
    }

    // Also verify [1, 2, 1024, T] adapter compatibility.
    let spec_1024 = Array4::<f32>::from_elem((1, 2, 1024, 32), 0.1);
    let out_1024 = cpu.forward(spec_1024.view()).unwrap();
    assert_eq!(out_1024.dim(), (1, 2, 1024, 32));
    assert!(out_1024.iter().all(|x| x.is_finite()));

    // Check Auto/DirectML parity if DirectML is active.
    let auto = HnsepSession::load(&pt_path, DevicePreference::Auto).unwrap();
    let out_auto = auto.forward(spec_1025.view()).unwrap();
    assert_eq!(out_auto.dim(), (1, 2, 1025, 32));
    assert!(out_auto.iter().all(|x| x.is_finite()));
    let max_diff = out_1025
        .iter()
        .zip(out_auto.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-2,
        "HNSEP CPU vs {} max_diff={} too large",
        auto.execution_provider(),
        max_diff
    );
}

#[test]
fn concurrent_gpu_or_cpu_sessions_are_thread_safe() {
    let vocoder = Arc::new(
        VocoderSession::load_with_config(
            real_vocoder_onnx_path(),
            &SessionConfig::with_device(DevicePreference::Auto),
        )
        .unwrap(),
    );
    let hnsep = Arc::new(
        HnsepSession::load_with_config(
            real_hnsep_pt_path(),
            &SessionConfig::with_device(DevicePreference::Auto),
        )
        .unwrap(),
    );

    let mut handles = Vec::new();
    for i in 0..4 {
        let v = Arc::clone(&vocoder);
        let h = Arc::clone(&hnsep);
        handles.push(thread::spawn(move || {
            let spec = Array4::<f32>::from_elem((1, 2, 1025, 32), 0.05 * (i + 1) as f32);
            let (mask, h_wait_ms) = h.forward_timed(spec.view()).unwrap();
            assert!(mask.iter().all(|x| x.is_finite()));
            assert!(h_wait_ms >= 0.0 && h_wait_ms.is_finite());

            let mel = Array3::<f32>::from_elem((1, 24, 128), -5.0);
            let f0 = Array2::<f32>::from_elem((1, 24), 220.0 + i as f32 * 20.0);
            let (wav, v_wait_ms) = v.synthesize_timed(mel.view(), f0.view()).unwrap();
            assert_eq!(wav.len(), 24 * 512);
            assert!(wav.iter().all(|x| x.is_finite()));
            assert!(v_wait_ms >= 0.0 && v_wait_ms.is_finite());
        }));
    }
    for handle in handles {
        handle.join().expect("worker thread must not panic");
    }
}
