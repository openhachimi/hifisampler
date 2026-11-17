use axum::body::Body;
use axum::http::{Request, StatusCode};
use hifi_core::audio::{read_wav, write_wav, AudioBuffer, DEFAULT_SAMPLE_RATE};
use hifi_server::config::ServerConfig;
use hifi_server::phrase_mgr::{
    note_sidecar_path, PhraseBatchNoteSpec, PhraseBatchRequest, WavtoolStepRequest,
};
use hifi_server::pipeline::RealResampleEngine;
use hifi_server::resampler::{OnnxRunner, ResampleEngine, ResampleError};
use hifi_server::routes::{build_router, AppState};
use hifi_server::stats::StatsCollector;
use ndarray::{Array4, ArrayView2};
use std::f32::consts::PI;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tower::ServiceExt;
use uuid::Uuid;

struct CountingMockOnnx {
    vocoder_calls: AtomicUsize,
}

impl CountingMockOnnx {
    fn new() -> Self {
        Self {
            vocoder_calls: AtomicUsize::new(0),
        }
    }
}

impl OnnxRunner for CountingMockOnnx {
    fn hnsep_uses_gpu(&self) -> bool {
        false
    }

    fn vocoder_uses_gpu(&self) -> bool {
        false
    }

    fn run_hnsep_mask(&self, stft_tensor: &Array4<f32>) -> Result<Array4<f32>, ResampleError> {
        let (_, _, f, t) = stft_tensor.dim();
        let mut mask = Array4::<f32>::zeros((1, 2, f, t));
        mask.slice_mut(ndarray::s![0, 0, .., ..]).fill(0.8);
        Ok(mask)
    }

    fn run_vocoder(&self, _mel: ArrayView2<f32>, f0: &[f32]) -> Result<Vec<f32>, ResampleError> {
        self.vocoder_calls.fetch_add(1, Ordering::SeqCst);
        let hop = 512usize;
        let sr = 44100.0_f32;
        let total = f0.len().max(1) * hop;
        let mut out = Vec::with_capacity(total);
        let mut phase = 0.0_f32;
        for i in 0..total {
            let hz = f0[(i / hop).min(f0.len().saturating_sub(1))];
            phase += 2.0 * PI * hz / sr;
            out.push(0.25 * phase.sin());
        }
        Ok(out)
    }
}

fn generate_sine_wav(path: &Path, dur_sec: f32) {
    let n = (DEFAULT_SAMPLE_RATE as f32 * dur_sec) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.4 * (2.0 * PI * 440.0 * i as f32 / DEFAULT_SAMPLE_RATE as f32).sin())
        .collect();
    write_wav(path, &AudioBuffer::new(samples, DEFAULT_SAMPLE_RATE, 1)).unwrap();
}

fn env_tokens(ovr_ms: f64) -> Vec<String> {
    [
        "0",
        "25",
        "25",
        "0",
        "100",
        "100",
        "0",
        &ovr_ms.to_string(),
        "0",
        "100",
        "100",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn make_resampler_args(in_wav: &Path, out_wav: &Path, tone: &str, len_ms: &str) -> Vec<String> {
    vec![
        in_wav.to_string_lossy().into_owned(),
        out_wav.to_string_lossy().into_owned(),
        tone.to_string(),
        "100".to_string(),
        "".to_string(),
        "0".to_string(),
        len_ms.to_string(),
        "0".to_string(),
        "0".to_string(),
        "100".to_string(),
        "0".to_string(),
        "!120".to_string(),
        "AA".to_string(),
    ]
}

#[test]
fn eight_note_temp_bat_phrase_uses_single_vocoder_call_and_backfills_stubs() {
    let tmp = std::env::temp_dir().join(format!("hifi_phrase_8n_{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("src.wav");
    generate_sine_wav(&in_wav, 0.5);

    let onnx = Arc::new(CountingMockOnnx::new());
    let engine = RealResampleEngine::new(ServerConfig::default(), onnx.clone()).unwrap();
    let stats = StatsCollector::default();

    let tones = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"];
    let mut note_outs = Vec::with_capacity(8);

    // 8 sequential /phrase/prepare calls -> 0 vocoder calls
    for (i, tone) in tones.iter().enumerate() {
        let out_wav = tmp.join(format!("res-{i:04}.wav"));
        let args = make_resampler_args(&in_wav, &out_wav, tone, "300");
        engine.prepare_phrase_note(&args, &stats).unwrap();
        assert!(note_sidecar_path(&out_wav).exists());
        assert_eq!(read_wav(&out_wav).unwrap().samples.len(), 64);
        note_outs.push(out_wav);
    }
    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 0);

    let phrase_out = tmp.join("cat-8notes.wav");
    // Steps 0..7 in temp.bat mode (expected_steps = 8, batch_id = 1001)
    for (i, note_wav) in note_outs.iter().enumerate() {
        let ovr = if i == 0 { 0.0 } else { 50.0 };
        engine
            .process_wavtool_step(
                WavtoolStepRequest {
                    output_file: phrase_out.to_string_lossy().into_owned(),
                    input_file: note_wav.to_string_lossy().into_owned(),
                    stp_ms: 0.0,
                    dur_ms: 250.0,
                    envelope: env_tokens(ovr),
                    commit: false,
                    batch_id: Some(1001),
                    expected_steps: Some(8),
                    fallback_resampler_args: None,
                },
                &stats,
            )
            .unwrap();
        if i < 7 {
            assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 0);
            assert!(!phrase_out.exists());
        }
    }

    // Step 7 triggers exactly ONE vocoder pass!
    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 1);
    assert!(phrase_out.exists());

    // Expected duration: 8 * 250ms - 7 * 50ms = 1650ms -> round(1.65 * 44100) = 72765 samples
    let rendered = read_wav(&phrase_out).unwrap();
    let expected_samples = (1.650 * 44100.0_f64).round() as usize;
    assert_eq!(rendered.samples.len(), expected_samples);
    assert!(rendered.samples.iter().all(|v| v.is_finite()));
    let peak = rendered
        .samples
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        peak > 0.05,
        "phrase output unexpectedly silent: peak={peak}"
    );

    // All 8 stub note WAVs were cleanly backfilled with full non-zero audio
    for note_wav in &note_outs {
        let backfilled = read_wav(note_wav).unwrap();
        assert!(backfilled.samples.len() > 64);
        let n_peak = backfilled
            .samples
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        assert!(n_peak > 0.01, "stub not backfilled: {}", note_wav.display());
    }

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn batch_id_change_resets_incomplete_session_and_fallback_recovers_missing_sidecar() {
    let tmp = std::env::temp_dir().join(format!("hifi_phrase_rst_{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("src.wav");
    generate_sine_wav(&in_wav, 0.5);

    let onnx = Arc::new(CountingMockOnnx::new());
    let engine = RealResampleEngine::new(ServerConfig::default(), onnx.clone()).unwrap();
    let stats = StatsCollector::default();
    let phrase_out = tmp.join("cat-reset.wav");

    // Partial aborted batch 77 step 0
    let n0 = tmp.join("res-0.wav");
    engine
        .prepare_phrase_note(&make_resampler_args(&in_wav, &n0, "C4", "250"), &stats)
        .unwrap();
    engine
        .process_wavtool_step(
            WavtoolStepRequest {
                output_file: phrase_out.to_string_lossy().into_owned(),
                input_file: n0.to_string_lossy().into_owned(),
                stp_ms: 0.0,
                dur_ms: 200.0,
                envelope: env_tokens(0.0),
                commit: false,
                batch_id: Some(77),
                expected_steps: Some(3),
                fallback_resampler_args: None,
            },
            &stats,
        )
        .unwrap();
    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 0);

    // New batch 88 with missing .melos recovered via fallback_resampler_args
    let n_missing_0 = tmp.join("cached-0.wav");
    let n_missing_1 = tmp.join("cached-1.wav");
    write_wav(
        &n_missing_0,
        &AudioBuffer::new(vec![0.0_f32; 64], DEFAULT_SAMPLE_RATE, 1),
    )
    .unwrap();
    write_wav(
        &n_missing_1,
        &AudioBuffer::new(vec![0.0_f32; 64], DEFAULT_SAMPLE_RATE, 1),
    )
    .unwrap();

    for (idx, (path, tone, ovr)) in [(&n_missing_0, "E4", 0.0), (&n_missing_1, "G4", 40.0)]
        .into_iter()
        .enumerate()
    {
        engine
            .process_wavtool_step(
                WavtoolStepRequest {
                    output_file: phrase_out.to_string_lossy().into_owned(),
                    input_file: path.to_string_lossy().into_owned(),
                    stp_ms: 0.0,
                    dur_ms: 200.0,
                    envelope: env_tokens(ovr),
                    commit: false,
                    batch_id: Some(88),
                    expected_steps: Some(2),
                    fallback_resampler_args: Some(make_resampler_args(&in_wav, path, tone, "250")),
                },
                &stats,
            )
            .unwrap();
        if idx == 0 {
            assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 0);
        }
    }

    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 1);
    let wav = read_wav(&phrase_out).unwrap();
    // Must be ONLY batch 88's length: 200 + 200 - 40 = 360ms -> 15876 samples (not polluted by batch 77!)
    assert_eq!(wav.samples.len(), (0.360 * 44100.0_f64).round() as usize);

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn unix_wavtool_mode_resets_when_output_deleted_and_single_note_post_writes_no_melos() {
    let tmp = std::env::temp_dir().join(format!("hifi_phrase_unix_{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("src.wav");
    generate_sine_wav(&in_wav, 0.5);

    let onnx = Arc::new(CountingMockOnnx::new());
    let engine = RealResampleEngine::new(ServerConfig::default(), onnx.clone()).unwrap();
    let stats = StatsCollector::default();

    let n0 = tmp.join("unix-n0.wav");
    let n1 = tmp.join("unix-n1.wav");
    engine
        .resample(&make_resampler_args(&in_wav, &n0, "C4", "250"), &stats)
        .unwrap();
    engine
        .resample(&make_resampler_args(&in_wav, &n1, "D4", "250"), &stats)
        .unwrap();

    // Critical E2E hygiene: single-note POST / MUST NOT write .melos onto disk
    assert!(!note_sidecar_path(&n0).exists());
    assert!(!note_sidecar_path(&n1).exists());

    let phrase_out = tmp.join("unix-cat.wav");
    // UnixWavtool deletes output before step 0
    engine
        .process_wavtool_step(
            WavtoolStepRequest {
                output_file: phrase_out.to_string_lossy().into_owned(),
                input_file: n0.to_string_lossy().into_owned(),
                stp_ms: 0.0,
                dur_ms: 200.0,
                envelope: env_tokens(0.0),
                commit: false,
                batch_id: None,
                expected_steps: None,
                fallback_resampler_args: None,
            },
            &stats,
        )
        .unwrap();
    assert_eq!(read_wav(&phrase_out).unwrap().samples.len(), 8820);

    engine
        .process_wavtool_step(
            WavtoolStepRequest {
                output_file: phrase_out.to_string_lossy().into_owned(),
                input_file: n1.to_string_lossy().into_owned(),
                stp_ms: 0.0,
                dur_ms: 200.0,
                envelope: env_tokens(40.0),
                commit: false,
                batch_id: None,
                expected_steps: None,
                fallback_resampler_args: None,
            },
            &stats,
        )
        .unwrap();
    assert_eq!(
        read_wav(&phrase_out).unwrap().samples.len(),
        (0.360 * 44100.0_f64).round() as usize
    );

    // Delete output_file to simulate next UnixWavtool phrase start -> resets to 200ms
    fs::remove_file(&phrase_out).unwrap();
    engine
        .process_wavtool_step(
            WavtoolStepRequest {
                output_file: phrase_out.to_string_lossy().into_owned(),
                input_file: n0.to_string_lossy().into_owned(),
                stp_ms: 0.0,
                dur_ms: 200.0,
                envelope: env_tokens(0.0),
                commit: false,
                batch_id: None,
                expected_steps: None,
                fallback_resampler_args: None,
            },
            &stats,
        )
        .unwrap();
    assert_eq!(read_wav(&phrase_out).unwrap().samples.len(), 8820);

    let _ = fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn phrase_http_routes_enforce_readiness_anti_csrf_and_batch_render() {
    let tmp = std::env::temp_dir().join(format!("hifi_phrase_http_{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("src.wav");
    let direct_wav = tmp.join("direct.wav");
    generate_sine_wav(&in_wav, 0.5);
    generate_sine_wav(&direct_wav, 0.3);

    let onnx = Arc::new(CountingMockOnnx::new());
    let engine: Arc<dyn ResampleEngine> =
        Arc::new(RealResampleEngine::new(ServerConfig::default(), onnx.clone()).unwrap());
    let state = AppState::new(2, engine);
    let app = build_router(state.clone());

    // Not ready -> 503
    let r503 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/phrase/prepare")
                .body(Body::from("a.wav b.wav C4 100"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r503.status(), StatusCode::SERVICE_UNAVAILABLE);

    state.set_ready(true);

    // Anti-CSRF -> 403
    let r403 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/phrase/wavtool")
                .header("Origin", "https://attacker.example")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r403.status(), StatusCode::FORBIDDEN);

    // POST /phrase/render batch with MelNote + Silence + DirectAudio
    let batch_out = tmp.join("batch.wav");
    let batch_req = PhraseBatchRequest {
        output_file: batch_out.to_string_lossy().into_owned(),
        notes: vec![
            PhraseBatchNoteSpec {
                resampler_args: Some(make_resampler_args(
                    &in_wav,
                    &tmp.join("unused.wav"),
                    "C4",
                    "250",
                )),
                wav_file: None,
                is_silence: false,
                stp_ms: 0.0,
                dur_ms: 200.0,
                envelope: env_tokens(0.0),
            },
            PhraseBatchNoteSpec {
                resampler_args: None,
                wav_file: None,
                is_silence: true,
                stp_ms: 0.0,
                dur_ms: 100.0,
                envelope: vec![],
            },
            PhraseBatchNoteSpec {
                resampler_args: None,
                wav_file: Some(direct_wav.to_string_lossy().into_owned()),
                is_silence: false,
                stp_ms: 0.0,
                dur_ms: 150.0,
                envelope: env_tokens(0.0),
            },
        ],
    };
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/phrase/render")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&batch_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 1);

    let out = read_wav(&batch_out).unwrap();
    assert_eq!(out.samples.len(), (0.450 * 44100.0_f64).round() as usize);

    let _ = fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn rejects_non_wav_output_and_respects_external_wav_overwrite() {
    let tmp = std::env::temp_dir().join(format!("hifi_phrase_ext_{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("src.wav");
    generate_sine_wav(&in_wav, 0.5);

    let onnx = Arc::new(CountingMockOnnx::new());
    let engine: Arc<dyn ResampleEngine> =
        Arc::new(RealResampleEngine::new(ServerConfig::default(), onnx.clone()).unwrap());
    let state = AppState::new(2, engine.clone());
    state.set_ready(true);
    let app = build_router(state);

    // 1. Non-.wav output_file rejected with HTTP 400 on /phrase/wavtool and /phrase/render
    let bad_wt = WavtoolStepRequest {
        output_file: tmp.join("bad.txt").to_string_lossy().into_owned(),
        input_file: in_wav.to_string_lossy().into_owned(),
        stp_ms: 0.0,
        dur_ms: 200.0,
        envelope: vec![],
        commit: true,
        batch_id: None,
        expected_steps: None,
        fallback_resampler_args: None,
    };
    let resp_wt = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/phrase/wavtool")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&bad_wt).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_wt.status(), StatusCode::BAD_REQUEST);

    let bad_batch = PhraseBatchRequest {
        output_file: tmp.join("bad.exe").to_string_lossy().into_owned(),
        notes: vec![],
    };
    let resp_batch = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/phrase/render")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&bad_batch).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_batch.status(), StatusCode::BAD_REQUEST);

    // 2. Prepare note, then externally overwrite note WAV -> cache/sidecar invalidated -> DirectAudio (0 vocoder calls)
    let stats = StatsCollector::default();
    let note_out = tmp.join("res-0.wav");
    engine
        .prepare_phrase_note(
            &make_resampler_args(&in_wav, &note_out, "C4", "250"),
            &stats,
        )
        .unwrap();
    assert!(note_sidecar_path(&note_out).exists());

    std::thread::sleep(std::time::Duration::from_millis(25));
    let ext_samples = vec![0.123_f32; 8820];
    write_wav(
        &note_out,
        &AudioBuffer::new(ext_samples, DEFAULT_SAMPLE_RATE, 1),
    )
    .unwrap();

    let phrase_out = tmp.join("cat-ext.wav");
    engine
        .process_wavtool_step(
            WavtoolStepRequest {
                output_file: phrase_out.to_string_lossy().into_owned(),
                input_file: note_out.to_string_lossy().into_owned(),
                stp_ms: 0.0,
                dur_ms: 200.0,
                envelope: env_tokens(0.0),
                commit: true,
                batch_id: Some(500),
                expected_steps: Some(1),
                fallback_resampler_args: Some(make_resampler_args(&in_wav, &note_out, "C4", "250")),
            },
            &stats,
        )
        .unwrap();

    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 0);
    assert_eq!(read_wav(&note_out).unwrap().samples.len(), 8820);
    assert_eq!(read_wav(&phrase_out).unwrap().samples.len(), 8820);

    let _ = fs::remove_dir_all(&tmp);
}
