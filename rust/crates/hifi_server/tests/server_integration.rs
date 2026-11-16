use hifi_core::audio::{write_wav, AudioBuffer, DEFAULT_SAMPLE_RATE};
use hifi_core::utau::parse_flags;
use hifi_server::beacon::write_locator_beacon_to;
use hifi_server::config::ServerConfig;
use hifi_server::pipeline::RealResampleEngine;
use hifi_server::resampler::{build_feature_cache_path, OnnxRunner, ResampleEngine, ResampleError};
use hifi_server::routes::{
    build_router, is_process_alive, run_idle_watchdog, run_parent_pid_watcher, AppState,
};
use hifi_server::stats::StatsCollector;
use ndarray::{Array4, ArrayView2};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

struct FakeOnnxRunner {
    hnsep_calls: AtomicUsize,
    vocoder_calls: AtomicUsize,
}

impl FakeOnnxRunner {
    fn new() -> Self {
        Self {
            hnsep_calls: AtomicUsize::new(0),
            vocoder_calls: AtomicUsize::new(0),
        }
    }
}

impl OnnxRunner for FakeOnnxRunner {
    fn hnsep_uses_gpu(&self) -> bool {
        false
    }

    fn vocoder_uses_gpu(&self) -> bool {
        false
    }

    fn run_hnsep_mask(&self, stft_tensor: &Array4<f32>) -> Result<Array4<f32>, ResampleError> {
        self.hnsep_calls.fetch_add(1, Ordering::SeqCst);
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

fn generate_sine_wav(path: &std::path::Path, dur_sec: f32) {
    let n = (DEFAULT_SAMPLE_RATE as f32 * dur_sec) as usize;
    let samples: Vec<f32> = (0..n)
        .map(|i| 0.4 * (2.0 * PI * 440.0 * i as f32 / DEFAULT_SAMPLE_RATE as f32).sin())
        .collect();
    write_wav(path, &AudioBuffer::new(samples, DEFAULT_SAMPLE_RATE, 1)).unwrap();
}

#[test]
fn real_dsp_pipeline_caches_hnsep_tension_fry_growl_lufs_and_nul() {
    let tmp = std::env::temp_dir().join(format!("hifi_real_pipe_{}", Uuid::new_v4().simple()));
    std::fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("note.wav");
    generate_sine_wav(&in_wav, 0.35);
    let out_wav = tmp.join("out.wav");

    let onnx = Arc::new(FakeOnnxRunner::new());
    let engine = RealResampleEngine::new(ServerConfig::default(), onnx.clone()).unwrap();
    let stats = StatsCollector::default();

    // 1. Null preview warms .hifi.bin cache without calling vocoder or recording stats
    engine
        .resample(
            &[
                in_wav.to_string_lossy().into_owned(),
                "nul".to_string(),
                "A4".to_string(),
                "100".to_string(),
            ],
            &stats,
        )
        .unwrap();
    let default_cache = build_feature_cache_path(&in_wav, &parse_flags(""));
    assert!(default_cache.exists());
    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 0);
    assert_eq!(stats.snapshot(0, true, None).total_requests, 0);

    // 2. Real render hits warmed .hifi.bin cache and runs vocoder + records stats
    engine
        .resample(
            &[
                in_wav.to_string_lossy().into_owned(),
                out_wav.to_string_lossy().into_owned(),
                "A4".to_string(),
                "100".to_string(),
                "P86A30".to_string(),
                "0".to_string(),
                "250".to_string(),
            ],
            &stats,
        )
        .unwrap();
    assert!(out_wav.exists());
    assert_eq!(onnx.vocoder_calls.load(Ordering::SeqCst), 1);
    let snap = stats.snapshot(0, true, None);
    assert_eq!(snap.total_requests, 1);
    assert_eq!(snap.cache_hits, 1);

    // 3. Ht50Ho40Hr-30HG30 exercises HNSEP + tension + Klatt + growl + v3 cache tag
    let out_fx = tmp.join("out_fx.wav");
    engine
        .resample(
            &[
                in_wav.to_string_lossy().into_owned(),
                out_fx.to_string_lossy().into_owned(),
                "C4".to_string(),
                "100".to_string(),
                "Ht50Ho40Hr-30HG30".to_string(),
                "0".to_string(),
                "250".to_string(),
            ],
            &stats,
        )
        .unwrap();
    assert!(out_fx.exists());
    assert_eq!(onnx.hnsep_calls.load(Ordering::SeqCst), 1);
    let v3_cache = build_feature_cache_path(&in_wav, &parse_flags("Ht50Ho40Hr-30HG30"));
    assert!(v3_cache.exists());

    // 4. Deep scoop triggers vocal fry path and reuses _hnsep cache when Hb120Hv80 runs
    let out_fry = tmp.join("out_fry.wav");
    engine
        .resample(
            &[
                in_wav.to_string_lossy().into_owned(),
                out_fry.to_string_lossy().into_owned(),
                "A3".to_string(),
                "100".to_string(),
                "Hb120Hv80".to_string(),
                "0".to_string(),
                "300".to_string(),
                "0".to_string(),
                "0".to_string(),
                "100".to_string(),
                "0".to_string(),
                "!120".to_string(),
                "AA#5//".to_string(),
            ],
            &stats,
        )
        .unwrap();
    assert!(out_fry.exists());
    assert_eq!(onnx.hnsep_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stats.snapshot(0, true, None).hnsep_cache_hits, 1);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn live_http_loopback_client_server_roundtrip_and_stats_schema() {
    let tmp = std::env::temp_dir().join(format!("hifi_http_e2e_{}", Uuid::new_v4().simple()));
    std::fs::create_dir_all(&tmp).unwrap();
    let in_wav = tmp.join("note.wav");
    generate_sine_wav(&in_wav, 0.25);
    let out_wav = tmp.join("out.wav");

    let onnx = Arc::new(FakeOnnxRunner::new());
    let engine = Arc::new(RealResampleEngine::new(ServerConfig::default(), onnx).unwrap());
    let state = AppState::new(2, engine);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let router = build_router(state.clone());

    let shutdown = state.shutdown_notify.clone();
    let srv_handle = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown.notified().await })
            .await
            .unwrap();
    });

    let (payload, media) = hifi_client::http::build_request_payload_with_protocol(
        &[
            in_wav.to_string_lossy().into_owned(),
            out_wav.to_string_lossy().into_owned(),
            "C4".into(),
            "100".into(),
        ],
        "json",
    );
    let outcome_503 = tokio::task::spawn_blocking({
        let payload = payload.clone();
        move || hifi_client::launcher::try_optimistic_post(port, &payload, media)
    })
    .await
    .unwrap();
    assert_eq!(
        outcome_503,
        hifi_client::launcher::FastPathOutcome::NeedsEnsureServer
    );

    state.set_ready(true);
    let outcome_ok = tokio::task::spawn_blocking(move || {
        hifi_client::launcher::try_optimistic_post(port, &payload, media)
    })
    .await
    .unwrap();
    assert_eq!(
        outcome_ok,
        hifi_client::launcher::FastPathOutcome::Succeeded
    );
    assert!(out_wav.exists());

    // Missing input -> 404
    let missing_payload = serde_json::json!({
        "args": [tmp.join("missing.wav").to_string_lossy(), out_wav.to_string_lossy(), "C4", "100"]
    })
    .to_string();
    let missing_resp = tokio::task::spawn_blocking(move || {
        hifi_client::http::http_request_loopback(
            port,
            "POST",
            "/",
            Some((&missing_payload, "application/json")),
            Duration::from_secs(1),
            Duration::from_secs(5),
        )
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(missing_resp.status, 404);

    let stats_resp = tokio::task::spawn_blocking(move || {
        hifi_client::http::http_request_loopback(
            port,
            "GET",
            "/stats",
            None,
            Duration::from_secs(1),
            Duration::from_secs(2),
        )
        .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(stats_resp.status, 200);
    let json_val: serde_json::Value = serde_json::from_str(&stats_resp.body).unwrap();
    assert_eq!(
        json_val.get("total_requests").and_then(|v| v.as_u64()),
        Some(2)
    );
    assert_eq!(
        json_val.get("success_count").and_then(|v| v.as_u64()),
        Some(1)
    );
    assert_eq!(
        json_val.get("error_count").and_then(|v| v.as_u64()),
        Some(1)
    );

    state.shutdown_notify.notify_waiters();
    srv_handle.await.unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn idle_watchdog_and_parent_pid_watcher_trigger_shutdown() {
    let onnx = Arc::new(FakeOnnxRunner::new());
    let engine = Arc::new(RealResampleEngine::new(ServerConfig::default(), onnx).unwrap());
    let state = AppState::new(1, engine);
    state.set_ready(true);
    state.touch_activity();

    let watchdog = tokio::spawn(run_idle_watchdog(state.clone(), 1));
    tokio::time::timeout(Duration::from_secs(4), state.shutdown_notify.notified())
        .await
        .expect("idle watchdog should trigger shutdown");
    watchdog.await.unwrap();

    let dir = std::env::temp_dir().join(format!("hifi_beacon_int_{}", Uuid::new_v4().simple()));
    let beacon = dir.join("server.json");
    write_locator_beacon_to(&beacon, &dir, 8572, true).unwrap();
    assert!(beacon.exists());
    let _ = std::fs::remove_dir_all(&dir);

    assert!(is_process_alive(std::process::id()));
    let state2 = AppState::new_initializing();
    let watcher = tokio::spawn(run_parent_pid_watcher(state2.clone(), 999_999_999));
    tokio::time::timeout(Duration::from_secs(4), state2.shutdown_notify.notified())
        .await
        .expect("parent watcher should trigger on nonexistent pid");
    watcher.await.unwrap();
}
