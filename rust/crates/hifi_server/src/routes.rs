use crate::args::split_arguments;
use crate::resampler::{ResampleEngine, ResampleError};
use crate::stats::StatsCollector;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, Semaphore};

/// Shared HTTP server state across handlers and watchdogs.
pub struct AppState {
    pub ready: AtomicBool,
    pub active_jobs: AtomicUsize,
    pub start_instant: Instant,
    pub last_activity_ms: AtomicU64,
    pub stats: Arc<StatsCollector>,
    pub worker_semaphore: Arc<RwLock<Arc<Semaphore>>>,
    pub engine: OnceLock<Arc<dyn ResampleEngine>>,
    pub shutdown_notify: Arc<Notify>,
}

impl AppState {
    pub fn new(workers: usize, engine: Arc<dyn ResampleEngine>) -> Arc<Self> {
        let state = Self::new_initializing();
        let _ = state.engine.set(engine);
        state.set_workers(workers);
        state
    }

    pub fn new_initializing() -> Arc<Self> {
        let stats = Arc::new(StatsCollector::default());
        stats.reset_start_time();
        Arc::new(Self {
            ready: AtomicBool::new(false),
            active_jobs: AtomicUsize::new(0),
            start_instant: Instant::now(),
            last_activity_ms: AtomicU64::new(0),
            stats,
            worker_semaphore: Arc::new(RwLock::new(Arc::new(Semaphore::new(1)))),
            engine: OnceLock::new(),
            shutdown_notify: Arc::new(Notify::new()),
        })
    }

    pub fn set_workers(&self, workers: usize) {
        let mut guard = self
            .worker_semaphore
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *guard = Arc::new(Semaphore::new(workers.max(1)));
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    pub fn touch_activity(&self) {
        let ms = self.start_instant.elapsed().as_millis() as u64;
        self.last_activity_ms.store(ms, Ordering::Release);
    }

    pub fn idle_elapsed(&self) -> Duration {
        let now_ms = self.start_instant.elapsed().as_millis() as u64;
        let last_ms = self.last_activity_ms.load(Ordering::Acquire);
        Duration::from_millis(now_ms.saturating_sub(last_ms))
    }
}

struct ActiveJobGuard<'a>(&'a AppState);

impl<'a> ActiveJobGuard<'a> {
    fn enter(state: &'a AppState) -> Self {
        state.active_jobs.fetch_add(1, Ordering::AcqRel);
        Self(state)
    }
}

impl<'a> Drop for ActiveJobGuard<'a> {
    fn drop(&mut self) {
        self.0.active_jobs.fetch_sub(1, Ordering::AcqRel);
        self.0.touch_activity();
    }
}

/// Anti-CSRF check: reject any request carrying `Origin` or `Sec-Fetch-Site` headers.
pub fn has_browser_origin_headers(headers: &HeaderMap) -> bool {
    headers.contains_key("origin") || headers.contains_key("sec-fetch-site")
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handle_get_root).post(handle_post_root))
        .route("/health", get(handle_health))
        .route("/stats", get(handle_stats))
        .with_state(state)
}

async fn handle_get_root(State(state): State<Arc<AppState>>) -> Response {
    state.touch_activity();
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, "Server Ready").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Server Initializing").into_response()
    }
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Response {
    let ready = state.ready.load(Ordering::Acquire);
    let active_jobs = state.active_jobs.load(Ordering::Acquire);
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({ "ready": ready, "active_jobs": active_jobs })),
    )
        .into_response()
}

async fn handle_stats(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if has_browser_origin_headers(&headers) {
        return (StatusCode::FORBIDDEN, Json(json!({ "error": "Forbidden" }))).into_response();
    }
    let ready = state.ready.load(Ordering::Acquire);
    let active_jobs = state.active_jobs.load(Ordering::Acquire);
    let snap = state.stats.snapshot(active_jobs, ready, None);
    (StatusCode::OK, Json(snap)).into_response()
}

async fn handle_post_root(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    state.touch_activity();

    if has_browser_origin_headers(&headers) {
        tracing::warn!("Rejected browser cross-origin POST request.");
        return (
            StatusCode::FORBIDDEN,
            "Forbidden: Browser requests are not allowed.",
        )
            .into_response();
    }

    if !state.ready.load(Ordering::Acquire) {
        tracing::warn!("POST arrived but server not ready.");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Server initializing, please retry.",
        )
            .into_response();
    }

    let sliced = match split_arguments(&body) {
        Ok(args) => args,
        Err(err) => {
            let msg = format!("Bad request arguments: {err}");
            tracing::warn!("{msg}");
            return (StatusCode::BAD_REQUEST, msg).into_response();
        }
    };

    let _job_guard = ActiveJobGuard::enter(&state);
    let req_t0 = Instant::now();

    let in_stem = Path::new(&sliced[0])
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&sliced[0]);
    let out_name = Path::new(&sliced[1])
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&sliced[1]);
    let note = format!("'{in_stem}' -> '{out_name}'");
    tracing::info!("Queued {note} ...");

    let Some(engine) = state.engine.get().cloned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "Server initializing, please retry.",
        )
            .into_response();
    };

    let sem = state
        .worker_semaphore
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let permit = match sem.acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            return (StatusCode::SERVICE_UNAVAILABLE, "Server shutting down.").into_response();
        }
    };

    let stats = state.stats.clone();
    let args_clone = sliced.clone();
    let res = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        engine.resample(&args_clone, &stats)
    })
    .await;

    let elapsed_ms = req_t0.elapsed().as_secs_f64() * 1000.0;
    match res {
        Ok(Ok(())) => (StatusCode::OK, format!("Success: {note}")).into_response(),
        Ok(Err(ResampleError::InputNotFound(detail))) => {
            let err = format!("Error processing: Input file not found ({detail}).");
            tracing::error!("{err}");
            state.stats.record_error(&note, &err, elapsed_ms);
            (StatusCode::NOT_FOUND, err).into_response()
        }
        Ok(Err(other)) => {
            let err = format!("Error processing note: ResampleError: {other}");
            tracing::error!("{err}");
            state.stats.record_error(&note, &err, elapsed_ms);
            (StatusCode::INTERNAL_SERVER_ERROR, err).into_response()
        }
        Err(join_err) => {
            let err = format!("Error processing note: JoinError: {join_err}");
            tracing::error!("{err}");
            state.stats.record_error(&note, &err, elapsed_ms);
            (StatusCode::INTERNAL_SERVER_ERROR, err).into_response()
        }
    }
}

/// Background idle watchdog task triggering graceful shutdown after `idle_timeout_secs`.
pub async fn run_idle_watchdog(state: Arc<AppState>, idle_timeout_secs: u64) {
    if idle_timeout_secs == 0 {
        return;
    }
    tracing::info!(
        "Idle auto-shutdown enabled: server will exit after {idle_timeout_secs}s of inactivity."
    );
    let poll_interval = Duration::from_secs(if idle_timeout_secs <= 5 { 1 } else { 5 });
    let timeout = Duration::from_secs(idle_timeout_secs);

    loop {
        tokio::select! {
            _ = tokio::time::sleep(poll_interval) => {
                if state.ready.load(Ordering::Acquire)
                    && state.active_jobs.load(Ordering::Acquire) == 0
                    && state.idle_elapsed() >= timeout
                {
                    tracing::info!(
                        "Server idle for {idle_timeout_secs}s with no active jobs. Shutting down..."
                    );
                    state.shutdown_notify.notify_waiters();
                    return;
                }
            }
            _ = state.shutdown_notify.notified() => {
                return;
            }
        }
    }
}

/// Background parent-PID exit watcher task triggering shutdown when parent terminates.
pub async fn run_parent_pid_watcher(state: Arc<AppState>, parent_pid: u32) {
    tracing::info!("Parent PID exit watcher active for PID {parent_pid}.");
    loop {
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                if !is_process_alive(parent_pid) {
                    tracing::info!("Parent process {parent_pid} exited. Shutting down server...");
                    state.shutdown_notify.notify_waiters();
                    return;
                }
            }
            _ = state.shutdown_notify.notified() => {
                return;
            }
        }
    }
}

#[cfg(windows)]
pub fn is_process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return false;
        }
        let mut exit_code: u32 = 0;
        let ok = GetExitCodeProcess(handle, &mut exit_code);
        CloseHandle(handle);
        ok != 0 && exit_code == STILL_ACTIVE as u32
    }
}

#[cfg(unix)]
pub fn is_process_alive(pid: u32) -> bool {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    let ret = unsafe { kill(pid as i32, 0) };
    if ret == 0 {
        return true;
    }
    // EPERM (1) means process exists but owned by another user
    std::io::Error::last_os_error().raw_os_error() == Some(1)
}

pub async fn wait_for_shutdown_signal(notify: Arc<Notify>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        if let Ok(mut sigterm) =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            sigterm.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received SIGINT/Ctrl+C, shutting down gracefully...");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM, shutting down gracefully...");
        }
        _ = notify.notified() => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::ResampleStats;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    struct MockEngine {
        fail_not_found: bool,
    }

    impl ResampleEngine for MockEngine {
        fn resample(
            &self,
            raw_args: &[String],
            stats: &StatsCollector,
        ) -> Result<(), ResampleError> {
            if self.fail_not_found {
                return Err(ResampleError::InputNotFound(raw_args[0].clone()));
            }
            if !crate::args::is_null_output(&raw_args[1]) {
                stats.record(ResampleStats::success(
                    "mock", 1.0, 2.0, 0.5, 3.5, false, false, 100, 100,
                ));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn routes_readiness_anti_csrf_and_errors() {
        let state = AppState::new(
            2,
            Arc::new(MockEngine {
                fail_not_found: false,
            }),
        );
        let app = build_router(state.clone());

        // Not ready -> 503
        let r_init = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r_init.status(), StatusCode::SERVICE_UNAVAILABLE);

        let r_health_503 = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r_health_503.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Ready -> 200
        state.set_ready(true);
        let r_ready = app
            .clone()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r_ready.status(), StatusCode::OK);

        // Anti-CSRF on POST / and GET /stats -> 403
        let r_csrf_post = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .header("Origin", "http://evil.com")
                    .body(Body::from(r#"{"args":["a.wav","b.wav","C4","100"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r_csrf_post.status(), StatusCode::FORBIDDEN);

        let r_csrf_stats = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/stats")
                    .header("Sec-Fetch-Site", "cross-site")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r_csrf_stats.status(), StatusCode::FORBIDDEN);

        // Valid POST / -> 200
        let r_ok = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .body(Body::from(r#"{"args":["a.wav","b.wav","C4","100"]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r_ok.status(), StatusCode::OK);

        // Bad args -> 400
        let r_bad = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .body(Body::from("only_one"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r_bad.status(), StatusCode::BAD_REQUEST);

        // Unknown route -> 404
        let r_404 = app
            .oneshot(
                Request::builder()
                    .uri("/unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r_404.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn missing_input_returns_404_and_records_error() {
        let state = AppState::new(
            1,
            Arc::new(MockEngine {
                fail_not_found: true,
            }),
        );
        state.set_ready(true);
        let app = build_router(state.clone());

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/")
                    .body(Body::from("missing.wav out.wav C4 100"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(state.stats.snapshot(0, true, None).error_count, 1);
    }
}
