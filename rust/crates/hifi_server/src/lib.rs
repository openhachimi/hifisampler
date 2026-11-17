//! `hifi_server`: Axum/Tokio inference HTTP server and synthesis orchestrator (`hifiserver`).

pub mod args;
pub mod beacon;
pub mod cache_locks;
pub mod cli;
pub mod config;
pub mod lock;
pub mod pipeline;
pub mod resampler;
pub mod routes;
pub mod stats;

use cli::{apply_managed_runtime_config, parse_server_cli_args};
use config::{load_server_config, resolve_server_root};
use lock::ServerInstanceLock;
use resampler::ResampleEngine;
use routes::{
    build_router, run_idle_watchdog, run_parent_pid_watcher, wait_for_shutdown_signal, AppState,
};
use std::sync::Arc;
use std::time::Duration;

pub type EngineFactory = Box<
    dyn FnOnce(
            &config::ServerConfig,
            &std::path::Path,
        ) -> Result<(Arc<dyn ResampleEngine>, usize), String>
        + Send,
>;

/// Run the `hifiserver` process with a caller-supplied backend factory.
pub fn run_server_with_factory(factory: EngineFactory) -> i32 {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = match parse_server_cli_args() {
        Ok(c) => c,
        Err(msg) => {
            if msg.starts_with("HifiSampler UTAU inference server") {
                println!("{msg}");
                return 0;
            }
            eprintln!("Error: {msg}");
            return 1;
        }
    };

    let server_root = resolve_server_root(cli.config.as_deref());

    if cli.register_only {
        match beacon::write_locator_beacon(&server_root, cli.port, cli.managed) {
            Ok(path) => {
                println!(
                    "Registered hifisampler locator beacon at: {}",
                    path.display()
                );
                return 0;
            }
            Err(err) => {
                eprintln!("Failed to write locator beacon: {err}");
                return 1;
            }
        }
    }

    let instance_lock = match ServerInstanceLock::try_acquire(cli.port, Duration::from_millis(500))
    {
        Ok(Some(lock)) => lock,
        Ok(None) => {
            let lock_path = lock::get_lock_file_path(cli.port);
            tracing::warn!(
                "Another instance seems to be running (lock file '{}' is held). Exiting.",
                lock_path.display()
            );
            return 0;
        }
        Err(err) => {
            tracing::error!("Failed to acquire single-instance lock: {err}");
            return 1;
        }
    };

    let mut cfg = match load_server_config(&server_root, cli.config.as_deref()) {
        Ok(c) => c,
        Err(err) => {
            tracing::error!("Configuration error: {err}");
            return 1;
        }
    };

    // Best-effort cleanup of stale orphan `.*.tmp` files leftover from hard crashes.
    let stale_min_age = Duration::from_secs(300);
    let _ = hifi_core::cache::cleanup_stale_tmp_files(&std::env::temp_dir(), stale_min_age);
    if let Some(beacon_dir) = beacon::default_beacon_path()
        .as_deref()
        .and_then(std::path::Path::parent)
    {
        let _ = hifi_core::cache::cleanup_stale_tmp_files(beacon_dir, stale_min_age);
    }

    let idle_timeout = apply_managed_runtime_config(&mut cfg, cli.managed, cli.idle_timeout);
    if let Err(err) = beacon::write_locator_beacon(&server_root, cli.port, cli.managed) {
        tracing::debug!("Could not write locator beacon: {err}");
    }

    tracing::info!(
        "hifiserver starting (root={}, managed={}, host={}, port={}, idle_timeout={}s)",
        server_root.display(),
        cli.managed,
        cli.host,
        cli.port,
        idle_timeout
    );
    tracing::info!(
        "Successfully acquired server lock: {}",
        instance_lock.path().display()
    );

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(err) => {
            tracing::error!("Failed to build Tokio runtime: {err}");
            return 1;
        }
    };

    let exit_code = rt.block_on(async move {
        let state = AppState::new_initializing();

        let bind_addr = format!("{}:{}", cli.host, cli.port);
        let listener = match tokio::net::TcpListener::bind(&bind_addr).await {
            Ok(l) => l,
            Err(err) => {
                tracing::error!("Failed to bind HTTP listener on {bind_addr}: {err}");
                return 1;
            }
        };

        let init_state = state.clone();
        let cfg_clone = cfg.clone();
        let root_clone = server_root.clone();
        let init_failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let init_failed_flag = init_failed.clone();

        let init_handle = tokio::spawn(async move {
            let res = tokio::task::spawn_blocking(move || factory(&cfg_clone, &root_clone)).await;
            match res {
                Ok(Ok((engine, resolved_workers))) => {
                    init_state.set_workers(resolved_workers);
                    let _ = init_state.engine.set(engine);
                    init_state.stats.reset_start_time();
                    init_state.touch_activity();
                    init_state.set_ready(true);
                    tracing::info!(
                        "Listening on http://{bind_addr}; axum + inference-workers={resolved_workers}"
                    );
                }
                Ok(Err(err)) => {
                    tracing::error!("Failed to initialize synthesis engine: {err}");
                    init_failed_flag.store(true, std::sync::atomic::Ordering::Release);
                    init_state.shutdown_notify.notify_waiters();
                }
                Err(join_err) => {
                    tracing::error!("Engine initialization panicked: {join_err}");
                    init_failed_flag.store(true, std::sync::atomic::Ordering::Release);
                    init_state.shutdown_notify.notify_waiters();
                }
            }
        });

        let watchdog_handle = (idle_timeout > 0)
            .then(|| tokio::spawn(run_idle_watchdog(state.clone(), idle_timeout)));
        let parent_handle = cli
            .parent_pid
            .map(|pid| tokio::spawn(run_parent_pid_watcher(state.clone(), pid)));

        let router = build_router(state.clone());
        let serve_res = axum::serve(listener, router)
            .with_graceful_shutdown(wait_for_shutdown_signal(state.shutdown_notify.clone()))
            .await;

        state.shutdown_notify.notify_waiters();
        let _ = init_handle.await;
        if let Some(h) = watchdog_handle {
            let _ = h.await;
        }
        if let Some(h) = parent_handle {
            let _ = h.await;
        }

        if init_failed.load(std::sync::atomic::Ordering::Acquire) {
            return 1;
        }
        if let Err(err) = serve_res {
            tracing::error!("HTTP server error: {err}");
            return 1;
        }
        0
    });

    drop(instance_lock);
    exit_code
}
