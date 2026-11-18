//! `hifi_client`: Fast-path UTAU resampler & phrase wavtool CLI clients (`hifisampler`, `melos`, `melos-wavtool`).

pub mod args;
pub mod bat_inspector;
pub mod config;
pub mod http;
pub mod launcher;
pub mod wavtool_cli;

use args::{is_null_output, parse_utau_args, translate_wine_path};
use bat_inspector::locate_and_inspect_temp_bat;
use config::{resolve_server_location, resolve_target_port};
use http::build_request_payload;
use launcher::{
    communicate_with_server_path, ensure_server_running_with_mode, try_optimistic_post_path,
    FastPathOutcome,
};
use std::path::Path;
use wavtool_cli::{looks_like_wavtool_invocation, run_wavtool};

/// Execute the standard `hifisampler` single-note CLI client workflow (`POST /`) and return exit code.
pub fn run(raw_args: &[String]) -> i32 {
    if raw_args.len() == 1 && (raw_args[0] == "-V" || raw_args[0] == "--version") {
        println!("hifisampler {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }
    run_resampler_with_endpoint(raw_args, "/", false)
}

/// Execute `melos` CLI: routes to wavtool when invoked in wavtool role/signature, or checks
/// `temp.bat` to route `melos` -> `/phrase/prepare` when paired with `melos-wavtool`, falling back
/// cleanly to single-note `POST /`.
pub fn run_melos(raw_args: &[String]) -> i32 {
    if raw_args.len() == 1 && (raw_args[0] == "-V" || raw_args[0] == "--version") {
        if looks_like_wavtool_invocation(raw_args) {
            println!("melos-wavtool {}", env!("CARGO_PKG_VERSION"));
        } else {
            println!("melos {}", env!("CARGO_PKG_VERSION"));
        }
        return 0;
    }
    if raw_args.is_empty() {
        eprintln!("[Main] No arguments provided.");
        return 1;
    }
    if looks_like_wavtool_invocation(raw_args) {
        return run_wavtool(raw_args);
    }

    let endpoint = match parse_utau_args(raw_args) {
        Ok(parsed) => {
            if !is_null_output(&parsed.out_file) {
                let forced = std::env::var("MELOS_PHRASE_PREPARE")
                    .ok()
                    .is_some_and(|v| v.trim() == "1");
                if forced {
                    "/phrase/prepare"
                } else if let Some(info) =
                    locate_and_inspect_temp_bat(Some(Path::new(&parsed.out_file)), None)
                {
                    if info.is_active_melos_phrase_for_note(&parsed.out_file) {
                        "/phrase/prepare"
                    } else {
                        "/"
                    }
                } else {
                    "/"
                }
            } else {
                "/"
            }
        }
        Err(_) => "/",
    };

    run_resampler_with_endpoint(raw_args, endpoint, true)
}

pub fn run_resampler_with_endpoint(
    raw_args: &[String],
    endpoint: &str,
    require_native_server: bool,
) -> i32 {
    if raw_args.is_empty() {
        eprintln!("[Main] No arguments provided.");
        return 1;
    }

    if let Err(msg) = parse_utau_args(raw_args) {
        eprintln!("[Main] {msg}");
        return 1;
    }

    let mut wire_args = raw_args.to_vec();
    if wire_args.len() >= 2 {
        wire_args[0] = translate_wine_path(&wire_args[0]);
        wire_args[1] = translate_wine_path(&wire_args[1]);
    }

    let port = resolve_target_port();
    let (payload, media_type) = build_request_payload(&wire_args);

    match try_optimistic_post_path(port, endpoint, &payload, media_type) {
        FastPathOutcome::Succeeded => return 0,
        FastPathOutcome::FatalError => return 1,
        FastPathOutcome::NeedsEnsureServer => {}
    }

    let loc = resolve_server_location();
    let port = resolve_target_port();
    if let Err(err) = ensure_server_running_with_mode(port, &loc, require_native_server) {
        eprintln!("[Main] Exiting due to critical error: {err}");
        return 1;
    }

    if let Err(err) = communicate_with_server_path(port, endpoint, &payload, media_type) {
        eprintln!("{err}");
        return 1;
    }

    0
}
