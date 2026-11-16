//! `hifi_client`: Fast-path UTAU resampler CLI client (`hifisampler`).

pub mod args;
pub mod config;
pub mod http;
pub mod launcher;

use args::{parse_utau_args, translate_wine_path};
use config::{resolve_server_location, resolve_target_port};
use http::build_request_payload;
use launcher::{
    communicate_with_server, ensure_server_running, try_optimistic_post, FastPathOutcome,
};

/// Execute the `hifisampler` CLI client workflow and return the process exit code.
pub fn run(raw_args: &[String]) -> i32 {
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

    match try_optimistic_post(port, &payload, media_type) {
        FastPathOutcome::Succeeded => return 0,
        FastPathOutcome::FatalError => return 1,
        FastPathOutcome::NeedsEnsureServer => {}
    }

    let loc = resolve_server_location();
    let port = resolve_target_port();
    if let Err(err) = ensure_server_running(port, &loc) {
        eprintln!("[Main] Exiting due to critical error: {err}");
        return 1;
    }

    if let Err(err) = communicate_with_server(port, &payload, media_type) {
        eprintln!("{err}");
        return 1;
    }

    0
}
