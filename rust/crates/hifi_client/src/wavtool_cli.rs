//! Wavtool CLI argument parser and client dispatch for `melos-wavtool` / dual-role `melos`.

use crate::args::translate_wine_path;
use crate::bat_inspector::{locate_and_inspect_temp_bat, BatInspection};
use crate::config::{resolve_server_location, resolve_target_port};
use crate::launcher::{
    communicate_with_server_path, ensure_server_running_with_mode, try_optimistic_post_path,
    FastPathOutcome,
};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedWavtoolCli {
    pub output_file: String,
    pub input_file: String,
    pub stp_ms: f64,
    pub dur_ms: f64,
    pub env_tokens: Vec<String>,
    pub commit: bool,
    pub batch_id: Option<u64>,
    pub expected_steps: Option<usize>,
}

#[derive(Debug, Serialize)]
struct WavtoolPayload {
    output_file: String,
    input_file: String,
    stp_ms: f64,
    dur_ms: f64,
    envelope: Vec<String>,
    commit: bool,
    batch_id: Option<u64>,
    expected_steps: Option<usize>,
    fallback_resampler_args: Option<Vec<String>>,
}

/// Lightweight parser mirroring `hifi_core::wavtool::parse_wavtool_duration_ms`.
pub fn parse_duration_ms_lightweight(raw: &str) -> Result<f64, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("Empty wavtool duration argument".to_string());
    }
    if let Some((ticks_raw, rest_raw)) = s.split_once('@') {
        let ticks: f64 = ticks_raw
            .trim()
            .parse()
            .map_err(|_| format!("Invalid wavtool tick count in '{s}'"))?;
        if !ticks.is_finite() {
            return Err(format!("Non-finite wavtool tick count in '{s}'"));
        }
        let rest = rest_raw.trim().strip_prefix('!').unwrap_or(rest_raw.trim());
        if rest.is_empty() {
            return Err(format!("Missing tempo after '@' in '{s}'"));
        }
        let bytes = rest.as_bytes();
        let mut sep_idx = None;
        for i in 1..bytes.len() {
            if (bytes[i] == b'+' || bytes[i] == b'-')
                && bytes[i - 1] != b'e'
                && bytes[i - 1] != b'E'
            {
                sep_idx = Some(i);
                break;
            }
        }
        let (tempo_raw, corr) = match sep_idx {
            Some(idx) => {
                let (t_part, c_part) = rest.split_at(idx);
                let c: f64 = c_part
                    .parse()
                    .map_err(|_| format!("Invalid wavtool duration correction in '{s}'"))?;
                if !c.is_finite() {
                    return Err(format!("Non-finite wavtool correction in '{s}'"));
                }
                (t_part, c)
            }
            None => (rest, 0.0),
        };
        let tempo: f64 = tempo_raw
            .parse()
            .map_err(|_| format!("Invalid wavtool tempo in '{s}'"))?;
        if !tempo.is_finite() {
            return Err(format!("Non-finite wavtool tempo in '{s}'"));
        }
        let ms = ticks * 125.0 / tempo.max(1.0) + corr;
        if !ms.is_finite() {
            return Err(format!("Non-finite wavtool duration in '{s}'"));
        }
        return Ok(ms);
    }
    let ms: f64 = s
        .parse()
        .map_err(|_| format!("Invalid wavtool duration '{s}'"))?;
    if !ms.is_finite() {
        return Err(format!("Non-finite wavtool duration '{s}'"));
    }
    Ok(ms)
}

pub fn parse_wavtool_cli(raw_args: &[String]) -> Result<ParsedWavtoolCli, String> {
    let mut positional = Vec::with_capacity(raw_args.len());
    let mut commit = false;
    let mut batch_id: Option<u64> = None;
    let mut expected_steps: Option<usize> = None;

    let mut i = 0;
    while i < raw_args.len() {
        let arg = raw_args[i].as_str();
        if arg == "--commit" {
            commit = true;
            i += 1;
        } else if let Some(val) = arg.strip_prefix("--batch-id=") {
            batch_id = Some(
                val.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("Invalid --batch-id value: '{val}'"))?,
            );
            i += 1;
        } else if arg == "--batch-id" {
            let next = raw_args
                .get(i + 1)
                .ok_or_else(|| "Missing value for --batch-id".to_string())?;
            batch_id = Some(
                next.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("Invalid --batch-id value: '{next}'"))?,
            );
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--expected-steps=") {
            expected_steps = Some(
                val.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --expected-steps value: '{val}'"))?,
            );
            i += 1;
        } else if arg == "--expected-steps" {
            let next = raw_args
                .get(i + 1)
                .ok_or_else(|| "Missing value for --expected-steps".to_string())?;
            expected_steps = Some(
                next.trim()
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --expected-steps value: '{next}'"))?,
            );
            i += 2;
        } else {
            positional.push(raw_args[i].clone());
            i += 1;
        }
    }

    if positional.len() < 4 {
        return Err(format!(
            "Expected at least 4 positional wavtool arguments (output, input, stp, dur), got {}.",
            positional.len()
        ));
    }

    let stp_ms = parse_duration_ms_lightweight(&positional[2])?;
    let dur_ms = parse_duration_ms_lightweight(&positional[3])?;
    let env_tokens = positional[4..].to_vec();
    match env_tokens.len() {
        0 | 2 => {}
        n if n >= 7 => {}
        other => {
            return Err(format!(
                "Invalid wavtool envelope token count: expected 0, 2, or >=7, got {other}"
            ));
        }
    }
    for tok in &env_tokens {
        let v: f64 = tok
            .trim()
            .parse()
            .map_err(|_| format!("Invalid wavtool envelope number: '{tok}'"))?;
        if !v.is_finite() {
            return Err(format!("Non-finite wavtool envelope number: '{tok}'"));
        }
    }

    Ok(ParsedWavtoolCli {
        output_file: translate_wine_path(&positional[0]),
        input_file: translate_wine_path(&positional[1]),
        stp_ms,
        dur_ms,
        env_tokens,
        commit,
        batch_id,
        expected_steps,
    })
}

fn melos_sidecar_exists(input_file: &str) -> bool {
    let mut os = Path::new(input_file).as_os_str().to_os_string();
    os.push(".melos");
    Path::new(&os).is_file()
}

pub fn build_wavtool_payload_with_inspection(
    parsed: &ParsedWavtoolCli,
    bat: Option<&BatInspection>,
) -> String {
    let (batch_id, expected_steps, fallback_resampler_args) =
        if let Some(info) = bat.filter(|b| b.matches_phrase_output(&parsed.output_file)) {
            let b_id = parsed.batch_id.or(Some(info.batch_id));
            let exp = parsed.expected_steps.or(if info.expected_steps > 0 {
                Some(info.expected_steps)
            } else {
                None
            });
            let fallback = if !melos_sidecar_exists(&parsed.input_file) {
                info.find_fallback_resampler_args(&parsed.input_file)
                    .cloned()
                    .map(|mut args| {
                        if args.len() >= 2 {
                            args[0] = translate_wine_path(&args[0]);
                            args[1] = translate_wine_path(&args[1]);
                        }
                        args
                    })
            } else {
                None
            };
            (b_id, exp, fallback)
        } else {
            (parsed.batch_id, parsed.expected_steps, None)
        };

    let payload = WavtoolPayload {
        output_file: parsed.output_file.clone(),
        input_file: parsed.input_file.clone(),
        stp_ms: parsed.stp_ms,
        dur_ms: parsed.dur_ms,
        envelope: parsed.env_tokens.clone(),
        commit: parsed.commit,
        batch_id,
        expected_steps,
        fallback_resampler_args,
    };
    serde_json::to_string(&payload).expect("serializing WavtoolPayload is infallible")
}

/// Runs `melos-wavtool` CLI workflow against `/phrase/wavtool`.
pub fn run_wavtool(raw_args: &[String]) -> i32 {
    if raw_args.len() == 1 && (raw_args[0] == "-V" || raw_args[0] == "--version") {
        println!("melos-wavtool {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }
    if raw_args.is_empty() {
        eprintln!("[Wavtool] No arguments provided.");
        return 1;
    }
    let parsed = match parse_wavtool_cli(raw_args) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("[Wavtool] {msg}");
            return 1;
        }
    };

    let bat = locate_and_inspect_temp_bat(
        Some(Path::new(&parsed.output_file)),
        Some(Path::new(&parsed.input_file)),
    );
    let payload = build_wavtool_payload_with_inspection(&parsed, bat.as_ref());
    let port = resolve_target_port();

    match try_optimistic_post_path(port, "/phrase/wavtool", &payload, "application/json") {
        FastPathOutcome::Succeeded => return 0,
        FastPathOutcome::FatalError => return 1,
        FastPathOutcome::NeedsEnsureServer => {}
    }

    let loc = resolve_server_location();
    let port = resolve_target_port();
    if let Err(err) = ensure_server_running_with_mode(port, &loc, true) {
        eprintln!("[Wavtool] Exiting due to critical error: {err}");
        return 1;
    }

    if let Err(err) =
        communicate_with_server_path(port, "/phrase/wavtool", &payload, "application/json")
    {
        eprintln!("{err}");
        return 1;
    }

    0
}

fn is_valid_midi_note_token(tok: &str) -> bool {
    let s = tok.trim();
    if s.is_empty() {
        return false;
    }
    let unsigned = s.strip_prefix(['+', '-']).unwrap_or(s);
    if !unsigned.is_empty() && unsigned.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    let bytes = s.as_bytes();
    if !bytes[0].is_ascii_alphabetic() || !('A'..='G').contains(&(bytes[0] as char)) {
        return false;
    }
    let mut idx = 1;
    if idx < bytes.len() && (bytes[idx] == b'#' || bytes[idx] == b'b') {
        idx += 1;
    }
    if idx >= bytes.len() {
        return false;
    }
    let oct = &s[idx..];
    let oct_digits = oct.strip_prefix('-').unwrap_or(oct);
    !oct_digits.is_empty() && oct_digits.bytes().all(|b| b.is_ascii_digit())
}

/// Detects whether `melos` CLI invocation is acting as Wavtool vs Resampler
/// (supports copying `melos.exe` directly into `OpenUtau/Wavtools/` Moresampler-style).
pub fn looks_like_wavtool_invocation(raw_args: &[String]) -> bool {
    if let Ok(role) = std::env::var("MELOS_ROLE") {
        let lower = role.trim().to_ascii_lowercase();
        if lower == "wavtool" {
            return true;
        }
        if lower == "resampler" {
            return false;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let stem = exe
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if stem.contains("wavtool") {
            return true;
        }
    }
    if raw_args.iter().any(|a| {
        a == "--commit" || a.starts_with("--batch-id") || a.starts_with("--expected-steps")
    }) {
        return true;
    }
    if raw_args.len() >= 4 {
        if raw_args[3].contains('@') {
            return true;
        }
        if !is_valid_midi_note_token(&raw_args[2]) && raw_args[2].trim().parse::<f64>().is_ok() {
            return true;
        }
        // $direct wavtool args have numeric stp + decimal duration (`{dur:F1}`) or 2-token env (`len == 6`)
        if raw_args[2].trim().parse::<f64>().is_ok() {
            let dur_tok = raw_args[3].trim();
            if dur_tok.contains('.') && dur_tok.parse::<f64>().is_ok() {
                return true;
            }
            if raw_args.len() == 6
                && dur_tok.parse::<f64>().is_ok()
                && !raw_args[4].trim().is_empty()
                && raw_args[4].trim().parse::<f64>().is_ok()
                && raw_args[5].trim().parse::<f64>().is_ok()
            {
                return true;
            }
        }
        // $direct wavtool args have numeric dur + >=7 numeric env tokens without `!tempo` at index 11
        if raw_args.len() >= 11
            && raw_args[4..11]
                .iter()
                .all(|t| t.trim().parse::<f64>().is_ok())
            && !raw_args.get(11).is_some_and(|s| s.trim().starts_with('!'))
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_role_detection_distinguishes_resampler_vs_wavtool() {
        // Resampler call (4 args and 13 args)
        let res_min: Vec<String> = ["in.wav", "out.wav", "A4", "100"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(!looks_like_wavtool_invocation(&res_min));

        let res_full: Vec<String> = [
            "in.wav", "out.wav", "C#4", "100", "g-5", "0", "300", "50", "-200", "100", "0", "!120",
            "AA",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(!looks_like_wavtool_invocation(&res_full));

        // Classic wavtool call (`480@120+0`)
        let wt_classic: Vec<String> = [
            "cat.wav",
            "res-1.wav",
            "0",
            "480@120+0",
            "0",
            "5",
            "35",
            "0",
            "100",
            "100",
            "0",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(looks_like_wavtool_invocation(&wt_classic));

        // $direct wavtool call (float stp / numeric env tokens without !tempo at idx 11)
        let wt_direct: Vec<String> = [
            "cat.wav", "src.wav", "10.5", "250.0", "0", "5", "35", "0", "100", "100", "0", "0",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(looks_like_wavtool_invocation(&wt_direct));

        // Integer MIDI pitch "60" with stp=0? Wait: check 13-arg resampler starting with MIDI integer "60":
        let res_midi_num: Vec<String> = [
            "in.wav", "out.wav", "60", "100", "", "0", "500", "0", "0", "100", "0", "!120", "AA",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        assert!(!looks_like_wavtool_invocation(&res_midi_num));

        // $direct wavtool call with stp="0" and decimal dur="250.0" (4 args and 6 args)
        let wt_direct_4: Vec<String> = ["cat.wav", "src.wav", "0", "250.0"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(looks_like_wavtool_invocation(&wt_direct_4));

        let wt_direct_2tok: Vec<String> = ["cat.wav", "src.wav", "0", "250", "0", "10"]
            .into_iter()
            .map(String::from)
            .collect();
        assert!(looks_like_wavtool_invocation(&wt_direct_2tok));

        assert!(parse_duration_ms_lightweight("480@inf").is_err());
        assert!(parse_duration_ms_lightweight("NaN").is_err());
        assert!(parse_duration_ms_lightweight("480@120+inf").is_err());
    }
}
