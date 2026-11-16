use serde_json::Value;
use std::path::{Path, PathBuf};

/// Returns true if `path` denotes a platform null sink (`nul` or `/dev/null`).
pub fn is_null_output(path: &str) -> bool {
    let lower = path.trim().to_ascii_lowercase();
    lower == "nul" || lower == "/dev/null"
}

/// Translate Wine-style paths (e.g. `Z:\home\...`, `C:\...`, `nul`) to POSIX paths.
pub fn translate_wine_path_posix(
    path_str: &str,
    wineprefix_env: Option<&str>,
    home_dir: Option<&Path>,
) -> String {
    if path_str.is_empty() {
        return String::new();
    }
    if is_null_output(path_str) {
        return "/dev/null".to_string();
    }

    let bytes = path_str.as_bytes();
    if bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
    {
        let drive = (bytes[0] as char).to_ascii_lowercase();
        let rest = path_str[3..].replace('\\', "/");
        if drive == 'z' {
            return format!("/{}", rest.trim_start_matches('/'));
        }

        let wine_prefix = resolve_wine_prefix(wineprefix_env, home_dir);
        if let Some(prefix) = wine_prefix {
            let dosdevice = prefix.join("dosdevices").join(format!("{drive}:"));
            if dosdevice.exists() {
                let mut out = dosdevice;
                for seg in rest.split('/').filter(|s| !s.is_empty()) {
                    out.push(seg);
                }
                return out.to_string_lossy().into_owned();
            }
        }
    }

    path_str.to_string()
}

fn resolve_wine_prefix(wineprefix_env: Option<&str>, home_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(raw) = wineprefix_env.map(str::trim).filter(|s| !s.is_empty()) {
        let p = if raw == "~" {
            home_dir.map(Path::to_path_buf)?
        } else if let Some(stripped) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
            home_dir?.join(stripped)
        } else {
            let pb = PathBuf::from(raw);
            if pb.is_absolute() {
                pb
            } else if let Some(h) = home_dir {
                h.join(pb)
            } else {
                pb
            }
        };
        return Some(p);
    }
    home_dir.map(|h| h.join(".wine"))
}

pub fn translate_wine_path(path_str: &str) -> String {
    #[cfg(windows)]
    {
        path_str.to_string()
    }
    #[cfg(not(windows))]
    {
        let env_prefix = std::env::var("WINEPREFIX").ok();
        let home = crate::beacon::home_dir();
        translate_wine_path_posix(path_str, env_prefix.as_deref(), home.as_deref())
    }
}

fn json_val_to_arg_string(val: &Value) -> Result<String, String> {
    match val {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(if *b { "True" } else { "False" }.to_string()),
        Value::Null => Ok("".to_string()),
        _ => Err("JSON argument elements must be scalar values.".to_string()),
    }
}

/// Split `in_file out_file` when tokens > 12 and paths contain spaces, matching `_AUDIO_EXT_SPLIT_RE`.
fn split_audio_ext_pair(joined: &str) -> Option<(String, String)> {
    const EXTS: &[&str] = &[".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"];
    let lower = joined.to_ascii_lowercase();
    let bytes = joined.as_bytes();

    let mut earliest: Option<(usize, usize)> = None;
    for ext in EXTS {
        let mut search_from = 0;
        while let Some(pos) = lower[search_from..].find(ext) {
            let abs_pos = search_from + pos;
            let end_ext = abs_pos + ext.len();
            if end_ext < bytes.len() && bytes[end_ext].is_ascii_whitespace() {
                match earliest {
                    None => earliest = Some((abs_pos, end_ext)),
                    Some((prev_pos, _)) if abs_pos < prev_pos => {
                        earliest = Some((abs_pos, end_ext))
                    }
                    _ => {}
                }
                break;
            }
            search_from = abs_pos + 1;
        }
    }

    let (_, end_ext) = earliest?;
    let in_part = &joined[..end_ext];
    let rest = joined[end_ext..].trim_start_matches(char::is_whitespace);
    if in_part.is_empty() || rest.is_empty() {
        return None;
    }
    Some((in_part.to_string(), rest.to_string()))
}

/// Parse JSON (`{"args": [...]}` or `[...]`) or `text/plain` (space- or newline-delimited) payload.
pub fn split_arguments(input: &str) -> Result<Vec<String>, String> {
    let stripped = input.trim();
    if stripped.is_empty() {
        return Err("Empty request payload.".to_string());
    }

    let mut args = if stripped.starts_with('{') || stripped.starts_with('[') {
        let data: Value =
            serde_json::from_str(stripped).map_err(|e| format!("Invalid JSON payload: {e}"))?;
        match data {
            Value::Object(ref map) => {
                let arr = map.get("args").and_then(|v| v.as_array()).ok_or_else(|| {
                    "JSON payload must be a list or an object with an 'args' list.".to_string()
                })?;
                arr.iter()
                    .map(json_val_to_arg_string)
                    .collect::<Result<Vec<_>, _>>()?
            }
            Value::Array(arr) => arr
                .iter()
                .map(json_val_to_arg_string)
                .collect::<Result<Vec<_>, _>>()?,
            _ => {
                return Err(
                    "JSON payload must be a list or an object with an 'args' list.".to_string(),
                );
            }
        }
    } else if stripped.contains('\n') {
        let lines: Vec<String> = stripped
            .split('\n')
            .map(|l| l.trim_end_matches('\r').to_string())
            .collect();
        if lines.len() < 4 {
            return Err(format!(
                "Expected at least 4 UTAU arguments, got {}.",
                lines.len()
            ));
        }
        lines
    } else {
        let tokens: Vec<&str> = stripped.split(' ').collect();
        if tokens.len() < 4 {
            return Err(format!(
                "Expected at least 4 UTAU arguments, got {}.",
                tokens.len()
            ));
        }
        if tokens.len() > 12 {
            let split_idx = tokens.len() - 11;
            let otherargs = &tokens[split_idx..];
            let file_path_strings = tokens[..split_idx].join(" ");
            let (in_file, out_file) =
                split_audio_ext_pair(&file_path_strings).ok_or_else(|| {
                    format!("Could not split input/output audio paths from: {file_path_strings}")
                })?;
            let mut combined = Vec::with_capacity(2 + otherargs.len());
            combined.push(in_file);
            combined.push(out_file);
            for a in otherargs {
                combined.push((*a).to_string());
            }
            combined
        } else {
            tokens.into_iter().map(str::to_string).collect()
        }
    };

    if args.len() < 4 {
        return Err(format!(
            "Expected at least 4 UTAU arguments (in_file, out_file, pitch, velocity), got {}.",
            args.len()
        ));
    }

    args[0] = translate_wine_path(&args[0]);
    args[1] = translate_wine_path(&args[1]);

    let out_lower = args[1].trim().to_ascii_lowercase();
    if out_lower != "nul" && out_lower != "/dev/null" && !out_lower.ends_with(".wav") {
        return Err(format!(
            "Output file must end with .wav or be a null device, got: {}",
            args[1]
        ));
    }

    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_space_separated_and_paths_with_spaces() {
        let basic = split_arguments("input.wav output.wav C4 100").unwrap();
        assert_eq!(basic, vec!["input.wav", "output.wav", "C4", "100"]);

        let spaces = split_arguments(
            r"C:\My Voice\in file.wav D:\Out Dir\out file.wav C4 100 g-5 0 1000 0 0 100 0 !100 AA",
        )
        .unwrap();
        assert_eq!(spaces[0], r"C:\My Voice\in file.wav");
        assert_eq!(spaces[1], r"D:\Out Dir\out file.wav");
        assert_eq!(spaces[2], "C4");
        assert_eq!(spaces.len(), 13);
    }

    #[test]
    fn split_json_and_newline_payloads() {
        let obj = split_arguments(r#"{"args":["in.wav","out.wav","C4","100","g-5"]}"#).unwrap();
        assert_eq!(obj, vec!["in.wav", "out.wav", "C4", "100", "g-5"]);

        let arr = split_arguments(r#"["in.wav","nul","C4",100]"#).unwrap();
        assert_eq!(
            arr[1].to_ascii_lowercase(),
            if cfg!(windows) { "nul" } else { "/dev/null" }
        );
        assert_eq!(arr[3], "100");

        let nl = split_arguments("in.wav\nout.wav\nC4\n100\n\n0\n300").unwrap();
        assert_eq!(nl.len(), 7);
        assert_eq!(nl[4], "");
    }

    #[test]
    fn split_rejects_invalid_payloads() {
        assert!(split_arguments("").is_err());
        assert!(split_arguments("in.wav out.wav C4").is_err());
        assert!(split_arguments("in.wav out.mp3 C4 100").is_err());
        assert!(split_arguments(r#"{"wrong":"key"}"#).is_err());
        assert!(split_arguments("{bad_json").is_err());
    }
}
