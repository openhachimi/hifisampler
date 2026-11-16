use std::path::{Path, PathBuf};

/// Parsed UTAU positional arguments passed to `hifisampler`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtauClientArgs {
    pub in_file: String,
    pub out_file: String,
    pub pitch: String,
    pub velocity: String,
    pub flags: String,
    pub offset: String,
    pub length: String,
    pub consonant: String,
    pub cutoff: String,
    pub volume: String,
    pub modulation: String,
    pub tempo: String,
    pub pitch_string: String,
}

/// Returns true if `path` denotes a platform null sink (`nul` or `/dev/null`).
pub fn is_null_output(path: &str) -> bool {
    let lower = path.trim().to_ascii_lowercase();
    lower == "nul" || lower == "/dev/null"
}

/// Returns true if `out_file` is either a null sink or ends with `.wav` (case-insensitive).
pub fn is_valid_output_path(path: &str) -> bool {
    if is_null_output(path) {
        return true;
    }
    path.trim().to_ascii_lowercase().ends_with(".wav")
}

/// Translate Wine-style Windows drive paths (e.g. `Z:\home\...`, `C:\...`, `nul`)
/// into native POSIX paths.
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
                let joined = join_posix_rel(&dosdevice, &rest);
                return joined.to_string_lossy().into_owned();
            }
        }
    }

    path_str.to_string()
}

fn resolve_wine_prefix(wineprefix_env: Option<&str>, home_dir: Option<&Path>) -> Option<PathBuf> {
    if let Some(raw) = wineprefix_env.map(str::trim).filter(|s| !s.is_empty()) {
        let expanded = if raw == "~" {
            home_dir.map(Path::to_path_buf)?
        } else if let Some(stripped) = raw.strip_prefix("~/").or_else(|| raw.strip_prefix("~\\")) {
            home_dir?.join(stripped)
        } else {
            let p = PathBuf::from(raw);
            if p.is_absolute() {
                p
            } else if let Some(h) = home_dir {
                h.join(p)
            } else {
                p
            }
        };
        return Some(expanded);
    }
    home_dir.map(|h| h.join(".wine"))
}

fn join_posix_rel(base: &Path, rest: &str) -> PathBuf {
    let mut out = base.to_path_buf();
    for seg in rest.split('/').filter(|s| !s.is_empty()) {
        out.push(seg);
    }
    out
}

/// Translate Wine paths on POSIX targets; on Windows returns the path unchanged.
pub fn translate_wine_path(path_str: &str) -> String {
    #[cfg(windows)]
    {
        path_str.to_string()
    }
    #[cfg(not(windows))]
    {
        let env_prefix = std::env::var("WINEPREFIX").ok();
        let home = crate::config::home_dir();
        translate_wine_path_posix(path_str, env_prefix.as_deref(), home.as_deref())
    }
}

/// Validate and parse 4..=13 positional UTAU CLI arguments, filling defaults for omitted slots.
pub fn parse_utau_args(raw_args: &[String]) -> Result<UtauClientArgs, String> {
    if raw_args.len() < 4 {
        return Err(format!(
            "Expected at least 4 UTAU arguments (in_file, out_file, pitch, velocity), got {}.",
            raw_args.len()
        ));
    }
    if !is_valid_output_path(&raw_args[1]) {
        return Err(format!(
            "Output file must end with .wav or be a null device, got: {}",
            raw_args[1]
        ));
    }

    let get_or = |idx: usize, default: &str| -> String {
        raw_args
            .get(idx)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    };

    Ok(UtauClientArgs {
        in_file: translate_wine_path(&raw_args[0]),
        out_file: translate_wine_path(&raw_args[1]),
        pitch: raw_args[2].clone(),
        velocity: raw_args[3].clone(),
        flags: get_or(4, ""),
        offset: get_or(5, "0"),
        length: get_or(6, "1000"),
        consonant: get_or(7, "0"),
        cutoff: get_or(8, "0"),
        volume: get_or(9, "100"),
        modulation: get_or(10, "0"),
        tempo: get_or(11, "!100"),
        pitch_string: get_or(12, "AA"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_and_wav_validation() {
        assert!(is_null_output("nul"));
        assert!(is_null_output("NUL"));
        assert!(is_null_output("/dev/null"));
        assert!(is_valid_output_path("out.WAV"));
        assert!(is_valid_output_path("NUL"));
        assert!(!is_valid_output_path("out.mp3"));
    }

    #[test]
    fn wine_z_drive_and_null_posix_translation() {
        assert_eq!(
            translate_wine_path_posix(r"Z:\home\user\sample.wav", None, None),
            "/home/user/sample.wav"
        );
        assert_eq!(
            translate_wine_path_posix("z:/tmp/a.wav", None, None),
            "/tmp/a.wav"
        );
        assert_eq!(translate_wine_path_posix("NUL", None, None), "/dev/null");
    }

    #[test]
    fn wine_custom_drive_dosdevices() {
        let tmp = std::env::temp_dir().join(format!("hifi_wine_test_{}", std::process::id()));
        let dos_c = tmp.join(".wine").join("dosdevices").join("c:");
        std::fs::create_dir_all(&dos_c).unwrap();
        let translated = translate_wine_path_posix(r"C:\Voice\test.wav", None, Some(&tmp));
        assert!(translated.ends_with("Voice/test.wav") || translated.ends_with(r"Voice\test.wav"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn parse_defaults_and_minimum_args() {
        let parsed =
            parse_utau_args(&["in.wav".into(), "out.wav".into(), "C4".into(), "100".into()])
                .unwrap();
        assert_eq!(parsed.flags, "");
        assert_eq!(parsed.length, "1000");
        assert_eq!(parsed.tempo, "!100");
        assert_eq!(parsed.pitch_string, "AA");

        assert!(parse_utau_args(&["a.wav".into(), "b.wav".into(), "C4".into()]).is_err());
        assert!(
            parse_utau_args(&["a.wav".into(), "b.mp3".into(), "C4".into(), "100".into()]).is_err()
        );
    }
}
