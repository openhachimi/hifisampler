//! Raw-byte inspector for OpenUtau `ExeWavtool.cs` / classic UTAU `temp.bat` scripts.
//!
//! Handles Windows ANSI codepages (Shift-JIS CP932, GBK CP936, etc.) safely without assuming UTF-8.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct BatInspection {
    pub bat_path: PathBuf,
    pub batch_id: u64,
    pub mtime_age_secs: Option<u64>,
    pub tool_path: Option<String>,
    pub output_path: Option<String>,
    pub expected_steps: usize,
    pub resampler_args_by_out: HashMap<String, Vec<String>>,
    pub resampler_args_by_name: HashMap<String, Vec<String>>,
}

impl BatInspection {
    /// Returns true if `@set tool=` points to a Melos wavtool (`melos*`).
    pub fn is_melos_wavtool(&self) -> bool {
        let Some(ref tool) = self.tool_path else {
            return false;
        };
        let fname = file_name_cmp(tool);
        let stem = fname.rsplit_once('.').map_or(fname.as_str(), |(s, _)| s);
        stem.starts_with("melos")
    }

    /// Returns true if `out_file` matches `@set output=` (filename or normalized path) inside `temp.bat`.
    pub fn matches_phrase_output(&self, out_file: &str) -> bool {
        let Some(ref bat_out) = self.output_path else {
            return false;
        };
        paths_match_or_same_filename(bat_out, out_file)
    }

    /// Resolves `@set output=` relative to `temp.bat`'s parent directory when relative.
    pub fn resolved_phrase_output_path(&self) -> Option<PathBuf> {
        let raw = self.output_path.as_deref()?.trim();
        if raw.is_empty() {
            return None;
        }
        let p = Path::new(raw);
        if p.is_absolute() {
            Some(p.to_path_buf())
        } else {
            Some(match self.bat_path.parent() {
                Some(dir) if !dir.as_os_str().is_empty() => dir.join(p),
                _ => p.to_path_buf(),
            })
        }
    }

    /// Returns true if the phrase output WAV declared in `temp.bat` already exists on disk
    /// (which indicates `temp.bat` finished a prior render or is stale).
    pub fn phrase_output_exists_on_disk(&self) -> bool {
        self.resolved_phrase_output_path()
            .is_some_and(|p| p.is_file())
    }

    /// Returns true only when `temp.bat` belongs to an active, in-flight `melos-wavtool` batch
    /// referencing `note_out_file`.
    pub fn is_active_melos_phrase_for_note(&self, note_out_file: &str) -> bool {
        self.is_melos_wavtool()
            && self.references_resampler_output(note_out_file)
            && self.mtime_age_secs.is_none_or(|age| age <= 300)
            && !self.phrase_output_exists_on_disk()
    }

    /// Returns true if `note_out_file` is referenced as a resampler `%temp%` target inside `temp.bat`.
    pub fn references_resampler_output(&self, note_out_file: &str) -> bool {
        self.find_fallback_resampler_args(note_out_file).is_some()
    }

    /// Finds the 13-element UTAU resampler argument list for `note_out_file`, if present in `temp.bat`.
    pub fn find_fallback_resampler_args(&self, note_out_file: &str) -> Option<&Vec<String>> {
        let norm_key = normalize_cmp_path(note_out_file);
        if let Some(found) = self.resampler_args_by_out.get(&norm_key) {
            return Some(found);
        }
        let name_key = file_name_cmp(note_out_file);
        if !name_key.is_empty() {
            if let Some(found) = self.resampler_args_by_name.get(&name_key) {
                let req_bare = !note_out_file.contains(['\\', '/']);
                let stored_bare = found.get(1).is_some_and(|p| !p.contains(['\\', '/']));
                if req_bare || stored_bare {
                    return Some(found);
                }
            }
        }
        None
    }
}

#[inline]
fn normalize_cmp_path(s: &str) -> String {
    let mut trimmed = s.trim().trim_matches('"');
    if cfg!(windows) {
        if let Some(stripped) = trimmed.strip_prefix(r"\\?\") {
            trimmed = stripped;
        }
        trimmed.replace('/', "\\").to_ascii_lowercase()
    } else {
        trimmed.replace('\\', "/")
    }
}

#[inline]
fn file_name_cmp(s: &str) -> String {
    let normalized = s.trim().trim_matches('"').replace('\\', "/");
    normalized
        .rsplit('/')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn paths_match_or_same_filename(a: &str, b: &str) -> bool {
    let na = normalize_cmp_path(a);
    let nb = normalize_cmp_path(b);
    if !na.is_empty() && na == nb {
        return true;
    }
    let fa = file_name_cmp(a);
    let fb = file_name_cmp(b);
    !fa.is_empty() && fa == fb
}

fn fnv1a_64_bat(bytes: &[u8], len: u64, mtime_ns: i128) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let mut feed = |slice: &[u8]| {
        for &b in slice {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    feed(&len.to_le_bytes());
    feed(&mtime_ns.to_le_bytes());
    feed(bytes);
    hash
}

/// Decodes raw bytes as UTF-8 if valid, otherwise falls back to the supplied Windows ANSI codepage
/// (passing `0` selects `CP_ACP` on Windows). On non-Windows hosts falls back cleanly to lossy UTF-8.
pub fn decode_bat_bytes_with_codepage(bytes: &[u8], codepage: u32) -> String {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s.to_string();
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Globalization::MultiByteToWideChar;
        if !bytes.is_empty() && bytes.len() <= i32::MAX as usize {
            unsafe {
                let wide_len = MultiByteToWideChar(
                    codepage,
                    0,
                    bytes.as_ptr(),
                    bytes.len() as i32,
                    std::ptr::null_mut(),
                    0,
                );
                if wide_len > 0 {
                    let mut wide = vec![0u16; wide_len as usize];
                    let written = MultiByteToWideChar(
                        codepage,
                        0,
                        bytes.as_ptr(),
                        bytes.len() as i32,
                        wide.as_mut_ptr(),
                        wide_len,
                    );
                    if written > 0 {
                        wide.truncate(written as usize);
                        return String::from_utf16_lossy(&wide);
                    }
                }
            }
        }
    }
    let _ = codepage;
    String::from_utf8_lossy(bytes).into_owned()
}

fn expand_batch_vars(input: &str, vars: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find('%') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        if let Some(end) = after.find('%') {
            if end == 0 {
                out.push('%');
            } else {
                let var_name = after[..end].to_ascii_lowercase();
                if let Some(val) = vars.get(&var_name) {
                    out.push_str(val.trim_matches('"'));
                } else {
                    out.push('%');
                    out.push_str(&after[..end]);
                    out.push('%');
                }
            }
            rest = &after[end + 1..];
        } else {
            out.push('%');
            out.push_str(after);
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

fn tokenize_batch_args(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut had_token = false;

    for ch in s.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                had_token = true;
            }
            c if c.is_ascii_whitespace() && !in_quotes => {
                if had_token {
                    tokens.push(std::mem::take(&mut cur));
                    had_token = false;
                }
            }
            c => {
                cur.push(c);
                had_token = true;
            }
        }
    }
    if had_token {
        tokens.push(cur);
    }
    tokens
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

/// Parses raw bytes from `temp.bat` (supporting arbitrary ANSI encodings via `codepage`).
pub fn parse_temp_bat_bytes(
    bat_path: PathBuf,
    bytes: &[u8],
    mtime_ns: i128,
    codepage: u32,
) -> BatInspection {
    let batch_id = fnv1a_64_bat(bytes, bytes.len() as u64, mtime_ns);
    let text = decode_bat_bytes_with_codepage(bytes, codepage);
    let mut vars: HashMap<String, String> = HashMap::new();
    let mut expected_steps = 0usize;
    let mut resampler_args_by_out: HashMap<String, Vec<String>> = HashMap::new();
    let mut resampler_args_by_name: HashMap<String, Vec<String>> = HashMap::new();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let no_bom = if line_idx == 0 {
            raw_line.trim_start_matches('\u{FEFF}')
        } else {
            raw_line
        };
        let line = no_bom
            .trim_start_matches(|c: char| c.is_ascii_whitespace() || c == '@')
            .trim_end();
        if line.is_empty() || line.starts_with("::") || line.starts_with(':') {
            continue;
        }
        if let Some(after_rem) = strip_prefix_ci(line, "rem") {
            if after_rem.is_empty() || after_rem.starts_with(|c: char| c.is_ascii_whitespace()) {
                continue;
            }
        }

        if let Some(after_set) = strip_prefix_ci(line, "set ") {
            if let Some((k, v)) = after_set.split_once('=') {
                let key = k.trim().to_ascii_lowercase();
                let expanded_val = expand_batch_vars(v.trim(), &vars);
                vars.insert(key, expanded_val);
            }
            continue;
        }

        // Ignore batch control statements such as `@if exist %temp% goto A`
        if strip_prefix_ci(line, "if ").is_some()
            || strip_prefix_ci(line, "del ").is_some()
            || strip_prefix_ci(line, "mkdir ").is_some()
            || strip_prefix_ci(line, "echo").is_some_and(|r| r.is_empty() || r.starts_with(' '))
            || strip_prefix_ci(line, "copy ").is_some()
        {
            continue;
        }

        // Check `@call %helper% ...` (OpenUtau ExeWavtool.cs / UTAU helper batch calls)
        if let Some(after_call) = strip_prefix_ci(line, "call ") {
            let tokens = tokenize_batch_args(after_call.trim());
            if tokens.is_empty() {
                continue;
            }
            let target_raw = &tokens[0];
            let target_exp = expand_batch_vars(target_raw, &vars);
            let is_helper = target_raw.eq_ignore_ascii_case("%helper%")
                || file_name_cmp(&target_exp).contains("helper");
            if is_helper {
                expected_steps += 1;
                if tokens.len() >= 9 {
                    let in_wav = expand_batch_vars(&tokens[1], &vars)
                        .trim_matches('"')
                        .to_string();
                    let tone = tokens[2].clone();
                    let offset = tokens[5].clone();
                    let dur_req = tokens[6].clone();
                    let consonant = tokens[7].clone();
                    let cutoff = tokens[8].clone();
                    let out_wav = vars
                        .get("temp")
                        .map(|v| v.trim_matches('"').to_string())
                        .unwrap_or_default();
                    let vel = vars
                        .get("vel")
                        .map(|v| v.trim_matches('"').to_string())
                        .unwrap_or_else(|| "100".to_string());
                    let flag = vars
                        .get("flag")
                        .map(|v| v.trim_matches('"').to_string())
                        .unwrap_or_default();

                    let mut param_toks: Vec<String> = vars
                        .get("params")
                        .map(|p| p.split_ascii_whitespace().map(str::to_string).collect())
                        .unwrap_or_default();
                    let volume = param_toks.first().cloned().unwrap_or_else(|| "100".into());
                    let modulation = param_toks.get(1).cloned().unwrap_or_else(|| "0".into());
                    let tempo = param_toks.get(2).cloned().unwrap_or_else(|| "!100".into());
                    let pitch_string = if param_toks.len() >= 4 {
                        param_toks.drain(3..).collect::<Vec<_>>().join("")
                    } else {
                        "AA".to_string()
                    };

                    if !out_wav.is_empty() {
                        let args = vec![
                            in_wav,
                            out_wav.clone(),
                            tone,
                            vel,
                            flag,
                            offset,
                            dur_req,
                            consonant,
                            cutoff,
                            volume,
                            modulation,
                            tempo,
                            pitch_string,
                        ];
                        resampler_args_by_out.insert(normalize_cmp_path(&out_wav), args.clone());
                        let fname = file_name_cmp(&out_wav);
                        if !fname.is_empty() {
                            resampler_args_by_name.insert(fname, args);
                        }
                    }
                }
            }
            continue;
        }

        // Direct `%tool%` invocation (`$direct=True` lines in ExeWavtool.cs)
        let tokens = tokenize_batch_args(line);
        if let Some(first) = tokens.first() {
            let expanded_cmd = expand_batch_vars(first, &vars);
            let tool_val = vars.get("tool").map(|s| s.trim_matches('"')).unwrap_or("");
            if first.eq_ignore_ascii_case("%tool%")
                || (!tool_val.is_empty() && paths_match_or_same_filename(&expanded_cmd, tool_val))
            {
                expected_steps += 1;
            }
        }
    }

    let tool_path = vars
        .get("tool")
        .map(|s| s.trim_matches('"').to_string())
        .filter(|s| !s.is_empty());
    let output_path = vars
        .get("output")
        .map(|s| s.trim_matches('"').to_string())
        .filter(|s| !s.is_empty());

    BatInspection {
        bat_path,
        batch_id,
        mtime_age_secs: None,
        tool_path,
        output_path,
        expected_steps,
        resampler_args_by_out,
        resampler_args_by_name,
    }
}

/// Locates `temp.bat` adjacent to `primary_path`, `secondary_path`, or current directory.
pub fn locate_and_inspect_temp_bat(
    primary_path: Option<&Path>,
    secondary_path: Option<&Path>,
) -> Option<BatInspection> {
    let mut candidates: Vec<PathBuf> = Vec::with_capacity(3);
    if let Some(p) = primary_path.and_then(Path::parent) {
        if !p.as_os_str().is_empty() {
            candidates.push(p.join("temp.bat"));
        }
    }
    if let Some(p) = secondary_path.and_then(Path::parent) {
        if !p.as_os_str().is_empty() {
            candidates.push(p.join("temp.bat"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("temp.bat"));
    }

    for cand in candidates {
        if let Ok(bytes) = fs::read(&cand) {
            let modified = fs::metadata(&cand).and_then(|m| m.modified()).ok();
            let mtime_ns = modified
                .map(|t| match t.duration_since(UNIX_EPOCH) {
                    Ok(d) => d.as_nanos() as i128,
                    Err(e) => -(e.duration().as_nanos() as i128),
                })
                .unwrap_or(0);
            let mtime_age_secs = modified
                .and_then(|t| SystemTime::now().duration_since(t).ok())
                .map(|d| d.as_secs());
            let mut info = parse_temp_bat_bytes(cand, &bytes, mtime_ns, 0);
            info.mtime_age_secs = mtime_age_secs;
            return Some(info);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_exe_wavtool_bat(oto_dir: &str, tool_path: &str, step_count: usize) -> Vec<u8> {
        let mut s = format!(
            "@rem project=\r\n\
             @set loadmodule=\r\n\
             @set tempo=120\r\n\
             @set samples=44100\r\n\
             @set oto={oto_dir}\r\n\
             @set tool={tool_path}\r\n\
             @set output=cat-1122334455667788.wav\r\n\
             @set helper=temp_helper.bat\r\n\
             @set cachedir={oto_dir}\r\n\
             @set flag=\"\"\r\n\
             @set env=0 5 35 0 100 100 0\r\n\
             @set stp=0\r\n\r\n"
        );
        for i in 0..step_count {
            if i + 1 == step_count {
                // Simulate a $direct=True note as final item
                s.push_str(
                    "@\"%tool%\" \"%output%\" \"%oto%\\src-direct.wav\" 10 250.0 0 5 35 0 100 100 0 0 0 0 100\r\n",
                );
            } else {
                s.push_str(&format!(
                    "@set resamp=C:\\OpenUtau\\Resamplers\\melos.exe\r\n\
                     @set params=100 0 !120 AA#2#\r\n\
                     @set flag=\"g-5\"\r\n\
                     @set env=0 5 35 0 100 100 0 50 0 100 100\r\n\
                     @set stp=12.5\r\n\
                     @set vel=100\r\n\
                     @set temp=\"%cachedir%\\res-note{i}.wav\"\r\n\
                     @call %helper% \"%oto%\\src-{i}.wav\" C4 480@120+0 20 15 300 40 -200 {i}\r\n"
                ));
            }
        }
        s.into_bytes()
    }

    #[test]
    fn parses_openutau_bat_counts_direct_and_extracts_fallback_args() {
        let bytes =
            sample_exe_wavtool_bat(r"C:\Cache", r"C:\OpenUtau\Wavtools\melos-wavtool.exe", 4);
        let info = parse_temp_bat_bytes(PathBuf::from(r"C:\Cache\temp.bat"), &bytes, 12345, 0);
        assert!(info.is_melos_wavtool());
        assert_eq!(info.expected_steps, 4);
        assert!(info.matches_phrase_output(r"C:\Cache\cat-1122334455667788.wav"));
        assert!(!info.matches_phrase_output(r"C:\Cache\cat-other.wav"));

        let fallback = info
            .find_fallback_resampler_args(r"C:\Cache\res-note1.wav")
            .expect("should recover 13 resampler args");
        assert_eq!(fallback.len(), 13);
        assert_eq!(fallback[0], r"C:\Cache\src-1.wav");
        assert_eq!(fallback[1], r"C:\Cache\res-note1.wav");
        assert_eq!(fallback[2], "C4");
        assert_eq!(fallback[3], "100");
        assert_eq!(fallback[4], "g-5");
        assert_eq!(fallback[5], "15");
        assert_eq!(fallback[6], "300");
        assert_eq!(fallback[7], "40");
        assert_eq!(fallback[8], "-200");
        assert_eq!(fallback[9], "100");
        assert_eq!(fallback[10], "0");
        assert_eq!(fallback[11], "!120");
        assert_eq!(fallback[12], "AA#2#");
    }

    #[test]
    fn raw_byte_parser_handles_shift_jis_and_gbk_paths_without_panicking() {
        // Shift-JIS `表` = [0x95, 0x5C] (trail byte 0x5C is ASCII `\`, notorious Shift-JIS trap!)
        let mut sjis_bat = Vec::new();
        sjis_bat.extend_from_slice(b"@set oto=C:\\");
        sjis_bat.extend_from_slice(&[0x95, 0x5C]); // `表`
        sjis_bat.extend_from_slice(b"\\Cache\r\n@set tool=C:\\Wavtools\\melos-wavtool.exe\r\n");
        sjis_bat.extend_from_slice(b"@set output=cat-sjis.wav\r\n");
        sjis_bat.extend_from_slice(b"@set helper=temp_helper.bat\r\n");
        sjis_bat.extend_from_slice(b"@set cachedir=%oto%\r\n");
        sjis_bat.extend_from_slice(b"@set params=100 0 !120 AA\r\n");
        sjis_bat.extend_from_slice(b"@set flag=\"\"\r\n@set vel=100\r\n");
        sjis_bat.extend_from_slice(b"@set temp=\"%cachedir%\\res-sjis.wav\"\r\n");
        sjis_bat
            .extend_from_slice(b"@call %helper% \"%oto%\\in.wav\" A4 480@120 0 0 300 0 0 0\r\n");

        let info_sjis = parse_temp_bat_bytes(PathBuf::from("temp.bat"), &sjis_bat, 1, 932);
        assert!(info_sjis.is_melos_wavtool());
        assert_eq!(info_sjis.expected_steps, 1);
        assert!(info_sjis.references_resampler_output("res-sjis.wav"));

        // GBK (CP936) raw Chinese bytes: `测试` = [0xB2, 0xE2, 0xCA, 0xD4]
        let mut gbk_bat = Vec::new();
        gbk_bat.extend_from_slice(b"@set oto=D:\\");
        gbk_bat.extend_from_slice(&[0xB2, 0xE2, 0xCA, 0xD4]);
        gbk_bat.extend_from_slice(b"\\Cache\r\n@set tool=D:\\Wavtools\\melos-wavtool.exe\r\n");
        gbk_bat.extend_from_slice(b"@set output=cat-gbk.wav\r\n");
        gbk_bat.extend_from_slice(b"@set helper=temp_helper.bat\r\n");
        gbk_bat.extend_from_slice(b"@set cachedir=%oto%\r\n");
        gbk_bat.extend_from_slice(b"@set params=100 0 !120 AA\r\n");
        gbk_bat.extend_from_slice(b"@set flag=\"\"\r\n@set vel=100\r\n");
        gbk_bat.extend_from_slice(b"@set temp=\"%cachedir%\\res-gbk.wav\"\r\n");
        gbk_bat.extend_from_slice(b"@call %helper% \"%oto%\\in.wav\" C4 480@120 0 0 300 0 0 0\r\n");

        let info_gbk = parse_temp_bat_bytes(PathBuf::from("temp.bat"), &gbk_bat, 2, 936);
        assert!(info_gbk.is_melos_wavtool());
        assert_eq!(info_gbk.expected_steps, 1);
        let args = info_gbk
            .find_fallback_resampler_args("res-gbk.wav")
            .unwrap();
        #[cfg(windows)]
        assert!(args[0].contains("测试"), "decoded={}", args[0]);
        let _ = args;
    }

    #[test]
    fn stale_bat_or_existing_output_disables_active_melos_phrase_routing() {
        let dir = std::env::temp_dir().join(format!("hifi_bat_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_string_lossy().into_owned();
        let bat_path = dir.join("temp.bat");
        let bytes = sample_exe_wavtool_bat(&dir_str, r"C:\Wavtools\melos-wavtool.exe", 2);
        let note0 = dir.join("res-note0.wav");
        let note0_s = note0.to_string_lossy().into_owned();

        let mut info = parse_temp_bat_bytes(bat_path, &bytes, 100, 0);
        assert!(info.is_active_melos_phrase_for_note(&note0_s));

        // Aged > 300s -> inactive
        info.mtime_age_secs = Some(301);
        assert!(!info.is_active_melos_phrase_for_note(&note0_s));
        info.mtime_age_secs = Some(5);
        assert!(info.is_active_melos_phrase_for_note(&note0_s));

        // Existing cat-*.wav output on disk -> inactive (switched back to convergence or completed)
        let phrase_out = dir.join("cat-1122334455667788.wav");
        fs::write(&phrase_out, b"RIFF").unwrap();
        assert!(!info.is_active_melos_phrase_for_note(&note0_s));

        // Basename mismatch across different parent dir -> not matched
        assert!(info
            .find_fallback_resampler_args(r"C:\OtherCache\res-note0.wav")
            .is_none());

        let _ = fs::remove_dir_all(&dir);
    }
}
