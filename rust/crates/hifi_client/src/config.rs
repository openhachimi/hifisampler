use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_PORT: u16 = 8572;

#[derive(Debug, Clone, Default)]
pub struct ServerLocation {
    pub server_root: Option<PathBuf>,
    pub server_exe: Option<PathBuf>,
    pub python_exe: Option<PathBuf>,
    pub script_path: Option<PathBuf>,
    pub launcher_script: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct ServerBeacon {
    pub server_root: Option<String>,
    pub python_exe: Option<String>,
    pub script_path: Option<String>,
    pub server_exe: Option<String>,
    pub port: Option<u16>,
    pub managed: Option<bool>,
}

/// Resolve the current user's home directory across Windows and POSIX.
pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        if let Ok(profile) = std::env::var("USERPROFILE") {
            if !profile.trim().is_empty() {
                return Some(PathBuf::from(profile));
            }
        }
        if let (Ok(drive), Ok(path)) = (std::env::var("HOMEDRIVE"), std::env::var("HOMEPATH")) {
            let combined = format!("{drive}{path}");
            if !combined.trim().is_empty() {
                return Some(PathBuf::from(combined));
            }
        }
        std::env::var("HOME")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
    }
}

pub fn beacon_file_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".hifisampler").join("server.json"))
}

pub fn read_beacon() -> Option<ServerBeacon> {
    let path = beacon_file_path()?;
    read_beacon_from_path(&path)
}

pub fn read_beacon_from_path(path: &Path) -> Option<ServerBeacon> {
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<ServerBeacon>(&content).ok()
}

/// Resolve target HTTP port with priority:
/// 1. `HIFISAMPLER_PORT` environment variable
/// 2. `~/.hifisampler/server.json` `"port"`
/// 3. Optional `"port"` in `config.yaml` (in `HIFISAMPLER_HOME` / exe dir / cwd)
/// 4. Default `8572`
pub fn resolve_target_port() -> u16 {
    let beacon = read_beacon();
    let config_port = find_port_in_nearby_config();
    resolve_target_port_with(
        std::env::var("HIFISAMPLER_PORT").ok().as_deref(),
        beacon.as_ref().and_then(|b| b.port),
        config_port,
    )
}

pub fn resolve_target_port_with(
    env_port: Option<&str>,
    beacon_port: Option<u16>,
    config_port: Option<u16>,
) -> u16 {
    if let Some(raw) = env_port {
        if let Ok(port) = raw.trim().parse::<u16>() {
            if port > 0 {
                return port;
            }
        }
    }
    if let Some(port) = beacon_port {
        if port > 0 {
            return port;
        }
    }
    if let Some(port) = config_port {
        if port > 0 {
            return port;
        }
    }
    DEFAULT_PORT
}

fn find_port_in_nearby_config() -> Option<u16> {
    let mut dirs = Vec::new();
    if let Ok(home) = std::env::var("HIFISAMPLER_HOME") {
        if !home.trim().is_empty() {
            dirs.push(PathBuf::from(home));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd);
    }

    for dir in dirs {
        let cfg = dir.join("config.yaml");
        if cfg.is_file() {
            if let Some(port) = parse_yaml_port_line(&cfg) {
                return Some(port);
            }
        }
    }
    None
}

fn parse_yaml_port_line(path: &Path) -> Option<u16> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let trimmed = line.split('#').next().unwrap_or("").trim();
        if let Some(rest) = trimmed.strip_prefix("port:") {
            if let Ok(p) = rest
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .parse::<u16>()
            {
                if p > 0 {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn server_bin_name() -> &'static str {
    if cfg!(windows) {
        "hifiserver.exe"
    } else {
        "hifiserver"
    }
}

fn launcher_script_name() -> &'static str {
    if cfg!(windows) {
        "start.bat"
    } else {
        "start.sh"
    }
}

fn is_server_root_dir(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    dir.join("config.yaml").is_file()
        || dir.join("config.default.yaml").is_file()
        || dir.join("hifiserver.py").is_file()
        || dir.join(launcher_script_name()).is_file()
        || dir
            .join("pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
            .is_dir()
}

fn find_server_exe_in_candidates(
    exe_dir: Option<&Path>,
    server_root: Option<&Path>,
    beacon: Option<&ServerBeacon>,
) -> Option<PathBuf> {
    let bin = server_bin_name();

    if let Ok(env_bin) = std::env::var("HIFISAMPLER_SERVER_EXE") {
        let p = PathBuf::from(env_bin.trim());
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(dir) = exe_dir {
        let cand = dir.join(bin);
        if cand.is_file() {
            return Some(cand);
        }
    }
    if let Some(root) = server_root {
        let direct = root.join(bin);
        if direct.is_file() {
            return Some(direct);
        }
        let release = root.join("rust").join("target").join("release").join(bin);
        if release.is_file() {
            return Some(release);
        }
        let debug = root.join("rust").join("target").join("debug").join(bin);
        if debug.is_file() {
            return Some(debug);
        }
    }
    if let Some(b) = beacon {
        if let Some(ref s_exe) = b.server_exe {
            let p = PathBuf::from(s_exe);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let s = path.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{rest}"));
        }
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return PathBuf::from(rest);
        }
    }
    path
}

fn canonicalize_clean(path: &Path) -> PathBuf {
    let p = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    strip_verbatim_prefix(p)
}

fn probe_server_dir(
    dir: &Path,
    exe_dir: Option<&Path>,
    beacon: Option<&ServerBeacon>,
) -> Option<ServerLocation> {
    if !is_server_root_dir(dir) {
        return None;
    }
    let full_dir = canonicalize_clean(dir);
    let script_path = full_dir.join("hifiserver.py");
    let launcher_path = full_dir.join(launcher_script_name());

    let portable_py = if cfg!(windows) {
        full_dir.join("env").join("python.exe")
    } else {
        full_dir.join("env").join("bin").join("python")
    };
    let venv_py = if cfg!(windows) {
        full_dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        full_dir.join(".venv").join("bin").join("python")
    };

    let mut python_exe = if portable_py.is_file() {
        Some(portable_py)
    } else if venv_py.is_file() {
        Some(venv_py)
    } else {
        None
    };

    if python_exe.is_none() {
        if let Some(b) = beacon {
            if let (Some(ref b_root), Some(ref b_py)) = (&b.server_root, &b.python_exe) {
                let b_py_path = PathBuf::from(b_py);
                let b_root_canon = canonicalize_clean(Path::new(b_root));
                if b_py_path.is_file() && b_root_canon == full_dir {
                    python_exe = Some(b_py_path);
                }
            }
        }
    }

    let server_exe = find_server_exe_in_candidates(exe_dir, Some(&full_dir), beacon);

    Some(ServerLocation {
        server_root: Some(full_dir),
        server_exe,
        python_exe,
        script_path: script_path.is_file().then_some(script_path),
        launcher_script: launcher_path.is_file().then_some(launcher_path),
    })
}

/// Resolve the server root directory and executable/script paths using:
/// 1. `HIFISAMPLER_HOME`
/// 2. Symlink target directory of current executable
/// 3. Executable's own directory (or its ancestors if inside `target/release`)
/// 4. Global locator beacon `~/.hifisampler/server.json`
/// 5. Current working directory
pub fn resolve_server_location() -> ServerLocation {
    let beacon = read_beacon();
    let current_exe = std::env::current_exe().ok();
    let exe_dir = current_exe
        .as_ref()
        .and_then(|p| p.parent().map(Path::to_path_buf));

    if let Ok(env_home) = std::env::var("HIFISAMPLER_HOME") {
        let p = PathBuf::from(env_home.trim());
        if !env_home.trim().is_empty() {
            if let Some(loc) = probe_server_dir(&p, exe_dir.as_deref(), beacon.as_ref()) {
                return loc;
            }
        }
    }

    if let Some(ref exe) = current_exe {
        if let Ok(target) = fs::canonicalize(exe) {
            if let Some(link_dir) = target.parent() {
                if let Some(loc) = probe_server_dir(link_dir, exe_dir.as_deref(), beacon.as_ref()) {
                    return loc;
                }
            }
        }
    }

    if let Some(ref dir) = exe_dir {
        if let Some(loc) = probe_server_dir(dir, exe_dir.as_deref(), beacon.as_ref()) {
            return loc;
        }
    }

    if let Some(ref b) = beacon {
        if let Some(ref sr) = b.server_root {
            if !sr.trim().is_empty() {
                if let Some(mut loc) =
                    probe_server_dir(Path::new(sr), exe_dir.as_deref(), beacon.as_ref())
                {
                    if loc.python_exe.is_none() {
                        if let Some(ref py) = b.python_exe {
                            let py_path = PathBuf::from(py);
                            if py_path.is_file() {
                                loc.python_exe = Some(py_path);
                            }
                        }
                    }
                    return loc;
                }
            }
        }
    }

    // Walk up ancestors from exe_dir (handles rust/target/release/hifisampler.exe)
    if let Some(ref dir) = exe_dir {
        let mut cursor = dir.parent();
        for _ in 0..4 {
            if let Some(anc) = cursor {
                if let Some(loc) = probe_server_dir(anc, exe_dir.as_deref(), beacon.as_ref()) {
                    return loc;
                }
                cursor = anc.parent();
            } else {
                break;
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if let Some(loc) = probe_server_dir(&cwd, exe_dir.as_deref(), beacon.as_ref()) {
            return loc;
        }
    }

    // Fallback: if only server_exe exists next to client exe, still allow launching
    if let Some(server_exe) =
        find_server_exe_in_candidates(exe_dir.as_deref(), None, beacon.as_ref())
    {
        let fallback_root = exe_dir.clone().or_else(|| std::env::current_dir().ok());
        return ServerLocation {
            server_root: fallback_root,
            server_exe: Some(server_exe),
            python_exe: None,
            script_path: None,
            launcher_script: None,
        };
    }

    ServerLocation::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_resolution_priority() {
        assert_eq!(
            resolve_target_port_with(Some("9100"), Some(8600), Some(8700)),
            9100
        );
        assert_eq!(resolve_target_port_with(None, Some(8600), Some(8700)), 8600);
        assert_eq!(
            resolve_target_port_with(Some("invalid"), Some(8600), Some(8700)),
            8600
        );
        assert_eq!(resolve_target_port_with(None, None, Some(8700)), 8700);
        assert_eq!(resolve_target_port_with(None, None, None), 8572);
    }
}
