use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LocatorBeacon {
    pub server_root: String,
    pub python_exe: String,
    pub script_path: String,
    pub server_exe: String,
    pub port: u16,
    pub managed: bool,
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

pub fn default_beacon_path() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".hifisampler").join("server.json"))
}

pub fn resolve_venv_python(root: &Path) -> Option<PathBuf> {
    let candidates = if cfg!(windows) {
        [
            root.join("env").join("python.exe"),
            root.join(".venv").join("Scripts").join("python.exe"),
        ]
    } else {
        [
            root.join("env").join("bin").join("python"),
            root.join(".venv").join("bin").join("python"),
        ]
    };
    candidates.into_iter().find(|p| p.is_file())
}

/// Atomically write `~/.hifisampler/server.json`.
/// Note: Intentionally preserved on shutdown so copied clients can discover `server_root`.
pub fn write_locator_beacon(
    server_root: &Path,
    port: u16,
    managed: bool,
) -> std::io::Result<PathBuf> {
    let target = default_beacon_path().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Unable to determine user home directory for ~/.hifisampler/server.json",
        )
    })?;
    write_locator_beacon_to(&target, server_root, port, managed)?;
    Ok(target)
}

pub fn write_locator_beacon_to(
    beacon_file: &Path,
    server_root: &Path,
    port: u16,
    managed: bool,
) -> std::io::Result<()> {
    let beacon_dir = beacon_file.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Beacon path has no parent",
        )
    })?;
    fs::create_dir_all(beacon_dir)?;

    let current_exe = std::env::current_exe()
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let py_exe = resolve_venv_python(server_root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let py_script = server_root.join("hifiserver.py");
    let script_path = if py_script.is_file() {
        py_script.to_string_lossy().into_owned()
    } else {
        current_exe.clone()
    };

    let payload = LocatorBeacon {
        server_root: server_root.to_string_lossy().into_owned(),
        python_exe: py_exe,
        script_path,
        server_exe: current_exe,
        port,
        managed,
    };

    let json_text = serde_json::to_string_pretty(&payload).map_err(std::io::Error::other)?;

    let tmp_path = beacon_dir.join(format!(".server_{}.json.tmp", Uuid::new_v4().simple()));
    struct TmpGuard<'a>(&'a Path);
    impl<'a> Drop for TmpGuard<'a> {
        fn drop(&mut self) {
            if self.0.exists() {
                let _ = fs::remove_file(self.0);
            }
        }
    }
    let guard = TmpGuard(&tmp_path);

    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(json_text.as_bytes())?;
        f.flush()?;
        f.sync_all()?;
    }

    fs::rename(&tmp_path, beacon_file)?;
    std::mem::forget(guard);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_beacon_write_contains_all_required_fields() {
        let dir =
            std::env::temp_dir().join(format!("hifi_beacon_test_{}", Uuid::new_v4().simple()));
        let beacon_path = dir.join(".hifisampler").join("server.json");
        write_locator_beacon_to(&beacon_path, &dir, 8999, true).unwrap();

        let text = fs::read_to_string(&beacon_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(parsed.get("server_root").is_some());
        assert!(parsed.get("python_exe").is_some());
        assert!(parsed.get("script_path").is_some());
        assert_eq!(parsed.get("port").and_then(|v| v.as_u64()), Some(8999));
        assert_eq!(parsed.get("managed").and_then(|v| v.as_bool()), Some(true));

        let _ = fs::remove_dir_all(&dir);
    }
}
