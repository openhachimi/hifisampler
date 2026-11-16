use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Resolve the per-port single-instance lockfile path matching `hifiserver.py:get_lock_file_path`.
pub fn get_lock_file_path(port: u16) -> PathBuf {
    let uid = resolve_uid_tag();
    let safe_uid: String = uid
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    std::env::temp_dir().join(format!("hifisampler-{port}-{safe_uid}.lock"))
}

fn resolve_uid_tag() -> String {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn getuid() -> u32;
        }
        let uid = unsafe { getuid() };
        uid.to_string()
    }
    #[cfg(not(unix))]
    {
        std::env::var("USERNAME")
            .or_else(|_| std::env::var("USER"))
            .unwrap_or_else(|_| "default".to_string())
    }
}

/// Exclusive single-instance advisory file lock held for the lifetime of the server.
pub struct ServerInstanceLock {
    _file: File,
    path: PathBuf,
}

impl ServerInstanceLock {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Try to acquire the exclusive instance lock within `timeout`.
    /// Returns `Ok(None)` when another server instance already holds the lock.
    pub fn try_acquire(port: u16, timeout: Duration) -> std::io::Result<Option<Self>> {
        let path = get_lock_file_path(port);
        Self::try_acquire_at(path, timeout)
    }

    pub fn try_acquire_at(path: PathBuf, timeout: Duration) -> std::io::Result<Option<Self>> {
        let deadline = Instant::now() + timeout;
        loop {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;

            match file.try_lock() {
                Ok(()) => {
                    return Ok(Some(Self { _file: file, path }));
                }
                Err(std::fs::TryLockError::WouldBlock) => {
                    if Instant::now() >= deadline {
                        return Ok(None);
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(std::fs::TryLockError::Error(e)) => {
                    return Err(e);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn lock_filename_contains_port_and_extension() {
        let p = get_lock_file_path(8599);
        let name = p.file_name().unwrap().to_string_lossy();
        assert!(name.contains("hifisampler-8599-"));
        assert!(name.ends_with(".lock"));
    }

    #[test]
    fn single_instance_lock_is_exclusive() {
        let path =
            std::env::temp_dir().join(format!("hifi_lock_test_{}.lock", Uuid::new_v4().simple()));
        let guard = ServerInstanceLock::try_acquire_at(path.clone(), Duration::from_millis(200))
            .unwrap()
            .expect("first lock should succeed");

        let second =
            ServerInstanceLock::try_acquire_at(path.clone(), Duration::from_millis(75)).unwrap();
        assert!(
            second.is_none(),
            "second lock must fail while first is held"
        );

        drop(guard);
        let third =
            ServerInstanceLock::try_acquire_at(path.clone(), Duration::from_millis(200)).unwrap();
        assert!(
            third.is_some(),
            "lock should succeed after first is dropped"
        );
        drop(third);
        let _ = std::fs::remove_file(&path);
    }
}
