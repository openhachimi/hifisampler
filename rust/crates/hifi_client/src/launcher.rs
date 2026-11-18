use crate::config::ServerLocation;
use crate::http::{http_request_loopback, HttpError};
use std::fs::{File, OpenOptions};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const MANAGED_IDLE_TIMEOUT_SECS: u64 = 600;
const STARTUP_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const SERVER_WAIT_TIMEOUT: Duration = Duration::from_secs(45);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastPathOutcome {
    Succeeded,
    NeedsEnsureServer,
    FatalError,
}

pub fn try_optimistic_post(port: u16, payload: &str, media_type: &str) -> FastPathOutcome {
    try_optimistic_post_path(port, "/", payload, media_type)
}

pub fn try_optimistic_post_path(
    port: u16,
    path: &str,
    payload: &str,
    media_type: &str,
) -> FastPathOutcome {
    let t0 = Instant::now();
    match http_request_loopback(
        port,
        "POST",
        path,
        Some((payload, media_type)),
        Duration::from_millis(300),
        Duration::from_secs(180),
    ) {
        Ok(resp) => {
            if (200..300).contains(&resp.status) {
                FastPathOutcome::Succeeded
            } else if resp.status == 503 {
                FastPathOutcome::NeedsEnsureServer
            } else {
                eprintln!(
                    "[Communicate] Error: Status {} {} - {}",
                    resp.status, resp.reason, resp.body
                );
                FastPathOutcome::FatalError
            }
        }
        Err(HttpError::ConnectRefusedOrUnreachable(_)) | Err(HttpError::ConnectTimeout) => {
            FastPathOutcome::NeedsEnsureServer
        }
        Err(ref err) if t0.elapsed() < Duration::from_millis(1500) => {
            let _ = err;
            FastPathOutcome::NeedsEnsureServer
        }
        Err(err) => {
            eprintln!(
                "[Communicate] Request failed after {}ms: {err}",
                t0.elapsed().as_millis()
            );
            FastPathOutcome::FatalError
        }
    }
}

pub fn is_server_ready(port: u16) -> bool {
    matches!(
        http_request_loopback(
            port,
            "GET",
            "/",
            None,
            Duration::from_millis(500),
            Duration::from_secs(2),
        ),
        Ok(resp) if resp.status == 200
    )
}

pub fn is_port_in_use(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(150)).is_ok()
}

pub fn startup_lock_path(port: u16) -> PathBuf {
    let raw_user = std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .unwrap_or_else(|_| "user".to_string());
    let safe_user: String = raw_user
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    std::env::temp_dir().join(format!("melos-client-startup-{port}-{safe_user}.lock"))
}

pub fn try_acquire_startup_lock(port: u16, timeout: Duration) -> Option<File> {
    let path = startup_lock_path(port);
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(file) = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
        {
            if file.try_lock().is_ok() {
                return Some(file);
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

struct SpawnedServerProcess {
    #[cfg(windows)]
    process_handle: windows_sys::Win32::Foundation::HANDLE,
    working_dir: PathBuf,
}

impl SpawnedServerProcess {
    fn poll_exit_code(&self) -> Option<u32> {
        #[cfg(windows)]
        unsafe {
            use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
            use windows_sys::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject};

            if !self.process_handle.is_null()
                && WaitForSingleObject(self.process_handle, 0) == WAIT_OBJECT_0
            {
                let mut code: u32 = 0;
                if GetExitCodeProcess(self.process_handle, &mut code) != 0 {
                    return Some(code);
                }
                return Some(1);
            }
            None
        }
        #[cfg(not(windows))]
        {
            None
        }
    }
}

#[cfg(windows)]
impl Drop for SpawnedServerProcess {
    fn drop(&mut self) {
        if !self.process_handle.is_null() {
            unsafe {
                windows_sys::Win32::Foundation::CloseHandle(self.process_handle);
            }
            self.process_handle = std::ptr::null_mut();
        }
    }
}

fn wait_for_server_with_child(
    port: u16,
    timeout: Duration,
    log_prefix: &str,
    child: Option<&SpawnedServerProcess>,
) -> Result<(), String> {
    println!(
        "{log_prefix} Waiting up to {}s for server at port {port}...",
        timeout.as_secs()
    );
    let t0 = Instant::now();
    let deadline = t0 + timeout;
    while Instant::now() < deadline {
        if is_server_ready(port) {
            println!(
                "{log_prefix} Server became ready after {:.1}s.",
                t0.elapsed().as_secs_f64()
            );
            return Ok(());
        }
        if let Some(proc_guard) = child {
            if let Some(code) = proc_guard.poll_exit_code() {
                if code != 0 {
                    return Err(format!(
                        "hifiserver exited prematurely with code {code} (check logs/hifiserver.log in '{}').",
                        proc_guard.working_dir.display()
                    ));
                }
            }
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    println!(
        "{log_prefix} Server did not become ready within {}s.",
        timeout.as_secs()
    );
    Err(format!(
        "Server launched (warming up in background), but did not become ready within {}s.",
        timeout.as_secs()
    ))
}

pub fn wait_for_server_to_start(port: u16, timeout: Duration, log_prefix: &str) -> bool {
    wait_for_server_with_child(port, timeout, log_prefix, None).is_ok()
}

pub fn ensure_server_running(port: u16, loc: &ServerLocation) -> Result<(), String> {
    ensure_server_running_with_mode(port, loc, false)
}

pub fn ensure_server_running_with_mode(
    port: u16,
    loc: &ServerLocation,
    require_native_server: bool,
) -> Result<(), String> {
    if is_server_ready(port) {
        return Ok(());
    }

    println!("[Main] Server not ready. Acquiring startup lock...");
    let _lock = try_acquire_startup_lock(port, STARTUP_LOCK_TIMEOUT);

    let mut proceed_to_communication = false;
    if _lock.is_some() {
        if is_server_ready(port) {
            proceed_to_communication = true;
        } else if !is_port_in_use(port) {
            println!("[Main] Port {port} is free. Launching Melos server...");
            if loc.server_root.is_none() && loc.server_exe.is_none() {
                return Err(
                    "Could not locate Melos server directory. Please run hifiserver --register-only once in the server folder to register ~/.melos/server.json, or set MELOS_HOME."
                        .to_string(),
                );
            }
            let child = launch_server_process_with_mode(loc, port, require_native_server)?;
            wait_for_server_with_child(port, SERVER_WAIT_TIMEOUT, "[InitialWait]", Some(&child))?;
            proceed_to_communication = true;
        } else {
            println!("[Main] Port {port} is in use; waiting for server initialization...");
        }
    } else {
        println!("[Main] Another client instance holds the startup lock; waiting for server...");
    }

    if !proceed_to_communication
        && wait_for_server_with_child(port, SERVER_WAIT_TIMEOUT, "[FinalWait]", None).is_err()
    {
        return Err("Error: Server failed readiness check within timeout.".to_string());
    }
    Ok(())
}

pub fn communicate_with_server(port: u16, payload: &str, media_type: &str) -> Result<(), String> {
    communicate_with_server_path(port, "/", payload, media_type)
}

pub fn communicate_with_server_path(
    port: u16,
    path: &str,
    payload: &str,
    media_type: &str,
) -> Result<(), String> {
    let max_retries = 2usize;
    let retry_delay = Duration::from_millis(800);

    for attempt in 1..=(max_retries + 1) {
        println!(
            "[Communicate] Sending POST {path} (Attempt {attempt}/{})...",
            max_retries + 1
        );
        match http_request_loopback(
            port,
            "POST",
            path,
            Some((payload, media_type)),
            Duration::from_secs(2),
            Duration::from_secs(180),
        ) {
            Ok(resp) if (200..300).contains(&resp.status) => {
                println!(
                    "[Communicate] Success (Attempt {attempt}, Status: {})",
                    resp.status
                );
                return Ok(());
            }
            Ok(resp) => {
                eprintln!(
                    "[Communicate] Error (Attempt {attempt}): Status {} {} - {}",
                    resp.status, resp.reason, resp.body
                );
                if resp.status == 503 && attempt <= max_retries {
                    println!("[Communicate] Server still initializing (503), retrying...");
                } else {
                    break;
                }
            }
            Err(HttpError::ReadTimeout) => {
                eprintln!("[Communicate] Request Timed Out (Attempt {attempt}).");
                break;
            }
            Err(err) => {
                eprintln!("[Communicate] Network Error (Attempt {attempt}): {err}");
            }
        }

        if attempt <= max_retries {
            std::thread::sleep(retry_delay);
        }
    }

    Err("[Communicate] Failed after all attempts.".to_string())
}

pub fn launch_server_process(loc: &ServerLocation, port: u16) -> Result<(), String> {
    launch_server_process_with_mode(loc, port, false).map(|_| ())
}

#[cfg(windows)]
fn launch_server_process_with_mode(
    loc: &ServerLocation,
    port: u16,
    require_native: bool,
) -> Result<SpawnedServerProcess, String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        CreateProcessW, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, PROCESS_INFORMATION,
        STARTUPINFOW,
    };

    let managed_args =
        format!("--managed --idle-timeout {MANAGED_IDLE_TIMEOUT_SECS} --port {port}");
    let working_dir = loc
        .server_root
        .clone()
        .or_else(|| {
            loc.server_exe
                .as_ref()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        })
        .ok_or_else(|| "Missing server working directory".to_string())?;

    let cmd_line = if let Some(ref server_exe) = loc.server_exe {
        format!("\"{}\" {managed_args}", server_exe.display())
    } else if require_native {
        return Err(
            "Native hifiserver.exe binary was not found. Please run start.bat or 'hifiserver.exe --register-only' in your Melos installation folder to register ~/.melos/server.json (or set MELOS_HOME)."
                .to_string(),
        );
    } else if let (Some(ref py), Some(ref script)) = (&loc.python_exe, &loc.script_path) {
        format!(
            "\"{}\" \"{}\" {managed_args}",
            py.display(),
            script.display()
        )
    } else if let Some(ref launcher) = loc.launcher_script {
        format!("cmd.exe /C \"\"{}\" {managed_args}\"", launcher.display())
    } else {
        return Err(
            "No hifiserver binary, Python script, or start.bat launcher found.".to_string(),
        );
    };

    let mut cmd_utf16: Vec<u16> = std::ffi::OsStr::new(&cmd_line)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let work_utf16: Vec<u16> = working_dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let title_utf16: Vec<u16> = std::ffi::OsStr::new("Melos Server")
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let mut si: STARTUPINFOW = unsafe { std::mem::zeroed() };
    si.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
    si.lpTitle = title_utf16.as_ptr() as *mut u16;

    let mut pi: PROCESS_INFORMATION = unsafe { std::mem::zeroed() };

    // CRITICAL: bInheritHandles = 0 (FALSE) prevents inheriting OpenUtau's redirected pipe handles.
    let created = unsafe {
        CreateProcessW(
            std::ptr::null(),
            cmd_utf16.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
            std::ptr::null(),
            work_utf16.as_ptr(),
            &si,
            &mut pi,
        )
    };

    if created == 0 {
        let err = std::io::Error::last_os_error();
        return Err(format!(
            "CreateProcessW failed ({err}) when launching '{cmd_line}'."
        ));
    }

    let pid = pi.dwProcessId;
    unsafe {
        if !pi.hThread.is_null() {
            CloseHandle(pi.hThread);
        }
    }

    println!(
        "[LaunchServer] Started detached Windows server (PID {pid}) in '{}'.",
        working_dir.display()
    );
    Ok(SpawnedServerProcess {
        process_handle: pi.hProcess,
        working_dir,
    })
}

#[cfg(not(windows))]
fn launch_server_process_with_mode(
    loc: &ServerLocation,
    port: u16,
    require_native: bool,
) -> Result<SpawnedServerProcess, String> {
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let working_dir = loc
        .server_root
        .clone()
        .or_else(|| {
            loc.server_exe
                .as_ref()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        })
        .ok_or_else(|| "Missing server working directory".to_string())?;

    let log_dir = working_dir.join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_file_path = log_dir.join("hifiserver.log");
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
        .map_err(|e| format!("Failed to open server log {}: {e}", log_file_path.display()))?;
    let log_err = log_file
        .try_clone()
        .map_err(|e| format!("Failed to clone log fd: {e}"))?;

    let mut cmd = if let Some(ref server_exe) = loc.server_exe {
        let mut c = Command::new(server_exe);
        c.args([
            "--managed",
            "--idle-timeout",
            &MANAGED_IDLE_TIMEOUT_SECS.to_string(),
            "--port",
            &port.to_string(),
        ]);
        c
    } else if require_native {
        return Err(
            "Native hifiserver binary was not found. Please run 'hifiserver --register-only' in your Melos installation folder to register ~/.melos/server.json (or set MELOS_HOME)."
                .to_string(),
        );
    } else if let (Some(ref py), Some(ref script)) = (&loc.python_exe, &loc.script_path) {
        let mut c = Command::new(py);
        c.arg(script).args([
            "--managed",
            "--idle-timeout",
            &MANAGED_IDLE_TIMEOUT_SECS.to_string(),
            "--port",
            &port.to_string(),
        ]);
        c
    } else if let Some(ref launcher) = loc.launcher_script {
        let mut c = Command::new("bash");
        c.arg(launcher).args([
            "--managed",
            "--idle-timeout",
            &MANAGED_IDLE_TIMEOUT_SECS.to_string(),
            "--port",
            &port.to_string(),
        ]);
        c
    } else {
        return Err("No hifiserver binary, Python script, or start.sh launcher found.".to_string());
    };

    cmd.current_dir(&working_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_err));

    unsafe {
        unsafe extern "C" {
            fn setsid() -> i32;
        }
        cmd.pre_exec(|| {
            setsid();
            Ok(())
        });
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn server process: {e}"))?;
    println!(
        "[LaunchServer] Launched detached background server (PID {}, log: {}).",
        child.id(),
        log_file_path.display()
    );
    Ok(SpawnedServerProcess { working_dir })
}
