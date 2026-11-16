use crate::config::ServerConfig;
use std::path::PathBuf;

pub const DEFAULT_PORT: u16 = 8572;
pub const DEFAULT_MANAGED_IDLE_TIMEOUT: u64 = 600;

/// Parsed CLI options for `hifiserver`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerCliArgs {
    pub register_only: bool,
    pub managed: bool,
    pub idle_timeout: Option<u64>,
    pub port: u16,
    pub host: String,
    pub config: Option<PathBuf>,
    pub parent_pid: Option<u32>,
}

/// Parse `hifiserver` CLI arguments with `HIFISAMPLER_PORT` fallback (CLI `--port` wins).
pub fn parse_server_cli_args_from(
    argv: &[String],
    env_port: Option<&str>,
) -> Result<ServerCliArgs, String> {
    let mut register_only = false;
    let mut managed = false;
    let mut idle_timeout_raw: Option<i64> = None;
    let mut port_raw: Option<i64> = None;
    let mut host = "127.0.0.1".to_string();
    let mut config: Option<PathBuf> = None;
    let mut parent_pid: Option<u32> = None;

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-h" | "--help" => {
                return Err(help_text());
            }
            "--register-only" => {
                register_only = true;
            }
            "--managed" => {
                managed = true;
            }
            "--idle-timeout" | "--idle-timeout-secs" => {
                i += 1;
                let val = argv
                    .get(i)
                    .ok_or_else(|| format!("{arg} requires an integer value"))?;
                idle_timeout_raw = Some(
                    val.parse::<i64>()
                        .map_err(|_| format!("Invalid integer for {arg}: {val}"))?,
                );
            }
            _ if arg.starts_with("--idle-timeout=") || arg.starts_with("--idle-timeout-secs=") => {
                let (_, val) = arg.split_once('=').unwrap();
                idle_timeout_raw = Some(
                    val.parse::<i64>()
                        .map_err(|_| format!("Invalid integer for --idle-timeout: {val}"))?,
                );
            }
            "-p" | "--port" => {
                i += 1;
                let val = argv
                    .get(i)
                    .ok_or_else(|| format!("{arg} requires a port number"))?;
                port_raw = Some(
                    val.parse::<i64>()
                        .map_err(|_| format!("Invalid integer for {arg}: {val}"))?,
                );
            }
            _ if arg.starts_with("--port=") => {
                let (_, val) = arg.split_once('=').unwrap();
                port_raw = Some(
                    val.parse::<i64>()
                        .map_err(|_| format!("Invalid integer for --port: {val}"))?,
                );
            }
            "--host" => {
                i += 1;
                let val = argv
                    .get(i)
                    .ok_or_else(|| "--host requires a bind address".to_string())?;
                host = val.clone();
            }
            _ if arg.starts_with("--host=") => {
                let (_, val) = arg.split_once('=').unwrap();
                host = val.to_string();
            }
            "--config" => {
                i += 1;
                let val = argv
                    .get(i)
                    .ok_or_else(|| "--config requires a file path".to_string())?;
                config = Some(PathBuf::from(val));
            }
            _ if arg.starts_with("--config=") => {
                let (_, val) = arg.split_once('=').unwrap();
                config = Some(PathBuf::from(val));
            }
            "--parent-pid" => {
                i += 1;
                let val = argv
                    .get(i)
                    .ok_or_else(|| "--parent-pid requires a PID".to_string())?;
                let pid = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid PID for --parent-pid: {val}"))?;
                if pid == 0 {
                    return Err("--parent-pid must be > 0".to_string());
                }
                parent_pid = Some(pid);
            }
            _ if arg.starts_with("--parent-pid=") => {
                let (_, val) = arg.split_once('=').unwrap();
                let pid = val
                    .parse::<u32>()
                    .map_err(|_| format!("Invalid PID for --parent-pid: {val}"))?;
                if pid == 0 {
                    return Err("--parent-pid must be > 0".to_string());
                }
                parent_pid = Some(pid);
            }
            other => {
                return Err(format!("Unrecognized argument: {other}"));
            }
        }
        i += 1;
    }

    let resolved_port = match port_raw {
        Some(p) => p,
        None => {
            if let Some(raw) = env_port.map(str::trim).filter(|s| !s.is_empty()) {
                raw.parse::<i64>()
                    .map_err(|_| format!("Invalid HIFISAMPLER_PORT: {raw}"))?
            } else {
                DEFAULT_PORT as i64
            }
        }
    };

    if !(1..=65535).contains(&resolved_port) {
        return Err(format!(
            "--port must be between 1 and 65535, got {resolved_port}"
        ));
    }

    let idle_timeout = match idle_timeout_raw {
        Some(v) if v < 0 => {
            return Err(format!("--idle-timeout must be >= 0, got {v}"));
        }
        Some(v) => Some(v as u64),
        None => None,
    };

    Ok(ServerCliArgs {
        register_only,
        managed,
        idle_timeout,
        port: resolved_port as u16,
        host,
        config,
        parent_pid,
    })
}

pub fn parse_server_cli_args() -> Result<ServerCliArgs, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let env_port = std::env::var("HIFISAMPLER_PORT").ok();
    parse_server_cli_args_from(&argv, env_port.as_deref())
}

/// Apply `--managed` / `--idle-timeout` overrides in memory without writing `config.yaml`.
pub fn apply_managed_runtime_config(
    cfg: &mut ServerConfig,
    managed: bool,
    idle_timeout: Option<u64>,
) -> u64 {
    let current_timeout = cfg.idle_timeout;

    if let Some(cli_timeout) = idle_timeout {
        let mut effective = cli_timeout;
        if managed && effective == 0 {
            effective = DEFAULT_MANAGED_IDLE_TIMEOUT;
        }
        cfg.idle_timeout = effective;
        return effective;
    }

    if managed {
        if current_timeout == 0 {
            cfg.idle_timeout = DEFAULT_MANAGED_IDLE_TIMEOUT;
            return DEFAULT_MANAGED_IDLE_TIMEOUT;
        }
        return current_timeout;
    }

    current_timeout
}

fn help_text() -> String {
    "HifiSampler UTAU inference server\n\n\
     Options:\n  \
       -p, --port <PORT>            HTTP listening port (1..65535, default 8572)\n  \
           --host <HOST>            Bind address (default 127.0.0.1)\n  \
           --config <PATH>          Path to YAML config file\n  \
           --managed                Run in client-managed lifecycle mode\n  \
           --idle-timeout <SECS>    Override idle auto-shutdown timeout in seconds\n  \
           --idle-timeout-secs <S>  Alias for --idle-timeout\n  \
           --parent-pid <PID>       Watch parent PID and exit when it terminates\n  \
           --register-only          Write ~/.hifisampler/server.json beacon and exit"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_and_env_override() {
        let a = parse_server_cli_args_from(&[], None).unwrap();
        assert_eq!(a.port, 8572);
        assert!(!a.managed);
        assert!(!a.register_only);
        assert_eq!(a.idle_timeout, None);

        let b = parse_server_cli_args_from(&[], Some("9100")).unwrap();
        assert_eq!(b.port, 9100);

        let c =
            parse_server_cli_args_from(&["--port".into(), "9200".into()], Some("9100")).unwrap();
        assert_eq!(c.port, 9200);
    }

    #[test]
    fn cli_validation_and_aliases() {
        let a = parse_server_cli_args_from(
            &[
                "--register-only".into(),
                "--managed".into(),
                "--idle-timeout-secs".into(),
                "720".into(),
                "--port".into(),
                "8600".into(),
                "--parent-pid".into(),
                "1234".into(),
            ],
            None,
        )
        .unwrap();
        assert!(a.register_only);
        assert!(a.managed);
        assert_eq!(a.idle_timeout, Some(720));
        assert_eq!(a.port, 8600);
        assert_eq!(a.parent_pid, Some(1234));

        assert!(parse_server_cli_args_from(&["--port".into(), "0".into()], None).is_err());
        assert!(parse_server_cli_args_from(&["--port".into(), "70000".into()], None).is_err());
        assert!(parse_server_cli_args_from(&["--idle-timeout".into(), "-5".into()], None).is_err());
    }

    #[test]
    fn managed_runtime_timeout_resolution() {
        let mut cfg = ServerConfig::default();
        assert_eq!(apply_managed_runtime_config(&mut cfg, false, None), 0);
        assert_eq!(apply_managed_runtime_config(&mut cfg, true, None), 600);
        assert_eq!(cfg.idle_timeout, 600);

        cfg.idle_timeout = 300;
        assert_eq!(apply_managed_runtime_config(&mut cfg, true, None), 300);
        assert_eq!(apply_managed_runtime_config(&mut cfg, true, Some(900)), 900);

        cfg.idle_timeout = 0;
        assert_eq!(apply_managed_runtime_config(&mut cfg, true, Some(0)), 600);
        assert_eq!(
            apply_managed_runtime_config(&mut cfg, false, Some(240)),
            240
        );
    }
}
