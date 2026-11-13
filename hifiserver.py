import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

version = '0.1.2'
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8572
DEFAULT_MANAGED_IDLE_TIMEOUT = 600


def parse_server_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for hifiserver.py with HIFISAMPLER_PORT fallback (CLI --port wins)."""
    parser = argparse.ArgumentParser(description="HifiSampler UTAU inference server")
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Write ~/.hifisampler/server.json locator beacon and exit.",
    )
    parser.add_argument(
        "--managed",
        action="store_true",
        help="Run in client-managed lifecycle mode with idle auto-shutdown.",
    )
    parser.add_argument(
        "--idle-timeout",
        "--idle-timeout-secs",
        dest="idle_timeout",
        type=int,
        default=None,
        help="Override runtime idle auto-shutdown timeout in seconds (does not modify config.yaml).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"HTTP listening port (overrides HIFISAMPLER_PORT, default {DEFAULT_PORT}).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.port is None:
        env_port_raw = os.environ.get("HIFISAMPLER_PORT", "").strip()
        args.port = int(env_port_raw) if env_port_raw else DEFAULT_PORT

    if args.port <= 0 or args.port > 65535:
        raise ValueError(f"--port must be between 1 and 65535, got {args.port}")
    if args.idle_timeout is not None and args.idle_timeout < 0:
        raise ValueError(f"--idle-timeout must be >= 0, got {args.idle_timeout}")

    return args


def apply_managed_runtime_config(config_obj, managed: bool = False, idle_timeout: int | None = None) -> int:
    """Apply --managed / --idle-timeout overrides in memory without writing config.yaml."""
    current_timeout = int(getattr(config_obj, "idle_timeout", 0) or 0)

    if idle_timeout is not None:
        effective_timeout = int(idle_timeout)
        if managed and effective_timeout <= 0:
            effective_timeout = DEFAULT_MANAGED_IDLE_TIMEOUT
        config_obj.idle_timeout = effective_timeout
        if managed:
            logging.info(
                f"Managed lifecycle active: runtime idle_timeout={effective_timeout}s "
                f"(from CLI; config.yaml untouched)."
            )
        else:
            logging.info(f"Runtime idle_timeout={effective_timeout}s applied from CLI (config.yaml untouched).")
        return effective_timeout

    if managed:
        if current_timeout <= 0:
            config_obj.idle_timeout = DEFAULT_MANAGED_IDLE_TIMEOUT
            logging.info(
                f"Managed lifecycle active: defaulting runtime idle_timeout to "
                f"{DEFAULT_MANAGED_IDLE_TIMEOUT}s (config.yaml idle_timeout={current_timeout}s untouched)."
            )
            return DEFAULT_MANAGED_IDLE_TIMEOUT
        logging.info(f"Managed lifecycle active: using configured idle_timeout={current_timeout}s.")
        return current_timeout

    return current_timeout


def get_venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        env_py = root / "env" / "python.exe"
        if env_py.exists():
            return env_py
        return root / ".venv" / "Scripts" / "python.exe"
    env_py = root / "env" / "bin" / "python"
    if env_py.exists():
        return env_py
    return root / ".venv" / "bin" / "python"


def write_locator_beacon(port: int = 8572, python_exe: str | None = None, managed: bool = False):
    """Write global locator beacon ~/.hifisampler/server.json so copied clients can find the server root."""
    try:
        beacon_dir = Path.home() / ".hifisampler"
        beacon_dir.mkdir(parents=True, exist_ok=True)
        beacon_file = beacon_dir / "server.json"

        venv_py = get_venv_python(PROJECT_ROOT)
        resolved_py = python_exe or (str(venv_py) if venv_py.exists() else sys.executable)

        payload = {
            "server_root": str(PROJECT_ROOT),
            "python_exe": str(resolved_py),
            "script_path": str(PROJECT_ROOT / "hifiserver.py"),
            "port": int(port),
            "managed": bool(managed),
        }

        fd, tmp_path = tempfile.mkstemp(dir=str(beacon_dir), prefix=".server_", suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, beacon_file)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    except Exception as e:
        logging.debug(f"Could not write locator beacon ~/.hifisampler/server.json: {e}")


def ensure_runtime_env():
    """If launched with system Python while a local .venv exists, transparently switch into .venv."""
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if os.environ.get("HIFISAMPLER_BOOTSTRAPPED") == "1":
        return

    venv_py = get_venv_python(PROJECT_ROOT)
    venv_dir = (PROJECT_ROOT / "env") if (PROJECT_ROOT / "env" / "python.exe").exists() else (PROJECT_ROOT / ".venv")
    if venv_py.exists():
        try:
            curr_prefix = Path(sys.prefix).resolve()
            target_prefix = venv_dir.resolve()
        except OSError:
            curr_prefix = Path(sys.prefix)
            target_prefix = venv_dir

        if curr_prefix != target_prefix:
            os.environ["HIFISAMPLER_BOOTSTRAPPED"] = "1"
            cmd = [str(venv_py), str(PROJECT_ROOT / "hifiserver.py"), *sys.argv[1:]]
            if sys.platform != "win32":
                os.execv(str(venv_py), cmd)
            else:
                sys.exit(subprocess.call(cmd))


def get_lock_file_path(port: int = 8572) -> Path:
    if hasattr(os, "getuid"):
        uid = str(os.getuid())
    else:
        uid = os.environ.get("USERNAME") or os.environ.get("USER") or "default"
    safe_uid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in uid)
    return Path(tempfile.gettempdir()) / f"hifisampler-{port}-{safe_uid}.lock"


if __name__ == '__main__':
    ensure_runtime_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cli_args = parse_server_cli_args(sys.argv[1:])
    port = cli_args.port

    if cli_args.register_only:
        write_locator_beacon(port=port, managed=cli_args.managed)
        print(f"Registered hifisampler locator beacon at: {Path.home() / '.hifisampler' / 'server.json'}")
        sys.exit(0)

    from filelock import FileLock, Timeout
    from config import CONFIG
    from backend import models
    from backend import server

    lock_file_path = get_lock_file_path(port=port)

    try:
        with FileLock(str(lock_file_path), timeout=0.5):
            apply_managed_runtime_config(
                CONFIG,
                managed=cli_args.managed,
                idle_timeout=cli_args.idle_timeout,
            )
            write_locator_beacon(port=port, managed=cli_args.managed)
            logging.info(f"hifisampler {version} (root: {PROJECT_ROOT}, managed={cli_args.managed}, port={port})")
            logging.info(f"Successfully acquired server lock: {lock_file_path}")
            models.initialize_models()
            server.run(port=port)
    except Timeout:
        logging.warning(
            f"Another instance seems to be running (lock file '{lock_file_path}' is held). Exiting.")
        sys.exit(0)
    except Exception as e:
        logging.error(
            f"Failed to initialize or start the server: {e}", exc_info=True)
        sys.exit(1)
