import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

version = '0.0.6-hifisampler'
PROJECT_ROOT = Path(__file__).resolve().parent


def get_venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def write_locator_beacon(port: int = 8572, python_exe: str | None = None):
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
    if venv_py.exists():
        try:
            curr_py = Path(sys.executable).resolve()
            target_py = venv_py.resolve()
        except OSError:
            curr_py = Path(sys.executable)
            target_py = venv_py

        if curr_py != target_py:
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
    port = int(os.environ.get("HIFISAMPLER_PORT", "8572"))
    write_locator_beacon(port=port)

    if "--register-only" in sys.argv:
        print(f"Registered hifisampler locator beacon at: {Path.home() / '.hifisampler' / 'server.json'}")
        sys.exit(0)

    from filelock import FileLock, Timeout
    from backend import models
    from backend import server

    lock_file_path = get_lock_file_path(port=port)

    try:
        with FileLock(str(lock_file_path), timeout=0.5):
            logging.info(f"hifisampler {version} (root: {PROJECT_ROOT})")
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
