import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

from hifiserver import PROJECT_ROOT, get_venv_python


def get_default_conda_base_path():
    """Attempts to find the default conda base path."""
    if sys.platform == "win32":
        possible_paths = [
            os.path.expanduser(r"~\miniconda3"),
            os.path.expanduser(r"~\anaconda3"),
            r"C:\miniconda3",
            r"C:\anaconda3",
            r"C:\ProgramData\Miniconda3",
            r"C:\ProgramData\Anaconda3",
        ]
    elif sys.platform in ("linux", "linux2", "darwin"):
        possible_paths = [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            "/opt/miniconda3",
            "/opt/anaconda3",
        ]
    else:
        return None

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def start_in_venv(venv_path: Path, python_script_path: Path) -> bool:
    """Start hifiserver.py directly in the current console using .venv."""
    python_exe = get_venv_python(venv_path.parent)
    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found in virtual environment: {python_exe}")
    if not python_script_path.exists():
        raise FileNotFoundError(f"Target Python script not found: {python_script_path}")

    command = [str(python_exe), str(python_script_path)]
    print(f"Starting server with venv: {' '.join(command)}")
    subprocess.run(command, check=True)
    return True


def _find_conda_executable(conda_base_path: str | None) -> str:
    if conda_base_path:
        candidates = (
            [os.path.join(conda_base_path, "Scripts", "conda.exe"), os.path.join(conda_base_path, "condabin", "conda.bat")]
            if sys.platform == "win32"
            else [os.path.join(conda_base_path, "bin", "conda")]
        )
        for cand in candidates:
            if os.path.exists(cand):
                return cand

    conda_in_path = shutil.which("conda")
    if conda_in_path:
        return conda_in_path

    default_base = get_default_conda_base_path()
    if default_base:
        candidates = (
            [os.path.join(default_base, "Scripts", "conda.exe"), os.path.join(default_base, "condabin", "conda.bat")]
            if sys.platform == "win32"
            else [os.path.join(default_base, "bin", "conda")]
        )
        for cand in candidates:
            if os.path.exists(cand):
                return cand

    raise FileNotFoundError(
        "Could not find Conda installation. Please use uv (.venv) or set conda_base_path in config.yaml."
    )


def start_in_conda_env(config: dict) -> bool:
    """Start hifiserver.py directly in the current console using Conda (without nested terminal windows)."""
    conda_env_name = config["conda_env_name"]
    python_script_path = Path(config["python_script_path"]).resolve()
    conda_base_path = config.get("conda_base_path")

    conda_bin = _find_conda_executable(conda_base_path)
    command = [conda_bin, "run", "--no-capture-output", "-n", conda_env_name, "python", str(python_script_path)]
    print(f"Starting server with Conda: {' '.join(command)}")
    subprocess.run(command, check=True)
    return True


def start_environment(config: dict) -> bool:
    os.chdir(PROJECT_ROOT)
    python_script_path = (PROJECT_ROOT / config.get("python_script_path", "./hifiserver.py")).resolve()
    env_path = PROJECT_ROOT / "env"
    venv_path = PROJECT_ROOT / ".venv"

    if (env_path / "python.exe").exists() or (venv_path.exists() and venv_path.is_dir()):
        return start_in_venv(venv_path, python_script_path)
    else:
        print(".venv folder not found. Falling back to Conda environment...")
        return start_in_conda_env(config)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    config_file_path = PROJECT_ROOT / "config.yaml"
    default_config_file_path = PROJECT_ROOT / "config.default.yaml"
    if not config_file_path.exists() and default_config_file_path.exists():
        try:
            shutil.copy(default_config_file_path, config_file_path)
        except OSError as e:
            print(f"Warning: Could not copy default config file: {e}")

    active_config_path = config_file_path if config_file_path.exists() else default_config_file_path
    env_config = {"conda_env_name": "hifisampler", "python_script_path": "./hifiserver.py"}

    if active_config_path.exists():
        try:
            with open(active_config_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f) or {}
            if isinstance(loaded_config, dict) and isinstance(loaded_config.get("env"), dict):
                env_config.update(loaded_config["env"])
        except Exception as e:
            print(f"Warning: Failed to read '{active_config_path}', using defaults: {e}")

    try:
        start_environment(env_config)
    except Exception as e:
        print(f"Failed to start hifiserver: {e}")
        sys.exit(1)
