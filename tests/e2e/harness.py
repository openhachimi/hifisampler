"""E2E test harness for HiFiSampler (Python baseline and Rust port).

Provides an opaque-box testing interface for launching servers, querying
HTTP endpoints, invoking CLI client binaries, manipulating audio WAV fixtures,
and inspecting HIFIBIN1 feature caches.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Generator, Sequence, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger("e2e.harness")


def find_free_port() -> int:
    """Find and return an ephemeral open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return s.getsockname()[1]


class E2EHarness:
    """Opaque-box testing harness supporting Python baseline and Rust port."""

    def __init__(
        self,
        target: str | None = None,
        root_dir: Path | str | None = None,
    ) -> None:
        if root_dir is None:
            # Default to repository root
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir).resolve()

        # Target selection: explicit > sys.argv > env var > auto-detect
        if target is None:
            # Check sys.argv for --target
            for arg in sys.argv:
                if arg.startswith("--target="):
                    target = arg.split("=", 1)[1].strip().lower()
                    break
            if target is None:
                target = os.environ.get("HIFISAMPLER_TEST_TARGET", "").strip().lower()

        if not target:
            # Auto-detect target: if rust binary exists, default to rust; else python
            rust_release = (
                self.root_dir
                / "rust"
                / "target"
                / "release"
                / ("hifisampler.exe" if os.name == "nt" else "hifisampler")
            )
            if rust_release.exists():
                target = "rust"
            else:
                target = "python"

        if target not in ("python", "rust"):
            raise ValueError(f"Invalid target '{target}'. Must be 'python' or 'rust'.")

        self.target = target
        self.python_exe = self._resolve_python_exe()

    def _resolve_python_exe(self) -> Path:
        """Locate the Python interpreter in .venv or system environment."""
        if os.name == "nt":
            candidates = [
                self.root_dir / ".venv" / "Scripts" / "python.exe",
                self.root_dir / "env" / "Scripts" / "python.exe",
                Path(sys.executable),
            ]
        else:
            candidates = [
                self.root_dir / ".venv" / "bin" / "python",
                self.root_dir / "env" / "bin" / "python",
                Path(sys.executable),
            ]

        for cand in candidates:
            if cand.exists() and os.access(cand, os.X_OK):
                return cand.resolve()
        return Path(sys.executable).resolve()

    def resolve_cli_cmd(self) -> list[str]:
        """Resolve command prefix to execute the client CLI binary."""
        if self.target == "rust":
            ext = ".exe" if os.name == "nt" else ""
            candidates = [
                self.root_dir / "rust" / "target" / "release" / f"hifisampler{ext}",
                self.root_dir / "rust" / "target" / "debug" / f"hifisampler{ext}",
            ]
            for cand in candidates:
                if cand.exists():
                    return [str(cand)]
            raise FileNotFoundError(
                f"Rust client binary not found under {self.root_dir / 'rust' / 'target'}"
            )

        # target == "python"
        py_script = self.root_dir / "hifisampler.py"
        if py_script.exists():
            return [str(self.python_exe), str(py_script)]

        exe_binary = self.root_dir / "hifisampler.exe"
        if os.name == "nt" and exe_binary.exists():
            return [str(exe_binary)]

        sh_script = self.root_dir / "hifisampler.sh"
        if os.name != "nt" and sh_script.exists():
            return ["bash", str(sh_script)]

        # Fallback to python hifisampler.py path even if not yet created
        return [str(self.python_exe), str(py_script)]

    def resolve_server_cmd(self, port: int = 8572) -> list[str]:
        """Resolve command prefix to execute the HTTP server binary."""
        if self.target == "rust":
            ext = ".exe" if os.name == "nt" else ""
            candidates = [
                self.root_dir / "rust" / "target" / "release" / f"hifiserver{ext}",
                self.root_dir / "rust" / "target" / "debug" / f"hifiserver{ext}",
            ]
            for cand in candidates:
                if cand.exists():
                    return [str(cand), "--port", str(port)]
            raise FileNotFoundError(
                f"Rust server binary not found under {self.root_dir / 'rust' / 'target'}"
            )

        # target == "python"
        server_py = self.root_dir / "hifiserver.py"
        return [str(self.python_exe), str(server_py), "--port", str(port)]

    def run_cli(
        self,
        args: Sequence[str],
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> subprocess.CompletedProcess[str]:
        """Execute the CLI client binary with given positional arguments."""
        cmd_prefix = self.resolve_cli_cmd()
        full_cmd = cmd_prefix + list(args)

        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        # Ensure project root is in PYTHONPATH
        py_path = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = (
            f"{self.root_dir}{os.pathsep}{py_path}" if py_path else str(self.root_dir)
        )

        logger.debug("Executing CLI: %s", full_cmd)
        return subprocess.run(
            full_cmd,
            cwd=str(self.root_dir),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @contextlib.contextmanager
    def managed_server(
        self,
        port: int | None = None,
        idle_timeout: int = 15,
        managed: bool = True,
        startup_timeout: float = 45.0,
    ) -> Generator[dict[str, Any], None, None]:
        """Launch background server, poll readiness probe, and ensure clean shutdown."""
        if port is None:
            port = 8572

        cmd = self.resolve_server_cmd(port=port)
        if managed:
            cmd.append("--managed")
        if idle_timeout is not None:
            cmd.extend(["--idle-timeout", str(idle_timeout)])

        run_env = os.environ.copy()
        run_env["HIFISAMPLER_PORT"] = str(port)
        py_path = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = (
            f"{self.root_dir}{os.pathsep}{py_path}" if py_path else str(self.root_dir)
        )

        logger.info("Starting managed server on port %d: %s", port, cmd)
        log_path = Path(tempfile.gettempdir()) / f"hifi_server_{port}_{os.getpid()}_{int(time.time()*1000)}.log"
        log_file = open(log_path, "w+", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(self.root_dir),
            env=run_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        server_info = {
            "proc": proc,
            "port": port,
            "url": f"http://127.0.0.1:{port}",
            "target": self.target,
        }

        # Poll GET / for status 200 "Server Ready"
        start_time = time.time()
        ready = False
        last_error = ""

        while time.time() - start_time < startup_timeout:
            if proc.poll() is not None:
                log_file.flush()
                log_file.seek(0)
                out = log_file.read()
                log_file.close()
                raise RuntimeError(
                    f"Server process terminated prematurely with exit code {proc.returncode}.\n"
                    f"OUTPUT: {out}"
                )
            try:
                status, body = self.http_get("/", port=port)
                if status == 200:
                    ready = True
                    break
                last_error = f"Status {status}: {body}"
            except Exception as e:
                last_error = str(e)
            time.sleep(0.25)

        if not ready:
            self._terminate_process(proc)
            log_file.flush()
            log_file.seek(0)
            out = log_file.read()
            log_file.close()
            raise TimeoutError(
                f"Server failed to become ready on port {port} within {startup_timeout}s. "
                f"Last error: {last_error}\nOUTPUT: {out}"
            )

        try:
            yield server_info
        finally:
            logger.info("Shutting down managed server (PID %d)...", proc.pid)
            self._terminate_process(proc)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
            try:
                log_file.close()
            except Exception:
                pass
            if log_path.exists():
                try:
                    log_path.unlink()
                except OSError:
                    pass

    def _terminate_process(self, proc: subprocess.Popen) -> None:
        """Cross-platform graceful process termination with force kill fallback."""
        if proc.poll() is not None:
            return

        if os.name == "nt":
            # On Windows, try taskkill with /T to kill child process trees
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=3.0,
                )
            except Exception:
                try:
                    proc.terminate()
                except OSError:
                    pass
        else:
            try:
                proc.terminate()
            except OSError:
                pass

        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass

    def http_get(
        self,
        endpoint: str,
        port: int = 8572,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> tuple[int, Union[str, dict[str, Any]]]:
        """Perform HTTP GET request returning (status_code, body)."""
        ep = "/" + endpoint.lstrip("/")
        url = f"http://127.0.0.1:{port}{ep}"
        req_headers = {"User-Agent": "HifiSampler-E2E-Harness"}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
                try:
                    return status, json.loads(raw)
                except json.JSONDecodeError:
                    return status, raw
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8") if e.fp else ""
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, raw
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e.reason}") from e

    def http_post(
        self,
        endpoint: str,
        payload: Union[dict[str, Any], list[Any], str],
        port: int = 8572,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, str]:
        """Perform HTTP POST request returning (status_code, response_text)."""
        ep = "/" + endpoint.lstrip("/")
        url = f"http://127.0.0.1:{port}{ep}"
        req_headers = {"User-Agent": "HifiSampler-E2E-Harness"}

        if isinstance(payload, (dict, list)):
            data = json.dumps(payload).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        else:
            data = str(payload).encode("utf-8")
            req_headers["Content-Type"] = "text/plain"

        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                text = resp.read().decode("utf-8")
                return status, text
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8") if e.fp else ""
            return e.code, text
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e.reason}") from e

    def parse_hifibin_header(self, path: Union[str, Path]) -> dict[str, Any]:
        """Parse 28-byte packed header `<8sIIId` from a HIFIBIN1 cache file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature cache file not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size < 28:
            raise ValueError(
                f"File {file_path} is too small for HIFIBIN1 header ({file_size} < 28 bytes)"
            )

        with open(file_path, "rb") as f:
            header_bytes = f.read(28)

        magic_raw, version, rows, cols, scale = struct.unpack("<8sIIId", header_bytes)
        magic_clean = magic_raw.rstrip(b"\x00").decode("latin1")
        expected_size = 28 + (rows * cols * 4)

        return {
            "magic": magic_clean,
            "version": version,
            "rows": rows,
            "cols": cols,
            "scale": scale,
            "file_size": file_size,
            "expected_size": expected_size,
            "size_valid": (file_size == expected_size),
        }

    def read_wav(self, path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """Load audio samples and sample rate using soundfile."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"WAV file not found: {file_path}")
        audio, sr = sf.read(str(file_path), dtype="float32")
        return audio, sr

    def generate_synthetic_wav(
        self,
        path: Union[str, Path],
        duration_sec: float = 0.5,
        freq: float = 440.0,
        sample_rate: int = 44100,
    ) -> str:
        """Create valid 16-bit PCM mono test WAV file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n_samples = max(1, int(round(sample_rate * duration_sec)))
        t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)
        samples = (0.5 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

        sf.write(str(out_path), samples, sample_rate, subtype="PCM_16", format="WAV")
        return str(out_path)

    @contextlib.contextmanager
    def temp_dir(self, prefix: str = "hifi_e2e_") -> Generator[Path, None, None]:
        """Create an isolated temporary directory fixture, automatically cleaned up."""
        tmp = Path(tempfile.mkdtemp(prefix=prefix))
        try:
            yield tmp
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
