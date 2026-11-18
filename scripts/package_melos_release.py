#!/usr/bin/env python3
"""Build multi-platform Melos release archives, run 3-stage isolated smoke tests, and emit release notes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform as py_platform
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_ENTRIES = {
    "model.ckpt",
    "model.pt",
    "hifisampler",
    "hifisampler.exe",
    "hifisampler.yaml",
}

PACKAGED_START_BAT = """\
@echo off
setlocal
cd /d "%~dp0"

if not exist "config.yaml" if exist "config.default.yaml" (
    copy /Y "config.default.yaml" "config.yaml" >nul
)

if not exist "%~dp0hifiserver.exe" (
    echo [Melos] Error: hifiserver.exe was not found in "%~dp0".
    if "%~1"=="" pause
    exit /b 1
)

"%~dp0hifiserver.exe" %*
set "_rc=%errorlevel%"
if %_rc% neq 0 (
    if "%~1"=="" pause
    exit /b %_rc%
)
"""

PACKAGED_START_SH = """\
#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "$SCRIPT_DIR/config.yaml" ] && [ -f "$SCRIPT_DIR/config.default.yaml" ]; then
    cp "$SCRIPT_DIR/config.default.yaml" "$SCRIPT_DIR/config.yaml"
fi

if [ ! -f "$SCRIPT_DIR/hifiserver" ]; then
    echo "[Melos] Error: hifiserver binary was not found in $SCRIPT_DIR" >&2
    exit 1
fi

chmod +x "$SCRIPT_DIR/hifiserver" "$SCRIPT_DIR/melos" "$SCRIPT_DIR/melos-wavtool" 2>/dev/null || true
exec "$SCRIPT_DIR/hifiserver" "$@"
"""


def detect_default_platform() -> str:
    sys_p = sys.platform.lower()
    mach = py_platform.machine().lower()
    if sys_p.startswith("win"):
        return "win64-dml"
    if sys_p == "darwin":
        return "macos-arm64-coreml" if ("arm" in mach or "aarch64" in mach) else "macos-x64-coreml"
    if "aarch64" in mach or "arm64" in mach:
        return "linux-arm64-aio"
    return "linux-x64-aio"


def derive_platform_meta(pkg_platform: str) -> dict[str, object]:
    p = pkg_platform.strip().lower()
    if p.startswith("win"):
        return {
            "platform": pkg_platform,
            "os_kind": "windows",
            "os_arch": p.split("-", 1)[0],
            "ext": "zip",
            "bins": ("melos.exe", "melos-wavtool.exe", "hifiserver.exe"),
            "launcher": "start.bat",
            "config_label": "Windows x64 DirectML All-in-One",
        }
    if p.startswith("macos-"):
        parts = p.split("-")
        arch = parts[1] if len(parts) >= 2 else "arm64"
        return {
            "platform": pkg_platform,
            "os_kind": "macos",
            "os_arch": f"macos-{arch}",
            "ext": "tar.gz",
            "bins": ("melos", "melos-wavtool", "hifiserver"),
            "launcher": "start.sh",
            "config_label": f"macOS ({arch}) CoreML/CPU All-in-One",
        }
    if p.startswith("linux-"):
        parts = p.split("-")
        arch = parts[1] if len(parts) >= 2 else "x64"
        return {
            "platform": pkg_platform,
            "os_kind": "linux",
            "os_arch": f"linux-{arch}",
            "ext": "tar.gz",
            "bins": ("melos", "melos-wavtool", "hifiserver"),
            "launcher": "start.sh",
            "config_label": f"Linux ({arch}) ONNX All-in-One",
        }
    raise ValueError(f"Unsupported --platform value: {pkg_platform!r}")


def make_aio_config_yaml(version_num: str, label: str) -> str:
    return f"""\
# Melos Default Configuration ({label})
# Edit config.yaml (or copy config.default.yaml -> config.yaml) to customize settings.
version: {version_num}

model:
  vocoder_path: "./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx"
  model_type: "onnx"
  hnsep_model_path: "./hnsep/vr/model.onnx"

audio:
  sample_rate: 44100
  win_size: 2048
  hop_size: 512
  origin_hop_size: 128
  n_mels: 128
  n_fft: 2048
  mel_fmin: 40
  mel_fmax: 16000

processing:
  wave_norm: true
  trim_silence: true
  silence_threshold: -52.0
  loop_mode: true
  auto_vocal_fry: true
  peak_limit: 1.0
  fill: 6

performance:
  max_workers: 2
  device: "auto"
  fp16: false
  idle_timeout: 0
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ArchiveWriter:
    """Unified .zip / .tar.gz writer supporting executable permissions and POSIX symlinks."""

    def __init__(self, path: Path, ext: str) -> None:
        self.path = path
        self.ext = ext
        if ext == "zip":
            self._zf = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6)
            self._tf = None
        elif ext == "tar.gz":
            self._zf = None
            self._tf = tarfile.open(path, "w:gz", compresslevel=6)
        else:
            raise ValueError(f"Unsupported archive ext: {ext}")

    def __enter__(self) -> "ArchiveWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._zf is not None:
            self._zf.close()
        if self._tf is not None:
            self._tf.close()

    def add_file(self, src: Path, arcname: str, executable: bool = False) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"Missing required release asset: {src}")
        if self._zf is not None:
            self._zf.write(src, arcname=arcname)
            return
        assert self._tf is not None
        tarinfo = self._tf.gettarinfo(str(src), arcname=arcname)
        tarinfo.uid = 0
        tarinfo.gid = 0
        tarinfo.uname = "root"
        tarinfo.gname = "root"
        tarinfo.mode = 0o755 if executable else 0o644
        with src.open("rb") as fh:
            self._tf.addfile(tarinfo, fh)

    def add_text(
        self,
        text: str,
        arcname: str,
        crlf: bool = False,
        executable: bool = False,
    ) -> None:
        normalized = text.replace("\r\n", "\n")
        if crlf:
            normalized = normalized.replace("\n", "\r\n")
        data = normalized.encode("utf-8")
        if self._zf is not None:
            info = zipfile.ZipInfo(arcname, date_time=(2026, 11, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if executable else 0o644) << 16
            self._zf.writestr(info, data)
            return
        assert self._tf is not None
        ti = tarfile.TarInfo(name=arcname)
        ti.size = len(data)
        ti.mtime = 1794988800
        ti.mode = 0o755 if executable else 0o644
        ti.uid = 0
        ti.gid = 0
        ti.uname = "root"
        ti.gname = "root"
        self._tf.addfile(ti, io.BytesIO(data))

    def add_symlink(self, arcname: str, target: str) -> None:
        if self._tf is None:
            return
        ti = tarfile.TarInfo(name=arcname)
        ti.type = tarfile.SYMTYPE
        ti.linkname = target
        ti.mtime = 1794988800
        ti.mode = 0o777
        ti.uid = 0
        ti.gid = 0
        ti.uname = "root"
        ti.gname = "root"
        self._tf.addfile(ti)


def find_ort_notice(ort_root: Path, filename: str) -> Path:
    direct = ort_root / filename
    if direct.is_file():
        return direct
    for pattern in (
        f"onnxruntime*.dist-info/**/{filename}",
        f"onnxruntime*.dist-info/{filename}",
    ):
        matches = sorted(ort_root.parent.glob(pattern))
        for m in matches:
            if m.is_file():
                return m
    raise FileNotFoundError(f"Could not locate ONNX Runtime notice '{filename}' near {ort_root}")


def collect_posix_ort_libs(ort_capi: Path, os_kind: str) -> tuple[list[tuple[Path, str]], list[tuple[str, str]]]:
    """Return ([(real_path, arc_basename), ...], [(symlink_name, target_basename), ...])."""
    if not ort_capi.is_dir():
        raise FileNotFoundError(f"ONNX Runtime capi directory not found: {ort_capi}")

    real_libs: dict[Path, str] = {}
    for entry in sorted(ort_capi.iterdir()):
        name = entry.name
        if "pybind11" in name.lower():
            continue
        if os_kind == "macos":
            match = name.startswith("libonnxruntime") and name.endswith(".dylib")
        else:
            match = name.startswith("libonnxruntime") and ".so" in name
        if not match or not entry.is_file():
            continue
        resolved = entry.resolve()
        if resolved not in real_libs:
            real_libs[resolved] = resolved.name

    if not real_libs:
        raise FileNotFoundError(f"No libonnxruntime shared library found in {ort_capi}")

    archived_names = set(real_libs.values())
    symlinks: list[tuple[str, str]] = []

    if os_kind == "macos":
        core_libs = sorted(
            n for n in archived_names if n.startswith("libonnxruntime.") and n.endswith(".dylib")
        )
        main_lib = next((n for n in core_libs if n != "libonnxruntime.dylib"), core_libs[0] if core_libs else None)
        if main_lib and "libonnxruntime.dylib" not in archived_names:
            symlinks.append(("libonnxruntime.dylib", main_lib))
    else:
        core_libs = sorted(n for n in archived_names if n.startswith("libonnxruntime.so"))
        main_lib = next((n for n in core_libs if n != "libonnxruntime.so"), core_libs[0] if core_libs else None)
        if main_lib:
            if "libonnxruntime.so" not in archived_names:
                symlinks.append(("libonnxruntime.so", main_lib))
            if "libonnxruntime.so.1" not in archived_names and main_lib != "libonnxruntime.so.1":
                symlinks.append(("libonnxruntime.so.1", main_lib))

    files = [(p, name) for p, name in real_libs.items()]
    return files, symlinks


def populate_common_binaries(
    writer: ArchiveWriter,
    repo_root: Path,
    version_num: str,
    meta: dict[str, object],
    stage_bin_dir: Path | None,
) -> None:
    is_win = meta["os_kind"] == "windows"
    bin_source = stage_bin_dir if stage_bin_dir is not None else (repo_root / "rust" / "target" / "release")
    prefix = "melos"

    for bin_name in meta["bins"]:  # type: ignore[attr-defined]
        writer.add_file(bin_source / bin_name, f"{prefix}/{bin_name}", executable=True)

    writer.add_file(repo_root / "rust" / "melos.yaml", f"{prefix}/melos.yaml")
    if is_win:
        writer.add_text(PACKAGED_START_BAT, f"{prefix}/start.bat", crlf=True, executable=True)
    else:
        writer.add_text(PACKAGED_START_SH, f"{prefix}/start.sh", crlf=False, executable=True)

    writer.add_file(repo_root / "LICENSE", f"{prefix}/LICENSE")
    writer.add_file(repo_root / "README.md", f"{prefix}/README.md")
    writer.add_file(repo_root / "README_zh_cn.md", f"{prefix}/README_zh_cn.md")
    writer.add_file(repo_root / "hnsep" / "LICENSE", f"{prefix}/hnsep/LICENSE")
    writer.add_text(
        make_aio_config_yaml(version_num, str(meta["config_label"])),
        f"{prefix}/config.default.yaml",
        crlf=is_win,
    )


def prepare_signed_bins_if_macos(repo_root: Path, os_kind: str) -> tuple[tempfile.TemporaryDirectory[str] | None, Path | None]:
    if os_kind != "macos" or sys.platform != "darwin" or shutil.which("codesign") is None:
        return None, None
    tmp = tempfile.TemporaryDirectory(prefix="melos_codesign_")
    tmp_path = Path(tmp.name)
    src_dir = repo_root / "rust" / "target" / "release"
    for name in ("melos", "melos-wavtool", "hifiserver"):
        src = src_dir / name
        dst = tmp_path / name
        shutil.copy2(src, dst)
        dst.chmod(0o755)
        subprocess.run(["codesign", "--force", "--sign", "-", str(dst)], check=True)
    return tmp, tmp_path


def list_archive_entries(archive_path: Path, ext: str) -> list[str]:
    if ext == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            return zf.namelist()
    with tarfile.open(archive_path, "r:gz") as tf:
        return [m.name for m in tf.getmembers() if not m.isdir()]


def assert_clean_archive(archive_path: Path, meta: dict[str, object], expect_models: bool) -> list[str]:
    names = list_archive_entries(archive_path, str(meta["ext"]))
    basenames = {Path(n).name.lower() for n in names if not n.endswith("/")}
    bad = basenames & FORBIDDEN_ENTRIES
    if bad:
        raise AssertionError(f"{archive_path.name} contains forbidden files: {sorted(bad)}")
    if any("pybind11" in b for b in basenames):
        raise AssertionError(f"{archive_path.name} contains unexpected pybind11 module")

    os_kind = str(meta["os_kind"])
    if not expect_models:
        if any(
            n.endswith(".onnx") or n.endswith(".dll") or n.endswith(".dylib") or ".so" in Path(n).name
            for n in names
        ):
            raise AssertionError(f"{archive_path.name} unexpectedly contains ONNX weights or shared libraries")
        return names

    bins = tuple(meta["bins"])  # type: ignore[arg-type]
    required_subpaths = {
        f"melos/{bins[0]}",
        f"melos/{bins[1]}",
        f"melos/{bins[2]}",
        "melos/melos.yaml",
        f"melos/{meta['launcher']}",
        "melos/config.default.yaml",
        "melos/config.yaml",
        "melos/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx",
        "melos/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/config.json",
        "melos/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/NOTICE.txt",
        "melos/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/NOTICE.zh-CN.txt",
        "melos/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/STATEMENTS.txt",
        "melos/hnsep/LICENSE",
        "melos/hnsep/vr/model.onnx",
        "melos/hnsep/vr/config.yaml",
        "melos/THIRD_PARTY_NOTICES/onnxruntime-LICENSE.txt",
        "melos/THIRD_PARTY_NOTICES/onnxruntime-ThirdPartyNotices.txt",
    }
    if os_kind == "windows":
        required_subpaths |= {
            "melos/onnxruntime.dll",
            "melos/onnxruntime_providers_shared.dll",
            "melos/DirectML.dll",
        }
    elif os_kind == "macos":
        if not any(n.startswith("melos/libonnxruntime") and n.endswith(".dylib") for n in names):
            raise AssertionError(f"{archive_path.name} missing macOS libonnxruntime dylib")
    else:
        if not any(n.startswith("melos/libonnxruntime.so") for n in names):
            raise AssertionError(f"{archive_path.name} missing Linux libonnxruntime.so")

    missing = required_subpaths - set(names)
    if missing:
        raise AssertionError(f"{archive_path.name} missing required AIO files: {sorted(missing)}")
    return names


def build_archives(
    repo_root: Path,
    ort_root: Path,
    models_root: Path,
    out_dir: Path,
    version_tag: str,
    pkg_platform: str,
) -> tuple[Path, Path, dict[str, object]]:
    meta = derive_platform_meta(pkg_platform)
    os_kind = str(meta["os_kind"])
    ext = str(meta["ext"])
    os_arch = str(meta["os_arch"])
    version_num = version_tag.lstrip("vV")

    out_dir.mkdir(parents=True, exist_ok=True)
    aio_path = out_dir / f"melos-{version_tag}-{pkg_platform}.{ext}"
    bins_path = out_dir / f"melos-{version_tag}-{os_arch}-bins-only.{ext}"

    codesign_tmp, stage_bin_dir = prepare_signed_bins_if_macos(repo_root, os_kind)
    try:
        print(f"[build] Writing {bins_path} ...")
        with ArchiveWriter(bins_path, ext) as writer:
            populate_common_binaries(writer, repo_root, version_num, meta, stage_bin_dir)
        bins_names = assert_clean_archive(bins_path, meta, expect_models=False)
        print(f"[build] {bins_path.name}: {len(bins_names)} entries, {bins_path.stat().st_size:,} bytes")

        print(f"[build] Writing {aio_path} ...")
        ort_capi = ort_root / "capi"
        vocoder_dir = models_root / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
        hnsep_vr_dir = models_root / "hnsep" / "vr"
        prefix = "melos"
        with ArchiveWriter(aio_path, ext) as writer:
            populate_common_binaries(writer, repo_root, version_num, meta, stage_bin_dir)
            writer.add_text(
                make_aio_config_yaml(version_num, str(meta["config_label"])),
                f"{prefix}/config.yaml",
                crlf=(os_kind == "windows"),
            )

            if os_kind == "windows":
                for dll_name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll", "DirectML.dll"):
                    writer.add_file(ort_capi / dll_name, f"{prefix}/{dll_name}")
            else:
                ort_files, ort_symlinks = collect_posix_ort_libs(ort_capi, os_kind)
                for src_lib, arc_basename in ort_files:
                    writer.add_file(src_lib, f"{prefix}/{arc_basename}", executable=True)
                for sym_name, sym_target in ort_symlinks:
                    writer.add_symlink(f"{prefix}/{sym_name}", sym_target)

            writer.add_file(
                find_ort_notice(ort_root, "LICENSE"),
                f"{prefix}/THIRD_PARTY_NOTICES/onnxruntime-LICENSE.txt",
            )
            writer.add_file(
                find_ort_notice(ort_root, "ThirdPartyNotices.txt"),
                f"{prefix}/THIRD_PARTY_NOTICES/onnxruntime-ThirdPartyNotices.txt",
            )

            voc_arc = f"{prefix}/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
            for name in ("model.onnx", "config.json", "NOTICE.txt", "NOTICE.zh-CN.txt", "STATEMENTS.txt"):
                writer.add_file(vocoder_dir / name, f"{voc_arc}/{name}")

            writer.add_file(hnsep_vr_dir / "model.onnx", f"{prefix}/hnsep/vr/model.onnx")
            writer.add_file(hnsep_vr_dir / "config.yaml", f"{prefix}/hnsep/vr/config.yaml")

        aio_names = assert_clean_archive(aio_path, meta, expect_models=True)
        print(f"[build] {aio_path.name}: {len(aio_names)} entries, {aio_path.stat().st_size:,} bytes")
        return aio_path, bins_path, meta
    finally:
        if codesign_tmp is not None:
            codesign_tmp.cleanup()


def write_synthetic_voice_wav(path: Path, sr: int = 44100, duration_s: float = 1.6) -> None:
    """Create a 16-bit mono 44.1kHz voiced harmonic sample outside the repo for smoke testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sr * duration_s)
    f0 = 261.625565  # C4
    samples = bytearray()
    for i in range(n_samples):
        t = i / sr
        env = 1.0
        if t < 0.04:
            env = t / 0.04
        elif t > duration_s - 0.06:
            env = max(0.0, (duration_s - t) / 0.06)
        phase = 2.0 * math.pi * f0 * t
        val = (
            0.42 * math.sin(phase)
            + 0.24 * math.sin(2.0 * phase + 0.2)
            + 0.14 * math.sin(3.0 * phase + 0.5)
            + 0.08 * math.sin(4.0 * phase)
            + 0.04 * math.sin(5.0 * phase)
        )
        breath = 0.02 * math.sin(2.0 * math.pi * 1733.0 * t + 0.7 * math.sin(2.0 * math.pi * 211.0 * t))
        pcm = max(-0.95, min(0.95, env * (val + breath)))
        samples.extend(struct.pack("<h", int(pcm * 32767.0)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(samples))


def read_wav_stats(path: Path) -> tuple[int, int, float, float, float]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sampwidth == 2:
        vals = [s / 32768.0 for (s,) in struct.iter_unpack("<h", raw)]
    elif sampwidth == 4:
        vals = [s for (s,) in struct.iter_unpack("<f", raw)]
    else:
        raise AssertionError(f"Unsupported sample width {sampwidth} in {path}")
    if channels > 1:
        vals = vals[::channels]
    if not vals or any(not math.isfinite(v) for v in vals):
        raise AssertionError(f"WAV {path} empty or contains non-finite samples")
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    peak = max(abs(v) for v in vals)
    dur_s = len(vals) / float(sr)
    return sr, len(vals), dur_s, rms, peak


def http_get_json(url: str, timeout: float = 2.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def choose_smoke_port(preferred: int = 8574) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def extract_archive(archive_path: Path, dest_dir: Path, ext: str) -> None:
    if ext == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
        return
    with tarfile.open(archive_path, "r:gz") as tf:
        if sys.version_info >= (3, 12):
            tf.extractall(dest_dir, filter="data")
        else:
            tf.extractall(dest_dir)


def kill_smoke_server(server_exe: Path) -> None:
    if os.name == "nt":
        norm = str(server_exe.resolve()).replace("'", "''")
        ps_cmd = (
            f"Get-CimInstance Win32_Process | Where-Object {{ $_.ExecutablePath -eq '{norm}' }} | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], capture_output=True, timeout=10)
    else:
        subprocess.run(["pkill", "-TERM", "-f", str(server_exe)], capture_output=True, timeout=5)
        time.sleep(0.3)
        subprocess.run(["pkill", "-KILL", "-f", str(server_exe)], capture_output=True, timeout=5)


def run_isolated_smoke_test(aio_path: Path, version_tag: str, meta: dict[str, object]) -> dict:
    version_num = version_tag.lstrip("vV")
    os_kind = str(meta["os_kind"])
    is_win = os_kind == "windows"
    bins = tuple(meta["bins"])  # type: ignore[arg-type]
    melos_bin, wavtool_bin, server_bin = bins

    smoke_root = Path(tempfile.mkdtemp(prefix="melos_aio_smoke_"))
    print(f"[smoke] Isolated temp root ({os_kind}): {smoke_root}")
    melos_dir = smoke_root / "extracted" / "melos"

    try:
        extract_archive(aio_path, smoke_root / "extracted", str(meta["ext"]))

        home_dir = smoke_root / "home"
        tmp_dir = smoke_root / "tmp"
        openutau_resamplers = smoke_root / "OpenUtau" / "Resamplers"
        openutau_wavtools = smoke_root / "OpenUtau" / "Wavtools"
        work_dir = smoke_root / "work"
        for d in (
            home_dir / "AppData" / "Roaming",
            home_dir / "AppData" / "Local",
            tmp_dir,
            openutau_resamplers,
            openutau_wavtools,
            work_dir / "oto",
            work_dir / "cache",
        ):
            d.mkdir(parents=True, exist_ok=True)

        if is_win:
            sanitized_env = {
                "SystemRoot": os.environ.get("SystemRoot", r"C:\Windows"),
                "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
                "ComSpec": os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe"),
                "PATH": r"C:\Windows\System32;C:\Windows",
                "USERNAME": os.environ.get("USERNAME", "melos_smoke"),
                "USERPROFILE": str(home_dir),
                "HOME": str(home_dir),
                "APPDATA": str(home_dir / "AppData" / "Roaming"),
                "LOCALAPPDATA": str(home_dir / "AppData" / "Local"),
                "TEMP": str(tmp_dir),
                "TMP": str(tmp_dir),
                "NO_PROXY": "127.0.0.1,localhost",
            }
        else:
            sanitized_env = {
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                "USER": os.environ.get("USER", "melos_smoke"),
                "HOME": str(home_dir),
                "XDG_CACHE_HOME": str(tmp_dir),
                "TMPDIR": str(tmp_dir),
                "NO_PROXY": "127.0.0.1,localhost",
            }

        # Stage 1: Verify --version on all 3 extracted binaries
        for bin_name, expected_prefix in (
            (server_bin, f"hifiserver {version_num}"),
            (melos_bin, f"melos {version_num}"),
            (wavtool_bin, f"melos-wavtool {version_num}"),
        ):
            res = subprocess.run(
                [str(melos_dir / bin_name), "--version"],
                cwd=str(melos_dir),
                env=sanitized_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )
            if res.returncode != 0 or expected_prefix not in res.stdout:
                raise AssertionError(
                    f"{bin_name} --version failed: rc={res.returncode}, stdout={res.stdout!r}, stderr={res.stderr!r}"
                )
        print(f"[smoke] Stage 1 passed: all 3 binaries report v{version_num}.")

        # Stage 2: Register locator beacon via extracted launcher --register-only
        smoke_port = choose_smoke_port(8574)
        extra_port_args = [] if smoke_port == 8574 else ["--port", str(smoke_port)]
        if is_win:
            reg_cmd = ["cmd.exe", "/d", "/c", " ".join(["start.bat", "--register-only", *extra_port_args])]
        else:
            start_sh = melos_dir / "start.sh"
            assert (start_sh.stat().st_mode & stat.S_IXUSR) != 0, "start.sh must retain +x permission"
            reg_cmd = ["/bin/bash", "./start.sh", "--register-only", *extra_port_args]

        reg_res = subprocess.run(
            reg_cmd,
            cwd=str(melos_dir),
            env=sanitized_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if reg_res.returncode != 0:
            raise RuntimeError(
                f"register-only failed with rc={reg_res.returncode}\nSTDOUT:\n{reg_res.stdout}\nSTDERR:\n{reg_res.stderr}"
            )
        melos_beacon = home_dir / ".melos" / "server.json"
        legacy_beacon = home_dir / ".hifisampler" / "server.json"
        if not melos_beacon.is_file():
            raise AssertionError(f"Expected beacon missing: {melos_beacon}")
        if legacy_beacon.exists():
            raise AssertionError(f"Melos registration unexpectedly wrote legacy beacon: {legacy_beacon}")
        beacon_data = json.loads(melos_beacon.read_text(encoding="utf-8"))
        if int(beacon_data.get("port", 0)) != smoke_port:
            raise AssertionError(f"Beacon port mismatch: {beacon_data}")
        if Path(beacon_data["server_root"]).resolve() != melos_dir.resolve():
            raise AssertionError(f"Beacon server_root mismatch: {beacon_data}")
        if Path(beacon_data["server_exe"]).resolve() != (melos_dir / server_bin).resolve():
            raise AssertionError(f"Beacon server_exe mismatch: {beacon_data}")
        print(f"[smoke] Stage 2 passed: ~/.melos/server.json registered on port {smoke_port}; ~/.hifisampler untouched.")

        # Stage 3: Simulate real OpenUtau layout & cold auto-launch via ~/.melos/server.json
        resamp_copied = openutau_resamplers / melos_bin
        wavtool_copied = openutau_wavtools / wavtool_bin
        shutil.copy2(melos_dir / melos_bin, resamp_copied)
        shutil.copy2(melos_dir / "melos.yaml", openutau_resamplers / "melos.yaml")
        shutil.copy2(melos_dir / wavtool_bin, wavtool_copied)
        if not is_win:
            resamp_copied.chmod(0o755)
            wavtool_copied.chmod(0o755)

        in_wav = work_dir / "oto" / "a.wav"
        write_synthetic_voice_wav(in_wav)

        out_wav = work_dir / "phrase_out.wav"
        note1_wav = work_dir / "cache" / "note1.wav"
        note2_wav = work_dir / "cache" / "note2.wav"

        sep = "\\\\" if is_win else "/"
        temp_bat = work_dir / "temp.bat"
        temp_bat.write_text(
            "@chcp 65001 >nul\r\n"
            "@set loadmodule=\r\n"
            "@set tempo=120\r\n"
            "@set samples=44100\r\n"
            f"@set oto={work_dir / 'oto'}\r\n"
            f"@set tool={wavtool_copied}\r\n"
            f"@set resamp={resamp_copied}\r\n"
            f"@set output={out_wav}\r\n"
            "@set helper=temp_helper.bat\r\n"
            f"@set cachedir={work_dir / 'cache'}\r\n"
            '@del "%output%" 2>nul\r\n'
            '@set flag="Ht15"\r\n'
            "@set vel=100\r\n"
            f"@set temp={note1_wav}\r\n"
            "@set params=100 0 !120 AA\r\n"
            f'@call %helper% "%oto%{sep}a.wav" C4 480@120+0 0 40 600 100 -300 0 10 0 100 100 0 0\r\n'
            "@if errorlevel 1 exit /b 1\r\n"
            '@set flag="Ht15"\r\n'
            "@set vel=100\r\n"
            f"@set temp={note2_wav}\r\n"
            "@set params=100 0 !120 AA\r\n"
            f'@call %helper% "%oto%{sep}a.wav" D4 480@120+0 0 40 600 100 -300 0 10 0 100 100 0 0\r\n'
            "@if errorlevel 1 exit /b 1\r\n",
            encoding="utf-8",
        )

        if is_win:
            helper_bat = work_dir / "temp_helper.bat"
            helper_bat.write_text(
                "@chcp 65001 >nul\r\n"
                '@if not exist "%temp%" "%resamp%" %1 "%temp%" %2 %vel% %flag% %5 %6 %7 %8 %params%\r\n'
                "@if errorlevel 1 exit /b 1\r\n"
                '@"%tool%" "%output%" "%temp%" %4 %3 0 10 0 100 100 0 0\r\n'
                "@if errorlevel 1 exit /b 1\r\n",
                encoding="utf-8",
            )
            bat_res = subprocess.run(
                ["cmd.exe", "/d", "/c", str(temp_bat)],
                cwd=str(work_dir),
                env=sanitized_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=90,
            )
            if bat_res.returncode != 0:
                log_path = melos_dir / "logs" / "hifiserver.log"
                log_dump = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else "<missing>"
                raise RuntimeError(
                    f"Cold auto-launch temp.bat exited with {bat_res.returncode}\nSTDOUT:\n{bat_res.stdout}\nSTDERR:\n{bat_res.stderr}\nSERVER LOG:\n{log_dump}"
                )
        else:
            steps = [
                [str(resamp_copied), str(in_wav), str(note1_wav), "C4", "100", "Ht15", "40", "600", "100", "-300", "100", "0", "!120", "AA"],
                [str(wavtool_copied), str(out_wav), str(note1_wav), "0", "480@120+0", "0", "10", "0", "100", "100", "0", "0"],
                [str(resamp_copied), str(in_wav), str(note2_wav), "D4", "100", "Ht15", "40", "600", "100", "-300", "100", "0", "!120", "AA"],
                [str(wavtool_copied), str(out_wav), str(note2_wav), "0", "480@120+0", "0", "10", "0", "100", "100", "0", "0"],
            ]
            for idx, cmd in enumerate(steps, 1):
                step_res = subprocess.run(
                    cmd,
                    cwd=str(work_dir),
                    env=sanitized_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                )
                if step_res.returncode != 0:
                    log_path = melos_dir / "logs" / "hifiserver.log"
                    log_dump = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else "<missing>"
                    raise RuntimeError(
                        f"POSIX step {idx} ({Path(cmd[0]).name}) failed rc={step_res.returncode}\nSTDOUT:\n{step_res.stdout}\nSTDERR:\n{step_res.stderr}\nSERVER LOG:\n{log_dump}"
                    )

        if not out_wav.is_file():
            raise AssertionError(f"Phrase output WAV not created: {out_wav}")
        sr, n_frames, dur_s, rms, peak = read_wav_stats(out_wav)
        if sr != 44100 or dur_s < 0.7 or rms <= 1e-4:
            raise AssertionError(
                f"Unexpected phrase output stats: sr={sr}, frames={n_frames}, dur={dur_s:.3f}s, rms={rms:.5f}"
            )

        health = http_get_json(f"http://127.0.0.1:{smoke_port}/health", timeout=2.0)
        stats = http_get_json(f"http://127.0.0.1:{smoke_port}/stats", timeout=2.0)
        recent = stats.get("recent", [])
        phrase_entries = [r for r in recent if str(r.get("note_name", "")).startswith("phrase:")]
        prep_entries = [r for r in recent if str(r.get("note_name", "")).startswith("prep:")]
        if (
            health.get("ready") is not True
            or stats.get("total_requests") != 3
            or stats.get("success_count") != 3
            or stats.get("error_count") != 0
            or len(prep_entries) != 2
            or len(phrase_entries) != 1
            or any(float(r.get("synthesis_ms", -1)) != 0.0 for r in prep_entries)
            or phrase_entries[0].get("success") is not True
            or float(phrase_entries[0].get("synthesis_ms", 0)) <= 0.0
            or int(phrase_entries[0].get("output_samples", 0)) <= 30000
        ):
            raise AssertionError(f"Unexpected /stats after cold phrase render: {json.dumps(stats)}")

        log_path = melos_dir / "logs" / "hifiserver.log"
        if not log_path.is_file():
            raise AssertionError(f"Expected managed log file missing: {log_path}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Loaded vocoder" not in log_text or "Loaded HN-SEP" not in log_text:
            raise AssertionError(f"Managed server log incomplete:\n{log_text}")

        print(f"[smoke] Stage 3 passed (cold auto-launch from copied OpenUtau folders on port {smoke_port}):")
        print(f"  WAV: {sr} Hz, {n_frames} samples ({dur_s:.3f}s), rms={rms:.4f}, peak={peak:.4f}")
        print(f"  /stats: {phrase_entries[0]['note_name']} synthesis_ms={phrase_entries[0]['synthesis_ms']:.2f}")
        print("  logs/hifiserver.log tail:")
        for line in log_text.strip().splitlines()[-10:]:
            print(f"    | {line}")

        return {
            "platform": meta["platform"],
            "port": smoke_port,
            "sr": sr,
            "samples": n_frames,
            "duration_s": round(dur_s, 3),
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "note_name": phrase_entries[0]["note_name"],
            "synthesis_ms": round(float(phrase_entries[0]["synthesis_ms"]), 2),
        }
    finally:
        kill_smoke_server(melos_dir / server_bin)
        for _ in range(10):
            try:
                if smoke_root.exists():
                    shutil.rmtree(smoke_root)
                break
            except PermissionError:
                time.sleep(0.3)
        if smoke_root.exists():
            raise RuntimeError(f"Failed to clean temporary smoke directory: {smoke_root}")
        print("[smoke] Temporary smoke directory deleted cleanly.")


def write_platform_summary_json(out_dir: Path, pkg_platform: str, assets: list[tuple[str, int, str]], smoke: dict | None) -> Path:
    summary_path = out_dir / f"summary-{pkg_platform}.json"
    summary = {
        "platform": pkg_platform,
        "assets": [{"name": n, "size": s, "sha256": d} for n, s, d in assets],
        "smoke": smoke,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Melos release archives and run isolated smoke tests.")
    parser.add_argument("--version", default="v0.2.0", help="Release tag version (default: v0.2.0)")
    parser.add_argument(
        "--platform",
        default=None,
        help="Target package flavor (win64-dml, macos-arm64-coreml, macos-x64-coreml, linux-x64-aio, linux-arm64-aio)",
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="Repository root directory")
    parser.add_argument(
        "--ort-dir",
        type=Path,
        default=None,
        help="Path to onnxruntime package root containing capi/, LICENSE, ThirdPartyNotices.txt",
    )
    parser.add_argument("--models-root", type=Path, default=None, help="Root directory containing ONNX model dirs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated archives and checksum summary",
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip isolated AIO smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pkg_platform = args.platform or detect_default_platform()
    repo_root = args.repo_root.resolve()
    ort_root = (args.ort_dir or (repo_root / ".venv" / "Lib" / "site-packages" / "onnxruntime")).resolve()
    models_root = (args.models_root or repo_root).resolve()
    out_dir = (args.out_dir or (repo_root / "TestResults" / "release_build")).resolve()

    aio_path, bins_path, meta = build_archives(
        repo_root, ort_root, models_root, out_dir, args.version, pkg_platform
    )
    smoke_summary = None if args.skip_smoke else run_isolated_smoke_test(aio_path, args.version, meta)

    melos_yaml = repo_root / "rust" / "melos.yaml"
    assets = [
        (aio_path.name, aio_path.stat().st_size, sha256_file(aio_path)),
        (bins_path.name, bins_path.stat().st_size, sha256_file(bins_path)),
        (melos_yaml.name, melos_yaml.stat().st_size, sha256_file(melos_yaml)),
    ]
    print(f"\n=== RELEASE ARTIFACT CHECKSUMS ({pkg_platform}) ===")
    for name, size, digest in assets:
        print(f"{name}: {size} bytes | sha256={digest}")

    summary_path = write_platform_summary_json(out_dir, pkg_platform, assets, smoke_summary)
    print(f"[summary] Wrote {summary_path}")


if __name__ == "__main__":
    main()
