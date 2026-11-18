#!/usr/bin/env python3
"""Build Melos release archives, run 3-stage isolated smoke tests, and emit release notes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import struct
import subprocess
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


def make_aio_config_yaml(version_num: str) -> str:
    return f"""\
# Melos Default Configuration (Windows x64 DirectML All-in-One)
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


def add_file(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing required release asset: {src}")
    zf.write(src, arcname=arcname)


def add_text(zf: zipfile.ZipFile, text: str, arcname: str, crlf: bool = False) -> None:
    normalized = text.replace("\r\n", "\n")
    if crlf:
        normalized = normalized.replace("\n", "\r\n")
    info = zipfile.ZipInfo(arcname, date_time=(2026, 11, 18, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, normalized.encode("utf-8"))


def populate_common_binaries(zf: zipfile.ZipFile, repo_root: Path, version_num: str) -> None:
    rust_release_dir = repo_root / "rust" / "target" / "release"
    prefix = "melos"
    add_file(zf, rust_release_dir / "melos.exe", f"{prefix}/melos.exe")
    add_file(zf, rust_release_dir / "melos-wavtool.exe", f"{prefix}/melos-wavtool.exe")
    add_file(zf, rust_release_dir / "hifiserver.exe", f"{prefix}/hifiserver.exe")
    add_file(zf, repo_root / "rust" / "melos.yaml", f"{prefix}/melos.yaml")
    add_text(zf, PACKAGED_START_BAT, f"{prefix}/start.bat", crlf=True)
    add_file(zf, repo_root / "LICENSE", f"{prefix}/LICENSE")
    add_file(zf, repo_root / "README.md", f"{prefix}/README.md")
    add_file(zf, repo_root / "README_zh_cn.md", f"{prefix}/README_zh_cn.md")
    add_file(zf, repo_root / "hnsep" / "LICENSE", f"{prefix}/hnsep/LICENSE")
    add_text(zf, make_aio_config_yaml(version_num), f"{prefix}/config.default.yaml", crlf=True)


def assert_clean_zip(zip_path: Path, expect_models: bool) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    basenames = {Path(n).name.lower() for n in names if not n.endswith("/")}
    bad = basenames & FORBIDDEN_ENTRIES
    if bad:
        raise AssertionError(f"{zip_path.name} contains forbidden files: {sorted(bad)}")
    if not expect_models:
        if any(n.endswith(".onnx") or n.endswith(".dll") for n in names):
            raise AssertionError(f"{zip_path.name} unexpectedly contains ONNX weights or DLLs")
    else:
        required_subpaths = {
            "melos/melos.exe",
            "melos/melos-wavtool.exe",
            "melos/hifiserver.exe",
            "melos/melos.yaml",
            "melos/start.bat",
            "melos/config.default.yaml",
            "melos/config.yaml",
            "melos/onnxruntime.dll",
            "melos/onnxruntime_providers_shared.dll",
            "melos/DirectML.dll",
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
        missing = required_subpaths - set(names)
        if missing:
            raise AssertionError(f"{zip_path.name} missing required AIO files: {sorted(missing)}")
    return names


def build_archives(
    repo_root: Path,
    ort_root: Path,
    models_root: Path,
    out_dir: Path,
    version_tag: str,
) -> tuple[Path, Path]:
    version_num = version_tag.lstrip("vV")
    out_dir.mkdir(parents=True, exist_ok=True)
    aio_zip = out_dir / f"melos-{version_tag}-win64-dml.zip"
    bins_zip = out_dir / f"melos-{version_tag}-win64-bins-only.zip"

    print(f"[build] Writing {bins_zip} ...")
    with zipfile.ZipFile(bins_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        populate_common_binaries(zf, repo_root, version_num)
    bins_names = assert_clean_zip(bins_zip, expect_models=False)
    print(f"[build] {bins_zip.name}: {len(bins_names)} entries, {bins_zip.stat().st_size:,} bytes")

    print(f"[build] Writing {aio_zip} ...")
    ort_capi = ort_root / "capi"
    vocoder_dir = models_root / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
    hnsep_vr_dir = models_root / "hnsep" / "vr"
    prefix = "melos"
    with zipfile.ZipFile(aio_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        populate_common_binaries(zf, repo_root, version_num)
        add_text(zf, make_aio_config_yaml(version_num), f"{prefix}/config.yaml", crlf=True)

        for dll_name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll", "DirectML.dll"):
            add_file(zf, ort_capi / dll_name, f"{prefix}/{dll_name}")
        add_file(zf, ort_root / "LICENSE", f"{prefix}/THIRD_PARTY_NOTICES/onnxruntime-LICENSE.txt")
        add_file(
            zf,
            ort_root / "ThirdPartyNotices.txt",
            f"{prefix}/THIRD_PARTY_NOTICES/onnxruntime-ThirdPartyNotices.txt",
        )

        voc_arc = f"{prefix}/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
        for name in ("model.onnx", "config.json", "NOTICE.txt", "NOTICE.zh-CN.txt", "STATEMENTS.txt"):
            add_file(zf, vocoder_dir / name, f"{voc_arc}/{name}")

        add_file(zf, hnsep_vr_dir / "model.onnx", f"{prefix}/hnsep/vr/model.onnx")
        add_file(zf, hnsep_vr_dir / "config.yaml", f"{prefix}/hnsep/vr/config.yaml")

    aio_names = assert_clean_zip(aio_zip, expect_models=True)
    print(f"[build] {aio_zip.name}: {len(aio_names)} entries, {aio_zip.stat().st_size:,} bytes")
    return aio_zip, bins_zip


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


def kill_server_by_exe_path(exe_path: Path) -> None:
    if os.name != "nt":
        return
    norm = str(exe_path.resolve()).replace("'", "''")
    ps_cmd = (
        f"Get-CimInstance Win32_Process | Where-Object {{ $_.ExecutablePath -eq '{norm}' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True,
        timeout=10,
    )


def run_isolated_smoke_test(aio_zip: Path, version_tag: str) -> dict:
    version_num = version_tag.lstrip("vV")
    smoke_root = Path(tempfile.mkdtemp(prefix="melos_aio_smoke_"))
    print(f"[smoke] Isolated temp root: {smoke_root}")
    melos_dir = smoke_root / "extracted" / "melos"

    try:
        with zipfile.ZipFile(aio_zip, "r") as zf:
            zf.extractall(smoke_root / "extracted")

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

        # Stage 1: Verify --version on all 3 extracted binaries
        for bin_name, expected_prefix in (
            ("hifiserver.exe", f"hifiserver {version_num}"),
            ("melos.exe", f"melos {version_num}"),
            ("melos-wavtool.exe", f"melos-wavtool {version_num}"),
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

        # Stage 2: Register locator beacon via extracted start.bat --register-only
        smoke_port = choose_smoke_port(8574)
        reg_args = "start.bat --register-only" if smoke_port == 8574 else f"start.bat --register-only --port {smoke_port}"
        reg_res = subprocess.run(
            ["cmd.exe", "/d", "/c", reg_args],
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
                f"{reg_args} failed with rc={reg_res.returncode}\nSTDOUT:\n{reg_res.stdout}\nSTDERR:\n{reg_res.stderr}"
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
        if Path(beacon_data["server_exe"]).resolve() != (melos_dir / "hifiserver.exe").resolve():
            raise AssertionError(f"Beacon server_exe mismatch: {beacon_data}")
        print(f"[smoke] Stage 2 passed: ~/.melos/server.json registered on port {smoke_port}; ~/.hifisampler untouched.")

        # Stage 3: Simulate real OpenUtau layout & cold auto-launch via ~/.melos/server.json (no server pre-started)
        shutil.copy2(melos_dir / "melos.exe", openutau_resamplers / "melos.exe")
        shutil.copy2(melos_dir / "melos.yaml", openutau_resamplers / "melos.yaml")
        shutil.copy2(melos_dir / "melos-wavtool.exe", openutau_wavtools / "melos-wavtool.exe")

        in_wav = work_dir / "oto" / "a.wav"
        write_synthetic_voice_wav(in_wav)

        helper_bat = work_dir / "temp_helper.bat"
        helper_bat.write_text(
            "@chcp 65001 >nul\r\n"
            '@if not exist "%temp%" "%resamp%" %1 "%temp%" %2 %vel% %flag% %5 %6 %7 %8 %params%\r\n'
            "@if errorlevel 1 exit /b 1\r\n"
            '@"%tool%" "%output%" "%temp%" %4 %3 0 10 0 100 100 0 0\r\n'
            "@if errorlevel 1 exit /b 1\r\n",
            encoding="utf-8",
        )

        temp_bat = work_dir / "temp.bat"
        temp_bat.write_text(
            "@chcp 65001 >nul\r\n"
            "@set loadmodule=\r\n"
            "@set tempo=120\r\n"
            "@set samples=44100\r\n"
            f"@set oto={work_dir / 'oto'}\r\n"
            f"@set tool={openutau_wavtools / 'melos-wavtool.exe'}\r\n"
            f"@set resamp={openutau_resamplers / 'melos.exe'}\r\n"
            f"@set output={work_dir / 'phrase_out.wav'}\r\n"
            "@set helper=temp_helper.bat\r\n"
            f"@set cachedir={work_dir / 'cache'}\r\n"
            '@del "%output%" 2>nul\r\n'
            '@set flag="Ht15"\r\n'
            "@set vel=100\r\n"
            f"@set temp={work_dir / 'cache' / 'note1.wav'}\r\n"
            "@set params=100 0 !120 AA\r\n"
            '@call %helper% "%oto%\\a.wav" C4 480@120+0 0 40 600 100 -300 0 10 0 100 100 0 0\r\n'
            "@if errorlevel 1 exit /b 1\r\n"
            '@set flag="Ht15"\r\n'
            "@set vel=100\r\n"
            f"@set temp={work_dir / 'cache' / 'note2.wav'}\r\n"
            "@set params=100 0 !120 AA\r\n"
            '@call %helper% "%oto%\\a.wav" D4 480@120+0 0 40 600 100 -300 0 10 0 100 100 0 0\r\n'
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

        out_wav = work_dir / "phrase_out.wav"
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
        has_ep = ("DmlExecutionProvider" in log_text) or ("CPUExecutionProvider" in log_text)
        if not has_ep or "Loaded vocoder" not in log_text or "Loaded HN-SEP" not in log_text:
            raise AssertionError(f"Managed server log incomplete:\n{log_text}")

        print(f"[smoke] Stage 3 passed (cold auto-launch from copied OpenUtau folders on port {smoke_port}):")
        print(f"  WAV: {sr} Hz, {n_frames} samples ({dur_s:.3f}s), rms={rms:.4f}, peak={peak:.4f}")
        print(f"  /stats: {phrase_entries[0]['note_name']} synthesis_ms={phrase_entries[0]['synthesis_ms']:.2f}")
        print("  logs/hifiserver.log tail:")
        for line in log_text.strip().splitlines()[-10:]:
            print(f"    | {line}")

        return {
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
        kill_server_by_exe_path(melos_dir / "hifiserver.exe")
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


def write_release_notes(
    out_dir: Path,
    version_tag: str,
    assets: list[tuple[str, int, str]],
    smoke_summary: dict | None,
) -> Path:
    notes_path = out_dir / f"RELEASE_NOTES_melos_{version_tag}.md"
    aio_zip_name = f"melos-{version_tag}-win64-dml.zip"
    bins_zip_name = f"melos-{version_tag}-win64-bins-only.zip"

    table_rows = "\n".join(
        f"| `{name}` | {size:,} bytes ({size / (1024 * 1024):.2f} MiB) | `{digest}` |"
        if size >= 1024 * 1024
        else f"| `{name}` | {size:,} bytes | `{digest}` |"
        for name, size, digest in assets
    )
    smoke_note = (
        f"- **独立环境三阶段全冷启动冒烟测试**：将 `{aio_zip_name}` 解压至仓库外目录，在纯净环境（`PATH=C:\\Windows\\System32;C:\\Windows`）依次验证 `--version` 校验、`start.bat --register-only` 写入 `~/.melos/server.json`（不触碰 `~/.hifisampler`），以及将 `melos.exe`+`melos.yaml` 与 `melos-wavtool.exe` 分别复制至模拟 `OpenUtau/Resamplers`、`OpenUtau/Wavtools` 后由客户端冷拉起 `hifiserver.exe --managed`（写入 `logs/hifiserver.log` 并启用 DirectML），完成带 `Ht15` 的 2 音符整句单次声码渲染（`{smoke_summary['note_name']}`，端口 `{smoke_summary['port']}`，输出 {smoke_summary['sr']} Hz / {smoke_summary['duration_s']}s，RMS={smoke_summary['rms']}，声码器耗时 {smoke_summary['synthesis_ms']} ms）。"
        if smoke_summary
        else "- **Smoke Test**: Skipped during this archive run."
    )

    notes = f"""\
# {version_tag}-melos — 纯 Rust 后端 & 整句 Mel+F0 连续声码渲染

本 Release 从 [`melos`](https://github.com/openhachimi/hifisampler/tree/melos) 分支独立发布，仅包含 **Melos**（纯 Rust 架构与连续短语合成引擎），与 `main` 分支的 Python `hifisampler` 保持独立区分（已设置 `--latest=false`，不改变仓库 `/releases/latest` 指向）。

> [!NOTE]
> **分支版本号奇偶约定 / Branch Versioning Convention**
> - **Melos（`melos` 分支）**：从 `0.2.0` 起步并采用**偶数次版本号**（`0.2.0`、`0.4.0`、`0.6.0`……，Release 标题为 `{version_tag}-melos`）。
> - **hifisampler（`main` 分支）**：采用**奇数次版本号**（`0.1.0`、`0.3.0`、`0.5.0`……，Release 标题为 `v0.1.0-hifi` 等）。
> - **同机并存隔离**：Melos 默认监听端口 `8574`（`http://127.0.0.1:8574`）并将服务端信标写入 `~/.melos/server.json`，与旧版端口 `8572` / `~/.hifisampler/server.json` 互相隔离，可同机并存。

---

## 📦 下载文件说明 / Download Assets

| 文件名 (Asset) | 大小 (Size) | SHA-256 |
| --- | --- | --- |
{table_rows}

- **`{aio_zip_name}`（推荐 / Windows x64 DirectML 一键整合包）**：解压即用，内置 `melos.exe`、`melos-wavtool.exe`、`hifiserver.exe`、`melos.yaml`、纯 Melos `start.bat`、`config.yaml`/`config.default.yaml`，以及已转换的 `pc_nsf_hifigan` ONNX 声码器、`hnsep` ONNX 谐波噪声分离模型和 Microsoft ONNX Runtime + DirectML 运行时 DLL；**无需安装 Python 或 CUDA**，在原生 OpenUtau 客户端即可使用整句 Mel+F0 连续合成。
- **`{bins_zip_name}`（纯开源二进制包 / Pure OSS Binaries Only）**：仅包含 Apache-2.0 编译的 `melos.exe`、`melos-wavtool.exe`、`hifiserver.exe`、`melos.yaml`、纯 Melos `start.bat`、`config.default.yaml` 与 `hnsep/LICENSE`；**不含**模型权重与闭源运行时 DLL，适合自行准备模型或合规审计使用。
- **`melos.yaml`**：OpenUtau 表情配置清单（与压缩包内相同，方便单独更新）。

---

## 🚀 原生 OpenUtau 快速配置 / Quick Setup (Native OpenUtau)

1. 下载并解压 **`{aio_zip_name}`** 到任意英文或中文路径。
2. **首次使用双击运行一次目录中的 `start.bat`**（自动生成 `config.yaml`、注册 `~/.melos/server.json` 并预热引擎；后续日常渲染无需提前启动，客户端会自动按需拉起托管服务，后台日志自动记录在 `logs/hifiserver.log`）。
3. 接入原生 OpenUtau（保持 `CLASSIC` 渲染器）：
   - 将 `melos.exe` 与 `melos.yaml` 复制到 OpenUtau 的 `Resamplers/` 目录；
   - 将 `melos-wavtool.exe` 复制到 OpenUtau 的 `Wavtools/` 目录；
   - 在轨道头将 **Resampler** 设为 `melos.exe`，**Wavtool** 设为 `melos-wavtool.exe`（若切回 `convergence` / `simple`，`melos.exe` 也会自动退回单音兼容模式）。

---

## ⚖️ 分发协议与模型许可证说明 / Distribution & Model License Terms

> [!IMPORTANT]
> **`{aio_zip_name}` 包含受非商业条款（Non-Commercial）约束的第三方预训练模型权重及闭源 DirectML 运行时组件，仅供个人研究、非商业艺术创作与技术评估；严禁将预训练模型权重或完整整合包用于任何商业用途。**

| 组件 (Component) | 适用许可证 / 条款 (License / Terms) | 说明与随附文件 (Notes & Bundled Notices) |
| --- | --- | --- |
| Melos Rust 源码与二进制（`hifiserver.exe` / `melos.exe` / `melos-wavtool.exe`） | **Apache-2.0** | 见包内 `LICENSE`（`{bins_zip_name}` 仅含本类与 `hnsep/LICENSE`）。 |
| `hnsep/` 脚本与许可证声明 | **MIT** | 上游 `yxlllc/vocal-remover`（`Copyright (c) 2019 tsurumeso`），见 `hnsep/LICENSE`。 |
| `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` ONNX 权重 | **CC BY-NC-SA 4.0（署名-非商业性使用-相同方式共享）** | 上游为 OpenVPI / DiffSinger 社区公开权重；包内保留原始 `NOTICE.txt`、`NOTICE.zh-CN.txt` 与 `STATEMENTS.txt`，未调整训练参数。 |
| `hnsep/vr/model.onnx`（240512） | **上游非商业声明 / Non-Commercial Notice** | 由 `yxlllc/vocal-remover` 发布的 `hnsep_240512` checkpoint 导出；上游仓库明确限制预训练模型权重不可用于商业用途。 |
| ONNX Runtime & DirectML DLL（`onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll`） | **MIT（ORT） / Microsoft Redistributable Notice（DirectML）** | 随附 `THIRD_PARTY_NOTICES/onnxruntime-LICENSE.txt` 与 `THIRD_PARTY_NOTICES/onnxruntime-ThirdPartyNotices.txt`。 |

---

## 🧪 验证摘要 / Verification Summary

{smoke_note}
"""
    notes_path.write_text(notes, encoding="utf-8")
    print(f"[notes] Wrote {notes_path}")
    return notes_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Melos release archives and run isolated smoke tests.")
    parser.add_argument("--version", default="v0.2.0", help="Release tag version (default: v0.2.0)")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT, help="Repository root directory")
    parser.add_argument(
        "--ort-dir",
        type=Path,
        default=None,
        help="Path to onnxruntime package root containing capi/*.dll, LICENSE, ThirdPartyNotices.txt",
    )
    parser.add_argument("--models-root", type=Path, default=None, help="Root directory containing ONNX model dirs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated zips and release notes",
    )
    parser.add_argument("--skip-smoke", action="store_true", help="Skip isolated AIO smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ort_root = (args.ort_dir or (repo_root / ".venv" / "Lib" / "site-packages" / "onnxruntime")).resolve()
    models_root = (args.models_root or repo_root).resolve()
    out_dir = (args.out_dir or (repo_root / "TestResults" / "release_build")).resolve()

    aio_zip, bins_zip = build_archives(repo_root, ort_root, models_root, out_dir, args.version)
    smoke_summary = None if args.skip_smoke else run_isolated_smoke_test(aio_zip, args.version)

    melos_yaml = repo_root / "rust" / "melos.yaml"
    assets = [
        (aio_zip.name, aio_zip.stat().st_size, sha256_file(aio_zip)),
        (bins_zip.name, bins_zip.stat().st_size, sha256_file(bins_zip)),
        (melos_yaml.name, melos_yaml.stat().st_size, sha256_file(melos_yaml)),
    ]
    print("\n=== RELEASE ARTIFACT CHECKSUMS ===")
    for name, size, digest in assets:
        print(f"{name}: {size} bytes | sha256={digest}")

    write_release_notes(out_dir, args.version, assets, smoke_summary)


if __name__ == "__main__":
    main()
