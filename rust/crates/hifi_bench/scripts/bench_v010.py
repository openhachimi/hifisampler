"""Isolated benchmark runner for v0.1.0 (tag v0.1.0 / 1b59be1).

Measures server cold-start, CLI cold/warm, cold/warm render wall latencies,
sequential and 2-worker concurrent 20-note phrase bursts, and peak process-tree RSS
without modifying C:\\Project\\hifi workspace configuration or beacon state.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def nearest_rank_percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return float(sorted_vals[idx])


def latency_summary(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "throughput_nps": 0.0,
        }
    s = sorted(samples_ms)
    mean_ms = statistics.fmean(s)
    return {
        "count": len(s),
        "mean_ms": mean_ms,
        "min_ms": s[0],
        "max_ms": s[-1],
        "p50_ms": nearest_rank_percentile(s, 50.0),
        "p95_ms": nearest_rank_percentile(s, 95.0),
        "throughput_nps": (1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    }


def generate_harmonic_vocal_wav(path: Path, freq_hz: float = 220.0, duration_sec: float = 1.0, sr: int = 44100) -> None:
    n_samples = max(1, int(round(sr * duration_sec)))
    fade_len = min(n_samples // 4, int(round(sr * 0.01)))
    samples = np.empty(n_samples, dtype=np.float32)
    rng = 0x5EED_CAFE_1234_5678
    mask64 = (1 << 64) - 1
    u32_max = float(0xFFFFFFFF)

    harm_scale = [1.0 / (k ** 1.2) for k in range(1, 11)]
    for i in range(n_samples):
        t = i / sr
        h = 0.0
        for k in range(1, 11):
            h += harm_scale[k - 1] * math.sin(2.0 * math.pi * freq_hz * k * t)
        rng = (rng * 6364136223846793005 + 1442695040888963407) & mask64
        u = ((rng >> 33) / u32_max) * 2.0 - 1.0
        if fade_len > 0 and i < fade_len:
            env = i / fade_len
        elif fade_len > 0 and i >= n_samples - fade_len:
            env = (n_samples - 1 - i) / fade_len
        else:
            env = 1.0
        val = env * (0.18 * h + 0.012 * u)
        samples[i] = max(-0.95, min(0.95, val))

    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def inspect_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        sw = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    all_finite = bool(np.all(np.isfinite(arr)))
    peak = float(np.max(np.abs(arr))) if len(arr) else 0.0
    rms = float(np.sqrt(np.mean(arr * arr))) if len(arr) else 0.0
    return {
        "samples": int(len(arr)),
        "all_finite": all_finite,
        "peak": peak,
        "rms_db": float(20.0 * math.log10(max(rms, 1e-12))),
    }


def process_tree_peak_rss_bytes(root_pid: int) -> int | None:
    if os.name != "nt":
        return None
    ps_script = (
        "$procs = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId; "
        f"$seen = @{{ {root_pid} = $true }}; "
        "$q = [System.Collections.Generic.Queue[int]]::new(); "
        f"$q.Enqueue({root_pid}); "
        "while ($q.Count -gt 0) { "
        "$cur = $q.Dequeue(); "
        "foreach ($p in $procs) { "
        "if ($p.ParentProcessId -eq $cur -and -not $seen.ContainsKey([int]$p.ProcessId)) { "
        "$seen[[int]$p.ProcessId] = $true; $q.Enqueue([int]$p.ProcessId) } } }; "
        "$total = [int64]0; "
        "foreach ($id in $seen.Keys) { "
        "$gp = Get-Process -Id $id -ErrorAction SilentlyContinue; "
        "if ($gp) { $total += [int64]$gp.PeakWorkingSet64 } }; "
        "Write-Output $total"
    )
    res = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    try:
        val = int(res.stdout.strip())
        return val if val > 0 else None
    except ValueError:
        return None


def kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def kill_v010_strays(v010_dir: Path) -> None:
    if os.name != "nt":
        return
    my_pid = os.getpid()
    needle = str(v010_dir).replace("'", "''")
    ps = (
        f"Get-CimInstance Win32_Process | Where-Object {{ "
        f"($_.ProcessId -ne $PID) -and ($_.ProcessId -ne {my_pid}) -and "
        f"$_.CommandLine -and $_.CommandLine.Contains('{needle}') -and $_.CommandLine.Contains('hifiserver.py') }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, check=False)


def http_get_ready(port: int) -> tuple[int, str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/", timeout=1.0) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def post_note(port: int, args: list[str]) -> float:
    payload = json.dumps({"args": args}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter()
    with opener.open(req, timeout=180.0) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if resp.status != 200:
            raise RuntimeError(f"POST returned {resp.status}: {body}")
    return (time.perf_counter() - t0) * 1000.0


class V010Server:
    def __init__(self, python_exe: Path, v010_dir: Path, home_dir: Path, port: int, log_path: Path):
        self.port = port
        self.log_path = log_path
        env = os.environ.copy()
        env["USERPROFILE"] = str(home_dir)
        env["HOME"] = str(home_dir)
        env["HIFISAMPLER_PORT"] = str(port)
        env["HIFISAMPLER_HOME"] = str(v010_dir)
        env["PYTHONUNBUFFERED"] = "1"
        self._log_file = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [str(python_exe), str(v010_dir / "hifiserver.py")],
            cwd=str(v010_dir),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout_sec: float = 120.0) -> float:
        t0 = time.perf_counter()
        deadline = t0 + timeout_sec
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                self._log_file.flush()
                logs = self.log_path.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(f"v0.1.0 server exited {self.proc.returncode}:\n{logs}")
            try:
                status, body = http_get_ready(self.port)
                if status == 200 and "Server Ready" in body:
                    return (time.perf_counter() - t0) * 1000.0
            except Exception:
                pass
            time.sleep(0.05)
        self._log_file.flush()
        logs = self.log_path.read_text(encoding="utf-8", errors="replace")
        raise TimeoutError(f"Timed out waiting for v0.1.0 server on port {self.port}:\n{logs}")

    def peak_rss_bytes(self) -> int | None:
        return process_tree_peak_rss_bytes(self.proc.pid)

    def shutdown(self) -> None:
        kill_tree(self.proc.pid)
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            self._log_file.close()
        except Exception:
            pass


def build_batch_jobs(in_wav: Path, out_dir: Path, mixed: bool) -> list[list[str]]:
    pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "G4", "E4"]
    flag_sets = ["", "P100", "g-10", "A30", "Hb120Hv90"] if mixed else [""]
    lengths = ["220", "280", "350", "400"]
    jobs = []
    for i in range(20):
        out_wav = out_dir / f"batch_{i}.wav"
        jobs.append(
            [
                str(in_wav),
                str(out_wav),
                pitches[i % len(pitches)],
                "100",
                flag_sets[i % len(flag_sets)],
                "0",
                lengths[i % len(lengths)],
                "0",
                "0",
                "100",
                "0",
                "!120",
                "AA",
            ]
        )
    return jobs


def run_batch(port: int, in_wav: Path, out_dir: Path, mixed: bool, concurrency: int) -> dict[str, object]:
    jobs = build_batch_jobs(in_wav, out_dir, mixed)
    client_times: list[float] = []
    t0 = time.perf_counter()
    if concurrency <= 1:
        for j in jobs:
            client_times.append(post_note(port, j))
    else:
        buckets = [jobs[w::concurrency] for w in range(concurrency)]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(lambda b: [post_note(port, j) for j in b], bucket) for bucket in buckets]
            for f in futs:
                client_times.extend(f.result())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    for i in range(20):
        out_wav = out_dir / f"batch_{i}.wav"
        if not out_wav.is_file():
            raise RuntimeError(f"Missing batch output: {out_wav}")
    return {
        "note_count": 20,
        "concurrency": concurrency,
        "wall_ms": wall_ms,
        "notes_per_sec": (20.0 * 1000.0) / max(wall_ms, 1e-6),
        "client_latency": latency_summary(client_times),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark v0.1.0 baseline in isolated directory")
    parser.add_argument("--v010-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--python-exe", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    v010_dir = args.v010_dir.resolve()
    repo_root = args.repo_root.resolve()
    py_exe = Path(os.path.abspath(str(args.python_exe)))
    cs_cli = v010_dir / "hifisampler.exe"
    if not cs_cli.is_file():
        raise FileNotFoundError(f"Missing {cs_cli}")

    work_dir = Path(tempfile.gettempdir()) / f"hifi_v010_bench_{uuid.uuid4().hex[:10]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    home_dir = work_dir / "isolated_home"
    home_dir.mkdir(parents=True, exist_ok=True)

    synth_wav = work_dir / "bench_harmonic_440.wav"
    generate_harmonic_vocal_wav(synth_wav, 220.0, 1.0, 44100)

    real_vocal_orig = repo_root / "TestResults/vocal_fry_demo/isolated_vb_cache/A3__ああいあうえあ.wav"
    real_vocal_copy = work_dir / "real_vocal_a3.wav"
    has_real_vocal = real_vocal_orig.is_file()
    if has_real_vocal:
        shutil.copy2(real_vocal_orig, real_vocal_copy)

    try:
        # 1. Server cold-start (2 repeats) + probe RSS
        print("[v0.1.0] 1/6 Measuring server cold-start latency...", flush=True)
        startup_ms: list[float] = []
        probe_rss: int | None = None
        for r in range(2):
            p = find_free_port()
            srv = V010Server(py_exe, v010_dir, home_dir, p, work_dir / f"v010_boot_{r}.log")
            try:
                ms = srv.wait_ready(120.0)
                print(f"[v0.1.0]   boot {r}: {ms:.1f} ms", flush=True)
                startup_ms.append(ms)
                if r == 1:
                    probe_in = work_dir / "probe_in.wav"
                    probe_out = work_dir / "probe_out.wav"
                    shutil.copy2(synth_wav, probe_in)
                    post_note(p, [str(probe_in), str(probe_out), "A4", "100", "Ht20", "0", "300", "0", "0", "100", "0", "!120", "AA"])
                    probe_rss = srv.peak_rss_bytes()
                    print(f"[v0.1.0]   probe_rss: {probe_rss}", flush=True)
            finally:
                srv.shutdown()

        # Register isolated beacon for C# CLI cold test (no .venv inside v010_dir)
        reg_env = os.environ.copy()
        reg_env["USERPROFILE"] = str(home_dir)
        reg_env["HOME"] = str(home_dir)
        subprocess.run(
            [str(py_exe), str(v010_dir / "hifiserver.py"), "--register-only"],
            cwd=str(v010_dir),
            env=reg_env,
            check=True,
            capture_output=True,
            timeout=30,
        )

        # CLI cold-start
        print("[v0.1.0] Measuring CLI cold-start...", flush=True)
        cli_port = find_free_port()
        cli_in = work_dir / "cli_cold_in.wav"
        cli_out = work_dir / "cli_cold_out.wav"
        shutil.copy2(synth_wav, cli_in)
        cli_env = os.environ.copy()
        cli_env["USERPROFILE"] = str(home_dir)
        cli_env["HOME"] = str(home_dir)
        cli_env["HIFISAMPLER_PORT"] = str(cli_port)
        cli_env["HIFISAMPLER_HOME"] = str(v010_dir)
        t0 = time.perf_counter()
        res = subprocess.run(
            [str(cs_cli), str(cli_in), str(cli_out), "A4", "100", "", "0", "300", "0", "0", "100", "0", "!120", "AA"],
            cwd=str(v010_dir),
            env=cli_env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        cli_cold_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[v0.1.0]   cli_cold_ms={cli_cold_ms:.1f}, rc={res.returncode}", flush=True)
        kill_v010_strays(v010_dir)
        if res.returncode != 0 or not cli_out.is_file():
            raise RuntimeError(f"v0.1.0 CLI cold failed ({res.returncode}): {res.stderr}\n{res.stdout}")

        # Start main long-lived v0.1.0 server
        main_port = find_free_port()
        srv = V010Server(py_exe, v010_dir, home_dir, main_port, work_dir / "v010_main.log")
        try:
            boot_main = srv.wait_ready(120.0)
            startup_ms.append(boot_main)
            print(f"[v0.1.0]   main server ready: {boot_main:.1f} ms", flush=True)

            # CLI warm (5 runs)
            cli_warm_dir = work_dir / "cli_warm"
            cli_warm_dir.mkdir(parents=True, exist_ok=True)
            cli_warm_in = cli_warm_dir / "in.wav"
            shutil.copy2(synth_wav, cli_warm_in)
            post_note(main_port, [str(cli_warm_in), str(cli_warm_dir / "prewarm.wav"), "A4", "100", "", "0", "300", "0", "0", "100", "0", "!120", "AA"])
            cli_warm_env = os.environ.copy()
            cli_warm_env["USERPROFILE"] = str(home_dir)
            cli_warm_env["HOME"] = str(home_dir)
            cli_warm_env["HIFISAMPLER_PORT"] = str(main_port)
            cli_warm_env["HIFISAMPLER_HOME"] = str(v010_dir)
            warm_samples: list[float] = []
            for i in range(5):
                out_w = cli_warm_dir / f"w_{i}.wav"
                t1 = time.perf_counter()
                r = subprocess.run(
                    [str(cs_cli), str(cli_warm_in), str(out_w), "A4", "100", "", "0", "300", "0", "0", "100", "0", "!120", "AA"],
                    cwd=str(v010_dir),
                    env=cli_warm_env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                warm_samples.append((time.perf_counter() - t1) * 1000.0)
                if r.returncode != 0 or not out_w.is_file():
                    raise RuntimeError(f"v0.1.0 CLI warm failed: {r.stderr}")
            cli_warm = latency_summary(warm_samples)
            print(f"[v0.1.0] 2/6 Cold render cases... (cli_warm_mean={cli_warm['mean_ms']:.1f}ms)", flush=True)

            # Cold render cases (no /stats in v0.1.0 -> HTTP wall_ms)
            cases = [
                ("default", "A4", "", "0", "400", "0", "0", False),
                ("Hb140Hv90", "A4", "Hb140Hv90", "0", "400", "0", "0", False),
                ("Ht50", "A4", "Ht50", "0", "400", "0", "0", False),
                ("HG60", "A4", "HG60", "0", "400", "0", "0", False),
                ("P100", "A4", "P100", "0", "400", "0", "0", False),
            ]
            if has_real_vocal:
                cases.extend([
                    ("real_vocal_default", "A3", "", "1200", "500", "80", "-1200", True),
                    ("real_vocal_Ht50P100", "C4", "Ht50P100", "1200", "500", "80", "-1200", True),
                ])

            cold_results: list[dict[str, object]] = []
            for name, pitch, flags, off, length, con, cut, use_real in cases:
                cdir = work_dir / "cold" / name
                cdir.mkdir(parents=True, exist_ok=True)
                cin = cdir / "input.wav"
                cout = cdir / "out.wav"
                shutil.copy2(real_vocal_copy if use_real else synth_wav, cin)
                w_ms = post_note(main_port, [str(cin), str(cout), pitch, "100", flags, off, length, con, cut, "100", "0", "!120", "AA"])
                cold_results.append({
                    "name": name,
                    "flags": flags,
                    "wall_ms": w_ms,
                    "audio": inspect_wav(cout),
                })

            print("[v0.1.0] 3/6 Warm cache...", flush=True)
            # Warm .hifi.npz hit (same-process repeat & fresh-server disk hit)
            ht_dir = work_dir / "cold" / "Ht50"
            warm_repeat_ms = post_note(main_port, [str(ht_dir / "input.wav"), str(ht_dir / "warm_repeat.wav"), "A4", "100", "Ht50", "0", "400", "0", "0", "100", "0", "!120", "AA"])

            disk_dir = work_dir / "warm_disk"
            disk_dir.mkdir(parents=True, exist_ok=True)
            disk_in = disk_dir / "input.wav"
            shutil.copy2(synth_wav, disk_in)
            post_note(main_port, [str(disk_in), str(disk_dir / "seed.wav"), "A4", "100", "Ht50", "0", "400", "0", "0", "100", "0", "!120", "AA"])
            npz_files = sorted(p.name for p in disk_dir.glob("*.hifi.npz"))
            disk_port = find_free_port()
            disk_srv = V010Server(py_exe, v010_dir, home_dir, disk_port, work_dir / "v010_disk.log")
            try:
                disk_srv.wait_ready(120.0)
                warm_disk_ms = post_note(disk_port, [str(disk_in), str(disk_dir / "warm.wav"), "A4", "100", "Ht50", "0", "400", "0", "0", "100", "0", "!120", "AA"])
            finally:
                disk_srv.shutdown()

            print("[v0.1.0] 4/6 Sequential & 2-worker concurrent batches...", flush=True)
            # Sequential & 2-worker concurrent 20-note batches
            seq_mix_dir = work_dir / "seq_mix"
            seq_mix_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(synth_wav, seq_mix_dir / "src.wav")
            seq_mixed = run_batch(main_port, seq_mix_dir / "src.wav", seq_mix_dir, True, 1)

            seq_warm_dir = work_dir / "seq_warm"
            seq_warm_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(synth_wav, seq_warm_dir / "src.wav")
            post_note(main_port, [str(seq_warm_dir / "src.wav"), str(seq_warm_dir / "pw.wav"), "C4", "100", "", "0", "250", "0", "0", "100", "0", "!120", "AA"])
            seq_warm = run_batch(main_port, seq_warm_dir / "src.wav", seq_warm_dir, False, 1)

            conc_mix_dir = work_dir / "conc_mix"
            conc_mix_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(synth_wav, conc_mix_dir / "src.wav")
            conc_mixed = run_batch(main_port, conc_mix_dir / "src.wav", conc_mix_dir, True, 2)

            conc_warm_dir = work_dir / "conc_warm"
            conc_warm_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(synth_wav, conc_warm_dir / "src.wav")
            post_note(main_port, [str(conc_warm_dir / "src.wav"), str(conc_warm_dir / "pw.wav"), "C4", "100", "", "0", "250", "0", "0", "100", "0", "!120", "AA"])
            conc_warm = run_batch(main_port, conc_warm_dir / "src.wav", conc_warm_dir, False, 2)

            full_peak_rss = srv.peak_rss_bytes()
            print(f"[v0.1.0] 5/6 full_peak_rss={full_peak_rss}", flush=True)
        finally:
            srv.shutdown()

        report = {
            "version": "0.1.0",
            "git_tag": "v0.1.0",
            "commit": "1b59be1",
            "server_startup_ms": startup_ms,
            "cli_cold_ms": cli_cold_ms,
            "cli_warm": cli_warm,
            "probe_peak_rss_bytes": probe_rss,
            "full_peak_rss_bytes": full_peak_rss,
            "cold_cases": cold_results,
            "warm_cache": {
                "cache_files": npz_files,
                "repeat_wall_ms": warm_repeat_ms,
                "disk_hit_wall_ms": warm_disk_ms,
            },
            "mixed_batch_seq": seq_mixed,
            "warm_batch_seq": seq_warm,
            "concurrent_mixed_batch": conc_mixed,
            "concurrent_warm_batch": conc_warm,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote v0.1.0 benchmark JSON to {args.json_out}")
        return 0
    finally:
        kill_v010_strays(v010_dir)
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
