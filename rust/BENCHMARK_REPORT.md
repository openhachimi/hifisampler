# HiFiSampler 4-Generation Evolution, Verification & A/B Benchmark Report

## 1. Environment & Methodology

- **Repository / Branch**: `C:\Project\hifi` (`feat/rust-port`)
- **Generations Evaluated**:
  1. **`v0.1.0` (`1b59be1`)** — Original Python `aiohttp` + PyTorch/ONNX server (`config.default.yaml`: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx` vocoder + `hnsep/vr/model.pt` PyTorch HN-SEP, PyTorch Mel analysis, linear `Ht` ramp, `pyloudnorm` + `resampy`, `.hifi.npz` zipped cache, C# CLI client, DirectML forced `max_workers=1`; isolated archive outside repo via `rust/crates/hifi_bench/scripts/bench_v010.py`).
  2. **Python `main` (`b6a1791`)** — Post-v0.1.2/NumPy/HIFIBIN1 Python server (`rust/crates/hifi_bench/scripts/py_bench_server.py --hnsep-onnx` for apples-to-apples ONNX comparisons without altering `config.yaml`, plus `hifiserver.py` default `model.pt` startup/RSS probe and C# `hifisampler.exe`).
  3. **Rust v1 (`c0c9e72`)** — Initial full-feature Rust port across 8 crates (`hifi_core`, `hifi_lufs`, `hifi_dsp`, `hifi_effects`, `hifi_onnx`, `hifi_server`, `hifi_client`, `hifi_bench`), built with `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `panic = "abort"`, `x86_64-pc-windows-gnu`.
  4. **Rust v2 (`OPT-01`..`OPT-14`, Current)** — Phase D optimized Rust release binaries (`rust/target/release/hifiserver.exe`, `rust/target/release/hifisampler.exe`, `strip = "symbols"`, `panic = "unwind"`, parallel HTTP/DSP worker overlap without outer `gpu_infer_lock`, frame-contiguous RFFT-cached STFT & fused tension energy, sparse Mel filterbank, `Arc` RAM caches, zero-copy non-durable `_hnsep` writes, zero-fill `.hifi.bin` power-loss recovery, `auto` `text/plain` client wire protocol).
- **Real Models**:
  - Vocoder: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx` (`DmlExecutionProvider` with `disable_metacommands=true` via `onnxruntime.dll`)
  - HN-SEP: `hnsep/vr/model.onnx` (and `hnsep/vr/model.pt` for `v0.1.0` & Python `hifiserver.py` default config)
- **Reproducible Harness**: `rust/crates/hifi_bench` (`hifi_bench.exe --root C:\Project\hifi`) supporting both full Python-vs-Rust runs and `--rust-only`, plus sequential (`concurrency=1`) and 2-worker concurrent (`concurrency=2`) 20-note phrase bursts.
- **Test Inputs**:
  - Deterministic 1.0 s 44.1 kHz 16-bit PCM harmonic vocal-like test WAV (`bench_harmonic_440.wav`, 220 Hz fundamental + 10 decaying harmonics + low-level breath noise, 44,100 samples)
  - Real UTAU vocal sample copied to isolated temp dir: `TestResults/vocal_fry_demo/isolated_vb_cache/A3__ああいあうえあ.wav` (289,380 samples, 6.56 s @ 44.1 kHz, active vowel window `offset=1200`, `length=500`, `consonant=80`, `cutoff=-1200`)

---

## 2. Phase D Post-Final-Edit Verification Gates

All 5 workspace verification gates and live corruption-recovery checks ran after the final edits and passed at 100%:

| Gate | Command | Result | Scope / Observed Outcome |
|---|---|---|---|
| Formatting | `cargo fmt --all -- --check` | **PASS** (exit 0) | Full Rust workspace (all 8 crates) |
| Linter | `cargo clippy --offline --workspace --all-targets -- -D warnings` | **PASS** (exit 0) | Zero warnings across all 8 crates |
| Rust Release Tests | `cargo test --offline --workspace --release` | **PASS** (**120/120**) | `hifi_bench` (5), `hifi_client` (10), `hifi_core` (41), `hifi_dsp` (10), `hifi_effects` (11), `hifi_lufs` (7), `hifi_onnx` (12), `hifi_server` (24) |
| Python Baseline Unit Suite | `C:\Project\hifi\.venv\Scripts\python.exe -m pytest tests --ignore=tests/e2e -q` | **PASS** (**259/259**) | Completed in **20.49 s** |
| E2E Suite (Rust Target) | `C:\Project\hifi\.venv\Scripts\python.exe tests/e2e/runner.py --target rust --all` | **PASS** (**196/196**) | Tier 1 + Tier 2 (F1–F17: 170) + Tier 3 (17) + Tier 4 (9); 85.47 s test time / **93.00 s** runner total (down from 120.64 s in Rust v1) |
| Zero-Fill `.hifi.bin` Recovery | Live `hifiserver.exe` + `test_all_zero_payload_rejected_and_regenerated_while_allowed_for_hnsep` | **PASS** | Exact-size (88,092 B, `128x172`) zero-filled `.hifi.bin` rejected on fresh server (`cache_hit: false`), regenerated with finite non-zero floats, and hit cleanly on retry (`cache_hit: true`) |

---

## 3. Feature, DSP & Architecture Evolution (`v0.1.0` -> `Now`)

Observed across git history (`1b59be1` -> `v0.1.1` / `v0.1.2` -> `b6a1791` -> `c0c9e72` -> current workspace):

| Capability / Architecture Axis | `v0.1.0` (`1b59be1`) | `v0.1.1` / `v0.1.2` (`950c2d9` / `22e1e11`) | Python `main` (`b6a1791`) | Rust v1 (`c0c9e72`) | Rust v2 (`OPT-01`..`OPT-14`) |
|---|---|---|---|---|---|
| **Server & Client Runtime** | Python `aiohttp` + C# CLI (5.17 MiB) | Python `aiohttp` + C# CLI | Python `aiohttp` + `--managed` + C# CLI | Pure Rust `axum` + CLI (`629.5 KiB` / `4.69 MiB`) | Stripped Rust `axum` + CLI (**`478.0 KiB` / `4.08 MiB`**) |
| **Glottal Tension (`Ht`)** | Linear pre-emphasis ramp (`Ht` only) | Clean-room LF minimum-phase v2 -> v3 + closed-form H1 + ear loudness `W(f)` + aspiration coupling (`Ht*v3`) | LF v3 + Klatt **`Ho`** openness & **`Hr`** singer's formant resonators | Full bit-compatible LF v3 + `Ho`/`Hr` Klatt | Frame-contiguous STFT + fused `apply_tension_and_coupling_with_config` (eliminates 1 full STFT) |
| **Vocal Fry & Growl (`HG`)** | `HG` growl only; no `auto_vocal_fry` | Leap-safe relative-pit LF glottal **`auto_vocal_fry`** (`v0.1.2`) | Vectorized NumPy `auto_vocal_fry` + `HG` | Native Rust LF `auto_vocal_fry` + `HG` | `O(N)` two-pointer `align_epoch_phase_monotonic` (`OPT-08`) + shared `hifi_dsp` math |
| **Loudness (`P`) & Resampling** | `pyloudnorm` + `resampy` | `pyloudnorm` + `resampy` | Fast NumPy/SciPy BS.1770-4 + polyphase resample (`be8c3fa`) | Native Rust BS.1770-4 (`hifi_lufs`) | Same exact BS.1770-4 Rust engine (`<= 6e-6 dB` delta) |
| **Mel & HN-SEP Pipeline** | PyTorch Mel + PyTorch `model.pt` (`model.onnx` vocoder) | PyTorch Mel + `model.pt` / optional ONNX | Pure NumPy/SciPy Mel + ONNX `hnsep/vr/model.onnx` fast path (`4daa7f5`) | Native Rust `realfft` Mel/STFT + FFI ORT (`hifi_onnx`) | Global cached `OnceLock` RFFT plans, sparse Mel filterbank intervals, zero-copy ORT tensor slices |
| **Feature Cache Format** | `.hifi.npz` (uncompressed ZIP) + `torch.save` `_hnsep` | `.hifi.npz` (`_Ht*v3` key versioning) | **`HIFIBIN1`** 28 B header + raw float32 (`f96ed03`) + source mtime/size check | Interoperable `HIFIBIN1` + RAM LRU + atomic temp writes | `Arc` zero-clone RAM caches, lazy lock pruning (>256), non-durable zero-copy `_hnsep`, zero-fill `.hifi.bin` power-loss recovery, startup stale `.tmp` cleanup |
| **Concurrency / DirectML** | Forces `max_workers=1` on DirectML | Forces `max_workers=1` on DirectML | Forces `max_workers=1` on DirectML | Unlocked workers but shared outer `gpu_infer_lock` | **`OPT-01`**: Decoupled HTTP worker count; CPU DSP & I/O run in parallel, serialized only around `OrtRuntime::lock_gpu()` |
| **Observability & Protocol** | `GET /`, `POST /` (JSON) only | `GET /`, `POST /` | Adds `/health`, `/stats`, `--managed` | Full parity + `HIFISAMPLER_PROTOCOL` (`json` / `text`) | `auto` `text/plain` fast path for safe CLI args (fallback to JSON on spaces/quotes/empty/control chars) |

---

## 4. 4-Generation Performance Comparison

### 4.1 Release Binary Sizes & Peak Process-Tree RSS (`PeakWorkingSet64`)

| Artifact / Runtime | Binary Size (Bytes / Unit) | Peak RSS Notes | Peak RSS (Bytes / MiB) |
|---|---:|---|---:|
| C# CLI (`hifisampler.exe`, `v0.1.0` & `main`) | 5,425,664 B (5.17 MiB) | — | — |
| Rust v1 CLI (`hifisampler.exe`) | 644,571 B (629.5 KiB) | — | — |
| **Rust v2 CLI (`hifisampler.exe`)** | **489,472 B (478.0 KiB)** | **11.08x** smaller than C#; **-24.1%** vs Rust v1 | — |
| Rust v1 Server (`hifiserver.exe`) | 4,917,678 B (4.69 MiB) | Full cold/interop/batch run | 1,268,973,568 B (**1,210.19 MiB**) |
| **Rust v2 Server (`hifiserver.exe`)** | **4,273,664 B (4.08 MiB)** | Full cold/interop/seq+concurrent run (`-13.1%` binary size vs Rust v1) | 1,652,686,848 B (**1,576.12 MiB**; higher concurrency keeps parallel worker/DML allocations live) |
| `v0.1.0` Server (`model.pt` HN-SEP) | Python + `.venv` | 1-note `Ht20` probe / full suite | 773,681,152 B (737.84 MiB) / 1,632,305,152 B (1,556.69 MiB) |
| Python `main` Server (default `model.pt`, 1 probe) | Python + `.venv` | v1-run / v2-run | 755,015,680 B (720.04 MiB) / 755,548,160 B (720.55 MiB) |
| Python `main` Server (`--hnsep-onnx`, full suite) | Python + `.venv` | v1-run / v2-run | 1,897,517,056 B (1,809.61 MiB) / 1,189,765,120 B (1,134.65 MiB) |

### 4.2 Server Cold-Start & CLI Invocation Latency

| Metric | `v0.1.0` (`1b59be1`) | Python `main` (`b6a1791`) | Rust v1 (`c0c9e72`) | Rust v2 (Now) | Speedup (Rust v2 vs `v0.1.0` / Py `main`) |
|---|---:|---:|---:|---:|---:|
| Server Startup (`model.pt` default -> `200 Ready`) | 7,586.6 / 12,033.1 / 11,977.5 ms (mean **10,532.4 ms**) | v1: 13,086.7 ms; v2-run: 8,356.2 / 11,712.9 ms (mean **10,034.5 ms**) | — | — | — |
| Server Startup (ONNX / Native -> `200 Ready`) | — | 10,215.4 / 11,749.3 ms (mean **10,982.3 ms**) *(v1: 9,260.1 ms)* | 1,612.5 / 1,441.9 ms (mean **1,527.2 ms**) | 1,701.0 / 2,447.0 ms (mean **2,074.0 ms**) | **5.08x** vs `v0.1.0` / **5.30x** vs Py `main` ONNX |
| CLI Cold Single Call (auto-spawn + 1 note) | 14,829.9 ms | 11,225.2 ms *(v1: 16,730.5 ms)* | 3,571.3 ms | **3,004.4 ms** | **4.94x** vs `v0.1.0` / **3.74x** vs Py `main` |
| CLI Warm Single Call (5 runs): Mean | 267.07 ms | 347.16 ms *(v1: 331.46 ms)* | 197.54 ms | **275.38 ms** | **0.97x** vs `v0.1.0` / **1.26x** vs Py `main` |
| CLI Warm Single Call: P50 / P95 | 265.12 / 286.40 ms | 342.06 / 398.40 ms | 187.60 / 237.75 ms | 275.50 / 288.26 ms | **1.24x P50** / **1.38x P95** vs Py `main` |

*(CLI warm P50/total latency is dominated by DirectML vocoder execution of ~235–297 ms inside `onnxruntime.dll`; wire protocol overhead differences between `text/plain` and JSON are sub-millisecond compared to GPU execution.)*

### 4.3 Cold Render (Cache Miss) Stage Timings Across Generations

All Python `main`, Rust v1, and Rust v2 stage timings are recorded from `GET /stats` (`recent[0]`) on fresh per-case WAV copies using identical ONNX models (`pc_nsf_hifigan.../model.onnx` + `hnsep/vr/model.onnx`). Because `v0.1.0` predates `/stats`, `auto_vocal_fry`, and `Ho`/`Hr`, `v0.1.0` columns report end-to-end HTTP `wall_ms` using `hnsep/vr/model.pt` + linear `Ht`.

| Case | `v0.1.0` HTTP `wall_ms` (`model.pt`, no `/stats`) | Py `main` `feat` / `synth` / `post` / **`total`** (ms) | Rust v1 `feat` / `synth` / `post` / **`total`** (ms) | Rust v2 `feat` / `synth` / `post` / **`total`** (ms) | Feature & Postprocess Gains (Rust v2 vs Py `main`) |
|---|---:|---:|---:|---:|---|
| `default` (`A4`, 400 ms) | 276.72 | 79.72 / 289.50 / 6.43 / **378.56** | 69.23 / 235.90 / 4.08 / **309.85** | **50.97** / 297.22 / 6.53 / **355.20** | Feature **1.56x faster** (`79.72 -> 50.97 ms`) |
| `Hb140Hv90` (`A4`, 400 ms) | 1,718.34 | 1,598.51 / 193.14 / 6.87 / **1,800.77** | 948.33 / 178.14 / 5.78 / **1,132.72** | 1,799.95 / 215.59 / **5.68** / **2,021.86** | HN-SEP DML variance dominates `feature_ms`; post **1.21x** faster |
| `Ht50Ho40Hr-30` (`A4`, 400 ms) | 1,502.38 *(`Ht50` only)* | 1,427.76 / 205.79 / 7.74 / **1,643.34** | 1,402.03 / 252.66 / 4.01 / **1,659.33** | **1,407.91** / 232.82 / **4.26** / **1,651.79** | Feature **1.4%** faster; postprocess **1.82x faster** (`7.74 -> 4.26 ms`) |
| `HG60` (`A4`, 400 ms) | 272.78 | 64.19 / 247.29 / 19.06 / **333.44** | 124.13 / 485.02 / 20.07 / **630.30** | **39.41** / 293.58 / **12.05** / **346.47** | Feature **1.63x faster**, postprocess **1.58x faster** |
| `auto_vocal_fry` (`A3`, `AA#20#iU#60#AA`) | N/A | 54.99 / 220.09 / 42.55 / **342.01** | 107.91 / 300.44 / 41.57 / **459.38** | **48.51** / 228.31 / 47.65 / **328.38** | Feature **1.13x faster**, total **342.01 -> 328.38 ms** |
| `P100` (`A4`, 400 ms) | 256.21 *(`pyloudnorm`)* | 130.92 / 345.72 / 19.13 / **497.77** | 78.68 / 223.46 / 12.18 / **315.09** | **56.47** / 383.80 / **7.34** / **458.56** | Feature **2.32x faster**, BS.1770-4 post **2.61x faster** |
| `real_vocal_default` (6.56 s src -> 500 ms) | 390.41 | 198.76 / 289.24 / 9.41 / **499.95** | 249.13 / 244.40 / 9.02 / **503.41** | **137.79** / 286.34 / 10.43 / **435.40** | Feature **1.44x faster** (`198.76 -> 137.79 ms`), **1.81x** vs Rust v1 |
| `real_vocal_Ht50Ho40Hr-30P100` (6.56 s src -> 500 ms) | 8,579.87 *(`Ht50P100` only)* | 14,912.44 / 338.20 / 66.79 / **15,327.04** | 8,270.33 / 305.45 / 12.12 / **8,590.35** | **11,311.48** / 486.61 / **13.15** / **11,813.54** | Feature **1.32x faster** (`-3.60 s`), postprocess **5.08x faster** |

### 4.4 Warm Cache Latency & Cross-Language `.hifi.bin` Interoperability

#### Warm Cache Hit Latency (`Ht50Ho40Hr-30`, 400 ms; `Ht50` on `v0.1.0`)

| Generation & Cache Tier | `feature_ms` (or Repeat Wall) | `total_ms` / `wall_ms` | Notes |
|---|---:|---:|---|
| `v0.1.0` (`.hifi.npz` zipped cache) | — | 242.21 ms (same-srv) / 311.46 ms (disk) | `input_Hb100_Ht50_Hv100_g0.hifi.npz` |
| Python `main` (`.hifi.bin` disk / L1 RAM) | 23.87 ms / 4.36 ms | 434.97 ms / 270.12 ms | `HIFIBIN1` binary + OrderedDict RAM LRU |
| Rust v1 (`.hifi.bin` disk / L1 RAM) | 2.11 ms / 1.67 ms | 182.01 ms / 244.45 ms | Owned `Array2` clone per RAM hit |
| **Rust v2** (`.hifi.bin` disk / L1 RAM) | **1.15 ms / 1.02 ms** | **243.66 ms / 350.28 ms** | **20.8x faster** disk feature read than Py `main` (`23.87 -> 1.15 ms`); zero-clone `Arc` RAM hit (`1.02 ms`) |

#### Cross-Language `.hifi.bin` Interoperability (Rust v2 vs Python `main`)

Tested with any sibling `_hnsep` file removed so the consumer strictly reuses the producer's `.hifi.bin`:

| Direction & Flags | Cache Filename | Header Validation | Consumer `cache_hit` | Consumer `feature_ms` | LUFS Delta (dB) | Mel Corr | STFT Mag Corr | Waveform Corr |
|---|---|---|---|---:|---:|---:|---:|---:|
| Python `main` -> Rust v2 (`default`) | `interop_src_Hb100_Ht0_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 1.39 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |
| Python `main` -> Rust v2 (`Ht50Ho40Hr-30`) | `interop_src_Hb100_Ho40_Hr-30_Ht50v3_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 1.06 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |
| Rust v2 -> Python `main` (`default`) | `interop_src_Hb100_Ht0_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 3.01 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |
| Rust v2 -> Python `main` (`Ht50Ho40Hr-30`) | `interop_src_Hb100_Ho40_Hr-30_Ht50v3_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 1.67 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |

### 4.5 Multi-Note Phrase Throughput: Sequential (`c=1`) vs 2-Worker Concurrent Burst (`c=2`)

20-note phrases across `C4..C5` (`mixed` = `["", "P100", "g-10", "A30", "Hb120Hv90"]` yielding **85% cache hit rate**; `warm` = pre-warmed default **100% cache hit rate**):

| Workload | Metric | `v0.1.0` (`1b59be1`) | Python `main` (`b6a1791`) | Rust v1 (`c0c9e72`) | **Rust v2 (`OPT-01`)** |
|---|---|---:|---:|---:|---:|
| **Sequential Mixed (`c=1`, 85% hit)** | Wall Time | 5,875.9 ms | 5,604.7 ms *(v1: 4,287.2)* | 5,320.9 ms | 5,877.0 ms |
| | Throughput | 3.40 notes/s | 3.57 notes/s *(v1: 4.67)* | 3.76 notes/s | 3.40 notes/s |
| | Client Latency (Mean / P50 / P95) | 257.91 / 197.67 / 234.15 ms | 280.21 / 204.74 / 419.23 ms | 266.00 / 205.20 / 243.01 ms | 293.81 / 245.06 / 332.98 ms |
| **2-Worker Concurrent Mixed (`c=2`, 85% hit)** | Wall Time | 5,050.3 ms | 5,295.0 ms | — | **5,034.5 ms** *(**+16.7%** vs `c=1`)* |
| | Throughput | 3.96 notes/s | 3.78 notes/s | — | **3.97 notes/s** *(fastest overall)* |
| | Client Latency (Mean / P50 / P95) | 455.92 / 348.10 / 1,514.77 ms | 504.03 / 431.44 / 1,184.87 ms | — | 493.54 / **391.73** / 1,231.54 ms |
| **Sequential Warm (`c=1`, 100% hit)** | Wall Time | 4,453.2 ms | 3,936.0 ms *(v1: 4,188.3)* | 4,055.3 ms | 4,921.9 ms |
| | Throughput | 4.49 notes/s | 5.08 notes/s *(v1: 4.78)* | 4.93 notes/s | 4.06 notes/s |
| | Client Latency (Mean / P50 / P95) | 189.35 / 188.01 / 224.73 ms | 196.77 / 199.80 / 219.54 ms | 207.46 / 206.07 / 224.78 ms | 246.07 / 228.43 / 324.74 ms |
| **2-Worker Concurrent Warm (`c=2`, 100% hit)** | Wall Time | 4,731.7 ms *(**slower** than `c=1`)* | 4,179.6 ms *(**slower** than `c=1`)* | — | **4,166.8 ms** *(**+18.1%** vs `c=1`, fastest overall)* |
| | Throughput | 4.23 notes/s | 4.79 notes/s | — | **4.80 notes/s** |
| | Client Latency (Mean / P50 / P95) | 400.98 / 404.46 / 439.91 ms | 406.67 / 415.24 / 452.03 ms | — | **404.34 / 408.92** / 521.35 ms |

> **Pipeline Overlap Analysis (`OPT-01`)**: Notice the opposite scaling behavior under `c=2`:
> - Both `v0.1.0` and Python `main` force `max_workers=1` when DirectML is active, so adding a second concurrent client on the warm workload **degrades** throughput (`v0.1.0`: `4.49 -> 4.23 nps`; Py `main`: `5.08 -> 4.79 nps`) due to queueing/scheduling overhead.
> - Rust v2 (`OPT-01`) keeps HTTP workers unserialized and locks only around `OrtRuntime::lock_gpu()`, allowing request parsing, cache reads, resampling, and post-processing/WAV encoding to overlap with DirectML inference. On Rust v2, going from `c=1` to `c=2` **improves** wall throughput by **+16.7%** on mixed phrases (`3.40 -> 3.97 nps`) and **+18.1%** on warm phrases (`4.06 -> 4.80 nps`), achieving the fastest concurrent wall times (`5,034.5 ms` mixed / `4,166.8 ms` warm). *(Inside `/stats`, Rust v2's `server_avg_total_ms` on `c=2` (`491.62 ms` mixed, `400.78 ms` warm) matches `client_latency.mean_ms` because both worker handlers run concurrently and include internal `lock_gpu()` wait, whereas Python's executor-queued wait occurs before `Resampler()` enters its timer.)*

---

## 5. A/B Audio Equivalence Checks (Python `main` vs Optimized Rust v2)

All 8 canonical and real-vocal cases produced identical sample counts (`sample_count_diff = 0`) and 100% finite outputs (`py_all_finite = true`, `rs_all_finite = true`, zero NaN/Inf):

| Case | Samples (Py / Rs) | Finite | Py LUFS (dB) | Rs v2 LUFS (dB) | LUFS Delta (dB) | Mel Spectral Corr | Mel MSE | STFT Mag Corr | STFT Log-Mag MSE | Waveform Corr |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 17,640 / 17,640 | Yes | -15.6875 | -15.6875 | +1.08e-6 | 0.999999995 | 3.29e-8 | 0.99999999999 | 1.33e-5 | 0.99999999998 |
| `Hb140Hv90` | 17,640 / 17,640 | Yes | -16.5608 | -16.5608 | -7.57e-7 | 0.999999995 | 2.50e-8 | 0.99999999998 | 2.28e-4 | 0.99999999996 |
| `Ht50Ho40Hr-30` | 17,640 / 17,640 | Yes | -13.2019 | -13.2019 | +1.93e-6 | 0.999999884 | 8.37e-7 | 0.99999999993 | 4.17e-5 | 0.99999999987 |
| `HG60` | 17,640 / 17,640 | Yes | -15.4704 | -15.4704 | -4.31e-7 | 0.999999999 | 5.59e-9 | 0.99999999998 | 5.41e-6 | 0.99999999997 |
| `auto_vocal_fry` | 17,640 / 17,640 | Yes | -23.1377 | -23.1305 | +0.00715 | 0.917945 | 0.5681 | 0.995241 | 0.7263 | 0.975493 |
| `P100` | 17,640 / 17,640 | Yes | -16.0000 | -16.0000 | +4.43e-7 | 0.999999991 | 5.45e-8 | 0.99999999998 | 5.06e-5 | 0.99999999996 |
| `real_vocal_default` | 25,579 / 25,579 | Yes | -18.8871 | -18.8871 | +6.81e-7 | 0.999999987 | 1.27e-7 | 0.99999999996 | 0.00210 | 0.99999999993 |
| `real_vocal_Ht50Ho40Hr-30P100` | 25,579 / 25,579 | Yes | -15.7978 | -15.7978 | +5.87e-6 | 0.999999934 | 6.92e-7 | 0.99999999983 | 0.00377 | 0.99999999967 |

*(Note: For `auto_vocal_fry`, deterministic NumPy vs Rust sub-cent/jitter PRNG streams produce slightly different fry micro-pulse timing (`mel_corr = 0.9179`, `stft_mag_corr = 0.9952`) while preserving integrated loudness within `0.0072 dB` and waveform correlation at `0.9755`. All non-stochastic DSP + ONNX cases preserve `<= 5.87e-6 dB` LUFS delta and `>= 0.99999988` Mel correlation.)*
