# HiFiSampler Rust Port (`hifisampler-rs`) — Verification & A/B Benchmark Report

## 1. Environment & Methodology

- **Repository / Branch**: `C:\Project\hifi` (`feat/rust-port`)
- **Models Evaluated**:
  - Vocoder: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx` (`DmlExecutionProvider` with `disable_metacommands=true` via `onnxruntime.dll`)
  - HN-SEP: `hnsep/vr/model.onnx` (and default `config.yaml` `./hnsep/vr/model.pt` noted separately for default Python `hifiserver.py` startup/RSS)
- **Python Baseline**: `C:\Project\hifi\.venv\Scripts\python.exe` (with `rust/crates/hifi_bench/scripts/py_bench_server.py --hnsep-onnx` for apples-to-apples ONNX comparisons without modifying `config.yaml`) and `C:\Project\hifi\hifisampler.exe` (C# CLI client)
- **Rust Port (`--release`)**: `rust/target/release/hifiserver.exe` and `rust/target/release/hifisampler.exe` (built with `opt-level = 3`, `lto = "thin"`, `codegen-units = 1`, `panic = "abort"`, `x86_64-pc-windows-gnu`)
- **Reproducible Harness**: `rust/crates/hifi_bench` (`cargo run --release -p hifi_bench -- --root C:\Project\hifi`)
- **Test Inputs**:
  - Deterministic 1.0 s 44.1 kHz 16-bit PCM harmonic vocal-like test WAV (`bench_harmonic_440.wav`, 220 Hz fundamental + 10 harmonics + low-level breath noise, 44,100 samples)
  - Real UTAU vocal sample copied to isolated temp dir: `TestResults/vocal_fry_demo/isolated_vb_cache/A3__ああいあうえあ.wav` (289,380 samples, 6.56 s @ 44.1 kHz, rendered from active vowel window `offset=1200`, `length=500`, `consonant=80`, `cutoff=-1200`)

---

## 2. Verification Gates Summary

| Gate | Command | Result | Scope / Notes |
|---|---|---|---|
| Formatting | `cargo fmt --all -- --check` | **PASS** (exit 0) | Full Rust workspace (`rust/crates/*`) |
| Linter | `cargo clippy --workspace --all-targets -- -D warnings` | **PASS** (exit 0) | All 8 workspace crates (0 warnings) |
| Rust Release Tests | `cargo test --workspace --release` | **PASS** (110/110) | `hifi_client` (10), `hifi_core` (37), `hifi_dsp` (9), `hifi_effects` (10), `hifi_lufs` (7), `hifi_onnx` (11), `hifi_server` (23), `hifi_bench` (3) |
| Python Unit Baseline | `pytest tests --ignore=tests/e2e -q` | **PASS** (259/259) | Verified in 38.01 s |
| E2E Suite (Rust Target) | `tests/e2e/runner.py --target rust --all` | **PASS** (196/196) | Tier 1 (85) + Tier 2 (85) + Tier 3 (17) + Tier 4 (9) in 120.64 s (100.0%) |

---

## 3. Binary Sizes & Peak Process-Tree Memory RSS

### 3.1 Release Binary Sizes

| Artifact | Size (bytes) | Size (KiB / MiB) | Comparison vs Baseline |
|---|---:|---:|---|
| `rust/target/release/hifisampler.exe` | 644,571 | 629.5 KiB (0.61 MiB) | **8.42x smaller** than C# `hifisampler.exe` |
| `rust/target/release/hifiserver.exe` | 4,917,678 | 4.69 MiB | Single native binary (vs Python `.venv` + PyTorch) |
| `hifisampler.exe` (C# baseline) | 5,425,664 | 5.17 MiB | Baseline CLI binary |

### 3.2 Peak Process-Tree Memory RSS (`PeakWorkingSet64`)

Measured across the entire process tree (including `uv` shim + child `cpython-3.12` interpreter for Python) after serving cold/warm/batch workloads:

| Server Process | Peak RSS (bytes) | Peak RSS (MiB) | Notes |
|---|---:|---:|---|
| Python (`hifiserver.py`, default `model.pt` + 1 probe) | 755,015,680 | 720.04 MiB | Startup + 1 `Ht20` probe render |
| Python (`py_bench_server.py --hnsep-onnx`, full suite) | 1,897,517,056 | 1,809.61 MiB | Full cold/interop/batch workload |
| Rust (`hifiserver.exe`, full suite) | 1,268,973,568 | 1,210.19 MiB | **33.1% lower peak RSS** vs Python ONNX server |

---

## 4. Cold-Start & CLI Invocation Latency

| Metric | Python Baseline | Rust (`--release`) | Speedup |
|---|---:|---:|---:|
| Server Startup (`hifiserver.py` default `model.pt` -> `200 Server Ready`) | 16,301.9 ms / 9,871.6 ms (mean **13,086.7 ms**) | — | — |
| Server Startup (`py_bench_server.py --hnsep-onnx` vs `hifiserver.exe`) | 10,181.8 ms / 8,338.4 ms (mean **9,260.1 ms**) | 1,612.5 ms / 1,441.9 ms (mean **1,527.2 ms**) | **6.06x** (vs Py ONNX) / **8.57x** (vs Py default) |
| Cold CLI Single Call (no server -> auto-spawn + render 1 note) | 16,730.5 ms (C# -> `hifiserver.py`) | **3,571.3 ms** (Rust -> `hifiserver.exe`) | **4.68x** |
| Warm CLI Single Call (5 runs, live server): Mean | 331.46 ms (C# client) | **197.54 ms** (Rust client) | **1.68x** |
| Warm CLI Single Call (5 runs, live server): P50 / P95 | 332.52 ms / 374.77 ms | **187.60 ms / 237.75 ms** | **1.77x P50** / **1.58x P95** |

---

## 5. Cold Render (Cache Miss) Stage Timings

Stage timings recorded from `GET /stats` (`recent[0]`) on fresh per-case WAV copies (guaranteed disk + L1 RAM cache miss). Both engines execute the same `pc_nsf_hifigan.../model.onnx` and `hnsep/vr/model.onnx` models on DirectML (`synthesis_ms` is GPU-bound in `onnxruntime.dll`):

| Case | Python `feature_ms` | Rust `feature_ms` | Python `synthesis_ms` | Rust `synthesis_ms` | Python `post_ms` | Rust `post_ms` | Python `total_ms` | Rust `total_ms` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` (`A4`, 400 ms) | 77.14 | 69.23 | 245.82 | 235.90 | 7.57 | 4.08 | 332.46 | **309.85** |
| `Hb140Hv90` (`A4`, 400 ms) | 1,201.70 | **948.33** | 177.42 | 178.14 | 6.23 | 5.78 | 1,386.37 | **1,132.72** |
| `Ht50Ho40Hr-30` (`A4`, 400 ms) | 1,192.27 | 1,402.03 | 210.27 | 252.66 | 9.32 | **4.01** | 1,413.53 | 1,659.33 |
| `HG60` (`A4`, 400 ms) | 83.57 | 124.13 | 322.28 | 485.02 | 63.05 | **20.07** | 474.72 | 630.30 |
| `auto_vocal_fry` (`A3`, `AA#20#iU#60#AA`, 400 ms) | 112.72 | **107.91** | 284.40 | 300.44 | 56.08 | **41.57** | 484.78 | **459.38** |
| `P100` (`A4`, 400 ms) | 62.55 | 78.68 | 233.45 | 223.46 | 12.03 | 12.18 | 309.58 | 315.09 |
| `real_vocal_default` (`A3`, 500 ms, 6.56 s src) | 188.25 | 249.13 | 256.23 | 244.40 | 9.46 | 9.02 | 455.81 | 503.41 |
| `real_vocal_Ht50Ho40Hr-30P100` (`C4`, 500 ms, 6.56 s src) | 8,337.06 | **8,270.33** | 336.92 | **305.45** | 24.07 | **12.12** | 8,700.88 | **8,590.35** |

---

## 6. Warm Render & Cross-Language `.hifi.bin` Interoperability

### 6.1 Warm Cache Hit Latency (`Ht50Ho40Hr-30`, 400 ms)

| Mode | Python `feature_ms` | Rust `feature_ms` | Python `total_ms` | Rust `total_ms` | Speedup (`feature` / `total`) |
|---|---:|---:|---:|---:|---:|
| Disk `.hifi.bin` hit (fresh server) | 38.00 ms | **2.11 ms** | 253.04 ms | **182.01 ms** | **18.0x** / **1.39x** |
| L1 RAM cache hit | 2.71 ms | **1.67 ms** | 258.77 ms | **244.45 ms** | **1.62x** / **1.06x** |

### 6.2 Cross-Language `.hifi.bin` Interoperability (`HIFIBIN1` Header + Equivalence)

Tested bidirectional feature cache reuse with any sibling `_hnsep` file removed so the consumer relies strictly on the producer's `.hifi.bin`:

| Direction & Flags | Cache Filename | Header Check | Consumer `cache_hit` | Consumer `feature_ms` | LUFS Delta (dB) | Mel Corr | STFT Mag Corr | Waveform Corr |
|---|---|---|---|---:|---:|---:|---:|---:|
| Python -> Rust (`default`) | `interop_src_Hb100_Ht0_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 1.12 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |
| Python -> Rust (`Ht50Ho40Hr-30`) | `interop_src_Hb100_Ho40_Hr-30_Ht50v3_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 0.82 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |
| Rust -> Python (`default`) | `interop_src_Hb100_Ht0_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 2.01 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |
| Rust -> Python (`Ht50Ho40Hr-30`) | `interop_src_Hb100_Ho40_Hr-30_Ht50v3_Hv100_g0.hifi.bin` | `HIFIBIN1` v1, 128x344, 176,156 B, finite | `true` | 1.50 ms | **0.000000** | **1.00000000** | **1.00000000** | **1.00000000** |

---

## 7. Multi-Note Batch Throughput (20-Note Phrase)

| Workload | Metric | Python (`--hnsep-onnx`) | Rust (`--release`) |
|---|---|---:|---:|
| **Mixed Cold+Warm 20-Note Phrase** (85% cache hit rate) | Wall Time | 4,287.2 ms | 5,320.9 ms |
| | Throughput | 4.67 notes/s | 3.76 notes/s |
| | Client Latency (Mean / P50 / P95) | 214.32 / 183.63 / 210.06 ms | 266.00 / 205.20 / 243.01 ms |
| | Server Batch Avg / P95 (`recent[0..20]`) | 210.37 / 206.58 ms | 263.80 / 240.77 ms |
| **Warm-Cache 20-Note Scale Burst** (100% cache hit rate) | Wall Time | 4,188.3 ms | **4,055.3 ms** |
| | Throughput | 4.78 notes/s | **4.93 notes/s** |
| | Client Latency (Mean / P50 / P95) | 209.29 / 209.40 / 247.77 ms | **207.46 / 206.07 / 224.78 ms** |
| | Server Batch Avg / P95 (`recent[0..20]`) | 204.49 / 243.45 ms | **200.64 / 222.51 ms** |

*(Note: In the mixed 20-note phrase, nearest-rank P95 index `round(0.95 * 19) = 18` reflects the 19th-fastest note while `max_ms` — 806.3 ms Py / 1,381.5 ms Rs from the initial cold HN-SEP render — shifts the mean slightly above P95.)*

---

## 8. A/B Audio Equivalence Checks (Python vs Rust)

All canonical and real-vocal cases produced identical sample counts (`sample_count_diff = 0`) and 100% finite outputs (`py_all_finite = true`, `rs_all_finite = true`, zero NaN/Inf):

| Case | Samples (Py / Rs) | Finite | Py LUFS (dB) | Rs LUFS (dB) | LUFS Delta (dB) | Mel Spectral Corr | Mel MSE | STFT Mag Corr | STFT Log-Mag MSE | Waveform Corr |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 17,640 / 17,640 | Yes | -15.6875 | -15.6875 | +5.66e-7 | 0.99999999 | 3.69e-8 | 0.99999999999 | 1.63e-5 | 0.99999999998 |
| `Hb140Hv90` | 17,640 / 17,640 | Yes | -16.5608 | -16.5608 | +2.42e-7 | 0.99999999 | 3.30e-8 | 0.99999999998 | 5.65e-5 | 0.99999999996 |
| `Ht50Ho40Hr-30` | 17,640 / 17,640 | Yes | -13.2019 | -13.2019 | +1.37e-6 | 0.99999988 | 8.60e-7 | 0.99999999994 | 5.75e-5 | 0.99999999987 |
| `HG60` | 17,640 / 17,640 | Yes | -15.4704 | -15.4704 | -9.09e-8 | 0.99999999 | 5.17e-9 | 0.99999999999 | 4.16e-6 | 0.99999999997 |
| `auto_vocal_fry` | 17,640 / 17,640 | Yes | -23.1377 | -23.1305 | +0.00715 | 0.917945 | 0.5681 | 0.995241 | 0.7264 | 0.975493 |
| `P100` | 17,640 / 17,640 | Yes | -16.0000 | -16.0000 | -2.73e-7 | 0.99999999 | 4.88e-8 | 0.99999999998 | 7.42e-5 | 0.99999999996 |
| `real_vocal_default` | 25,579 / 25,579 | Yes | -18.8871 | -18.8871 | -2.44e-7 | 0.99999999 | 1.11e-7 | 0.99999999996 | 0.00188 | 0.99999999992 |
| `real_vocal_Ht50Ho40Hr-30P100` | 25,579 / 25,579 | Yes | -15.7978 | -15.7978 | +5.98e-6 | 0.99999995 | 5.01e-7 | 0.99999999983 | 0.00374 | 0.99999999968 |

*(Note: For `auto_vocal_fry`, subtle sub-cent/jitter RNG stream differences between NumPy and Rust explain `mel_corr = 0.9179` and `stft_mag_corr = 0.9952` while keeping integrated loudness within `0.007 dB` and waveform correlation at `0.9755`. All non-stochastic DSP + ONNX cases achieve `>= 0.9999998` Mel correlation and `<= 6e-6 dB` LUFS delta.)*
