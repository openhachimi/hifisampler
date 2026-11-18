# Melos

[中文文档](README_zh_cn.md) | [English Document](README.md) | [Python Branch (hifisampler / `main`)](https://github.com/openhachimi/hifisampler/tree/main)

A pure-Rust neural UTAU / OpenUtau singing synthesis engine (`melos` branch) built on [pc-nsf-hifigan](https://github.com/openvpi/vocoders) and HN-SEP, supporting **full-phrase continuous Mel + F0 synthesis directly in the native OpenUtau client**. Runs natively across Windows, Linux, and macOS with support for NVIDIA, AMD, and Intel GPUs as well as CPU.

**For Jinriki VOCALOID / OtoMAD use, please check out our [Hachimisampler](https://github.com/openhachimi/hachimisampler).**

---

## Why Melos?

Traditional UTAU / OpenUtau resamplers are constrained to per-note synthesis, producing isolated `.wav` clips and handing them to a wavtool for linear waveform crossfading:
- Running the neural vocoder independently on each note initializes fresh harmonic phases per note, often causing `-2.4` to `-3.2 dB` phase-cancellation dips, comb filtering, or "double-voice" blur across legato VCV transitions;
- A 20-note phrase invokes the GPU vocoder 20 separate times, paying repeated kernel launch and padding overhead.

**Melos** is a full-stack Rust rewrite of the inference server and clients, bringing continuous phrase-level Mel synthesis directly to the **native OpenUtau client (`CLASSIC` renderer) and classic UTAU**:

| Capability | Classic Single-Note Splicing (`convergence`, etc.) | Melos Phrase Mode (`melos` + `melos-wavtool`) |
| :--- | :--- | :--- |
| **Client Support** | Native OpenUtau / UTAU client | **Native OpenUtau / UTAU client** |
| **Splicing Domain** | Independent per-note WAVs + waveform crossfade | **Power-normalized continuous Mel + continuous MIDI F0 splicing** |
| **GPU Vocoder Invocations** | $N$ calls for $N$ notes | **1 call per phrase** (with 8-frame front/tail context padding & exact crop) |
| **Legato / VCV Transitions** | Independent phase per note; prone to overlap dips | **Continuous NSF harmonic excitation phase & smooth formant morphing** |
| **Phrase Render Speed** | 1.0x baseline | **~1.5x–2.3x faster wall time** (**2.0x–2.7x** less GPU vocoder time) |
| **Cache Backfill Safety** | Per-note `.wav` | Automatically backfills real per-note `.wav` caches for seamless `convergence` fallback |

### Architecture Highlights

1. **Pure-Rust Native Stack ([`rust/`](rust/))**:
   - Zero Python or PyTorch runtime required in production: `hifiserver` (~`4.3 MiB`), `melos` (`553 KiB`), and `melos-wavtool` (`517 KiB`) compile to standalone native binaries with sub-millisecond CLI startup.
   - Native ONNX Runtime GPU/CPU execution (**DirectML** out of the box on Windows for AMD / Intel / NVIDIA discrete & integrated GPUs, plus **CUDA**, **CoreML**, and multi-threaded **CPU** fallback).
   - `HIFIBIN1` binary feature cache (`< 1 ms` load, backward-compatible with legacy `.hifi.npz`), concurrent HTTP + DSP worker overlap, and managed server lifecycle (`--managed`, `--idle-timeout`, automatic server discovery & on-demand launch).
2. **Expressive Acoustic DSP**:
   - Physical Glottal Tension (`Ht`), Klatt Vocal Tract Openness (`Ho`), 3.2 kHz Singer's Formant (`Hr`), automatic glottal vocal fry on deep pitch scoops (`auto_vocal_fry`), Growl (`HG`), ITU-R BS.1770-4 loudness normalization (`P`), pitch-slope vibrato amplitude modulation (`A`), and independent HN-SEP breath/voicing control (`Hb` / `Hv`).
3. **4-Generation Evolution & Real-Model Benchmarks**:
   - Complete cross-generation latency, throughput, audio equivalence, and phrase continuity benchmarks are documented in [`rust/BENCHMARK_REPORT.md`](rust/BENCHMARK_REPORT.md).

---

## How to Use?

### Method 1: All-in-One (AIO) Package (Recommended on Windows — Ready Out of the Box)

No Rust toolchain, Python installation, or manual dependency/model setup required:

1. Download the Melos All-in-One package from [Releases](https://github.com/openhachimi/hifisampler/releases) (e.g. `melos-v*-win64-dml.zip` — universal Windows archive bundling DirectML GPU acceleration for AMD / Intel / NVIDIA discrete & integrated GPUs as well as CPU, plus pre-packaged ONNX models) and extract it.
2. **Double-click `start.bat` once on first use** (registers the server location and starts the background inference service).
   - Once registered, you **do not need to open `start.bat` manually** in daily OpenUtau use — `melos.exe` automatically starts `hifiserver.exe` in the background on demand whenever rendering starts.
3. **Connect to the native OpenUtau client (using the default `CLASSIC` renderer)**:
   - **Full-Phrase Continuous Mel Mode (Recommended)**:
     1. Copy `melos.exe` and `melos.yaml` into OpenUtau's `Resamplers` folder.
     2. Copy `melos-wavtool.exe` into OpenUtau's `Wavtools` folder *(Tip: `melos.exe` includes automatic Resampler/Wavtool dual-role detection, so copying `melos.exe` into `Wavtools` works too).*
     3. In your OpenUtau track header, set **Resampler** = `melos.exe` and **Wavtool** = `melos-wavtool.exe`.
   - **Single-Note Compatibility Mode (Optional)**: If you prefer OpenUtau's built-in `convergence` or `simple` wavtools, copy only `melos.exe` + `melos.yaml` into `Resamplers`; `melos.exe` automatically detects non-Melos wavtools and falls back cleanly to single-note WAV synthesis.
4. In OpenUtau Piano Roll, click **`⚙` (Configure Expressions) -> "Add all expressions suggested by renderers" -> "Apply"** to import all supported flags.

---

### Method 2: Building from Source (Windows / Linux / macOS)

1. Install a standard Rust toolchain (`rustup`, `edition = 2021`) and build the release binaries:
   ```bash
   cd rust
   cargo build --release --bins
   ```
   Binaries are emitted to `rust/target/release/`:
   - `hifiserver` (`/ .exe`): HTTP + ONNX acoustic inference server
   - `melos` (`/ .exe`): Phrase-aware resampler CLI client with dual-role auto-detection
   - `melos-wavtool` (`/ .exe`): Continuous phrase Mel + F0 wavtool CLI client
   - `hifi_bench` (`/ .exe`): Latency, throughput, audio equivalence, and phrase continuity benchmark suite
2. Place the ONNX Runtime shared library (`onnxruntime.dll` with DirectML support on Windows, or configure `ORT_DYLIB_PATH`) and model weights in the engine root folder:
   - Vocoder: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx` (upstream: [openvpi/vocoders Releases](https://github.com/openvpi/vocoders/releases))
   - Harmonic-Noise Separator (optional; required only for `Hb` / `Hv` / `Ht` / `Ho` / `Hr`): `hnsep/vr/model.onnx` (upstream weights: [yxlllc/vocal-remover `hnsep_240512`](https://github.com/yxlllc/vocal-remover/releases/tag/hnsep_240512), exportable via `hnsep/convert_hnsep_to_onnx.py`; setting `hnsep_model_path: ""` in `config.yaml` runs the server without HN-SEP)
3. Copy `config.default.yaml` to `config.yaml` (`model_type: "onnx"`), register the server once, and place the client binaries in OpenUtau (`~/.local/share/OpenUtau/Resamplers/` and `Wavtools/` on Linux / macOS alongside [`rust/melos.yaml`](rust/melos.yaml)):
   ```powershell
   .\hifiserver.exe --register-only
   ```

---

## Hardware Acceleration & `config.yaml` Options

| Setting | Default | Description |
| :--- | :--- | :--- |
| `performance.device` | `"auto"` | Execution provider preference: `"auto"` (DirectML on Windows, CUDA on Linux, CoreML on macOS, with automatic CPU fallback), `"dml"`, `"cuda"`, `"coreml"`, or `"cpu"` |
| `performance.max_workers` | `2` | Worker concurrency (`2`–`4` recommended; HTTP/DSP stages pipeline-overlap with GPU inference) |
| `performance.idle_timeout` | `0` | Auto-shutdown timer in seconds (`0` = stay resident; `--managed` auto-started servers default to `600`s when `0` to free GPU VRAM) |
| `processing.wave_norm` | `true` | Enables ITU-R BS.1770-4 (-16 LUFS) loudness normalization used with the `P` flag |
| `processing.auto_vocal_fry` | `true` | Automatically synthesizes natural glottal vocal fry on deep pitch scoops/releases (yields when `HG` growl is active) |
| `processing.loop_mode` | `true` | Enables mirrored Mel frame looping on stretched notes (can also be toggled per note via `He` when `false`) |
| `processing.peak_limit` | `1.0` | Final waveform peak limiter ceiling (`0.0`–`1.0`) |

### Telemetry & Health Endpoints

`hifiserver` listens on `http://127.0.0.1:8572` by default (configurable via `--port` or `config.yaml`):
- `GET /health`: Readiness state, active in-flight job count, and server uptime
- `GET /stats`: Request counts, feature & HN-SEP cache hit rates, P95 latency, and per-stage timing breakdown (feature / synthesis / postprocess)

---

## Implemented Flags

Import `melos.yaml` in the native OpenUtau client to enable all supported expression flags:

- **g (`gen`):** Adjust gender / formants (positive = deeper/masculine, negative = brighter/feminine).
  - Range: `-100` to `100` | Default: `0`
- **Hb:** Breathiness / aperiodic noise component percentage.
  - Range: `0` to `500` | Default: `100`
- **Hv:** Voicing / harmonic component percentage.
  - Range: `0` to `150` | Default: `100`
- **Ht:** Glottal vocal tension (spectral tilt & aperiodic coupling).
  - Range: `-100` to `100` | Default: `0`
- **Ho:** Vocal tract openness (positive = open/bright upper formants, negative = covered/warm tone).
  - Range: `-100` to `100` | Default: `0`
- **Hr:** Singer's formant resonance around 3.2 kHz (positive = added projection/ring, negative = softer 3 kHz glare).
  - Range: `-100` to `100` | Default: `0`
- **HG:** Growl / distortion effect.
  - Range: `0` to `100` | Default: `0`
- **P (`norm`):** Note-level loudness normalization strength targeting -16 LUFS (requires `wave_norm: true`).
  - Range: `0` to `100` | OpenUtau expression default: `86` (CLI fallback default: `100`)
- **t:** Fine pitch shift in units of 10 cents (`t+10` = +1 semitone).
  - Range: `-120` to `120` | Default: `0`
- **A:** Pitch-slope amplitude modulation for natural vibrato & portamento dynamics.
  - Range: `-100` to `100` | Default: `0`
- **G:** Force regenerate feature cache for the current note.
  - Toggle flag (no value needed)
- **He:** Enable mirrored Mel loop mode for the current note (when `loop_mode: false` in `config.yaml`).
  - Toggle flag (no value needed)

_Note: `B` and `V` were named `Hb` and `Hv` to avoid collisions with standard UTAU flags that carry different semantics._

---

## Distribution & Model License Terms

> **All-in-One (AIO) Package Distribution Policy (Non-Commercial Only):**
> While Melos's own source code and standalone binaries (without bundled weights) are open-source under Apache-2.0, **any All-in-One archive bundling pretrained model weights is strictly restricted to non-commercial use only**. Selling, charging redistribution fees, or bundling the AIO package into commercial software or services is prohibited, and redistributors must comply with all component licenses below:

1. **Melos Engine Source Code & Binaries — [Apache License 2.0](LICENSE)**
   - All original source code under this repository (including the `rust/` workspace) is licensed under Apache-2.0.
   - Network definition and ONNX export helper scripts under `hnsep/` (`nets.py`, `layers.py`, `convert_hnsep_to_onnx.py`) originate from [`tsurumeso/vocal-remover`](https://github.com/tsurumeso/vocal-remover) and [`yxlllc/vocal-remover`](https://github.com/yxlllc/vocal-remover) under the **[MIT License](hnsep/LICENSE)** (`Copyright (c) 2019 tsurumeso`).
2. **`pc-nsf-hifigan` Vocoder Weights — [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
   - Provided by the [OpenVPI Team](https://github.com/openvpi/vocoders) (DiffSinger Community Vocoder Project; AIO bundles the original official OpenVPI ONNX weights).
   - Any redistribution containing these weights **must**:
     1. Never be sold or distributed for any fee (`NonCommercial`);
     2. Include a copy of or link to the CC BY-NC-SA 4.0 license and share derivatives under the same terms (`ShareAlike`);
     3. Preserve `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/NOTICE.txt`, `NOTICE.zh-CN.txt`, and `STATEMENTS.txt` intact with the complete voice-contributor and dataset acknowledgement list;
     4. Document any modifications if weights are ever altered (the AIO ships the original OpenVPI weights with no parameter changes).
3. **`hnsep` Separation Weights — Copyright [`yxlllc`](https://github.com/yxlllc) (No Explicit Weight License; Non-Commercial Bundle Only)**
   - Originate from [`yxlllc/vocal-remover` Release `hnsep_240512`](https://github.com/yxlllc/vocal-remover/releases/tag/hnsep_240512) (`hnsep/vr/model.onnx` is a direct graph export of `vr/model.pt` via `convert_hnsep_to_onnx.py` without altering trained parameters).
   - Note: While the upstream code repository is MIT-licensed, the `hnsep_240512.zip` archive does not carry an explicit separate license covering pretrained weights or training datasets. For strict redistribution or commercial-compliance builds, do **not** bundle `hnsep` weights — set `hnsep_model_path: ""` in `config.yaml` to run cleanly under pure Apache-2.0 / CC terms without HN-SEP, or have end users fetch the weights directly from upstream.

---

## Acknowledgments

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for pc-nsf-hifigan
- [yxlllc](https://github.com/yxlllc) & [tsurumeso](https://github.com/tsurumeso) for vocal-remover / HN-SEP
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
