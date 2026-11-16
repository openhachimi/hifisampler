# Project: hifisampler Rust Port (`hifisampler-rs`)

## Architecture
- **Workspace Layout**: Cargo workspace located at `rust/` containing clean-room crates/modules:
  - `rust/hifi_core`: Audio I/O (WAV reading/writing), HIFIBIN1 binary feature cache with mtime/size validation, atomic file swapping, legacy npz fallback.
  - `rust/hifi_lufs`: ITU-R BS.1770-4 LUFS loudness measurement and envelope-aware normalization.
  - `rust/hifi_dsp`: Signal processing algorithms: HNSEP STFT complex framing, Akima pitch interpolation, Ht v3 tension engine (continuous-time Fant LF glottal model, adaptive F0, Itakura-Saito Rd0, formant plateau, Hilbert cepstrum minimum-phase filtering, aperiodic noise coupling), Klatt resonators (Ho openness, Hr singer's formant), Vocal Fry detection & 3-band post-shaper, Growl Butterworth split & square LFO.
  - `rust/hifi_onnx`: Pure ONNX inference engine via `ort` crate for neural vocoder and HNSEP with cross-platform EP fallback (DirectML/CUDA/ROCm/CoreML/CPU).
  - `rust/hifi_server`: Axum/Tokio HTTP server with `--managed` lifecycle supervision, idle watchdog, file locking, locator beacon, `/health`, and `/stats` telemetry.
  - `rust/hifi_client`: Lightweight CLI client parsing 13 UTAU arguments, Wine path translation, 300ms optimistic POST, detached process spawning without inherited handles.
  - `rust/hifi_bench`: Benchmarking harness and audio equivalence validator.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Branch & Toolchain Setup | Create `feat/rust-port` branch and set up Rust toolchain in `rust/` | M1 | Survey 3 / R1 |
| 2 | Audio I/O | 44.1 kHz 16-bit PCM / float32 WAV read/write with mono/stereo handling | M1 | Survey 1 / R1 |
| 3 | HIFIBIN1 Feature Cache | 28-byte packed header (`<8sIIId`), raw float32 payload, mtime invalidation, atomic tmp file swap, legacy npz reading | M1 | Survey 1 / R1 |
| 4 | BS.1770-4 LUFS Normalization | K-weighting SOS filters, 400ms sliding block, dual gating (-70 LUFS / -10 LU), envelope-aware gain protection | M2 | Survey 1 / R1 |
| 5 | Pitch & Expression (F0 / Akima) | UTAU base64/RLE pitch curve parsing, Akima 1D interpolation onto Mel grid | M2 | Survey 1 / R1 |
| 6 | Ht v3 Glottal Tension Engine | Fant LF glottal solver, adaptive F0, Itakura-Saito Rd0, formant plateau protection, Hilbert cepstral filter, aperiodic noise coupling | M2 | Survey 1 / R1 |
| 7 | Ho & Hr Klatt Resonators | 2nd-order Klatt resonators (Ho at source F0, Hr at 3200 Hz singer's formant) | M2 | Survey 1 / R1 |
| 8 | Vocal Fry & Growl | O(1) early exit, 2-pass stationary basin detector, 3-band post-shaper; 400Hz Butterworth split, 80Hz square LFO | M2 | Survey 1 / R1 |
| 9 | Neural Vocoder ONNX Inference | Model `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx` inference (`mel` [1, N, 128], `f0` [1, N] -> `waveform` [1, N*512]) | M3 | Survey 1 / R2 |
| 10 | HNSEP ONNX Separation | Centered STFT, complex 2-channel tensor, unvoiced/voiced separation, linear bypass when expressions are zero | M3 | Survey 1 / R1 |
| 11 | Cross-Platform EP Fallback | DirectML (with `disable_metacommands="true"`, `max_workers=1`) / CUDA / ROCm / CoreML with automatic CPU fallback | M3 | Survey 2 / R2 |
| 12 | CLI Client (`hifisampler`) | 13 UTAU positional arguments, Wine path translation, 300ms optimistic POST, detached child process launch (`bInheritHandles=false`) | M4 | Survey 2 / R1 |
| 13 | HTTP Server & Routes | `GET /` (200/503), `POST /` (job queue), anti-CSRF browser check (`Origin`/`Sec-Fetch-Site` rejection) | M4 | Survey 2 / R1 |
| 14 | `--managed` Lifecycle & Beacon | 600s idle watchdog auto-shutdown, single-instance file lock, atomic beacon `~/.hifisampler/server.json` | M4 | Survey 2 / R1 |
| 15 | Telemetry Endpoints | `/stats` (200-item ring buffer, nearest-rank P95 latency, cache hit rates) and `/health` | M4 | Survey 2 / R1 |
| 16 | Audio Equivalence Verification | Tier 1 DSP, Tier 2 ONNX, Tier 3 End-to-End A/B equivalence against Python baseline across 10 canonical note configs | M5 | Survey 3 / R3 |
| 17 | Latency & Throughput Benchmark | Side-by-side performance benchmark comparing Python baseline vs Rust port on standard synthesis inputs | M5 | Survey 3 / R3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Branch & Core Cache / Audio I/O | Branch `feat/rust-port`, Rust toolchain setup, workspace layout, WAV I/O, HIFIBIN1 binary cache | none | IN_PROGRESS (sub_orch_m1: 0570856a-f638-4d27-97e3-c087bde86e38) |
| M2 | DSP & Audio Normalization | BS.1770-4 LUFS, Akima pitch, Ht v3 tension, Ho/Hr resonators, Vocal Fry, Growl | M1 | PLANNED |
| M3 | Pure ONNX Inference & EPs | Vocoder ONNX, HNSEP ONNX, DirectML/CUDA/ROCm/CoreML/CPU fallback hierarchy | M1 | PLANNED |
| M4 | Server, Client & Managed Lifecycle | CLI client (13 args, detached spawn), HTTP server, `--managed` watchdog, beacon, `/stats`, `/health` | M2, M3 | PLANNED |
| M5 | Final E2E Audio Equivalence & Benchmark | Full synthesis pipeline verification against 10 note configs, A/B audio fidelity equivalence, latency/throughput benchmark | M4 | PLANNED |

## Interface Contracts
### `hifi_core` ↔ `hifi_dsp` / `hifi_onnx`
- `HifiBinHeader`: 28 bytes packed (`magic: [u8; 8]`, `version: u32`, `rows: u32`, `cols: u32`, `scale: f64`).
- `HifiBinCache`: `load_cache(wav_path: &Path) -> Result<Option<Array2<f32>>, CacheError>`, `save_cache(wav_path: &Path, features: ArrayView2<f32>) -> Result<(), CacheError>`.
- Audio Buffer: `AudioBuffer { samples: Vec<f32>, sample_rate: u32 }`.

### `hifi_dsp` ↔ `hifi_onnx`
- Resampler Parameters: `ResampleParams { in_file: PathBuf, out_file: PathBuf, pitch: String, velocity: f32, flags: ResampleFlags, offset: f32, length_req: f32, fixed_len: f32, end_blank: f32, volume: f32, modulation: f32, tempo: f32, pitchbend: Vec<i16> }`.
- Acoustic Features: `AcousticFeatures { mel: Array2<f32>, f0: Array1<f32> }`.

### `hifi_onnx` ↔ `hifi_server`
- Vocoder Model: `VocoderSession::synthesize(mel: ArrayView2<f32>, f0: ArrayView1<f32>) -> Result<Vec<f32>, OrtError>`.
- Execution Provider Config: `ExecutionProviderPreference { device: DevicePreference, allow_fallback: bool }`.

### `hifi_server` ↔ Client / Telemetry
- Routes: `GET /` -> 200 "Server Ready" / 503 "Server Initializing".
- `POST /` -> payload `{"args": [arg0, arg1, ...]}` -> 200 "Success: '{stem}' -> '{out}'" / 400 / 404 / 500.
- `GET /health` -> `{"ready": bool, "active_jobs": u32}`.
- `GET /stats` -> `StatsSnapshot` JSON matching Python `backend/stats.py` schema.

## Code Layout
```
c:\Project\hifi\
├── rust/
│   ├── Cargo.toml
│   ├── crates/
│   │   ├── hifi_core/      # Audio I/O, HIFIBIN1 cache, types
│   │   ├── hifi_lufs/      # BS.1770-4 LUFS & envelope gain
│   │   ├── hifi_dsp/       # HNSEP, F0, Ht v3, Ho, Hr, Vocal Fry, Growl
│   │   ├── hifi_onnx/      # Pure ONNX Runtime engine & EP fallback
│   │   ├── hifi_server/    # Axum HTTP server & managed lifecycle
│   │   ├── hifi_client/    # hifisampler CLI client binary
│   │   └── hifi_bench/     # Benchmarking & A/B equivalence suite
├── tests/                  # Existing Python test suite & reference assets
├── util/vocal_fry (1).py   # [RESTRICTED: NEVER TOUCH OR COMMIT]
└── PROJECT.md              # Project architecture & status index
```
