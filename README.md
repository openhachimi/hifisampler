# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

A neural UTAU / OpenUtau resampler based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders) for virtual singers, supporting Windows, Linux, and macOS with NVIDIA CUDA, AMD ROCm, and ONNX Runtime acceleration.

**For Jinriki please use our [Hachimisampler](https://github.com/openhachimi/hachimisampler).**

## Why is it called hifisampler?

Hifisampler was modified from [straycatresampler](https://github.com/UtaUtaUtau/straycat), replacing the original WORLD vocoder with pc-nsf-hifigan.

## What makes pc-nsf-hifigan different from traditional vocoders?

pc-nsf-hifigan employs neural networks to upsample input features, offering clearer audio quality than traditional vocoders. It is an improvement over standard nsf-hifigan, supporting F0 inputs that do not match the original Mel spectrogram, making it suitable for UTAU / OpenUtau resampling.

## How to use?

### 1. Integrated Environment Package (Recommended on Windows)

1. Download the latest [release](https://github.com/openhachimi/hifisampler/releases) package, extract it, and run `start.bat` once to initialize the rendering service (this automatically registers a global locator beacon at `~/.hifisampler/server.json`).
2. Add the client to OpenUtau's `Resamplers` folder:
   - **Direct Copy (Recommended, supports server auto-start)**: Once `start.bat` or `uv run hifiserver.py` has been run at least once, you can directly copy `hifisampler.exe` and `hifisampler.yaml` into `OpenUtau\Resamplers\` without needing `mklink`!
   - **Or Symbolic Link**:
     ```cmd
     mklink "C:\[OpenUtau Path]\Resamplers\hifisampler.exe" "C:\[Project Path]\hifisampler.exe"
     ```
3. Set the UTAU / OpenUtau resampler to `hifisampler.exe`.

### 2. Linux / macOS & OpenUtau Setup

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and extract the source code and model files into the project folder.
2. Run `./start.sh` once (or `uv run hifiserver.py --register-only`) to initialize `.venv` and register the `~/.hifisampler/server.json` locator beacon.
3. Connect to OpenUtau (choose either method):
   - **Method A (OpenUtau Built-in Hifisampler)**: Start `./start.sh` and select the built-in `hifisampler` resampler in OpenUtau.
   - **Method B (External Resampler with Auto-Start & Expression Manifest)**: Copy `hifisampler.sh` (or the native `hifisampler-linux-x64` binary) and `hifisampler.yaml` into `~/.local/share/OpenUtau/Resamplers/`:
     ```bash
     cp hifisampler.sh ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     cp hifisampler.yaml ~/.local/share/OpenUtau/Resamplers/hifisampler.sh.yaml
     chmod +x ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     ```
     *Note: On Linux, if the server is not running when OpenUtau invokes `hifisampler.sh`, it will first attempt to launch a visible GUI terminal (`ptyxis`, `kgx`, `gnome-terminal`, `konsole`, `kitty`, `alacritty`, `wezterm`, `foot`, `xfce4-terminal`, etc.) detached via `setsid`, and automatically fall back to a background daemon logging to `logs/hifiserver.log` if no GUI terminal is found.*

### 3. AMD ROCm / NVIDIA CUDA / FP16 Configuration

1. **AMD ROCm on Linux (Recommended: Native PyTorch ROCm + `.ckpt`/`.pt` models)**:
   - Install ROCm PyTorch inside the project environment:
     ```bash
     uv pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch
     ```
   - Configure `config.yaml` to use the PyTorch checkpoints and optional FP16 inference (`MIOPEN_FIND_MODE=FAST` and startup kernel warmup are enabled automatically, while STFT, complex masks, and NSF phase accumulation stay in FP32 to preserve pitch accuracy):
     ```yaml
     model:
       vocoder_path: "./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"
       model_type: "ckpt"
       hnsep_model_path: "./hnsep/vr/model.pt"
     performance:
       max_workers: 2
       device: "auto"      # "auto", "cuda", "rocm", "mps", or "cpu"
       fp16: true          # Enable FP16 inference for PyTorch GPU models
       idle_timeout: 0     # Auto-shutdown server after N seconds of inactivity (0 = disabled)
     ```
2. **NVIDIA CUDA / CPU**:
   - `pyproject.toml` defaults to `pytorch-cu129`. For CPU-only installations, switch `[tool.uv.sources]` to `pytorch-cpu`.

## Implemented Flags

Place `hifisampler.yaml` alongside the client in OpenUtau's `Resamplers/` folder (e.g. `hifisampler.yaml` or `hifisampler.sh.yaml`) to automatically register all flags in OpenUtau:

- **g:** Adjust gender/formants.
  - Range: `-600` to `600` | Default: `0`
- **Hb:** Adjust breath/noise.
  - Range: `0` to `500` | Default: `100`
- **Hv:** Adjust voice/harmonic.
  - Range: `0` to `150` | Default: `100`
- **HG:** Vocal fry/growl.
  - Range: `0` to `100` | Default: `0`
- **P:** Normalize loudness at the note level, targeting -16 LUFS. Enable this by setting `wave_norm` to `true` in `config.yaml`.
  - Range: `0` to `100` | Default: `100`
- **t:** Shift the pitch by a specific amount, in cents (1 cent = 1/100 of a semitone).
  - Range: `-1200` to `1200` | Default: `0`
- **Ht:** Adjust tension.
  - Range: `-100` to `100` | Default: `0`
- **A:** Modulate amplitude based on pitch variations for natural vibrato.
  - Range: `-100` to `100` | Default: `0`
- **G:** Force regeneration of feature cache (overwrites existing cache on disk).
  - No value needed
- **He:** Enable Mel spectrum loop mode for long notes.
  - No value needed

_Note: The flags `B` and `V` were renamed to `Hb` and `Hv` respectively because they conflict with other UTAU flags but have different definitions._

## Acknowledgments

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
