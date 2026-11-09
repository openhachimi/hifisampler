# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

A neural UTAU / OpenUtau resampler based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders) for virtual singers. Runs on Windows, Linux, and macOS with support for NVIDIA, AMD, and Intel GPUs as well as CPU.

**For Jinriki VOCALOID / OtoMAD use, please check out our [Hachimisampler](https://github.com/openhachimi/hachimisampler).**

## Why is it called hifisampler?

hifisampler was modified from [straycatresampler](https://github.com/UtaUtaUtau/straycat), replacing the traditional WORLD vocoder with the neural vocoder pc-nsf-hifigan.

## What makes pc-nsf-hifigan different from traditional vocoders?

pc-nsf-hifigan uses neural networks to upsample acoustic features, producing clearer and more natural audio quality than traditional algorithmic vocoders. It also supports F0 (pitch) inputs that differ from the original sample's Mel spectrogram, making it ideal for UTAU / OpenUtau pitch shifting and resampling.

---

## How to Use?

Choose any of the installation methods below based on your operating system and preference.

### Method 1: All-in-One Package (Easiest on Windows)

1. Download the latest [Release](https://github.com/openhachimi/hifisampler/releases) package, extract it, and double-click `start.bat` once to start the rendering server.
2. Connect to OpenUtau (choose any option):
   - **Option A (Recommended — Direct Copy)**: After running `start.bat` at least once, simply **copy** `hifisampler.exe` and `hifisampler.yaml` into OpenUtau's `Resamplers` folder (no administrator privileges or symbolic links required; OpenUtau can also auto-start the server when needed).
   - **Option B (OpenUtau Built-in Tool)**: Keep `start.bat` running and select the built-in `hifisampler` resampler directly inside OpenUtau.
   - **Option C (Symbolic Link)**:
     ```cmd
     mklink "C:\[OpenUtau Path]\Resamplers\hifisampler.exe" "C:\[Project Path]\hifisampler.exe"
     ```
3. Set your resampler in UTAU / OpenUtau to `hifisampler.exe`.

---

### Method 2: Manual Installation with `uv` (Windows / Linux / macOS)

1. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
2. Download the source code and model files from [Releases](https://github.com/openhachimi/hifisampler/releases) and extract them into the project folder.
3. **Start the server**:
   - **Windows**: Double-click `start.bat` (on first launch, it automatically detects your GPU and sets up `.venv`).
   - **Linux / macOS**: Run `./start.sh` in the project directory. On first launch, it automatically detects your hardware (NVIDIA CUDA / AMD ROCm / CPU), installs dependencies, and generates `config.yaml`.
4. **Connect to OpenUtau**:
   - **Windows**: Copy `hifisampler.exe` and `hifisampler.yaml` into `OpenUtau\Resamplers\`.
   - **Linux / macOS**: Either select OpenUtau's built-in `hifisampler` while the server is running, or copy `hifisampler.sh` (or `hifisampler-linux-x64`) and `hifisampler.yaml` into `~/.local/share/OpenUtau/Resamplers/`:
     ```bash
     cp hifisampler.sh ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     cp hifisampler.yaml ~/.local/share/OpenUtau/Resamplers/hifisampler.sh.yaml
     chmod +x ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     ```

---

### Method 3: Manual Installation with Conda / pip

1. Create and activate a Python 3.11 environment:
   ```bash
   conda create -n hifisampler python=3.11 -y
   conda activate hifisampler
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the model files from [Releases](https://github.com/openhachimi/hifisampler/releases) into the project folder, and install the appropriate PyTorch version from the [PyTorch website](https://pytorch.org/) according to the GPU guide below.
4. Start the rendering server:
   ```bash
   conda activate hifisampler
   python hifiserver.py
   ```

---

## GPU & Hardware Recommendations (NVIDIA / AMD / Intel / CPU)

By default, hifisampler uses **ONNX mode**, which works out of the box across all hardware. You can customize `config.yaml` based on your setup:

| OS & Hardware | Recommended Mode (`config.yaml`) | Notes |
| :--- | :--- | :--- |
| **Windows + AMD / Intel GPU** | **ONNX mode (Default)** | Official PyTorch ROCm is largely unavailable on Windows. **Simply keep the default `model_type: "onnx"`** — DirectML acceleration is built in and automatically uses your AMD or Intel GPU (including integrated GPUs) with zero extra setup. |
| **Windows / Linux + NVIDIA GPU** | **ONNX (Default) or PyTorch (`ckpt`)** | Default ONNX works out of the box. To use PyTorch models instead, uncomment the `.ckpt` / `.pt` paths in `config.yaml` (you can also set `fp16: true` for faster rendering). |
| **Linux + AMD GPU (ROCm)** | **PyTorch (`ckpt`) or ONNX** | `./start.sh` automatically detects AMD GPUs and installs ROCm PyTorch on Linux. You can switch `config.yaml` to `ckpt` mode (setting `fp16: true` is also recommended). |
| **CPU Only** | **ONNX mode (Default)** | Keep the default settings. `start.bat` and `start.sh` automatically install the lightweight CPU environment when no dedicated GPU is detected. |

### Useful `config.yaml` Settings

- **`wave_norm`** (default `true`): Enables note-level loudness normalization (used with the `P` flag).
- **`fp16`** (default `false`): When running PyTorch (`ckpt`) models on a GPU, set to `true` for faster half-precision inference and lower VRAM usage.
- **`idle_timeout`** (default `0`): Auto-shutdown timer in seconds. `0` keeps the server running indefinitely; setting it to e.g. `600` will automatically shut down the server after 10 minutes of inactivity to free GPU VRAM (the client will automatically start it again on the next render).

---

## Implemented Flags

Placing `hifisampler.yaml` in OpenUtau's `Resamplers` folder alongside the client allows OpenUtau to automatically display all supported expression flags:

- **g (`gen`):** Adjust gender / formants (aligned with standard UTAU resamplers: positive values = deeper/masculine, negative values = brighter/feminine).
  - Range: `-100` to `100` | Default: `0`
- **Hb:** Adjust breathiness / noise component percentage.
  - Range: `0` to `500` | Default: `100`
- **Hv:** Adjust voicing / harmonic component percentage.
  - Range: `0` to `150` | Default: `100`
- **Ht:** Adjust vocal tension.
  - Range: `-100` to `100` | Default: `0`
- **HG:** Vocal fry / growl effect.
  - Range: `0` to `100` | Default: `0`
- **P (`norm`):** Note-level loudness normalization strength targeting -16 LUFS (requires `wave_norm: true` in `config.yaml`).
  - Range: `0` to `100` | Default: `100`
- **t:** Pitch shift in units of 10 cents (`t+10` = +1 semitone, matching standard UTAU resamplers).
  - Range: `-120` to `120` | Default: `0`
- **A:** Modulate amplitude based on pitch changes for more natural vibrato.
  - Range: `-100` to `100` | Default: `0`
- **G:** Force regenerate feature cache for the note (overwrites existing cache).
  - Toggle flag (no value needed)
- **He:** Enable Mel spectrogram loop mode for long notes (when `loop_mode: false` in `config.yaml`).
  - Toggle flag (no value needed)

_Note: The flags `B` and `V` were renamed to `Hb` and `Hv` respectively to avoid conflicts with standard UTAU flags that have different definitions._

## Acknowledgments

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
