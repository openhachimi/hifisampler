# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

A new UTAU resampler based on [pc-nsf-hifigan](https://github.com/openvpi/vocoders) for virtual singer.

**For Jinriki please use our [Hachimisampler](https://github.com/openhachimi/hachimisampler).**

## Why is it called hifisampler?

Hifisampler was modified from [straycatresampler](https://github.com/UtaUtaUtau/straycat), replacing the original WORLD with pc-nsf-hifigan.

## What makes pc-nsf-hifigan different from traditional vocoders?

pc-nsf-hifigan employs neural networks to upsample the input features, offering clearer audio quality than traditional vocoders. It is an improvement over the traditional nsf-hifigan, supporting f0 inputs that do not match mel, making it suitable for UTAU resampling.

## How to use?

Three installation methods are provided; choose the one that best suits your needs and preferences.

### Using Integrated Environment Package (Recommended for NVIDIA GPU)

1. Download the latest [release](https://github.com/openhachimi/hifisampler/releases) package and extract it. Run `start.bat` to start the rendering service.
2. If you're using the experimental server auto-start feature (Optional, but not recommended), keep `config.default.yaml`, `hifiserver.py`, `hifisampler.exe`, and `launch_server.py` in the same directory. It's best to keep the original file structure after extracting the release. For OpenUTAU, you can create a symbolic link to place `hifisampler.exe` in the Resamplers folder.

   ```cmd
   mklink "C:\[OpenUTAU Path]\Resamplers\hifisampler.exe" "C:\[Project Path]\hifisampler.exe"
   ```

3. Set the UTAU resampler to `hifisampler.exe` and ensure the rendering service is running.

### Manual Installation using uv

0. Install `uv` following the instructions in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).
1. Download and extract the source code from the [latest release](https://github.com/openhachimi/hifisampler/releases). Then, navigate into the extracted folder.
2. Download model files from release assets. Unzip and place it in the project folder.
3. Fill in the configuration details in `config.yaml`. If this is your first time using the software, modify `config.default.yaml` instead. The `config.yaml` file will be automatically generated upon the first run.
4. Depending on your hardware, you can select a suitable CUDA version for acceleration. To do this, modify the `tool.uv.sources` section in `pyproject.toml`. For example, to enable CUDA acceleration:

   ```toml
   [tool.uv.sources]
   torch = [
      { index = "pytorch-cu128" },
   ]
   ```

    If you're using the CPU version, set it as follows:
  
    ```toml
    [tool.uv.sources]
     torch = [
         { index = "pytorch-cpu" },
     ]
    ```

5. If you're using the experimental server auto-start feature (Optional, but not recommended), keep `config.default.yaml`, `hifiserver.py`, `hifisampler.exe`, and `launch_server.py` in the same directory. It's best to keep the original file structure after extracting the release. For OpenUTAU, you can create a symbolic link to place `hifisampler.exe` in the Resamplers folder.

   ```cmd
   mklink "C:\[OpenUTAU Path]\Resamplers\hifisampler.exe" "C:\[Project Path]\hifisampler.exe"
   ```

6. Before each use, run `hifiserver.py` to start the rendering service. If you're using the experimental server auto-start feature, you can skip this step. Enter the following command in your terminal:

   ```bash
   uv run hifiserver.py
   ```

7. Set the resampler in UTAU to `hifisampler.exe` and ensure the rendering service is running.

### Manual Installation using conda/pip

1. Install Python 3.10 and run the following commands (it's strongly recommended to use conda for easier environment management):

   ```bash
   pip install -r requirements.txt
   ```

2. Download the CUDA version of PyTorch from the Torch website (If you're certain about only using the ONNX version, then downloading the CPU version of PyTorch is fine).
3. Download model files from release assets. Unzip and place it in the project folder.
4. If you're using the experimental server auto-start feature (Optional, but not recommended), keep `config.default.yaml`, `hifiserver.py`, `hifisampler.exe`, and `launch_server.py` in the same directory. It's best to keep the original file structure after extracting the release. For OpenUTAU, you can create a symbolic link to place `hifisampler.exe` in the Resamplers folder.

   ```cmd
   mklink "C:\[OpenUTAU Path]\Resamplers\hifisampler.exe" "C:\[Project Path]\hifisampler.exe"
   ```

5. Download the [release](https://github.com/openhachimi/hifisampler/releases), unzip it, and run 'hifiserver.py'.
6. Set UTAU's resampler to `hifisampler.exe`.

## Implemented flags

- **g:** Adjust gender/formants.
  - Range: `-600` to `600` | Default: `0`
- **Hb:** Adjust breath/noise.
  - Range: `0` to `500` | Default: `100`
- **Hv:** Adjust voice/harmonic.
  - Range: `0` to `150` | Default: `100`
- **HG:** Vocal fry/growl.
  - Range: `0` to `100` | Default: `0`
- **P:** Normalize loudness at the note level, targeting -16 LUFS. Enable this by setting `wave_norm` to `true` in your `config.yaml` file.
  - Range: `0` to `100` | Default: `100`
- **t:** Shift the pitch by a specific amount, in cents. 1 cent = 1/100 of a semitone.
  - Range: `-1200` to `1200` | Default: `0`
- **Ht:** Adjust tension.
  - Range: `-100` to `100` | Default: `0`
- **A:** Modulating the amplitude based on pitch variations, which helps creating a more realistic vibrato.
  - Range: `-100` to `100` | Default: `0`
- **G:** Force to regenerate feature cache (Ignoring existed cache).
  - No value needed
- **He:** Toggle Mel spectrum extension mode against `config.yaml` (`processing.loop_mode`).
  - If `processing.loop_mode=false`, `He` enables loop mode.
  - If `processing.loop_mode=true`, `He` switches to stretch mode.
  - No value needed

_Note: The flags `B` and `V` were renamed to `Hb` and `Hv` respectively because they conflict with other UTAU flags but have different definitions._

## Other Notes

- If using server-side auto-start (Experimental), closing the terminal window or rendering process during server startup may cause the server to freeze. You can try manually releasing the file lock on `hifisampler.exe`. We recommend manually starting the rendering service using `./start.bat` to avoid issues.

## Acknowledgments

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
