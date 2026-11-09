# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders) 的 UTAU / OpenUtau 重采样器，支持 Windows、Linux 与 macOS，兼容 NVIDIA、AMD、Intel 显卡及纯 CPU 运行。

**如果您用于人力 VOCALOID（音MAD），推荐使用我们的 [Hachimisampler](https://github.com/openhachimi/hachimisampler)。**

## 为什么叫 hifisampler?

hifisampler 由 [straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用神经网络声码器 pc-nsf-hifigan 替换了传统的 WORLD 声码器。

## pc-nsf-hifigan 和传统声码器有什么不同?

pc-nsf-hifigan 采用神经网络对输入特征进行上采样，音质比传统算法声码器更加清晰自然；同时它支持输入与原始采样不匹配的音高（f0），非常适合用于 UTAU / OpenUtau 的变调与重采样。

---

## 如何使用?

提供以下安装方式，您可以根据自己的操作系统和习惯任选其一。

### 方式一：使用环境整合包（Windows 最简方式）

1. 下载最新的 [Release](https://github.com/openhachimi/hifisampler/releases) 整合包并解压，双击运行一次 `start.bat` 启动渲染服务。
2. 将客户端接入 OpenUtau（以下方法任选其一）：
   - **方法 A（推荐，直接复制）**：运行过一次 `start.bat` 后，直接把解压目录里的 `hifisampler.exe` 和 `hifisampler.yaml` **复制**到 OpenUtau 的 `Resamplers` 文件夹即可（无需管理员权限，也支持由 OpenUtau 自动拉起服务端）。
   - **方法 B（使用 OpenUtau 内置工具）**：保持 `start.bat` 窗口运行，直接在 OpenUtau 中选择内置的 `hifisampler` 重采样器。
   - **方法 C（创建软链接）**：
     ```cmd
     mklink "C:\[OpenUtau路径]\Resamplers\hifisampler.exe" "C:\[项目路径]\hifisampler.exe"
     ```
3. 在 UTAU / OpenUtau 中将重采样器设为 `hifisampler.exe` 即可开始渲染。

---

### 方式二：使用 uv 手动安装（Windows / Linux / macOS 通用）

1. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
2. 下载 [Release](https://github.com/openhachimi/hifisampler/releases) 中的源码与模型文件，解压并放入同一项目文件夹内。
3. **启动服务**：
   - **Windows**：双击运行 `start.bat`（或在终端执行 `uv run hifiserver.py`）。
   - **Linux / macOS**：在项目目录终端执行 `./start.sh`（或 `uv run hifiserver.py`）。首次运行会自动生成 `config.yaml` 并完成环境配置。
4. **接入 OpenUtau**：
   - **Windows**：将 `hifisampler.exe` 和 `hifisampler.yaml` 复制到 `OpenUtau\Resamplers\` 目录。
   - **Linux / macOS**：可以直接使用 OpenUtau 内置的 `hifisampler`；或者将项目目录下的 `hifisampler.sh`（或 `hifisampler-linux-x64`）与 `hifisampler.yaml` 复制到 `~/.local/share/OpenUtau/Resamplers/`（复制 `hifisampler.yaml` 时若配合脚本使用，可重命名为 `hifisampler.sh.yaml` 以自动加载表情参数）：
     ```bash
     cp hifisampler.sh ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     cp hifisampler.yaml ~/.local/share/OpenUtau/Resamplers/hifisampler.sh.yaml
     chmod +x ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     ```

---

### 方式三：使用 Conda 手动安装

1. 安装 Miniconda，在项目文件夹下打开终端并创建虚拟环境：
   ```bash
   conda create -n hifisampler python=3.10 -y
   conda activate hifisampler
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 从 [Release](https://github.com/openhachimi/hifisampler/releases) 下载模型文件解压放入项目文件夹，并根据下方显卡说明在 [PyTorch 官网](https://pytorch.org/) 安装对应版本的 `torch`。
4. 启动渲染服务：
   ```bash
   conda activate hifisampler
   python hifiserver.py
   ```

---

## 不同显卡（NVIDIA / AMD / CPU）配置建议

项目默认使用兼容性最好的 **ONNX 模式**，开箱即用。您可以根据自己的硬件在 `config.yaml` 中调整以获得最佳速度：

| 您的硬件与系统 | 推荐模型模式 (`config.yaml`) | 配置说明 |
| :--- | :--- | :--- |
| **Windows + AMD / Intel 显卡** | **ONNX 模式（默认）** | Windows 下 PyTorch 官方几乎不支持 ROCm，**请直接保持默认的 `model_type: "onnx"`**。程序已内置 DirectML 加速，会自动调用 AMD / Intel 显卡（含核显），无需任何额外配置。 |
| **Windows / Linux + NVIDIA 显卡** | **ONNX（默认）或 PyTorch (`ckpt`)** | 默认 ONNX 即可加速；若想使用 PyTorch 模型，取消 `config.yaml` 中 `.ckpt` / `.pt` 的注释，并可将 `fp16` 设为 `true` 进一步提速。 |
| **Linux + AMD 显卡 (ROCm)** | **PyTorch (`ckpt`) 或 ONNX** | Linux 下支持原生 ROCm 加速。若使用 `.ckpt` / `.pt` 模型，可在项目环境中执行：<br>`uv pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch`<br>并在 `config.yaml` 中切换为 `ckpt` 模式，推荐同时开启 `fp16: true`。 |
| **纯 CPU（无独立显卡）** | **ONNX 模式（默认）** | 保持默认配置即可。若使用 `uv` 手动安装且不需要下载体积较大的 CUDA 组件，可将 `pyproject.toml` 中的 `pytorch-cu129` 改为 `pytorch-cpu`。 |

### 常用 `config.yaml` 选项说明

- **`wave_norm`**（默认 `true`）：开启音符粒度的响度平衡，配合 `P` 参数使用。
- **`fp16`**（默认 `false`）：使用 GPU 运行 PyTorch (`ckpt`) 模型时，设为 `true` 可开启半精度加速并降低显存占用。
- **`idle_timeout`**（默认 `0`）：空闲自动关闭时间（单位：秒）。设为 `0` 表示服务端常驻；若设为如 `600`，则 10 分钟不渲染后服务端会自动退出以释放显卡显存，下次在 OpenUtau 渲染时客户端会自动重新拉起服务端。

---

## 已实现的 Flags 参数

将 `hifisampler.yaml` 一并放入 OpenUtau 的 `Resamplers` 文件夹后，OpenUtau 会自动识别并显示以下全部表情参数：

- **g:** 调整性别 / 共振峰。
  - 范围: `-600` 到 `600` | 默认: `0`
- **Hb:** 控制气息 / 噪波成分的音量。
  - 范围: `0` 到 `500` | 默认: `100`
- **Hv:** 控制发声 / 谐波成分的音量。
  - 范围: `0` 到 `150` | 默认: `100`
- **Ht:** 调整声线张力（Tension）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **HG:** 怒音 / 嘶吼效果（Growl）。
  - 范围: `0` 到 `100` | 默认: `0`
- **P:** 音符响度标准化强度（以 -16 LUFS 为基准，需在 `config.yaml` 中保持 `wave_norm: true`）。
  - 范围: `0` 到 `100` | 默认: `100`
- **t:** 音高微调，单位为音分（1 音分 = 1/100 半音）。
  - 范围: `-1200` 到 `1200` | 默认: `0`
- **A:** 随音高变化自动调制音量振幅，使颤音更自然。
  - 范围: `-100` 到 `100` | 默认: `0`
- **G:** 强制重新生成当前音符的特征缓存（忽略并覆盖已有缓存）。
  - 开关型参数，无需数值
- **He:** 为长音启用 Mel 频谱循环模式（当 `config.yaml` 中 `loop_mode: false` 时可按音符单独开启）。
  - 开关型参数，无需数值

_注：由于 `B` 和 `V` 与其他传统 UTAU 重采样器的参数重名但含义不同，因此在 hifisampler 中分别命名为 `Hb` 和 `Hv`。_

## 感谢

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
