# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders) 的新一代 UTAU / OpenUtau 重采样器，支持 Windows、Linux 与 macOS，原生兼容 NVIDIA CUDA、AMD ROCm 与 ONNX Runtime。

## 为什么叫 hifisampler?

hifisampler 是由 [straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用 pc-nsf-hifigan 替换了原来的 WORLD。

## pc-nsf-hifigan 和其它传统 vocoder 有什么不同?

pc-nsf-hifigan 采用神经网络对输入的特征进行上采样，音质比传统的 vocoder 更加清晰。
pc-nsf-hifigan 是传统 nsf-hifigan 的改进，支持输入与 mel 不匹配的 f0，因此可以用于 UTAU / OpenUtau 的重采样。

## 如何使用?

提供三种安装方式，按自己需要和喜好选择。

### 1. 使用环境整合包（Windows 推荐）

1. 下载最新的 [release](https://github.com/openhachimi/hifisampler/releases) 整合包并解压，运行一次 `start.bat` 启动渲染服务（首次启动会自动在 `~/.hifisampler/server.json` 写入全局定位信标）。
2. 将客户端放入 OpenUtau 的 `Resamplers` 文件夹：
   - **直接复制（推荐，支持服务端自启动）**：运行过一次 `start.bat` 或 `uv run hifiserver.py` 后，直接把 `hifisampler.exe` 和 `hifisampler.yaml` 复制到 `OpenUtau\Resamplers\` 目录即可，无需创建软链接！
   - **或创建软链接**：
     ```cmd
     mklink "C:\[OpenUtau路径]\Resamplers\hifisampler.exe" "C:\[项目路径]\hifisampler.exe"
     ```
3. 在 OpenUtau 中将重采样器设置为 `hifisampler.exe`。

### 2. Linux / macOS 与 OpenUtau 配置指南

1. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，解压源码与模型文件到项目目录。
2. 在项目目录下运行一次 `./start.sh`（或 `uv run hifiserver.py --register-only`）完成环境初始化与全局信标 `~/.hifisampler/server.json` 注册。
3. 将客户端接入 OpenUtau 的 `Resamplers` 目录（二选一）：
   - **方式 A（OpenUtau 内置 Hifisampler 工具）**：启动 `./start.sh` 后，直接在 OpenUtau 中选择内置 `hifisampler`（连接 `127.0.0.1:8572`）。
   - **方式 B（外部重采样器 + 自动拉起服务端 + 表情参数自动加载）**：将项目根目录下的 `hifisampler.sh`（或编译好的 `hifisampler-linux-x64`）与 `hifisampler.yaml` 直接复制或软链接到 `~/.local/share/OpenUtau/Resamplers/`：
     ```bash
     cp hifisampler.sh ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     cp hifisampler.yaml ~/.local/share/OpenUtau/Resamplers/hifisampler.sh.yaml
     chmod +x ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     ```
     *注：在 Linux 下若尚未启动服务端，`hifisampler.sh` 被 OpenUtau 调用时会自动优先弹出桌面终端窗口（支持 `ptyxis`、`kgx`、`gnome-terminal`、`konsole`、`kitty`、`alacritty`、`wezterm`、`foot`、`xfce4-terminal` 等），若无可用终端则自动退化为后台静默守护进程并将日志输出到 `logs/hifiserver.log`。*

### 3. AMD ROCm / NVIDIA CUDA / FP16 加速配置

1. **AMD ROCm（Linux 推荐 PyTorch 原生 ROCm + `.ckpt`/`.pt` 模型）**：
   - 安装 ROCm 版 PyTorch：
     ```bash
     uv pip install --index-url https://download.pytorch.org/whl/rocm6.3 torch
     ```
   - 在 `config.yaml` 中启用 PyTorch 模型与可选 FP16 半精度加速（服务端已内置 `MIOPEN_FIND_MODE=FAST` 与启动预热 Warmup，且对 STFT、复数掩码与 NSF 相位累加保持 FP32 高精度防破音）：
     ```yaml
     model:
       vocoder_path: "./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"
       model_type: "ckpt"
       hnsep_model_path: "./hnsep/vr/model.pt"
     performance:
       max_workers: 2
       device: "auto"      # 自动识别 CUDA / ROCm / MPS / CPU
       fp16: true          # 开启 GPU FP16 推理加速
       idle_timeout: 0     # 空闲自动关闭服务端释放显存秒数（0 为常驻不关闭）
     ```
2. **NVIDIA CUDA / CPU**：
   - 默认 `pyproject.toml` 配置为 `pytorch-cu129`；若仅使用 CPU，可在 `pyproject.toml` 的 `[tool.uv.sources]` 中改为 `pytorch-cpu`。

### 4. 使用 conda 手动安装

1. 创建并激活虚拟环境：
   ```bash
   conda create -n hifisampler python=3.10 -y
   conda activate hifisampler
   pip install -r requirements.txt
   ```
2. 根据硬件前往 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA / ROCm / CPU 版本的 `torch`。
3. 运行 `python hifiserver.py` 启动服务。

## 已实现的 flags

将 `hifisampler.yaml` 放入 OpenUtau 的 `Resamplers` 目录（与客户端同名，如 `hifisampler.yaml` 或 `hifisampler.sh.yaml`）即可在 OpenUtau 中自动加载以下全部表情参数：

- **g:** 调整性别/共振峰。
  - 范围: `-600` 到 `600` | 默认: `0`
- **Hb:** 控制气息/噪波成分的量。
  - 范围: `0` 到 `500` | 默认: `100`
- **Hv:** 控制发声/谐波成分的量。
  - 范围: `0` 到 `150` | 默认: `100`
- **HG:** 怒音/嘶吼效果。
  - 范围: `0` 到 `100` | 默认: `0`
- **P:** 以 -16 LUFS 为基准进行音符粒度的响度标准化。需要在 `config.yaml` 中设置 `wave_norm` 为 `true`。
  - 范围: `0` 到 `100` | 默认: `100`
- **t:** 移动特定的音高，单位为音分，1 分=1/100 个半音。
  - 范围: `-1200` 到 `1200` | 默认: `0`
- **Ht:** 调整张力。
  - 范围: `-100` 到 `100` | 默认: `0`
- **A:** 根据音高的变化调制振幅，有助于生成相对真实的颤音。
  - 范围: `-100` 到 `100` | 默认: `0`
- **G:** 强制重新生成特征缓存（忽略已有缓存并覆写磁盘缓存）。
  - 无需数值
- **He:** 为长音启用 Mel 频谱循环模式。
  - 无需数值

_注：由于 `B` 和 `V` 与其他 UTAU flags 名称冲突但定义不同，因此分别更名为 `Hb` 和 `Hv`。_

## 感谢

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
