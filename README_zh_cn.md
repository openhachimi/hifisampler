# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders) 的 UTAU / OpenUtau 重采样器，支持 Windows、Linux 与 macOS，兼容 NVIDIA、AMD、Intel 显卡及纯 CPU 运行。

**如果您用于人力 VOCALOID（音MAD），推荐使用我们的 [HachiShifter](https://github.com/openhachimi/hachishifter) 这是灵感源于知名调音软件的全流程人力鬼畜工具。**

## 下一代 Rust 引擎分支：Melos（`melos` 分支）

当前 `main` 分支为 Python 版 **hifisampler**；如果您需要纯 Rust 重写的下一代引擎，请前往独立的 **[`melos` 分支（Melos）](https://github.com/openhachimi/hifisampler/tree/melos)**：

- **直接支持原生 OpenUtau 客户端的整句连续 Mel 合成（`melos` + `melos-wavtool`）**：无需更换 OpenUtau 客户端，即可将“逐音独立合成 WAV 再做波形交叉淡化”升级为**整句连续 Mel 频谱 + F0 拼接、单次 GPU 声码器直出**（含首尾上下文帧填充与裁切），消除音符交叠处的相位抵消凹陷，连音过渡更自然，整句渲染提速约 **1.5x–2.3x**。
- **纯 Rust 原生架构 + 一键整合包**：脱离 Python / PyTorch 运行时依赖，超轻量客户端（约 500 KiB）配合原生 ONNX Runtime（DirectML / CUDA / CoreML / CPU）推理服务端，并支持独立单音兼容用法与全套表现力表情参数。
- **版本号奇偶区分规则与独立并存**：为避免混淆，**Melos（`melos` 分支）采用偶数次版本号**（从 `0.2.0` 起步，后续 `0.2.0` / `0.4.0` / `0.6.0`……，对应 Release `v0.2.0-melos`），**hifisampler（`main` 分支）采用奇数次版本号**（后续 `0.1.0` / `0.3.0` / `0.5.0`……，对应 Release `v0.1.0-hifi`）。Melos 默认使用独立端口 `8574` 与信标 `~/.melos/server.json`，可与本机原有引擎互不干扰地并存。

👉 **前往查看：[Melos 中文文档 (`melos` 分支)](https://github.com/openhachimi/hifisampler/blob/melos/README_zh_cn.md) | [`melos` 分支代码主页](https://github.com/openhachimi/hifisampler/tree/melos)**

---

## 版本选择与预览版说明（`v0.1.0` / `v0.1.1` / `v0.1.2`）

> GitHub Releases 页面默认展示标记为 `Latest` 的 **`v0.1.0`** 正式版；如果您想使用新版 **`Ht` 声门张力模型** 或 **免 Flag 自动气泡音**，请前往 [Releases 页面](https://github.com/openhachimi/hifisampler/releases) 选择标记为 **`Pre-release`** 的 **`v0.1.1`** 或 **`v0.1.2`**。

| 版本 | 类型 | 核心特性与适用人群 |
| :--- | :--- | :--- |
| **[`v0.1.0`](https://github.com/openhachimi/hifisampler/releases/tag/v0.1.0)** | 正式版 (`Latest`) | 稳定基线版本：三级内存特征缓存、DirectML 固定分桶（彻底消除动态长度重编译卡顿）、客户端按需自拉起服务端、完整 `hifisampler.yaml` 表情映射。 |
| **[`v0.1.1`](https://github.com/openhachimi/hifisampler/releases/tag/v0.1.1)** | 预览版 (`Pre-release`) | **LF 声门模型张力（`Ht`）v3 重构**：使用 Fant LF 声门导数模型 + 2048 点最小相位倒谱滤波替代旧版线性斜坡；支持动态 `Rd` 生理映射、闭式 H1 基频锚点与耳道等响度补偿 `W(f)`（压紧不炸音不失真、松弛气声不丢基频音量），内置清辅音双门控保护的浊音气声联动及 `_htv3` 缓存隔离。 |
| **[`v0.1.2`](https://github.com/openhachimi/hifisampler/releases/tag/v0.1.2)** | 预览版 (`Pre-release`) | **免 Flag 相对音高（`pit`）自动气泡音合成** + **完整包含 `v0.1.1` 的 `Ht` v3 改进**：无需额外 Flag，在钢琴卷帘将 `pit` 相对当前音符向下拉低（低于 `-5.6` 半音渐入、`-14` 至 `-17` 半音展开）即可自动合成连续声门气泡音；内置三阶跨八度大跳免疫检测避免常规滑音误触发，可通过 `config.yaml` 中 `processing.auto_vocal_fry`（默认 `true`）随时开关。 |

---

## 为什么叫 hifisampler?

hifisampler 由 [straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用神经网络声码器 pc-nsf-hifigan 替换了传统的 WORLD 声码器。

## pc-nsf-hifigan 和传统声码器有什么不同?

pc-nsf-hifigan 采用神经网络对输入特征进行上采样，音质比传统算法声码器更加清晰自然；同时它支持输入与原始采样不匹配的音高（f0），非常适合用于 UTAU / OpenUtau 的变调与重采样。

---

## 如何使用?

提供以下安装方式，您可以根据自己的操作系统和习惯任选其一。

### 方式一：使用环境整合一键包（Windows 最简方式，免装 Python / 依赖 / 模型）

1. 前往 [Release](https://github.com/openhachimi/hifisampler/releases) 根据显卡下载对应的一键整合包并解压（正式版为 `v0.1.0`，如需新版 `Ht` 或自动气泡音可下载 `Pre-release` 中的 `v0.1.1` / `v0.1.2`，三者均提供完整一键包；已下载 `v0.1.0` 一键包的用户也可下载 `v0.1.1` / `v0.1.2` 源码 `.zip` 直接覆盖根目录升级）：
   - **`hifisampler-v*-win64-dml.zip`（通用推荐 / A卡 / I卡 / 核显 / CPU）**：内置完整 Python 环境、全部模型与 DirectML 硬件加速，体积小巧，所有 Windows 显卡开箱即用。
   - **`hifisampler-v*-win64-cuda.7z`（NVIDIA 独立显卡专属）**：内置完整 Python 环境、全部模型与 CUDA 版 PyTorch（支持 RTX 20/30/40/50 系显卡），默认开启 `ckpt + fp16` 半精度加速。
2. **首次使用时双击运行一次解压目录中的 `start.bat`**（会自动记录服务端路径并启动服务，无需联网下载依赖）。
3. 将客户端接入 OpenUtau（以下方法任选其一）：
   - **方法 A（推荐，直接复制 + 自动拉起服务端）**：将解压目录里的 `hifisampler.exe` 和 `hifisampler.yaml` **复制**到 OpenUtau 的 `Resamplers` 文件夹即可。
     - **服务端自启动说明**：只要首次运行过一次 `start.bat`，以后打开 OpenUtau 渲染时**无需再手动运行 `start.bat`**——当 `hifisampler.exe` 检测到后台服务未启动时，会自动为您拉起服务端控制台窗口并在就绪后继续渲染。（若日后移动了服务端文件夹位置，只需在新位置重新运行一次 `start.bat` 即可更新路径。）
   - **方法 B（使用 OpenUtau 内置工具）**：保持 `start.bat` 窗口运行，直接在 OpenUtau 中选择内置的 `hifisampler` 重采样器（注：内置重采样器不支持自动拉起服务端，需手动先开 `start.bat`）。
   - **方法 C（创建软链接）**：
     ```cmd
     mklink "C:\[OpenUtau路径]\Resamplers\hifisampler.exe" "C:\[项目路径]\hifisampler.exe"
     ```
3. 在 UTAU / OpenUtau 中将重采样器设为 `hifisampler.exe` 即可开始渲染。

---

### 方式二：使用 uv 手动安装（Windows / Linux / macOS 通用）

1. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
2. 下载 [Release](https://github.com/openhachimi/hifisampler/releases) 中的源码与模型文件，解压并放入同一项目文件夹内。
3. **首次启动服务**：
   - **Windows**：双击运行一次 `start.bat`（首次运行会自动按您的显卡类型创建 `.venv`、安装依赖并注册服务端路径）。
   - **Linux / macOS**：在项目目录终端执行一次 `./start.sh`（首次运行会自动检测 NVIDIA CUDA / AMD ROCm / CPU、配置环境并注册服务端路径）。
4. **接入 OpenUtau（支持自动拉起服务端）**：
   - **Windows**：将 `hifisampler.exe` 和 `hifisampler.yaml` 复制到 `OpenUtau\Resamplers\` 目录，后续在 OpenUtau 中渲染时会自动拉起服务端。
   - **Linux / macOS**：将项目目录下的 `hifisampler.sh` 与 `hifisampler.yaml` 复制到 `~/.local/share/OpenUtau/Resamplers/`（复制 `hifisampler.yaml` 时重命名为 `hifisampler.sh.yaml` 以自动加载表情参数），后续渲染时同样支持自动拉起服务端：
     ```bash
     cp hifisampler.sh ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     cp hifisampler.yaml ~/.local/share/OpenUtau/Resamplers/hifisampler.sh.yaml
     chmod +x ~/.local/share/OpenUtau/Resamplers/hifisampler.sh
     ```

---

### 方式三：使用 Conda 手动安装

1. 安装 Miniconda，在项目文件夹下打开终端并创建虚拟环境：
   ```bash
   conda create -n hifisampler python=3.11 -y
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
| **Linux + AMD 显卡 (ROCm)** | **PyTorch (`ckpt`) 或 ONNX** | Linux 下 `./start.sh` 会自动检测并安装 ROCm 版 PyTorch。您可在 `config.yaml` 中切换为 `ckpt` 模式，推荐同时开启 `fp16: true`。 |
| **纯 CPU（无独立显卡）** | **ONNX 模式（默认）** | 保持默认配置即可。`start.bat` / `start.sh` 在未检测到独立显卡时会自动安装轻量 CPU 版环境。 |

### 常用 `config.yaml` 选项说明

- **`auto_vocal_fry`**（仅 `v0.1.2+` 可用，默认 `true`）：当检测到相对当前音符深幅下潜的 `pit` 音高线时自动合成自然声门气泡音（带跨八度大跳免疫保护，开启 `HG` 怒音时自动避让；设为 `false` 可关闭）。
- **`wave_norm`**（默认 `true`）：开启音符粒度的响度平衡，配合 `P` 参数使用。
- **`fp16`**（默认 `false`）：使用 GPU 运行 PyTorch (`ckpt`) 模型时，设为 `true` 可开启半精度加速并降低显存占用。
- **`idle_timeout`**（默认 `0`）：空闲自动关闭时间（单位：秒）。设为 `0` 表示服务端常驻；若设为如 `600`，则 10 分钟不渲染后服务端会自动退出以释放显卡显存，下次在 OpenUtau 渲染时客户端会自动重新拉起服务端。

---

## 已实现的 Flags 参数

将 `hifisampler.yaml` 一并放入 OpenUtau 的 `Resamplers` 文件夹并重启 OpenUtau 后，在钢琴卷帘左下角点击 **`⚙`（表情设置） ->「添加渲染器建议的所有表情」->「应用」**，即可一键导入以下全部表情参数：

- **g (`gen`):** 调整性别 / 共振峰（与其他主流采样器一致：正值偏成熟/男声，负值偏幼态/女声）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **Hb:** 控制气息 / 噪波成分的音量百分比。
  - 范围: `0` 到 `500` | 默认: `100`
- **Hv:** 控制发声 / 谐波成分的音量百分比。
  - 范围: `0` 到 `150` | 默认: `100`
- **Ht:** 调整声线张力（Tension）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **Ho:** 调整口腔/咽腔开口度（Openness，正值敞亮饱满、负值含蓄拢音）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **Hr:** 调整歌手共振峰（Singer's Formant，围绕 3.2 kHz 增强穿透金属芯或柔化刺耳峰）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **HG:** 怒音 / 嘶吼效果（Growl）。
  - 范围: `0` 到 `100` | 默认: `0`
- **P (`norm`):** 音符响度标准化强度（以 -16 LUFS 为基准，需在 `config.yaml` 中保持 `wave_norm: true`）。
  - 范围: `0` 到 `100` | 默认: `100`
- **t:** 音高微调，单位为 10 音分（`t+10` = 升高 1 个半音，与传统 UTAU 采样器一致）。
  - 范围: `-120` 到 `120` | 默认: `0`
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
