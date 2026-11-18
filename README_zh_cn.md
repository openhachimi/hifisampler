# Melos

[中文文档](README_zh_cn.md) | [English Document](README.md) | [Python 分支（hifisampler / `main`）](https://github.com/openhachimi/hifisampler/blob/main/README_zh_cn.md)

基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders) 与 HN-SEP 构建的纯 Rust 神经网络 UTAU / OpenUtau 歌声合成引擎（`melos` 分支），直接支持原生 OpenUtau 客户端实现**整句连续 Mel + F0 单次声码器合成**，跨平台支持 Windows、Linux 与 macOS，兼容 NVIDIA、AMD、Intel 显卡及纯 CPU 运行。

**如果您用于人力 VOCALOID（音MAD），推荐使用我们的 [Hachimisampler](https://github.com/openhachimi/hachimisampler)。**

---

## 为什么选择 Melos？

传统 UTAU / OpenUtau 重采样器受限于单音调用协议，只能**逐个音符独立运行声码器生成 `.wav`，再由拼接器在波形层做线性交叉淡化**：
- 每个音符单独走一遍神经网络声码器，各自生成独立的谐波初始相位，在连音（VCV）交叠区容易因相位不一致产生 `-2.4 ~ -3.2 dB` 的能量凹陷、梳状滤波或“双影感”；
- 一句 20 个音符的乐句需要反复启动 20 次 GPU 声码器推理，存在大量重复开销。

**Melos** 用纯 Rust 重写了全栈推理服务端与客户端，直接面向**原生 OpenUtau 客户端（`CLASSIC` 渲染器）与传统 UTAU** 引入整句连续 Mel 合成能力：

| 特性维度 | 传统单音拼接模式（`convergence` 等） | Melos 整句合成模式（`melos` + `melos-wavtool`） |
| :--- | :--- | :--- |
| **客户端支持** | 原生 OpenUtau / UTAU 客户端 | **原生 OpenUtau / UTAU 客户端** |
| **拼接与合成域** | 逐音独立生成 WAV 后做波形交叉淡化 | **能量域归一化连续 Mel + 连续 MIDI F0 拼接** |
| **GPU 声码器调用次数** | $N$ 个音符调用 $N$ 次 | **整句仅调用 1 次**（含首尾各 8 帧上下文填充与精准裁切） |
| **连音交叠过渡质量** | 相位独立，易出现音量凹陷与衔接痕迹 | **整句共享连续 NSF 正弦激励相位，共振峰平滑自然** |
| **整句渲染速度** | 1.0x 基准 | **提速约 1.5x–2.3x**（声码器耗时降低 **2.0x–2.7x**） |
| **缓存回填兼容** | 单音 `.wav` | 合成后自动事务性回填各音符真实 `.wav`，切回 `convergence` 无缝可用 |

### 核心架构亮点

1. **纯 Rust 原生高性能架构（[`rust/`](rust/)）**：
   - 彻底摆脱生产环境的 Python / PyTorch 运行时依赖：服务端 `hifiserver` 约 `4.3 MiB`，客户端 `melos`（`553 KiB`）与 `melos-wavtool`（`517 KiB`）均为毫秒级启动的原生可执行文件。
   - 原生集成 ONNX Runtime 硬件加速（Windows 下默认内置 **DirectML**，无需配置即可调用 AMD / Intel / NVIDIA 独显与核显；同时支持 **CUDA**、**CoreML** 与多线程 **CPU** 自动回退）。
   - 采用 `HIFIBIN1` 二进制特征缓存（加载 `< 1 ms`，向下兼容旧版 `.hifi.npz`）、HTTP/DSP 并发流水线重叠，以及守护进程自管理生命周期（`--managed`、`--idle-timeout`、自动寻址与按需拉起）。
2. **丰富的物理声学表现力参数**：
   - 内置声门张力模型（`Ht`）、Klatt 声道开口度（`Ho`）、3.2 kHz 歌手共振峰（`Hr`）、深滑音自动气泡音（`auto_vocal_fry`）、怒音嘶吼（`HG`）、ITU-R BS.1770-4 响度归一化（`P`）、颤音斜率振幅调制（`A`）及谐噪分离独立控制（`Hb` / `Hv`）。
3. **四代演进与真实模型基准**：
   - 完整实测对比与性能指标请参阅 [`rust/BENCHMARK_REPORT.md`](rust/BENCHMARK_REPORT.md)。

---

## 如何使用？

### 方式一：使用 All-in-One 一键整合包（Windows / macOS / Linux 推荐，开箱即用）

无需安装 Rust、Python 或手动配置任何环境与模型依赖：

1. 前往 [Releases](https://github.com/openhachimi/hifisampler/releases) 下载对应系统的 Melos All-in-One 一键整合包（Release 标题 `v0.2.0-melos`）：
   - **Windows x64**：`melos-v0.2.0-win64-dml.zip`（内置 DirectML 加速与完整 ONNX 模型，支持 AMD / Intel / NVIDIA 独显、核显及纯 CPU）
   - **macOS（Apple Silicon M 系列）**：`melos-v0.2.0-macos-arm64-coreml.tar.gz`（内置 arm64 CoreML / CPU ONNX Runtime 与完整 ONNX 模型）
   - **macOS（Intel）**：`melos-v0.2.0-macos-x64-coreml.tar.gz`（内置 x86_64 CoreML / CPU ONNX Runtime 与完整 ONNX 模型）
   - **Linux x64 / ARM64**：`melos-v0.2.0-linux-x64-aio.tar.gz` / `melos-v0.2.0-linux-arm64-aio.tar.gz`（内置 ONNX Runtime 与完整 ONNX 模型；若需只含 Apache-2.0 纯程序的压缩包，也可按平台选择 `*-bins-only.*`）
   - **版本号约定（与 `main` 分支区分）**：为避免与同仓库的 `hifisampler`（`main` 分支，采用奇数次版本号 `0.1.0` / `0.3.0` / `0.5.0`……）混淆，**Melos（`melos` 分支）从 `0.2.0` 起步并采用偶数次版本号**（`0.2.0`、`0.4.0`、`0.6.0`……）。同时 Melos 默认使用独立端口 `8574` 与独立信标 `~/.melos/server.json`，可与旧版同机共存互不干扰。
2. **首次使用时初始化注册并预热服务**：
   - **Windows**：双击运行一次目录中的 `start.bat`（会自动生成 `config.yaml`、注册 `~/.melos/server.json` 并启动后台推理服务）。
   - **macOS / Linux**：解压后先清理一次 macOS 下载隔离属性（仅 macOS 需要：`xattr -cr ./melos`），然后运行 `bash ./melos/start.sh --register-only`（或直接 `bash ./melos/start.sh` 前台启动观察日志）。
   - 注册完成后，日常在 OpenUtau 中**无需手动提前运行启动脚本**——每次渲染时 `melos` / `melos.exe` 检测到服务端未运行，都会自动在后台按需拉起服务端并在就绪后完成渲染。
3. **接入原生 OpenUtau 客户端（保持默认的 `CLASSIC` 渲染器）**：
   - **整句连续 Mel 合成（推荐）**：
     1. 将整合包中的 `melos.exe`（macOS / Linux 为 `melos`）与 `melos.yaml` 复制到 OpenUtau 的 `Resamplers` 文件夹（macOS / Linux 默认位于 `~/.local/share/OpenUtau/Resamplers/` 或 `~/Library/Application Support/OpenUtau/Resamplers/`）；
     2. 将 `melos-wavtool.exe`（macOS / Linux 为 `melos-wavtool`）复制到 OpenUtau 的 `Wavtools` 文件夹（注：`melos` 内置双角色自动识别，复制一份 `melos` / `melos.exe` 放入 `Wavtools` 文件夹同样可用）；
     3. 在 OpenUtau 轨道头将 **重采样器（Resampler）** 设为 `melos.exe`（/ `melos`），将 **拼接器（Wavtool）** 设为 `melos-wavtool.exe`（/ `melos-wavtool`）。
   - **单音兼容用法（可选）**：若需搭配 OpenUtau 内置 `convergence` / `simple` 拼接器，仅将 `melos.exe`（/ `melos`）与 `melos.yaml` 放入 `Resamplers` 文件夹即可，`melos` 会自动回退为标准单音合成。
4. 在钢琴卷帘左下角点击 **`⚙`（表情设置） ->「添加渲染器建议的所有表情」->「应用」**，即可使用全部表情参数。

---

### 方式二：从源码编译（Windows / Linux / macOS）

1. 确保已安装 Rust 工具链（`rustup`，`edition = 2021`），编译发布版二进制：
   ```bash
   cd rust
   cargo build --release --bins
   ```
   编译产物位于 `rust/target/release/`：
   - `hifiserver`（/ `hifiserver.exe`）：HTTP + ONNX 声学推理服务端
   - `melos`（/ `melos.exe`）：整句特征准备与双角色自识别重采样客户端
   - `melos-wavtool`（/ `melos-wavtool.exe`）：整句连续 Mel + F0 拼接与单次声码器触发工具
   - `hifi_bench`（/ `hifi_bench.exe`）：延迟、吞吐、声学等价性与整句连续性基准套件
2. 在引擎根目录放置 ONNX Runtime 动态库（Windows 下为支持 DirectML 的 `onnxruntime.dll`，或通过 `ORT_DYLIB_PATH` 指定）以及模型权重：
   - 声码器模型：`pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx`（上游获取：[openvpi/vocoders Releases](https://github.com/openvpi/vocoders/releases)）
   - 谐噪分离模型（可选，启用 `Hb` / `Hv` / `Ht` / `Ho` / `Hr` 时需要）：`hnsep/vr/model.onnx`（上游权重：[yxlllc/vocal-remover `hnsep_240512`](https://github.com/yxlllc/vocal-remover/releases/tag/hnsep_240512)，可用 `hnsep/convert_hnsep_to_onnx.py` 导出；在 `config.yaml` 中将 `hnsep_model_path` 设为 `""` 时可在无 HN-SEP 情况下独立运行）
3. 复制 `config.default.yaml` 为 `config.yaml` 并确认 `model_type: "onnx"` 与模型路径正确后，运行一次服务端注册并接入 OpenUtau（Linux / macOS 将 `melos` 与 [`rust/melos.yaml`](rust/melos.yaml) 放入 `~/.local/share/OpenUtau/Resamplers/`，`melos-wavtool` 放入 `~/.local/share/OpenUtau/Wavtools/`）：
   ```powershell
   .\hifiserver.exe --register-only
   ```

---

## 硬件加速与常用 `config.yaml` 配置

| 选项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `performance.device` | `"auto"` | 推理设备偏好：`"auto"`（Windows 优先 DirectML，Linux 优先 CUDA，macOS 优先 CoreML，不可用时自动回退 CPU）、`"dml"`、`"cuda"`、`"coreml"` 或 `"cpu"` |
| `performance.max_workers` | `2` | 并发处理工作线程数（HTTP / DSP 与 GPU 推理自动流水线重叠，推荐 `2`–`4`） |
| `performance.idle_timeout` | `0` | 空闲自动退出秒数（`0` 为常驻；`--managed` 客户端自拉起模式下若未配置默认启用 `600` 秒超时释放显存） |
| `processing.wave_norm` | `true` | 开启基于 ITU-R BS.1770-4（-16 LUFS）的音符响度平衡，配合 `P` 参数使用 |
| `processing.auto_vocal_fry` | `true` | 检测深幅音高下潜（Scoop/Release）时自动合成自然声门气泡音（开启 `HG` 怒音时自动避让） |
| `processing.loop_mode` | `true` | 长音超采样时启用镜像循环延展（也可在设为 `false` 时通过 `He` 参数按音符单独开启） |
| `processing.peak_limit` | `1.0` | 输出波形峰值限幅上限（`0.0`–`1.0`） |

### 服务端状态与诊断接口

`hifiserver` 默认监听 `http://127.0.0.1:8574`（可通过 `--port` 参数、`MELOS_PORT` 或 `config.yaml` 修改，默认与 `hifisampler` 错开端口与 `~/.melos/server.json` 信标以便并存）：
- `GET /health`：返回服务就绪状态、在途任务数与运行时间
- `GET /stats`：返回总请求数、特征缓存/HN-SEP 缓存命中率、P95 延迟及各阶段耗时分布（特征提取 / 声码器推理 / 后处理）

---

## 已实现的 Flags 参数

导入 `melos.yaml` 后即可在原生 OpenUtau 客户端中使用以下全部表情参数：

- **g (`gen`):** 调整性别 / 共振峰（正值偏成熟/男声，负值偏明亮/女声）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **Hb:** 控制气息 / 非周期噪波成分百分比。
  - 范围: `0` 到 `500` | 默认: `100`
- **Hv:** 控制发声 / 周期谐波成分百分比。
  - 范围: `0` 到 `150` | 默认: `100`
- **Ht:** 调整声门张力（Tension，物理频谱倾斜与非周期耦合）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **Ho:** 调整声道开口度（Openness，正值敞亮通透、负值柔和拢音）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **Hr:** 调整歌手共振峰（Singer's Formant，围绕 3.2 kHz 增强穿透金属芯或柔化高频毛刺）。
  - 范围: `-100` 到 `100` | 默认: `0`
- **HG:** 怒音 / 嘶吼效果（Growl）。
  - 范围: `0` 到 `100` | 默认: `0`
- **P (`norm`):** 音符响度标准化强度（目标 -16 LUFS，需 `wave_norm: true`）。
  - 范围: `0` 到 `100` | OpenUtau 表情默认: `86`（CLI 缺省默认 `100`）
- **t:** 音高微调，单位为 10 音分（`t+10` = 升高 1 个半音）。
  - 范围: `-120` 到 `120` | 默认: `0`
- **A:** 随音高斜率自动调制音量振幅，增强颤音与滑音动态自然度。
  - 范围: `-100` 到 `100` | 默认: `0`
- **G:** 强制重新生成当前音符的特征缓存（覆盖已有缓存）。
  - 开关型参数，无需数值
- **He:** 为当前长音启用 Mel 频谱镜像循环模式（当 `loop_mode: false` 时单独生效）。
  - 开关型参数，无需数值

_注：为避免与传统 UTAU 重采样器同名但定义不同的 `B`、`V` 参数冲突，Melos 中分别命名为 `Hb` 和 `Hv`。_

---

## 分发协议与模型许可证说明

> **All-in-One（AIO）整合包整体分发约束：**
> 仓库本体源码与纯可执行程序基于 Apache-2.0 开源；由于 All-in-One 一键整合包内部捆绑了第三方预训练神经网络模型权重，**任何包含预训练模型权重的 All-in-One 整合包仅限非商业用途（Non-Commercial Only），禁止售卖、收费分发或作为商业产品的一部分重新分发**，并须遵循下方各组件的许可要求：

1. **Melos 引擎源码与二进制程序 — [Apache License 2.0](LICENSE)**
   - `rust/` 下的全部 Rust 引擎代码及本仓库原创代码采用 Apache-2.0 协议。
   - `hnsep/` 目录下的网络结构与 ONNX 导出辅助脚本（`nets.py`、`layers.py`、`convert_hnsep_to_onnx.py`）源自 [`tsurumeso/vocal-remover`](https://github.com/tsurumeso/vocal-remover) 与 [`yxlllc/vocal-remover`](https://github.com/yxlllc/vocal-remover)，遵循 **[MIT License](hnsep/LICENSE)**（Copyright (c) 2019 tsurumeso）。
2. **`pc-nsf-hifigan` 声码器预训练权重 — [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
   - 由 [OpenVPI Team](https://github.com/openvpi/vocoders) 提供（DiffSinger 社区声码器企划，AIO 内置原版官方 ONNX 导出权重）。
   - 二次分发任何包含该声码器权重的整合包或镜像时，**必须**：
     1. 不得售卖或就二次分发收取任何费用（`NC` 非商业约束）；
     2. 附上 CC BY-NC-SA 4.0 许可证副本或链接，若涉及权重修改需以相同协议共享（`SA`）；
     3. 完整保留 `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/` 目录下的 `NOTICE.txt`、`NOTICE.zh-CN.txt` 与 `STATEMENTS.txt`，不得删改其中的全体声库授权人及公开数据集致谢名单；
     4. 若对权重做过微调或修改，需附加修改说明（本整合包保持原版官方权重未作参数调整）。
3. **`hnsep` 谐噪分离预训练权重 — 原作者 [`yxlllc`](https://github.com/yxlllc) 版权所有（无独立商业授权）**
   - 来自 [`yxlllc/vocal-remover` Release `hnsep_240512`](https://github.com/yxlllc/vocal-remover/releases/tag/hnsep_240512)（`hnsep/vr/model.onnx` 为原版 `vr/model.pt` 经 `convert_hnsep_to_onnx.py` 等价导出的 ONNX 推理图，未变更训练参数）。
   - 注意：上游源码仓库使用 MIT License，但 `hnsep_240512.zip` 权重压缩包本身未附带单独的权重或训练数据许可证声明。若您需要进行严格的二次分发或商业合规审计，请**勿**再分发 `hnsep` 权重文件——可在 `config.yaml` 中将 `hnsep_model_path` 设为空字符串 `""` 以无 HN-SEP 方式纯 Apache-2.0/CC 合规运行，或引导终端用户从上游 Release 自行获取。

---

## 感谢

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [yxlllc](https://github.com/yxlllc) & [tsurumeso](https://github.com/tsurumeso) for vocal-remover / HN-SEP
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
