# hifisampler

[中文文档](README_zh_cn.md) | [English Document](README.md)

一个基于 [pc-nsf-hifigan](https://github.com/openvpi/vocoders)的新的 utau 重采样器。

## 为什么叫 hifisampler?

hifisampler 是由 [straycatresampler](https://github.com/UtaUtaUtau/straycat) 修改而来，用 pc-nsf-hifigan 替换了原来的 world。

## pc-nsf-hifigan 和其它传统 vocoder 有什么不同?

pc-nsf-hifigan 采用神经网络对输入的特征进行上采样，音质比传统的 vocoder 更加清晰。
pc-nsf-hifigan 是传统 nsf-hifigan 的改进，支持输入与 mel 不匹配的 f0，因此可以用于 utau 的重采样。

## 如何使用?

提供三种安装方式，按自己需要和喜好选择。

### 使用环境整合包（推荐 NVIDIA GPU）

1. 下载最新的 [release](https://github.com/openhachimi/hifisampler/releases) 整合包并解压，运行`start.bat` 启动渲染服务。
2. （可选但不推荐）如果采用服务端自启动（实验性），则需要保持 `config.default.yaml`, `hifiserver.py`, `hifisampler.exe` 以及 `launch_server.py` 四个文件在同一目录下。建议解压后保持原文件结构不变。OpenUTAU 可以采用创建软链接的方式将 `hifisampler.exe` 链接到 Resamplers 文件夹。

   ```cmd
   mklink "C:\[OpenUTAU路径]\Resamplers\hifisampler.exe" "C:\[项目路径]\hifisampler.exe"
   ```

3. 将 utau 的重采样器设置为 `hifisampler.exe`，并确保渲染服务已启动。

### 使用 uv 手动安装

0. 安装 uv，安装方法请参考 [uv 文档](https://docs.astral.sh/uv/getting-started/installation/)。
1. 下载 [release](https://github.com/openhachimi/hifisampler/releases) 源码并解压，进入文件夹。
2. 从 release assets 下载模型文件，解压后放入项目文件夹。
3. 在 config.yaml 中填入配置（如果是首次使用，则在 config.default.yaml 中修改，首次运行时会自动生成 config.yaml 文件）。
4. 根据个人需求，可以选择适合自己硬件的 cuda 版本以获取加速，具体而言，在 `pyproject.toml` 中，如果需要 cuda 加速，则需要修改配置文件的 `tool.uv.sources` 部分，如：

   ```toml
   [tool.uv.sources]
   torch = [
      { index = "pytorch-cu128" },
   ]
   ```

   如果使用 CPU，则：

   ```toml
   [tool.uv.sources]
    torch = [
        { index = "pytorch-cpu" },
    ]
    ```

5. （可选但不推荐）如果采用服务端自启动（实验性），则需要保持 `config.default.yaml`, `hifiserver.py`, `hifisampler.exe` 以及 `launch_server.py` 四个文件在同一目录下。建议解压后保持原文件结构不变。OpenUTAU 可以采用创建软链接的方式将 `hifisampler.exe` 链接到 Resamplers 文件夹。

   ```cmd
   mklink "C:\[OpenUTAU路径]\Resamplers\hifisampler.exe" "C:\[项目路径]\hifisampler.exe"
   ```

6. 每次使用前运行 `hifiserver.py` 启动渲染服务。如果采用服务端自启动（实验性）则可以跳过此步骤。在终端输入

   ```bash
   uv run hifiserver.py
   ```

7. 将 utau 的重采样器设置为 `hifisampler.exe`，并确保渲染服务已启动。

### 使用 conda 手动安装

0. 下载 [release](https://github.com/openhachimi/hifisampler/releases) 解压，进入文件夹，如果有嵌套的话继续打开直到显示有 hifiserver.py 文件。

1. 安装 miniconda ，安装好后在刚刚打开的文件夹右击，点在终端中打开，输入

   ```bash
   conda create -n hifisampler python=3.8 -y
   ```

   创建完成后，输入

   ```bash
   conda activate hifisampler
   ```

   即可进入虚拟环境。虚拟环境只需创建一次，以后直接进入有 hifiserver.py 的文件夹打开终端输入 activate 命令即可  
   进入成功后会发现终端工作目录前面有 ( hifisamper ) 标志  
   如果报错说 conda 不是内部外部命令，是环境变量没设置好，搜索： conda 设置环境变量，按要求操作即可

2. 安装依赖，输入

   ```bash
   pip install -r requirements.txt
   ```

3. 从 release assets 下载模型文件，解压后放入项目文件夹。
4. 在 [torch 官网] (<https://pytorch.org/>) 下载 cuda 版本的 pytorch ( 如果你确定只使用 onnx 版，那么可以下载 cpu 版的 pytorch )
   具体安装方法：进入后往下滑，看到 INSTALL PYTORCH 以及一个表格，PyTorch Build 选 Stable , Your OS 选你的操作系统 , Package 选 pip , Language 选 python , Compute Platform 如果要下载 gpu 版就选带 cuda 的，下载 cpu 版选 cpu ，然后复制 Run this Command 右边表格里的命令到终端运行
5. （可选但不推荐）如果采用服务端自启动（实验性），则需要保持 `config.default.yaml`, `hifiserver.py`, `hifisampler.exe` 以及 `launch_server.py` 四个文件在同一目录下。建议解压后保持原文件结构不变。OpenUTAU 可以采用创建软链接的方式将 `hifisampler.exe` 链接到 Resamplers 文件夹。

   ```cmd
   mklink "C:\[OpenUTAU路径]\Resamplers\hifisampler.exe" "C:\[项目路径]\hifisampler.exe"
   ```

6. 每次使用前运行 `hifiserver.py` 启动渲染服务。如果采用服务端自启动（实验性）则可以跳过此步骤。在终端输入

   ```bash
   conda activate hifisampler
   python hifiserver.py
   ```

7. 将 utau 的重采样器设置为 `hifisampler.exe`，并确保渲染服务已启动。

## 已实现的 flags

- **g:** 调整性别/共振峰。
  - 范围: `-600` 到 `600` | 默认: `0`
- **Hb:** 控制气息/噪波成分的量。
  - 范围: `0` 到 `500` | 默认: `100`
- **Hv:** 控制发声/谐波成分的量。
  - 范围: `0` 到 `150` | 默认: `100`
- **HG:** 怒音/嘶吼效果。
  - 范围: `0` 到 `100` | 默认: `0`
- **P:** 以 -16 LUFS 为基准进行音符粒度的响度标准化。需要在 config.yaml 中设置 `wave_norm` 为 `true`。
  - 范围: `0` 到 `100` | 默认: `100`
- **t:** 移动特定的音高，单位为音分，1 分=1/100 个半音。
  - 范围: `-1200` 到 `1200` | 默认: `0`
- **Ht:** 调整张力。
  - 范围: `-100` 到 `100` | 默认: `0`
- **A:** 根据音高的变化调制振幅，有助于生成相对真实的颤音。
  - 范围: `-100` 到 `100` | 默认: `0`
- **G:** 强制重新生成特征缓存（忽略已有缓存）。
  - 无需数值
- **He:** 基于 `config.yaml`（`processing.loop_mode`）切换 Mel 频谱延长模式。
  - 当 `processing.loop_mode=false` 时，`He` 会启用循环模式。
  - 当 `processing.loop_mode=true` 时，`He` 会切换为拉伸模式。
  - 无需数值

_注：由于 `B` 和 `V` 与其他 UTAU flags 名称冲突但定义不同，因此分别更名为 `Hb` 和 `Hv`。_

## 其他注意事项

- 如果采用服务端自启动（实验性），在服务器启动过程中关闭终端窗口或渲染，可能导致服务端卡住，可以尝试手动解除 `hifisampler.exe` 的文件占用。推荐手动使用 `./start.bat` 启动渲染服务。

## 感谢

- [yjzxkxdn](https://github.com/yjzxkxdn)
- [openvpi](https://github.com/openvpi) for the pc-nsf-hifigan
- [MinaminoTenki](https://github.com/Lanhuace-Wan)
- [Linkzerosss](https://github.com/Linkzerosss)
- [MUTED64](https://github.com/MUTED64)
- [mili-tan](https://github.com/mili-tan)
