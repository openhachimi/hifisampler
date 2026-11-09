import logging
import os
from pathlib import Path
import uuid
import numpy as np
import resampy
import torch
from config import CONFIG
import soundfile as sf

if CONFIG.wave_norm:
    try:
        import pyloudnorm as pyln
        logging.info("pyloudnorm imported for wave normalization.")
    except ImportError:
        logging.warning(
            "pyloudnorm not found, wave normalization disabled.")
        CONFIG.wave_norm = False  # Disable if import fails


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400, strength=100
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)
        strength: strength of the normalization. Defaults to 100.

    Returns:
        loudness normalized audio
    """
    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-8:
        return audio

    original_length = len(audio)
    original_audio = audio.copy()  # 保存原始音频用于后续处理
    voiced_frames = []
    first_voiced = 0
    hop_length = int(rate * 0.01)

    if CONFIG.trim_silence:
        def get_rms_db(audio_segment):
            if len(audio_segment) == 0:
                return -np.inf
            rms = np.sqrt(np.mean(np.square(audio_segment)))
            if rms < 1e-10:  # 避免log(0)错误
                return -np.inf
            return 20 * np.log10(rms)

        frame_length = int(rate * 0.02)  # 20ms窗口
        hop_length = int(rate * 0.01)    # 10ms步长

        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms_db = get_rms_db(frame)
            rms_values.append(rms_db)

        # 使用阈值检测有声帧
        voiced_frames = [i for i, rms in enumerate(
            rms_values) if rms > CONFIG.silence_threshold]

        if voiced_frames:
            first_voiced = voiced_frames[0]
            last_voiced = voiced_frames[-1]

            # 添加一些余量，避免突然截断
            padding_frames = int(rate * 0.1) // hop_length  # 添加100ms的余量

            # 确保不超出边界
            start_sample = max(0, first_voiced * hop_length)
            end_sample = min(len(audio), (last_voiced + 1 +
                             padding_frames) * hop_length + frame_length)

            trimmed_audio = audio[start_sample:end_sample]
            logging.debug(
                f'Trimmed silence: {len(audio)} -> {len(trimmed_audio)} samples')

            # 使用截取后的音频进行响度标准化
            audio = trimmed_audio

    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-8:
        return original_audio

    # 如果音频长度小于最小块大小，进行填充
    min_samples = int(rate * block_size)
    if len(audio) < min_samples:
        padding_length = min_samples - len(audio)
        audio = np.pad(audio, (0, padding_length), mode='reflect')

    # Measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)
    if not np.isfinite(_loudness):
        return original_audio

    # Apply strength to calculate the target loudness
    final_loudness = _loudness + (loudness - _loudness) * strength / 100

    # Loudness normalize audio to [loudness] LUFS
    audio = pyln.normalize.loudness(audio, _loudness, final_loudness)

    # 如果启用了无声截取功能，需要恢复原始长度
    if CONFIG.trim_silence:
        # 创建一个全零数组作为输出
        output = np.zeros(original_length, dtype=original_audio.dtype)

        # 将标准化后的音频放回原位置，并添加平滑过渡
        if voiced_frames:  # 确保有声音帧存在
            start_sample = max(0, first_voiced * int(hop_length))

            # 计算需要放回的音频长度
            available_length = min(len(audio), original_length - start_sample)

            # 创建一个逐渐衰减的窗函数，用于音频的尾部淡出
            # 最多200ms或音频长度的1/4
            fade_length = min(int(rate * 0.2), available_length // 4)
            fade_out = np.ones(available_length)

            if fade_length > 0:
                # 在末尾应用淡出效果
                fade_out[-fade_length:] = np.linspace(1.0, 0.0, fade_length)

            # 应用淡出效果并放回原位置
            output[start_sample:start_sample +
                   available_length] = audio[:available_length] * fade_out
            # 如果原始音频有后续部分，应用交叉淡入淡出
            if start_sample + available_length < original_length:
                remain_length = original_length - \
                    (start_sample + available_length)
                crossfade_length = min(fade_length, remain_length)

                if crossfade_length > 0:
                    crossfade_start = start_sample + available_length
                    # 从原始音频获取剩余部分
                    remain_audio = original_audio[crossfade_start:original_length]

                    # 应用淡入效果到原始音频的剩余部分
                    fade_in = np.ones(remain_length)
                    fade_in[:crossfade_length] = np.linspace(
                        0.0, 1.0, crossfade_length)

                    # 填充剩余部分
                    output[crossfade_start:original_length] = remain_audio * fade_in
        else:  # 如果没有声音帧，直接返回原始音频
            output = original_audio[:original_length]

        audio = output

    # 如果原始音频短于block_size，裁剪回原始长度
    if original_length < min_samples:
        audio = audio[:original_length]

    return audio


def pre_emphasis_base_tension(wave, b):
    """
    Args:
        wave: [1, 1, t]
    """
    orig_dtype = wave.dtype
    wave = wave.float()
    original_length = wave.size(-1)
    pad_length = (CONFIG.hop_size - (original_length %
                  CONFIG.hop_size)) % CONFIG.hop_size
    wave = torch.nn.functional.pad(
        wave, (0, pad_length), mode='constant', value=0)
    wave = wave.squeeze(1)

    window = torch.hann_window(CONFIG.win_size, device=wave.device, dtype=torch.float32)
    spec = torch.stft(
        wave,
        CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window,
        return_complex=True
    )

    fft_bin = CONFIG.n_fft // 2 + 1
    x0 = fft_bin / ((CONFIG.sample_rate / 2) / 1500)
    freq_filter = (-b / x0) * torch.arange(0, fft_bin, device=wave.device, dtype=torch.float32) + b
    gain = torch.exp(torch.clamp(freq_filter, min=-2, max=2)).unsqueeze(0).unsqueeze(2)

    filtered_wave = torch.istft(
        spec * gain,
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window
    )

    original_max = torch.max(torch.abs(wave))
    filtered_max = torch.max(torch.abs(filtered_wave))
    if filtered_max > 1e-8:
        filtered_wave = filtered_wave * \
            (original_max / filtered_max) * (np.clip(b/(-15), 0, 0.33) + 1)
    else:
        filtered_wave = torch.zeros_like(filtered_wave)
    filtered_wave = filtered_wave.unsqueeze(1)
    filtered_wave = filtered_wave[:, :, :original_length]

    return filtered_wave.to(dtype=orig_dtype)


def read_wav(loc):
    """Read audio files supported by soundfile and resample to 44.1kHz if needed. Mixes down to mono if needed.

    Parameters
    ----------
    loc : str or file
        Input audio file.

    Returns
    -------
    ndarray
        Data read from WAV file remapped to [-1, 1] and in 44.1kHz
    """
    if isinstance(loc, str):  # make sure input is Path
        loc = Path(loc)

    exists = loc.exists()
    if not exists:  # check for alternative files (case-insensitive for Linux)
        supported_exts = {'.' + ext.lower() for ext in sf.available_formats().keys()}
        for ext in sf.available_formats().keys():
            for candidate_ext in ('.' + ext.lower(), '.' + ext.upper()):
                candidate = loc.with_suffix(candidate_ext)
                if candidate.exists():
                    loc = candidate
                    exists = True
                    break
            if exists:
                break

        if not exists and loc.parent.exists():
            target_stem = loc.stem.lower()
            for item in loc.parent.iterdir():
                if item.stem.lower() == target_stem and item.suffix.lower() in supported_exts:
                    loc = item
                    exists = True
                    break

    if not exists:
        raise FileNotFoundError(f"No supported audio file was found for: {loc}")

    x, fs = sf.read(str(loc))
    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than stereo
        x = np.mean(x, axis=1)

    if fs != CONFIG.sample_rate:
        x = resampy.resample(x, fs, CONFIG.sample_rate)

    return x


def save_wav(loc, x):
    """Atomically save audio data into a 16-bit PCM WAV file.

    Parameters
    ----------
    loc : str or file
        Output WAV file.

    x : ndarray
        Audio data in 44.1kHz within [-1, 1].
    """
    out_path = Path(loc)
    tmp_path = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(tmp_path), x, CONFIG.sample_rate, subtype='PCM_16', format='WAV')
        os.replace(str(tmp_path), str(out_path))
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        logging.error(f"Error saving WAV file {loc}: {e}")
        raise
