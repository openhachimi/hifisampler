import logging
import os
from pathlib import Path
import threading
import uuid
import numpy as np
import resampy
import torch
from config import CONFIG
from util.cache_manager import cache_manager
import soundfile as sf

if CONFIG.wave_norm:
    try:
        import pyloudnorm as pyln
        logging.info("pyloudnorm imported for wave normalization.")
    except ImportError:
        logging.warning(
            "pyloudnorm not found, wave normalization disabled.")
        CONFIG.wave_norm = False  # Disable if import fails

_METER_CACHE = {}
_METER_LOCK = threading.Lock()
_PRE_EMPHASIS_CACHE = {}
_PRE_EMPHASIS_LOCK = threading.Lock()


def _get_loudness_meter(rate: int, block_size: float):
    key = (int(rate), float(block_size))
    with _METER_LOCK:
        meter = _METER_CACHE.get(key)
        if meter is None:
            meter = pyln.Meter(rate, block_size=block_size)
            _METER_CACHE[key] = meter
        return meter


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _peak_abs(x: np.ndarray) -> float:
    return max(-float(x.min()), float(x.max())) if x.size > 0 else 0.0


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
    if len(audio) == 0 or _peak_abs(audio) < 1e-8:
        return audio

    original_length = len(audio)
    original_audio = audio.copy()  # 保存原始音频用于后续处理
    voiced_frames = []
    first_voiced = 0
    hop_length = int(rate * 0.01)

    if CONFIG.trim_silence:
        frame_length = int(rate * 0.02)  # 20ms窗口
        hop_length = int(rate * 0.01)    # 10ms步长

        # Match range(0, len(audio) - frame_length, hop_length) frame count exactly
        num_frames = max(0, (len(audio) - frame_length + hop_length - 1) // hop_length)
        if num_frames > 0:
            cs = np.empty(len(audio) + 1, dtype=np.float64)
            cs[0] = 0.0
            np.cumsum(np.square(audio, dtype=np.float64), out=cs[1:])
            starts = np.arange(num_frames) * hop_length
            mean_sq = (cs[starts + frame_length] - cs[starts]) / frame_length
            thresh_sq = max(1e-20, float(10.0 ** (CONFIG.silence_threshold / 10.0)))
            voiced_frames = np.flatnonzero(mean_sq > thresh_sq).tolist()

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
        elif num_frames > 0:
            return original_audio

    if len(audio) == 0 or _peak_abs(audio) < 1e-8:
        return original_audio

    # 如果音频长度小于最小块大小，仅对测量副本进行填充，避免反射回声混入输出尾部
    min_samples = int(rate * block_size)
    if len(audio) < min_samples:
        padding_length = min_samples - len(audio)
        measure_audio = np.pad(audio, (0, padding_length), mode='reflect')
    else:
        measure_audio = audio

    # Measure the loudness first using cached BS.1770 meter
    meter = _get_loudness_meter(rate, block_size)
    _loudness = meter.integrated_loudness(measure_audio)
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


def _get_pre_emphasis_tensors(device: torch.device):
    key = (CONFIG.win_size, CONFIG.n_fft, str(device))
    with _PRE_EMPHASIS_LOCK:
        cached = _PRE_EMPHASIS_CACHE.get(key)
        if cached is None:
            window = torch.hann_window(CONFIG.win_size, device=device, dtype=torch.float32)
            fft_bin = CONFIG.n_fft // 2 + 1
            arange_bins = torch.arange(0, fft_bin, device=device, dtype=torch.float32)
            cached = (window, arange_bins, fft_bin)
            _PRE_EMPHASIS_CACHE[key] = cached
        return cached


def _fant_lf_poles(rd: float, f0_ref: float = 240.0):
    """
    Compute analytical LF glottal spectral poles (fa, Fg) and open quotient Oq
    from Fant (1995) 'The LF-model revisited: Transformations and frequency domain analysis'
    and Doval et al. (2006) 'The spectrum of glottal flow models'.
    Valid physiological range: Rd in [0.25, 3.00].
    """
    rd = float(np.clip(rd, 0.25, 3.00))
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    # Fant (1995) Eq. 4: 0.11 * Rd = (0.5 + 1.2 * Rk) * (Rk / (4 * Rg) + Ra)
    rg_denom = 4.0 * (0.11 * rd - ra * (0.5 + 1.2 * rk))
    rg = (rk * (0.5 + 1.2 * rk)) / max(rg_denom, 1e-4)
    oq = (1.0 + rk) / (2.0 * rg)

    fa = f0_ref / (2.0 * np.pi * ra)
    fg = f0_ref / (2.0 * oq)
    return fa, fg, oq


def _estimate_harmonic_f0(mean_power: torch.Tensor, sr: int, n_fft: int) -> float:
    """
    Zero-extra-STFT source fundamental frequency (f0_src) estimator from the
    voiced-frame mean power spectrum via 1D Wiener-Khinchin autocorrelation.
    Returns f0_src clamped in [95.0, 580.0] Hz (falls back to 240.0 Hz on silence/noise).
    """
    total_power = float(torch.sum(mean_power).item())
    if total_power < 1e-10:
        return 240.0

    autocorr = torch.fft.irfft(mean_power, n=n_fft)
    r0 = float(autocorr[0].item())
    if r0 <= 1e-10:
        return 240.0

    lag_min = max(2, int(sr / 580.0))
    lag_max = min(n_fft // 2 - 2, int(sr / 95.0))
    if lag_max <= lag_min + 2:
        return 240.0

    seg = autocorr[lag_min:lag_max + 1]
    max_val = float(torch.max(seg).item())
    if max_val < 0.25 * r0:
        return 240.0

    # Pick the first local peak >= 0.82 * max_val to prevent subharmonic octave-down jumps
    threshold = 0.82 * max_val
    is_local_peak = (seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] >= threshold)
    peak_indices = torch.nonzero(is_local_peak, as_tuple=False)
    if peak_indices.numel() > 0:
        rel_idx = int(peak_indices[0].item()) + 1
    else:
        rel_idx = int(torch.argmax(seg).item())

    tau = lag_min + rel_idx
    # 3-point parabolic interpolation for sub-sample lag accuracy
    if 1 <= tau < n_fft // 2 - 1:
        y_prev = float(autocorr[tau - 1].item())
        y_curr = float(autocorr[tau].item())
        y_next = float(autocorr[tau + 1].item())
        denom = y_prev - 2.0 * y_curr + y_next
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_prev - y_next) / denom
            tau_interp = tau + float(np.clip(delta, -0.5, 0.5))
        else:
            tau_interp = float(tau)
    else:
        tau_interp = float(tau)

    return float(np.clip(sr / max(tau_interp, 1.0), 95.0, 580.0))


def _apply_formant_plateau_protection(
    log_gain: torch.Tensor,
    mean_power: torch.Tensor,
    freqs: torch.Tensor,
    sr: int,
    n_fft: int,
    alpha_prot: float = 0.50,
) -> torch.Tensor:
    """
    Formant-Preserving Slope-Damped Plateau (Option C):
    Extracts vowel formant peaks (F1, F2, F3) via band-pass cepstral liftering
    (quefrency q in [4, sr/1200]) and damps the local slope of log_gain across
    formant peaks while preserving the full macroscopic LF spectral tilt.
    """
    if float(torch.sum(mean_power).item()) < 1e-10:
        return log_gain

    log_spec_mean = 0.5 * torch.log(torch.clamp(mean_power, min=1e-12))
    cep = torch.fft.irfft(log_spec_mean, n=n_fft)

    q_low = 4
    q_high = min(n_fft // 2 - 1, max(12, int(sr / 1200.0)))
    cep_bp = torch.zeros_like(cep)
    cep_bp[q_low:q_high] = cep[q_low:q_high]
    cep_bp[-q_high + 1:-q_low + 1] = cep[-q_high + 1:-q_low + 1]

    formant_prominence = torch.fft.rfft(cep_bp, n=n_fft).real
    vowel_band = (freqs >= 220.0) & (freqs <= 3600.0)
    pos_prom = torch.relu(formant_prominence) * vowel_band.to(dtype=log_gain.dtype)
    max_prom = torch.max(pos_prom)
    if float(max_prom.item()) < 1e-5:
        return log_gain

    formant_mask = torch.clamp(pos_prom / (max_prom + 1e-6), 0.0, 1.0)

    # Damp local derivative d(log_gain)/df inside formant peak bandwidths [220 Hz, 3600 Hz]
    dg = log_gain[1:] - log_gain[:-1]
    mask_mid = 0.5 * (formant_mask[1:] + formant_mask[:-1])
    low_idx = min(log_gain.numel() - 2, max(1, int(200.0 * n_fft / sr)))
    ref_idx = min(log_gain.numel() - 1, max(low_idx + 2, int(4500.0 * n_fft / sr)))

    dg_damped = dg.clone()
    dg_damped[low_idx:ref_idx] = dg[low_idx:ref_idx] * (1.0 - alpha_prot * mask_mid[low_idx:ref_idx])

    # Rescale the damped increments in [low_idx, ref_idx] so the total tilt at ref_idx is 100% preserved
    orig_span = float(torch.sum(dg[low_idx:ref_idx]).item())
    damped_span = float(torch.sum(dg_damped[low_idx:ref_idx]).item())
    if abs(damped_span) > 1e-5:
        dg_damped[low_idx:ref_idx] = dg_damped[low_idx:ref_idx] * (orig_span / damped_span)

    log_gain_prot = torch.zeros_like(log_gain)
    log_gain_prot[0] = log_gain[0]
    log_gain_prot[1:] = log_gain[0] + torch.cumsum(dg_damped, dim=0)

    return log_gain_prot


def pre_emphasis_base_tension(wave, b):
    """
    Clean-room minimum-phase LF glottal spectral tilt filter with wide dynamic range,
    adaptive source-f0 anchoring, and formant-preserving cepstral liftering (Option C),
    based on Fant (1995) and Doval et al. (2006).

    Args:
        wave: [1, 1, t] periodic harmonic waveform tensor
        b: float tension parameter (-Ht / 50, in [-2.0, 2.0]; b < 0 is higher tension)
    """
    if abs(float(b)) < 1e-6:
        return wave

    orig_dtype = wave.dtype
    wave = wave.float()
    original_length = wave.size(-1)
    pad_length = (CONFIG.hop_size - (original_length %
                  CONFIG.hop_size)) % CONFIG.hop_size
    min_stft_len = CONFIG.n_fft // 2 + 1
    if original_length + pad_length < min_stft_len:
        pad_length = min_stft_len - original_length
    wave = torch.nn.functional.pad(
        wave, (0, pad_length), mode='constant', value=0)
    wave = wave.squeeze(1)

    window, arange_bins, _ = _get_pre_emphasis_tensors(wave.device)
    spec = torch.stft(
        wave,
        CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window,
        return_complex=True
    )

    b_clamped = float(np.clip(b, -2.0, 2.0))

    # Compute energy-weighted mean power spectrum across frames for adaptive f0_src and formant liftering
    power_spec = spec.real ** 2 + spec.imag ** 2  # [1, F, T]
    frame_energy = torch.sum(power_spec, dim=1, keepdim=True)  # [1, 1, T]
    total_energy = torch.sum(frame_energy)
    if float(total_energy.item()) > 1e-10:
        mean_power = torch.sum(power_spec * frame_energy, dim=2).squeeze(0) / (total_energy + 1e-12)
    else:
        mean_power = torch.mean(power_spec, dim=2).squeeze(0)

    f0_src = _estimate_harmonic_f0(mean_power, int(CONFIG.sample_rate), int(CONFIG.n_fft))

    # Wide-range physiological Rd mapping in [0.25, 3.00] around neutral Rd0 = 1.0
    rd0 = 1.0
    rd_target = float(np.clip(3.0 ** (b_clamped / 2.0), 0.25, 3.00))

    fa0, _, oq0 = _fant_lf_poles(rd0, f0_ref=f0_src)
    fa_t, fg_t, oq_t = _fant_lf_poles(rd_target, f0_ref=f0_src)

    freqs = arange_bins * (float(CONFIG.sample_rate) / float(CONFIG.n_fft))

    # 1. Cascade return-phase & open-asymmetry spectral tilt pole ratio (Doval et al. 2006)
    # Wide dynamic range cascade order for dramatic yet smooth +/-100 excursion
    tilt_order = 1.85 if b_clamped < 0.0 else 2.25
    gain_return = torch.pow(
        (1.0 + (freqs / fa0) ** 2) / (1.0 + (freqs / fa_t) ** 2),
        0.5 * tilt_order,
    )

    # 2. Open-phase glottal formant H1-H2 shelf, anchored to unity (0 dB) at DC
    oq_ratio = (oq_t / oq0) ** 2.8
    shelf_corner = 2.2 * fg_t
    gain_open_raw = torch.sqrt(
        (oq_ratio ** 2 + (freqs / shelf_corner) ** 2) / (1.0 + (freqs / shelf_corner) ** 2)
    )
    dc_weight = (freqs ** 2) / (freqs ** 2 + 45.0 ** 2)
    gain_open = 1.0 + dc_weight * (gain_open_raw - 1.0)

    # 3. High-frequency (>8.5 kHz) ultra-treble protection to prevent boosting >10 kHz hiss
    log_gain = torch.log(torch.clamp(gain_return * gain_open, min=1e-4))
    hf_taper = 1.0 / (1.0 + (freqs / 8500.0) ** 2)
    log_gain = torch.where(
        log_gain > 0.0,
        log_gain * (0.72 + 0.28 * hf_taper),
        log_gain,
    )

    # 4. Option C: Formant-preserving slope-damped plateau on vowel formants F1/F2/F3
    log_gain = _apply_formant_plateau_protection(
        log_gain, mean_power, freqs, int(CONFIG.sample_rate), int(CONFIG.n_fft), alpha_prot=0.50
    )

    # 5. Minimum-phase complex transfer function via causal real cepstrum folding
    cepstrum = torch.fft.irfft(log_gain, n=CONFIG.n_fft)
    cep_min = torch.zeros_like(cepstrum)
    half_n = CONFIG.n_fft // 2
    cep_min[0] = cepstrum[0]
    cep_min[1:half_n] = 2.0 * cepstrum[1:half_n]
    cep_min[half_n] = cepstrum[half_n]
    gain = torch.exp(torch.fft.rfft(cep_min, n=CONFIG.n_fft)).unsqueeze(0).unsqueeze(2)

    filtered_wave = torch.istft(
        spec * gain,
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window,
        length=wave.size(-1),
    )

    original_rms = torch.sqrt(torch.mean(wave ** 2))
    filtered_rms = torch.sqrt(torch.mean(filtered_wave ** 2))
    if filtered_rms > 1e-8:
        # Equal-loudness glottal flow compensation across spectral tilt
        loudness_comp = float(10.0 ** (b_clamped * 0.5 / 20.0))
        filtered_wave = filtered_wave * (original_rms / filtered_rms) * loudness_comp
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

    resolved_loc = Path(os.path.normcase(os.path.abspath(str(loc))))

    cached_wav = cache_manager.load_wav_cache(resolved_loc)
    if cached_wav is not None:
        return cached_wav

    x, fs = sf.read(str(loc), dtype='float32')
    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than stereo
        x = np.mean(x, axis=1, dtype=np.float32)

    if fs != CONFIG.sample_rate:
        x = resampy.resample(x, fs, CONFIG.sample_rate).astype(np.float32, copy=False)

    cache_manager.save_wav_cache(resolved_loc, x)
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
