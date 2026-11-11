import logging
import math
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
_LF_GLOTTAL_TABLE = None
_LF_GLOTTAL_LOCK = threading.Lock()


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


def _middle_ear_loudness_weight(freqs: torch.Tensor) -> torch.Tensor:
    """
    Psychoacoustic middle-ear / equal-loudness frequency weighting W(f)
    with 300 Hz low-frequency floor, 400 Hz double pole, 1200 Hz presence zero,
    3100 Hz upper pole, and 0.33 Stevens compression exponent.
    """
    fc = torch.clamp(freqs, min=300.0)
    fc2 = fc * fc
    num = (fc2 * fc2) * (fc2 + 1200.0 ** 2)
    den = ((fc2 + 400.0 ** 2) ** 2) * (fc2 + 3100.0 ** 2)
    return torch.pow(num / torch.clamp(den, min=1e-20), 0.33)


def _get_pre_emphasis_tensors(device: torch.device):
    key = (CONFIG.win_size, CONFIG.n_fft, CONFIG.sample_rate, str(device))
    with _PRE_EMPHASIS_LOCK:
        cached = _PRE_EMPHASIS_CACHE.get(key)
        if cached is None:
            window = torch.hann_window(CONFIG.win_size, device=device, dtype=torch.float32)
            fft_bin = CONFIG.n_fft // 2 + 1
            arange_bins = torch.arange(0, fft_bin, device=device, dtype=torch.float32)
            freqs = arange_bins * (float(CONFIG.sample_rate) / float(CONFIG.n_fft))
            ear_weight = _middle_ear_loudness_weight(freqs)
            cached = (window, arange_bins, fft_bin, freqs, ear_weight)
            _PRE_EMPHASIS_CACHE[key] = cached
        return cached


def _solve_lf_unit_period_params(rd: float):
    """
    Solve normalized continuous-time Fant (1995) LF glottal pulse parameters on unit period T0 = 1.0
    for Rd in [0.06, 3.00], using a C^inf-smooth softplus return-phase corner near Rd = 0.21
    so Te + Ta < 1.0 and parameter derivatives remain strictly continuous everywhere.
    """
    rd = float(np.clip(rd, 0.06, 3.00))
    tau = 0.035
    x_s = (rd - 0.2083333) / tau
    softplus_val = tau * (max(x_s, 0.0) + math.log1p(math.exp(-abs(x_s))))
    ra = 0.0008 + 0.048 * softplus_val
    rk = 0.224 + 0.118 * rd
    denom = (0.11 * rd) / (0.5 + 1.2 * rk) - ra
    rg = (0.25 * rk) / max(denom, 1e-4)

    tp = 1.0 / (2.0 * rg)
    ta = ra
    te = min(tp * (1.0 + rk), 1.0 - ta - 1e-4)
    wg = math.pi / tp
    tc = max(1.0 - te, 1e-5)

    # 1. Return-phase decay rate eps: 1 - eps * ta - exp(-eps * tc) = 0
    if tc <= ta + 1e-7:
        eps = 1.0 / max(ta, 1e-6)
    else:
        lo = 1e-4
        hi = 1.0 / max(ta, 1e-6)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if 1.0 - mid * ta - math.exp(-mid * tc) > 0.0:
                lo = mid
            else:
                hi = mid
        eps = 0.5 * (lo + hi)

    exp_ret = math.exp(-eps * tc)
    i_ret = (1.0 - exp_ret) / (eps * eps * ta) - (tc * exp_ret) / (eps * ta)

    # 2. Opening-phase growth rate alpha: strictly decreasing A_open(alpha) == i_ret
    sin_wte = math.sin(wg * te)
    cos_wte = math.cos(wg * te)

    def a_open(a: float) -> float:
        exp_neg = math.exp(-a * te) if a >= -50.0 else math.exp(min(-a * te, 60.0))
        return (-wg * exp_neg - a * sin_wte + wg * cos_wte) / (sin_wte * (a * a + wg * wg))

    a_lo = -1.0
    while a_open(a_lo) < i_ret and a_lo > -500.0:
        a_lo *= 2.0
    a_hi = 1.0
    while a_open(a_hi) > i_ret and a_hi < 1e6:
        a_hi *= 2.0
    for _ in range(52):
        a_mid = 0.5 * (a_lo + a_hi)
        if a_open(a_mid) > i_ret:
            a_lo = a_mid
        else:
            a_hi = a_mid
    alpha = 0.5 * (a_lo + a_hi)
    return te, ta, wg, eps, alpha


def _get_lf_glottal_table():
    """
    Lazily build and cache the exact closed-form continuous-time LF harmonic log-magnitude table
    ln(g_tilde_k(Rd)) for Rd in [0.07, 3.00] (N_RD = 96) and harmonic index k = 1..256,
    normalized strictly at the fundamental (g_tilde_1(Rd) == 1.0, i.e. 0.00 dB at H1),
    along with its arc-length equalized perceptual tilt coordinate psi_grid in [0.0, 1.0].
    """
    global _LF_GLOTTAL_TABLE
    with _LF_GLOTTAL_LOCK:
        if _LF_GLOTTAL_TABLE is not None:
            return _LF_GLOTTAL_TABLE

        n_rd = 96
        k_max = 256
        rd_grid = np.linspace(0.07, 3.00, n_rd, dtype=np.float64)
        ks = np.arange(1, k_max + 1, dtype=np.float64)
        w = 2.0 * np.pi * ks
        jw = 1j * w

        log_table = np.zeros((n_rd, k_max), dtype=np.float64)
        for i, rd in enumerate(rd_grid):
            te, ta, wg, eps, alpha = _solve_lf_unit_period_params(float(rd))
            tc = max(1.0 - te, 1e-5)
            sin_wte = math.sin(wg * te)
            cos_wte = math.cos(wg * te)
            neg_inv_sin = -1.0 / sin_wte

            exp_neg_ate = math.exp(-alpha * te) if alpha >= -50.0 else math.exp(min(-alpha * te, 60.0))
            e_j_wte = np.exp(-jw * te)
            g_open = neg_inv_sin * (
                wg * exp_neg_ate + e_j_wte * ((alpha - jw) * sin_wte - wg * cos_wte)
            ) / ((alpha - jw) ** 2 + wg * wg)

            exp_ret = math.exp(-eps * tc)
            e_pos_jwte = np.exp(jw * te)
            g_ret = (e_j_wte / (eps * ta)) * (
                (1.0 - exp_ret * e_pos_jwte) / (eps + jw) - exp_ret * (1.0 - e_pos_jwte) / jw
            )

            g_mag = np.abs(g_open + g_ret) / ks
            g1 = max(float(g_mag[0]), 1e-12)
            g_tilde = np.maximum(g_mag / g1, 1e-6)
            log_table[i, :] = np.log(g_tilde)
            log_table[i, 0] = 0.0

        # Construct strictly monotonic perceptual arc-length coordinate psi_grid in [0, 1]
        # (psi=0 at tense Rd=0.07, psi=1 at lax Rd=3.00) from harmonics k=2..24 log-slope steps
        step_norms = np.sqrt(np.mean(np.diff(log_table[:, 1:24], axis=0) ** 2, axis=1)) + 1e-5
        psi_cum = np.concatenate([[0.0], np.cumsum(step_norms)])
        psi_grid = psi_cum / psi_cum[-1]

        power_templates = np.exp(2.0 * log_table[:, :80])  # [N_RD, 80] for <= 8 kHz Rd0 estimation
        _LF_GLOTTAL_TABLE = (
            rd_grid.astype(np.float32),
            log_table.astype(np.float32),
            power_templates.astype(np.float32),
            psi_grid.astype(np.float32),
        )
        return _LF_GLOTTAL_TABLE


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

    # Pick the first local peak >= 0.82 * max_val (including boundary lag_min) to prevent subharmonic octave-down jumps
    threshold = 0.82 * max_val
    left_nb = autocorr[lag_min - 1:lag_max]
    right_nb = autocorr[lag_min + 1:lag_max + 2]
    is_local_peak = (seg >= left_nb) & (seg >= right_nb) & (seg >= threshold)
    peak_indices = torch.nonzero(is_local_peak, as_tuple=False)
    if peak_indices.numel() > 0:
        rel_idx = int(peak_indices[0].item())
    else:
        rel_idx = int(torch.argmax(seg).item())

    tau = lag_min + rel_idx
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


def _estimate_baseline_rd(
    mean_power: torch.Tensor,
    formant_env_log: torch.Tensor,
    freqs: torch.Tensor,
    f0_src: float,
    sr: int,
    n_fft: int,
) -> float:
    """
    Estimate intrinsic source glottal shape Rd0 of the recording (<= 8 kHz) free of
    studio proximity-effect and inter-harmonic valley bias by integrating harmonic peak
    neighborhoods around k * f0_src and matching against LF glottal templates over k >= 2
    alongside broadband harmonic presence-to-fundamental balance.
    Returns Rd0 in [0.45, 1.35] (low Rd0 for sharp/pressed 'edge'/'power' voices, ~1.0-1.3 for modal/lax).
    """
    total_p = float(torch.sum(mean_power).item())
    if total_p < 1e-10:
        return 1.0

    # 1. Broadband harmonic presence-to-low balance directly on mean_power (avoids inter-harmonic valley division)
    low_band = (freqs >= max(110.0, 0.80 * f0_src)) & (freqs <= max(850.0, 3.2 * f0_src))
    pres_band = (freqs >= 1500.0) & (freqs <= 6500.0)
    e_low = float(torch.sum(mean_power[low_band]).item()) + 1e-12
    e_pres = float(torch.sum(mean_power[pres_band]).item()) + 1e-12
    pres_ratio_db = 10.0 * math.log10(e_pres / e_low)

    # 2. Harmonic-peak k >= 2 Itakura-Saito slope against LF templates
    k_est = min(56, int(8000.0 / max(f0_src, 95.0)))
    rd_is = 1.0
    if k_est >= 5:
        mp_np = mean_power.detach().cpu().numpy()
        bin_hz = float(sr) / float(n_fft)
        half_bw_bins = max(1, int(round(0.32 * f0_src / bin_hz)))
        n_bins = mp_np.shape[0]
        q_obs = np.empty(k_est - 1, dtype=np.float64)
        for k in range(2, k_est + 1):
            fk = k * f0_src
            x_lip2 = (max(fk, 350.0) / 4250.0) ** 2
            lip_p = (2.0765167 * x_lip2) / (2.0765167 + x_lip2)
            c_bin = int(round(fk / bin_hz))
            lo_b = max(1, c_bin - half_bw_bins)
            hi_b = min(n_bins, c_bin + half_bw_bins + 1)
            peak_p = float(np.max(mp_np[lo_b:hi_b])) if hi_b > lo_b else 1e-10
            q_obs[k - 2] = max(peak_p / max(lip_p, 1e-4), 1e-10)

        q_sum = float(np.sum(q_obs))
        if q_sum > 1e-10:
            q_norm = np.maximum(q_obs / q_sum, 1e-8)
            rd_grid, _, p_templates, _ = _get_lf_glottal_table()
            tpl = p_templates[:, 1:k_est].astype(np.float64)
            tpl_norm = np.maximum(tpl / np.maximum(np.sum(tpl, axis=1, keepdims=True), 1e-12), 1e-8)
            ratio = q_norm[np.newaxis, :] / tpl_norm
            is_div = np.mean(ratio - np.log(ratio) - 1.0, axis=1)
            rd_is = float(rd_grid[int(np.argmin(is_div))])

    # Map presence ratio [-24 dB (lax/mellow) .. +8 dB (sharp/edge)] smoothly to [1.32 .. 0.45]
    u_bright = float(np.clip((pres_ratio_db + 24.0) / 32.0, 0.0, 1.0))
    rd_bright = 1.32 - 0.87 * u_bright
    rd_blend = 0.65 * rd_bright + 0.35 * float(np.clip(rd_is, 0.40, 1.60))
    return float(np.clip(rd_blend, 0.45, 1.35))


def _extract_cepstral_formant_envelope(
    mean_power: torch.Tensor, sr: int, n_fft: int
) -> torch.Tensor:
    """
    Extract zero-macroscopic-tilt vocal-tract formant log-envelope (quefrency q in [3, sr/1150])
    used both for formant-free baseline Rd0 estimation and formant plateau protection.
    """
    if float(torch.sum(mean_power).item()) < 1e-10:
        return torch.zeros_like(mean_power)

    log_spec_mean = 0.5 * torch.log(torch.clamp(mean_power, min=1e-12))
    cep = torch.fft.irfft(log_spec_mean, n=n_fft)

    q_low = 3
    q_high = min(n_fft // 2 - 1, max(12, int(sr / 1150.0)))
    cep_bp = torch.zeros_like(cep)
    cep_bp[q_low:q_high] = cep[q_low:q_high]
    cep_bp[-q_high + 1:-q_low + 1] = cep[-q_high + 1:-q_low + 1]
    return torch.fft.rfft(cep_bp, n=n_fft).real


def _apply_formant_plateau_protection(
    log_gain: torch.Tensor,
    formant_prominence: torch.Tensor,
    freqs: torch.Tensor,
    sr: int,
    n_fft: int,
    alpha_prot: float = 0.50,
) -> torch.Tensor:
    """
    Formant-Preserving Slope-Damped Plateau:
    Damps the local derivative d(log_gain)/df across vowel formant peaks (F1, F2, F3)
    in [220 Hz, 3600 Hz] while preserving 100% of the macroscopic LF glottal spectral tilt.
    """
    vowel_band = (freqs >= 220.0) & (freqs <= 3600.0)
    pos_prom = torch.relu(formant_prominence) * vowel_band.to(dtype=log_gain.dtype)
    max_prom = torch.max(pos_prom)
    if float(max_prom.item()) < 1e-5:
        return log_gain

    formant_mask = torch.clamp(pos_prom / (max_prom + 1e-6), 0.0, 1.0)

    dg = log_gain[1:] - log_gain[:-1]
    mask_mid = 0.5 * (formant_mask[1:] + formant_mask[:-1])
    low_idx = min(log_gain.numel() - 2, max(1, int(200.0 * n_fft / sr)))
    ref_idx = min(log_gain.numel() - 1, max(low_idx + 2, int(4500.0 * n_fft / sr)))

    dg_damped = dg.clone()
    dg_damped[low_idx:ref_idx] = dg[low_idx:ref_idx] * (1.0 - alpha_prot * mask_mid[low_idx:ref_idx])

    orig_span = float(torch.sum(dg[low_idx:ref_idx]).item())
    damped_span = float(torch.sum(dg_damped[low_idx:ref_idx]).item())
    if abs(damped_span) > 1e-5:
        span_scale = float(np.clip(orig_span / damped_span, 0.5, 2.0))
        dg_damped[low_idx:ref_idx] = dg_damped[low_idx:ref_idx] * span_scale

    log_gain_prot = torch.zeros_like(log_gain)
    log_gain_prot[0] = log_gain[0]
    log_gain_prot[1:] = log_gain[0] + torch.cumsum(dg_damped, dim=0)
    return log_gain_prot


def _interp_lf_log_spectrum(rd_val: float, f0_src: float, freqs_np: np.ndarray) -> np.ndarray:
    """
    Evaluate continuous ripple-free LF log-glottal spectrum ln(g_tilde(f / f0_src; Rd))
    anchored strictly at 0.0 dB for f <= f0_src (H1 pivot invariance).
    """
    rd_grid, log_table, _, _ = _get_lf_glottal_table()
    pos = float(np.clip((rd_val - rd_grid[0]) / (rd_grid[1] - rd_grid[0]), 0.0, len(rd_grid) - 1.0001))
    i0 = int(math.floor(pos))
    frac = pos - i0
    row = (1.0 - frac) * log_table[i0] + frac * log_table[i0 + 1]  # [K_max]

    ks = np.arange(1, len(row) + 1, dtype=np.float32)
    nu = np.maximum(freqs_np / max(f0_src, 1.0), 1.0)
    return np.interp(nu, ks, row).astype(np.float32)


def pre_emphasis_base_tension(wave, b):
    """
    Clean-room v3 minimum-phase LF glottal tension engine:
    1. Adaptive source-f0 & Itakura-Saito baseline Rd0 estimation with sharp-voice protection;
    2. Exact closed-form continuous-time LF harmonic ratio anchored strictly at H1 (f = f0_src);
    3. Bidirectional high-frequency counter-damping ramp (+/- 0.8 Np = +/- 6.95 dB over 8.5-11.5 kHz);
    4. Cepstral formant plateau protection (F1/F2/F3 vowel invariance) & Hilbert minimum-phase;
    5. Psychoacoustic middle-ear loudness preservation W(f) clamped safely in [0.45, 2.0].

    Args:
        wave: [1, 1, t] periodic harmonic waveform tensor
        b: float tension parameter (-Ht / 50, in [-2.0, 2.0]; b < 0 is higher tension)
    """
    if abs(float(b)) < 1e-6:
        return wave

    orig_dtype = wave.dtype
    wave = wave.float()
    original_length = wave.size(-1)
    pad_length = (CONFIG.hop_size - (original_length % CONFIG.hop_size)) % CONFIG.hop_size
    min_stft_len = CONFIG.n_fft // 2 + 1
    if original_length + pad_length < min_stft_len:
        pad_length = min_stft_len - original_length
    wave = torch.nn.functional.pad(wave, (0, pad_length), mode='constant', value=0)
    wave = wave.squeeze(1)

    window, _, _, freqs, ear_weight = _get_pre_emphasis_tensors(wave.device)
    spec = torch.stft(
        wave,
        CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window,
        return_complex=True,
    )

    b_clamped = float(np.clip(b, -2.0, 2.0))
    sr = int(CONFIG.sample_rate)
    n_fft = int(CONFIG.n_fft)

    # 1. Voiced-energy-weighted mean power spectrum, adaptive f0_src, and baseline Rd0
    power_spec = spec.real ** 2 + spec.imag ** 2  # [1, F, T]
    frame_energy = torch.sum(power_spec, dim=1, keepdim=True)  # [1, 1, T]
    total_energy = float(torch.sum(frame_energy).item())
    if total_energy <= 1e-12:
        return torch.zeros_like(wave).unsqueeze(1)[:, :, :original_length].to(dtype=orig_dtype)

    mean_power = torch.sum(power_spec * frame_energy, dim=2).squeeze(0) / (total_energy + 1e-12)
    f0_src = _estimate_harmonic_f0(mean_power, sr, n_fft)
    formant_prominence = _extract_cepstral_formant_envelope(mean_power, sr, n_fft)
    rd0 = _estimate_baseline_rd(mean_power, formant_prominence, freqs, f0_src, sr, n_fft)

    # 2. Perceptual arc-length Rd warping with soft-knee rail protection on sharp/intense voices
    rd_grid, _, _, psi_grid = _get_lf_glottal_table()

    if b_clamped < 0.0:
        # Positive tension (Ht > 0, b in [-2, 0]): map smoothly down to voice-adaptive tense floor
        # Gives mellow voices (psi0 ~ 0.72) full ~13 dB upward travel while protecting sharp voices (psi0 ~ 0.38)
        rd_base = rd0
        psi0 = float(np.interp(rd_base, rd_grid, psi_grid))
        u = -b_clamped / 2.0
        psi_floor = min(psi0, 0.042 + 0.030 * max(0.0, 0.55 - psi0))
        psi_target = psi_floor + (psi0 - psi_floor) * (1.0 - math.tanh(1.75 * u) / math.tanh(1.75))
        tilt_scale = 1.28
    else:
        # Negative tension (Ht < 0, b in (0, +2]): guarantee full downward softening excursion
        # even on mellow voices by capping rd_base <= 0.92 while preserving wider travel on sharp voices
        rd_base = min(rd0, 0.92)
        psi0 = float(np.interp(rd_base, rd_grid, psi_grid))
        drive = (b_clamped / 2.0) * 1.65
        psi_target = psi0 + (1.0 - psi0) * math.tanh(drive)
        tilt_scale = 1.48

    rd_target = float(np.interp(psi_target, psi_grid, rd_grid))

    # 3. Closed-form LF harmonic log-ratio (strictly 0 dB at and below H1 = f0_src)
    freqs_np = freqs.detach().cpu().numpy()
    log_g0 = _interp_lf_log_spectrum(rd_base, f0_src, freqs_np)
    log_gt = _interp_lf_log_spectrum(rd_target, f0_src, freqs_np)
    raw_delta_np = tilt_scale * (log_gt - log_g0)
    log_gain = torch.from_numpy(raw_delta_np).to(device=wave.device, dtype=torch.float32)

    # 4. Bidirectional high-frequency counter-damping ramp (+/- 0.8 Np = +/- 6.95 dB over [8.2k, 11.5k] Hz)
    # Prevents >9 kHz sibilant harshness on Ht > 0 AND preserves vocal "air" on Ht < 0
    hf_ramp = torch.clamp((freqs - 8200.0) / 3300.0, min=0.0, max=1.0)
    hf_counter = torch.clamp(-hf_ramp * log_gain, min=-0.80, max=0.80)
    log_gain = log_gain + hf_counter

    # 5. Formant-preserving slope-damped plateau across vowel formants F1/F2/F3
    log_gain = _apply_formant_plateau_protection(
        log_gain, formant_prominence, freqs, sr, n_fft, alpha_prot=0.50
    )

    # 6. Minimum-phase complex transfer function via causal real cepstrum folding
    cepstrum = torch.fft.irfft(log_gain, n=n_fft)
    cep_min = torch.zeros_like(cepstrum)
    half_n = n_fft // 2
    cep_min[0] = cepstrum[0]
    cep_min[1:half_n] = 2.0 * cepstrum[1:half_n]
    cep_min[half_n] = cepstrum[half_n]
    gain = torch.exp(torch.fft.rfft(cep_min, n=n_fft)).unsqueeze(0).unsqueeze(2)

    filtered_wave = torch.istft(
        spec * gain,
        n_fft=n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window,
        length=wave.size(-1),
    )

    # 7. Psychoacoustic middle-ear loudness preservation W(f) with full downward attenuation headroom
    # Prevents +8 dB clipping when UTAU P-norm is inactive while lifting H1 chest warmth on negative Ht
    mag_sq = torch.exp(2.0 * torch.clamp(log_gain, min=-8.0, max=8.0))
    l_orig = float(torch.sum(mean_power * ear_weight).item())
    l_mod = float(torch.sum(mean_power * mag_sq * ear_weight).item())
    orig_rms = float(torch.sqrt(torch.mean(wave ** 2)).item())
    filtered_rms = float(torch.sqrt(torch.mean(filtered_wave ** 2)).item())

    if filtered_rms > 1e-8 and l_mod > 1e-18 and l_orig > 1e-18:
        g_loud_w = math.sqrt(l_orig / l_mod)
        g_rms = orig_rms / filtered_rms
        if b_clamped < 0.0:
            g_loud = min(0.78 * g_loud_w + 0.22 * g_rms, 1.18 * g_rms)
            expressive_trim = float(10.0 ** (-0.30 * b_clamped / 20.0))
        else:
            g_loud = min(1.85, 0.72 * g_loud_w + 0.28 * g_rms, 1.22 * g_rms)
            expressive_trim = 1.0
        filtered_wave = filtered_wave * (g_loud * expressive_trim)
    else:
        filtered_wave = torch.zeros_like(filtered_wave)

    filtered_wave = filtered_wave.unsqueeze(1)[:, :, :original_length]
    return filtered_wave.to(dtype=orig_dtype)


def couple_aperiodic_to_tension(
    aperiodic_wave: torch.Tensor,
    orig_voiced_wave: torch.Tensor,
    mod_voiced_wave: torch.Tensor,
    b: float,
) -> torch.Tensor:
    """
    Clean-room voiced aspiration noise coupling with 5-frame boundary anchor stabilization:
    - Leaves unvoiced consonants (s, sh, t, k, ch) and silence 100% bit-identical (voiced_mask == 0);
    - Smoothly couples high-frequency (> 1.6 kHz) voiced aspiration turbulence inside vowels
      to track harmonic spectral energy changes induced by Ht, eliminating exposed hiss on soft Ht < 0
      and matching glottal turbulence on bright Ht > 0.
    """
    if abs(float(b)) < 1e-6:
        return aperiodic_wave

    orig_dtype = aperiodic_wave.dtype
    n_samples = aperiodic_wave.size(-1)
    if n_samples == 0:
        return aperiodic_wave

    min_stft_len = CONFIG.n_fft // 2 + 1
    ap = aperiodic_wave.float().squeeze(1)
    v_orig = orig_voiced_wave.float().squeeze(1)
    v_mod = mod_voiced_wave.float().squeeze(1)

    pad_len = (CONFIG.hop_size - (n_samples % CONFIG.hop_size)) % CONFIG.hop_size
    if n_samples + pad_len < min_stft_len:
        pad_len = min_stft_len - n_samples
    if pad_len > 0:
        ap_pad = torch.nn.functional.pad(ap, (0, pad_len))
        vo_pad = torch.nn.functional.pad(v_orig, (0, pad_len))
        vm_pad = torch.nn.functional.pad(v_mod, (0, pad_len))
    else:
        ap_pad, vo_pad, vm_pad = ap, v_orig, v_mod

    window, _, _, freqs, _ = _get_pre_emphasis_tensors(aperiodic_wave.device)
    spec_ap = torch.stft(ap_pad, CONFIG.n_fft, hop_length=CONFIG.hop_size, win_length=CONFIG.win_size, window=window, return_complex=True)
    spec_vo = torch.stft(vo_pad, CONFIG.n_fft, hop_length=CONFIG.hop_size, win_length=CONFIG.win_size, window=window, return_complex=True)
    spec_vm = torch.stft(vm_pad, CONFIG.n_fft, hop_length=CONFIG.hop_size, win_length=CONFIG.win_size, window=window, return_complex=True)

    p_ap = torch.sum(spec_ap.real ** 2 + spec_ap.imag ** 2, dim=1).squeeze(0)  # [T_f]
    p_vo_bins = (spec_vo.real ** 2 + spec_vo.imag ** 2).squeeze(0)  # [F, T_f]
    p_vm_bins = (spec_vm.real ** 2 + spec_vm.imag ** 2).squeeze(0)  # [F, T_f]
    p_vo = torch.sum(p_vo_bins, dim=0)  # [T_f]

    max_vo = float(torch.max(p_vo).item())
    if max_vo < 1e-10:
        return aperiodic_wave

    # Voiced frame detector: strong harmonic energy AND harmonic dominance over fricative noise
    harm_ratio = p_vo / (p_vo + p_ap + 1e-12)
    rel_level = p_vo / (max_vo + 1e-12)
    voiced_bool = ((rel_level > 3e-4) & (harm_ratio > 0.55)).detach().cpu().numpy()
    n_frames = len(voiced_bool)
    if not np.any(voiced_bool):
        return aperiodic_wave

    # Compute high-frequency (1.6 kHz - 9.0 kHz) harmonic log-amplitude shift per frame
    hf_mask = ((freqs >= 1600.0) & (freqs <= 9000.0)).to(dtype=torch.float32).unsqueeze(1)
    eh_orig = torch.sum(p_vo_bins * hf_mask, dim=0).detach().cpu().numpy()
    eh_mod = torch.sum(p_vm_bins * hf_mask, dim=0).detach().cpu().numpy()
    delta_l_frames = 0.5 * np.log((eh_mod + 1e-12) / (eh_orig + 1e-12))

    # Apply 5-frame onset/offset boundary anchor stabilization on contiguous voiced intervals
    stabilized_delta = np.zeros(n_frames, dtype=np.float32)
    voiced_weight = np.zeros(n_frames, dtype=np.float32)
    idx = 0
    while idx < n_frames:
        if not voiced_bool[idx]:
            idx += 1
            continue
        end_idx = idx + 1
        while end_idx < n_frames and voiced_bool[end_idx]:
            end_idx += 1
        seg_len = end_idx - idx
        margin = 0 if seg_len < 4 else (seg_len // 3 if seg_len <= 10 else 5)
        n_l = idx + margin
        n_r = end_idx - 1 - margin
        stabilized_delta[idx:end_idx] = delta_l_frames[idx:end_idx]
        if margin > 0 and n_l <= n_r:
            stabilized_delta[idx:n_l] = delta_l_frames[n_l]
            stabilized_delta[n_r + 1:end_idx] = delta_l_frames[n_r]
        voiced_weight[idx:end_idx] = 1.0
        idx = end_idx

    # Smooth voiced_weight with a 3-tap kernel strictly inside voiced support so unvoiced stays 0.0
    if n_frames >= 3:
        w_smooth = np.convolve(voiced_weight, np.array([0.25, 0.50, 0.25], dtype=np.float32), mode='same')
        voiced_weight = np.where(voiced_bool, w_smooth, 0.0)

    coupled_log = np.clip(0.48 * stabilized_delta, -0.95, 0.62) * voiced_weight  # [-8.25 dB, +5.39 dB]
    coupled_log_t = torch.from_numpy(coupled_log).to(device=aperiodic_wave.device, dtype=torch.float32)  # [T_f]

    # Zero-phase smooth high-shelf (> 1.6 kHz) on stochastic noise avoids any comb phase cancellation
    s_hf = ((freqs / 1600.0) ** 2) / (1.0 + (freqs / 1600.0) ** 2)  # [F]
    freq_profile = (0.22 + 0.78 * s_hf).unsqueeze(1)  # [F, 1]
    noise_gain = torch.exp(freq_profile * coupled_log_t.unsqueeze(0)).unsqueeze(0)  # [1, F, T_f]

    ap_filtered = torch.istft(
        spec_ap * noise_gain,
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=window,
        length=ap_pad.size(-1),
    )[:, :n_samples]

    # Interpolate voiced_weight to sample level so unvoiced consonants & silence are 100% bit-identical
    frame_centers = np.arange(n_frames, dtype=np.float32) * float(CONFIG.hop_size)
    sample_idx = np.arange(n_samples, dtype=np.float32)
    sample_mask_np = np.interp(sample_idx, frame_centers, voiced_weight, left=0.0, right=0.0).astype(np.float32)
    sample_mask = torch.from_numpy(sample_mask_np).to(device=aperiodic_wave.device, dtype=torch.float32).unsqueeze(0)

    ap_orig = ap[:, :n_samples]
    ap_out = ap_orig + sample_mask * (ap_filtered - ap_orig)
    return ap_out.unsqueeze(1).to(dtype=orig_dtype)


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
