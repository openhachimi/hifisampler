"""Lightweight ITU-R BS.1770-4 LUFS measurement, envelope-aware gain protection,
and polyphase resampling utilities with zero ``pyloudnorm`` or ``resampy`` dependencies.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
from scipy.signal import resample_poly, sosfilt

# ITU-R BS.1770-4 / Brecht DeMan exact K-weighting prototype parameters.
_HS_FC = 1681.9744509555319
_HS_GAIN_DB = 3.99984385397
_HS_Q = 0.7071752369554193

_HP_FC = 38.13547087613982
_HP_Q = 0.5003270373253953

# Precomputed exact 44100 Hz ITU-R BS.1770-4 K-weighting SOS matrix.
_K_WEIGHT_SOS_44100 = (
    (
        1.5308412300498355,
        -2.6509799951536985,
        1.1690790799210682,
        1.0,
        -1.6636551132560204,
        0.7125954280732254,
    ),
    (
        1.0,
        -2.0,
        1.0,
        1.0,
        -1.9891696736297957,
        0.9891990357870394,
    ),
)

# ITU-R BS.1770-4 channel weights [L, R, C, Ls, Rs].
_BS1770_CHANNEL_WEIGHTS = (1.0, 1.0, 1.0, 1.41, 1.41)
_ABS_GATE_LUFS = -70.0
_ABS_GATE_POWER = 10.0 ** ((_ABS_GATE_LUFS + 0.691) / 10.0)
_REL_GATE_RATIO = 10.0 ** (-10.0 / 10.0)  # -10 LU


@lru_cache(maxsize=64)
def _k_weighting_sos(sample_rate: int) -> np.ndarray:
    """Return cached (2, 6) float64 SOS coefficients for ITU-R BS.1770-4 K-weighting."""
    sr = int(sample_rate)
    if sr <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if sr == 44100:
        return np.asarray(_K_WEIGHT_SOS_44100, dtype=np.float64)
    if sr <= int(2 * _HS_FC):
        raise ValueError(
            f"sample_rate={sr} Hz is too low for K-weighting high-shelf at {_HS_FC:.1f} Hz"
        )

    # Stage 1: head-related high-shelf via exact prewarped bilinear transform.
    k_hs = math.tan(math.pi * _HS_FC / sr)
    vh = 10.0 ** (_HS_GAIN_DB / 20.0)
    vb = vh ** 0.499666774155
    a0_hs = 1.0 + k_hs / _HS_Q + k_hs * k_hs
    b0_hs = (vh + vb * k_hs / _HS_Q + k_hs * k_hs) / a0_hs
    b1_hs = 2.0 * (k_hs * k_hs - vh) / a0_hs
    b2_hs = (vh - vb * k_hs / _HS_Q + k_hs * k_hs) / a0_hs
    a1_hs = 2.0 * (k_hs * k_hs - 1.0) / a0_hs
    a2_hs = (1.0 - k_hs / _HS_Q + k_hs * k_hs) / a0_hs

    # Stage 2: RLB high-pass via exact prewarped bilinear transform.
    k_hp = math.tan(math.pi * _HP_FC / sr)
    a0_hp = 1.0 + k_hp / _HP_Q + k_hp * k_hp
    a1_hp = 2.0 * (k_hp * k_hp - 1.0) / a0_hp
    a2_hp = (1.0 - k_hp / _HP_Q + k_hp * k_hp) / a0_hp

    sos = np.asarray(
        [
            [b0_hs, b1_hs, b2_hs, 1.0, a1_hs, a2_hs],
            [1.0, -2.0, 1.0, 1.0, a1_hp, a2_hp],
        ],
        dtype=np.float64,
    )
    return sos


def _channel_weights(num_channels: int) -> np.ndarray:
    """Return ITU-R BS.1770-4 channel weighting factors."""
    if num_channels <= len(_BS1770_CHANNEL_WEIGHTS):
        return np.asarray(_BS1770_CHANNEL_WEIGHTS[:num_channels], dtype=np.float64)
    weights = np.ones(num_channels, dtype=np.float64)
    weights[: len(_BS1770_CHANNEL_WEIGHTS)] = _BS1770_CHANNEL_WEIGHTS
    return weights


def measure_lufs(audio: np.ndarray, sample_rate: int, block_size: float = 0.400) -> float:
    """Measure integrated gated loudness in LUFS according to ITU-R BS.1770-4.

    Supports mono ``(N,)`` and multichannel ``(N, C)`` inputs. Returns ``-inf``
    for empty, silent, or fully gated signals.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if block_size <= 0.0:
        raise ValueError(f"block_size must be positive, got {block_size!r}")

    data = np.asarray(audio, dtype=np.float64)
    if data.size == 0:
        return float("-inf")
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {data.shape}")

    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    sos = _k_weighting_sos(int(sample_rate))
    weighted = sosfilt(sos, data, axis=0)
    weights = _channel_weights(weighted.shape[1])

    # Instantaneous weighted power per sample across channels.
    sq_weighted = np.square(weighted) @ weights
    num_samples = sq_weighted.shape[0]
    block_samples = max(1, int(round(sample_rate * block_size)))
    hop_samples = max(1, block_samples // 4)  # 75% overlap (100 ms hop for 400 ms block)

    if num_samples < block_samples:
        mean_sq = float(np.mean(sq_weighted))
        if mean_sq < 1e-20:
            return float("-inf")
        return float(-0.691 + 10.0 * math.log10(mean_sq))

    cs = np.empty(num_samples + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(sq_weighted, out=cs[1:])

    starts = np.arange(0, num_samples - block_samples + 1, hop_samples, dtype=np.int64)
    block_powers = (cs[starts + block_samples] - cs[starts]) / float(block_samples)

    # Absolute gate: >= -70 LUFS
    abs_mask = block_powers >= _ABS_GATE_POWER
    if not np.any(abs_mask):
        return float("-inf")

    mean_abs = float(np.mean(block_powers[abs_mask]))
    rel_gate = mean_abs * _REL_GATE_RATIO

    # Relative gate: > (abs-gated mean - 10 LU) and >= -70 LUFS
    rel_mask = abs_mask & (block_powers > rel_gate)
    if not np.any(rel_mask):
        return float("-inf")

    mean_rel = float(np.mean(block_powers[rel_mask]))
    if mean_rel <= 0.0:
        return float("-inf")
    return float(-0.691 + 10.0 * math.log10(mean_rel))


def apply_envelope_aware_gain(
    audio: np.ndarray,
    target_gain: float,
    sample_rate: int,
    low_level_protect_db: float = -40.0,
    tail_peak_limit_dbfs: float = -6.0,
    tail_mode: str = "preserve_relative",
    max_boost_db: float = 18.0,
) -> np.ndarray:
    """Apply gain with envelope-aware tail protection to avoid boosting breath tails.

    Uses 20 ms frame RMS (10 ms hop), p90 of non-silent frames (>-60 dB) for
    sustained-level estimation, 50 ms moving-average envelope smoothing, and
    low-level tail peak limiting.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

    audio_f32 = np.asarray(audio, dtype=np.float32)
    if audio_f32.size == 0:
        return audio_f32.copy()

    gain = max(0.0, float(target_gain))
    if max_boost_db is not None:
        gain = min(gain, 10.0 ** (float(max_boost_db) / 20.0))

    if abs(gain - 1.0) < 0.01:
        return audio_f32.copy()

    mode = str(tail_mode).strip().lower()
    if mode == "preserve_relative":
        fade_start_db, fade_end_db, tail_gain_floor_ratio = 8.0, 24.0, 0.45
    elif mode in ("clip", "legacy_protect"):
        fade_start_db, fade_end_db, tail_gain_floor_ratio = 10.0, 20.0, 0.0
    else:
        raise ValueError(
            f"Unsupported tail_mode {tail_mode!r}; expected 'preserve_relative' or 'clip'"
        )

    n = audio_f32.shape[0]
    frame_len = int(sample_rate * 0.02)
    hop_len = int(sample_rate * 0.01)
    if frame_len <= 0 or hop_len <= 0 or n < frame_len:
        return (audio_f32 * np.float32(gain)).astype(np.float32, copy=False)

    # Frame-level RMS in dB via prefix sums.
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(np.square(audio_f32, dtype=np.float64), out=cs[1:])
    starts = np.arange(0, n - frame_len + 1, hop_len, dtype=np.int64)
    if starts.size == 0:
        return (audio_f32 * np.float32(gain)).astype(np.float32, copy=False)

    rms = np.sqrt(np.maximum(0.0, (cs[starts + frame_len] - cs[starts]) / float(frame_len)))
    frame_rms = np.where(rms < 1e-10, -100.0, 20.0 * np.log10(np.maximum(rms, 1e-10)))

    non_silent = frame_rms[frame_rms > -60.0]
    if non_silent.size == 0:
        return (audio_f32 * np.float32(gain)).astype(np.float32, copy=False)

    non_silent_sorted = np.sort(non_silent)
    p90_idx = min(int(non_silent_sorted.size * 0.9), non_silent_sorted.size - 1)
    sustained_db = float(non_silent_sorted[p90_idx])

    threshold_high = sustained_db - fade_start_db
    threshold_low = sustained_db - fade_end_db
    db_range = max(1e-12, threshold_high - threshold_low)

    relative = np.clip((frame_rms - threshold_low) / db_range, 0.0, 1.0)
    blend = tail_gain_floor_ratio + (1.0 - tail_gain_floor_ratio) * relative
    frame_gains = 1.0 + (gain - 1.0) * blend

    if gain > 1.0:
        protected_gain = 1.0 + (gain - 1.0) * tail_gain_floor_ratio
        frame_gains = np.where(
            frame_rms <= low_level_protect_db,
            np.minimum(frame_gains, protected_gain),
            frame_gains,
        )

    # 50 ms centered moving-average smoothing over frame gains.
    smooth_frames = max(1, int(round((sample_rate * 0.05) / float(hop_len))))
    if smooth_frames > 1 and frame_gains.size > smooth_frames:
        half = smooth_frames // 2
        fg_cs = np.empty(frame_gains.size + 1, dtype=np.float64)
        fg_cs[0] = 0.0
        np.cumsum(frame_gains, out=fg_cs[1:])
        idx = np.arange(frame_gains.size, dtype=np.int64)
        lo = np.maximum(0, idx - half)
        hi = np.minimum(frame_gains.size, idx + half + 1)
        smoothed_gains = (fg_cs[hi] - fg_cs[lo]) / (hi - lo)
    else:
        smoothed_gains = frame_gains

    # Linearly interpolate frame gains to per-sample gains.
    sample_pos = np.arange(n, dtype=np.float64) / float(hop_len)
    last_frame = smoothed_gains.size - 1
    frame_idx = np.minimum(sample_pos.astype(np.int64), last_frame)
    next_idx = np.minimum(frame_idx + 1, last_frame)
    frac = np.where(frame_idx < last_frame, sample_pos - frame_idx, 0.0)

    sample_gains = smoothed_gains[frame_idx] * (1.0 - frac) + smoothed_gains[next_idx] * frac
    output = audio_f32.astype(np.float64) * sample_gains

    if gain > 1.0:
        tail_peak_limit = 10.0 ** (float(tail_peak_limit_dbfs) / 20.0)
        if tail_peak_limit > 0.0:
            sample_db = frame_rms[frame_idx] * (1.0 - frac) + frame_rms[next_idx] * frac
            low_mask = sample_db <= low_level_protect_db
            if np.any(low_mask):
                # Smooth gain clamping on protected low-level regions instead of hard sample clipping
                low_orig = np.abs(audio_f32[low_mask].astype(np.float64))
                safe_gains = np.where(
                    low_orig * sample_gains[low_mask] > tail_peak_limit,
                    tail_peak_limit / np.maximum(low_orig, 1e-12),
                    sample_gains[low_mask],
                )
                output[low_mask] = audio_f32[low_mask].astype(np.float64) * safe_gains

    return output.astype(np.float32, copy=False)


def _find_voiced_span(
    audio: np.ndarray,
    rate: int,
    silence_threshold: float,
) -> tuple[int, int] | None:
    """Return ``(start_sample, end_sample)`` of voiced audio with 100 ms tail padding,
    or ``None`` if no frames exceed ``silence_threshold``.
    """
    frame_length = int(rate * 0.02)
    hop_length = int(rate * 0.01)
    if frame_length <= 0 or hop_length <= 0 or len(audio) < frame_length:
        return (0, len(audio))

    num_frames = max(0, (len(audio) - frame_length) // hop_length + 1)
    if num_frames <= 0:
        return (0, len(audio))

    cs = np.empty(len(audio) + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(np.square(audio, dtype=np.float64), out=cs[1:])
    starts = np.arange(num_frames, dtype=np.int64) * hop_length
    mean_sq = (cs[starts + frame_length] - cs[starts]) / float(frame_length)
    thresh_sq = max(1e-20, 10.0 ** (float(silence_threshold) / 10.0))
    voiced = np.flatnonzero(mean_sq > thresh_sq)
    if voiced.size == 0:
        return None

    first_voiced = int(voiced[0])
    last_voiced = int(voiced[-1])
    padding_frames = int(rate * 0.1) // hop_length
    start_sample = max(0, first_voiced * hop_length)
    end_sample = min(len(audio), (last_voiced + 1 + padding_frames) * hop_length + frame_length)
    return (start_sample, end_sample)


def loudness_norm_fast(
    audio: np.ndarray,
    rate: int,
    peak: float = -1.0,
    loudness: float = -23.0,
    block_size: float = 0.400,
    strength: float = 100,
    trim_silence: bool = True,
    silence_threshold: float = -52.0,
    max_boost_db: float = 18.0,
    low_level_protect_db: float = -40.0,
    tail_peak_limit_dbfs: float = -6.0,
    tail_mode: str = "preserve_relative",
) -> np.ndarray:
    """Fast BS.1770-4 loudness normalization with optional silence trimming and envelope protection."""
    del peak  # Preserved for signature compatibility with util.audio.loudness_norm.

    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate!r}")

    original = np.asarray(audio, dtype=np.float32)
    original_length = original.shape[0]
    if original_length == 0:
        return original.copy()

    if not np.all(np.isfinite(original)):
        original = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    peak_abs = float(np.max(np.abs(original)))
    if peak_abs < 1e-8 or float(strength) < 0.01:
        return original.copy()

    span: tuple[int, int] | None = None
    work_audio = original
    if trim_silence:
        span = _find_voiced_span(original, rate, silence_threshold)
        if span is None:
            return original.copy()
        start_sample, end_sample = span
        work_audio = original[start_sample:end_sample]
        if work_audio.size == 0 or float(np.max(np.abs(work_audio))) < 1e-8:
            return original.copy()

    min_samples = max(1, int(rate * block_size))
    if len(work_audio) < min_samples:
        pad_len = min_samples - len(work_audio)
        pad_mode = "reflect" if len(work_audio) > 1 else "edge"
        measure_audio = np.pad(work_audio, (0, pad_len), mode=pad_mode)
    else:
        measure_audio = work_audio

    current_lufs = measure_lufs(measure_audio, rate, block_size=block_size)
    if not math.isfinite(current_lufs):
        return original.copy()

    final_loudness = current_lufs + (float(loudness) - current_lufs) * float(strength) / 100.0
    gain_db = final_loudness - current_lufs
    if max_boost_db is not None:
        gain_db = min(gain_db, float(max_boost_db))
    target_gain = 10.0 ** (gain_db / 20.0)

    normalized = apply_envelope_aware_gain(
        work_audio,
        target_gain=target_gain,
        sample_rate=rate,
        low_level_protect_db=low_level_protect_db,
        tail_peak_limit_dbfs=tail_peak_limit_dbfs,
        tail_mode=tail_mode,
        max_boost_db=max_boost_db,
    )

    if trim_silence and span is not None:
        start_sample, _ = span
        output = original.copy()
        available_length = min(len(normalized), original_length - start_sample)
        fade_length = min(int(rate * 0.2), available_length // 4)

        if available_length > 0:
            lead_xfade = min(int(rate * 0.005), start_sample, available_length // 4)
            blend_weights = np.ones(available_length, dtype=np.float32)
            if lead_xfade > 0:
                blend_weights[:lead_xfade] = np.linspace(0.0, 1.0, lead_xfade, dtype=np.float32)
            if fade_length > 0 and (start_sample + available_length) < original_length:
                blend_weights[-fade_length:] = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
            orig_slice = original[start_sample : start_sample + available_length]
            output[start_sample : start_sample + available_length] = (
                normalized[:available_length] * blend_weights + orig_slice * (1.0 - blend_weights)
            )

        return output

    return normalized[:original_length].astype(np.float32, copy=False)


def resample_poly_fast(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality polyphase resampling returning float32 with exact expected length."""
    src_sr = int(orig_sr)
    dst_sr = int(target_sr)
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(f"Sample rates must be positive, got orig_sr={orig_sr}, target_sr={target_sr}")

    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[0] == 0 or src_sr == dst_sr:
        return arr.copy()

    expected_len = int(round(arr.shape[0] * dst_sr / src_sr))
    if expected_len <= 0:
        return np.zeros((0,) + arr.shape[1:], dtype=np.float32)

    g = math.gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g

    if max(up, down) > 1000:
        frac = Fraction(dst_sr, src_sr).limit_denominator(512)
        up = frac.numerator
        down = frac.denominator

    y = resample_poly(arr, up, down, axis=0, padtype="line").astype(np.float32, copy=False)

    if y.shape[0] > expected_len:
        y = y[:expected_len]
    elif y.shape[0] < expected_len:
        pad_width = [(0, expected_len - y.shape[0])] + [(0, 0)] * (y.ndim - 1)
        y = np.pad(y, pad_width, mode="constant")

    return y.astype(np.float32, copy=False)
