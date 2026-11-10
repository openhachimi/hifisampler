from functools import lru_cache
import numpy as np
from scipy.signal import butter, sosfilt

_MAX_VIBRATO_CENTS = 100.0
_HP_CUTOFF_HZ      = 20.0
_MIN_NYQ_FRAC      = 0.01


@lru_cache(maxsize=64)
def _get_highpass_sos(order: int, norm_freq: float):
    return butter(order, norm_freq, "high", output="sos")


def _highpass(audio: np.ndarray, sr: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    nyq   = sr / 2
    norm  = float(np.clip(cutoff / nyq, _MIN_NYQ_FRAC, 0.99))
    sos   = _get_highpass_sos(4, round(norm, 6))
    high  = np.asarray(sosfilt(sos, audio), dtype=audio.dtype)
    return high, audio - high


def _square_lfo(num_samples: int, sr: int, freq: float) -> np.ndarray:
    half_cycles = (np.arange(num_samples, dtype=np.float32) * np.float32(2.0 * freq / sr)).astype(np.int32)
    return (1 - ((half_cycles & 1) << 1)).astype(np.float32)


def _apply_pitch_modulation(band: np.ndarray,
                            sr: int,
                            lfo: np.ndarray,
                            strength: float) -> np.ndarray:
    n_samples = len(band)
    exp_arg = float(strength * _MAX_VIBRATO_CENTS / 1200.0)
    r_pos = 2.0 ** exp_arg
    r_neg = 2.0 ** (-exp_arg)
    ratio = (0.5 * (r_pos + r_neg)) + lfo * (0.5 * (r_pos - r_neg))
    cumsum  = np.cumsum(ratio, dtype=np.float64) - float(ratio[0])
    arange_idx = np.arange(n_samples, dtype=np.float64)
    ideal   = arange_idx * np.mean(ratio)
    drift   = cumsum - ideal

    if n_samples > 100:
        hp_norm = round(float(_HP_CUTOFF_HZ / (sr / 2)), 6)
        drift   = sosfilt(_get_highpass_sos(2, hp_norm), drift)

    idx       = np.clip(arange_idx + drift, 0.0, float(n_samples - 1))
    modulated = np.interp(idx, arange_idx, band).astype(band.dtype, copy=False)

    rms_orig  = float(np.sqrt(np.mean(np.square(band, dtype=np.float64))))
    rms_new   = float(np.sqrt(np.mean(np.square(modulated, dtype=np.float64))))
    if rms_new > 1e-10:
        modulated = modulated * (rms_orig / rms_new)
    return modulated


def growl(audio: np.ndarray,
          sample_rate: int,
          *,
          frequency: float = 80.0,
          strength: float = 0.5,
          freq_low: float = 400.0) -> np.ndarray:
    if len(audio) == 0 or strength == 0 or frequency <= 0:
        return audio.copy()

    band, complement = _highpass(audio, sample_rate, freq_low)
    lfo              = _square_lfo(len(audio), sample_rate, frequency)
    band             = _apply_pitch_modulation(band, sample_rate, lfo, strength)

    return complement + band
