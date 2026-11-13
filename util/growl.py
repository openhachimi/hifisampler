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


def _rms_f64(x: np.ndarray) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    acc = 0.0
    step = 16384
    for start in range(0, n, step):
        seg = x[start:start + step].astype(np.float64, copy=False)
        acc += float(np.dot(seg, seg))
    return float(np.sqrt(acc / n))


def _square_lfo(num_samples: int, sr: int, freq: float) -> np.ndarray:
    lfo = np.arange(num_samples, dtype=np.float32)
    lfo *= np.float32(2.0 * freq / sr)
    half_cycles = lfo.astype(np.int32)
    half_cycles &= 1
    lfo[:] = half_cycles
    lfo *= np.float32(-2.0)
    lfo += np.float32(1.0)
    return lfo


def _apply_pitch_modulation(band: np.ndarray,
                            sr: int,
                            lfo: np.ndarray,
                            strength: float) -> np.ndarray:
    n_samples = len(band)
    if n_samples == 0:
        return band.copy()

    exp_arg = float(strength * _MAX_VIBRATO_CENTS / 1200.0)
    r_pos = 2.0 ** exp_arg
    r_neg = 2.0 ** (-exp_arg)
    half_diff = 0.5 * (r_pos - r_neg)

    # Cancel mean ratio analytically so phase drift stays bounded in float64 without extra full-length buffers.
    drift = np.cumsum(lfo, dtype=np.float64)
    drift -= float(lfo[0])
    mean_lfo = float(np.mean(lfo, dtype=np.float64))
    idx = np.arange(n_samples, dtype=np.float64)
    if mean_lfo != 0.0:
        drift -= idx * mean_lfo
    drift *= half_diff

    if n_samples > 100:
        hp_norm = round(float(np.clip(_HP_CUTOFF_HZ / (sr / 2), 1e-6, 0.99)), 6)
        drift = sosfilt(_get_highpass_sos(2, hp_norm), drift)

    idx += drift
    np.clip(idx, 0.0, float(n_samples - 1), out=idx)

    # Uniform-grid linear interpolation directly in band.dtype (O(N) indexing, no float64 output copy).
    i0 = idx.astype(np.intp)
    idx -= i0
    frac = idx.astype(band.dtype, copy=False)
    i1 = np.minimum(i0 + 1, n_samples - 1)
    b0 = band[i0]
    modulated = band[i1]
    modulated -= b0
    modulated *= frac
    modulated += b0

    rms_orig = _rms_f64(band)
    rms_new = _rms_f64(modulated)
    if rms_new > 1e-10:
        scale = rms_orig / rms_new
        modulated *= band.dtype.type(scale) if np.issubdtype(band.dtype, np.floating) else scale
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

    complement += band
    return complement
