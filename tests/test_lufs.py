import math

import numpy as np
import pytest

from util.lufs import (
    _k_weighting_sos,
    apply_envelope_aware_gain,
    loudness_norm_fast,
    measure_lufs,
    resample_poly_fast,
)

try:
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover
    pyln = None


def _make_sine(freq: float, sr: int, duration: float = 1.5, amp: float = 0.4) -> np.ndarray:
    t = np.linspace(0.0, duration, int(round(sr * duration)), endpoint=False, dtype=np.float64)
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


def _make_mixed_signal(sr: int, duration: float = 2.0, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, duration, int(round(sr * duration)), endpoint=False, dtype=np.float64)
    sig = (
        0.25 * np.sin(2.0 * math.pi * 220.0 * t)
        + 0.15 * np.sin(2.0 * math.pi * 880.0 * t)
        + 0.08 * np.sin(2.0 * math.pi * 2640.0 * t)
        + 0.03 * rng.normal(size=t.shape[0])
    )
    return sig.astype(np.float32)


class TestMeasureLufs:
    @pytest.mark.parametrize("sr", [44100, 48000, 22050, 16000])
    def test_matches_pyloudnorm_sine_and_mixed(self, sr):
        if pyln is None:
            pytest.skip("pyloudnorm is not installed in environment")

        meter = pyln.Meter(sr)
        for sig in (_make_sine(440.0, sr, 1.5, 0.5), _make_sine(100.0, sr, 1.5, 0.4), _make_mixed_signal(sr)):
            expected = meter.integrated_loudness(sig.astype(np.float64))
            actual = measure_lufs(sig, sr)
            assert math.isfinite(actual)
            assert abs(actual - expected) < 0.15

    def test_matches_pyloudnorm_multichannel(self):
        if pyln is None:
            pytest.skip("pyloudnorm is not installed in environment")

        sr = 44100
        left = _make_sine(440.0, sr, 1.2, 0.35)
        right = _make_mixed_signal(sr, 1.2)
        stereo = np.stack([left, right], axis=1)
        expected = pyln.Meter(sr).integrated_loudness(stereo.astype(np.float64))
        actual = measure_lufs(stereo, sr)
        assert abs(actual - expected) < 0.15

    def test_silence_empty_and_short_audio_safety(self):
        assert measure_lufs(np.array([], dtype=np.float32), 44100) == float("-inf")
        assert measure_lufs(np.zeros(44100, dtype=np.float32), 44100) == float("-inf")
        assert measure_lufs(np.zeros(500, dtype=np.float32), 44100) == float("-inf")

        # Below -70 LUFS absolute gate -> -inf
        ultra_quiet = _make_sine(440.0, 44100, 1.0, amp=1e-5)
        assert measure_lufs(ultra_quiet, 44100) == float("-inf")

        # Short (< 400 ms) voiced signal returns finite loudness consistent with long tone
        short_sine = _make_sine(440.0, 44100, duration=0.1, amp=0.5)
        long_sine = _make_sine(440.0, 44100, duration=1.0, amp=0.5)
        short_lufs = measure_lufs(short_sine, 44100)
        long_lufs = measure_lufs(long_sine, 44100)
        assert math.isfinite(short_lufs)
        assert abs(short_lufs - long_lufs) < 0.15

    def test_k_weighting_sos_is_cached(self):
        _k_weighting_sos.cache_clear()
        sos1 = _k_weighting_sos(44100)
        sos2 = _k_weighting_sos(44100)
        sos3 = _k_weighting_sos(48000)
        assert sos1 is sos2
        assert sos1.shape == (2, 6)
        assert sos3.shape == (2, 6)
        assert _k_weighting_sos.cache_info().hits >= 1


class TestEnvelopeAwareGain:
    def test_prevents_tail_pop_and_respects_peak_limit(self):
        sr = 44100
        body_len = int(0.6 * sr)
        tail_len = int(0.6 * sr)

        t_body = np.arange(body_len, dtype=np.float64) / sr
        body = 0.08 * np.sin(2.0 * math.pi * 440.0 * t_body)

        # Quiet breath tail (~ -46 dBFS RMS) with a transient spike inside the low-level tail
        t_tail = np.arange(tail_len, dtype=np.float64) / sr
        tail = 0.006 * np.sin(2.0 * math.pi * 1200.0 * t_tail)
        tail[tail_len // 2] = 0.09  # local spike inside low-level tail
        audio = np.concatenate([body, tail]).astype(np.float32)

        target_gain = 10.0 ** (16.0 / 20.0)  # +16 dB boost
        tail_limit_dbfs = -6.0
        tail_limit_lin = 10.0 ** (tail_limit_dbfs / 20.0)

        out_preserve = apply_envelope_aware_gain(
            audio,
            target_gain=target_gain,
            sample_rate=sr,
            low_level_protect_db=-40.0,
            tail_peak_limit_dbfs=tail_limit_dbfs,
            tail_mode="preserve_relative",
        )
        out_clip = apply_envelope_aware_gain(
            audio,
            target_gain=target_gain,
            sample_rate=sr,
            low_level_protect_db=-40.0,
            tail_peak_limit_dbfs=tail_limit_dbfs,
            tail_mode="clip",
        )

        assert out_preserve.dtype == np.float32
        assert out_clip.dtype == np.float32
        assert np.all(np.isfinite(out_preserve))
        assert np.all(np.isfinite(out_clip))

        # Sustained body gets full boost
        body_rms_in = float(np.sqrt(np.mean(audio[: int(0.4 * sr)] ** 2)))
        body_rms_out = float(np.sqrt(np.mean(out_preserve[: int(0.4 * sr)] ** 2)))
        assert body_rms_out / body_rms_in > target_gain * 0.9

        # Low-level tail spike is clamped <= tail_peak_limit_dbfs
        tail_slice = slice(body_len + int(0.1 * sr), len(audio))
        assert float(np.max(np.abs(out_preserve[tail_slice]))) <= tail_limit_lin + 1e-6
        assert float(np.max(np.abs(out_clip[tail_slice]))) <= tail_limit_lin + 1e-6

        # preserve_relative retains more relative tail level than clip/legacy_protect
        quiet_tail_slice = slice(body_len + int(0.1 * sr), body_len + tail_len // 2 - 500)
        rms_preserve = float(np.sqrt(np.mean(out_preserve[quiet_tail_slice] ** 2)))
        rms_clip = float(np.sqrt(np.mean(out_clip[quiet_tail_slice] ** 2)))
        assert rms_preserve > rms_clip

    def test_max_boost_cap_and_edge_cases(self):
        sr = 44100
        sine = _make_sine(440.0, sr, 0.5, amp=0.01)
        out = apply_envelope_aware_gain(sine, target_gain=100.0, sample_rate=sr, max_boost_db=18.0)
        max_expected_gain = 10.0 ** (18.0 / 20.0)
        assert float(np.max(np.abs(out))) <= float(np.max(np.abs(sine))) * max_expected_gain * 1.01

        zeros = np.zeros(2000, dtype=np.float32)
        out_zeros = apply_envelope_aware_gain(zeros, target_gain=4.0, sample_rate=sr)
        assert out_zeros.dtype == np.float32
        assert float(np.max(np.abs(out_zeros))) == 0.0


class TestLoudnessNormFast:
    def test_normalizes_to_target_lufs_and_respects_strength(self):
        sr = 44100
        sig = _make_mixed_signal(sr, duration=1.5) * np.float32(0.25)
        in_lufs = measure_lufs(sig, sr)

        norm_100 = loudness_norm_fast(sig, sr, loudness=-16.0, strength=100, trim_silence=False)
        assert norm_100.dtype == np.float32
        assert np.all(np.isfinite(norm_100))
        assert abs(measure_lufs(norm_100, sr) - (-16.0)) < 0.6

        norm_50 = loudness_norm_fast(sig, sr, loudness=-16.0, strength=50, trim_silence=False)
        expected_mid = in_lufs + (-16.0 - in_lufs) * 0.5
        assert abs(measure_lufs(norm_50, sr) - expected_mid) < 0.6

        norm_0 = loudness_norm_fast(sig, sr, loudness=-16.0, strength=0, trim_silence=False)
        assert np.allclose(norm_0, sig)

    def test_trim_silence_crossfade_and_zero_safety(self):
        sr = 44100
        # Pure silence stays float32 zero without NaN
        zeros = np.zeros(sr, dtype=np.float32)
        out_zeros = loudness_norm_fast(zeros, sr, loudness=-16.0)
        assert out_zeros.dtype == np.float32
        assert np.all(np.isfinite(out_zeros))
        assert float(np.max(np.abs(out_zeros))) == 0.0

        # Short burst + silent tail: tail after padding remains silent (no reflect leak)
        full_len = int(sr * 0.6)
        burst_len = int(sr * 0.12)
        short_burst = np.zeros(full_len, dtype=np.float32)
        short_burst[:burst_len] = _make_sine(440.0, sr, duration=0.12, amp=0.15)
        normed = loudness_norm_fast(short_burst, sr, loudness=-16.0, trim_silence=True)
        assert len(normed) == full_len
        assert normed.dtype == np.float32
        assert np.all(np.isfinite(normed))
        tail_start = int(sr * 0.30)
        assert float(np.max(np.abs(normed[tail_start:]))) == 0.0

    def test_envelope_protection_prevents_tail_boost_explosion(self):
        sr = 44100
        body = _make_sine(440.0, sr, duration=0.8, amp=0.05)
        tail = _make_sine(1500.0, sr, duration=0.8, amp=0.005)
        tail[len(tail) // 2] = 0.08
        audio = np.concatenate([body, tail]).astype(np.float32)

        normed = loudness_norm_fast(
            audio,
            sr,
            loudness=-12.0,
            trim_silence=False,
            max_boost_db=18.0,
            low_level_protect_db=-40.0,
            tail_peak_limit_dbfs=-6.0,
            tail_mode="preserve_relative",
        )
        tail_limit_lin = 10.0 ** (-6.0 / 20.0)
        tail_region = normed[len(body) + int(0.1 * sr) :]
        assert float(np.max(np.abs(tail_region))) <= tail_limit_lin + 1e-6


class TestResamplePolyFast:
    @pytest.mark.parametrize(
        "orig_sr,target_sr,num_samples",
        [
            (48000, 44100, 48000),
            (48000, 44100, 47999),
            (22050, 44100, 22050),
            (22050, 44100, 12345),
            (44100, 48000, 44100),
            (44100, 44101, 44100),  # triggers limit_denominator(512) fallback
        ],
    )
    def test_exact_length_and_dtype(self, orig_sr, target_sr, num_samples):
        t = np.arange(num_samples, dtype=np.float64) / orig_sr
        x = (0.5 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
        y = resample_poly_fast(x, orig_sr, target_sr)
        assert y.dtype == np.float32
        assert len(y) == round(num_samples * target_sr / orig_sr)
        assert np.all(np.isfinite(y))

    @pytest.mark.parametrize("orig_sr,target_sr", [(48000, 44100), (22050, 44100)])
    def test_preserves_spectrum_and_rejects_aliasing(self, orig_sr, target_sr):
        duration = 1.0
        n = int(orig_sr * duration)
        t = np.arange(n, dtype=np.float64) / orig_sr
        f1, f2 = 440.0, 3500.0
        x = (0.5 * np.sin(2.0 * math.pi * f1 * t) + 0.3 * np.sin(2.0 * math.pi * f2 * t)).astype(
            np.float32
        )

        y = resample_poly_fast(x, orig_sr, target_sr)
        window = np.hanning(len(y))
        mag = np.abs(np.fft.rfft(y * window))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / target_sr)

        # Find two dominant peaks
        peak1_idx = int(np.argmax(mag[freqs < 1500.0]))
        peak2_idx = int(np.argmax(mag[(freqs >= 1500.0) & (freqs < 6000.0)])) + int(
            np.sum(freqs < 1500.0)
        )
        assert abs(freqs[peak1_idx] - f1) < 2.0
        assert abs(freqs[peak2_idx] - f2) < 2.0

        # Spurious out-of-band energy (> 8 kHz) should be strongly suppressed
        oob_mask = freqs > 8000.0
        assert float(np.max(mag[oob_mask])) < float(mag[peak1_idx]) * 1e-3

    def test_empty_and_identity(self):
        empty = np.array([], dtype=np.float32)
        assert len(resample_poly_fast(empty, 48000, 44100)) == 0
        x = _make_sine(440.0, 44100, 0.1)
        y = resample_poly_fast(x, 44100, 44100)
        assert y.dtype == np.float32
        assert np.array_equal(x, y)
