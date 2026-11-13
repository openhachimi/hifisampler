import math
import numpy as np
import pytest
from scipy.signal import butter, sosfilt, sosfiltfilt

from config import CONFIG
from util.growl import _get_highpass_sos, growl
from util.vocal_fry import (
    _get_crossover_sos,
    _get_hanning_kernel,
    _pulse_rms_integral,
    apply_lf_vocal_fry,
    prepare_vocal_fry_excitation,
)


def _reference_growl(audio: np.ndarray, sample_rate: int, frequency: float = 80.0, strength: float = 0.5, freq_low: float = 400.0) -> np.ndarray:
    if len(audio) == 0 or strength == 0 or frequency <= 0:
        return audio.copy()
    nyq = sample_rate / 2
    norm = float(np.clip(freq_low / nyq, 0.01, 0.99))
    sos = butter(4, round(norm, 6), "high", output="sos")
    band = np.asarray(sosfilt(sos, audio), dtype=audio.dtype)
    complement = audio - band

    n_samples = len(band)
    half_cycles = (np.arange(n_samples, dtype=np.float32) * np.float32(2.0 * frequency / sample_rate)).astype(np.int32)
    lfo = (1 - ((half_cycles & 1) << 1)).astype(np.float32)

    exp_arg = float(strength * 100.0 / 1200.0)
    r_pos = 2.0 ** exp_arg
    r_neg = 2.0 ** (-exp_arg)
    ratio = (0.5 * (r_pos + r_neg)) + lfo.astype(np.float64) * (0.5 * (r_pos - r_neg))
    cumsum = np.cumsum(ratio, dtype=np.float64) - float(ratio[0])
    arange_idx = np.arange(n_samples, dtype=np.float64)
    drift = cumsum - arange_idx * np.mean(ratio)
    if n_samples > 100:
        sos2 = butter(2, round(float(20.0 / (sample_rate / 2)), 6), "high", output="sos")
        drift = sosfilt(sos2, drift)
    idx = np.clip(arange_idx + drift, 0.0, float(n_samples - 1))
    modulated = np.interp(idx, arange_idx, band).astype(band.dtype, copy=False)
    rms_orig = float(np.sqrt(np.mean(np.square(band, dtype=np.float64))))
    rms_new = float(np.sqrt(np.mean(np.square(modulated, dtype=np.float64))))
    if rms_new > 1e-10:
        modulated = modulated * (rms_orig / rms_new)
    return complement + modulated


def _reference_apply_lf_vocal_fry(render: np.ndarray, fry_ctx: dict, sr: int) -> np.ndarray:
    t_mel = fry_ctx["t_mel"]
    w_fry = fry_ctx["w_fry"]
    f0_eff = fry_ctx.get("f0_eff", fry_ctx.get("f0_carrier"))
    new_start = fry_ctx["new_start"]
    new_end = fry_ctx["new_end"]
    seed = int(fry_ctx.get("seed", 20261111))

    n_samples = len(render)
    audio_t = np.linspace(new_start, new_end, num=n_samples, endpoint=False, dtype=np.float64)
    w_s = np.interp(audio_t, t_mel, w_fry, left=w_fry[0], right=w_fry[-1])
    if float(np.max(w_s)) <= 1e-4:
        return render
    f0_s = np.clip(np.interp(audio_t, t_mel, f0_eff, left=f0_eff[0], right=f0_eff[-1]), 20.0, 1200.0)
    period_s = 1.0 / f0_s

    x64 = render.astype(np.float64, copy=False)
    if n_samples > 64:
        above_sub = sosfiltfilt(_get_crossover_sos(int(sr), 155.0, "high"), x64)
        sub_band = x64 - above_sub
        high_band = sosfiltfilt(_get_crossover_sos(int(sr), 900.0, "high"), above_sub)
        mid_band = above_sub - high_band
    else:
        sub_band = 0.2 * x64
        mid_band = 0.4 * x64
        high_band = 0.4 * x64

    dphase = f0_s / float(sr)
    phase = np.cumsum(dphase, dtype=np.float64)
    burst_energy = 0.35 * (mid_band ** 2) + 0.65 * (high_band ** 2)
    c_inst = (w_s * burst_energy) * np.exp(-1j * (2.0 * np.pi) * phase)
    z_glob = complex(np.sum(c_inst))
    if abs(z_glob) > 1e-16:
        z_fb = z_glob / abs(z_glob)
        step = 1024
        n_blk = max(1, int(math.ceil(n_samples / step)))
        pad_n = n_blk * step - n_samples
        c_pad = np.pad(c_inst, (0, pad_n), mode="constant") if pad_n > 0 else c_inst
        c_blk = np.mean(c_pad.reshape(n_blk, step), axis=1)
        win_l = min(11, max(3, (n_blk // 2) * 2 + 1))
        kern = np.hanning(win_l + 2)[1:-1]
        kern /= np.sum(kern)
        c_re = np.convolve(np.pad(c_blk.real, win_l // 2, mode="edge"), kern, mode="valid")[:n_blk]
        c_im = np.convolve(np.pad(c_blk.imag, win_l // 2, mode="edge"), kern, mode="valid")[:n_blk]
        c_sm = (c_re + 1j * c_im) + 0.08 * float(np.max(np.abs(c_blk)) + 1e-12) * z_fb
        phi_blk = np.unwrap(-np.angle(c_sm)) / (2.0 * np.pi)
        blk_t = (np.arange(n_blk, dtype=np.float64) + 0.5) * step
        phi_raw = np.interp(np.arange(n_samples, dtype=np.float64), blk_t, phi_blk)
        dphi = np.diff(phi_raw, prepend=phi_raw[0])
        max_dphi = 0.18 * dphase
        phi_peak = float(phi_raw[0]) + np.cumsum(np.clip(dphi, -max_dphi, max_dphi))
    else:
        phi_peak = 0.0

    aligned = phase - phi_peak + 0.050
    cycle_idx = np.floor(aligned).astype(np.int64)
    tau = (aligned - cycle_idx) * period_s
    tau_prev = tau + period_s

    c_min = int(cycle_idx[0]) - 2
    c_max = int(cycle_idx[-1]) + 2
    n_cyc = max(4, c_max - c_min + 1)
    rng = np.random.default_rng(seed ^ 0x5A3C91)
    shim_table = rng.standard_normal(n_cyc)
    cyc_grid = np.arange(c_min, c_min + n_cyc, dtype=np.int64)
    amp_table = np.clip(1.0 - 0.34 * (cyc_grid & 1).astype(np.float64) + 0.11 * shim_table, 0.45, 1.18)
    amp_cur = amp_table[np.clip(cycle_idx - c_min, 0, n_cyc - 1)]
    amp_prev = amp_table[np.clip(cycle_idx - 1 - c_min, 0, n_cyc - 1)]

    tau_a_mid, tau_d_mid = 0.00075, 0.0044
    tau_a_hi, tau_d_hi = 0.00040, 0.00235
    env_mid = (
        amp_cur * (1.0 - np.exp(-tau / tau_a_mid)) * np.exp(-tau / tau_d_mid)
        + amp_prev * (1.0 - np.exp(-tau_prev / tau_a_mid)) * np.exp(-tau_prev / tau_d_mid)
    )
    env_hi = (
        amp_cur * (1.0 - np.exp(-tau / tau_a_hi)) * np.exp(-tau / tau_d_hi)
        + amp_prev * (1.0 - np.exp(-tau_prev / tau_a_hi)) * np.exp(-tau_prev / tau_d_hi)
    )
    rms_mid = np.sqrt(_pulse_rms_integral(tau_a_mid, tau_d_mid) / period_s) * 0.86
    rms_hi = np.sqrt(_pulse_rms_integral(tau_a_hi, tau_d_hi) / period_s) * 0.86
    fry_wave = (
        0.42 * sub_band * (env_mid * np.minimum(0.85 / np.maximum(rms_mid, 1e-4), 1.9))
        + mid_band * (env_mid * np.minimum(1.28 / np.maximum(rms_mid, 1e-4), 3.10))
        + high_band * (env_hi * np.minimum(1.52 / np.maximum(rms_hi, 1e-4), 3.65))
    )
    ceil_peak = max(0.96 * float(np.max(np.abs(x64))), 0.12)
    fry_wave = ceil_peak * np.tanh(fry_wave / ceil_peak)
    return ((1.0 - w_s) * x64 + w_s * fry_wave).astype(render.dtype, copy=False)


class TestGrowlFastPath:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("n_samples", [50, 101, 4410, 44100])
    def test_growl_numerical_equivalence(self, dtype, n_samples):
        sr = 44100
        t = np.arange(n_samples, dtype=np.float64) / sr
        sig = (
            0.35 * np.sin(2.0 * np.pi * 220.0 * t)
            + 0.25 * np.sin(2.0 * np.pi * 880.0 * t)
            + 0.15 * np.sin(2.0 * np.pi * 2400.0 * t)
        ).astype(dtype)

        got = growl(sig, sr, frequency=80.0, strength=0.65, freq_low=400.0)
        ref = _reference_growl(sig, sr, frequency=80.0, strength=0.65, freq_low=400.0)

        assert got.dtype == dtype
        assert got.shape == sig.shape
        assert np.all(np.isfinite(got))
        atol = 1e-5 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=atol)

    @pytest.mark.parametrize("n_samples", [0, 1, 2, 17, 100])
    def test_growl_boundary_and_silence(self, n_samples):
        sr = 44100
        zeros = np.zeros(n_samples, dtype=np.float32)
        out_z = growl(zeros, sr, frequency=80.0, strength=0.8)
        assert out_z.shape == (n_samples,)
        assert out_z.dtype == np.float32
        assert np.all(out_z == 0.0)

        ones = np.ones(n_samples, dtype=np.float32) * 0.25
        out_bypass = growl(ones, sr, frequency=0.0, strength=0.5)
        np.testing.assert_array_equal(out_bypass, ones)
        assert out_bypass is not ones

        out_zero_str = growl(ones, sr, frequency=80.0, strength=0.0)
        np.testing.assert_array_equal(out_zero_str, ones)
        assert out_zero_str is not ones

        out_valid = growl(ones, sr, frequency=120.0, strength=1.0, freq_low=300.0)
        assert np.all(np.isfinite(out_valid))

    def test_growl_sos_cache_hits(self):
        _get_highpass_sos.cache_clear()
        sig = np.random.default_rng(0).standard_normal(4096).astype(np.float32) * 0.1
        growl(sig, 44100, frequency=80.0, strength=0.5)
        growl(sig, 44100, frequency=90.0, strength=0.8)
        assert _get_highpass_sos.cache_info().hits >= 2


class TestVocalFryFastPath:
    def _make_fry_fixture(self, sr: int = 44100, n_frames: int = 80):
        thop = CONFIG.hop_size / sr
        t_mel = np.arange(n_frames, dtype=np.float64) * thop
        f0_flat = np.full(n_frames, 329.63, dtype=np.float32)
        mel_voiced = np.full((CONFIG.n_mels, n_frames), -3.0, dtype=np.float32)
        mel_voiced[:35, :] = 0.8

        s_u = np.linspace(0.0, 1.0, 16)
        scurve_dn = (-1900.0 * (3.0 * s_u ** 2 - 2.0 * s_u ** 3)).astype(np.int32)
        pb = np.concatenate([
            np.zeros(20, dtype=np.int32),
            scurve_dn,
            np.full(90, -1900, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        ])
        ctx = prepare_vocal_fry_excitation(
            pitchbend=pb,
            base_pitch_midi=64.0,
            t_flag_val=0.0,
            tempo=120.0,
            new_start=0.0,
            new_end=n_frames * thop,
            t_mel=t_mel,
            f0_render=f0_flat,
            mel_render=mel_voiced,
            sr=sr,
            n_fft=CONFIG.n_fft,
            mel_fmin=CONFIG.mel_fmin,
            mel_fmax=CONFIG.mel_fmax,
            new_con=0.03,
        )
        return ctx, t_mel, thop

    @pytest.mark.parametrize("n_frames", [80, 180])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_apply_lf_vocal_fry_equivalence(self, n_frames, dtype):
        sr = CONFIG.sample_rate
        ctx, t_mel, thop = self._make_fry_fixture(sr=sr, n_frames=n_frames)
        assert ctx is not None

        n_samples = int(round(n_frames * thop * sr))
        audio_t = np.linspace(0.0, n_frames * thop, num=n_samples, endpoint=False, dtype=np.float64)
        f0_inst = np.interp(audio_t, t_mel, ctx["f0_eff"])
        ph_inst = np.cumsum(f0_inst / sr)
        carrier = (
            0.25 * np.cos(2.0 * np.pi * 620.0 * audio_t) * (0.55 + 0.45 * np.cos(2.0 * np.pi * ph_inst))
            + 0.18 * np.cos(2.0 * np.pi * 1750.0 * audio_t) * (0.55 + 0.45 * np.cos(2.0 * np.pi * ph_inst))
        ).astype(dtype)

        got = apply_lf_vocal_fry(carrier, ctx, sr)
        ref = _reference_apply_lf_vocal_fry(carrier, ctx, sr)

        assert got.dtype == dtype
        assert got.shape == carrier.shape
        assert np.all(np.isfinite(got))
        atol = 1e-5 if dtype == np.float32 else 1e-11
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=atol)

    @pytest.mark.parametrize("n_samples", [0, 1, 31, 32, 64, 65, 1024, 1025, 8192, 8193])
    def test_apply_lf_vocal_fry_chunk_and_length_boundaries(self, n_samples):
        sr = CONFIG.sample_rate
        ctx, _, _ = self._make_fry_fixture(sr=sr, n_frames=80)
        assert ctx is not None

        rng = np.random.default_rng(123)
        wav = (rng.standard_normal(n_samples) * 0.2).astype(np.float32)
        out = apply_lf_vocal_fry(wav, ctx, sr)
        assert out.shape == (n_samples,)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

        if n_samples >= 32:
            ref = _reference_apply_lf_vocal_fry(wav, ctx, sr)
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

        zeros = np.zeros(n_samples, dtype=np.float32)
        out_z = apply_lf_vocal_fry(zeros, ctx, sr)
        assert np.all(out_z == 0.0)

    def test_apply_lf_vocal_fry_nonuniform_and_single_frame(self):
        sr = 44100
        wav = (0.2 * np.sin(2.0 * np.pi * 440.0 * np.arange(2048) / sr)).astype(np.float32)

        # Single-frame t_mel
        ctx_1 = {
            "t_mel": np.array([0.0], dtype=np.float64),
            "w_fry": np.array([0.8], dtype=np.float64),
            "f0_eff": np.array([32.0], dtype=np.float64),
            "new_start": 0.0,
            "new_end": 2048 / sr,
            "seed": 42,
        }
        got_1 = apply_lf_vocal_fry(wav, ctx_1, sr)
        ref_1 = _reference_apply_lf_vocal_fry(wav, ctx_1, sr)
        np.testing.assert_allclose(got_1, ref_1, rtol=1e-5, atol=1e-5)

        # Non-uniform t_mel
        ctx_nu = {
            "t_mel": np.array([0.0, 0.005, 0.02, 0.046], dtype=np.float64),
            "w_fry": np.array([0.0, 0.9, 0.85, 0.1], dtype=np.float64),
            "f0_eff": np.array([220.0, 31.0, 35.0, 180.0], dtype=np.float64),
            "new_start": 0.0,
            "new_end": 2048 / sr,
            "seed": 99,
        }
        got_nu = apply_lf_vocal_fry(wav, ctx_nu, sr)
        ref_nu = _reference_apply_lf_vocal_fry(wav, ctx_nu, sr)
        np.testing.assert_allclose(got_nu, ref_nu, rtol=1e-5, atol=1e-5)

    def test_prepare_vocal_fry_edge_cases_and_extreme_pitchbend(self):
        sr = CONFIG.sample_rate
        thop = CONFIG.hop_size / sr

        # Empty / None / tiny inputs
        assert prepare_vocal_fry_excitation(
            None, 64.0, 0.0, 120.0, 0.0, 0.5,
            np.arange(10, dtype=np.float64) * thop,
            np.full(10, 330.0, dtype=np.float32),
            np.zeros((CONFIG.n_mels, 10), dtype=np.float32),
            sr, CONFIG.n_fft, CONFIG.mel_fmin, CONFIG.mel_fmax,
        ) is None

        assert prepare_vocal_fry_excitation(
            np.array([-2000], dtype=np.int32), 64.0, 0.0, 120.0, 0.0, 0.5,
            np.arange(10, dtype=np.float64) * thop,
            np.full(10, 330.0, dtype=np.float32),
            np.zeros((CONFIG.n_mels, 10), dtype=np.float32),
            sr, CONFIG.n_fft, CONFIG.mel_fmin, CONFIG.mel_fmax,
        ) is None

        # Extreme negative & positive pitchbend swings with 1-frame and 3-frame mel
        for n_frames in (1, 3, 60):
            t_mel = np.arange(n_frames, dtype=np.float64) * thop
            f0 = np.full(n_frames, 261.63, dtype=np.float32)
            mel = np.full((CONFIG.n_mels, n_frames), -3.0, dtype=np.float32)
            mel[:35, :] = 0.8
            pb_extreme = np.concatenate([
                np.full(40, -4800, dtype=np.int32),
                np.full(20, 4800, dtype=np.int32),
                np.full(40, -3600, dtype=np.int32),
                np.zeros(1, dtype=np.int32),
            ])
            res = prepare_vocal_fry_excitation(
                pb_extreme, 60.0, 0.0, 120.0, 0.0, max(n_frames * thop, thop),
                t_mel, f0, mel, sr, CONFIG.n_fft, CONFIG.mel_fmin, CONFIG.mel_fmax,
            )
            if res is not None:
                assert np.all(np.isfinite(res["mel_render"]))
                assert np.all(np.isfinite(res["f0_render"]))
                assert np.all(np.isfinite(res["w_fry"]))
                assert np.all(np.isfinite(res["f0_eff"]))

        # Flat deep pitchbend with trailing 0 (exercises Akima constant-pitch branch)
        n_frames = 60
        t_mel = np.arange(n_frames, dtype=np.float64) * thop
        mel = np.full((CONFIG.n_mels, n_frames), -3.0, dtype=np.float32)
        mel[:35, :] = 0.8
        pb_flat_deep = np.concatenate([np.full(80, -1900, dtype=np.int32), np.zeros(1, dtype=np.int32)])
        res_flat = prepare_vocal_fry_excitation(
            pb_flat_deep, 64.0, 0.0, 120.0, 0.0, n_frames * thop,
            t_mel, np.full(n_frames, 329.63, dtype=np.float32), mel,
            sr, CONFIG.n_fft, CONFIG.mel_fmin, CONFIG.mel_fmax,
        )
        assert res_flat is not None
        assert np.all(np.isfinite(res_flat["f0_render"]))

        # Uniform floor mel (-15.0) -> finite output without NaN/Inf
        res_sil = prepare_vocal_fry_excitation(
            pb_flat_deep, 64.0, 0.0, 120.0, 0.0, n_frames * thop,
            t_mel, np.full(n_frames, 329.63, dtype=np.float32),
            np.full((CONFIG.n_mels, n_frames), -15.0, dtype=np.float32),
            sr, CONFIG.n_fft, CONFIG.mel_fmin, CONFIG.mel_fmax,
        )
        if res_sil is not None:
            assert np.all(np.isfinite(res_sil["mel_render"]))
            assert np.all(np.isfinite(res_sil["f0_render"]))

    def test_kernel_and_sos_caches(self):
        _get_hanning_kernel.cache_clear()
        _get_crossover_sos.cache_clear()
        k1 = _get_hanning_kernel(7)
        k2 = _get_hanning_kernel(7)
        assert k1 is k2
        assert _get_hanning_kernel.cache_info().hits == 1
        s1 = _get_crossover_sos(44100, 155.0, "high")
        s2 = _get_crossover_sos(44100, 155.0, "high")
        assert s1 is s2
        assert _get_crossover_sos.cache_info().hits == 1
