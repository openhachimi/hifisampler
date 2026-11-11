import os
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from backend import models
from backend.resampler import Resampler
from config import CONFIG
from util.cache_manager import cache_manager


@pytest.fixture(scope="module", autouse=True)
def initialize_global_models():
    """Ensure neural models are initialized once for the module test session."""
    models.initialize_models()
    assert (models.vocoder is not None) or (models.ort_session is not None)
    assert models.mel_analyzer is not None
    assert models.hnsep_model is not None


class TestResamplerPipeline:
    """End-to-end integration tests for model initialization and audio rendering."""

    def test_model_initialization_state(self):
        assert CONFIG.model_type in ("onnx", "ckpt")
        if CONFIG.model_type == "onnx":
            assert models.ort_session is not None
        else:
            assert models.vocoder is not None

        assert models.hnsep_model is not None
        assert models.mel_analyzer is not None

    def test_render_basic_note(self, synthetic_wav_pair):
        in_path, out_path = synthetic_wav_pair

        # Render a basic A4 note (440 Hz) for 300 ms at tempo 120
        Resampler(
            in_file=str(in_path),
            out_file=str(out_path),
            pitch="A4",
            velocity=100,
            flags="",
            offset=0,
            length=300,
            volume=100,
            tempo="!120",
            pitch_string="AA",
        )

        assert out_path.exists()
        audio, fs = sf.read(str(out_path), dtype="float32")
        assert fs == CONFIG.sample_rate
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0.001
        assert np.max(np.abs(audio)) <= 1.0

    def test_render_null_output_preview_mode(self, synthetic_wav_file, caplog):
        # Preview / cache pre-warming mode using null output device
        # Resampler skips resample() when output is 'nul' or '/dev/null'
        import logging
        with caplog.at_level(logging.INFO):
            Resampler(
                in_file=str(synthetic_wav_file),
                out_file="nul",
                pitch="C4",
                velocity=100,
                flags="",
                length=200,
                tempo="!120",
            )
            assert "Null output file. Skipping..." in caplog.text

            caplog.clear()
            Resampler(
                in_file=str(synthetic_wav_file),
                out_file="/dev/null",
                pitch="C4",
                velocity=100,
                flags="",
                length=200,
                tempo="!120",
            )
            assert "Null output file. Skipping..." in caplog.text

    @pytest.mark.parametrize(
        "flag_test",
        [
            "g-10",         # Gender shift
            "t5",           # Pitch shift
            "P80",          # Loudness normalization
            "A40",          # Amplitude modulation
            "HG25",         # Growl / vocal fry
            "Hb100Hv100",   # Balanced breath and voicing
        ],
    )
    def test_render_with_utau_flags(self, synthetic_wav_file, tmp_path, flag_test):
        out_path = tmp_path / f"out_flag_{flag_test}.wav"
        Resampler(
            in_file=str(synthetic_wav_file),
            out_file=str(out_path),
            pitch="E4",
            velocity=100,
            flags=flag_test,
            length=250,
            tempo="!120",
            pitch_string="AA",
        )

        assert out_path.exists()
        audio, fs = sf.read(str(out_path), dtype="float32")
        assert fs == CONFIG.sample_rate
        assert len(audio) > 0
        assert np.max(np.abs(audio)) > 0.001

    def test_features_caching_and_force_generate(self, synthetic_wav_pair):
        in_path, out_path1 = synthetic_wav_pair
        out_path2 = out_path1.with_name("note_out_2.wav")

        # 1st run: generates features and disk cache .hifi.npz
        Resampler(
            in_file=str(in_path),
            out_file=str(out_path1),
            pitch="A4",
            velocity=100,
            flags="Hb100Hv100g0",
            length=200,
            tempo="!120",
        )
        assert out_path1.exists()

        # Check that cache file was generated
        cache_files = list(in_path.parent.glob("*.hifi.npz"))
        assert len(cache_files) >= 1

        # 2nd run: loads from cache
        Resampler(
            in_file=str(in_path),
            out_file=str(out_path2),
            pitch="A4",
            velocity=100,
            flags="Hb100Hv100g0",
            length=200,
            tempo="!120",
        )
        assert out_path2.exists()

        # 3rd run with 'G' flag: forces generation
        out_path3 = out_path1.with_name("note_out_3.wav")
        Resampler(
            in_file=str(in_path),
            out_file=str(out_path3),
            pitch="A4",
            velocity=100,
            flags="Hb100Hv100g0G",
            length=200,
            tempo="!120",
        )
        assert out_path3.exists()

    def test_boundary_timing_and_pitchbend(self, synthetic_wav_pair):
        in_path, out_path = synthetic_wav_pair

        # Test short note, non-zero offset/consonant/cutoff, and pitchbend string with RLE
        Resampler(
            in_file=str(in_path),
            out_file=str(out_path),
            pitch="G4",
            velocity=80,
            flags="g0",
            offset=50,
            length=150,
            consonant=20,
            cutoff=-50,
            volume=90,
            modulation=0,
            tempo="!140",
            pitch_string="AA#5#AB#5",
        )

        assert out_path.exists()
        audio, fs = sf.read(str(out_path), dtype="float32")
        assert fs == CONFIG.sample_rate
        assert len(audio) > 0

    def test_short_waveform_stft_and_valueless_flags(self, synthetic_wav_pair):
        import torch
        from util.audio import pre_emphasis_base_tension, loudness_norm
        from util.growl import growl

        # 1. Short waveform (< n_fft // 2) in PitchAdjustableMelSpectrogram and pre_emphasis_base_tension
        short_wav = torch.randn(1, 1, 128, dtype=torch.float32, device=CONFIG.device)
        emph = pre_emphasis_base_tension(short_wav, 0.5)
        assert emph.shape == short_wav.shape

        short_mel = models.mel_analyzer(short_wav.squeeze(1), key_shift=2, speed=1.0)
        assert short_mel.ndim == 3 and short_mel.shape[1] == CONFIG.n_mels

        # 2. Empty audio in growl
        empty_np = np.array([], dtype=np.float32)
        assert len(growl(empty_np, CONFIG.sample_rate, strength=0.5)) == 0

        # 3. Short-note loudness_norm with trim_silence does not leak reflected padding into tail
        full_len = int(CONFIG.sample_rate * 0.5)
        short_burst = np.zeros(full_len, dtype=np.float32)
        burst_end = int(CONFIG.sample_rate * 0.1)
        t = np.arange(burst_end, dtype=np.float32) / CONFIG.sample_rate
        short_burst[:burst_end] = 0.25 * np.sin(2 * np.pi * 440.0 * t)
        normed = loudness_norm(short_burst, CONFIG.sample_rate, loudness=-16.0)
        assert len(normed) == full_len
        # Tail after burst + 100ms padding + 20ms frame must remain silent (zero)
        tail_start = int(CONFIG.sample_rate * 0.25)
        assert np.max(np.abs(normed[tail_start:])) == 0.0

        # 4. auto_render=False, G0, He0, and valueless A flag
        in_path, out_path = synthetic_wav_pair
        r = Resampler(
            in_file=str(in_path),
            out_file=str(out_path),
            pitch="C4",
            velocity=100,
            flags="G0He0A",
            length="240.6",
            tempo="!0",
            pitch_string="AA",
            auto_render=False,
        )
        assert r.length == 241
        assert r.tempo == 1.0
        r.render()
        assert out_path.exists()

    def test_lf_tension_invariants_and_cache_versioning(self, synthetic_wav_pair):
        import torch
        from util.audio import (
            _estimate_baseline_rd,
            _extract_cepstral_formant_envelope,
            couple_aperiodic_to_tension,
            pre_emphasis_base_tension,
        )

        # 1. Pure silence safety (no NaN / zero division)
        silent_wav = torch.zeros(1, 1, 4096, dtype=torch.float32, device=CONFIG.device)
        silent_out = pre_emphasis_base_tension(silent_wav, -2.0)
        assert torch.all(torch.isfinite(silent_out))
        assert torch.max(torch.abs(silent_out)).item() == 0.0

        # 2. Monotonic spectral tilt across b in [+2.0, +1.0, 0.0, -1.0, -2.0] (Ht from -100 to +100)
        sr = CONFIG.sample_rate
        t = torch.arange(0, sr // 2, dtype=torch.float32, device=CONFIG.device) / sr
        harmonic_sig = torch.zeros_like(t)
        for k in range(1, 20):
            harmonic_sig = harmonic_sig + (1.0 / (k ** 1.3)) * torch.sin(2.0 * np.pi * 220.0 * k * t)
        harmonic_wav = (0.2 * harmonic_sig).unsqueeze(0).unsqueeze(0)

        hf_ratios = []
        for b_val in [2.0, 1.0, 0.0, -1.0, -2.0]:
            out = pre_emphasis_base_tension(harmonic_wav, b_val)
            assert out.shape == harmonic_wav.shape
            assert torch.all(torch.isfinite(out))
            spec = torch.abs(torch.fft.rfft(out.squeeze())) ** 2
            freqs = torch.fft.rfftfreq(out.numel(), d=1.0 / sr, device=CONFIG.device)
            e_low = torch.sum(spec[(freqs >= 150.0) & (freqs < 800.0)]).item()
            e_high = torch.sum(spec[(freqs >= 1500.0) & (freqs < 6000.0)]).item()
            hf_ratios.append(e_high / (e_low + 1e-12))

        for i in range(len(hf_ratios) - 1):
            assert hf_ratios[i + 1] > hf_ratios[i]

        # 3. Adaptive baseline Rd0 distinguishes sharp/pressed voice from lax/soft voice
        n_fft = CONFIG.n_fft
        freqs_stft = torch.arange(0, n_fft // 2 + 1, device=CONFIG.device, dtype=torch.float32) * (sr / n_fft)
        sharp_sig = sum((1.0 / (k ** 0.7)) * torch.sin(2.0 * np.pi * 220.0 * k * t) for k in range(1, 25))
        lax_sig = sum((1.0 / (k ** 2.4)) * torch.sin(2.0 * np.pi * 220.0 * k * t) for k in range(1, 25))
        p_sharp = torch.abs(torch.fft.rfft(sharp_sig[:n_fft] * torch.hann_window(n_fft, device=CONFIG.device))) ** 2
        p_lax = torch.abs(torch.fft.rfft(lax_sig[:n_fft] * torch.hann_window(n_fft, device=CONFIG.device))) ** 2
        rd0_sharp = _estimate_baseline_rd(
            p_sharp, _extract_cepstral_formant_envelope(p_sharp, sr, n_fft), freqs_stft, 220.0, sr, n_fft
        )
        rd0_lax = _estimate_baseline_rd(
            p_lax, _extract_cepstral_formant_envelope(p_lax, sr, n_fft), freqs_stft, 220.0, sr, n_fft
        )
        assert rd0_sharp < rd0_lax

        # 4. Voiced aspiration coupling preserves unvoiced consonant onset bit-identically
        n_tot = sr // 2
        consonant_end = int(0.12 * sr)
        gen = torch.Generator(device=CONFIG.device).manual_seed(42)
        ap_noise = 0.05 * torch.randn(1, 1, n_tot, generator=gen, device=CONFIG.device)
        voiced_part = torch.zeros(1, 1, n_tot, dtype=torch.float32, device=CONFIG.device)
        voiced_part[:, :, consonant_end + 2048:] = harmonic_wav[:, :, consonant_end + 2048:]
        voiced_soft = pre_emphasis_base_tension(voiced_part, 2.0)  # Ht = -100
        ap_coupled = couple_aperiodic_to_tension(ap_noise, voiced_part, voiced_soft, 2.0)
        assert torch.equal(ap_coupled[:, :, :consonant_end], ap_noise[:, :, :consonant_end])
        # Inside voiced vowel tail, high-frequency aspiration energy is attenuated for Ht = -100
        tail_orig = torch.sum(ap_noise[:, :, consonant_end + 4096:] ** 2).item()
        tail_mod = torch.sum(ap_coupled[:, :, consonant_end + 4096:] ** 2).item()
        assert tail_mod < tail_orig

        # 5. Non-zero Ht appends v3 version tag in cache filename
        in_path, out_path = synthetic_wav_pair
        Resampler(
            in_file=str(in_path),
            out_file=str(out_path),
            pitch="A4",
            velocity=100,
            flags="Ht50",
            length=200,
            tempo="!120",
        )
        v3_caches = list(in_path.parent.glob("*Ht50v3*.hifi.npz"))
        assert len(v3_caches) == 1

    def test_relative_pit_vocal_fry_invariants(self, synthetic_wav_pair):
        from util.vocal_fry import apply_lf_vocal_fry, prepare_vocal_fry_excitation

        sr = CONFIG.sample_rate
        n_frames = 60
        thop = CONFIG.hop_size / sr
        t_mel = np.arange(n_frames, dtype=np.float64) * thop
        f0_flat = np.full(n_frames, 329.63, dtype=np.float32)  # E4
        mel_voiced = np.full((CONFIG.n_mels, n_frames), -3.0, dtype=np.float32)
        mel_voiced[:35, :] = 0.8  # strong voiced low-F1 energy

        # 1. Zero-regression early exit on flat notes, <= 1-octave (-1200c) portamentos, and empty mel frames
        assert (
            prepare_vocal_fry_excitation(
                pitchbend=np.full(10, -1900, dtype=np.int32),
                base_pitch_midi=64,
                t_flag_val=0.0,
                tempo=120.0,
                new_start=0.0,
                new_end=0.1,
                t_mel=np.zeros(0, dtype=np.float64),
                f0_render=np.zeros(0, dtype=np.float32),
                mel_render=np.zeros((CONFIG.n_mels, 0), dtype=np.float32),
                sr=sr,
                n_fft=CONFIG.n_fft,
                mel_fmin=CONFIG.mel_fmin,
                mel_fmax=CONFIG.mel_fmax,
            )
            is None
        )
        for safe_pb in [np.zeros(40, dtype=np.int32), np.full(40, -1200, dtype=np.int32)]:
            ctx = prepare_vocal_fry_excitation(
                pitchbend=safe_pb,
                base_pitch_midi=64,
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
            )
            assert ctx is None

        # 2. Unvoiced fricative frame protection (high-frequency hiss dominates -> w_fry == 0)
        mel_fric = np.full((CONFIG.n_mels, n_frames), -6.0, dtype=np.float32)
        mel_fric[85:, :] = 1.2
        deep_pb = np.concatenate([np.full(50, -1900, dtype=np.int32), np.zeros(1, dtype=np.int32)])
        ctx_fric = prepare_vocal_fry_excitation(
            pitchbend=deep_pb,
            base_pitch_midi=64,
            t_flag_val=0.0,
            tempo=120.0,
            new_start=0.0,
            new_end=n_frames * thop,
            t_mel=t_mel,
            f0_render=f0_flat,
            mel_render=mel_fric,
            sr=sr,
            n_fft=CONFIG.n_fft,
            mel_fmin=CONFIG.mel_fmin,
            mel_fmax=CONFIG.mel_fmax,
        )
        assert ctx_fric is None

        # 3. Melodic >1-octave leap immunity ('nian C3 -> xiang F4' +1700c upward & downward portamentos)
        # Pitch tick spacing at 120 BPM is 60 / (120 * 96) = 5.2083 ms/tick
        # (a) VCV/CV upward leap: 35 ticks (~182 ms) at -1700c inside pre-consonant [0..0.20s],
        #     followed by 16 ticks (~83 ms) steep S-curve climb to 0c, then steady 0c vowel body
        s_u = np.linspace(0.0, 1.0, 16)
        scurve_up = (-1700.0 * (1.0 - (3.0 * s_u**2 - 2.0 * s_u**3))).astype(np.int32)
        pb_up_vcv = np.concatenate([
            np.full(35, -1700, dtype=np.int32),
            scurve_up,
            np.zeros(85, dtype=np.int32),
        ])
        assert (
            prepare_vocal_fry_excitation(
                pitchbend=pb_up_vcv,
                base_pitch_midi=65,  # F4
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
                new_con=0.20,
            )
            is None
        )

        # (b) Pure-vowel upward leap (new_con=0.0): 16 ticks (~83 ms) overlap at -1700c + 16 ticks climb to 0c
        pb_up_vowel = np.concatenate([
            np.full(16, -1700, dtype=np.int32),
            scurve_up,
            np.zeros(90, dtype=np.int32),
        ])
        assert (
            prepare_vocal_fry_excitation(
                pitchbend=pb_up_vowel,
                base_pitch_midi=65,
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
                new_con=0.0,
            )
            is None
        )

        # (c) Large downward tail portamento (F4 -> C3, -1700c at very end of note)
        scurve_dn = (-1700.0 * (3.0 * s_u**2 - 2.0 * s_u**3)).astype(np.int32)
        n_total_ticks = int(round((n_frames * thop) / (60.0 / (120.0 * 96.0))))
        pb_dn_tail = np.concatenate([
            np.zeros(max(10, n_total_ticks - 22), dtype=np.int32),
            scurve_dn,
            np.full(5, -1700, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        ])
        assert (
            prepare_vocal_fry_excitation(
                pitchbend=pb_dn_tail,
                base_pitch_midi=65,
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
                new_con=0.06,
            )
            is None
        )

        # 4. Sustained intentional fry (-1900c inside vowel body) activates continuous-glide LF vocal fry engine
        pb_intentional_fry = np.concatenate([
            np.zeros(25, dtype=np.int32),
            scurve_dn,
            np.full(65, -1900, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
        ])
        ctx_fry = prepare_vocal_fry_excitation(
            pitchbend=pb_intentional_fry,
            base_pitch_midi=64,
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
            new_con=0.04,
        )
        assert ctx_fry is not None
        f_fry_min = float(np.min(ctx_fry["f0_render"]))
        assert 22.0 <= f_fry_min <= 52.0  # Continuous creaky bubble pulse rate [22..52 Hz]

        dur_samples = int(round(n_frames * thop * sr))
        audio_t = np.linspace(0.0, n_frames * thop, num=dur_samples, endpoint=False, dtype=np.float64)
        f0_inst = np.interp(audio_t, t_mel, ctx_fry["f0_eff"])
        ph_inst = np.cumsum(f0_inst / sr)
        # Synthesize broadband harmonic carrier locked to f0_inst (250..3200 Hz formants)
        carrier = (
            0.22 * np.cos(2.0 * np.pi * 680.0 * audio_t) * (0.55 + 0.45 * np.cos(2.0 * np.pi * ph_inst))
            + 0.16 * np.cos(2.0 * np.pi * 1650.0 * audio_t) * (0.55 + 0.45 * np.cos(2.0 * np.pi * ph_inst))
        ).astype(np.float32)

        shaped = apply_lf_vocal_fry(carrier, ctx_fry, sr)
        assert shaped.shape == carrier.shape
        assert np.all(np.isfinite(shaped))
        tail_slice = slice(int(0.35 * sr), dur_samples - int(0.03 * sr))
        win = int(0.0025 * sr)
        env_orig = np.sqrt(np.convolve(carrier[tail_slice] ** 2, np.ones(win) / win, mode="valid") + 1e-12)
        env_fry = np.sqrt(np.convolve(shaped[tail_slice] ** 2, np.ones(win) / win, mode="valid") + 1e-12)
        mod_orig = np.percentile(env_orig, 95) / (np.percentile(env_orig, 15) + 1e-9)
        mod_fry = np.percentile(env_fry, 95) / (np.percentile(env_fry, 15) + 1e-9)
        assert mod_fry > mod_orig * 2.5
        assert mod_fry > 4.0
