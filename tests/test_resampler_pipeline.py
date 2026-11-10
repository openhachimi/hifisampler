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
        from util.audio import pre_emphasis_base_tension

        # 1. Pure silence safety (no NaN / zero division)
        silent_wav = torch.zeros(1, 1, 4096, dtype=torch.float32, device=CONFIG.device)
        silent_out = pre_emphasis_base_tension(silent_wav, -2.0)
        assert torch.all(torch.isfinite(silent_out))
        assert torch.max(torch.abs(silent_out)).item() == 0.0

        # 2. Monotonic spectral tilt across b in [+2.0, +1.0, 0.0, -1.0, -2.0] (Ht from -100 to +100)
        sr = CONFIG.sample_rate
        t = torch.arange(0, sr // 2, dtype=torch.float32, device=CONFIG.device) / sr
        harmonic_sig = torch.zeros_like(t)
        for k in range(1, 16):
            harmonic_sig = harmonic_sig + (1.0 / k) * torch.sin(2.0 * np.pi * 220.0 * k * t)
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

        # 3. Non-zero Ht appends v2 version tag in cache filename
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
        v2_caches = list(in_path.parent.glob("*Ht50v2*.hifi.npz"))
        assert len(v2_caches) == 1
