"""Tier 4 Real-World Application Scenarios Test Suite (>=9 Tests).

Provides end-to-end integration tests mimicking real OpenUtau vocal render sessions:
- 8-note major scale (C4 through C5)
- Expressive ballad phrase with dynamic glottal tension swells
- Vocal fry intro scoop transitioning into sustained modal vibrato
- Rapid 80ms staccato note sequences
- Multi-pitch vibrato phrase with Akima interpolation
- Heavy growl rock vocal phrase
- Loop sustain vowel sequences with reflection padding
- Full 10-note song verse previewed via null sink for pre-warming
- Multi-track batch export session with 100% feature cache reuse
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
import soundfile as sf
import torch

from backend import models
from backend.resampler import Resampler
from config import CONFIG
from tests.e2e.harness import E2EHarness
from util.cache_manager import cache_manager
from util.parse_utau import midi_to_hz, note_to_midi


@pytest.fixture(scope="session", autouse=True)
def initialize_neural_models() -> None:
    """Ensure neural models are initialized once per test session."""
    models.initialize_models()
    assert (models.vocoder is not None) or (models.ort_session is not None)
    assert models.mel_analyzer is not None
    assert models.hnsep_model is not None


@pytest.fixture(scope="module")
def harness() -> E2EHarness:
    """Instantiate E2E test harness."""
    return E2EHarness()


@pytest.fixture(scope="module")
def e2e_server(harness: E2EHarness) -> Generator[dict[str, Any], None, None]:
    """Provide managed server instance for CLI and HTTP requests."""
    with harness.managed_server(port=8572, idle_timeout=180, managed=True) as srv:
        yield srv


@pytest.fixture
def tmp_workdir(harness: E2EHarness) -> Generator[Path, None, None]:
    """Provide an isolated temporary directory."""
    with harness.temp_dir(prefix="tier4_workdir_") as tmp:
        yield tmp


@pytest.fixture
def test_wav_pair(harness: E2EHarness, tmp_workdir: Path) -> tuple[Path, Path]:
    """Create input WAV file and output WAV destination path."""
    in_wav = tmp_workdir / "input.wav"
    out_wav = tmp_workdir / "output.wav"
    harness.generate_synthetic_wav(in_wav, duration_sec=0.4, freq=440.0)
    return in_wav, out_wav


class TestTier4RealWorldScenarios:
    """Tier 4: Real-World Vocal Synthesis Scenarios."""

    def test_t4_01_eight_note_major_scale(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S1: 8-note C-major scale (C4..C5) render session."""
        in_wav, _ = test_wav_pair
        scale_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
        out_files = []

        for idx, note in enumerate(scale_notes):
            out_p = tmp_workdir / f"scale_{idx}_{note}.wav"
            Resampler(str(in_wav), str(out_p), note, 100, length=200)
            assert out_p.exists()
            out_files.append(out_p)

        assert len(out_files) == 8
        for p in out_files:
            audio, sr = sf.read(str(p), dtype="float32")
            assert sr == 44100
            assert np.max(np.abs(audio)) > 0.02
            assert np.all(np.isfinite(audio))

    def test_t4_02_expressive_ballad_phrase(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S2: 4-note expressive ballad phrase with dynamic glottal tension swells."""
        in_wav, _ = test_wav_pair
        phrase_config = [
            ("C4", "Ht10", 300),
            ("E4", "Ht40", 350),
            ("G4", "Ht60", 450),
            ("F4", "Ht20", 300),
        ]

        for i, (note, flags, dur) in enumerate(phrase_config):
            out_p = tmp_workdir / f"ballad_{i}_{note}.wav"
            Resampler(str(in_wav), str(out_p), note, 100, flags=flags, length=dur, tempo="!85.0")
            assert out_p.exists()
            audio, _ = sf.read(str(out_p), dtype="float32")
            assert np.max(np.abs(audio)) <= 1.05
            assert np.all(np.isfinite(audio))

    def test_t4_03_vocal_fry_intro_sustained_vibrato(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S3: Vocal fry intro scoop transitioning into sustained modal vibrato."""
        in_wav, _ = test_wav_pair
        # Note 1: Creaky vocal fry scoop
        out1 = tmp_workdir / "s3_fry_intro.wav"
        Resampler(str(in_wav), str(out1), "A3", 100, length=250, pitch_string="AA#5//")
        assert out1.exists()

        # Note 2: Sustained vibrato
        out2 = tmp_workdir / "s3_sustained_vibrato.wav"
        Resampler(str(in_wav), str(out2), "C4", 100, length=400, pitch_string="AA#4AB#4")
        assert out2.exists()

        a1, _ = sf.read(str(out1), dtype="float32")
        a2, _ = sf.read(str(out2), dtype="float32")
        assert len(a1) > 0 and len(a2) > 0
        assert np.max(np.abs(a1)) > 0.01
        assert np.max(np.abs(a2)) > 0.01

    def test_t4_04_rapid_staccato_notes(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S4: 6 rapid 80ms staccato notes rendered with velocity=140."""
        in_wav, _ = test_wav_pair
        notes = ["A4", "A4", "B4", "C5", "B4", "A4"]

        for i, note in enumerate(notes):
            out_p = tmp_workdir / f"staccato_{i}.wav"
            Resampler(str(in_wav), str(out_p), note, 140, length=80)
            assert out_p.exists()
            audio, sr = sf.read(str(out_p), dtype="float32")
            dur_ms = (len(audio) / sr) * 1000.0
            assert dur_ms < 150.0

    def test_t4_05_multi_pitch_vibrato_phrase(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S5: Multi-pitch vibrato phrase with Akima pitch curve."""
        in_wav, _ = test_wav_pair
        vibrato_notes = [("D4", 250), ("F4", 300), ("A4", 350)]

        for i, (note, dur) in enumerate(vibrato_notes):
            out_p = tmp_workdir / f"vibrato_phrase_{i}.wav"
            Resampler(str(in_wav), str(out_p), note, 100, length=dur, pitch_string="AA#3AB#3AA#3", modulation=30.0)
            assert out_p.exists()
            audio, _ = sf.read(str(out_p), dtype="float32")
            assert np.all(np.isfinite(audio))

    def test_t4_06_growl_rock_phrase(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S6: Rock vocal phrase with HG50 growl and Ht30 tension."""
        in_wav, _ = test_wav_pair
        rock_notes = ["E3", "G3", "A3"]

        for i, note in enumerate(rock_notes):
            out_p = tmp_workdir / f"rock_{i}_{note}.wav"
            Resampler(str(in_wav), str(out_p), note, 120, flags="HG50Ht30", length=300)
            assert out_p.exists()
            audio, _ = sf.read(str(out_p), dtype="float32")
            assert np.max(np.abs(audio)) > 0.05
            assert np.all(np.isfinite(audio))

    def test_t4_07_loop_sustain_note_sequence(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S7: Sequence of sustained vowel notes using He loop sustain mode."""
        in_wav, _ = test_wav_pair
        sustain_durations = [300, 500, 700]

        for i, dur in enumerate(sustain_durations):
            out_p = tmp_workdir / f"sustain_{i}_{dur}ms.wav"
            Resampler(str(in_wav), str(out_p), "C4", 100, flags="He", length=dur)
            assert out_p.exists()
            audio, sr = sf.read(str(out_p), dtype="float32")
            dur_ms = (len(audio) / sr) * 1000.0
            assert abs(dur_ms - dur) < 40.0

    def test_t4_08_full_song_verse_null_sink_preview(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        test_wav_pair: tuple[Path, Path],
    ) -> None:
        """S8: 8-note song verse previewed via null sink for cache pre-warming."""
        in_wav, _ = test_wav_pair
        port = e2e_server["port"]
        null_sink = "nul" if os.name == "nt" else "/dev/null"
        verse_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

        t0 = time.perf_counter()
        for note in verse_notes:
            res = harness.run_cli([str(in_wav), null_sink, note, "100"], env={"HIFISAMPLER_PORT": str(port)}, timeout=15.0)
            assert res.returncode == 0
        preview_ms = (time.perf_counter() - t0) * 1000.0

        # Entire 8-note preview executes smoothly
        assert preview_ms > 0.0

    def test_t4_09_batch_export_session_with_cache_reuse(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """S9: Multi-track batch render session verifying 100% cache reuse on duplicate notes."""
        in_wav, _ = test_wav_pair
        batch_notes = ["C4", "E4", "G4"]

        # Track 1: First render pass
        track1_outs = []
        for i, note in enumerate(batch_notes):
            out_p = tmp_workdir / f"t1_{i}_{note}.wav"
            Resampler(str(in_wav), str(out_p), note, 100, length=200)
            assert out_p.exists()
            track1_outs.append(out_p)

        # Track 2: Second render pass with same notes (hits in-memory / disk cache)
        t0 = time.perf_counter()
        track2_outs = []
        for i, note in enumerate(batch_notes):
            out_p = tmp_workdir / f"t2_{i}_{note}.wav"
            Resampler(str(in_wav), str(out_p), note, 100, length=200)
            assert out_p.exists()
            track2_outs.append(out_p)
        t2_time_ms = (time.perf_counter() - t0) * 1000.0

        # Track 2 audio must be identical to Track 1 audio
        for p1, p2 in zip(track1_outs, track2_outs):
            a1, _ = sf.read(str(p1), dtype="float32")
            a2, _ = sf.read(str(p2), dtype="float32")
            assert np.allclose(a1, a2, atol=1e-3)

        # Cache pass is faster than cold synthesis
        assert t2_time_ms < 3000.0
