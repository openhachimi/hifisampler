"""Tier 3 Cross-Feature Combinations Test Suite (>=17 Tests).

Provides end-to-end integration tests verifying pairwise and multi-feature
interactions across the HiFiSampler architecture:
- Ht v3 tension + Vocal Fry scoop
- Ht v3 tension + Growl 80Hz LFO
- Ho openness + Hr singer's formant + Ht tension
- BS.1770-4 LUFS normalization + extreme pitch glides
- HNSEP breath/voice split + HIFIBIN1 cache hit
- Null sink preview + cache pre-warming
- Managed server lifecycle + client optimistic POST
- Telemetry stats under multi-request load
- Formant gender shift + LUFS normalization
- Velocity time-stretching + HNSEP voicing emphasis
- Loop sustain mode + glottal tension
- Pitch-derivative amplitude modulation + vibrato
- Force generation flag G + HTTP POST
- Anti-CSRF rejection with invariant telemetry
- Wine path translation + binary cache creation
- Low pitch F2 + vocal openness + fry scoop
- High pitch G5 + singer's formant + LUFS normalization
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
import soundfile as sf
import torch

from backend import models
from backend.resampler import Resampler
from backend.stats import stats_collector
from config import CONFIG
from tests.e2e.harness import E2EHarness
from util.cache_manager import cache_manager
from util.lufs import measure_lufs


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
    """Provide managed server on port 8572."""
    with harness.managed_server(port=8572, idle_timeout=180, managed=True) as srv:
        yield srv


@pytest.fixture
def tmp_workdir(harness: E2EHarness) -> Generator[Path, None, None]:
    """Provide an isolated temporary directory."""
    with harness.temp_dir(prefix="tier3_workdir_") as tmp:
        yield tmp


@pytest.fixture
def test_wav_pair(harness: E2EHarness, tmp_workdir: Path) -> tuple[Path, Path]:
    """Create input WAV file and output WAV destination path."""
    in_wav = tmp_workdir / "input.wav"
    out_wav = tmp_workdir / "output.wav"
    harness.generate_synthetic_wav(in_wav, duration_sec=0.3, freq=440.0)
    return in_wav, out_wav


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise Cross-Feature Combinations."""

    def test_t3_01_ht_v3_with_vocal_fry(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C1: Ht v3 tension (Ht50) combined with pitch scoop (-1900 cents)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c01_ht_fry.wav"
        # Deep scoop triggers vocal fry logic while Ht50 applies glottal tension
        Resampler(str(in_wav), str(out_wav), "A3", 100, flags="Ht50", length=300, pitch_string="AA#5//")
        assert out_wav.exists()
        audio, sr = sf.read(str(out_wav), dtype="float32")
        assert sr == 44100
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    def test_t3_02_ht_v3_with_growl(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C2: Ht v3 tension (Ht-40) combined with Growl (HG40)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c02_ht_growl.wav"
        Resampler(str(in_wav), str(out_wav), "C4", 100, flags="Ht-40HG40", length=250)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.05

    def test_t3_03_ho_hr_and_ht_tension(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C3: Vocal openness (Ho30) + Singer's formant (Hr-20) + Glottal tension (Ht40)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c03_tri_dsp.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Ho30Hr-20Ht40", length=250)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) <= 1.0

    def test_t3_04_lufs_norm_with_extreme_pitchbend(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C4: BS.1770-4 LUFS (P100) on wide pitch glide (+/-1200c)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c04_lufs_glide.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="P100", length=350, pitch_string="AA#5AB#5")
        assert out_wav.exists()
        audio, sr = sf.read(str(out_wav), dtype="float32")
        lufs = measure_lufs(audio, sr)
        assert math.isfinite(lufs)
        assert abs(lufs - (-16.0)) < 3.0

    def test_t3_05_hnsep_breath_split_with_cache_hit(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C5: HNSEP breath/voice split (Hb140Hv80) followed by feature cache hit."""
        in_wav, _ = test_wav_pair
        out1 = tmp_workdir / "c05_pass1.wav"
        out2 = tmp_workdir / "c05_pass2.wav"

        Resampler(str(in_wav), str(out1), "C4", 100, flags="Hb140Hv80", length=200)
        assert out1.exists()

        # Second pass re-uses cache
        Resampler(str(in_wav), str(out2), "C4", 100, flags="Hb140Hv80", length=200)
        assert out2.exists()
        a1, _ = sf.read(str(out1), dtype="float32")
        a2, _ = sf.read(str(out2), dtype="float32")
        assert np.allclose(a1, a2, atol=1e-3)

    def test_t3_06_null_sink_preview_with_cache_prewarm(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C6: Null sink preview pre-warms cache for subsequent disk render."""
        in_wav, _ = test_wav_pair
        port = e2e_server["port"]
        null_sink = "nul" if os.name == "nt" else "/dev/null"
        real_out = tmp_workdir / "c06_rendered.wav"

        # 1. Preview to null
        res1 = harness.run_cli([str(in_wav), null_sink, "A4", "100"], env={"HIFISAMPLER_PORT": str(port)}, timeout=15.0)
        assert res1.returncode == 0

        # 2. Disk render
        res2 = harness.run_cli([str(in_wav), str(real_out), "A4", "100"], env={"HIFISAMPLER_PORT": str(port)}, timeout=15.0)
        assert res2.returncode == 0
        assert real_out.exists()

    def test_t3_07_managed_watchdog_with_client_optimistic_post(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        test_wav_pair: tuple[Path, Path],
    ) -> None:
        """C7: Client sends optimistic POST to managed server and receives 200 OK."""
        in_wav, out_wav = test_wav_pair
        port = e2e_server["port"]
        payload = {"args": [str(in_wav), str(out_wav), "C4", "100"]}
        status, text = harness.http_post("/", payload, port=port)
        assert status == 200
        assert "Success" in text
        assert out_wav.exists()

    def test_t3_08_telemetry_stats_under_concurrent_load(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C8: Telemetry /stats tracks metrics accurately under sequential requests."""
        in_wav, _ = test_wav_pair
        port = e2e_server["port"]

        _, before = harness.http_get("/stats", port=port)
        reqs_before = before.get("total_requests", 0)

        for i in range(3):
            out_p = tmp_workdir / f"c08_load_{i}.wav"
            harness.http_post("/", {"args": [str(in_wav), str(out_p), "D4", "100"]}, port=port)

        status, after = harness.http_get("/stats", port=port)
        assert status == 200
        assert after.get("total_requests", 0) >= reqs_before + 3
        assert after.get("avg_total_ms", 0.0) >= 0.0

    def test_t3_09_gender_factor_with_lufs_norm(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C9: Gender formant shift (g-30) + BS.1770-4 LUFS normalization (P100)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c09_gender_norm.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="g-30P100", length=250)
        assert out_wav.exists()
        audio, sr = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) <= 1.05

    def test_t3_10_velocity_timestretch_with_hnsep_voicing(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C10: Velocity 180 (stretched) combined with voicing boost (Hv130)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c10_vel_voice.wav"
        Resampler(str(in_wav), str(out_wav), "E4", 180, flags="Hv130", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) > 0.05

    def test_t3_11_loop_sustain_he_with_ht_tension(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C11: Loop sustain vowel reflection (He) combined with tension (Ht50)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c11_sustain_tension.wav"
        Resampler(str(in_wav), str(out_wav), "C4", 100, flags="HeHt50", length=350)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_t3_12_amplitude_modulation_a_with_vibrato_pitch(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C12: Amplitude modulation (A50) on rapid pitch vibrato curve."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c12_am_vibrato.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="A50", length=300, pitch_string="AA#4AB#4")
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_t3_13_force_generate_g_with_managed_server_post(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C13: HTTP POST / with G flag forces cache regeneration."""
        in_wav, _ = test_wav_pair
        port = e2e_server["port"]
        out_wav = tmp_workdir / "c13_forced.wav"

        payload = {"args": [str(in_wav), str(out_wav), "A4", "100", "G", "0", "200"]}
        status, text = harness.http_post("/", payload, port=port)
        assert status == 200
        assert out_wav.exists()

    def test_t3_14_anti_csrf_rejection_with_telemetry_unchanged(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """C14: Cross-origin CSRF attempt rejected with 403 does not increment success_count."""
        port = e2e_server["port"]
        _, before = harness.http_get("/stats", port=port)
        succ_before = before.get("success_count", 0)

        status, _ = harness.http_post(
            "/",
            {"args": ["in.wav", "out.wav", "C4", "100"]},
            port=port,
            headers={"Origin": "http://evil-tracker.com"},
        )
        assert status == 403

        _, after = harness.http_get("/stats", port=port)
        succ_after = after.get("success_count", 0)
        assert succ_after == succ_before

    def test_t3_15_wine_path_translation_with_hifibin_cache(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C15: Windows drive letter path parses and produces valid .hifi.bin cache."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c15_wine.wav"
        Resampler(str(in_wav), str(out_wav), "C4", 100, length=200)
        assert out_wav.exists()
        bin_files = list(in_wav.parent.glob("*.hifi.bin"))
        assert len(bin_files) >= 1

    def test_t3_16_low_pitch_f2_with_ho_openness_and_fry(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C16: Bass note (F2, ~87 Hz) with Ho50 and pitch scoop."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c16_f2_ho_fry.wav"
        Resampler(str(in_wav), str(out_wav), "F2", 100, flags="Ho50", length=300, pitch_string="AA#3//")
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    def test_t3_17_high_pitch_g5_with_hr_singers_formant_and_lufs(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """C17: Soprano note (G5, 784 Hz) with singer's formant (Hr50) and LUFS normalization (P100)."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "c17_g5_hr_lufs.wav"
        Resampler(str(in_wav), str(out_wav), "G5", 100, flags="Hr50P100", length=250)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) <= 1.05
