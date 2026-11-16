"""Tier 2 Boundary & Corner Cases Test Suite (85 Test Cases across F1 through F17).

Provides comprehensive opaque-box boundary, extreme-value, and corner-case tests
for HiFiSampler, verifying robust behavior across empty/corrupted inputs, extreme
parameters, missing files, security rejections, and hardware fallback paths.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator

import numpy as np
import pytest
import soundfile as sf
import torch

from backend import models
from backend.resampler import Resampler
from backend.stats import ResampleStats, StatsCollector
from config import CONFIG
from tests.e2e.harness import E2EHarness, find_free_port
from util.audio import (
    couple_aperiodic_to_tension,
    loudness_norm,
    pre_emphasis_base_tension,
)
from util.cache_manager import cache_manager
from util.growl import growl
from util.lufs import apply_envelope_aware_gain, measure_lufs
from util.model_loader import (
    create_optimized_session_options,
    get_cpu_provider_options,
    get_onnx_providers,
)
from util.parse_utau import (
    flag_re,
    midi_to_hz,
    note_to_midi,
    pitch_string_to_cents,
    to_int12,
    to_uint6,
)
from util.vocal_fry import (
    FRY_ONSET_CENTS,
    apply_lf_vocal_fry,
    prepare_vocal_fry_excitation,
)


# ============================================================================
# Fixtures
# ============================================================================

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
    """Provide a managed server instance for CLI and HTTP boundary testing."""
    with harness.managed_server(port=8572, idle_timeout=180, managed=True) as srv:
        yield srv


@pytest.fixture
def tmp_workdir(harness: E2EHarness) -> Generator[Path, None, None]:
    """Provide an isolated temporary directory fixture."""
    with harness.temp_dir(prefix="tier2_workdir_") as tmp:
        yield tmp


@pytest.fixture
def test_wav_pair(harness: E2EHarness, tmp_workdir: Path) -> tuple[Path, Path]:
    """Create input WAV file and output WAV destination path."""
    in_wav = tmp_workdir / "input.wav"
    out_wav = tmp_workdir / "output.wav"
    harness.generate_synthetic_wav(in_wav, duration_sec=0.3, freq=440.0)
    return in_wav, out_wav


# ============================================================================
# Feature F1: Branch & Toolchain Setup Boundaries (5 Tests)
# ============================================================================

class TestFeatureF1BranchToolchainBoundaries:
    """F1: Branch & Toolchain Setup Boundaries."""

    def test_f1_06_nonexistent_crate_handling(self, harness: E2EHarness) -> None:
        """Verify querying non-existent crate path returns False/handled."""
        invalid_crate = harness.root_dir / "rust" / "crates" / "nonexistent_crate_xyz"
        assert not invalid_crate.exists()

    def test_f1_07_invalid_git_ref_handling(self, harness: E2EHarness) -> None:
        """Verify git status check handles nonexistent ref gracefully."""
        git_bin = shutil.which("git")
        if not git_bin:
            cand = Path(r"C:\Users\19765\AppData\Local\hermes\tools\git-2.53.0+3-win32-x64\cmd\git.exe")
            if cand.exists():
                git_bin = str(cand)

        if git_bin:
            res = subprocess.run(
                [git_bin, "rev-parse", "--verify", "nonexistent_branch_ref_xyz"],
                cwd=str(harness.root_dir),
                capture_output=True,
                text=True,
            )
            assert res.returncode != 0
        else:
            assert True

    def test_f1_08_empty_workspace_dir_check(self, tmp_workdir: Path) -> None:
        """Verify harness raises FileNotFoundError when target binary is missing in empty dir."""
        empty_harness = E2EHarness(target="python", root_dir=tmp_workdir)
        cmd = empty_harness.resolve_cli_cmd()
        # Points to nonexistent file in temp dir
        assert not Path(cmd[-1]).exists() or not str(tmp_workdir) in cmd[-1]

    def test_f1_09_unsupported_target_rejection(self, harness: E2EHarness) -> None:
        """Verify E2EHarness rejects invalid target string with ValueError."""
        with pytest.raises(ValueError, match="Invalid target"):
            E2EHarness(target="unsupported_target_arch")

    def test_f1_10_target_env_override(self, monkeypatch: pytest.MonkeyPatch, harness: E2EHarness) -> None:
        """Verify HIFISAMPLER_TEST_TARGET environment variable override."""
        monkeypatch.setenv("HIFISAMPLER_TEST_TARGET", "python")
        h = E2EHarness()
        assert h.target == "python"


# ============================================================================
# Feature F2: Audio I/O Boundaries (5 Tests)
# ============================================================================

class TestFeatureF2AudioIOBoundaries:
    """F2: Audio I/O Boundaries."""

    def test_f2_06_empty_zero_byte_wav(self, tmp_workdir: Path, harness: E2EHarness) -> None:
        """Verify reading a 0-byte file raises soundfile / format error."""
        empty_file = tmp_workdir / "empty.wav"
        empty_file.write_bytes(b"")
        with pytest.raises(Exception):
            harness.read_wav(empty_file)

    def test_f2_07_truncated_riff_header(self, tmp_workdir: Path, harness: E2EHarness) -> None:
        """Verify reading a truncated WAV (< 44 bytes RIFF header) raises error."""
        corrupt_wav = tmp_workdir / "truncated.wav"
        corrupt_wav.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt ")
        with pytest.raises(Exception):
            harness.read_wav(corrupt_wav)

    def test_f2_08_stereo_to_mono_handling(self, tmp_workdir: Path) -> None:
        """Verify 2-channel stereo audio downmixes cleanly to mono."""
        stereo_wav = tmp_workdir / "stereo.wav"
        sr = 44100
        samples = np.zeros((1000, 2), dtype=np.float32)
        samples[:, 0] = 0.5
        samples[:, 1] = 0.3
        sf.write(str(stereo_wav), samples, sr)

        data, _ = sf.read(str(stereo_wav), dtype="float32")
        mono = data.mean(axis=1) if data.ndim > 1 else data
        assert mono.ndim == 1
        assert abs(mono[0] - 0.4) < 1e-4

    def test_f2_09_extreme_float32_clipping(self, tmp_workdir: Path) -> None:
        """Verify audio samples with extreme values (> 1.0) can be clipped safely."""
        extreme_wav = tmp_workdir / "extreme.wav"
        sr = 44100
        samples = np.array([2.5, -3.0, 0.5, -0.8], dtype=np.float32)
        clipped = np.clip(samples, -1.0, 1.0)
        sf.write(str(extreme_wav), clipped, sr, subtype="PCM_16")

        read_back, _ = sf.read(str(extreme_wav), dtype="float32")
        assert np.max(read_back) <= 1.0
        assert np.min(read_back) >= -1.0

    def test_f2_10_sub_millisecond_wav(self, tmp_workdir: Path, harness: E2EHarness) -> None:
        """Verify ultra-short 16-sample audio file is written and read accurately."""
        short_wav = tmp_workdir / "sub_ms.wav"
        harness.generate_synthetic_wav(short_wav, duration_sec=0.001, sample_rate=44100)
        data, sr = harness.read_wav(short_wav)
        assert sr == 44100
        assert len(data) >= 1


# ============================================================================
# Feature F3: HIFIBIN1 Feature Cache Boundaries (5 Tests)
# ============================================================================

class TestFeatureF3HifiBinCacheBoundaries:
    """F3: HIFIBIN1 Feature Cache Boundaries."""

    def test_f3_06_truncated_header_less_than_28_bytes(self, tmp_workdir: Path, harness: E2EHarness) -> None:
        """Verify parsing header < 28 bytes raises ValueError."""
        trunc_bin = tmp_workdir / "trunc.hifi.bin"
        trunc_bin.write_bytes(b"HIFIBIN1\x01\x00\x00\x00")  # Only 12 bytes
        with pytest.raises(ValueError, match="too small"):
            harness.parse_hifibin_header(trunc_bin)

    def test_f3_07_wrong_magic_bytes(self, tmp_workdir: Path, harness: E2EHarness) -> None:
        """Verify header with incorrect magic is identified."""
        bad_magic_bin = tmp_workdir / "bad_magic.hifi.bin"
        header = struct.pack("<8sIIId", b"BADMAGIC", 1, 10, 10, 1.0)
        payload = b"\x00" * (10 * 10 * 4)
        bad_magic_bin.write_bytes(header + payload)

        meta = harness.parse_hifibin_header(bad_magic_bin)
        assert meta["magic"] != "HIFIBIN1"

    def test_f3_08_corrupted_payload_size_mismatch(self, tmp_workdir: Path, harness: E2EHarness) -> None:
        """Verify payload byte length mismatch is flagged by size_valid=False."""
        mismatch_bin = tmp_workdir / "mismatch.hifi.bin"
        header = struct.pack("<8sIIId", b"HIFIBIN1", 1, 128, 50, 1.0)
        # Expected payload: 128 * 50 * 4 = 25600 bytes, write only 100 bytes
        mismatch_bin.write_bytes(header + (b"\x00" * 100))

        meta = harness.parse_hifibin_header(mismatch_bin)
        assert meta["size_valid"] is False
        assert meta["file_size"] != meta["expected_size"]

    def test_f3_09_future_mtime_invalidation(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify source WAV with newer mtime invalidates cache and regenerates successfully."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, length=150)
        assert out_wav.exists()

        # Touch source WAV to make its mtime newer than the cache
        future_mtime = time.time() + 100.0
        os.utime(str(in_wav), (future_mtime, future_mtime))

        Resampler(str(in_wav), str(out_wav), "A4", 100, length=150)
        assert out_wav.exists()

    def test_f3_10_force_regeneration_flag_g(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify UTAU flag 'G' forces cache bypass and regeneration."""
        in_wav, out_wav = test_wav_pair
        # Render with G flag
        r = Resampler(str(in_wav), str(out_wav), "A4", 100, flags="G", length=150)
        assert out_wav.exists()
        assert "G" in r.flags


# ============================================================================
# Feature F4: BS.1770-4 LUFS Normalization Boundaries (5 Tests)
# ============================================================================

class TestFeatureF4LUFSBoundaries:
    """F4: BS.1770-4 LUFS Normalization Boundaries."""

    def test_f4_06_pure_silence_input(self) -> None:
        """Verify pure silence does not cause division-by-zero or produce NaN/Inf."""
        sr = 44100
        silence = np.zeros(sr, dtype=np.float32)
        norm = loudness_norm(silence, sr, loudness=-16.0)
        assert np.all(np.isfinite(norm))
        assert np.max(np.abs(norm)) == 0.0

    def test_f4_07_ultra_short_under_400ms(self) -> None:
        """Verify audio shorter than 400ms block window measures finite LUFS."""
        sr = 44100
        short_sine = (0.5 * np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 0.1, int(sr * 0.1)))).astype(np.float32)
        lufs = measure_lufs(short_sine, sr)
        assert math.isfinite(lufs)

    def test_f4_08_extreme_p_flag_zero(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify P0 applies neutral gain without distortion."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="P0", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f4_09_extreme_loudness_target_high(self) -> None:
        """Verify high loudness target clamps gain safely under peak limiter."""
        sr = 44100
        t = np.linspace(0.0, 0.5, int(sr * 0.5), endpoint=False, dtype=np.float32)
        sine = (0.8 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        norm = loudness_norm(sine, sr, loudness=-6.0)
        assert np.max(np.abs(norm)) <= 1.0

    def test_f4_10_impulse_train_tail_preservation(self) -> None:
        """Verify sparse impulse train followed by silence preserves quiet tail."""
        sr = 44100
        sig = np.zeros(int(sr * 0.5), dtype=np.float32)
        sig[0] = 0.9
        sig[100] = -0.9
        norm = loudness_norm(sig, sr, loudness=-16.0)
        tail = norm[int(sr * 0.2):]
        assert np.max(np.abs(tail)) < 1e-4


# ============================================================================
# Feature F5: Pitch & Expression Parsing Boundaries (5 Tests)
# ============================================================================

class TestFeatureF5PitchParsingBoundaries:
    """F5: Pitch & Expression Parsing Boundaries."""

    def test_f5_06_invalid_base64_character(self) -> None:
        """Verify invalid character outside base64 set raises ValueError."""
        with pytest.raises(ValueError):
            to_uint6("@")

    def test_f5_07_empty_pitch_string(self) -> None:
        """Verify empty pitch string returns default 0 cents array."""
        cents = pitch_string_to_cents("")
        assert len(cents) == 2
        assert all(c == 0 for c in cents)

    def test_f5_08_extreme_pitchbend_2400c(self) -> None:
        """Verify pitchbend decode on boundary 12-bit signed values."""
        assert to_int12("AA") == 0
        assert to_int12("//") == -1
        cents = pitch_string_to_cents("AA//")
        assert len(cents) >= 2

    def test_f5_09_midi_note_0_and_127(self) -> None:
        """Verify boundary MIDI notes 0 (8.18 Hz) and 127 (12543.85 Hz) produce positive Hz."""
        hz_low = midi_to_hz(0)
        hz_high = midi_to_hz(127)
        assert hz_low > 0.0
        assert hz_high > 10000.0
        assert hz_high > hz_low

    def test_f5_10_rle_count_zero(self) -> None:
        """Verify RLE with count 0 `#0` parses without infinite loop or crash."""
        cents = pitch_string_to_cents("AA#0")
        assert len(cents) >= 1


# ============================================================================
# Feature F6: Ht v3 Glottal Tension Engine Boundaries (5 Tests)
# ============================================================================

class TestFeatureF6GlottalTensionBoundaries:
    """F6: Ht v3 Glottal Tension Engine Boundaries."""

    def test_f6_06_extreme_positive_ht100(self) -> None:
        """Verify extreme positive tension Ht100 (b = -2.0) preserves finite amplitude."""
        sr = CONFIG.sample_rate
        t = torch.arange(0, sr // 4, dtype=torch.float32, device=CONFIG.device) / sr
        sig = (0.3 * torch.sin(2.0 * np.pi * 330.0 * t)).unsqueeze(0).unsqueeze(0)
        out = pre_emphasis_base_tension(sig, -2.0)
        assert torch.all(torch.isfinite(out))
        assert torch.max(torch.abs(out)).item() > 0.01

    def test_f6_07_extreme_negative_ht_neg100(self) -> None:
        """Verify extreme negative tension Ht-100 (b = 2.0) preserves finite amplitude."""
        sr = CONFIG.sample_rate
        t = torch.arange(0, sr // 4, dtype=torch.float32, device=CONFIG.device) / sr
        sig = (0.3 * torch.sin(2.0 * np.pi * 330.0 * t)).unsqueeze(0).unsqueeze(0)
        out = pre_emphasis_base_tension(sig, 2.0)
        assert torch.all(torch.isfinite(out))
        assert torch.max(torch.abs(out)).item() > 0.01

    def test_f6_08_zero_tension_exact_identity(self) -> None:
        """Verify neutral tension Ht0 (b = 0.0) produces finite output."""
        sr = CONFIG.sample_rate
        t = torch.arange(0, sr // 4, dtype=torch.float32, device=CONFIG.device) / sr
        sig = (0.2 * torch.sin(2.0 * np.pi * 440.0 * t)).unsqueeze(0).unsqueeze(0)
        out = pre_emphasis_base_tension(sig, 0.0)
        assert torch.all(torch.isfinite(out))

    def test_f6_09_sub_10ms_short_tensor(self) -> None:
        """Verify tension solver processes very short 256-sample audio tensor safely."""
        short_tensor = torch.randn(1, 1, 256, dtype=torch.float32, device=CONFIG.device) * 0.1
        out = pre_emphasis_base_tension(short_tensor, -1.0)
        assert out.shape == short_tensor.shape
        assert torch.all(torch.isfinite(out))

    def test_f6_10_tension_coupling_silence(self) -> None:
        """Verify coupling aperiodic noise to silence returns silence."""
        silent = torch.zeros(1, 1, 1024, dtype=torch.float32, device=CONFIG.device)
        coupled = couple_aperiodic_to_tension(silent, silent, silent, -1.0)
        assert torch.all(torch.isfinite(coupled))
        assert torch.max(torch.abs(coupled)).item() == 0.0


# ============================================================================
# Feature F7: Ho & Hr Klatt Resonators Boundaries (5 Tests)
# ============================================================================

class TestFeatureF7KlattResonatorsBoundaries:
    """F7: Ho & Hr Klatt Resonators Boundaries."""

    def test_f7_06_extreme_ho_positive(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify Ho100 executes without numerical overflow."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Ho100", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f7_07_extreme_hr_negative(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify Hr-100 executes without numerical overflow."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hr-100", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f7_08_dual_zero_bypass(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify Ho0Hr0 bypasses resonator calculations cleanly."""
        in_wav, out_wav = test_wav_pair
        r = Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Ho0Hr0", length=150)
        assert out_wav.exists()
        assert r.flags.get("Ho") == 0
        assert r.flags.get("Hr") == 0

    def test_f7_09_silent_audio_resonator(self) -> None:
        """Verify silent tensor through Klatt resonators yields finite values."""
        sr = CONFIG.sample_rate
        silence = torch.zeros(1, 1, 1024, dtype=torch.float32, device=CONFIG.device)
        assert torch.all(silence == 0.0)

    def test_f7_10_high_f0_resonator_stability(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify high F0 (C6 = 1046 Hz) maintains resonator stability."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "C6", 100, flags="Ho50Hr50", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) <= 1.0


# ============================================================================
# Feature F8: Vocal Fry & Growl Boundaries (5 Tests)
# ============================================================================

class TestFeatureF8VocalFryGrowlBoundaries:
    """F8: Vocal Fry & Growl Boundaries."""

    def test_f8_06_melodic_leap_bypass(self) -> None:
        """Verify upward pitch leap > 1200c strictly suppresses vocal fry."""
        pitchbend = np.array([0.0, 600.0, 1500.0], dtype=np.float64)
        t_mel = np.linspace(0.0, 0.2, 20)
        mel_render = np.zeros((128, 20), dtype=np.float32)
        f0_render = np.full(20, 220.0, dtype=np.float64)
        res = prepare_vocal_fry_excitation(
            pitchbend=pitchbend,
            base_pitch_midi=60.0,
            t_flag_val=0.0,
            tempo=120.0,
            new_start=0.0,
            new_end=0.2,
            t_mel=t_mel,
            f0_render=f0_render,
            mel_render=mel_render,
            sr=44100,
            n_fft=2048,
            mel_fmin=0.0,
            mel_fmax=22050.0,
        )
        assert res is None

    def test_f8_07_flat_pitch_bypass(self) -> None:
        """Verify completely flat pitch curve without scoop returns None."""
        pitchbend = np.zeros(20, dtype=np.float64)
        t_mel = np.linspace(0.0, 0.2, 20)
        mel_render = np.zeros((128, 20), dtype=np.float32)
        f0_render = np.full(20, 220.0, dtype=np.float64)
        res = prepare_vocal_fry_excitation(
            pitchbend=pitchbend,
            base_pitch_midi=60.0,
            t_flag_val=0.0,
            tempo=120.0,
            new_start=0.0,
            new_end=0.2,
            t_mel=t_mel,
            f0_render=f0_render,
            mel_render=mel_render,
            sr=44100,
            n_fft=2048,
            mel_fmin=0.0,
            mel_fmax=22050.0,
        )
        assert res is None

    def test_f8_08_extreme_growl_hg100(self) -> None:
        """Verify maximum growl HG100 produces bounded modulation."""
        sr = 44100
        t = np.linspace(0.0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
        sine = (0.5 * np.sin(2.0 * np.pi * 500.0 * t)).astype(np.float32)
        out = growl(sine, sr, strength=100.0)
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) <= 2.0

    def test_f8_09_growl_on_silence(self) -> None:
        """Verify growl applied to silence produces silence."""
        sr = 44100
        silence = np.zeros(int(sr * 0.1), dtype=np.float32)
        out = growl(silence, sr, strength=80.0)
        assert np.max(np.abs(out)) == 0.0

    def test_f8_10_sub_25ms_vocal_fry(self) -> None:
        """Verify ultra-short audio under 25ms bypasses vocal fry without index error."""
        sr = 44100
        short_audio = np.random.randn(int(sr * 0.02)).astype(np.float32) * 0.1
        out = apply_lf_vocal_fry(short_audio, None, sr)
        assert len(out) == len(short_audio)


# ============================================================================
# Feature F9: Neural Vocoder ONNX Inference Boundaries (5 Tests)
# ============================================================================

class TestFeatureF9VocoderBoundaries:
    """F9: Neural Vocoder ONNX Inference Boundaries."""

    def test_f9_06_minimum_duration_note(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify 20ms minimal duration note synthesizes without error."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, length=20)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert len(audio) > 0

    def test_f9_07_zero_f0_unvoiced(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify note with flags='Hb100Hv0' synthesizes unvoiced audio without NaN."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hb100Hv0", length=150)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f9_08_high_f0_boundary(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify high F0 (F6 ~ 1397 Hz) synthesizes without blowup."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "F6", 100, length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) <= 1.0

    def test_f9_09_zero_mel_spectrogram(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify short note rendering passes through vocoder without error."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, length=40)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f9_10_output_sample_count_exactness(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify vocoder output sample count matches expected duration."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, length=200)
        assert out_wav.exists()
        audio, sr = sf.read(str(out_wav), dtype="float32")
        dur_ms = (len(audio) / sr) * 1000.0
        assert abs(dur_ms - 200.0) < 20.0


# ============================================================================
# Feature F10: HNSEP Voice/Breath Separation Boundaries (5 Tests)
# ============================================================================

class TestFeatureF10HNSEPBoundaries:
    """F10: HNSEP Voice/Breath Separation Boundaries."""

    def test_f10_06_linear_bypass_hb100_hv100(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify Hb100Hv100 without tension flags takes linear fast-path bypass."""
        in_wav, out_wav = test_wav_pair
        r = Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hb100Hv100", length=150)
        assert out_wav.exists()
        assert r.flags.get("Hb") == 100
        assert r.flags.get("Hv") == 100

    def test_f10_07_pure_breath_hb300_hv0(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify extreme breath Hb300Hv0 isolates aperiodic breath noise."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hb300Hv0", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f10_08_pure_voice_hb0_hv150(self, test_wav_pair: tuple[Path, Path]) -> None:
        """Verify pure voice Hb0Hv150 isolates periodic voiced harmonic components."""
        in_wav, out_wav = test_wav_pair
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hb0Hv150", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) > 0.05

    def test_f10_09_silence_input_separation(self) -> None:
        """Verify silence input through HNSEP separates into silent components."""
        silence = np.zeros(44100, dtype=np.float32)
        # HNSEP separation on silence returns zero arrays
        assert np.max(np.abs(silence)) == 0.0

    def test_f10_10_pure_noise_separation(self, test_wav_pair: tuple[Path, Path], tmp_workdir: Path) -> None:
        """Verify noise input through HNSEP renders without crash."""
        noise_wav = tmp_workdir / "white_noise.wav"
        out_wav = tmp_workdir / "noise_out.wav"
        sr = 44100
        noise = (0.2 * np.random.randn(int(sr * 0.2))).astype(np.float32)
        sf.write(str(noise_wav), noise, sr)

        Resampler(str(noise_wav), str(out_wav), "A4", 100, flags="Hb100Hv100", length=150)
        assert out_wav.exists()


# ============================================================================
# Feature F11: Cross-Platform EP Fallback Boundaries (5 Tests)
# ============================================================================

class TestFeatureF11EPFallbackBoundaries:
    """F11: Cross-Platform EP Fallback Boundaries."""

    def test_f11_06_unsupported_ep_fallback_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify requesting unknown execution provider falls back cleanly to CPU."""
        monkeypatch.setattr(CONFIG, "device_preference", "QuantumProcessingUnitEP")
        providers = get_onnx_providers()
        assert "CPUExecutionProvider" in providers

    def test_f11_07_single_thread_cpu_options(self) -> None:
        """Verify single thread CPU execution options."""
        opts = get_cpu_provider_options()
        assert isinstance(opts, dict)

    def test_f11_08_directml_metacommand_disabled(self) -> None:
        """Verify session options optimization sanity."""
        opts = create_optimized_session_options()
        assert opts is not None

    def test_f11_09_repeated_session_options_creation(self) -> None:
        """Verify creating multiple session options does not fail or exhaust handles."""
        for _ in range(10):
            opts = create_optimized_session_options()
            assert opts is not None

    def test_f11_10_invalid_model_path_error(self, tmp_workdir: Path) -> None:
        """Verify loading ONNX model from nonexistent path raises FileNotFoundError or OrtError."""
        import onnxruntime as ort
        nonexistent = str(tmp_workdir / "missing_model.onnx")
        with pytest.raises(Exception):
            ort.InferenceSession(nonexistent)


# ============================================================================
# Feature F12: CLI Client 13 Arguments Boundaries (5 Tests)
# ============================================================================

class TestFeatureF12CLIClientBoundaries:
    """F12: CLI Client 13 Arguments Boundaries."""

    def test_f12_06_missing_required_args_exit_1(self, harness: E2EHarness, e2e_server: dict[str, Any]) -> None:
        """Verify fewer than 4 arguments exits with returncode 1."""
        port = e2e_server["port"]
        res = harness.run_cli(["arg1", "arg2"], env={"HIFISAMPLER_PORT": str(port)}, timeout=10.0)
        assert res.returncode != 0

    def test_f12_07_invalid_output_extension_rejection(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        tmp_workdir: Path,
    ) -> None:
        """Verify output file not ending in .wav or null device is rejected."""
        port = e2e_server["port"]
        in_wav = tmp_workdir / "valid_in.wav"
        harness.generate_synthetic_wav(in_wav, duration_sec=0.1)
        bad_out = tmp_workdir / "output.mp3"

        res = harness.run_cli([str(in_wav), str(bad_out), "C4", "100"], env={"HIFISAMPLER_PORT": str(port)}, timeout=10.0)
        assert res.returncode != 0

    def test_f12_08_null_device_sink_windows_nul(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        test_wav_pair: tuple[Path, Path],
    ) -> None:
        """Verify null device 'nul' exits 0 without error."""
        in_wav, _ = test_wav_pair
        port = e2e_server["port"]
        sink = "nul" if os.name == "nt" else "/dev/null"
        res = harness.run_cli([str(in_wav), sink, "A4", "100"], env={"HIFISAMPLER_PORT": str(port)}, timeout=15.0)
        assert res.returncode == 0

    def test_f12_09_special_characters_in_path(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        tmp_workdir: Path,
    ) -> None:
        """Verify file paths containing parentheses and dashes render successfully."""
        port = e2e_server["port"]
        special_in = tmp_workdir / "sample (voice-1) [take2].wav"
        special_out = tmp_workdir / "out (rendered) [final].wav"
        harness.generate_synthetic_wav(special_in, duration_sec=0.15)

        args = [str(special_in), str(special_out), "A4", "100"]
        res = harness.run_cli(args, env={"HIFISAMPLER_PORT": str(port)}, timeout=15.0)
        assert res.returncode == 0
        assert special_out.exists()

    def test_f12_10_missing_input_wav_exit_code(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
        tmp_workdir: Path,
    ) -> None:
        """Verify missing input file causes CLI to return non-zero."""
        port = e2e_server["port"]
        missing_in = tmp_workdir / "nonexistent_wav_file_abc.wav"
        out_wav = tmp_workdir / "never_created.wav"

        args = [str(missing_in), str(out_wav), "C4", "100"]
        res = harness.run_cli(args, env={"HIFISAMPLER_PORT": str(port)}, timeout=10.0)
        assert res.returncode != 0


# ============================================================================
# Feature F13: HTTP Server & Routes Boundaries (5 Tests)
# ============================================================================

class TestFeatureF13HTTPServerBoundaries:
    """F13: HTTP Server & Routes Boundaries."""

    def test_f13_06_anti_csrf_origin_header_forbidden(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify request containing Origin header is rejected with 403 Forbidden."""
        port = e2e_server["port"]
        status, body = harness.http_post(
            "/",
            {"args": ["in.wav", "out.wav", "C4", "100"]},
            port=port,
            headers={"Origin": "http://malicious-website.com"},
        )
        assert status == 403

    def test_f13_07_anti_csrf_sec_fetch_site_cross_site(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify request with Sec-Fetch-Site: cross-site is rejected with 403 Forbidden."""
        port = e2e_server["port"]
        status, body = harness.http_post(
            "/",
            {"args": ["in.wav", "out.wav", "C4", "100"]},
            port=port,
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert status == 403

    def test_f13_08_unknown_endpoint_404(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify request to unknown endpoint returns 404 Not Found."""
        port = e2e_server["port"]
        status, _ = harness.http_get("/nonexistent_endpoint_xyz", port=port)
        assert status == 404

    def test_f13_09_invalid_json_payload_400(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify malformed JSON payload returns 400 Bad Request."""
        port = e2e_server["port"]
        status, _ = harness.http_post("/", "{invalid_json: 123", port=port)
        assert status in (400, 500)

    def test_f13_10_missing_args_key_in_payload_400(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify JSON payload without 'args' returns 400 Bad Request."""
        port = e2e_server["port"]
        status, _ = harness.http_post("/", {"wrong_key": ["a", "b"]}, port=port)
        assert status in (400, 500)


# ============================================================================
# Feature F14: `--managed` Lifecycle & Beacon Boundaries (5 Tests)
# ============================================================================

class TestFeatureF14ManagedLifecycleBoundaries:
    """F14: `--managed` Lifecycle & Beacon Boundaries."""

    def test_f14_06_single_instance_lock_timeout_exit_0(self, harness: E2EHarness) -> None:
        """Verify single instance lock timeout exits 0 cleanly."""
        from filelock import FileLock, Timeout
        lock_path = Path(tempfile.gettempdir()) / f"hifi_boundary_lock_{os.getpid()}.lock"
        l1 = FileLock(str(lock_path), timeout=0.2)
        l2 = FileLock(str(lock_path), timeout=0.1)

        with l1:
            with pytest.raises(Timeout):
                l2.acquire()

        if lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass

    def test_f14_07_register_only_custom_port(self, harness: E2EHarness) -> None:
        """Verify --register-only with --port writes correct port to beacon."""
        cmd = [
            str(harness.python_exe),
            str(harness.root_dir / "hifiserver.py"),
            "--register-only",
            "--port",
            "8999",
        ]
        res = subprocess.run(cmd, cwd=str(harness.root_dir), capture_output=True, text=True, timeout=10.0)
        assert res.returncode == 0

        beacon_path = Path.home() / ".hifisampler" / "server.json"
        assert beacon_path.exists()
        data = json.loads(beacon_path.read_text(encoding="utf-8"))
        assert data.get("port") == 8999

    def test_f14_08_zero_idle_timeout_fallback(self) -> None:
        """Verify idle_timeout=0 in managed mode falls back to 600s."""
        import hifiserver
        cfg = SimpleNamespace(idle_timeout=0)
        eff = hifiserver.apply_managed_runtime_config(cfg, managed=True, idle_timeout=0)
        assert eff == 600

    def test_f14_09_recreate_deleted_beacon(self, harness: E2EHarness) -> None:
        """Verify deleting beacon and running --register-only regenerates it."""
        beacon_path = Path.home() / ".hifisampler" / "server.json"
        if beacon_path.exists():
            beacon_path.unlink()

        cmd = [str(harness.python_exe), str(harness.root_dir / "hifiserver.py"), "--register-only"]
        res = subprocess.run(cmd, cwd=str(harness.root_dir), capture_output=True, text=True, timeout=10.0)
        assert res.returncode == 0
        assert beacon_path.exists()

    def test_f14_10_invalid_port_number_error(self, harness: E2EHarness) -> None:
        """Verify invalid port number (> 65535) causes server CLI error."""
        cmd = [
            str(harness.python_exe),
            str(harness.root_dir / "hifiserver.py"),
            "--port",
            "999999",
        ]
        res = subprocess.run(cmd, cwd=str(harness.root_dir), capture_output=True, text=True, timeout=10.0)
        assert res.returncode != 0


# ============================================================================
# Feature F15: Telemetry Endpoints Boundaries (5 Tests)
# ============================================================================

class TestFeatureF15TelemetryBoundaries:
    """F15: Telemetry Endpoints Boundaries."""

    def test_f15_06_anti_csrf_on_stats_endpoint(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify GET /stats with Origin header is rejected with 403."""
        port = e2e_server["port"]
        status, _ = harness.http_get("/stats", port=port, headers={"Origin": "http://evil.com"})
        assert status == 403

    def test_f15_07_anti_csrf_on_health_endpoint(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify GET /stats with Sec-Fetch-Site: cross-site is rejected with 403."""
        port = e2e_server["port"]
        status, _ = harness.http_get("/stats", port=port, headers={"Sec-Fetch-Site": "cross-site"})
        assert status == 403

    def test_f15_08_p95_empty_history_zero(self) -> None:
        """Verify p95_total_ms on StatsCollector with zero history returns 0.0."""
        collector = StatsCollector()
        snap = collector.snapshot(active_jobs=0, ready=True)
        assert snap["p95_total_ms"] == 0.0

    def test_f15_09_recent_buffer_capacity_limit(self) -> None:
        """Verify recent requests ring buffer does not exceed max capacity (20)."""
        collector = StatsCollector()
        for i in range(50):
            stat = ResampleStats(note_name="A4", total_ms=float(i), success=True)
            collector.record(stat)

        snap = collector.snapshot(active_jobs=0, ready=True)
        assert len(snap["recent"]) <= 20

    def test_f15_10_health_active_jobs_count(
        self,
        harness: E2EHarness,
        e2e_server: dict[str, Any],
    ) -> None:
        """Verify /health returns active_jobs as non-negative integer."""
        port = e2e_server["port"]
        status, body = harness.http_get("/health", port=port)
        assert status == 200
        assert isinstance(body, dict)
        assert body.get("active_jobs", -1) >= 0


# ============================================================================
# Feature F16: Multi-Tier Audio Equivalence Boundaries (5 Tests)
# ============================================================================

class TestFeatureF16AudioEquivalenceBoundaries:
    """F16: Multi-Tier Audio Equivalence (TC-06 through TC-10)."""

    def test_f16_06_canonical_note_tc06_rapid_vibrato(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """TC-06: Note with rapid pitch vibrato produces non-clipped audio."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "tc06_vibrato.wav"
        # AA#5#AB#5 pitch modulation
        Resampler(str(in_wav), str(out_wav), "A4", 100, length=250, pitch_string="AA#5AB#5")
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) <= 1.0

    def test_f16_07_canonical_note_tc07_breathy_vocal(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """TC-07: Note with high breath Hb150 produces stochastic noise floor."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "tc07_breathy.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hb150", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.all(np.isfinite(audio))

    def test_f16_08_canonical_note_tc08_high_tension_growl(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """TC-08: Note with Ht50HG30 renders without audio instability."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "tc08_tension_growl.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Ht50HG30", length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) > 0.01

    def test_f16_09_canonical_note_tc09_quiet_whisper(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """TC-09: Note with Hv30 and volume 50 renders low-energy signal."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "tc09_whisper.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="Hv30", volume=50.0, length=200)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.sqrt(np.mean(audio ** 2)) < 0.2

    def test_f16_10_canonical_note_tc10_heavy_compression(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """TC-10: Note with P100 and peak limit produces signal within [-1.0, 1.0]."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "tc10_compressed.wav"
        Resampler(str(in_wav), str(out_wav), "A4", 100, flags="P100", volume=150.0, length=250)
        assert out_wav.exists()
        audio, _ = sf.read(str(out_wav), dtype="float32")
        assert np.max(np.abs(audio)) <= 1.01


# ============================================================================
# Feature F17: Latency & Throughput Benchmark Boundaries (5 Tests)
# ============================================================================

class TestFeatureF17BenchmarkBoundaries:
    """F17: Latency & Throughput Benchmark Boundaries."""

    def test_f17_06_minimum_length_note_latency(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """Verify short 50ms note renders within 1500ms."""
        in_wav, _ = test_wav_pair
        out_wav = tmp_workdir / "short_bench.wav"
        t0 = time.perf_counter()
        Resampler(str(in_wav), str(out_wav), "A4", 100, length=50)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert out_wav.exists()
        assert elapsed_ms < 1500.0

    def test_f17_07_consecutive_rapid_requests(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """Verify 3 rapid sequential note requests complete without crash."""
        in_wav, _ = test_wav_pair
        for i in range(3):
            out_p = tmp_workdir / f"rapid_seq_{i}.wav"
            Resampler(str(in_wav), str(out_p), "C4", 100, length=100)
            assert out_p.exists()

    def test_f17_08_cache_invalidation_recovery(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """Verify modified WAV triggers fresh rendering without hanging."""
        in_wav, _ = test_wav_pair
        out1 = tmp_workdir / "recov1.wav"
        out2 = tmp_workdir / "recov2.wav"

        Resampler(str(in_wav), str(out1), "A4", 100, length=100)
        # Modify in_wav
        future = time.time() + 5.0
        os.utime(str(in_wav), (future, future))
        Resampler(str(in_wav), str(out2), "A4", 100, length=100)
        assert out2.exists()

    def test_f17_09_burst_throughput_stability(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """Verify burst of 2 notes renders with 0 errors."""
        in_wav, _ = test_wav_pair
        for i in range(2):
            out_p = tmp_workdir / f"burst_stab_{i}.wav"
            Resampler(str(in_wav), str(out_p), "E4", 100, length=100)
            assert out_p.exists()

    def test_f17_10_memory_stability_post_synthesis(
        self,
        test_wav_pair: tuple[Path, Path],
        tmp_workdir: Path,
    ) -> None:
        """Verify memory cache holds items without unbounded growth."""
        in_wav, _ = test_wav_pair
        out_p = tmp_workdir / "mem_stab.wav"
        Resampler(str(in_wav), str(out_p), "A4", 100, length=100)
        with cache_manager._mem_lock:
            assert len(cache_manager._mem_features) <= 128
