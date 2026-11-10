import copy
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from config import CONFIG
from util.cache_manager import cache_manager


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings", "ignore::pytest_asyncio.plugin.PytestDeprecationWarning"
    )


@pytest.fixture
def sample_rate():
    return 44100


@pytest.fixture
def synthetic_sine_wave(sample_rate):
    """Generate 1.0 second of pure 440Hz sine wave at 44.1kHz in memory."""
    duration = 1.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)
    # Amplitude 0.5 to stay well below clipping
    sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    return sine.astype(np.float32)


@pytest.fixture
def synthetic_silence(sample_rate):
    """Generate 1.0 second of silence at 44.1kHz in memory."""
    duration = 1.0
    return np.zeros(int(sample_rate * duration), dtype=np.float32)


@pytest.fixture
def synthetic_noise(sample_rate):
    """Generate 1.0 second of pseudo-random white noise in memory."""
    duration = 1.0
    rng = np.random.default_rng(seed=42)
    noise = rng.uniform(-0.2, 0.2, int(sample_rate * duration)).astype(np.float32)
    return noise


@pytest.fixture
def synthetic_wav_file(tmp_path, synthetic_sine_wave, sample_rate):
    """Create a temporary on-disk WAV file containing synthetic sine wave."""
    wav_path = tmp_path / "test_input.wav"
    sf.write(str(wav_path), synthetic_sine_wave, sample_rate, subtype="PCM_16", format="WAV")
    yield wav_path
    if wav_path.exists():
        try:
            wav_path.unlink()
        except OSError:
            pass


@pytest.fixture
def synthetic_wav_pair(tmp_path, synthetic_sine_wave, sample_rate):
    """Create input WAV file and output WAV path pair in isolated directory."""
    in_path = tmp_path / "note_in.wav"
    out_path = tmp_path / "note_out.wav"
    sf.write(str(in_path), synthetic_sine_wave, sample_rate, subtype="PCM_16", format="WAV")
    yield in_path, out_path
    for p in (in_path, out_path):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


@pytest.fixture
def clean_cache():
    """Ensure in-memory caches and locks are cleared before and after test."""
    with cache_manager._mem_lock:
        cache_manager._mem_features.clear()
        cache_manager._mem_hnsep.clear()
        cache_manager._mem_wav.clear()
    with cache_manager._locks_lock:
        cache_manager._locks.clear()
    yield
    with cache_manager._mem_lock:
        cache_manager._mem_features.clear()
        cache_manager._mem_hnsep.clear()
        cache_manager._mem_wav.clear()
    with cache_manager._locks_lock:
        cache_manager._locks.clear()


@pytest.fixture
def restore_config():
    """Preserve and restore CONFIG attributes across test runs."""
    saved_attrs = {
        "device": CONFIG.device,
        "device_preference": getattr(CONFIG, "device_preference", "auto"),
        "model_type": getattr(CONFIG, "model_type", "onnx"),
        "max_workers": getattr(CONFIG, "max_workers", 2),
        "fp16": getattr(CONFIG, "fp16", False),
        "wave_norm": getattr(CONFIG, "wave_norm", True),
        "loop_mode": getattr(CONFIG, "loop_mode", False),
    }
    yield
    for k, v in saved_attrs.items():
        setattr(CONFIG, k, v)
