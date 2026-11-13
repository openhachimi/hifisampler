import os
from unittest.mock import patch
import numpy as np
import pytest

from util.cache_manager import (
    CacheManager,
    _BIN_HEADER,
    _BIN_MAGIC,
    _BIN_VERSION,
    resolve_feature_cache_paths,
)


class TestBinaryCacheFormatAndRoundtrip:
    """Verify HIFIBIN1 header layout, zero-ZIP roundtrip, and immutability."""

    def test_header_constant_size_is_28_bytes(self):
        assert _BIN_HEADER.size == 28
        packed = _BIN_HEADER.pack(_BIN_MAGIC, _BIN_VERSION, 128, 50, 0.75)
        assert len(packed) == 28
        assert packed[:8] == b"HIFIBIN1"

    def test_binary_roundtrip_exact_values_and_freezing(self, tmp_path):
        cm_writer = CacheManager()
        cache_path = tmp_path / "note.hifi.bin"
        mel = np.random.default_rng(42).standard_normal((128, 64), dtype=np.float32)
        features = {"mel_origin": mel, "scale": 0.625}

        saved = cm_writer.save_features_cache(cache_path, features)
        assert cache_path.exists()
        assert cache_path.stat().st_size == _BIN_HEADER.size + 128 * 64 * 4
        assert cache_path.read_bytes()[:8] == _BIN_MAGIC
        assert np.array_equal(saved["mel_origin"], mel)
        assert saved["scale"] == pytest.approx(0.625)
        assert not saved["mel_origin"].flags.writeable

        # Cold-start reader verifies real disk deserialization without np.load
        cm_reader = CacheManager()
        with patch("numpy.load", side_effect=AssertionError("np.load must not be called for HIFIBIN1")):
            loaded = cm_reader.load_features_cache(cache_path)

        assert loaded is not None
        assert loaded["mel_origin"].shape == (128, 64)
        assert loaded["mel_origin"].dtype == np.float32
        assert np.array_equal(loaded["mel_origin"], mel)
        assert loaded["scale"] == pytest.approx(0.625, rel=0, abs=0)
        assert not loaded["mel_origin"].flags.writeable

    def test_non_contiguous_mel_and_numpy_scalar_scale(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "non_contig.hifi.bin"
        base = np.arange(20 * 12, dtype=np.float32).reshape(12, 20)
        mel_t = base.T[:, ::2]  # Fortran/strided non-contiguous view
        assert not mel_t.flags.c_contiguous

        features = {"mel_origin": mel_t, "scale": np.array(0.33333334, dtype=np.float32)}
        cm.save_features_cache(cache_path, features)

        cm_cold = CacheManager()
        loaded = cm_cold.load_features_cache(cache_path)
        assert loaded is not None
        assert loaded["mel_origin"].flags.c_contiguous
        assert np.array_equal(loaded["mel_origin"], mel_t)
        assert loaded["scale"] == pytest.approx(float(features["scale"]))

    def test_empty_frame_dimension_roundtrip(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "empty_frames.hifi.bin"
        mel = np.zeros((128, 0), dtype=np.float32)
        cm.save_features_cache(cache_path, {"mel_origin": mel, "scale": 1.0})

        loaded = CacheManager().load_features_cache(cache_path)
        assert loaded is not None
        assert loaded["mel_origin"].shape == (128, 0)
        assert loaded["scale"] == 1.0


class TestBinaryCacheCorruptionAndErrors:
    """Verify corrupted/truncated files return None and PermissionError falls back to RAM."""

    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            b"HIFIBIN",
            b"HIFIBIN1\x01\x00\x00\x00",  # truncated header (< 28 bytes)
            _BIN_HEADER.pack(_BIN_MAGIC, 99, 2, 2, 1.0) + b"\x00" * 16,  # bad version
            _BIN_HEADER.pack(_BIN_MAGIC, _BIN_VERSION, 10, 10, 1.0) + b"\x00" * 16,  # truncated body
            _BIN_HEADER.pack(_BIN_MAGIC, _BIN_VERSION, 2, 2, 1.0) + b"\x00" * 20,  # trailing garbage
            b"not-a-valid-cache-file-at-all",
        ],
    )
    def test_corrupted_or_truncated_binary_returns_none(self, tmp_path, payload):
        cm = CacheManager()
        cache_path = tmp_path / "broken.hifi.bin"
        cache_path.write_bytes(payload)

        assert cm.load_features_cache(cache_path) is None

    def test_permission_error_falls_back_to_memory_and_cleans_tmp(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "readonly.hifi.bin"
        features = {"mel_origin": np.ones((8, 8), dtype=np.float32), "scale": 0.5}

        with patch("os.replace", side_effect=PermissionError("Mock Permission Denied")):
            saved = cm.save_features_cache(cache_path, features)

        assert saved is not None
        assert np.array_equal(saved["mel_origin"], features["mel_origin"])
        assert saved["scale"] == 0.5
        assert not cache_path.exists()
        # No leftover .tmp files
        assert list(tmp_path.glob("*.tmp")) == []

        # L1 RAM fallback still serves the features
        loaded = cm.load_features_cache(cache_path)
        assert loaded is not None
        assert np.array_equal(loaded["mel_origin"], features["mel_origin"])


class TestBinaryCacheInvalidationAndFallback:
    """Verify source_path mtime invalidation, .npz fallback, and .bin vs .npz routing."""

    def test_source_wav_mtime_invalidates_binary_cache(self, tmp_path):
        cm = CacheManager()
        src_wav = tmp_path / "voice.wav"
        src_wav.write_bytes(b"old")
        bin_path = tmp_path / "voice.hifi.bin"

        old_feat = {"mel_origin": np.ones((4, 4), dtype=np.float32), "scale": 1.0}
        cm.save_features_cache(bin_path, old_feat, source_path=src_wav)
        assert cm.load_features_cache(bin_path, source_path=src_wav) is not None

        # Simulate stale cache whose mtime predates source_path
        past_ns = src_wav.stat().st_mtime_ns - 1_000_000_000
        os.utime(bin_path, ns=(past_ns, past_ns))

        assert cm.load_features_cache(bin_path, source_path=src_wav) is None

        new_feat = {"mel_origin": np.full((4, 4), 3.0, dtype=np.float32), "scale": 0.25}
        saved = cm.save_features_cache(bin_path, new_feat, force_generate=False, source_path=src_wav)
        assert np.array_equal(saved["mel_origin"], new_feat["mel_origin"])
        reloaded = CacheManager().load_features_cache(bin_path, source_path=src_wav)
        assert reloaded is not None
        assert reloaded["scale"] == 0.25

    def test_resolve_feature_cache_paths_and_legacy_npz_fallback(self, tmp_path):
        bin_path = tmp_path / "sample.hifi.bin"
        npz_path = tmp_path / "sample.hifi.npz"

        # Neither exists -> (bin, bin)
        assert resolve_feature_cache_paths(bin_path) == (bin_path, bin_path)

        # Only .npz exists -> read falls back to .npz, write targets .bin
        cm = CacheManager()
        legacy_feat = {"mel_origin": np.full((6, 5), 2.5, dtype=np.float32), "scale": 0.8}
        cm.save_features_cache(npz_path, legacy_feat)

        read_p, write_p = resolve_feature_cache_paths(bin_path)
        assert read_p == npz_path
        assert write_p == bin_path

        # load_features_cache(bin_path) smoothly falls back to existing .npz
        loaded_fallback = CacheManager().load_features_cache(bin_path)
        assert loaded_fallback is not None
        assert np.array_equal(loaded_fallback["mel_origin"], legacy_feat["mel_origin"])
        assert loaded_fallback["scale"] == pytest.approx(0.8)

        # save_features_cache(bin_path) still writes .bin even when .npz exists
        new_feat = {"mel_origin": np.full((6, 5), 9.0, dtype=np.float32), "scale": 0.4}
        cm.save_features_cache(bin_path, new_feat)
        assert bin_path.exists()
        assert resolve_feature_cache_paths(bin_path) == (bin_path, bin_path)
        loaded_bin = CacheManager().load_features_cache(bin_path)
        assert np.array_equal(loaded_bin["mel_origin"], new_feat["mel_origin"])

    def test_bin_vs_npz_explicit_routing_and_auto_probe(self, tmp_path):
        cm = CacheManager()
        mel = np.arange(12, dtype=np.float32).reshape(3, 4)
        features = {"mel_origin": mel, "scale": 0.5}

        # .npz default stays NPZ
        npz_default = tmp_path / "default.hifi.npz"
        cm.save_features_cache(npz_default, features)
        assert npz_default.read_bytes()[:2] == b"PK"

        # .npz with binary=True writes HIFIBIN1, auto-probed on load
        npz_forced_bin = tmp_path / "forced_bin.hifi.npz"
        cm.save_features_cache(npz_forced_bin, features, binary=True)
        assert npz_forced_bin.read_bytes()[:8] == _BIN_MAGIC
        with patch("numpy.load", side_effect=AssertionError("Should auto-probe HIFIBIN1")):
            loaded = CacheManager().load_features_cache(npz_forced_bin)
        assert np.array_equal(loaded["mel_origin"], mel)

        # .bin with binary=False writes NPZ, auto-probed on load
        bin_forced_npz = tmp_path / "forced_npz.hifi.bin"
        cm.save_features_cache(bin_forced_npz, features, binary=False)
        assert bin_forced_npz.read_bytes()[:2] == b"PK"
        loaded_npz = CacheManager().load_features_cache(bin_forced_npz)
        assert np.array_equal(loaded_npz["mel_origin"], mel)
        assert loaded_npz["scale"] == pytest.approx(0.5)

        # Non-standard extra keys on .bin fall back safely to NPZ
        non_std_bin = tmp_path / "extra.hifi.bin"
        cm.save_features_cache(non_std_bin, {"mel_origin": mel, "scale": 0.5, "extra": np.array([1])})
        assert non_std_bin.read_bytes()[:2] == b"PK"
        loaded_extra = CacheManager().load_features_cache(non_std_bin)
        assert np.array_equal(loaded_extra["extra"], np.array([1]))
