from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch
import numpy as np
import pytest
import torch

from util.cache_manager import (
    CacheManager,
    _MAX_FEATURES_RAM_ITEMS,
    _MAX_HNSEP_RAM_ITEMS,
    _MAX_WAV_RAM_ITEMS,
    cache_manager,
)


class TestMemoryLRUEviction:
    """Test in-memory LRU eviction policy across all cache tiers."""

    def test_features_lru_eviction(self):
        cm = CacheManager()
        # Insert _MAX_FEATURES_RAM_ITEMS items
        for i in range(_MAX_FEATURES_RAM_ITEMS):
            key = f"feature_{i}"
            cm._put_mem_features(key, None, {"val": np.array([i], dtype=np.float32)})

        assert len(cm._mem_features) == _MAX_FEATURES_RAM_ITEMS
        assert "feature_0" in cm._mem_features

        # Access feature_0 to bump its recency
        with cm._mem_lock:
            cm._mem_features.move_to_end("feature_0")

        # Insert one more item; feature_1 should be evicted instead of feature_0
        cm._put_mem_features("feature_overflow", None, {"val": np.array([999], dtype=np.float32)})

        assert len(cm._mem_features) == _MAX_FEATURES_RAM_ITEMS
        assert "feature_0" in cm._mem_features
        assert "feature_1" not in cm._mem_features
        assert "feature_overflow" in cm._mem_features

    def test_hnsep_lru_eviction(self):
        cm = CacheManager()
        for i in range(_MAX_HNSEP_RAM_ITEMS):
            key = f"hnsep_{i}"
            cm._put_mem_hnsep(key, None, torch.tensor([float(i)]))

        assert len(cm._mem_hnsep) == _MAX_HNSEP_RAM_ITEMS
        assert "hnsep_0" in cm._mem_hnsep

        # Insert overflow item
        cm._put_mem_hnsep("hnsep_overflow", None, torch.tensor([999.0]))
        assert len(cm._mem_hnsep) == _MAX_HNSEP_RAM_ITEMS
        assert "hnsep_0" not in cm._mem_hnsep
        assert "hnsep_overflow" in cm._mem_hnsep

    def test_wav_lru_eviction(self, tmp_path):
        cm = CacheManager()
        files = []
        for i in range(_MAX_WAV_RAM_ITEMS + 1):
            p = tmp_path / f"test_{i}.wav"
            p.write_bytes(b"dummy")
            files.append(p)

        # Fill up to max
        for i in range(_MAX_WAV_RAM_ITEMS):
            cm.save_wav_cache(files[i], np.array([float(i)], dtype=np.float32))

        assert len(cm._mem_wav) == _MAX_WAV_RAM_ITEMS
        first_key = str(files[0])
        assert first_key in cm._mem_wav

        # Insert one more
        cm.save_wav_cache(files[-1], np.array([999.0], dtype=np.float32))
        assert len(cm._mem_wav) == _MAX_WAV_RAM_ITEMS
        assert first_key not in cm._mem_wav
        assert str(files[-1]) in cm._mem_wav


class TestWavCacheBehavior:
    """Test audio waveform caching, stat signature verification, and immutability."""

    def test_save_and_load_wav_cache(self, tmp_path):
        cm = CacheManager()
        wav_file = tmp_path / "sample.wav"
        wav_file.write_bytes(b"dummy_wav_data")

        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cm.save_wav_cache(wav_file, data)

        loaded = cm.load_wav_cache(wav_file)
        assert loaded is not None
        assert np.allclose(loaded, data)

        # Ensure returned copy is mutable without corrupting internal cache
        loaded[0] = 999.0
        reloaded = cm.load_wav_cache(wav_file)
        assert reloaded[0] == pytest.approx(0.1)

    def test_file_modification_invalidates_cache(self, tmp_path):
        cm = CacheManager()
        wav_file = tmp_path / "sample_mod.wav"
        wav_file.write_bytes(b"initial_data")

        data = np.array([1.0, 2.0], dtype=np.float32)
        cm.save_wav_cache(wav_file, data)
        assert cm.load_wav_cache(wav_file) is not None

        # Modify file size/mtime
        time.sleep(0.01)
        wav_file.write_bytes(b"modified_much_longer_data")

        # Cache should detect signature mismatch and return None
        assert cm.load_wav_cache(wav_file) is None

    def test_deleted_file_returns_none(self, tmp_path):
        cm = CacheManager()
        wav_file = tmp_path / "to_delete.wav"
        wav_file.write_bytes(b"temp")

        cm.save_wav_cache(wav_file, np.array([1.0], dtype=np.float32))
        wav_file.unlink()

        assert cm.load_wav_cache(wav_file) is None


class TestFeaturesCachePersistence:
    """Test disk NPZ persistence, feature freezing, and fallback safety."""

    def test_features_save_and_load_roundtrip(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "test_note.hifi.npz"

        features = {
            "mel_origin": np.random.randn(128, 50).astype(np.float32),
            "scale": np.array(0.5, dtype=np.float32),
        }

        saved = cm.save_features_cache(cache_path, features)
        assert cache_path.exists()
        assert np.allclose(saved["mel_origin"], features["mel_origin"])
        assert saved["scale"] == pytest.approx(0.5)

        # Load from disk
        loaded = cm.load_features_cache(cache_path)
        assert loaded is not None
        assert np.allclose(loaded["mel_origin"], features["mel_origin"])
        assert loaded["scale"] == pytest.approx(0.5)

    def test_feature_freezing_prevents_in_place_corruption(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        frozen = CacheManager._freeze_features({"mel": arr})
        assert not frozen["mel"].flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            frozen["mel"][0] = 99.0

    def test_force_generate_bypasses_cache(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "bypass.hifi.npz"
        features = {"mel_origin": np.ones((10, 10), dtype=np.float32), "scale": 1.0}
        cm.save_features_cache(cache_path, features)

        # With force_generate=True, load returns None
        assert cm.load_features_cache(cache_path, force_generate=True) is None

    def test_corrupted_npz_graceful_recovery(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "corrupted.hifi.npz"
        cache_path.write_bytes(b"Not an NPZ file header at all!")

        # Should log warning and return None, rather than raising exception
        loaded = cm.load_features_cache(cache_path)
        assert loaded is None

    def test_disk_write_permission_error_falls_back_to_memory(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "readonly.hifi.npz"
        features = {"mel_origin": np.zeros((4, 4), dtype=np.float32), "scale": 1.0}

        # Mock np.savez to raise PermissionError
        with patch("numpy.savez", side_effect=PermissionError("Mock Permission Denied")):
            saved = cm.save_features_cache(cache_path, features)
            # Should not crash; returns in-memory features
            assert saved is not None
            assert np.allclose(saved["mel_origin"], features["mel_origin"])


class TestHnsepCachePersistence:
    """Test HN-SEP PyTorch tensor cache operations."""

    def test_hnsep_save_and_load_roundtrip(self, tmp_path):
        cm = CacheManager()
        cache_path = tmp_path / "sample_hnsep"
        tensor_data = torch.randn(1, 1, 1024, dtype=torch.float32)

        saved = cm.save_hnsep_cache(cache_path, tensor_data, device="cpu")
        assert cache_path.exists()
        assert torch.allclose(saved, tensor_data)

        loaded = cm.load_hnsep_cache(cache_path, device="cpu")
        assert loaded is not None
        assert torch.allclose(loaded, tensor_data)

        # Mutating returned tensor in-place on CPU must not corrupt the cached tensor
        loaded.zero_()
        reloaded = cm.load_hnsep_cache(cache_path, device="cpu")
        assert torch.allclose(reloaded, tensor_data)

    def test_source_wav_mtime_invalidates_stale_caches(self, tmp_path):
        cm = CacheManager()
        src_wav = tmp_path / "source.wav"
        src_wav.write_bytes(b"old_wav")
        feat_path = tmp_path / "source.hifi.npz"
        hnsep_path = tmp_path / "source_hnsep"

        cm.save_features_cache(feat_path, {"mel_origin": np.ones((4, 4), dtype=np.float32), "scale": 1.0})
        cm.save_hnsep_cache(hnsep_path, torch.ones(1, 1, 16), device="cpu")

        assert cm.load_features_cache(feat_path, source_path=src_wav) is not None
        assert cm.load_hnsep_cache(hnsep_path, device="cpu", source_path=src_wav) is not None

        # Touch source.wav with a strictly newer mtime
        future_ns = feat_path.stat().st_mtime_ns + 1_000_000_000
        os.utime(src_wav, ns=(future_ns, future_ns))

        assert cm.load_features_cache(feat_path, source_path=src_wav) is None
        assert cm.load_hnsep_cache(hnsep_path, device="cpu", source_path=src_wav) is None

        # Saving new features/hnsep with force_generate=False and source_path=src_wav must overwrite stale disk files
        new_feat = {"mel_origin": np.full((4, 4), 7.0, dtype=np.float32), "scale": 2.0}
        new_hnsep = torch.full((1, 1, 16), 9.0)
        saved_feat = cm.save_features_cache(feat_path, new_feat, force_generate=False, source_path=src_wav)
        saved_hnsep = cm.save_hnsep_cache(hnsep_path, new_hnsep, device="cpu", force_generate=False, source_path=src_wav)
        assert np.allclose(saved_feat["mel_origin"], 7.0)
        assert torch.allclose(saved_hnsep, new_hnsep)


class TestConcurrencyAndLocks:
    """Test re-entrant locks and multi-threaded race safety."""

    def test_get_lock_returns_same_rlock_for_equivalent_paths(self, tmp_path):
        cm = CacheManager()
        p1 = tmp_path / "test_file.npz"
        p2 = tmp_path / "TEST_FILE.NPZ"

        lock1 = cm.get_lock(p1)
        lock2 = cm.get_lock(p2)
        # On Windows (normcase), these resolve to the exact same lock
        if os.name == "nt":
            assert lock1 is lock2
        assert isinstance(lock1, type(threading.RLock()))

    def test_concurrent_multithreaded_cache_access(self, tmp_path):
        cm = CacheManager()
        num_threads = 8
        items_per_thread = 20
        errors = []

        def worker(thread_id):
            try:
                for j in range(items_per_thread):
                    target_path = tmp_path / f"thread_{thread_id % 3}_item_{j}.npz"
                    features = {
                        "data": np.full((10, 10), thread_id * 100 + j, dtype=np.float32),
                        "scale": float(j),
                    }
                    cm.save_features_cache(target_path, features)
                    loaded = cm.load_features_cache(target_path)
                    assert loaded is not None
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(worker, i) for i in range(num_threads)]
            for f in futures:
                f.result()

        assert len(errors) == 0
