from collections import OrderedDict
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

_MAX_FEATURES_RAM_ITEMS = 128
_MAX_HNSEP_RAM_ITEMS = 64
_MAX_WAV_RAM_ITEMS = 64


class CacheManager:
    def __init__(self):
        self._locks: Dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()
        self._mem_lock = threading.Lock()
        self._mem_features: OrderedDict[str, Tuple[Optional[Tuple[int, int]], Dict[str, Any]]] = OrderedDict()
        self._mem_hnsep: OrderedDict[str, Tuple[Optional[Tuple[int, int]], torch.Tensor]] = OrderedDict()
        self._mem_wav: OrderedDict[str, Tuple[Tuple[int, int], np.ndarray]] = OrderedDict()

    def _resolve_key(self, path: Path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    def _stat_sig(self, path: Path) -> Optional[Tuple[int, int]]:
        try:
            st = path.stat()
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def get_lock(self, cache_path: Path) -> threading.RLock:
        cache_key = self._resolve_key(cache_path)
        with self._locks_lock:
            if cache_key not in self._locks:
                self._locks[cache_key] = threading.RLock()
            return self._locks[cache_key]

    @staticmethod
    def _freeze_features(features: Dict[str, Any]) -> Dict[str, Any]:
        frozen = {}
        for k, v in features.items():
            if isinstance(v, np.ndarray):
                arr = np.ascontiguousarray(v)
                arr.flags.writeable = False
                frozen[k] = arr
            else:
                frozen[k] = v
        return frozen

    def _put_mem_features(self, key: str, sig: Optional[Tuple[int, int]], features: Dict[str, Any]) -> Dict[str, Any]:
        frozen = self._freeze_features(features)
        with self._mem_lock:
            self._mem_features[key] = (sig, frozen)
            self._mem_features.move_to_end(key)
            while len(self._mem_features) > _MAX_FEATURES_RAM_ITEMS:
                self._mem_features.popitem(last=False)
        return frozen

    def _put_mem_hnsep(self, key: str, sig: Optional[Tuple[int, int]], cpu_tensor: torch.Tensor) -> None:
        with self._mem_lock:
            self._mem_hnsep[key] = (sig, cpu_tensor)
            self._mem_hnsep.move_to_end(key)
            while len(self._mem_hnsep) > _MAX_HNSEP_RAM_ITEMS:
                self._mem_hnsep.popitem(last=False)

    def load_wav_cache(self, resolved_path: Path) -> Optional[np.ndarray]:
        key = str(resolved_path)
        sig = self._stat_sig(resolved_path)
        if sig is None:
            return None
        with self._mem_lock:
            entry = self._mem_wav.get(key)
            if entry is not None:
                cached_sig, arr = entry
                if cached_sig == sig:
                    self._mem_wav.move_to_end(key)
                    return arr.copy()
                else:
                    del self._mem_wav[key]
        return None

    def save_wav_cache(self, resolved_path: Path, wav_array: np.ndarray) -> None:
        key = str(resolved_path)
        sig = self._stat_sig(resolved_path)
        if sig is None:
            return
        stored = np.array(wav_array, dtype=np.float32, copy=True, order="C")
        stored.flags.writeable = False
        with self._mem_lock:
            self._mem_wav[key] = (sig, stored)
            self._mem_wav.move_to_end(key)
            while len(self._mem_wav) > _MAX_WAV_RAM_ITEMS:
                self._mem_wav.popitem(last=False)

    def load_features_cache(
        self, cache_path: Path, force_generate: bool = False, source_path: Optional[Path] = None
    ) -> Any:
        key = self._resolve_key(cache_path)
        if force_generate:
            with self._mem_lock:
                self._mem_features.pop(key, None)
            return None

        with self.get_lock(cache_path):
            disk_sig = self._stat_sig(cache_path)
            if source_path is not None and disk_sig is not None:
                src_sig = self._stat_sig(source_path)
                if src_sig is not None and disk_sig[0] < src_sig[0]:
                    with self._mem_lock:
                        self._mem_features.pop(key, None)
                    return None

            with self._mem_lock:
                entry = self._mem_features.get(key)
                if entry is not None:
                    cached_sig, cached_features = entry
                    if disk_sig is not None and cached_sig == disk_sig:
                        self._mem_features.move_to_end(key)
                        return dict(cached_features)
                    elif disk_sig is None and cached_sig is None:
                        self._mem_features.move_to_end(key)
                        return dict(cached_features)
                    else:
                        del self._mem_features[key]

            if disk_sig is not None:
                try:
                    with np.load(str(cache_path), allow_pickle=False) as data:
                        features = {
                            k: (data[k].item() if data[k].ndim == 0 and k == "scale" else data[k])
                            for k in data.files
                        }
                    frozen = self._put_mem_features(key, disk_sig, features)
                    logging.debug(f'Cache loaded successfully: {cache_path.name}')
                    return dict(frozen)
                except (EOFError, OSError, ValueError, KeyError) as e:
                    logging.warning(f'Failed to load cache {cache_path.name}: {type(e).__name__}: {e}')
        return None

    def load_hnsep_cache(
        self, cache_path: Path, device: str, force_generate: bool = False, source_path: Optional[Path] = None
    ) -> Any:
        key = self._resolve_key(cache_path)
        if force_generate:
            with self._mem_lock:
                self._mem_hnsep.pop(key, None)
            return None

        target_dev = torch.device(device)
        with self.get_lock(cache_path):
            disk_sig = self._stat_sig(cache_path)
            if source_path is not None and disk_sig is not None:
                src_sig = self._stat_sig(source_path)
                if src_sig is not None and disk_sig[0] < src_sig[0]:
                    with self._mem_lock:
                        self._mem_hnsep.pop(key, None)
                    return None

            with self._mem_lock:
                entry = self._mem_hnsep.get(key)
                if entry is not None:
                    cached_sig, cpu_tensor = entry
                    if (disk_sig is not None and cached_sig == disk_sig) or (disk_sig is None and cached_sig is None):
                        self._mem_hnsep.move_to_end(key)
                        return cpu_tensor.clone() if cpu_tensor.device == target_dev else cpu_tensor.to(target_dev)
                    else:
                        del self._mem_hnsep[key]

            if disk_sig is not None:
                try:
                    cpu_data = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                    if isinstance(cpu_data, torch.Tensor):
                        cpu_data = cpu_data.detach().cpu()
                        self._put_mem_hnsep(key, disk_sig, cpu_data)
                        logging.debug(f'Hnsep cache loaded successfully: {cache_path.name}')
                        return cpu_data.clone() if cpu_data.device == target_dev else cpu_data.to(target_dev)
                    return cpu_data
                except Exception as e:
                    logging.warning(f'Failed to load hnsep cache {cache_path.name} (regenerating): {type(e).__name__}: {e}')
        return None

    def save_features_cache(
        self,
        cache_path: Path,
        features: Dict[str, Any],
        force_generate: bool = False,
        source_path: Optional[Path] = None,
    ):
        key = self._resolve_key(cache_path)
        with self.get_lock(cache_path):
            if not force_generate:
                existing = self.load_features_cache(cache_path, force_generate=False, source_path=source_path)
                if existing is not None:
                    logging.debug(f'Cache already exists, using existing: {cache_path.name}')
                    return existing

            temp_base = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            temp_npz_path = Path(str(temp_base) + ".npz")
            try:
                # Use uncompressed np.savez to avoid high-entropy float32 zlib CPU bottlenecks
                np.savez(str(temp_base), **features)
                os.replace(str(temp_npz_path), str(cache_path))
                disk_sig = self._stat_sig(cache_path)
                frozen = self._put_mem_features(key, disk_sig, features)
                logging.debug(f'Features saved successfully to {cache_path}')
                return dict(frozen)
            except (PermissionError, OSError) as e:
                logging.warning(f'Could not write cache to {cache_path} (read-only or permission denied), using in-memory features: {e}')
                if temp_npz_path.exists():
                    try:
                        temp_npz_path.unlink()
                    except OSError:
                        pass
                frozen = self._put_mem_features(key, None, features)
                return dict(frozen)
            except Exception as e:
                logging.error(f'Error saving cache {cache_path}: {e}', exc_info=True)
                if temp_npz_path.exists():
                    try:
                        temp_npz_path.unlink()
                    except OSError:
                        pass
                frozen = self._put_mem_features(key, None, features)
                return dict(frozen)

    def save_hnsep_cache(
        self,
        cache_path: Path,
        data: Any,
        device: str = 'cpu',
        force_generate: bool = False,
        source_path: Optional[Path] = None,
    ):
        key = self._resolve_key(cache_path)
        with self.get_lock(cache_path):
            if not force_generate:
                existing = self.load_hnsep_cache(
                    cache_path, device=device, force_generate=False, source_path=source_path
                )
                if existing is not None:
                    logging.debug(f'Hnsep cache already exists, using existing: {cache_path.name}')
                    return existing

            cpu_data = data.detach().cpu().clone() if isinstance(data, torch.Tensor) else data
            temp_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save(cpu_data, str(temp_path))
                os.replace(str(temp_path), str(cache_path))
                disk_sig = self._stat_sig(cache_path)
                if isinstance(cpu_data, torch.Tensor):
                    self._put_mem_hnsep(key, disk_sig, cpu_data)
                logging.debug(f'Hnsep data saved successfully to {cache_path}')
                return data
            except (PermissionError, OSError) as e:
                logging.warning(f'Could not write hnsep cache to {cache_path} (read-only or permission denied), using in-memory tensor: {e}')
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                if isinstance(cpu_data, torch.Tensor):
                    self._put_mem_hnsep(key, None, cpu_data)
                return data
            except Exception as e:
                logging.error(f'Error saving hnsep cache {cache_path}: {e}', exc_info=True)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                if isinstance(cpu_data, torch.Tensor):
                    self._put_mem_hnsep(key, None, cpu_data)
                return data


# 全局缓存管理器实例
cache_manager = CacheManager()
