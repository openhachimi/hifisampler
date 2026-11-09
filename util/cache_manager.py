import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch


class CacheManager:
    def __init__(self):
        self._locks: Dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()

    def get_lock(self, cache_path: Path) -> threading.RLock:
        try:
            cache_key = str(cache_path.resolve())
        except OSError:
            cache_key = str(cache_path)
        with self._locks_lock:
            if cache_key not in self._locks:
                self._locks[cache_key] = threading.RLock()
            return self._locks[cache_key]

    def load_features_cache(self, cache_path: Path, force_generate: bool = False) -> Any:
        if force_generate:
            return None

        with self.get_lock(cache_path):
            if cache_path.exists():
                try:
                    with np.load(str(cache_path)) as data:
                        features = {k: data[k] for k in data.files}
                    logging.debug(f'Cache loaded successfully: {cache_path.name}')
                    return features
                except (EOFError, OSError, ValueError, KeyError) as e:
                    logging.warning(f'Failed to load cache {cache_path.name}: {type(e).__name__}: {e}')
        return None

    def load_hnsep_cache(self, cache_path: Path, device: str, force_generate: bool = False) -> Any:
        if force_generate:
            return None

        with self.get_lock(cache_path):
            if cache_path.exists():
                try:
                    data = torch.load(str(cache_path), map_location=device, weights_only=True)
                    logging.debug(f'Hnsep cache loaded successfully: {cache_path.name}')
                    return data
                except Exception as e:
                    logging.warning(f'Failed to load hnsep cache {cache_path.name} (regenerating): {type(e).__name__}: {e}')
        return None

    def save_features_cache(self, cache_path: Path, features: Dict[str, Any], force_generate: bool = False):
        with self.get_lock(cache_path):
            if not force_generate and cache_path.exists():
                try:
                    with np.load(str(cache_path)) as data:
                        existing_features = {k: data[k] for k in data.files}
                    logging.debug(f'Cache already exists, using existing: {cache_path.name}')
                    return existing_features
                except Exception:
                    pass

            temp_base = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            temp_npz_path = Path(str(temp_base) + ".npz")
            try:
                np.savez_compressed(str(temp_base), **features)
                os.replace(str(temp_npz_path), str(cache_path))
                logging.debug(f'Features saved successfully to {cache_path}')
                return features
            except (PermissionError, OSError) as e:
                logging.warning(f'Could not write cache to {cache_path} (read-only or permission denied), using in-memory features: {e}')
                if temp_npz_path.exists():
                    try:
                        temp_npz_path.unlink()
                    except OSError:
                        pass
                return features
            except Exception as e:
                logging.error(f'Error saving cache {cache_path}: {e}', exc_info=True)
                if temp_npz_path.exists():
                    try:
                        temp_npz_path.unlink()
                    except OSError:
                        pass
                return features

    def save_hnsep_cache(self, cache_path: Path, data: Any, device: str = 'cpu', force_generate: bool = False):
        with self.get_lock(cache_path):
            if not force_generate and cache_path.exists():
                try:
                    existing_data = torch.load(str(cache_path), map_location=device, weights_only=True)
                    logging.debug(f'Hnsep cache already exists, using existing: {cache_path.name}')
                    return existing_data
                except Exception:
                    pass

            temp_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                torch.save(data, str(temp_path))
                os.replace(str(temp_path), str(cache_path))
                logging.debug(f'Hnsep data saved successfully to {cache_path}')
                return data
            except (PermissionError, OSError) as e:
                logging.warning(f'Could not write hnsep cache to {cache_path} (read-only or permission denied), using in-memory tensor: {e}')
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                return data
            except Exception as e:
                logging.error(f'Error saving hnsep cache {cache_path}: {e}', exc_info=True)
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                return data


# 全局缓存管理器实例
cache_manager = CacheManager()
