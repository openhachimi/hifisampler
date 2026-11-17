use hifi_core::cache::{load_cache_with_options, save_cache_with_options, stat_sig, StatSig};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_HNSEP_RAM_ITEMS: usize = 64;

#[derive(Clone)]
struct HnsepEntry {
    disk_sig: Option<StatSig>,
    src_sig: Option<StatSig>,
    harmonic: Arc<Vec<f32>>,
}

/// Bounded 64-entry RAM LRU + optional `.hifi.bin` disk cache for separated HN-SEP harmonic audio.
pub struct HnsepCache {
    entries: Mutex<(VecDeque<String>, HashMap<String, HnsepEntry>)>,
}

impl Default for HnsepCache {
    fn default() -> Self {
        Self {
            entries: Mutex::new((
                VecDeque::with_capacity(MAX_HNSEP_RAM_ITEMS),
                HashMap::with_capacity(MAX_HNSEP_RAM_ITEMS),
            )),
        }
    }
}

/// Returns sibling `<stem>_hnsep.hifi.bin` path (distinct from Python's `.pt` `_hnsep` file).
pub fn hnsep_disk_cache_path(in_file: &Path) -> PathBuf {
    let stem = in_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    in_file.with_file_name(format!("{stem}_hnsep.hifi.bin"))
}

fn cache_key(path: &Path) -> String {
    let s = path.to_string_lossy().into_owned();
    if cfg!(windows) {
        s.replace('/', "\\").to_ascii_lowercase()
    } else {
        s
    }
}

impl HnsepCache {
    /// Load cached harmonic waveform from RAM LRU or disk `.hifi.bin` if still fresh vs `source_wav`.
    pub fn load(
        &self,
        disk_path: &Path,
        source_wav: &Path,
        force_generate: bool,
    ) -> Option<Arc<Vec<f32>>> {
        let key = cache_key(disk_path);
        if force_generate {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let (order, map) = &mut *guard;
            Self::remove_locked(order, map, &key);
            return None;
        }

        let disk_sig = stat_sig(disk_path);
        let src_sig = stat_sig(source_wav);
        if let (Some(d), Some(s)) = (disk_sig, src_sig) {
            if d.mtime_ns < s.mtime_ns {
                let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
                let (order, map) = &mut *guard;
                Self::remove_locked(order, map, &key);
                return None;
            }
        }

        {
            let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            let (order, map) = &mut *guard;
            if let Some(entry) = map.get(&key).cloned() {
                let src_stale =
                    matches!((src_sig, entry.src_sig), (Some(cur), Some(prev)) if cur != prev);
                if !src_stale && entry.disk_sig == disk_sig {
                    Self::touch_locked(order, &key);
                    return Some(entry.harmonic);
                }
                Self::remove_locked(order, map, &key);
            }
        }

        if let Ok(Some(cached)) = load_cache_with_options(disk_path, Some(source_wav), false) {
            let harmonic = Arc::new(cached.mel_origin.into_raw_vec_and_offset().0);
            self.put_mem(&key, disk_sig, src_sig, harmonic.clone());
            return Some(harmonic);
        }

        None
    }

    /// Save separated harmonic waveform to disk `.hifi.bin` (best-effort) and RAM LRU.
    pub fn save(&self, disk_path: &Path, source_wav: &Path, harmonic: Vec<f32>) -> Arc<Vec<f32>> {
        let arc = Arc::new(harmonic);
        if let Err(err) =
            save_cache_with_options(disk_path, 1, arc.len(), arc.as_slice(), 1.0, false)
        {
            tracing::debug!(
                "Could not write HN-SEP disk cache '{}': {err}",
                disk_path.display()
            );
        }

        let key = cache_key(disk_path);
        let disk_sig = stat_sig(disk_path);
        let src_sig = stat_sig(source_wav);
        self.put_mem(&key, disk_sig, src_sig, arc.clone());
        arc
    }

    fn put_mem(
        &self,
        key: &str,
        disk_sig: Option<StatSig>,
        src_sig: Option<StatSig>,
        harmonic: Arc<Vec<f32>>,
    ) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let (order, map) = &mut *guard;
        if map.contains_key(key) {
            Self::touch_locked(order, key);
            map.insert(
                key.to_string(),
                HnsepEntry {
                    disk_sig,
                    src_sig,
                    harmonic,
                },
            );
            return;
        }
        while map.len() >= MAX_HNSEP_RAM_ITEMS {
            if let Some(evicted) = order.pop_front() {
                map.remove(&evicted);
            } else {
                break;
            }
        }
        order.push_back(key.to_string());
        map.insert(
            key.to_string(),
            HnsepEntry {
                disk_sig,
                src_sig,
                harmonic,
            },
        );
    }

    fn touch_locked(order: &mut VecDeque<String>, key: &str) {
        if let Some(pos) = order.iter().position(|k| k == key) {
            order.remove(pos);
        }
        order.push_back(key.to_string());
    }

    fn remove_locked(
        order: &mut VecDeque<String>,
        map: &mut HashMap<String, HnsepEntry>,
        key: &str,
    ) {
        if map.remove(key).is_some() {
            if let Some(pos) = order.iter().position(|k| k == key) {
                order.remove(pos);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn hnsep_cache_roundtrip_and_invalidation() {
        let dir =
            std::env::temp_dir().join(format!("hifi_hnsep_cache_{}", Uuid::new_v4().simple()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("voice.wav");
        std::fs::write(&src, b"RIFF").unwrap();
        let disk = hnsep_disk_cache_path(&src);

        let cache = HnsepCache::default();
        assert!(cache.load(&disk, &src, false).is_none());

        cache.save(&disk, &src, vec![0.25, -0.5, 0.75]);
        let hit = cache.load(&disk, &src, false).unwrap();
        assert_eq!(&*hit, &[0.25_f32, -0.5, 0.75]);

        // Cold cache reads back from disk `.hifi.bin` (and accepts valid silent all-zero harmonic).
        let cold_cache = HnsepCache::default();
        let disk_hit = cold_cache.load(&disk, &src, false).unwrap();
        assert_eq!(&*disk_hit, &[0.25_f32, -0.5, 0.75]);

        let silent_src = dir.join("silent.wav");
        std::fs::write(&silent_src, b"RIFF").unwrap();
        let silent_disk = hnsep_disk_cache_path(&silent_src);
        cold_cache.save(&silent_disk, &silent_src, vec![0.0, 0.0, 0.0]);
        let cold_silent = HnsepCache::default();
        assert_eq!(
            &*cold_silent.load(&silent_disk, &silent_src, false).unwrap(),
            &[0.0_f32, 0.0, 0.0]
        );

        assert!(cache.load(&disk, &src, true).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
