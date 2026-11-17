//! HIFIBIN1 binary feature cache and legacy `.npz` fallback reader/writer.
//!
//! Header layout: exactly 28 bytes packed little-endian `<8sIIId`
//! - `magic`: `*b"HIFIBIN1"` (8 bytes)
//! - `version`: `u32` = 1 (4 bytes, little-endian)
//! - `rows`: `u32` (4 bytes, little-endian)
//! - `cols`: `u32` (4 bytes, little-endian)
//! - `scale`: `f64` (8 bytes, little-endian)

use crate::audio::AudioBuffer;
use ndarray::{Array2, ArrayView2};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use uuid::Uuid;

/// Maximum number of feature matrices cached in L1 RAM (`util/cache_manager.py`).
pub const MAX_FEATURES_RAM_ITEMS: usize = 128;

/// Maximum number of decoded WAV waveforms cached in L1 RAM (`util/cache_manager.py`).
pub const MAX_WAV_RAM_ITEMS: usize = 64;

/// Threshold above which `CacheManager::get_lock` lazily prunes idle lock entries.
const LOCK_PRUNE_THRESHOLD: usize = 256;

/// File metadata signature matching Python `(st.st_mtime_ns, st.st_size)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatSig {
    pub mtime_ns: i128,
    pub size: u64,
}

#[inline]
fn stat_sig_from_meta(meta: &fs::Metadata) -> Option<StatSig> {
    let modified = meta.modified().ok()?;
    let mtime_ns = match modified.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i128,
        Err(e) => -(e.duration().as_nanos() as i128),
    };
    Some(StatSig {
        mtime_ns,
        size: meta.len(),
    })
}

/// Reads `(mtime_ns, size)` metadata signature for `path`, returning `None` if unavailable.
pub fn stat_sig(path: &Path) -> Option<StatSig> {
    let meta = fs::metadata(path).ok()?;
    stat_sig_from_meta(&meta)
}

/// Computes the legacy `.npz` fallback path corresponding to a `.bin` cache path.
fn sibling_npz_path(cache_path: &Path) -> PathBuf {
    if let Some(stem) = cache_path.file_name().and_then(|s| s.to_str()) {
        if let Some(stripped) = stem.strip_suffix(".bin") {
            let new_name = format!("{stripped}.npz");
            return match cache_path.parent() {
                Some(p) if !p.as_os_str().is_empty() => p.join(new_name),
                _ => PathBuf::from(new_name),
            };
        }
    }
    cache_path.with_extension("npz")
}

/// Resolves `(read_path, write_path)` for feature cache files, falling back to an existing
/// `.npz` file for reads while always targeting `.bin` for writes when given a `.bin` or bare path.
pub fn resolve_feature_cache_paths(base_path: &Path) -> (PathBuf, PathBuf) {
    let ext_lower = base_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());

    match ext_lower.as_deref() {
        Some("bin") => {
            let bin_path = base_path.to_path_buf();
            let npz_path = base_path.with_extension("npz");
            let read_path = if bin_path.exists() {
                bin_path.clone()
            } else if npz_path.exists() {
                npz_path
            } else {
                bin_path.clone()
            };
            (read_path, bin_path)
        }
        Some("npz") => {
            let npz_path = base_path.to_path_buf();
            let bin_path = base_path.with_extension("bin");
            let read_path = if npz_path.exists() {
                npz_path.clone()
            } else if bin_path.exists() {
                bin_path
            } else {
                npz_path.clone()
            };
            (read_path, npz_path)
        }
        _ => {
            let name = base_path.to_string_lossy();
            let (bin_path, npz_path) = if name.to_ascii_lowercase().ends_with(".hifi") {
                (
                    PathBuf::from(format!("{name}.bin")),
                    PathBuf::from(format!("{name}.npz")),
                )
            } else {
                (
                    PathBuf::from(format!("{name}.hifi.bin")),
                    PathBuf::from(format!("{name}.hifi.npz")),
                )
            };
            let read_path = if bin_path.exists() {
                bin_path.clone()
            } else if npz_path.exists() {
                npz_path
            } else {
                bin_path.clone()
            };
            (read_path, bin_path)
        }
    }
}

/// Atomically renames `tmp` to `dest` with up to 3 exponential retries (`20ms`, `80ms`, `200ms`)
/// when encountering transient Windows sharing/access errors (`ERROR_ACCESS_DENIED=5`,
/// `ERROR_SHARING_VIOLATION=32`, or `ErrorKind::PermissionDenied`).
pub fn rename_with_retry(tmp: &Path, dest: &Path) -> io::Result<()> {
    const BACKOFF_MS: [u64; 3] = [20, 80, 200];
    let mut attempt = 0;
    loop {
        match fs::rename(tmp, dest) {
            Ok(()) => return Ok(()),
            Err(e) if attempt < BACKOFF_MS.len() && is_retryable_rename_err(&e) => {
                std::thread::sleep(Duration::from_millis(BACKOFF_MS[attempt]));
                attempt += 1;
            }
            Err(e) => return Err(e),
        }
    }
}

#[inline]
fn is_retryable_rename_err(e: &io::Error) -> bool {
    if e.kind() == io::ErrorKind::PermissionDenied {
        return true;
    }
    matches!(e.raw_os_error(), Some(5) | Some(32))
}

/// Best-effort cleanup of orphan `.*.tmp` files inside `dir` older than `min_age`.
/// Returns the number of files removed.
pub fn cleanup_stale_tmp_files(dir: &Path, min_age: Duration) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    let now = SystemTime::now();
    let mut removed = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !(name.starts_with('.') && name.ends_with(".tmp")) {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        if !min_age.is_zero() {
            let Ok(modified) = meta.modified() else {
                continue;
            };
            let Ok(age) = now.duration_since(modified) else {
                continue;
            };
            if age < min_age {
                continue;
            }
        }
        if fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Best-effort cleanup of all orphan `.*.tmp` files inside `dir`.
/// Returns the number of files removed.
pub fn cleanup_orphan_tmp_files(dir: &Path) -> usize {
    cleanup_stale_tmp_files(dir, Duration::ZERO)
}

/// 8-byte ASCII magic identifier for HIFIBIN1 feature cache files.
pub const HIFIBIN_MAGIC: &[u8; 8] = b"HIFIBIN1";

/// Format version number (currently 1).
pub const HIFIBIN_VERSION: u32 = 1;

/// Exact packed header size in bytes without padding (8 + 4 + 4 + 4 + 8 = 28).
pub const HEADER_SIZE: usize = 28;

/// Binary header representation for `HIFIBIN1` cache files.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HifiBinHeader {
    /// File magic bytes, must equal `*b"HIFIBIN1"`.
    pub magic: [u8; 8],
    /// Cache format version, must equal `1`.
    pub version: u32,
    /// Number of mel frequency bins (rows, typically 128).
    pub rows: u32,
    /// Number of time frames (columns, N).
    pub cols: u32,
    /// Dynamic range scaling factor.
    pub scale: f64,
}

impl HifiBinHeader {
    /// Creates a new header with standard magic and version.
    pub fn new(rows: u32, cols: u32, scale: f64) -> Self {
        Self {
            magic: *HIFIBIN_MAGIC,
            version: HIFIBIN_VERSION,
            rows,
            cols,
            scale,
        }
    }

    /// Serializes the header into an exact 28-byte array in little-endian format.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.rows.to_le_bytes());
        buf[16..20].copy_from_slice(&self.cols.to_le_bytes());
        buf[20..28].copy_from_slice(&self.scale.to_le_bytes());
        buf
    }

    /// Deserializes and validates a header from a 28-byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CacheError> {
        if bytes.len() < HEADER_SIZE {
            return Err(CacheError::TruncatedHeader(bytes.len()));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if magic != *HIFIBIN_MAGIC {
            return Err(CacheError::InvalidMagic(magic));
        }

        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != HIFIBIN_VERSION {
            return Err(CacheError::UnsupportedVersion(version));
        }

        let rows = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let cols = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        let scale = f64::from_le_bytes(bytes[20..28].try_into().unwrap());

        if !scale.is_finite() {
            return Err(CacheError::NonFiniteScale(scale));
        }

        Ok(Self {
            magic,
            version,
            rows,
            cols,
            scale,
        })
    }

    /// Reads a 28-byte header from a reader.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, CacheError> {
        let mut buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut buf)?;
        Self::from_bytes(&buf)
    }

    /// Writes the 28-byte header to a writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), CacheError> {
        let bytes = self.to_bytes();
        writer.write_all(&bytes)?;
        Ok(())
    }
}

/// In-memory representation of cached acoustic features.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedFeatures {
    /// 2D mel-spectrogram origin matrix of shape `[rows, cols]`.
    pub mel_origin: Array2<f32>,
    /// Dynamic range scale factor.
    pub scale: f64,
}

/// Errors occurring during cache serialization or deserialization.
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Truncated binary cache header: {0} bytes (expected 28)")]
    TruncatedHeader(usize),

    #[error("Invalid binary cache magic: {0:?}")]
    InvalidMagic([u8; 8]),

    #[error("Unsupported binary cache version: {0}")]
    UnsupportedVersion(u32),

    #[error("Non-finite scale in binary cache: {0}")]
    NonFiniteScale(f64),

    #[error("Non-finite payload element detected in binary cache")]
    NonFinitePayload,

    #[error("Zero-filled payload detected in feature cache (requires regeneration)")]
    ZeroFilledPayload,

    #[error(
        "Binary cache size mismatch: expected {expected} bytes for ({rows}, {cols}), got {actual}"
    )]
    SizeMismatch {
        expected: u64,
        actual: u64,
        rows: u32,
        cols: u32,
    },

    #[error("NPZ reader error: {0}")]
    Npz(String),

    #[error("Array shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("Feature matrix dimensions ({rows}x{cols}) exceed u32 limits")]
    DimensionOverflow { rows: usize, cols: usize },

    #[error("Atomic rename failed for '{tmp}' -> '{dest}': {source}")]
    AtomicRenameFailed {
        tmp: PathBuf,
        dest: PathBuf,
        #[source]
        source: io::Error,
    },
}

fn is_stale_from_sig(
    cache_sig: StatSig,
    source_path: &Path,
    known_src_sig: Option<Option<StatSig>>,
) -> Result<bool, CacheError> {
    let src_sig = match known_src_sig {
        Some(Some(sig)) => sig,
        Some(None) => return Ok(false),
        None => match fs::metadata(source_path) {
            Ok(m) => match stat_sig_from_meta(&m) {
                Some(sig) => sig,
                None => return Ok(false),
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(CacheError::Io(e)),
        },
    };
    Ok(cache_sig.mtime_ns < src_sig.mtime_ns)
}

/// Atomically saves a contiguous `[rows, cols]` `f32` payload to disk using `HIFIBIN1`,
/// calling `sync_all()` only when `durable == true`.
pub fn save_cache_with_options(
    cache_path: &Path,
    rows: usize,
    cols: usize,
    flat_data: &[f32],
    scale: f64,
    durable: bool,
) -> Result<(), CacheError> {
    if !scale.is_finite() {
        return Err(CacheError::NonFiniteScale(scale));
    }

    // Validate dimensions fit u32 and match flat_data.len() without overflow.
    let rows_u32 = u32::try_from(rows).map_err(|_| CacheError::DimensionOverflow { rows, cols })?;
    let cols_u32 = u32::try_from(cols).map_err(|_| CacheError::DimensionOverflow { rows, cols })?;
    let _ = ArrayView2::from_shape((rows, cols), flat_data)?;

    if let Some(parent) = cache_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file_name = cache_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("cache.bin");
    let tmp_file_name = format!(".{file_name}.{}.tmp", Uuid::new_v4().simple());
    let tmp_path = match cache_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(&tmp_file_name),
        _ => PathBuf::from(&tmp_file_name),
    };

    // RAII temporary file cleanup on failure
    struct TmpCleanup<'a>(&'a Path);
    impl<'a> Drop for TmpCleanup<'a> {
        fn drop(&mut self) {
            if self.0.exists() {
                let _ = fs::remove_file(self.0);
            }
        }
    }
    let cleanup = TmpCleanup(&tmp_path);

    let header = HifiBinHeader::new(rows_u32, cols_u32, scale);

    {
        let file = File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        header.write_to(&mut writer)?;

        #[cfg(target_endian = "little")]
        {
            let byte_slice = bytemuck::cast_slice::<f32, u8>(flat_data);
            writer.write_all(byte_slice)?;
        }
        #[cfg(not(target_endian = "little"))]
        {
            for &val in flat_data {
                writer.write_all(&val.to_le_bytes())?;
            }
        }

        writer.flush()?;
        if durable {
            writer
                .into_inner()
                .map_err(|e| e.into_error())?
                .sync_all()?;
        }
    }

    rename_with_retry(&tmp_path, cache_path).map_err(|e| CacheError::AtomicRenameFailed {
        tmp: tmp_path.clone(),
        dest: cache_path.to_path_buf(),
        source: e,
    })?;

    // Disarm cleanup guard on successful rename
    std::mem::forget(cleanup);

    Ok(())
}

/// Atomically saves acoustic features to disk using the `HIFIBIN1` binary format (`durable = true`).
pub fn save_cache(cache_path: &Path, mel: &Array2<f32>, scale: f64) -> Result<(), CacheError> {
    let rows = mel.nrows();
    let cols = mel.ncols();
    let standard = mel.as_standard_layout();
    if let Some(slice) = standard.as_slice() {
        save_cache_with_options(cache_path, rows, cols, slice, scale, true)
    } else {
        let owned = standard.to_owned();
        save_cache_with_options(
            cache_path,
            rows,
            cols,
            owned.as_slice().unwrap_or(&[]),
            scale,
            true,
        )
    }
}

fn validate_payload(floats: &[f32], reject_all_zero: bool) -> Result<(), CacheError> {
    let mut all_zero = reject_all_zero && !floats.is_empty();
    for &v in floats {
        if !v.is_finite() {
            return Err(CacheError::NonFinitePayload);
        }
        if all_zero && v != 0.0 {
            all_zero = false;
        }
    }
    if all_zero {
        return Err(CacheError::ZeroFilledPayload);
    }
    Ok(())
}

fn load_cache_with_options_inner(
    cache_path: &Path,
    source_wav_path: Option<&Path>,
    known_src_sig: Option<Option<StatSig>>,
    reject_all_zero: bool,
) -> Result<Option<CachedFeatures>, CacheError> {
    let file = match File::open(cache_path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let candidate_npz = sibling_npz_path(cache_path);
            let npz_file = match File::open(&candidate_npz) {
                Ok(f) => f,
                Err(e2) if e2.kind() == io::ErrorKind::NotFound => return Ok(None),
                Err(e2) => return Err(CacheError::Io(e2)),
            };
            if let Some(source_path) = source_wav_path {
                let meta = npz_file.metadata()?;
                if let Some(cache_sig) = stat_sig_from_meta(&meta) {
                    if is_stale_from_sig(cache_sig, source_path, known_src_sig)? {
                        return Ok(None);
                    }
                }
            }
            return load_npz_from_reader(BufReader::new(npz_file), reject_all_zero);
        }
        Err(e) => return Err(CacheError::Io(e)),
    };

    let meta = file.metadata()?;
    let file_len = meta.len();

    if let Some(source_path) = source_wav_path {
        if let Some(cache_sig) = stat_sig_from_meta(&meta) {
            if is_stale_from_sig(cache_sig, source_path, known_src_sig)? {
                return Ok(None);
            }
        }
    }

    let mut reader = BufReader::new(file);
    let mut magic_buf = [0u8; 8];
    if let Err(e) = reader.read_exact(&mut magic_buf) {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            return Err(CacheError::TruncatedHeader(file_len as usize));
        }
        return Err(CacheError::Io(e));
    }

    if magic_buf == *HIFIBIN_MAGIC {
        let mut rest_header = [0u8; HEADER_SIZE - 8];
        if let Err(e) = reader.read_exact(&mut rest_header) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                return Err(CacheError::TruncatedHeader(file_len as usize));
            }
            return Err(CacheError::Io(e));
        }

        let mut header_buf = [0u8; HEADER_SIZE];
        header_buf[0..8].copy_from_slice(&magic_buf);
        header_buf[8..HEADER_SIZE].copy_from_slice(&rest_header);

        let header = HifiBinHeader::from_bytes(&header_buf)?;

        let num_elements_u64 = (header.rows as u64).checked_mul(header.cols as u64).ok_or(
            CacheError::SizeMismatch {
                expected: u64::MAX,
                actual: file_len,
                rows: header.rows,
                cols: header.cols,
            },
        )?;
        let expected_size = num_elements_u64
            .checked_mul(4)
            .and_then(|b| b.checked_add(HEADER_SIZE as u64))
            .ok_or(CacheError::SizeMismatch {
                expected: u64::MAX,
                actual: file_len,
                rows: header.rows,
                cols: header.cols,
            })?;
        if file_len != expected_size {
            return Err(CacheError::SizeMismatch {
                expected: expected_size,
                actual: file_len,
                rows: header.rows,
                cols: header.cols,
            });
        }

        let num_elements =
            usize::try_from(num_elements_u64).map_err(|_| CacheError::SizeMismatch {
                expected: expected_size,
                actual: file_len,
                rows: header.rows,
                cols: header.cols,
            })?;

        let mut floats = vec![0.0_f32; num_elements];
        if num_elements > 0 {
            #[cfg(target_endian = "little")]
            {
                reader.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut floats))?;
            }
            #[cfg(not(target_endian = "little"))]
            {
                let mut buf4 = [0u8; 4];
                for dst in &mut floats {
                    reader.read_exact(&mut buf4)?;
                    *dst = f32::from_le_bytes(buf4);
                }
            }
        }

        validate_payload(&floats, reject_all_zero)?;

        let mel_origin =
            Array2::from_shape_vec((header.rows as usize, header.cols as usize), floats)?;

        Ok(Some(CachedFeatures {
            mel_origin,
            scale: header.scale,
        }))
    } else {
        // Not a HIFIBIN1 magic; attempt legacy NPZ decode
        reader.seek(SeekFrom::Start(0))?;
        load_npz_from_reader(reader, reject_all_zero)
    }
}

/// Reads cached acoustic features from disk with configurable all-zero payload rejection.
///
/// Validates `source_wav_path` mtime staleness, rejects non-finite payload floats,
/// and when `reject_all_zero == true`, rejects non-empty all-zero payloads so power-loss
/// zero-filled feature caches regenerate automatically.
pub fn load_cache_with_options(
    cache_path: &Path,
    source_wav_path: Option<&Path>,
    reject_all_zero: bool,
) -> Result<Option<CachedFeatures>, CacheError> {
    load_cache_with_options_inner(cache_path, source_wav_path, None, reject_all_zero)
}

/// Reads cached acoustic features from disk (`reject_all_zero = true`).
///
/// Features are validated against `source_wav_path` (if provided): if `cache.mtime < source.mtime`,
/// the cache is deemed stale and `Ok(None)` is returned.
/// If `cache_path` is missing, transparently checks for a sibling `.npz` archive.
pub fn load_cache(
    cache_path: &Path,
    source_wav_path: Option<&Path>,
) -> Result<Option<CachedFeatures>, CacheError> {
    load_cache_with_options_inner(cache_path, source_wav_path, None, true)
}

/// Decodes features from an `.npz` stream.
fn load_npz_from_reader<R: Read + Seek>(
    reader: R,
    reject_all_zero: bool,
) -> Result<Option<CachedFeatures>, CacheError> {
    let mut npz =
        ndarray_npy::NpzReader::new(reader).map_err(|e| CacheError::Npz(e.to_string()))?;

    let mel_origin: Array2<f32> = npz
        .by_name("mel_origin.npy")
        .or_else(|_| npz.by_name("mel_origin"))
        .map_err(|e| CacheError::Npz(e.to_string()))?;

    let scale = if let Ok(s) = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix0>("scale.npy")
        .or_else(|_| npz.by_name("scale"))
    {
        s.into_scalar()
    } else if let Ok(s) = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix0>("scale.npy")
        .or_else(|_| npz.by_name("scale"))
    {
        s.into_scalar() as f64
    } else if let Ok(s) = npz
        .by_name::<ndarray::OwnedRepr<f64>, ndarray::Ix1>("scale.npy")
        .or_else(|_| npz.by_name("scale"))
    {
        s.iter().next().copied().unwrap_or(1.0)
    } else if let Ok(s) = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix1>("scale.npy")
        .or_else(|_| npz.by_name("scale"))
    {
        s.iter().next().map(|&v| v as f64).unwrap_or(1.0)
    } else {
        1.0
    };

    if !scale.is_finite() {
        return Err(CacheError::NonFiniteScale(scale));
    }

    if let Some(slice) = mel_origin.as_slice() {
        validate_payload(slice, reject_all_zero)?;
    } else {
        let mut all_zero = reject_all_zero && !mel_origin.is_empty();
        for &v in mel_origin.iter() {
            if !v.is_finite() {
                return Err(CacheError::NonFinitePayload);
            }
            if all_zero && v != 0.0 {
                all_zero = false;
            }
        }
        if all_zero {
            return Err(CacheError::ZeroFilledPayload);
        }
    }

    Ok(Some(CachedFeatures { mel_origin, scale }))
}

struct FeatureRamEntry {
    disk_sig: Option<StatSig>,
    src_sig: Option<StatSig>,
    features: Arc<CachedFeatures>,
}

struct LruCache<V> {
    cap: usize,
    order: VecDeque<String>,
    map: HashMap<String, V>,
}

impl<V> LruCache<V> {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            order: VecDeque::new(),
            map: HashMap::new(),
        }
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.to_string());
    }

    fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        if self.map.contains_key(key) {
            self.touch(key);
            self.map.get_mut(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, val: V) {
        if self.map.contains_key(&key) {
            self.touch(&key);
            self.map.insert(key, val);
            return;
        }
        while self.map.len() >= self.cap {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, val);
    }

    fn remove(&mut self, key: &str) {
        if self.map.remove(key).is_some() {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
            }
        }
    }
}

/// Thread-safe L1 RAM LRU + L2 disk `HIFIBIN1` cache manager mirroring `util/cache_manager.py`.
///
/// Validates both cache file and source WAV `(mtime_ns, size)` signatures (`StatSig`),
/// supports zero-copy `Arc` handles for RAM hits, transparent `.hifi.npz` fallback reads,
/// and lazily-pruned per-path locks.
pub struct CacheManager {
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    mem_features: Mutex<LruCache<FeatureRamEntry>>,
    mem_wav: Mutex<LruCache<(StatSig, Arc<AudioBuffer>)>>,
}

impl Default for CacheManager {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheManager {
    /// Creates a new `CacheManager` with standard LRU capacities (128 features, 64 WAVs).
    pub fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
            mem_features: Mutex::new(LruCache::new(MAX_FEATURES_RAM_ITEMS)),
            mem_wav: Mutex::new(LruCache::new(MAX_WAV_RAM_ITEMS)),
        }
    }

    fn resolve_key(path: &Path) -> String {
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        };
        let s = abs.to_string_lossy().to_string();
        if cfg!(windows) {
            s.replace('/', "\\").to_ascii_lowercase()
        } else {
            s
        }
    }

    /// Returns a per-cache-path lock for coordinating concurrent feature generation,
    /// lazily pruning idle lock entries when the map exceeds 256 entries.
    pub fn get_lock(&self, cache_path: &Path) -> Arc<Mutex<()>> {
        let key = Self::resolve_key(cache_path);
        let mut guard = self.locks.lock().unwrap_or_else(|p| p.into_inner());
        if guard.len() > LOCK_PRUNE_THRESHOLD {
            guard.retain(|_, v| Arc::strong_count(v) > 1);
        }
        guard
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Returns the number of currently tracked per-path locks (for diagnostics/tests).
    pub fn tracked_lock_count(&self) -> usize {
        self.locks.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Loads decoded WAV audio from L1 RAM cache as a zero-copy `Arc` if `(mtime_ns, size)` matches `path`.
    pub fn load_wav_cache_arc(&self, path: &Path) -> Option<Arc<AudioBuffer>> {
        let sig = stat_sig(path)?;
        let key = Self::resolve_key(path);
        let mut guard = self.mem_wav.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((cached_sig, buf)) = guard.get_mut(&key) {
            if *cached_sig == sig {
                return Some(Arc::clone(buf));
            }
            guard.remove(&key);
        }
        None
    }

    /// Loads decoded WAV audio from L1 RAM cache if `(mtime_ns, size)` matches `path`.
    pub fn load_wav_cache(&self, path: &Path) -> Option<AudioBuffer> {
        self.load_wav_cache_arc(path)
            .map(|arc| arc.as_ref().clone())
    }

    /// Stores decoded WAV audio `Arc` in L1 RAM cache keyed by `(mtime_ns, size)`.
    pub fn save_wav_cache_arc(&self, path: &Path, audio: Arc<AudioBuffer>) {
        let Some(sig) = stat_sig(path) else {
            return;
        };
        let key = Self::resolve_key(path);
        let mut guard = self.mem_wav.lock().unwrap_or_else(|p| p.into_inner());
        guard.insert(key, (sig, audio));
    }

    /// Stores decoded WAV audio in L1 RAM cache keyed by `(mtime_ns, size)`.
    pub fn save_wav_cache(&self, path: &Path, audio: &AudioBuffer) {
        self.save_wav_cache_arc(path, Arc::new(audio.clone()));
    }

    /// Loads acoustic features from L1 RAM or L2 disk (`HIFIBIN1` or legacy `.npz` fallback)
    /// as a zero-copy `Arc<CachedFeatures>`, invalidating when `source_path` has a newer `mtime_ns`
    /// or altered `(mtime_ns, size)`. Corrupted or zero-filled disk files return `None`.
    pub fn load_features_cache_arc(
        &self,
        cache_path: &Path,
        force_generate: bool,
        source_path: Option<&Path>,
    ) -> Option<Arc<CachedFeatures>> {
        let key = Self::resolve_key(cache_path);
        if force_generate {
            let mut guard = self.mem_features.lock().unwrap_or_else(|p| p.into_inner());
            guard.remove(&key);
            return None;
        }

        // Single metadata lookup for cache_path; only probe sibling .npz if .bin is missing.
        let disk_sig = stat_sig(cache_path).or_else(|| stat_sig(&sibling_npz_path(cache_path)));
        let src_sig = source_path.and_then(stat_sig);

        if let (Some(d), Some(s)) = (disk_sig, src_sig) {
            if d.mtime_ns < s.mtime_ns {
                let mut guard = self.mem_features.lock().unwrap_or_else(|p| p.into_inner());
                guard.remove(&key);
                return None;
            }
        }

        {
            let mut guard = self.mem_features.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(entry) = guard.get_mut(&key) {
                let src_stale = matches!(
                    (src_sig, entry.src_sig),
                    (Some(cur), Some(prev)) if cur != prev
                );
                let disk_matches = entry.disk_sig == disk_sig;
                if !src_stale && disk_matches {
                    return Some(Arc::clone(&entry.features));
                }
                guard.remove(&key);
            }
        }

        // If neither .bin nor .npz exists on disk, avoid a redundant File::open ENOENT syscall.
        disk_sig?;

        match load_cache_with_options_inner(cache_path, source_path, Some(src_sig), true) {
            Ok(Some(features)) => {
                let arc = Arc::new(features);
                let mut guard = self.mem_features.lock().unwrap_or_else(|p| p.into_inner());
                guard.insert(
                    key,
                    FeatureRamEntry {
                        disk_sig,
                        src_sig,
                        features: Arc::clone(&arc),
                    },
                );
                Some(arc)
            }
            Ok(None) | Err(_) => None,
        }
    }

    /// Loads acoustic features from L1 RAM or L2 disk (`HIFIBIN1` or legacy `.npz` fallback).
    pub fn load_features_cache(
        &self,
        cache_path: &Path,
        force_generate: bool,
        source_path: Option<&Path>,
    ) -> Option<CachedFeatures> {
        self.load_features_cache_arc(cache_path, force_generate, source_path)
            .map(|arc| arc.as_ref().clone())
    }

    /// Saves acoustic features atomically to `.hifi.bin` and updates L1 RAM cache, returning an `Arc`.
    /// If writing to disk fails (e.g. read-only directory), still caches in L1 RAM.
    pub fn save_features_cache_arc(
        &self,
        cache_path: &Path,
        mel: &Array2<f32>,
        scale: f64,
        force_generate: bool,
        source_path: Option<&Path>,
    ) -> Result<Arc<CachedFeatures>, CacheError> {
        if !scale.is_finite() {
            return Err(CacheError::NonFiniteScale(scale));
        }

        if !force_generate {
            if let Some(existing) = self.load_features_cache_arc(cache_path, false, source_path) {
                return Ok(existing);
            }
        }

        let features = Arc::new(CachedFeatures {
            mel_origin: mel.clone(),
            scale,
        });
        let key = Self::resolve_key(cache_path);
        let src_sig = source_path.and_then(stat_sig);

        let disk_sig = match save_cache(cache_path, mel, scale) {
            Ok(()) => stat_sig(cache_path),
            Err(_) => None,
        };

        let mut guard = self.mem_features.lock().unwrap_or_else(|p| p.into_inner());
        guard.insert(
            key,
            FeatureRamEntry {
                disk_sig,
                src_sig,
                features: Arc::clone(&features),
            },
        );
        Ok(features)
    }

    /// Saves acoustic features atomically to `.hifi.bin` and updates L1 RAM cache.
    pub fn save_features_cache(
        &self,
        cache_path: &Path,
        mel: &Array2<f32>,
        scale: f64,
        force_generate: bool,
        source_path: Option<&Path>,
    ) -> Result<CachedFeatures, CacheError> {
        self.save_features_cache_arc(cache_path, mel, scale, force_generate, source_path)
            .map(|arc| arc.as_ref().clone())
    }
}
