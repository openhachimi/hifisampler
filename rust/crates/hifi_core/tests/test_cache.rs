use approx::assert_relative_eq;
use hifi_core::{
    load_cache, resolve_feature_cache_paths, save_cache, stat_sig, AudioBuffer, CacheError,
    CacheManager, HifiBinHeader, DEFAULT_SAMPLE_RATE, HEADER_SIZE, HIFIBIN_MAGIC, HIFIBIN_VERSION,
};
use ndarray::{Array0, Array2};
use std::fs::{self, File};
use std::io::Write;
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn test_header_constant_size_is_28_bytes() {
    assert_eq!(HEADER_SIZE, 28);
    let header = HifiBinHeader::new(128, 50, 0.75);
    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), 28);
    assert_eq!(&bytes[0..8], HIFIBIN_MAGIC);

    let parsed = HifiBinHeader::from_bytes(&bytes).expect("Failed to parse valid header");
    assert_eq!(parsed.magic, *HIFIBIN_MAGIC);
    assert_eq!(parsed.version, HIFIBIN_VERSION);
    assert_eq!(parsed.rows, 128);
    assert_eq!(parsed.cols, 50);
    assert_relative_eq!(parsed.scale, 0.75, epsilon = 1e-7);
}

#[test]
fn test_header_endianness_matches_python_struct() {
    // Equivalent to Python struct.pack("<8sIIId", b"HIFIBIN1", 1, 128, 50, 0.75)
    let header = HifiBinHeader::new(128, 50, 0.75);
    let bytes = header.to_bytes();

    // Magic: b"HIFIBIN1"
    assert_eq!(&bytes[0..8], b"HIFIBIN1");
    // Version: 1 as u32 LE -> [0x01, 0x00, 0x00, 0x00]
    assert_eq!(&bytes[8..12], &[1, 0, 0, 0]);
    // Rows: 128 as u32 LE -> [0x80, 0x00, 0x00, 0x00]
    assert_eq!(&bytes[12..16], &[128, 0, 0, 0]);
    // Cols: 50 as u32 LE -> [0x32, 0x00, 0x00, 0x00]
    assert_eq!(&bytes[16..20], &[50, 0, 0, 0]);
    // Scale: 0.75 as f64 LE -> IEEE 754 double
    let expected_scale_bytes = 0.75f64.to_le_bytes();
    assert_eq!(&bytes[20..28], &expected_scale_bytes);
}

#[test]
fn test_cache_save_and_load_roundtrip() {
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("test_sample.hifi.bin");

    let rows = 128;
    let cols = 64;
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push((r as f32 * 0.1) + (c as f32 * 0.01));
        }
    }
    let mel = Array2::from_shape_vec((rows, cols), data).unwrap();
    let scale = 0.8125;

    save_cache(&cache_path, &mel, scale).expect("Save cache failed");
    assert!(cache_path.exists());

    // Exact expected size: 28 + (128 * 64 * 4) = 28 + 32768 = 32796 bytes
    let file_size = fs::metadata(&cache_path).unwrap().len();
    assert_eq!(file_size, 28 + (128 * 64 * 4));

    let loaded = load_cache(&cache_path, None)
        .expect("Load cache failed")
        .expect("Cache returned None");

    assert_eq!(loaded.mel_origin.shape(), &[128, 64]);
    assert_relative_eq!(loaded.scale, scale, epsilon = 1e-7);

    for (a, b) in mel.iter().zip(loaded.mel_origin.iter()) {
        assert_relative_eq!(*a, *b, epsilon = 1e-6);
    }
}

#[test]
fn test_empty_audio_cache_cols_zero() {
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("empty.hifi.bin");

    let rows = 128;
    let cols = 0;
    let mel = Array2::from_shape_vec((rows, cols), vec![]).unwrap();
    let scale = 1.0;

    save_cache(&cache_path, &mel, scale).expect("Save empty cache failed");
    assert_eq!(fs::metadata(&cache_path).unwrap().len(), 28);

    let loaded = load_cache(&cache_path, None)
        .expect("Load empty cache failed")
        .expect("Cache returned None");

    assert_eq!(loaded.mel_origin.shape(), &[128, 0]);
    assert_eq!(loaded.scale, 1.0);
}

#[test]
fn test_mtime_invalidation_when_wav_is_newer() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("sample.wav");
    let cache_path = dir.path().join("sample.hifi.bin");

    // 1. Create cache first
    let mel = Array2::from_shape_vec((128, 10), vec![0.5f32; 1280]).unwrap();
    save_cache(&cache_path, &mel, 1.0).unwrap();

    // 2. Sleep slightly and create source WAV newer than cache
    std::thread::sleep(Duration::from_millis(50));
    {
        let mut f = File::create(&wav_path).unwrap();
        f.write_all(b"fake wav data").unwrap();
    }

    // Now wav_mtime > cache_mtime -> cache should be invalidated
    let result = load_cache(&cache_path, Some(&wav_path)).unwrap();
    assert!(
        result.is_none(),
        "Expected cache miss due to newer source WAV"
    );

    // Conversely, if cache is newer than or equal to WAV, it should load
    std::thread::sleep(Duration::from_millis(50));
    save_cache(&cache_path, &mel, 1.0).unwrap();
    let valid_result = load_cache(&cache_path, Some(&wav_path)).unwrap();
    assert!(valid_result.is_some(), "Expected valid cache hit");
}

#[test]
fn test_corrupted_header_detection() {
    let dir = tempdir().unwrap();

    // 1. Bad magic
    let bad_magic_path = dir.path().join("bad_magic.bin");
    {
        let mut f = File::create(&bad_magic_path).unwrap();
        let mut header = HifiBinHeader::new(128, 10, 1.0).to_bytes();
        header[0..8].copy_from_slice(b"NOTMAGIC");
        f.write_all(&header).unwrap();
    }
    // With bad magic and no valid NPZ, it should fail
    let res = load_cache(&bad_magic_path, None);
    assert!(res.is_err());

    // 2. Unsupported version
    let bad_version_path = dir.path().join("bad_ver.bin");
    {
        let mut f = File::create(&bad_version_path).unwrap();
        let mut header = HifiBinHeader::new(128, 10, 1.0).to_bytes();
        header[8..12].copy_from_slice(&2u32.to_le_bytes()); // version 2
        f.write_all(&header).unwrap();
    }
    let res = load_cache(&bad_version_path, None);
    assert!(res.is_err());

    // 3. Size mismatch
    let size_mismatch_path = dir.path().join("size_mismatch.bin");
    {
        let mut f = File::create(&size_mismatch_path).unwrap();
        let header = HifiBinHeader::new(128, 10, 1.0).to_bytes(); // expects 28 + 128*10*4 = 5148 bytes
        f.write_all(&header).unwrap();
        f.write_all(&[0u8; 100]).unwrap(); // only wrote 100 bytes of payload
    }
    let res = load_cache(&size_mismatch_path, None);
    match res {
        Err(CacheError::SizeMismatch {
            expected, actual, ..
        }) => {
            assert_eq!(expected, 28 + 128 * 10 * 4);
            assert_eq!(actual, 128);
        }
        other => panic!("Expected SizeMismatch, got {:?}", other),
    }

    // 4. Truncated header (< 28 bytes)
    let trunc_path = dir.path().join("trunc.bin");
    {
        let mut f = File::create(&trunc_path).unwrap();
        f.write_all(b"HIFIBIN1short").unwrap();
    }
    let res = load_cache(&trunc_path, None);
    assert!(res.is_err());
}

#[test]
fn test_legacy_npz_fallback() {
    let dir = tempdir().unwrap();
    let bin_path = dir.path().join("song.hifi.bin");
    let npz_path = dir.path().join("song.hifi.npz");

    // Write a genuine .npz archive using ndarray_npy
    let mel = Array2::from_shape_vec((128, 16), vec![0.42f32; 128 * 16]).unwrap();
    let scale: Array0<f64> = Array0::from_elem((), 0.95f64);

    {
        let file = File::create(&npz_path).unwrap();
        let mut npz = ndarray_npy::NpzWriter::new(file);
        npz.add_array("mel_origin", &mel).unwrap();
        npz.add_array("scale", &scale).unwrap();
        npz.finish().unwrap();
    }

    assert!(npz_path.exists());
    assert!(!bin_path.exists());

    // Requesting song.hifi.bin should automatically find song.hifi.npz
    let loaded = load_cache(&bin_path, None)
        .expect("Failed to load legacy npz fallback")
        .expect("Expected Some(CachedFeatures)");

    assert_eq!(loaded.mel_origin.shape(), &[128, 16]);
    assert_relative_eq!(loaded.scale, 0.95, epsilon = 1e-6);
    assert_relative_eq!(loaded.mel_origin[[0, 0]], 0.42, epsilon = 1e-6);
}

#[test]
fn test_atomic_save_leaves_no_tmp_files() {
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("clean.hifi.bin");

    let mel = Array2::from_shape_vec((128, 5), vec![0.1f32; 128 * 5]).unwrap();
    save_cache(&cache_path, &mel, 1.0).unwrap();

    // Check directory entries
    let entries: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0], "clean.hifi.bin");
}

#[test]
fn test_cache_manager_stat_sig_invalidation_and_npz_non_finite_scale() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("voice.wav");
    let bin_path = dir.path().join("voice.hifi.bin");
    let npz_path = dir.path().join("voice.hifi.npz");

    {
        let mut f = File::create(&wav_path).unwrap();
        f.write_all(b"v1").unwrap();
    }
    let wav_sig_1 = stat_sig(&wav_path).unwrap();

    let cm = CacheManager::new();
    let mel = Array2::from_shape_vec((8, 4), vec![1.0f32; 32]).unwrap();
    cm.save_features_cache(&bin_path, &mel, 0.75, false, Some(&wav_path))
        .unwrap();
    assert!(cm
        .load_features_cache(&bin_path, false, Some(&wav_path))
        .is_some());

    // Change source wav size while updating file -> stat_sig (mtime_ns, size) invalidates RAM entry
    std::thread::sleep(Duration::from_millis(30));
    {
        let mut f = File::create(&wav_path).unwrap();
        f.write_all(b"v2-longer-wav").unwrap();
    }
    let wav_sig_2 = stat_sig(&wav_path).unwrap();
    assert_ne!(wav_sig_1, wav_sig_2);
    assert!(cm
        .load_features_cache(&bin_path, false, Some(&wav_path))
        .is_none());

    // WAV cache invalidation on size/mtime change
    let buf = AudioBuffer::new(vec![0.1, 0.2], DEFAULT_SAMPLE_RATE, 1);
    cm.save_wav_cache(&wav_path, &buf);
    assert!(cm.load_wav_cache(&wav_path).is_some());
    std::thread::sleep(Duration::from_millis(30));
    {
        let mut f = File::create(&wav_path).unwrap();
        f.write_all(b"v3-even-longer").unwrap();
    }
    assert!(cm.load_wav_cache(&wav_path).is_none());

    // resolve_feature_cache_paths falls back to .npz when .bin absent
    fs::remove_file(&bin_path).unwrap();
    let scale_nan = Array0::from_elem((), f64::NAN);
    {
        let file = File::create(&npz_path).unwrap();
        let mut npz = ndarray_npy::NpzWriter::new(file);
        npz.add_array("mel_origin", &mel).unwrap();
        npz.add_array("scale", &scale_nan).unwrap();
        npz.finish().unwrap();
    }
    let (read_p, write_p) = resolve_feature_cache_paths(&bin_path);
    assert_eq!(read_p, npz_path);
    assert_eq!(write_p, bin_path);
    assert!(matches!(
        load_cache(&bin_path, None),
        Err(CacheError::NonFiniteScale(_))
    ));
}
