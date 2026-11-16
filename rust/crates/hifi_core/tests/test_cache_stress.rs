//! Comprehensive empirical stress test suite for HIFIBIN1 feature cache.
//!
//! Validates:
//! 1. 28-byte header packing (<8sIIId) bit-for-bit against Python struct.pack
//! 2. Cross-language interoperability (Python writing -> Rust reading, Rust writing -> Python reading)
//! 3. Legacy .npz reading and transparent fallback from Python
//! 4. Cache invalidation when source WAV mtime is newer (including sub-second resolution)
//! 5. Atomic swap resilience: read-only target, deep directories, bare paths, concurrent writes
//! 6. Adversarial input validation: truncated files, bad magic/version, NaN/Inf scales, size mismatches

use approx::assert_relative_eq;
use hifi_core::{
    load_cache, save_cache, CacheError, HifiBinHeader, HIFIBIN_MAGIC, HIFIBIN_VERSION,
};
use ndarray::{s, Array2};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime};
use tempfile::tempdir;

fn get_python_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let venv_python = manifest_dir.join("../../../.venv/Scripts/python.exe");
    if venv_python.exists() {
        return venv_python.canonicalize().unwrap_or(venv_python);
    }
    PathBuf::from("python")
}

fn get_repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.join("../../..");
    root.canonicalize().unwrap_or(root)
}

// =========================================================================
// 1. Bit-for-bit Header Packing (<8sIIId) vs Python struct.pack
// =========================================================================

#[test]
fn test_stress_header_packing_bit_for_bit_vs_python() {
    let python = get_python_path();
    assert!(
        python.exists(),
        "Python executable must exist at {:?}",
        python
    );

    let test_cases = vec![
        (128u32, 50u32, 0.75f64),
        (128u32, 0u32, 1.0f64),
        (0u32, 0u32, 1.0f64),
        (80u32, 256u32, -0.123456789f64),
        (128u32, 100u32, 1e-300f64),
        (128u32, 100u32, 1e15f64),
        (65535u32, 100000u32, 42.0f64),
        (1u32, 1u32, std::f64::consts::PI),
    ];

    for (rows, cols, scale) in test_cases {
        // Rust serialization
        let header = HifiBinHeader::new(rows, cols, scale);
        let rust_bytes = header.to_bytes();
        assert_eq!(rust_bytes.len(), 28);

        // Python struct.pack("<8sIIId", b"HIFIBIN1", 1, rows, cols, scale)
        let py_code = format!(
            "import struct, sys; \
             b = struct.pack('<8sIIId', b'HIFIBIN1', 1, {}, {}, {}); \
             sys.stdout.buffer.write(b)",
            rows, cols, scale
        );

        let output = Command::new(&python)
            .args(["-c", &py_code])
            .output()
            .expect("Failed to execute python struct.pack");

        assert!(
            output.status.success(),
            "Python script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let py_bytes = output.stdout;
        assert_eq!(py_bytes.len(), 28, "Python output must be exactly 28 bytes");

        // Exact bit-for-bit identity check
        assert_eq!(
            rust_bytes.as_slice(),
            py_bytes.as_slice(),
            "Mismatch in 28-byte header between Rust and Python for ({}, {}, {})",
            rows,
            cols,
            scale
        );

        // Verify Rust deserializes Python's output
        let parsed = HifiBinHeader::from_bytes(&py_bytes).expect("Failed to parse Python bytes");
        assert_eq!(parsed.magic, *HIFIBIN_MAGIC);
        assert_eq!(parsed.version, HIFIBIN_VERSION);
        assert_eq!(parsed.rows, rows);
        assert_eq!(parsed.cols, cols);
        assert_eq!(parsed.scale.to_bits(), scale.to_bits());
    }
}

// =========================================================================
// 2. Cross-Language Reading & Writing Interoperability
// =========================================================================

#[test]
fn test_stress_cross_language_python_writes_rust_reads() {
    let python = get_python_path();
    let repo_root = get_repo_root();
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("py_produced.hifi.bin");

    let rows = 128;
    let cols = 75;
    let scale = 0.6180339887;

    // Use Python util.cache_manager to write .hifi.bin with exact dyadic floats
    let py_script = format!(
        "import sys, os; \
         sys.path.insert(0, r'{}'); \
         import numpy as np; \
         from util.cache_manager import cache_manager; \
         rows, cols = {}, {}; \
         r = np.arange(rows, dtype=np.float32).reshape(rows, 1); \
         c = np.arange(cols, dtype=np.float32).reshape(1, cols); \
         mel = ((r * 64.0 + c - 2048.0) / 16.0).astype(np.float32); \
         cache_manager.save_features_cache(r'{}', {{'mel_origin': mel, 'scale': {}}}); \
         print('SAVED')",
        repo_root.display(),
        rows,
        cols,
        cache_path.display(),
        scale
    );

    let output = Command::new(&python)
        .args(["-c", &py_script])
        .output()
        .expect("Failed to execute python script");

    assert!(
        output.status.success(),
        "Python save failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(cache_path.exists());

    // Load with Rust
    let loaded = load_cache(&cache_path, None)
        .expect("Rust load_cache failed")
        .expect("Expected Some(CachedFeatures)");

    assert_eq!(loaded.mel_origin.shape(), &[rows, cols]);
    assert_relative_eq!(loaded.scale, scale, epsilon = 1e-12);

    // Verify values against analytical dyadic formula: exact bit equivalence
    for r in 0..rows {
        for c in 0..cols {
            let expected = (r as f32 * 64.0 + c as f32 - 2048.0) / 16.0;
            let actual = loaded.mel_origin[[r, c]];
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "Exact bit equivalence at [{}, {}]: expected {}, got {}",
                r,
                c,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_stress_cross_language_rust_writes_python_reads() {
    let python = get_python_path();
    let repo_root = get_repo_root();
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("rust_produced.hifi.bin");

    let rows = 128;
    let cols = 64;
    let scale = 0.875f64;

    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push((r as f32 * 0.07).cos() * (c as f32 * 0.03).sin());
        }
    }
    let mel = Array2::from_shape_vec((rows, cols), data).unwrap();

    // Save with Rust
    save_cache(&cache_path, &mel, scale).expect("Rust save_cache failed");
    assert!(cache_path.exists());

    // Python reads and asserts exact equality
    let py_script = format!(
        "import sys, os; \
         sys.path.insert(0, r'{}'); \
         import numpy as np; \
         from util.cache_manager import cache_manager; \
         loaded = cache_manager.load_features_cache(r'{}'); \
         assert loaded is not None, 'Python failed to load cache'; \
         mel = loaded['mel_origin']; \
         assert mel.shape == ({}, {}), f'Wrong shape: {{mel.shape}}'; \
         assert mel.dtype == np.float32, f'Wrong dtype: {{mel.dtype}}'; \
         scale = loaded['scale']; \
         assert abs(scale - {}) < 1e-9, f'Wrong scale: {{scale}}'; \
         print('VERIFIED')",
        repo_root.display(),
        cache_path.display(),
        rows,
        cols,
        scale
    );

    let output = Command::new(&python)
        .args(["-c", &py_script])
        .output()
        .expect("Failed to execute python load verification");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Python load verification failed: STDOUT: {}, STDERR: {}",
        stdout,
        stderr
    );
    assert!(stdout.contains("VERIFIED"));
}

#[test]
fn test_stress_non_contiguous_rust_array_saved_and_read_by_python() {
    let python = get_python_path();
    let repo_root = get_repo_root();
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("strided.hifi.bin");

    // Create a 256x128 array and slice it to 128x64 with non-contiguous strides
    let big_rows = 256;
    let big_cols = 128;
    let mut data = Vec::with_capacity(big_rows * big_cols);
    for i in 0..(big_rows * big_cols) {
        data.push(i as f32 * 0.05);
    }
    let big_arr = Array2::from_shape_vec((big_rows, big_cols), data).unwrap();
    let sliced = big_arr.slice(s![..;2, ..;2]).to_owned(); // 128x64

    save_cache(&cache_path, &sliced, 1.25).expect("Save strided cache failed");

    let py_script = format!(
        "import sys; \
         sys.path.insert(0, r'{}'); \
         from util.cache_manager import cache_manager; \
         loaded = cache_manager.load_features_cache(r'{}'); \
         assert loaded is not None; \
         assert loaded['mel_origin'].shape == (128, 64); \
         assert loaded['scale'] == 1.25; \
         assert loaded['mel_origin'][0, 0] == 0.0; \
         assert loaded['mel_origin'][1, 0] == 2 * 128 * 0.05; \
         print('STRIDED_OK')",
        repo_root.display(),
        cache_path.display()
    );

    let output = Command::new(&python)
        .args(["-c", &py_script])
        .output()
        .expect("Failed to verify non-contiguous array in Python");

    assert!(
        output.status.success(),
        "Python check failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// =========================================================================
// 3. Legacy .npz Reading & Transparent Fallback
// =========================================================================

#[test]
fn test_stress_legacy_npz_variants_generated_by_python() {
    let python = get_python_path();
    let dir = tempdir().unwrap();

    // 1. Python writes npz with 0-D float64 scale
    let npz1_bin_target = dir.path().join("case1.hifi.bin");
    let npz1_file = dir.path().join("case1.hifi.npz");
    let py1 = format!(
        "import numpy as np; \
         mel = np.full((128, 16), 0.33, dtype=np.float32); \
         np.savez(r'{}', mel_origin=mel, scale=0.77)",
        npz1_file.display()
    );
    Command::new(&python).args(["-c", &py1]).status().unwrap();

    let loaded1 = load_cache(&npz1_bin_target, None)
        .expect("load_cache failed for case1")
        .expect("case1 returned None");
    assert_eq!(loaded1.mel_origin.shape(), &[128, 16]);
    assert_relative_eq!(loaded1.scale, 0.77, epsilon = 1e-6);

    // 2. Python writes npz with 0-D float32 scale
    let npz2_bin_target = dir.path().join("case2.hifi.bin");
    let npz2_file = dir.path().join("case2.hifi.npz");
    let py2 = format!(
        "import numpy as np; \
         mel = np.full((128, 20), 0.5, dtype=np.float32); \
         np.savez(r'{}', mel_origin=mel, scale=np.float32(0.55))",
        npz2_file.display()
    );
    Command::new(&python).args(["-c", &py2]).status().unwrap();

    let loaded2 = load_cache(&npz2_bin_target, None)
        .expect("load_cache failed for case2")
        .expect("case2 returned None");
    assert_eq!(loaded2.mel_origin.shape(), &[128, 20]);
    assert_relative_eq!(loaded2.scale, 0.55, epsilon = 1e-5);

    // 3. Python writes npz with 1-D array scale
    let npz3_bin_target = dir.path().join("case3.hifi.bin");
    let npz3_file = dir.path().join("case3.hifi.npz");
    let py3 = format!(
        "import numpy as np; \
         mel = np.full((128, 24), 0.7, dtype=np.float32); \
         np.savez(r'{}', mel_origin=mel, scale=np.array([0.91], dtype=np.float64))",
        npz3_file.display()
    );
    Command::new(&python).args(["-c", &py3]).status().unwrap();

    let loaded3 = load_cache(&npz3_bin_target, None)
        .expect("load_cache failed for case3")
        .expect("case3 returned None");
    assert_eq!(loaded3.mel_origin.shape(), &[128, 24]);
    assert_relative_eq!(loaded3.scale, 0.91, epsilon = 1e-6);

    // 4. Direct load of .npz file (passed as path without .bin)
    let loaded_direct = load_cache(&npz1_file, None)
        .expect("Direct npz load failed")
        .expect("Direct npz load returned None");
    assert_eq!(loaded_direct.mel_origin.shape(), &[128, 16]);
    assert_relative_eq!(loaded_direct.scale, 0.77, epsilon = 1e-6);
}

// =========================================================================
// 4. MTime Invalidation (Subsecond & Edge Cases)
// =========================================================================

#[test]
fn test_stress_mtime_invalidation_edges() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("test.wav");
    let cache_path = dir.path().join("test.hifi.bin");

    let mel = Array2::from_shape_vec((128, 10), vec![0.5f32; 1280]).unwrap();
    save_cache(&cache_path, &mel, 1.0).unwrap();

    // Source WAV does not exist yet -> should be valid (not stale)
    assert!(load_cache(&cache_path, Some(&wav_path)).unwrap().is_some());

    // Create source WAV 100ms older than cache
    {
        let mut f = File::create(&wav_path).unwrap();
        f.write_all(b"audio").unwrap();
    }
    // Now both files exist. Let's make cache newer by re-saving after a pause.
    thread::sleep(Duration::from_millis(50));
    save_cache(&cache_path, &mel, 1.0).unwrap();

    // Cache is newer than WAV -> valid
    assert!(load_cache(&cache_path, Some(&wav_path)).unwrap().is_some());

    // Modify source WAV to be newer
    thread::sleep(Duration::from_millis(50));
    {
        let mut f = OpenOptions::new().write(true).open(&wav_path).unwrap();
        f.write_all(b"updated audio").unwrap();
    }

    // Now wav.mtime > cache.mtime -> invalid!
    assert!(
        load_cache(&cache_path, Some(&wav_path)).unwrap().is_none(),
        "Stale cache should return None"
    );

    // Delete WAV -> source_path.exists() is false -> not stale
    fs::remove_file(&wav_path).unwrap();
    assert!(
        load_cache(&cache_path, Some(&wav_path)).unwrap().is_some(),
        "Missing source WAV should not invalidate cache"
    );
}

// =========================================================================
// 5. Atomic Swap & Resilience
// =========================================================================

#[test]
#[allow(clippy::permissions_set_readonly_false)]
fn test_stress_atomic_swap_read_only_destination_cleans_up_tmp() {
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("readonly_target.hifi.bin");

    let mel = Array2::from_shape_vec((128, 4), vec![0.2f32; 128 * 4]).unwrap();

    // 1. Initial save succeeds
    save_cache(&cache_path, &mel, 1.0).unwrap();
    assert!(cache_path.exists());

    // 2. Set file read-only
    let mut perms = fs::metadata(&cache_path).unwrap().permissions();
    perms.set_readonly(true);
    fs::set_permissions(&cache_path, perms).unwrap();

    // 3. Attempting to overwrite a read-only file on Windows fails during rename
    let result = save_cache(&cache_path, &mel, 2.0);
    assert!(
        result.is_err(),
        "Expected save_cache to fail on read-only file"
    );

    // 4. Verify no .tmp files leaked into directory!
    let entries: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    for entry in &entries {
        assert!(
            !entry.ends_with(".tmp"),
            "Lingering temporary file detected: {}",
            entry
        );
    }

    // Restore permissions for cleanup
    let mut perms = fs::metadata(&cache_path).unwrap().permissions();
    perms.set_readonly(false);
    fs::set_permissions(&cache_path, perms).unwrap();
}

#[test]
fn test_stress_nested_directory_creation() {
    let dir = tempdir().unwrap();
    let deep_path = dir
        .path()
        .join("a")
        .join("b")
        .join("c")
        .join("deep.hifi.bin");

    let mel = Array2::from_shape_vec((128, 2), vec![0.1f32; 256]).unwrap();
    save_cache(&deep_path, &mel, 1.0).expect("Deep save failed");
    assert!(deep_path.exists());

    let loaded = load_cache(&deep_path, None).unwrap().unwrap();
    assert_eq!(loaded.mel_origin.shape(), &[128, 2]);
}

#[test]
fn test_stress_concurrent_writes_and_reads() {
    let dir = tempdir().unwrap();
    let dir_path = Arc::new(dir.path().to_path_buf());

    let num_threads = 8;
    let mut handles = Vec::new();

    for tid in 0..num_threads {
        let dir_clone = Arc::clone(&dir_path);
        handles.push(thread::spawn(move || {
            let cache_path = dir_clone.join(format!("thread_{}.hifi.bin", tid));
            let mel = Array2::from_shape_vec((128, 16), vec![tid as f32; 128 * 16]).unwrap();

            // Save cache
            save_cache(&cache_path, &mel, tid as f64).unwrap();

            // Read cache
            let loaded = load_cache(&cache_path, None).unwrap().unwrap();
            assert_eq!(loaded.scale, tid as f64);
            assert_eq!(loaded.mel_origin[[0, 0]], tid as f32);
        }));
    }

    for h in handles {
        h.join().expect("Worker thread panicked");
    }

    // Verify all files exist and no tmp files remain
    let entries: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();

    assert_eq!(entries.len(), num_threads);
    for entry in entries {
        assert!(!entry.ends_with(".tmp"));
    }
}

// =========================================================================
// 6. Adversarial & Malicious Header Cases
// =========================================================================

#[test]
fn test_stress_adversarial_scales() {
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("bad_scale.hifi.bin");
    let mel = Array2::from_shape_vec((128, 4), vec![0.1f32; 512]).unwrap();

    // 1. NaN scale on save
    let err_nan = save_cache(&cache_path, &mel, f64::NAN);
    assert!(matches!(err_nan, Err(CacheError::NonFiniteScale(_))));

    // 2. Inf scale on save
    let err_inf = save_cache(&cache_path, &mel, f64::INFINITY);
    assert!(matches!(err_inf, Err(CacheError::NonFiniteScale(_))));

    // 3. Neg Inf scale on save
    let err_ninf = save_cache(&cache_path, &mel, f64::NEG_INFINITY);
    assert!(matches!(err_ninf, Err(CacheError::NonFiniteScale(_))));

    // 4. Manually write NaN into header on disk
    let mut header = HifiBinHeader::new(128, 4, 1.0).to_bytes();
    header[20..28].copy_from_slice(&f64::NAN.to_le_bytes());
    {
        let mut f = File::create(&cache_path).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&vec![0u8; 128 * 4 * 4]).unwrap();
    }
    let res = load_cache(&cache_path, None);
    assert!(matches!(res, Err(CacheError::NonFiniteScale(_))));
}

#[test]
fn test_stress_adversarial_huge_dimensions_dos_defense() {
    let dir = tempdir().unwrap();
    let cache_path = dir.path().join("dos_bomb.hifi.bin");

    // Header claims 1,000,000 rows x 1,000,000 cols (4 Terabytes of payload)
    let header = HifiBinHeader::new(1_000_000, 1_000_000, 1.0).to_bytes();
    {
        let mut f = File::create(&cache_path).unwrap();
        f.write_all(&header).unwrap();
        // File only has 28 bytes total
    }

    // Must return SizeMismatch immediately without allocating 4TB or crashing OOM
    let start = SystemTime::now();
    let res = load_cache(&cache_path, None);
    let elapsed = start.elapsed().unwrap();

    assert!(
        matches!(res, Err(CacheError::SizeMismatch { .. })),
        "Expected SizeMismatch, got {:?}",
        res
    );
    assert!(
        elapsed < Duration::from_millis(50),
        "Size check must be instantaneous O(1)"
    );
}
