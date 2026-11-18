use approx::assert_relative_eq;
use hifi_lufs::{
    apply_envelope_aware_gain, apply_peak_limit, compute_lufs_target_gain, k_weighting_sos,
    loudness_norm, loudness_norm_fast, measure_lufs, measure_lufs_multichannel, peak_normalize,
    resample_poly_fast, EnvelopeGainOptions, LoudnessNormOptions, LufsError, TailMode,
    K_WEIGHT_SOS_44100,
};
use std::f64::consts::PI;
use std::path::PathBuf;
use std::process::Command;

fn python_exe() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "../../../.venv/Scripts/python.exe",
        "../../../.venv/bin/python",
    ] {
        let cand = manifest_dir.join(rel);
        if cand.exists() {
            return cand.canonicalize().unwrap_or(cand);
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.join("../../..");
    root.canonicalize().unwrap_or(root)
}

fn make_sine(freq: f64, sr: u32, duration: f64, amp: f32) -> Vec<f32> {
    let n = (sr as f64 * duration).round() as usize;
    (0..n)
        .map(|i| amp * ((2.0 * PI * freq * (i as f64) / (sr as f64)).sin() as f32))
        .collect()
}

fn make_mixed(sr: u32, duration: f64) -> Vec<f32> {
    let n = (sr as f64 * duration).round() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sr as f64;
            let v = 0.25 * (2.0 * PI * 220.0 * t).sin()
                + 0.15 * (2.0 * PI * 880.0 * t).sin()
                + 0.08 * (2.0 * PI * 2640.0 * t).sin()
                + 0.02 * (2.0 * PI * 5100.0 * t).cos();
            v as f32
        })
        .collect()
}

#[test]
fn test_k_weighting_sos_44100_and_arbitrary_rates() {
    let sos_441 = k_weighting_sos(44100).unwrap();
    assert_eq!(sos_441, K_WEIGHT_SOS_44100);

    // Invalid rates
    assert!(matches!(
        k_weighting_sos(0),
        Err(LufsError::InvalidSampleRate(0))
    ));
    assert!(matches!(
        k_weighting_sos(3000),
        Err(LufsError::SampleRateTooLowForKWeighting { .. })
    ));

    // Compare 48000, 22050, 16000 against Python util.lufs._k_weighting_sos
    let py_code = format!(
        "import sys, json; sys.path.insert(0, r'{}'); \
         from util.lufs import _k_weighting_sos; \
         out = {{str(sr): _k_weighting_sos(sr).tolist() for sr in (16000, 22050, 44100, 48000, 96000)}}; \
         print(json.dumps(out))",
        repo_root().display()
    );
    let output = Command::new(python_exe())
        .args(["-c", &py_code])
        .output()
        .expect("Failed to execute python");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let trimmed = stdout.trim();

    for sr in [16000u32, 22050, 44100, 48000, 96000] {
        let sos = k_weighting_sos(sr).unwrap();
        // Verify key marker in JSON output and check numerical values by querying single rate
        assert!(trimmed.contains(&format!("\"{}\":", sr)));
        let single_py = format!(
            "import sys; sys.path.insert(0, r'{}'); \
             from util.lufs import _k_weighting_sos; \
             s = _k_weighting_sos({}); \
             print(','.join(f'{{v:.17g}}' for row in s for v in row))",
            repo_root().display(),
            sr
        );
        let out = Command::new(python_exe())
            .args(["-c", &single_py])
            .output()
            .unwrap();
        assert!(out.status.success());
        let vals: Vec<f64> = String::from_utf8(out.stdout)
            .unwrap()
            .trim()
            .split(',')
            .map(|x| x.parse::<f64>().unwrap())
            .collect();
        assert_eq!(vals.len(), 12);
        for r in 0..2 {
            for c in 0..6 {
                assert_relative_eq!(sos[r][c], vals[r * 6 + c], epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn test_measure_lufs_matches_python_across_rates_and_multichannel() {
    for sr in [16000u32, 22050, 44100, 48000] {
        let sig = make_mixed(sr, 1.5);
        let rust_lufs = measure_lufs(&sig, sr).unwrap();

        let py_code = format!(
            "import sys, math; sys.path.insert(0, r'{}'); \
             import numpy as np; from util.lufs import measure_lufs; \
             n = int(round({} * 1.5)); \
             t = np.arange(n, dtype=np.float64) / {}; \
             sig = (0.25*np.sin(2*math.pi*220*t) + 0.15*np.sin(2*math.pi*880*t) + 0.08*np.sin(2*math.pi*2640*t) + 0.02*np.cos(2*math.pi*5100*t)).astype(np.float32); \
             print(f'{{measure_lufs(sig, {}):.12f}}')",
            repo_root().display(),
            sr,
            sr,
            sr
        );
        let out = Command::new(python_exe())
            .args(["-c", &py_code])
            .output()
            .unwrap();
        assert!(out.status.success());
        let py_lufs: f64 = String::from_utf8(out.stdout)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_relative_eq!(rust_lufs, py_lufs, epsilon = 1e-4);
    }

    // Multichannel stereo test
    let sr = 44100u32;
    let left = make_sine(440.0, sr, 1.2, 0.35);
    let right = make_mixed(sr, 1.2);
    let mut stereo = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.iter().zip(right.iter()) {
        stereo.push(*l);
        stereo.push(*r);
    }
    let rust_stereo = measure_lufs_multichannel(&stereo, sr, 2, 0.400).unwrap();
    let py_stereo_code = format!(
        "import sys, math; sys.path.insert(0, r'{}'); \
         import numpy as np; from util.lufs import measure_lufs; \
         sr = 44100; n = int(round(sr * 1.2)); \
         t = np.arange(n, dtype=np.float64) / sr; \
         l = (0.35 * np.sin(2*math.pi*440*t)).astype(np.float32); \
         r = (0.25*np.sin(2*math.pi*220*t) + 0.15*np.sin(2*math.pi*880*t) + 0.08*np.sin(2*math.pi*2640*t) + 0.02*np.cos(2*math.pi*5100*t)).astype(np.float32); \
         st = np.stack([l, r], axis=1); \
         print(f'{{measure_lufs(st, sr):.12f}}')",
        repo_root().display()
    );
    let out = Command::new(python_exe())
        .args(["-c", &py_stereo_code])
        .output()
        .unwrap();
    assert!(out.status.success());
    let py_stereo: f64 = String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_relative_eq!(rust_stereo, py_stereo, epsilon = 1e-4);
}

#[test]
fn test_measure_lufs_silence_gates_and_short() {
    assert_eq!(measure_lufs(&[], 44100).unwrap(), f64::NEG_INFINITY);
    assert_eq!(
        measure_lufs(&vec![0.0; 44100], 44100).unwrap(),
        f64::NEG_INFINITY
    );
    assert_eq!(
        measure_lufs(&vec![0.0; 500], 44100).unwrap(),
        f64::NEG_INFINITY
    );

    // Below -70 LUFS absolute gate
    let ultra_quiet = make_sine(440.0, 44100, 1.0, 1e-5);
    assert_eq!(
        measure_lufs(&ultra_quiet, 44100).unwrap(),
        f64::NEG_INFINITY
    );

    // Non-finite samples sanitized safely
    let mut with_nan = make_sine(440.0, 44100, 1.0, 0.25);
    with_nan[10] = f32::NAN;
    with_nan[20] = f32::INFINITY;
    assert!(measure_lufs(&with_nan, 44100).unwrap().is_finite());

    // Relative gate (-10 LU): loud segment (0.8s @ 0.35) + quiet segment (0.8s @ 0.02, -24.86 dB).
    // 5 pure-loud blocks + 3 transition blocks (0.75, 0.50, 0.25 of P_loud) pass -10 LU gate;
    // 5 pure-quiet blocks are dropped by relative gate -> exact mean = 6.5/8 = -0.902 dB vs loud_only
    // (whereas without relative gating, mean across all 13 blocks would drop by ~3.0 dB).
    let loud = make_sine(440.0, 44100, 0.8, 0.35);
    let quiet = make_sine(440.0, 44100, 0.8, 0.02);
    let mut combined = loud.clone();
    combined.extend_from_slice(&quiet);
    let loud_only = measure_lufs(&loud, 44100).unwrap();
    let gated = measure_lufs(&combined, 44100).unwrap();
    let expected_delta_db = 10.0 * (6.5_f64 / 8.0).log10();
    assert!(
        (gated - (loud_only + expected_delta_db)).abs() < 0.05,
        "gated={gated}, loud_only={loud_only}, expected_delta={expected_delta_db}"
    );
    assert!(
        gated > loud_only - 1.5,
        "relative gate failed to reject quiet blocks"
    );

    // Short (<400 ms) signal matches long tone
    let short_sine = make_sine(440.0, 44100, 0.1, 0.5);
    let long_sine = make_sine(440.0, 44100, 1.0, 0.5);
    let short_lufs = measure_lufs(&short_sine, 44100).unwrap();
    let long_lufs = measure_lufs(&long_sine, 44100).unwrap();
    assert!(short_lufs.is_finite());
    assert!((short_lufs - long_lufs).abs() < 0.15);
}

#[test]
fn test_envelope_aware_gain_tail_protection_and_python_parity() {
    let sr = 44100u32;
    let body_len = (0.6 * sr as f64) as usize;
    let tail_len = (0.6 * sr as f64) as usize;

    let mut audio = Vec::with_capacity(body_len + tail_len);
    for i in 0..body_len {
        let t = i as f64 / sr as f64;
        audio.push((0.08 * (2.0 * PI * 440.0 * t).sin()) as f32);
    }
    for i in 0..tail_len {
        let t = i as f64 / sr as f64;
        audio.push((0.006 * (2.0 * PI * 1200.0 * t).sin()) as f32);
    }
    audio[body_len + tail_len / 2] = 0.09;

    let target_gain = 10.0_f64.powf(16.0 / 20.0);
    let tail_limit_dbfs = -6.0;
    let tail_limit_lin = 10.0_f32.powf((tail_limit_dbfs / 20.0) as f32);

    let out_preserve = apply_envelope_aware_gain(
        &audio,
        target_gain,
        sr,
        EnvelopeGainOptions {
            low_level_protect_db: -40.0,
            tail_peak_limit_dbfs: tail_limit_dbfs,
            tail_mode: TailMode::PreserveRelative,
            max_boost_db: Some(18.0),
        },
    )
    .unwrap();

    let out_clip = apply_envelope_aware_gain(
        &audio,
        target_gain,
        sr,
        EnvelopeGainOptions {
            low_level_protect_db: -40.0,
            tail_peak_limit_dbfs: tail_limit_dbfs,
            tail_mode: TailMode::Clip,
            max_boost_db: Some(18.0),
        },
    )
    .unwrap();

    // Sustained body gets full boost
    let body_rms_in: f64 = (audio[..(0.4 * sr as f64) as usize]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        / (0.4 * sr as f64))
        .sqrt();
    let body_rms_out: f64 = (out_preserve[..(0.4 * sr as f64) as usize]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        / (0.4 * sr as f64))
        .sqrt();
    assert!(body_rms_out / body_rms_in > target_gain * 0.9);

    // Tail peak limiter enforced
    let tail_start = body_len + (0.1 * sr as f64) as usize;
    let max_preserve_tail = out_preserve[tail_start..]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    let max_clip_tail = out_clip[tail_start..]
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);
    assert!(max_preserve_tail <= tail_limit_lin + 1e-6);
    assert!(max_clip_tail <= tail_limit_lin + 1e-6);

    // PreserveRelative retains more quiet tail than Clip
    let quiet_end = body_len + tail_len / 2 - 500;
    let rms_preserve: f64 = out_preserve[tail_start..quiet_end]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let rms_clip: f64 = out_clip[tail_start..quiet_end]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(rms_preserve > rms_clip);

    // Numerical parity with Python util.lufs.apply_envelope_aware_gain
    let py_code = format!(
        "import sys, math; sys.path.insert(0, r'{}'); \
         import numpy as np; from util.lufs import apply_envelope_aware_gain; \
         sr = 44100; body_len = int(0.6*sr); tail_len = int(0.6*sr); \
         body = 0.08 * np.sin(2*math.pi*440*np.arange(body_len)/sr); \
         tail = 0.006 * np.sin(2*math.pi*1200*np.arange(tail_len)/sr); \
         tail[tail_len//2] = 0.09; \
         audio = np.concatenate([body, tail]).astype(np.float32); \
         out = apply_envelope_aware_gain(audio, 10**(16/20), sr); \
         sys.stdout.buffer.write(out.tobytes())",
        repo_root().display()
    );
    let py_out = Command::new(python_exe())
        .args(["-c", &py_code])
        .output()
        .unwrap();
    assert!(py_out.status.success());
    let py_samples: Vec<f32> = py_out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    assert_eq!(py_samples.len(), out_preserve.len());
    let max_diff = out_preserve
        .iter()
        .zip(py_samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-5, "max_diff={max_diff}");
}

#[test]
fn test_loudness_norm_fast_strength_trim_and_python_parity() {
    let sr = 44100u32;
    let sig: Vec<f32> = make_mixed(sr, 1.5).into_iter().map(|v| v * 0.25).collect();
    let in_lufs = measure_lufs(&sig, sr).unwrap();

    let norm_100 = loudness_norm(&sig, sr, -16.0, 100.0, false, -52.0).unwrap();
    assert!((measure_lufs(&norm_100, sr).unwrap() - (-16.0)).abs() < 0.6);

    let norm_50 = loudness_norm(&sig, sr, -16.0, 50.0, false, -52.0).unwrap();
    let expected_mid = in_lufs + (-16.0 - in_lufs) * 0.5;
    assert!((measure_lufs(&norm_50, sr).unwrap() - expected_mid).abs() < 0.6);

    let norm_0 = loudness_norm(&sig, sr, -16.0, 0.0, false, -52.0).unwrap();
    assert_eq!(norm_0, sig);

    // Silence & sub-gate (-inf LUFS) stay finite and unexploded
    let zeros = vec![0.0_f32; sr as usize];
    let out_zeros = loudness_norm(&zeros, sr, -16.0, 100.0, true, -52.0).unwrap();
    assert!(out_zeros.iter().all(|&v| v == 0.0));

    let ultra_quiet = make_sine(440.0, sr, 0.8, 1e-5);
    let out_quiet = loudness_norm(&ultra_quiet, sr, -16.0, 100.0, true, -52.0).unwrap();
    assert_eq!(out_quiet, ultra_quiet);

    // Short burst with silence trimming & 5ms crossfade
    let full_len = (sr as f64 * 0.6) as usize;
    let burst_len = (sr as f64 * 0.12) as usize;
    let mut short_burst = vec![0.0_f32; full_len];
    let burst = make_sine(440.0, sr, 0.12, 0.15);
    short_burst[sr as usize / 10..sr as usize / 10 + burst_len].copy_from_slice(&burst);
    let normed = loudness_norm_fast(
        &short_burst,
        sr,
        LoudnessNormOptions {
            target_loudness: -16.0,
            trim_silence: true,
            ..LoudnessNormOptions::default()
        },
    )
    .unwrap();
    assert_eq!(normed.len(), full_len);
    assert!(normed.iter().all(|v| v.is_finite()));
    let tail_start = (sr as f64 * 0.45) as usize;
    assert_eq!(
        normed[tail_start..]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max),
        0.0
    );

    // Parity against Python loudness_norm_fast on trimmed signal with lead+tail crossfade
    let py_code = format!(
        "import sys, math; sys.path.insert(0, r'{}'); \
         import numpy as np; from util.lufs import loudness_norm_fast; \
         sr = 44100; full_len = int(sr*0.6); burst_len = int(sr*0.12); \
         x = np.zeros(full_len, dtype=np.float32); \
         t = np.arange(burst_len, dtype=np.float64)/sr; \
         x[sr//10:sr//10+burst_len] = (0.15 * np.sin(2*math.pi*440*t)).astype(np.float32); \
         y = loudness_norm_fast(x, sr, loudness=-16.0, trim_silence=True); \
         sys.stdout.buffer.write(y.tobytes())",
        repo_root().display()
    );
    let py_out = Command::new(python_exe())
        .args(["-c", &py_code])
        .output()
        .unwrap();
    assert!(py_out.status.success());
    let py_samples: Vec<f32> = py_out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    let max_diff = normed
        .iter()
        .zip(py_samples.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-5, "loudness_norm_fast max_diff={max_diff}");
}

#[test]
fn test_resample_poly_fast_lengths_spectrum_and_python_parity() {
    let cases = [
        (48000u32, 44100u32, 4800usize),
        (48000, 44100, 4799),
        (22050, 44100, 2205),
        (22050, 44100, 1235),
        (44100, 48000, 4410),
        (16000, 44100, 1601),
        (44100, 44101, 4410), // triggers limit_denominator(512) -> 1/1
    ];

    for &(orig_sr, target_sr, n) in &cases {
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / orig_sr as f64;
                (0.5 * (2.0 * PI * 440.0 * t).sin() + 0.25 * (2.0 * PI * 1760.0 * t).cos()) as f32
            })
            .collect();
        let rust_y = resample_poly_fast(&x, orig_sr, target_sr).unwrap();
        let expected_len =
            ((n as f64 * target_sr as f64) / orig_sr as f64).round_ties_even() as usize;
        assert_eq!(rust_y.len(), expected_len);
        assert!(rust_y.iter().all(|v| v.is_finite()));

        let py_code = format!(
            "import sys, math; sys.path.insert(0, r'{}'); \
             import numpy as np; from util.lufs import resample_poly_fast; \
             t = np.arange({}, dtype=np.float64) / {}; \
             x = (0.5*np.sin(2*math.pi*440*t) + 0.25*np.cos(2*math.pi*1760*t)).astype(np.float32); \
             y = resample_poly_fast(x, {}, {}); \
             sys.stdout.buffer.write(y.tobytes())",
            repo_root().display(),
            n,
            orig_sr,
            orig_sr,
            target_sr
        );
        let py_out = Command::new(python_exe())
            .args(["-c", &py_code])
            .output()
            .unwrap();
        assert!(py_out.status.success());
        let py_y: Vec<f32> = py_out
            .stdout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(rust_y.len(), py_y.len());
        let max_diff = rust_y
            .iter()
            .zip(py_y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 2e-6,
            "orig_sr={orig_sr}, target_sr={target_sr}, n={n}, max_diff={max_diff}"
        );
    }

    // Empty, single-sample, and identity checks
    assert!(resample_poly_fast(&[], 48000, 44100).unwrap().is_empty());
    let single = resample_poly_fast(&[0.42_f32], 22050, 44100).unwrap();
    assert_eq!(single.len(), 2);
    assert!(single.iter().all(|v| v.is_finite()));

    let orig = make_sine(440.0, 44100, 0.1, 0.5);
    assert_eq!(resample_poly_fast(&orig, 44100, 44100).unwrap(), orig);
}

#[test]
fn test_peak_normalize_and_peak_limit_helpers() {
    let quiet = vec![0.1_f32, -0.2, 0.05];
    let normed = peak_normalize(&quiet, -6.0);
    let expected_peak = 10.0_f32.powf(-6.0 / 20.0);
    let actual_peak = normed.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert_relative_eq!(actual_peak, expected_peak, epsilon = 1e-6);

    let mut hot = vec![0.5_f32, -1.5, 0.25];
    apply_peak_limit(&mut hot, 0.99);
    let limited_peak = hot.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert_relative_eq!(limited_peak, 0.99, epsilon = 1e-6);
}

#[test]
fn test_compute_lufs_target_gain_matches_norm_and_respects_strength() {
    let sr = 44100u32;
    let tone = make_sine(440.0, sr, 0.8, 0.08);
    let tone_lufs = measure_lufs(&tone, sr).unwrap();

    // P100 targeting -16.0 LUFS
    let opts_full = LoudnessNormOptions {
        target_loudness: -16.0,
        strength: 100.0,
        trim_silence: true,
        silence_threshold: -52.0,
        ..LoudnessNormOptions::default()
    };
    let gain_full = compute_lufs_target_gain(&tone, sr, opts_full).unwrap();
    let expected_db = (-16.0 - tone_lufs).min(18.0);
    let expected_gain = 10.0_f64.powf(expected_db / 20.0);
    assert_relative_eq!(gain_full, expected_gain, epsilon = 1e-4);

    // P50 halves the dB shift
    let opts_half = LoudnessNormOptions {
        strength: 50.0,
        ..opts_full
    };
    let gain_half = compute_lufs_target_gain(&tone, sr, opts_half).unwrap();
    let expected_half_gain = 10.0_f64.powf((0.5 * (-16.0 - tone_lufs)).min(18.0) / 20.0);
    assert_relative_eq!(gain_half, expected_half_gain, epsilon = 1e-4);

    // P0 and silent buffers yield unity 1.0
    let opts_zero = LoudnessNormOptions {
        strength: 0.0,
        ..opts_full
    };
    assert_relative_eq!(
        compute_lufs_target_gain(&tone, sr, opts_zero).unwrap(),
        1.0,
        epsilon = 1e-12
    );
    let silent = vec![0.0_f32; 4410];
    assert_relative_eq!(
        compute_lufs_target_gain(&silent, sr, opts_full).unwrap(),
        1.0,
        epsilon = 1e-12
    );
}
