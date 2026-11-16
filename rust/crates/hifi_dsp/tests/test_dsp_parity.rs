use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use hifi_dsp::hnsep_stft::{istft_1d, stft_1d};
use hifi_dsp::mel::MAX_WINDOW_CACHE_SIZE;
use hifi_dsp::tension::{
    estimate_baseline_rd, extract_cepstral_formant_envelope, middle_ear_loudness_weight,
    solve_lf_unit_period_params,
};
use hifi_dsp::{
    apply_bounded_mask, apply_bounded_mask_tensor, couple_aperiodic_to_tension,
    dynamic_range_compression, dynamic_range_compression_default, midi_to_hz, periodic_hann_window,
    pre_emphasis_base_tension, reflect_pad_1d, resample_mel_and_pitch, slaney_mel_filterbank,
    Akima1DInterpolator, HnsepStft, MelConfig, MelSpectrogram, PadMode, ResampleTimeParams,
};
use ndarray::{Array2, Array4};
use num_complex::Complex32;

#[derive(Debug)]
enum FixtureData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
}

#[derive(Debug)]
struct FixtureRecord {
    shape: Vec<usize>,
    data: FixtureData,
}

impl FixtureRecord {
    fn as_f32(&self) -> &[f32] {
        match &self.data {
            FixtureData::F32(v) => v,
            _ => panic!("expected f32 fixture, got {:?}", self.data),
        }
    }

    fn as_f64(&self) -> &[f64] {
        match &self.data {
            FixtureData::F64(v) => v,
            _ => panic!("expected f64 fixture, got {:?}", self.data),
        }
    }

    fn as_i32(&self) -> &[i32] {
        match &self.data {
            FixtureData::I32(v) => v,
            _ => panic!("expected i32 fixture, got {:?}", self.data),
        }
    }

    fn as_array2_f32(&self) -> Array2<f32> {
        assert_eq!(self.shape.len(), 2);
        Array2::from_shape_vec((self.shape[0], self.shape[1]), self.as_f32().to_vec()).unwrap()
    }

    fn as_array4_f32(&self) -> Array4<f32> {
        assert_eq!(self.shape.len(), 4);
        Array4::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]),
            self.as_f32().to_vec(),
        )
        .unwrap()
    }
}

fn load_fixtures() -> HashMap<String, FixtureRecord> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("parity_fixtures.bin");
    let bytes = fs::read(&path).expect("parity_fixtures.bin should exist");
    assert_eq!(&bytes[..8], b"PARITY01");
    let mut off = 8usize;

    let read_u32 = |buf: &[u8], pos: &mut usize| -> u32 {
        let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    };

    let n_records = read_u32(&bytes, &mut off) as usize;
    let mut map = HashMap::with_capacity(n_records);

    for _ in 0..n_records {
        let name_len = read_u32(&bytes, &mut off) as usize;
        let kind = bytes[off];
        off += 1;
        let ndim = read_u32(&bytes, &mut off) as usize;
        let name = std::str::from_utf8(&bytes[off..off + name_len])
            .unwrap()
            .to_string();
        off += name_len;

        let mut shape = Vec::with_capacity(ndim);
        let mut num_elems = 1usize;
        for _ in 0..ndim {
            let d = read_u32(&bytes, &mut off) as usize;
            shape.push(d);
            num_elems *= d;
        }

        let data = match kind {
            0 => {
                let mut v = Vec::with_capacity(num_elems);
                for _ in 0..num_elems {
                    v.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                    off += 4;
                }
                FixtureData::F32(v)
            }
            1 => {
                let mut v = Vec::with_capacity(num_elems);
                for _ in 0..num_elems {
                    v.push(f64::from_le_bytes(bytes[off..off + 8].try_into().unwrap()));
                    off += 8;
                }
                FixtureData::F64(v)
            }
            2 => {
                let mut v = Vec::with_capacity(num_elems);
                for _ in 0..num_elems {
                    v.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                    off += 4;
                }
                FixtureData::I32(v)
            }
            other => panic!("unexpected fixture dtype tag {other}"),
        };

        map.insert(name, FixtureRecord { shape, data });
    }
    assert_eq!(off, bytes.len());
    map
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn test_parity_slaney_mel_and_drc() {
    let fix = load_fixtures();
    let fb_ref = fix["slaney_fb"].as_array2_f32();
    let fb_got = slaney_mel_filterbank(44100.0, 2048, 128, 40.0, Some(16000.0));
    assert_eq!(fb_got.dim(), fb_ref.dim());
    let diff = max_abs_diff_f32(fb_got.as_slice().unwrap(), fb_ref.as_slice().unwrap());
    assert!(diff < 1e-6, "slaney_mel_filterbank max_diff={diff}");

    let audio = fix["audio_main"].as_f32();
    let mel = MelSpectrogram::with_defaults();

    let got_default = mel.compute(audio, 0.0, 1.0).unwrap();
    let ref_default = fix["mel_default"].as_array2_f32();
    assert_eq!(got_default.dim(), ref_default.dim());
    let diff_def = max_abs_diff_f32(
        got_default.as_slice().unwrap(),
        ref_default.as_slice().unwrap(),
    );
    assert!(diff_def < 1e-4, "mel_default max_diff={diff_def}");

    let got_drc = dynamic_range_compression_default(got_default.view());
    let ref_drc = fix["mel_default_drc"].as_array2_f32();
    let diff_drc = max_abs_diff_f32(got_drc.as_slice().unwrap(), ref_drc.as_slice().unwrap());
    assert!(diff_drc < 1e-4, "mel_default_drc max_diff={diff_drc}");

    let got_up = mel.compute(audio, 1.2, 0.9).unwrap();
    let ref_up = fix["mel_shift_up"].as_array2_f32();
    assert_eq!(got_up.dim(), ref_up.dim());
    let diff_up = max_abs_diff_f32(got_up.as_slice().unwrap(), ref_up.as_slice().unwrap());
    assert!(diff_up < 1e-4, "mel_shift_up max_diff={diff_up}");

    // Gender g=1 => key_shift=-0.06 => n_fft_new=2041 (odd!)
    let got_odd = mel.compute_with_gender(audio, 1.0, 1.0).unwrap();
    let ref_odd = fix["mel_shift_odd"].as_array2_f32();
    assert_eq!(got_odd.dim(), ref_odd.dim());
    let diff_odd = max_abs_diff_f32(got_odd.as_slice().unwrap(), ref_odd.as_slice().unwrap());
    assert!(diff_odd < 1e-4, "mel_shift_odd max_diff={diff_odd}");

    // Short signal + center=true
    let mel_center = MelSpectrogram::new(MelConfig {
        center: true,
        ..MelConfig::default()
    })
    .unwrap();
    let got_short = mel_center.compute(&audio[..120], -0.8, 1.0).unwrap();
    let ref_short = fix["mel_short_center"].as_array2_f32();
    assert_eq!(got_short.dim(), ref_short.dim());
    let diff_short = max_abs_diff_f32(got_short.as_slice().unwrap(), ref_short.as_slice().unwrap());
    assert!(diff_short < 1e-4, "mel_short_center max_diff={diff_short}");

    // Custom C and clip_val DRC check
    let small = Array2::from_shape_vec((2, 2), vec![0.0_f32, -1.0, 1e-5, 2.0]).unwrap();
    let drc_custom = dynamic_range_compression(small.view(), 1.5, 1e-7);
    assert!((drc_custom[[0, 0]] - (1.5_f32 * 1e-7).ln()).abs() < 1e-6);
    assert!((drc_custom[[0, 1]] - (1.5_f32 * 1e-7).ln()).abs() < 1e-6);
    assert!((drc_custom[[1, 1]] - (1.5_f32 * 2.0).ln()).abs() < 1e-6);
}

#[test]
fn test_mel_lru_cache_and_thread_safety() {
    let mel = Arc::new(MelSpectrogram::with_defaults());
    let audio: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44100.0).sin() * 0.25)
        .collect();
    for i in 0..(MAX_WINDOW_CACHE_SIZE + 10) {
        let _ = mel.compute(&audio, -5.0 + 0.25 * i as f64, 1.0).unwrap();
    }
    assert_eq!(mel.hann_cache_len(), MAX_WINDOW_CACHE_SIZE);

    let ref_mel = mel.compute(&audio, 1.2, 1.0).unwrap();
    let mut handles = Vec::new();
    for _ in 0..6 {
        let m = Arc::clone(&mel);
        let a = audio.clone();
        handles.push(thread::spawn(move || m.compute(&a, 1.2, 1.0).unwrap()));
    }
    for h in handles {
        let res = h.join().unwrap();
        let diff = max_abs_diff_f32(res.as_slice().unwrap(), ref_mel.as_slice().unwrap());
        assert!(diff < 1e-6, "thread safety mismatch: {diff}");
    }
}

#[test]
fn test_parity_hnsep_stft_and_istft() {
    let fix = load_fixtures();
    let audio = fix["audio_main"].as_f32();
    let hnsep = HnsepStft::default();

    for (suffix, slice_len) in [("100", 100usize), ("main", audio.len())] {
        let prep = hnsep.prepare_input(&audio[..slice_len]).unwrap();
        let ref_in = fix[&format!("hnsep_in_{suffix}")].as_array4_f32();
        let mask_in = fix[&format!("hnsep_mask_{suffix}")].as_array4_f32();
        let ref_out = fix[&format!("hnsep_out_{suffix}")].as_f32();

        assert_eq!(prep.tensor.dim(), ref_in.dim());
        let diff_in = max_abs_diff_f32(prep.tensor.as_slice().unwrap(), ref_in.as_slice().unwrap());
        assert!(diff_in < 1e-4, "hnsep_in_{suffix} max_diff={diff_in}");

        let got_out = hnsep.reconstruct_from_mask(&prep, mask_in.view()).unwrap();
        let diff_out = max_abs_diff_f32(&got_out, ref_out);
        assert!(diff_out < 1e-4, "hnsep_out_{suffix} max_diff={diff_out}");
    }

    // Verify centered reflect-padded STFT -> ISTFT identity roundtrip
    let win = periodic_hann_window(2048);
    let sig = &audio[..8192];
    let spec = stft_1d(sig, 2048, 512, &win, true, PadMode::Reflect).unwrap();
    let recon = istft_1d(spec.view(), 2048, 512, &win, true, Some(sig.len())).unwrap();
    let rt_diff = max_abs_diff_f32(sig, &recon);
    assert!(rt_diff < 1e-5, "STFT/ISTFT roundtrip error {rt_diff}");

    // Verify bounded_mask clamping: |m_bounded| < 1
    let mut raw_mask = Array2::from_elem((4, 4), Complex32::new(50.0, -50.0));
    apply_bounded_mask(&mut raw_mask, 1e-8);
    for c in raw_mask.iter() {
        assert!(c.norm() <= 1.0 + 1e-6);
    }

    // Verify 1024-bin edge-replication fallback in reconstruct_from_mask
    let prep = hnsep.prepare_input(&audio[..100]).unwrap();
    let n_frames = prep.spec.ncols();
    let mask_1024 = Array4::<f32>::ones((1, 2, 1024, n_frames));
    let recon_1024 = hnsep
        .reconstruct_from_mask(&prep, mask_1024.view())
        .unwrap();
    assert_eq!(recon_1024.len(), 100);
    assert!(recon_1024.iter().all(|v| v.is_finite()));
}

#[test]
fn test_reflect_pad_1d_and_window_edge_cases() {
    assert_eq!(periodic_hann_window(0), Vec::<f32>::new());
    assert_eq!(periodic_hann_window(1), vec![1.0]);
    let w4 = periodic_hann_window(4);
    assert!((w4[0] - 0.0).abs() < 1e-6);
    assert!((w4[1] - 0.5).abs() < 1e-6);
    assert!((w4[2] - 1.0).abs() < 1e-6);
    assert!((w4[3] - 0.5).abs() < 1e-6);

    // Negative pad crops and short signal zero-prepads
    let p = reflect_pad_1d(&[1.0, 2.0, 3.0, 4.0], -1, -1);
    assert_eq!(p, vec![2.0, 3.0]);

    let short = reflect_pad_1d(&[1.0, 2.0], 3, 3);
    // Zero-padded to len 4 -> [1, 2, 0, 0], reflect(3,3) -> [0, 0, 2, 1, 2, 0, 0, 0, 2, 1]
    assert_eq!(
        short,
        vec![0.0, 0.0, 2.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0]
    );
}

#[test]
fn test_parity_akima_and_time_resampling() {
    let fix = load_fixtures();

    for name in ["step", "nonunif", "2", "3"] {
        let x = fix[&format!("akima_{name}_x")].as_f64();
        let y = fix[&format!("akima_{name}_y")].as_f64();
        let q = fix[&format!("akima_{name}_q")].as_f64();
        let expected = fix[&format!("akima_{name}_out")].as_f64();

        let spline = Akima1DInterpolator::new(x, y).unwrap();
        let got = spline.evaluate_many(q);
        let diff = max_abs_diff_f64(&got, expected);
        assert!(diff < 1e-12, "akima_{name} max_diff={diff}");

        // Out-of-bounds returns NaN when extrapolate=false
        assert!(spline.evaluate(x[0] - 0.1).is_nan());
        assert!(spline.evaluate(x[x.len() - 1] + 0.1).is_nan());
    }

    let mel_origin = fix["mel_origin_synth"].as_array2_f32();
    let pb_cents = fix["pb_cents"].as_i32();

    // Case 1: loop_mode = true, cutoff < 0
    let p_loop = ResampleTimeParams {
        loop_mode: true,
        base_pitch_midi: 69.0,
        velocity: 85.0,
        offset_ms: 40.0,
        length_ms: 350.0,
        consonant_ms: 30.0,
        cutoff_ms: -60.0,
        t_flag: 5.0,
        tempo: 135.0,
        ..ResampleTimeParams::default()
    };
    let res_loop = resample_mel_and_pitch(mel_origin.view(), pb_cents, &p_loop).unwrap();
    let ref_mel_loop = fix["mel_r_loop"].as_array2_f32();
    assert_eq!(res_loop.mel_render.dim(), ref_mel_loop.dim());
    assert!(
        max_abs_diff_f32(
            res_loop.mel_render.as_slice().unwrap(),
            ref_mel_loop.as_slice().unwrap()
        ) < 1e-5
    );
    assert!(max_abs_diff_f32(&res_loop.f0_render, fix["f0_r_loop"].as_f32()) < 1e-4);
    assert!(max_abs_diff_f64(&res_loop.pitch_render, fix["pitch_r_loop"].as_f64()) < 1e-11);
    assert!(max_abs_diff_f64(&res_loop.t_mel, fix["t_mel_loop"].as_f64()) < 1e-12);
    let sc_loop = fix["scalars_loop"].as_f64();
    assert!((res_loop.new_start - sc_loop[0]).abs() < 1e-12);
    assert!((res_loop.new_end - sc_loop[1]).abs() < 1e-12);
    assert!((res_loop.new_con - sc_loop[2]).abs() < 1e-12);
    assert_eq!(res_loop.cut_l, sc_loop[3] as usize);
    assert_eq!(res_loop.cut_r, sc_loop[4] as usize);

    // Case 2: loop_mode = false, cutoff > 0
    let p_noloop = ResampleTimeParams {
        loop_mode: false,
        base_pitch_midi: 69.0,
        velocity: 120.0,
        offset_ms: 20.0,
        length_ms: 420.0,
        consonant_ms: 25.0,
        cutoff_ms: 30.0,
        t_flag: -3.0,
        tempo: 110.0,
        ..ResampleTimeParams::default()
    };
    let res_noloop = resample_mel_and_pitch(mel_origin.view(), pb_cents, &p_noloop).unwrap();
    let ref_mel_noloop = fix["mel_r_noloop"].as_array2_f32();
    assert_eq!(res_noloop.mel_render.dim(), ref_mel_noloop.dim());
    assert!(
        max_abs_diff_f32(
            res_noloop.mel_render.as_slice().unwrap(),
            ref_mel_noloop.as_slice().unwrap()
        ) < 1e-5
    );
    assert!(max_abs_diff_f32(&res_noloop.f0_render, fix["f0_r_noloop"].as_f32()) < 1e-4);
    assert!(max_abs_diff_f64(&res_noloop.pitch_render, fix["pitch_r_noloop"].as_f64()) < 1e-11);
    let sc_noloop = fix["scalars_noloop"].as_f64();
    assert!((res_noloop.new_start - sc_noloop[0]).abs() < 1e-12);
    assert!((res_noloop.new_end - sc_noloop[1]).abs() < 1e-12);
    assert_eq!(res_noloop.cut_l, sc_noloop[3] as usize);
    assert_eq!(res_noloop.cut_r, sc_noloop[4] as usize);

    // Constant pitch check (A4 = 440 Hz)
    let res_const = resample_mel_and_pitch(
        mel_origin.view(),
        &[0, 0, 0],
        &ResampleTimeParams {
            base_pitch_midi: 69.0,
            length_ms: 100.0,
            ..ResampleTimeParams::default()
        },
    )
    .unwrap();
    assert!((res_const.f0_render[0] - 440.0).abs() < 1e-4);
    assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-12);
}

#[test]
fn test_parity_tension_and_aperiodic_coupling() {
    let fix = load_fixtures();

    let rds = fix["lf_rds"].as_f64();
    let lf_ref = fix["lf_params"].as_f64();
    for (i, &rd) in rds.iter().enumerate() {
        let (te, ta, wg, eps, alpha) = solve_lf_unit_period_params(rd);
        let base = i * 5;
        assert!((te - lf_ref[base]).abs() < 1e-12, "rd={rd} te mismatch");
        assert!((ta - lf_ref[base + 1]).abs() < 1e-12, "rd={rd} ta mismatch");
        assert!((wg - lf_ref[base + 2]).abs() < 1e-12, "rd={rd} wg mismatch");
        assert!(
            (eps - lf_ref[base + 3]).abs() < 1e-11,
            "rd={rd} eps mismatch"
        );
        assert!(
            (alpha - lf_ref[base + 4]).abs() < 1e-9,
            "rd={rd} alpha mismatch"
        );
    }

    let harm_sig = fix["harm_sig"].as_f32();

    let got_pos = pre_emphasis_base_tension(harm_sig, -1.0, 0.0, 0.0).unwrap();
    let diff_pos = max_abs_diff_f32(&got_pos, fix["tension_pos"].as_f32());
    assert!(diff_pos < 1e-4, "tension_pos max_diff={diff_pos}");

    let got_neg = pre_emphasis_base_tension(harm_sig, 1.5, 0.0, 0.0).unwrap();
    let diff_neg = max_abs_diff_f32(&got_neg, fix["tension_neg"].as_f32());
    assert!(diff_neg < 1e-4, "tension_neg max_diff={diff_neg}");

    let got_hohr = pre_emphasis_base_tension(harm_sig, -1.2, 45.0, -35.0).unwrap();
    let diff_hohr = max_abs_diff_f32(&got_hohr, fix["tension_hohr"].as_f32());
    assert!(diff_hohr < 1e-4, "tension_hohr max_diff={diff_hohr}");

    let got_short = pre_emphasis_base_tension(&harm_sig[..128], 0.5, 0.0, 0.0).unwrap();
    let diff_short = max_abs_diff_f32(&got_short, fix["tension_short"].as_f32());
    assert!(diff_short < 1e-4, "tension_short max_diff={diff_short}");

    // Aperiodic coupling parity + bit-identical consonant prefix
    let ap_noise = fix["ap_noise_1d"].as_f32();
    let voiced_part = fix["voiced_part_1d"].as_f32();
    let voiced_soft = fix["voiced_soft_1d"].as_f32();
    let got_ap =
        couple_aperiodic_to_tension(ap_noise, voiced_part, voiced_soft, 2.0, -40.0, 30.0).unwrap();
    let ref_ap = fix["ap_coupled"].as_f32();
    let diff_ap = max_abs_diff_f32(&got_ap, ref_ap);
    assert!(diff_ap < 1e-4, "ap_coupled max_diff={diff_ap}");

    let consonant_end = (0.12 * 44100.0) as usize;
    assert_eq!(&got_ap[..consonant_end], &ap_noise[..consonant_end]);
}

#[test]
fn test_tension_invariants_and_rd0_ordering() {
    // Silence returns zero without NaNs
    let silent = vec![0.0_f32; 4096];
    let silent_out = pre_emphasis_base_tension(&silent, -2.0, 0.0, 0.0).unwrap();
    assert_eq!(silent_out.len(), 4096);
    assert!(silent_out.iter().all(|&x| x == 0.0));

    // Bypass when b=ho=hr=0
    let probe = vec![0.125_f32; 128];
    let bypass = pre_emphasis_base_tension(&probe, 0.0, 0.0, 0.0).unwrap();
    assert_eq!(bypass, probe);

    // Sharp vs lax baseline Rd0 ordering
    let sr = 44100u32;
    let n_fft = 2048usize;
    let win = periodic_hann_window(n_fft);
    let mut sharp = vec![0.0_f32; n_fft];
    let mut lax = vec![0.0_f32; n_fft];
    for (i, (s_out, l_out)) in sharp.iter_mut().zip(lax.iter_mut()).enumerate() {
        let t = i as f64 / sr as f64;
        for k in 1..25 {
            let phase = (2.0 * std::f64::consts::PI * 220.0 * k as f64 * t).sin();
            *s_out += (1.0 / (k as f64).powf(0.7)) as f32 * phase as f32;
            *l_out += (1.0 / (k as f64).powf(2.4)) as f32 * phase as f32;
        }
    }
    let spec_s = stft_1d(&sharp, n_fft, 512, &win, false, PadMode::Constant(0.0)).unwrap();
    let spec_l = stft_1d(&lax, n_fft, 512, &win, false, PadMode::Constant(0.0)).unwrap();
    let p_sharp: Vec<f32> = spec_s.column(0).iter().map(|c| c.norm_sqr()).collect();
    let p_lax: Vec<f32> = spec_l.column(0).iter().map(|c| c.norm_sqr()).collect();
    let freqs: Vec<f32> = (0..=(n_fft / 2))
        .map(|k| (k as f64 * sr as f64 / n_fft as f64) as f32)
        .collect();
    let _env_s = extract_cepstral_formant_envelope(&p_sharp, sr, n_fft).unwrap();
    let rd0_sharp = estimate_baseline_rd(&p_sharp, &freqs, 220.0, sr, n_fft);
    let rd0_lax = estimate_baseline_rd(&p_lax, &freqs, 220.0, sr, n_fft);
    assert!(
        rd0_sharp < rd0_lax,
        "expected rd0_sharp ({rd0_sharp}) < rd0_lax ({rd0_lax})"
    );

    // Middle-ear loudness weight sanity
    let w = middle_ear_loudness_weight(&[100.0, 300.0, 1200.0, 3100.0, 10000.0]);
    assert!(w.iter().all(|v| v.is_finite() && *v > 0.0));
}

fn band_energy_ratio(wave: &[f32], lo_num: f64, hi_num: f64, lo_den: f64, hi_den: f64) -> f64 {
    let win = periodic_hann_window(wave.len());
    let spec = stft_1d(
        wave,
        wave.len(),
        wave.len(),
        &win,
        false,
        PadMode::Constant(0.0),
    )
    .unwrap();
    let df = 44100.0 / wave.len() as f64;
    let mut e_num = 0.0_f64;
    let mut e_den = 0.0_f64;
    for (k, c) in spec.column(0).iter().enumerate() {
        let f = k as f64 * df;
        let p = c.norm_sqr() as f64;
        if f >= lo_num && f < hi_num {
            e_num += p;
        }
        if f >= lo_den && f < hi_den {
            e_den += p;
        }
    }
    e_num / (e_den + 1e-12)
}

#[test]
fn test_tension_ht_ho_hr_monotonicity() {
    let fix = load_fixtures();
    let harm_sig = fix["harm_sig"].as_f32();

    // 1. Ht tilt monotonicity across b in [+2.0, +1.0, 0.0, -1.0, -2.0]
    let mut ht_ratios = Vec::new();
    for &b in &[2.0, 1.0, 0.0, -1.0, -2.0] {
        let out = pre_emphasis_base_tension(harm_sig, b, 0.0, 0.0).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
        ht_ratios.push(band_energy_ratio(&out, 1500.0, 6000.0, 150.0, 800.0));
    }
    for w in ht_ratios.windows(2) {
        assert!(w[1] > w[0], "expected Ht ratio increase: {:?}", ht_ratios);
    }

    // 2. Ho openness monotonicity across [-80, 0, +80]
    let mut ho_ratios = Vec::new();
    for &ho in &[-80.0, 0.0, 80.0] {
        let out = pre_emphasis_base_tension(harm_sig, 0.0, ho, 0.0).unwrap();
        ho_ratios.push(band_energy_ratio(&out, 1500.0, 6000.0, 150.0, 500.0));
    }
    assert!(ho_ratios[0] < ho_ratios[1] && ho_ratios[1] < ho_ratios[2]);

    // 3. Hr 3.2 kHz singer's formant monotonicity across [-80, 0, +80]
    let mut hr_ratios = Vec::new();
    for &hr in &[-80.0, 0.0, 80.0] {
        let out = pre_emphasis_base_tension(harm_sig, 0.0, 0.0, hr).unwrap();
        hr_ratios.push(band_energy_ratio(&out, 2700.0, 3700.0, 150.0, 500.0));
    }
    assert!(hr_ratios[0] < hr_ratios[1] && hr_ratios[1] < hr_ratios[2]);

    // 4. Bypass when b=0, ho=0, hr=0 in couple_aperiodic_to_tension
    let bypass_ap =
        couple_aperiodic_to_tension(harm_sig, harm_sig, harm_sig, 0.0, 0.0, 0.0).unwrap();
    assert_eq!(bypass_ap, harm_sig);
}

#[test]
fn test_edge_cases_batch_akima_hnsep() {
    let mel = MelSpectrogram::with_defaults();
    let a: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44100.0).sin())
        .collect();
    let b: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).cos())
        .collect();
    let mut flat = Vec::with_capacity(8192);
    flat.extend_from_slice(&a);
    flat.extend_from_slice(&b);
    let batch = Array2::from_shape_vec((2, 4096), flat).unwrap();
    let batch_mel = mel.compute_batch(batch.view(), 0.0, 1.0).unwrap();
    assert_eq!(batch_mel.dim().0, 2);
    assert_eq!(batch_mel.dim().1, 128);

    // Non-contiguous (transposed) batch view
    let transposed_src = batch.t().to_owned();
    let non_contig_view = transposed_src.t();
    let batch_mel_nc = mel.compute_batch(non_contig_view, 0.0, 1.0).unwrap();
    assert!(
        max_abs_diff_f32(
            batch_mel.as_slice().unwrap(),
            batch_mel_nc.as_slice().unwrap()
        ) < 1e-6
    );

    // Akima extrapolate=true and error cases
    assert!(Akima1DInterpolator::new(&[0.0], &[1.0]).is_err());
    assert!(Akima1DInterpolator::new(&[0.0, 0.0], &[1.0, 2.0]).is_err());
    assert!(Akima1DInterpolator::new(&[0.0, 1.0], &[1.0]).is_err());
    let ext = Akima1DInterpolator::with_extrapolate(&[0.0, 1.0], &[2.0, 4.0], true).unwrap();
    assert!((ext.evaluate(2.0) - 6.0).abs() < 1e-12);

    // HN-SEP empty input + apply_bounded_mask_tensor
    let hnsep = HnsepStft::default();
    let empty_prep = hnsep.prepare_input(&[]).unwrap();
    let empty_mask = Array4::<f32>::zeros(empty_prep.tensor.dim());
    let empty_out = hnsep
        .reconstruct_from_mask(&empty_prep, empty_mask.view())
        .unwrap();
    assert!(empty_out.is_empty());

    let mut mask_t = Array4::<f32>::from_elem((1, 2, 4, 4), 10.0);
    apply_bounded_mask_tensor(&mut mask_t, 1e-8).unwrap();
    for f in 0..4 {
        for t in 0..4 {
            let r = mask_t[[0, 0, f, t]];
            let i = mask_t[[0, 1, f, t]];
            assert!((r * r + i * i).sqrt() <= 1.0 + 1e-6);
        }
    }

    // 1-frame mel resampling fallback
    let one_frame = Array2::<f32>::from_elem((128, 1), -2.5);
    let r1 =
        resample_mel_and_pitch(one_frame.view(), &[0], &ResampleTimeParams::default()).unwrap();
    assert_eq!(r1.mel_render.nrows(), 128);
    assert!(r1.mel_render.ncols() >= 1);
}
