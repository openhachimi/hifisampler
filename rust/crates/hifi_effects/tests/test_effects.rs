use std::f64::consts::PI;

use approx::assert_abs_diff_eq;
use hifi_effects::{
    apply_amplitude_modulation, apply_hg_growl, apply_lf_vocal_fry, apply_lf_vocal_fry_in_place,
    apply_scale_inverse, apply_volume_and_peak_limit, butter_sos, growl, growl_with_params,
    prepare_vocal_fry_excitation, slaney_mel_filterbank, sosfilt, sosfilt_zi, sosfiltfilt,
    Akima1DInterpolator, ButterBand, FryExcitationResult, VocalFryInput,
};
use ndarray::Array2;

const SR: u32 = 44100;
const N_FFT: usize = 2048;
const HOP_SIZE: usize = 512;
const N_MELS: usize = 128;
const MEL_FMIN: f64 = 40.0;
const MEL_FMAX: f64 = 16000.0;

fn make_fry_fixture(n_frames: usize) -> (Option<FryExcitationResult>, Vec<f64>, f64) {
    let thop = (HOP_SIZE as f64) / (SR as f64);
    let t_mel: Vec<f64> = (0..n_frames).map(|i| (i as f64) * thop).collect();
    let f0_flat = vec![329.63_f32; n_frames];
    let mut mel = Array2::<f32>::from_elem((N_MELS, n_frames), -3.0);
    for m in 0..35 {
        for t in 0..n_frames {
            mel[[m, t]] = 0.8;
        }
    }

    let mut pb = vec![0.0_f64; 20];
    for k in 0..16 {
        let u = (k as f64) / 15.0;
        let s = -1900.0 * (3.0 * u * u - 2.0 * u * u * u);
        pb.push((s as i32) as f64);
    }
    pb.extend(std::iter::repeat_n(-1900.0_f64, 90));
    pb.push(0.0);

    let input = VocalFryInput {
        pitchbend: &pb,
        base_pitch_midi: 64.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: (n_frames as f64) * thop,
        t_mel: &t_mel,
        f0_render: &f0_flat,
        mel_render: mel.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: Some(0.03),
    };
    let ctx = prepare_vocal_fry_excitation(&input);
    (ctx, t_mel, thop)
}

#[test]
fn test_butter_sos_and_zi_match_scipy() {
    // 2nd order highpass 155 Hz @ 44100
    let sos_155 = butter_sos(2, 155.0 / 22050.0, ButterBand::Highpass);
    let expected_155 = [
        0.9845057087473873,
        -1.9690114174947746,
        0.9845057087473873,
        1.0,
        -1.9687713300228558,
        0.9692515049666944,
    ];
    for i in 0..6 {
        assert_abs_diff_eq!(sos_155[0][i], expected_155[i], epsilon = 1e-13);
    }
    let zi_155 = sosfilt_zi(&sos_155);
    assert_abs_diff_eq!(zi_155[0][0], -0.9845057087473873, epsilon = 1e-13);
    assert_abs_diff_eq!(zi_155[0][1], 0.9845057087473873, epsilon = 1e-13);

    // 4th order highpass 400 Hz rounded @ 44100
    let wn4 = ((400.0 / 22050.0_f64) * 1e6).round() / 1e6;
    let sos4 = butter_sos(4, wn4, ButterBand::Highpass);
    let expected_4 = [
        [
            0.9282307106725765,
            -1.856461421345153,
            0.9282307106725765,
            1.0,
            -1.8969272203836025,
            0.9000120486909986,
        ],
        [1.0, -2.0, 1.0, 1.0, -1.9541562351842994, 0.9573341306810765],
    ];
    for s in 0..2 {
        for i in 0..6 {
            assert_abs_diff_eq!(sos4[s][i], expected_4[s][i], epsilon = 1e-13);
        }
    }
    let zi4 = sosfilt_zi(&sos4);
    assert_abs_diff_eq!(zi4[0][0], -0.9282307106725765, epsilon = 1e-13);
    assert_abs_diff_eq!(zi4[1][0], 0.0, epsilon = 1e-13);
    assert_abs_diff_eq!(zi4[1][1], 0.0, epsilon = 1e-13);

    // 4th order lowpass 0.25
    let sos_lp = butter_sos(4, 0.25, ButterBand::Lowpass);
    let expected_lp = [
        [
            0.010209480791203138,
            0.020418961582406275,
            0.010209480791203138,
            1.0,
            -0.8553979327751704,
            0.20971535775655478,
        ],
        [1.0, 2.0, 1.0, 1.0, -1.1130298541633479, 0.5740619150839545],
    ];
    for s in 0..2 {
        for i in 0..6 {
            assert_abs_diff_eq!(sos_lp[s][i], expected_lp[s][i], epsilon = 1e-13);
        }
    }
    let zi_lp = sosfilt_zi(&sos_lp);
    assert_abs_diff_eq!(zi_lp[0][0], 0.1050485344389486, epsilon = 1e-13);
    assert_abs_diff_eq!(zi_lp[1][0], 0.8847419847698488, epsilon = 1e-13);

    // 3rd order (odd order) highpass 0.30
    let sos3 = butter_sos(3, 0.30, ButterBand::Highpass);
    let expected_3 = [
        [
            0.3744526925901595,
            -0.748905385180319,
            0.3744526925901595,
            1.0,
            -0.32491969623290634,
            0.0,
        ],
        [1.0, -1.0, 0.0, 1.0, -0.8369977874388261, 0.4239856889474125],
    ];
    for s in 0..2 {
        for i in 0..6 {
            assert_abs_diff_eq!(sos3[s][i], expected_3[s][i], epsilon = 1e-13);
        }
    }
}

#[test]
fn test_sosfiltfilt_matches_scipy_and_zero_phase_stability() {
    // Exact parity vs scipy.signal.sosfiltfilt
    let x: Vec<f64> = (0..120)
        .map(|i| {
            let t1 = 12.0 * (i as f64) / 119.0;
            let t2 = 40.0 * (i as f64) / 119.0;
            t1.sin() + 0.3 * t2.cos()
        })
        .collect();
    let sos2 = butter_sos(2, 155.0 / 22050.0, ButterBand::Highpass);
    let y = sosfiltfilt(&sos2, &x).expect("len 120 > edge 9");
    let head_exp = [
        0.09305548541185593,
        0.1778439820498291,
        0.22998671290493414,
        0.25390004860455506,
        0.2569614937359048,
        0.2485832023009636,
        0.2390608899295337,
        0.23832797144916007,
    ];
    let tail_exp = [
        -0.17369240260555785,
        -0.13791903706832043,
        -0.12490382604607203,
        -0.13018459088229495,
        -0.14633140215438917,
        -0.163832772329909,
        -0.17222348809585414,
        -0.16132720772903958,
    ];
    for i in 0..8 {
        assert_abs_diff_eq!(y[i], head_exp[i], epsilon = 1e-11);
        assert_abs_diff_eq!(y[120 - 8 + i], tail_exp[i], epsilon = 1e-11);
    }

    // Zero-phase symmetry check on centered impulse
    let n = 2001;
    let center = n / 2;
    let mut impulse = vec![0.0_f64; n];
    impulse[center] = 1.0;
    let sos4 = butter_sos(4, 400.0 / 22050.0, ButterBand::Highpass);
    let resp = sosfiltfilt(&sos4, &impulse).unwrap();
    for k in 1..400 {
        assert_abs_diff_eq!(resp[center - k], resp[center + k], epsilon = 1e-12);
    }
    // Peak remains strictly at center (zero phase lag)
    let peak_idx = resp
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .unwrap()
        .0;
    assert_eq!(peak_idx, center);

    // Extreme low cutoff long-duration stability (20 Hz @ 44100 Hz, 2 seconds)
    let sos_low = butter_sos(2, 20.0 / 22050.0, ButterBand::Highpass);
    let long_dc_step = vec![1.0_f64; 88200];
    let filtered_dc = sosfiltfilt(&sos_low, &long_dc_step).unwrap();
    assert!(filtered_dc.iter().all(|v| v.is_finite() && v.abs() < 1e-10));

    // Pole stability for causal sosfilt
    let causal_resp = sosfilt(&sos4, &impulse, None);
    assert!(causal_resp.iter().all(|v| v.is_finite()));
    assert!(causal_resp[n - 1].abs() < 1e-8);
}

#[test]
fn test_akima_matches_scipy() {
    let ax = [0.0, 0.05, 0.12, 0.20, 0.35, 0.50];
    let ay = [60.0, 58.5, 52.0, 48.0, 47.0, 64.0];
    let aq = [0.0, 0.025, 0.08, 0.16, 0.28, 0.42, 0.50];
    let expected = [
        60.0,
        59.68001930501931,
        55.77753169242141,
        49.71107701990402,
        46.27647340573656,
        52.132571428571424,
        64.0,
    ];
    let akima = Akima1DInterpolator::new(&ax, &ay);
    let got = akima.eval_clamped(&aq);
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(g, e, epsilon = 1e-12);
    }

    // 2-point linear edge case & flat curve safety
    let lin = Akima1DInterpolator::new(&[0.0, 2.0], &[10.0, 20.0]);
    assert_abs_diff_eq!(lin.eval(0.5), 12.5, epsilon = 1e-12);
    let flat = Akima1DInterpolator::new(&[0.0, 1.0, 2.0, 3.0], &[5.0, 5.0, 5.0, 5.0]);
    assert_abs_diff_eq!(flat.eval(1.7), 5.0, epsilon = 1e-12);
}

#[test]
fn test_growl_numerical_closeness_and_boundaries() {
    // 1. Exact numerical closeness vs util/growl.py for n=101 and n=4410
    for &(n, exp_head, exp_tail, exp_rms, exp_sum) in &[
        (
            101_usize,
            [
                0.0_f32, 0.09252908, 0.17873265, 0.25296727, 0.3108754, 0.34984198, 0.36925134,
                0.37051576,
            ],
            [
                -0.06188175_f32,
                -0.008419216,
                0.036200255,
                0.06793204,
                0.08445677,
                0.08540924,
                0.07237804,
                0.048676476,
            ],
            0.32129287891107294_f64,
            23.195788510143757_f64,
        ),
        (
            4410_usize,
            [
                0.0_f32, 0.09535247, 0.18264933, 0.2560727, 0.31140578, 0.34647924, 0.36136657,
                0.35830823,
            ],
            [
                -0.28495672_f32,
                -0.31269354,
                -0.3467012,
                -0.3797022,
                -0.4040294,
                -0.4125347,
                -0.3994478,
                -0.36108738,
            ],
            0.328338417809238_f64,
            -1.2177807954140007_f64,
        ),
    ] {
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = (i as f64) / (SR as f64);
                (0.35 * (2.0 * PI * 220.0 * t).sin()
                    + 0.25 * (2.0 * PI * 880.0 * t).sin()
                    + 0.15 * (2.0 * PI * 2400.0 * t).sin()) as f32
            })
            .collect();
        let got = growl_with_params(&sig, SR, 80.0, 0.65, 400.0);
        assert_eq!(got.len(), n);
        assert!(got.iter().all(|v| v.is_finite()));
        for i in 0..8 {
            assert_abs_diff_eq!(got[i], exp_head[i], epsilon = 1e-5);
            assert_abs_diff_eq!(got[n - 8 + i], exp_tail[i], epsilon = 1e-5);
        }
        let rms = (got.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64).sqrt();
        let sum: f64 = got.iter().map(|&v| v as f64).sum();
        assert_abs_diff_eq!(rms, exp_rms, epsilon = 1e-5);
        assert_abs_diff_eq!(sum, exp_sum, epsilon = 1e-4);
    }

    // 2. Boundaries & bypasses
    assert!(growl(&[], SR, 0.5).is_empty());
    for n in [0_usize, 1, 2, 17, 100] {
        let zeros = vec![0.0_f32; n];
        let out_z = growl(&zeros, SR, 0.8);
        assert!(out_z.iter().all(|&v| v == 0.0));

        let ones = vec![0.25_f32; n];
        assert_eq!(growl_with_params(&ones, SR, 0.0, 0.5, 400.0), ones);
        assert_eq!(growl(&ones, SR, 0.0), ones);
        assert_eq!(apply_hg_growl(&ones, SR, 0), ones);
        assert_eq!(apply_hg_growl(&ones, SR, -10), ones);
        let valid = growl_with_params(&ones, SR, 120.0, 1.0, 300.0);
        assert!(valid.iter().all(|v| v.is_finite()));
    }

    // 3. RMS energy preservation within +/- 1.0 dB and high-band modulation
    let n_half = (SR / 2) as usize;
    let audio_880: Vec<f32> = (0..n_half)
        .map(|i| (0.3 * (2.0 * PI * 880.0 * (i as f64) / (SR as f64)).sin()) as f32)
        .collect();
    let mod_880 = growl(&audio_880, SR, 0.5);
    let rms_in =
        (audio_880.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n_half as f64).sqrt();
    let rms_out = (mod_880.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n_half as f64).sqrt();
    let db_diff = 20.0 * (rms_out / (rms_in + 1e-12)).log10();
    assert!(db_diff.abs() < 1.0, "db_diff={db_diff}");

    let max_diff: f32 = audio_880
        .iter()
        .zip(mod_880.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max_diff > 0.05);

    // Extreme HG100 boundedness
    let hg100 = apply_hg_growl(&audio_880, SR, 100);
    assert!(hg100.iter().all(|v| v.is_finite() && v.abs() <= 2.0));
}

#[test]
fn test_vocal_fry_o1_early_exit() {
    let thop = (HOP_SIZE as f64) / (SR as f64);
    let n_frames = 60;
    let t_mel: Vec<f64> = (0..n_frames).map(|i| (i as f64) * thop).collect();
    let f0_flat = vec![329.63_f32; n_frames];
    let mut mel = Array2::<f32>::from_elem((N_MELS, n_frames), -3.0);
    for m in 0..35 {
        for t in 0..n_frames {
            mel[[m, t]] = 0.8;
        }
    }

    // 1. Flat, <= 1-octave (-1200c), boundary (-1250c), tiny pitchbend, empty mel -> None
    for pb in [
        vec![-1900.0],
        vec![0.0; 40],
        vec![-600.0; 40],
        vec![-1200.0; 40],
        vec![-1250.0; 40],
    ] {
        let res = prepare_vocal_fry_excitation(&VocalFryInput {
            pitchbend: &pb,
            base_pitch_midi: 64.0,
            t_flag_val: 0.0,
            tempo: 120.0,
            new_start: 0.0,
            new_end: (n_frames as f64) * thop,
            t_mel: &t_mel,
            f0_render: &f0_flat,
            mel_render: mel.view(),
            sr: SR,
            n_fft: N_FFT,
            mel_fmin: MEL_FMIN,
            mel_fmax: MEL_FMAX,
            new_con: None,
        });
        assert!(res.is_none());
    }

    let empty_mel = Array2::<f32>::zeros((N_MELS, 0));
    assert!(prepare_vocal_fry_excitation(&VocalFryInput {
        pitchbend: &[-1900.0; 10],
        base_pitch_midi: 64.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: 0.1,
        t_mel: &[],
        f0_render: &[],
        mel_render: empty_mel.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: None,
    })
    .is_none());

    // 2. apply_lf_vocal_fry O(1) early return on None, <32 samples, and all-zero fry_env
    let render: Vec<f32> = (0..4096)
        .map(|i| (0.25 * (2.0 * PI * 440.0 * (i as f64) / (SR as f64)).sin()) as f32)
        .collect();
    assert_eq!(apply_lf_vocal_fry(&render, None, SR), render);

    let zero_ctx = FryExcitationResult {
        mel_mod: mel.clone(),
        f0_mod: f0_flat.clone(),
        fry_env: vec![0.0_f64; n_frames],
        f0_eff: vec![329.63_f64; n_frames],
        t_mel: t_mel.clone(),
        new_start: 0.0,
        new_end: (n_frames as f64) * thop,
        seed: 1,
    };
    assert_eq!(apply_lf_vocal_fry(&render, Some(&zero_ctx), SR), render);

    let mut render_ip = render.clone();
    apply_lf_vocal_fry_in_place(&mut render_ip, Some(&zero_ctx), SR);
    assert_eq!(render_ip, render);

    let short = vec![0.1_f32; 31];
    let (active_ctx, _, _) = make_fry_fixture(80);
    assert_eq!(apply_lf_vocal_fry(&short, active_ctx.as_ref(), SR), short);
}

#[test]
fn test_vocal_fry_melodic_leap_immunity_and_fricative_guard() {
    let thop = (HOP_SIZE as f64) / (SR as f64);
    let n_frames = 60;
    let t_mel: Vec<f64> = (0..n_frames).map(|i| (i as f64) * thop).collect();
    let f0_flat = vec![329.63_f32; n_frames];
    let mut mel_voiced = Array2::<f32>::from_elem((N_MELS, n_frames), -3.0);
    for m in 0..35 {
        for t in 0..n_frames {
            mel_voiced[[m, t]] = 0.8;
        }
    }

    // (a) Unvoiced fricative protection
    let mut mel_fric = Array2::<f32>::from_elem((N_MELS, n_frames), -6.0);
    for m in 85..N_MELS {
        for t in 0..n_frames {
            mel_fric[[m, t]] = 1.2;
        }
    }
    let mut deep_pb = vec![-1900.0_f64; 50];
    deep_pb.push(0.0);
    assert!(prepare_vocal_fry_excitation(&VocalFryInput {
        pitchbend: &deep_pb,
        base_pitch_midi: 64.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: (n_frames as f64) * thop,
        t_mel: &t_mel,
        f0_render: &f0_flat,
        mel_render: mel_fric.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: None,
    })
    .is_none());

    // (b) VCV upward leap ('nian C3 -> xiang F4')
    let mut scurve_up = Vec::with_capacity(16);
    let mut scurve_dn = Vec::with_capacity(16);
    for k in 0..16 {
        let u = (k as f64) / 15.0;
        let s = 3.0 * u * u - 2.0 * u * u * u;
        scurve_up.push(((-1700.0 * (1.0 - s)) as i32) as f64);
        scurve_dn.push(((-1700.0 * s) as i32) as f64);
    }

    let mut pb_up_vcv = vec![-1700.0_f64; 35];
    pb_up_vcv.extend_from_slice(&scurve_up);
    pb_up_vcv.extend(std::iter::repeat_n(0.0_f64, 85));
    assert!(prepare_vocal_fry_excitation(&VocalFryInput {
        pitchbend: &pb_up_vcv,
        base_pitch_midi: 65.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: (n_frames as f64) * thop,
        t_mel: &t_mel,
        f0_render: &f0_flat,
        mel_render: mel_voiced.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: Some(0.20),
    })
    .is_none());

    // (c) Pure-vowel upward leap
    let mut pb_up_vowel = vec![-1700.0_f64; 16];
    pb_up_vowel.extend_from_slice(&scurve_up);
    pb_up_vowel.extend(std::iter::repeat_n(0.0_f64, 90));
    assert!(prepare_vocal_fry_excitation(&VocalFryInput {
        pitchbend: &pb_up_vowel,
        base_pitch_midi: 65.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: (n_frames as f64) * thop,
        t_mel: &t_mel,
        f0_render: &f0_flat,
        mel_render: mel_voiced.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: Some(0.0),
    })
    .is_none());

    // (d) Downward tail portamento
    let n_total_ticks = (((n_frames as f64) * thop) / (60.0 / (120.0 * 96.0))).round() as usize;
    let mut pb_dn_tail = vec![0.0_f64; n_total_ticks.saturating_sub(22).max(10)];
    pb_dn_tail.extend_from_slice(&scurve_dn);
    pb_dn_tail.extend(std::iter::repeat_n(-1700.0_f64, 5));
    pb_dn_tail.push(0.0);
    assert!(prepare_vocal_fry_excitation(&VocalFryInput {
        pitchbend: &pb_dn_tail,
        base_pitch_midi: 65.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: (n_frames as f64) * thop,
        t_mel: &t_mel,
        f0_render: &f0_flat,
        mel_render: mel_voiced.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: Some(0.06),
    })
    .is_none());
}

#[test]
fn test_vocal_fry_python_parity_and_shaping() {
    // 1. Slaney mel filterbank parity
    let fb = slaney_mel_filterbank(SR, N_FFT, N_MELS, MEL_FMIN, MEL_FMAX);
    assert_abs_diff_eq!(fb.row(0).sum(), 0.04762832, epsilon = 1e-6);
    assert_abs_diff_eq!(fb.row(50).sum(), 0.046_012_72_f32, epsilon = 1e-6);
    assert_abs_diff_eq!(fb.row(127).sum(), 0.046422288, epsilon = 1e-6);

    // 2. prepare_vocal_fry_excitation parity on w_fry and mel_mod
    let (ctx_opt, t_mel, thop) = make_fry_fixture(80);
    let ctx = ctx_opt.expect("intentional fry basin must trigger");
    assert_eq!(ctx.seed, 2106217256);

    let expected_w_fry_slice = [
        0.0,
        0.003729623232274191,
        0.03482672246659081,
        0.132892778014964,
        0.3128184239458679,
        0.5424289617950817,
        0.7604449572576797,
        0.9123789017093065,
        0.9824532557784027,
        0.9991130946969147,
        1.0,
        1.0,
        1.0,
        1.0,
    ];
    for (idx, &exp) in expected_w_fry_slice.iter().enumerate() {
        assert_abs_diff_eq!(ctx.fry_env[8 + idx], exp, epsilon = 1e-6);
    }

    let expected_mel_head = [
        0.52992666_f32,
        0.5690155,
        0.6088106,
        0.64887995,
        0.6847141,
        0.7145931,
        0.7442968,
        0.77092236,
    ];
    let expected_mel_mid = [
        -2.6990247_f32,
        -2.6956663,
        -2.6928358,
        -2.6901684,
        -2.68766,
        -2.6853297,
        -2.6832516,
        -2.6814258,
    ];
    for i in 0..8 {
        assert_abs_diff_eq!(ctx.mel_mod[[i, 30]], expected_mel_head[i], epsilon = 1e-5);
        assert_abs_diff_eq!(
            ctx.mel_mod[[40 + i, 30]],
            expected_mel_mid[i],
            epsilon = 1e-5
        );
    }

    let f0_min = ctx.f0_mod.iter().copied().fold(f32::INFINITY, f32::min);
    let f0_max = ctx.f0_mod.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!((20.0..=52.0).contains(&f0_min), "f0_min={f0_min}");
    assert!(f0_max <= 335.0, "f0_max={f0_max}");

    // 3. apply_lf_vocal_fry produces strong creaky bubble envelope modulation
    let n_samples = ((80.0 * thop) * (SR as f64)).round() as usize;
    let mut ph_inst = 0.0_f64;
    let mut carrier = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = (i as f64) * (80.0 * thop) / (n_samples as f64);
        let f0_i = hifi_effects::interp_1d_clamped(&t_mel, &ctx.f0_eff, t);
        ph_inst += f0_i / (SR as f64);
        let c = 0.22 * (2.0 * PI * 680.0 * t).cos() * (0.55 + 0.45 * (2.0 * PI * ph_inst).cos())
            + 0.16 * (2.0 * PI * 1650.0 * t).cos() * (0.55 + 0.45 * (2.0 * PI * ph_inst).cos());
        carrier.push(c as f32);
    }

    let shaped = apply_lf_vocal_fry(&carrier, Some(&ctx), SR);
    assert_eq!(shaped.len(), n_samples);
    assert!(shaped.iter().all(|v| v.is_finite()));

    let tail_s = (0.35 * SR as f64) as usize;
    let tail_e = n_samples - (0.03 * SR as f64) as usize;
    let win = (0.0025 * SR as f64) as usize;
    let env_stats = |buf: &[f32]| -> f64 {
        let mut envs = Vec::new();
        for start in tail_s..=(tail_e - win) {
            let e: f64 = buf[start..(start + win)]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / (win as f64);
            envs.push((e + 1e-12).sqrt());
        }
        envs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95 = envs[(envs.len() - 1) * 95 / 100];
        let p15 = envs[(envs.len() - 1) * 15 / 100];
        p95 / (p15 + 1e-9)
    };
    let mod_orig = env_stats(&carrier);
    let mod_fry = env_stats(&shaped);
    assert!(
        mod_fry > mod_orig * 2.5 && mod_fry > 4.0,
        "mod_orig={mod_orig}, mod_fry={mod_fry}"
    );

    // 4. Chunk boundaries & silence safety
    for &n in &[0_usize, 1, 31, 32, 64, 65, 1024, 1025, 8192, 8193] {
        let zeros = vec![0.0_f32; n];
        let out_z = apply_lf_vocal_fry(&zeros, Some(&ctx), SR);
        assert!(out_z.iter().all(|&v| v == 0.0));
        let sig: Vec<f32> = (0..n).map(|i| 0.2 * ((i as f32) * 0.1).sin()).collect();
        let out_s = apply_lf_vocal_fry(&sig, Some(&ctx), SR);
        assert_eq!(out_s.len(), n);
        assert!(out_s.iter().all(|v| v.is_finite()));
    }

    // 5. Flat deep pitchbend with trailing 0 (exercises Akima constant-pitch branch)
    let mut pb_flat = vec![-1900.0_f64; 80];
    pb_flat.push(0.0);
    let mut mel_60 = Array2::<f32>::from_elem((N_MELS, 60), -3.0);
    for m in 0..35 {
        for t in 0..60 {
            mel_60[[m, t]] = 0.8;
        }
    }
    let res_flat = prepare_vocal_fry_excitation(&VocalFryInput {
        pitchbend: &pb_flat,
        base_pitch_midi: 64.0,
        t_flag_val: 0.0,
        tempo: 120.0,
        new_start: 0.0,
        new_end: 60.0 * thop,
        t_mel: &t_mel[..60],
        f0_render: &vec![329.63_f32; 60],
        mel_render: mel_60.view(),
        sr: SR,
        n_fft: N_FFT,
        mel_fmin: MEL_FMIN,
        mel_fmax: MEL_FMAX,
        new_con: None,
    })
    .expect("flat deep basin triggers");
    assert!(res_flat.f0_mod.iter().all(|v| v.is_finite()));
}

#[test]
fn test_amplitude_modulation_and_volume_scaling() {
    let thop = (HOP_SIZE as f64) / (SR as f64);
    let n_frames = 80;
    let t_mel: Vec<f64> = (0..n_frames).map(|i| (i as f64) * thop).collect();
    let pitch_render: Vec<f64> = (0..n_frames)
        .map(|i| 64.0 + 3.0 * (2.0 * PI * (i as f64) / 20.0).sin())
        .collect();

    let n_samples = 4096_usize;
    let new_end = (n_frames as f64) * thop;
    let sig_in: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = (i as f64) * new_end / (n_samples as f64);
            (0.3 * (2.0 * PI * 440.0 * t).sin()) as f32
        })
        .collect();

    // A=0 is an exact no-op
    let mut noop = sig_in.clone();
    apply_amplitude_modulation(&mut noop, 0, &pitch_render, &t_mel, thop, 0.0, new_end);
    assert_eq!(noop, sig_in);

    // A=50 exact numerical match vs backend/resampler.py
    let mut am50 = sig_in.clone();
    apply_amplitude_modulation(&mut am50, 50, &pitch_render, &t_mel, thop, 0.0, new_end);
    let expected_head = [
        0.0_f32,
        0.33441547,
        0.5413275,
        0.5422504,
        0.33708277,
        0.0040511275,
        -0.3301165,
        -0.53841376,
    ];
    let expected_mid = [
        -0.5561434_f32,
        -0.402655,
        -0.09575726,
        0.24795504,
        0.49773517,
        0.55841994,
        0.40668565,
        0.10001125,
    ];
    for i in 0..8 {
        assert_abs_diff_eq!(am50[i], expected_head[i], epsilon = 1e-5);
        assert_abs_diff_eq!(am50[1000 + i], expected_mid[i], epsilon = 1e-5);
    }
    let am_rms = (am50.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n_samples as f64).sqrt();
    assert_abs_diff_eq!(am_rms, 0.2551886369178049, epsilon = 1e-5);

    // Clamping A to [-100, 100]
    let mut am100 = sig_in.clone();
    let mut am250 = sig_in.clone();
    apply_amplitude_modulation(&mut am100, 100, &pitch_render, &t_mel, thop, 0.0, new_end);
    apply_amplitude_modulation(&mut am250, 250, &pitch_render, &t_mel, thop, 0.0, new_end);
    assert_eq!(am100, am250);

    // Unscaling & volume peak limit
    let mut buf = vec![0.25_f32, -0.5, 0.1];
    apply_scale_inverse(&mut buf, 0.5);
    assert_abs_diff_eq!(buf[0], 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(buf[1], -1.0, epsilon = 1e-6);

    let mut vol_buf = vec![0.6_f32, -0.8, 0.3];
    apply_volume_and_peak_limit(&mut vol_buf, 200.0, 1.0); // scaled_peak = 1.6 > 1.0 -> peak becomes 1.0
    assert_abs_diff_eq!(vol_buf[1], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(vol_buf[0], 0.75, epsilon = 1e-6);

    let mut vol_half = vec![0.6_f32, -0.4];
    apply_volume_and_peak_limit(&mut vol_half, 50.0, 1.0);
    assert_abs_diff_eq!(vol_half[0], 0.3, epsilon = 1e-6);
    assert_abs_diff_eq!(vol_half[1], -0.2, epsilon = 1e-6);

    let mut vol_zero = vec![0.6_f32, -0.4];
    apply_volume_and_peak_limit(&mut vol_zero, 0.0, 1.0);
    assert_eq!(vol_zero, vec![0.0_f32, 0.0]);
}
