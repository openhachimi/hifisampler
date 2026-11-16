use approx::assert_relative_eq;
use hifi_core::{
    is_null_device, read_wav, read_wav_raw, resolve_audio_path, write_wav, AudioBuffer, AudioError,
    DEFAULT_SAMPLE_RATE,
};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use tempfile::tempdir;

#[test]
fn test_audio_buffer_duration() {
    let samples = vec![0.0f32; 44100 * 2];
    let buf = AudioBuffer::new(samples, DEFAULT_SAMPLE_RATE, 1);
    assert_relative_eq!(buf.duration_seconds(), 2.0, epsilon = 1e-6);

    let empty = AudioBuffer::new(vec![], DEFAULT_SAMPLE_RATE, 1);
    assert_eq!(empty.duration_seconds(), 0.0);
}

#[test]
fn test_is_null_device() {
    assert!(is_null_device(Path::new("nul")));
    assert!(is_null_device(Path::new("NUL")));
    assert!(is_null_device(Path::new("/dev/null")));
    assert!(is_null_device(Path::new("  /DEV/NULL  ")));
    assert!(!is_null_device(Path::new("null")));
    assert!(!is_null_device(Path::new("output.wav")));
}

#[test]
fn test_null_device_write_bypass() {
    let buf = AudioBuffer::new(vec![0.1, 0.2, 0.3], DEFAULT_SAMPLE_RATE, 1);
    assert!(write_wav(Path::new("nul"), &buf).is_ok());
    assert!(write_wav(Path::new("/dev/null"), &buf).is_ok());
}

#[test]
fn test_wav_read_write_roundtrip() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("roundtrip.wav");

    // Synthesize 440 Hz sine wave for 0.05s (2205 samples)
    let count = 2205;
    let mut original = Vec::with_capacity(count);
    for i in 0..count {
        let t = i as f32 / DEFAULT_SAMPLE_RATE as f32;
        original.push((2.0 * PI * 440.0 * t).sin() * 0.75);
    }

    let input_buffer = AudioBuffer::new(original.clone(), DEFAULT_SAMPLE_RATE, 1);
    write_wav(&wav_path, &input_buffer).expect("Failed to write WAV");

    assert!(wav_path.exists());

    let output_buffer = read_wav(&wav_path).expect("Failed to read WAV");
    assert_eq!(output_buffer.sample_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(output_buffer.channels, 1);
    assert_eq!(output_buffer.samples.len(), count);

    // Maximum quantization error for 16-bit PCM is 1.0 / 32768.0 ≈ 3.05e-5
    let max_quant_error = 1.0f32 / 32768.0f32 + 1e-6f32;
    for (orig, loaded) in original.iter().zip(output_buffer.samples.iter()) {
        assert!(
            (orig - loaded).abs() <= max_quant_error,
            "Diff {} exceeded max error {}",
            (orig - loaded).abs(),
            max_quant_error
        );
    }
}

#[test]
fn test_stereo_downmix_to_mono() {
    let dir = tempdir().unwrap();
    let stereo_wav = dir.path().join("stereo.wav");

    let spec = WavSpec {
        channels: 2,
        sample_rate: DEFAULT_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    {
        let file = File::create(&stereo_wav).unwrap();
        let mut writer = WavWriter::new(BufWriter::new(file), spec).unwrap();
        // L = 0.2 * 32768 = 6553.6 -> 6553
        // R = 0.6 * 32768 = 19660.8 -> 19660
        writer.write_sample(6553i16).unwrap();
        writer.write_sample(19660i16).unwrap();
        writer.finalize().unwrap();
    }

    let audio = read_wav(&stereo_wav).expect("Failed to read stereo wav");
    assert_eq!(audio.channels, 1);
    assert_eq!(audio.samples.len(), 1);

    let expected = ((6553.0 / 32768.0) + (19660.0 / 32768.0)) / 2.0;
    assert_relative_eq!(audio.samples[0], expected, epsilon = 1e-5);
}

#[test]
fn test_read_24bit_pcm() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("test_24bit.wav");

    let spec = WavSpec {
        channels: 1,
        sample_rate: DEFAULT_SAMPLE_RATE,
        bits_per_sample: 24,
        sample_format: SampleFormat::Int,
    };

    {
        let file = File::create(&wav_path).unwrap();
        let mut writer = WavWriter::new(BufWriter::new(file), spec).unwrap();
        // 0.5 * 8388608 = 4194304
        writer.write_sample(4194304i32).unwrap();
        writer.finalize().unwrap();
    }

    let audio = read_wav(&wav_path).expect("Failed to read 24-bit wav");
    assert_eq!(audio.channels, 1);
    assert_relative_eq!(audio.samples[0], 0.5, epsilon = 1e-6);
}

#[test]
fn test_read_32bit_float() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("test_32float.wav");

    let spec = WavSpec {
        channels: 1,
        sample_rate: DEFAULT_SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    {
        let file = File::create(&wav_path).unwrap();
        let mut writer = WavWriter::new(BufWriter::new(file), spec).unwrap();
        writer.write_sample(0.12345f32).unwrap();
        writer.write_sample(-0.6789f32).unwrap();
        writer.finalize().unwrap();
    }

    let audio = read_wav(&wav_path).expect("Failed to read 32-bit float wav");
    assert_relative_eq!(audio.samples[0], 0.12345, epsilon = 1e-6);
    assert_relative_eq!(audio.samples[1], -0.6789, epsilon = 1e-6);
}

#[test]
fn test_case_insensitive_path_resolution() {
    let dir = tempdir().unwrap();
    let upper_wav = dir.path().join("VocalSample.WAV");

    let spec = WavSpec {
        channels: 1,
        sample_rate: DEFAULT_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    {
        let file = File::create(&upper_wav).unwrap();
        let mut writer = WavWriter::new(BufWriter::new(file), spec).unwrap();
        writer.write_sample(1000i16).unwrap();
        writer.finalize().unwrap();
    }

    // Attempt to resolve using lowercase
    let lower_request = dir.path().join("vocalsample.wav");
    let resolved = resolve_audio_path(&lower_request).expect("Resolution failed");
    assert!(resolved.exists());
    assert_eq!(
        resolved
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_lowercase(),
        "vocalsample.wav"
    );
}

#[test]
fn test_missing_file_returns_not_found() {
    let res = read_wav(Path::new("C:/definitely/does_not_exist_at_all.wav"));
    match res {
        Err(AudioError::FileNotFound(_)) => (),
        other => panic!("Expected FileNotFound, got {:?}", other),
    }
}

#[test]
fn test_nan_and_out_of_range_clamping() {
    let dir = tempdir().unwrap();
    let wav_path = dir.path().join("clamped.wav");

    let samples = vec![f32::NAN, 1.5, -2.0, 0.0];
    let buf = AudioBuffer::new(samples, DEFAULT_SAMPLE_RATE, 1);

    write_wav(&wav_path, &buf).expect("Write with non-finite/out-of-bounds failed");
    let loaded = read_wav(&wav_path).expect("Read failed");

    // NaN should map to 0.0
    assert_relative_eq!(loaded.samples[0], 0.0, epsilon = 1e-4);
    // 1.5 clamped to 32767 / 32768
    assert_relative_eq!(loaded.samples[1], 32767.0 / 32768.0, epsilon = 1e-4);
    // -2.0 clamped to -32768 / 32768 = -1.0
    assert_relative_eq!(loaded.samples[2], -1.0, epsilon = 1e-4);
    // 0.0 maps to 0.0
    assert_relative_eq!(loaded.samples[3], 0.0, epsilon = 1e-4);
}

#[test]
fn test_read_wav_resamples_non_44100_to_default_sample_rate() {
    let dir = tempdir().unwrap();
    let wav_48k = dir.path().join("tone_48k.wav");

    let src_sr = 48000u32;
    let src_len = 4800usize; // 0.1s
    let spec = WavSpec {
        channels: 1,
        sample_rate: src_sr,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    {
        let file = File::create(&wav_48k).unwrap();
        let mut writer = WavWriter::new(BufWriter::new(file), spec).unwrap();
        for i in 0..src_len {
            let t = i as f32 / src_sr as f32;
            writer
                .write_sample((2.0 * PI * 440.0 * t).sin() * 0.5)
                .unwrap();
        }
        writer.finalize().unwrap();
    }

    let raw = read_wav_raw(&wav_48k).expect("read_wav_raw failed");
    assert_eq!(raw.sample_rate, 48000);
    assert_eq!(raw.samples.len(), src_len);

    let resampled = read_wav(&wav_48k).expect("read_wav failed");
    assert_eq!(resampled.sample_rate, DEFAULT_SAMPLE_RATE);
    assert_eq!(resampled.channels, 1);
    assert_eq!(resampled.samples.len(), 4410);
    assert!(resampled.samples.iter().all(|v| v.is_finite()));

    // Verify 440 Hz tone accuracy in interior after resampling
    let mut max_err = 0.0_f32;
    for (i, &actual) in resampled.samples.iter().enumerate().skip(200).take(4000) {
        let t = i as f32 / DEFAULT_SAMPLE_RATE as f32;
        let expected = (2.0 * PI * 440.0 * t).sin() * 0.5;
        max_err = max_err.max((actual - expected).abs());
    }
    assert!(
        max_err < 1e-3,
        "48k->44.1k resampling interior max_err={max_err}"
    );
}
