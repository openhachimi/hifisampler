//! Benchmarking and A/B audio equivalence verification library for `hifisampler-rs`.
//!
//! Provides deterministic WAV test fixture generation, ITU-R BS.1770-4 LUFS and
//! spectral/Mel equivalence metrics, nearest-rank latency statistics, and
//! `HIFIBIN1` binary feature cache inspection.

use hifi_core::audio::{read_wav, write_wav, AudioBuffer, DEFAULT_SAMPLE_RATE};
use hifi_core::cache::{load_cache, HifiBinHeader, HEADER_SIZE};
use hifi_dsp::hnsep_stft::stft_1d;
use hifi_dsp::{dynamic_range_compression_default, MelConfig, MelSpectrogram, PadMode};
use hifi_lufs::measure_lufs;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;

/// Summary of statistical latency and throughput metrics across repeated runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LatencySummary {
    pub count: usize,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub throughput_nps: f64,
}

impl LatencySummary {
    /// Compute nearest-rank latency percentiles and throughput from millisecond durations.
    pub fn from_samples(samples_ms: &[f64]) -> Self {
        if samples_ms.is_empty() {
            return Self {
                count: 0,
                mean_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                throughput_nps: 0.0,
            };
        }
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let sum: f64 = sorted.iter().sum();
        let mean_ms = sum / sorted.len() as f64;
        let p50_ms = nearest_rank_percentile(&sorted, 50.0);
        let p95_ms = nearest_rank_percentile(&sorted, 95.0);
        let throughput_nps = if mean_ms > 0.0 { 1000.0 / mean_ms } else { 0.0 };
        Self {
            count: sorted.len(),
            mean_ms,
            min_ms: sorted[0],
            max_ms: sorted[sorted.len() - 1],
            p50_ms,
            p95_ms,
            throughput_nps,
        }
    }
}

/// Nearest-rank percentile matching `backend/stats.py::_percentile`.
pub fn nearest_rank_percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64).round() as isize;
    let clamped = idx.clamp(0, sorted_values.len() as isize - 1) as usize;
    sorted_values[clamped]
}

/// Quantitative A/B equivalence metrics comparing a Python reference WAV against a Rust WAV.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioEquivalenceReport {
    pub py_samples: usize,
    pub rs_samples: usize,
    pub sample_count_diff: isize,
    pub py_all_finite: bool,
    pub rs_all_finite: bool,
    pub py_peak: f32,
    pub rs_peak: f32,
    pub py_rms_db: f64,
    pub rs_rms_db: f64,
    pub py_lufs: f64,
    pub rs_lufs: f64,
    pub lufs_delta_db: f64,
    pub mel_spectral_corr: f64,
    pub mel_spectral_mse: f64,
    pub stft_mag_corr: f64,
    pub stft_log_mag_mse: f64,
    pub waveform_corr: f64,
}

/// Parsed summary of a `HIFIBIN1` binary feature cache file on disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HifiBinSummary {
    pub magic: String,
    pub version: u32,
    pub rows: u32,
    pub cols: u32,
    pub scale: f64,
    pub file_size: u64,
    pub expected_size: u64,
    pub size_valid: bool,
    pub all_finite: bool,
}

/// Generate a deterministic 16-bit PCM mono harmonic vocal-like test WAV file.
///
/// Includes fundamental + 10 decaying harmonics, smooth 10 ms fade-in/out envelope,
/// and deterministic pseudo-random low-level breath noise so HN-SEP, glottal tension,
/// Klatt resonators, growl, and vocal fry all receive realistic spectral content.
pub fn generate_harmonic_vocal_wav(
    path: &Path,
    freq_hz: f64,
    duration_sec: f64,
    sample_rate: u32,
) -> Result<(), String> {
    let n_samples = ((sample_rate as f64 * duration_sec).round() as usize).max(1);
    let fade_len = ((sample_rate as f64 * 0.01).round() as usize).min(n_samples / 4);
    let mut samples = Vec::with_capacity(n_samples);
    let mut rng_state: u64 = 0x5EED_CAFE_1234_5678;

    for i in 0..n_samples {
        let t = i as f64 / sample_rate as f64;
        let mut harmonic = 0.0_f64;
        for k in 1..=10 {
            let amp = 1.0 / (k as f64).powf(1.2);
            harmonic += amp * (2.0 * std::f64::consts::PI * freq_hz * k as f64 * t).sin();
        }
        // Deterministic LCG breath component (~ -32 dB relative)
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((rng_state >> 33) as f64 / (u32::MAX as f64)) * 2.0 - 1.0;
        let env = if fade_len > 0 && i < fade_len {
            i as f64 / fade_len as f64
        } else if fade_len > 0 && i >= n_samples - fade_len {
            (n_samples - 1 - i) as f64 / fade_len as f64
        } else {
            1.0
        };
        let val = env * (0.18 * harmonic + 0.012 * u);
        samples.push((val as f32).clamp(-0.95, 0.95));
    }

    let buf = AudioBuffer::new(samples, sample_rate, 1);
    write_wav(path, &buf).map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

/// Generate a pure sine 16-bit PCM mono WAV matching `tests/e2e/harness.py`.
pub fn generate_sine_wav(
    path: &Path,
    freq_hz: f64,
    duration_sec: f64,
    sample_rate: u32,
) -> Result<(), String> {
    let n_samples = ((sample_rate as f64 * duration_sec).round() as usize).max(1);
    let samples: Vec<f32> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (0.5 * (2.0 * std::f64::consts::PI * freq_hz * t).sin()) as f32
        })
        .collect();
    let buf = AudioBuffer::new(samples, sample_rate, 1);
    write_wav(path, &buf).map_err(|e| format!("Failed to write {}: {e}", path.display()))
}

/// Inspect a `HIFIBIN1` file on disk and validate its 28-byte header and float32 matrix.
pub fn inspect_hifibin(path: &Path) -> Result<HifiBinSummary, String> {
    let meta =
        std::fs::metadata(path).map_err(|e| format!("Failed to stat {}: {e}", path.display()))?;
    let file_size = meta.len();
    let mut f = File::open(path).map_err(|e| format!("Failed to open {}: {e}", path.display()))?;
    let mut hdr_bytes = [0u8; HEADER_SIZE];
    f.read_exact(&mut hdr_bytes)
        .map_err(|e| format!("Failed to read header of {}: {e}", path.display()))?;

    let hdr = HifiBinHeader::from_bytes(&hdr_bytes)
        .map_err(|e| format!("Invalid header in {}: {e}", path.display()))?;
    let expected_size = HEADER_SIZE as u64 + (hdr.rows as u64) * (hdr.cols as u64) * 4;
    let cached = load_cache(path, None)
        .map_err(|e| format!("Failed to load cache {}: {e}", path.display()))?
        .ok_or_else(|| format!("load_cache returned None for {}", path.display()))?;
    let all_finite = cached.mel_origin.iter().all(|v| v.is_finite()) && cached.scale.is_finite();

    let magic = String::from_utf8_lossy(&hdr.magic)
        .trim_end_matches('\0')
        .to_string();

    Ok(HifiBinSummary {
        magic,
        version: hdr.version,
        rows: hdr.rows,
        cols: hdr.cols,
        scale: hdr.scale,
        file_size,
        expected_size,
        size_valid: file_size == expected_size,
        all_finite,
    })
}

/// Compare two WAV files on disk for acoustic, LUFS, and spectral equivalence.
pub fn compare_wav_files(py_wav: &Path, rs_wav: &Path) -> Result<AudioEquivalenceReport, String> {
    let py_buf = read_wav(py_wav)
        .map_err(|e| format!("Failed to read Python WAV {}: {e}", py_wav.display()))?;
    let rs_buf = read_wav(rs_wav)
        .map_err(|e| format!("Failed to read Rust WAV {}: {e}", rs_wav.display()))?;
    if py_buf.sample_rate != rs_buf.sample_rate {
        return Err(format!(
            "Sample rate mismatch: py={} vs rs={}",
            py_buf.sample_rate, rs_buf.sample_rate
        ));
    }
    compare_audio_buffers(&py_buf.samples, &rs_buf.samples, py_buf.sample_rate)
}

/// Compare two mono float32 audio buffers at `sample_rate` for A/B equivalence.
pub fn compare_audio_buffers(
    py: &[f32],
    rs: &[f32],
    sample_rate: u32,
) -> Result<AudioEquivalenceReport, String> {
    let py_all_finite = py.iter().all(|v| v.is_finite());
    let rs_all_finite = rs.iter().all(|v| v.is_finite());
    let py_peak = py.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let rs_peak = rs.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);

    let py_rms_db = rms_dbfs(py);
    let rs_rms_db = rms_dbfs(rs);

    let py_lufs = measure_lufs(py, sample_rate)
        .map_err(|e| format!("measure_lufs failed on Python audio: {e}"))?;
    let rs_lufs = measure_lufs(rs, sample_rate)
        .map_err(|e| format!("measure_lufs failed on Rust audio: {e}"))?;
    let lufs_delta_db = if py_lufs.is_finite() && rs_lufs.is_finite() {
        rs_lufs - py_lufs
    } else {
        f64::NAN
    };

    let min_len = py.len().min(rs.len());
    let waveform_corr = if min_len > 1 {
        pearson_corr_f32(&py[..min_len], &rs[..min_len])
    } else {
        0.0
    };

    let (mel_spectral_corr, mel_spectral_mse) = compare_mel_spectra(py, rs, sample_rate)?;
    let (stft_mag_corr, stft_log_mag_mse) = compare_stft_spectra(py, rs)?;

    Ok(AudioEquivalenceReport {
        py_samples: py.len(),
        rs_samples: rs.len(),
        sample_count_diff: rs.len() as isize - py.len() as isize,
        py_all_finite,
        rs_all_finite,
        py_peak,
        rs_peak,
        py_rms_db,
        rs_rms_db,
        py_lufs,
        rs_lufs,
        lufs_delta_db,
        mel_spectral_corr,
        mel_spectral_mse,
        stft_mag_corr,
        stft_log_mag_mse,
        waveform_corr,
    })
}

fn rms_dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mean_sq = samples
        .iter()
        .map(|&x| {
            let v = x as f64;
            v * v
        })
        .sum::<f64>()
        / samples.len() as f64;
    let rms = mean_sq.sqrt();
    if rms <= 1e-12 {
        -240.0
    } else {
        20.0 * rms.log10()
    }
}

fn compare_mel_spectra(py: &[f32], rs: &[f32], sample_rate: u32) -> Result<(f64, f64), String> {
    let mel = MelSpectrogram::new(MelConfig {
        sample_rate,
        n_fft: 2048,
        win_size: 2048,
        hop_length: 512,
        f_min: 40.0,
        f_max: 16000.0,
        n_mels: 128,
        center: false,
    })
    .map_err(|e| format!("MelSpectrogram init failed: {e}"))?;

    let pad_to_min = |s: &[f32]| -> Vec<f32> {
        if s.len() >= 2048 {
            s.to_vec()
        } else {
            let mut out = vec![0.0_f32; 2048];
            out[..s.len()].copy_from_slice(s);
            out
        }
    };

    let py_pad = pad_to_min(py);
    let rs_pad = pad_to_min(rs);
    let py_mel = dynamic_range_compression_default(
        mel.compute(&py_pad, 0.0, 1.0)
            .map_err(|e| format!("py mel failed: {e}"))?
            .view(),
    );
    let rs_mel = dynamic_range_compression_default(
        mel.compute(&rs_pad, 0.0, 1.0)
            .map_err(|e| format!("rs mel failed: {e}"))?
            .view(),
    );

    Ok(compare_2d_aligned(&py_mel, &rs_mel))
}

fn compare_stft_spectra(py: &[f32], rs: &[f32]) -> Result<(f64, f64), String> {
    let n_fft = 2048usize;
    let hop = 512usize;
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n_fft as f32).cos()))
        .collect();

    let py_spec = stft_1d(py, n_fft, hop, &window, true, PadMode::Reflect)
        .map_err(|e| format!("py stft failed: {e}"))?;
    let rs_spec = stft_1d(rs, n_fft, hop, &window, true, PadMode::Reflect)
        .map_err(|e| format!("rs stft failed: {e}"))?;

    let bins = py_spec.nrows().min(rs_spec.nrows());
    let frames = py_spec.ncols().min(rs_spec.ncols());
    if bins == 0 || frames == 0 {
        return Ok((0.0, 0.0));
    }

    let mut py_mag = Vec::with_capacity(bins * frames);
    let mut rs_mag = Vec::with_capacity(bins * frames);
    let mut log_mse_sum = 0.0_f64;
    let count = (bins * frames) as f64;

    for b in 0..bins {
        for t in 0..frames {
            let pm = py_spec[(b, t)].norm() as f64;
            let rm = rs_spec[(b, t)].norm() as f64;
            py_mag.push(pm);
            rs_mag.push(rm);
            let dl = rm.max(1e-6).ln() - pm.max(1e-6).ln();
            log_mse_sum += dl * dl;
        }
    }

    let corr = pearson_corr_f64(&py_mag, &rs_mag);
    Ok((corr, log_mse_sum / count))
}

fn compare_2d_aligned(a: &Array2<f32>, b: &Array2<f32>) -> (f64, f64) {
    let rows = a.nrows().min(b.nrows());
    let cols = a.ncols().min(b.ncols());
    if rows == 0 || cols == 0 {
        return (0.0, 0.0);
    }
    let mut av = Vec::with_capacity(rows * cols);
    let mut bv = Vec::with_capacity(rows * cols);
    let mut mse_sum = 0.0_f64;
    for r in 0..rows {
        for c in 0..cols {
            let x = a[(r, c)] as f64;
            let y = b[(r, c)] as f64;
            av.push(x);
            bv.push(y);
            let d = y - x;
            mse_sum += d * d;
        }
    }
    let corr = pearson_corr_f64(&av, &bv);
    (corr, mse_sum / (rows * cols) as f64)
}

fn pearson_corr_f32(a: &[f32], b: &[f32]) -> f64 {
    let af: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let bf: Vec<f64> = b.iter().map(|&v| v as f64).collect();
    pearson_corr_f64(&af, &bf)
}

fn pearson_corr_f64(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean_a = a[..n].iter().sum::<f64>() / nf;
    let mean_b = b[..n].iter().sum::<f64>() / nf;
    let mut num = 0.0_f64;
    let mut den_a = 0.0_f64;
    let mut den_b = 0.0_f64;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let den = (den_a * den_b).sqrt();
    if den <= 1e-18 {
        if (den_a - den_b).abs() <= 1e-18 {
            1.0
        } else {
            0.0
        }
    } else {
        (num / den).clamp(-1.0, 1.0)
    }
}

/// Minimal HTTP/1.1 loopback response for benchmark requests.
#[derive(Debug, Clone)]
pub struct LoopbackHttpResponse {
    pub status: u16,
    pub body: String,
}

/// Send an HTTP/1.1 request over loopback `127.0.0.1:{port}` without proxies.
pub fn http_request_loopback(
    port: u16,
    method: &str,
    path: &str,
    body: Option<(&str, &str)>,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<LoopbackHttpResponse, String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, connect_timeout)
        .map_err(|e| format!("Connection failed: {e}"))?;
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(read_timeout));
    let _ = stream.set_write_timeout(Some(read_timeout));

    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nUser-Agent: hifi-bench\r\n"
    );
    if let Some((payload, content_type)) = body {
        req.push_str(&format!(
            "Content-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{payload}",
            payload.len()
        ));
    } else {
        req.push_str("\r\n");
    }

    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("Write failed: {e}"))?;
    stream.flush().map_err(|e| format!("Flush failed: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .map_err(|e| format!("Read status line failed: {e}"))?;
    let mut parts = status_line.trim_end_matches(['\r', '\n']).splitn(3, ' ');
    let _http_ver = parts.next();
    let status: u16 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("Invalid HTTP status line: {status_line:?}"))?;

    let mut content_length: Option<usize> = None;
    let mut is_chunked = false;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("Read header failed: {e}"))?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            let kl = k.trim().to_ascii_lowercase();
            let vl = v.trim();
            if kl == "content-length" {
                content_length = vl.parse().ok();
            } else if kl == "transfer-encoding" && vl.to_ascii_lowercase().contains("chunked") {
                is_chunked = true;
            }
        }
    }

    let body_bytes = if is_chunked {
        let mut out = Vec::new();
        loop {
            let mut size_line = String::new();
            if reader
                .read_line(&mut size_line)
                .map_err(|e| format!("Read chunk size failed: {e}"))?
                == 0
            {
                break;
            }
            let hex = size_line.trim().split(';').next().unwrap_or("0").trim();
            let sz = usize::from_str_radix(hex, 16)
                .map_err(|_| format!("Invalid chunk size: {size_line:?}"))?;
            if sz == 0 {
                break;
            }
            let mut buf = vec![0u8; sz];
            reader
                .read_exact(&mut buf)
                .map_err(|e| format!("Read chunk body failed: {e}"))?;
            out.extend_from_slice(&buf);
            let mut crlf = [0u8; 2];
            let _ = reader.read_exact(&mut crlf);
        }
        out
    } else if let Some(len) = content_length {
        let mut buf = vec![0u8; len];
        reader
            .read_exact(&mut buf)
            .map_err(|e| format!("Read body failed: {e}"))?;
        buf
    } else {
        let mut buf = Vec::new();
        reader
            .read_to_end(&mut buf)
            .map_err(|e| format!("Read EOF body failed: {e}"))?;
        buf
    };

    Ok(LoopbackHttpResponse {
        status,
        body: String::from_utf8_lossy(&body_bytes).into_owned(),
    })
}

/// Bind and immediately release an ephemeral loopback TCP port on `127.0.0.1`.
pub fn find_free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .map_err(|e| format!("Failed to bind ephemeral port: {e}"))?;
    let addr = listener
        .local_addr()
        .map_err(|e| format!("Failed to inspect ephemeral port: {e}"))?;
    Ok(addr.port())
}

/// Read peak working set RSS in bytes for a running (or unclosed) Windows child process.
#[cfg(windows)]
pub fn process_peak_rss_bytes(child: &std::process::Child) -> Option<u64> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };

    let handle = child.as_raw_handle();
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let ok = unsafe { K32GetProcessMemoryInfo(handle as _, &mut counters, counters.cb) };
    if ok != 0 {
        Some(counters.PeakWorkingSetSize as u64)
    } else {
        None
    }
}

/// Read total peak working set RSS in bytes across `root_pid` and all descendant processes.
#[cfg(windows)]
pub fn process_tree_peak_rss_bytes(root_pid: u32) -> Option<u64> {
    let script = format!(
        "$procs = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId; \
         $seen = @{{ {root_pid} = $true }}; \
         $queue = [System.Collections.Generic.Queue[int]]::new(); \
         $queue.Enqueue({root_pid}); \
         while ($queue.Count -gt 0) {{ \
             $cur = $queue.Dequeue(); \
             foreach ($p in $procs) {{ \
                 if ($p.ParentProcessId -eq $cur -and -not $seen.ContainsKey([int]$p.ProcessId)) {{ \
                     $seen[[int]$p.ProcessId] = $true; \
                     $queue.Enqueue([int]$p.ProcessId); \
                 }} \
             }} \
         }}; \
         $total = [int64]0; \
         foreach ($id in $seen.Keys) {{ \
             $gp = Get-Process -Id $id -ErrorAction SilentlyContinue; \
             if ($gp) {{ $total += [int64]$gp.PeakWorkingSet64 }} \
         }}; \
         Write-Output $total"
    );
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let val: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    (val > 0).then_some(val)
}

#[cfg(not(windows))]
pub fn process_tree_peak_rss_bytes(root_pid: u32) -> Option<u64> {
    let status_path = format!("/proc/{root_pid}/status");
    let text = std::fs::read_to_string(status_path).ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(windows))]
pub fn process_peak_rss_bytes(child: &std::process::Child) -> Option<u64> {
    let status_path = format!("/proc/{}/status", child.id());
    let text = std::fs::read_to_string(status_path).ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Return the default sample rate constant (`44100`).
pub fn default_sample_rate() -> u32 {
    DEFAULT_SAMPLE_RATE
}

#[cfg(test)]
mod tests {
    use super::*;
    use hifi_core::cache::save_cache;
    use tempfile::tempdir;

    #[test]
    fn nearest_rank_and_latency_summary_match_stats_contract() {
        let vals: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let summary = LatencySummary::from_samples(&vals);
        assert_eq!(summary.count, 100);
        assert!((summary.mean_ms - 50.5).abs() < 1e-9);
        assert!((summary.min_ms - 1.0).abs() < 1e-9);
        assert!((summary.max_ms - 100.0).abs() < 1e-9);
        assert!((summary.p50_ms - 51.0).abs() < 1e-9);
        assert!((summary.p95_ms - 95.0).abs() < 1e-9);
        assert!((summary.throughput_nps - (1000.0 / 50.5)).abs() < 1e-9);
    }

    #[test]
    fn audio_equivalence_identity_and_perturbation() {
        let sr = 44100u32;
        let dir = tempdir().unwrap();
        let wav_a = dir.path().join("a.wav");
        let wav_b = dir.path().join("b.wav");
        generate_harmonic_vocal_wav(&wav_a, 220.0, 0.4, sr).unwrap();
        generate_sine_wav(&wav_b, 440.0, 0.4, sr).unwrap();

        let identical = compare_wav_files(&wav_a, &wav_a).unwrap();
        assert_eq!(identical.sample_count_diff, 0);
        assert!(identical.py_all_finite && identical.rs_all_finite);
        assert!(identical.lufs_delta_db.abs() < 1e-9);
        assert!((identical.mel_spectral_corr - 1.0).abs() < 1e-6);
        assert!(identical.mel_spectral_mse < 1e-10);
        assert!((identical.stft_mag_corr - 1.0).abs() < 1e-6);
        assert!(identical.stft_log_mag_mse < 1e-10);
        assert!((identical.waveform_corr - 1.0).abs() < 1e-6);

        let distinct = compare_wav_files(&wav_a, &wav_b).unwrap();
        assert!(distinct.mel_spectral_corr < 0.9);
        assert!(distinct.mel_spectral_mse > 0.1);
    }

    #[test]
    fn hifibin_inspector_validates_saved_cache() {
        let dir = tempdir().unwrap();
        let bin_path = dir.path().join("test_Hb100_Ht0_Hv100_g0.hifi.bin");
        let mel = Array2::<f32>::from_elem((128, 32), -2.5);
        save_cache(&bin_path, &mel, 0.75).unwrap();

        let summary = inspect_hifibin(&bin_path).unwrap();
        assert_eq!(summary.magic, "HIFIBIN1");
        assert_eq!(summary.version, 1);
        assert_eq!(summary.rows, 128);
        assert_eq!(summary.cols, 32);
        assert!((summary.scale - 0.75).abs() < 1e-12);
        assert!(summary.size_valid);
        assert!(summary.all_finite);
    }
}
