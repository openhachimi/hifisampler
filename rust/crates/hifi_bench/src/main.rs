use hifi_bench::{
    compare_wav_files, find_free_port, generate_harmonic_vocal_wav, http_request_loopback,
    inspect_hifibin, process_peak_rss_bytes, process_tree_peak_rss_bytes, AudioEquivalenceReport,
    HifiBinSummary, LatencySummary,
};
use hifi_core::audio::{read_wav, write_wav, AudioBuffer};
use hifi_core::utau::parse_flags;
use hifi_core::wavtool::WavtoolEnvelope;
use hifi_lufs::measure_lufs;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Debug, Clone)]
struct NoteCase {
    name: &'static str,
    pitch: &'static str,
    velocity: &'static str,
    flags: &'static str,
    offset: &'static str,
    length: &'static str,
    consonant: &'static str,
    cutoff: &'static str,
    tempo: &'static str,
    pitch_string: &'static str,
    use_real_vocal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StageTiming {
    feature_ms: f64,
    synthesis_ms: f64,
    postprocess_ms: f64,
    total_ms: f64,
    wall_ms: f64,
    cache_hit: bool,
    hnsep_cache_hit: bool,
    input_samples: usize,
    output_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdCaseComparison {
    name: String,
    flags: String,
    pitch: String,
    length: String,
    pitch_string: String,
    python: Option<StageTiming>,
    rust: StageTiming,
    equivalence: Option<AudioEquivalenceReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WarmAndInteropResult {
    py_warm_disk: Option<StageTiming>,
    py_warm_ram: Option<StageTiming>,
    rs_warm_disk: StageTiming,
    rs_warm_ram: StageTiming,
    py_to_rs_default: Option<InteropCheck>,
    py_to_rs_fx: Option<InteropCheck>,
    rs_to_py_default: Option<InteropCheck>,
    rs_to_py_fx: Option<InteropCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InteropCheck {
    cache_file_name: String,
    header: HifiBinSummary,
    consumer_cache_hit: bool,
    consumer_timing: StageTiming,
    equivalence: AudioEquivalenceReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BatchThroughputResult {
    note_count: usize,
    concurrency: usize,
    wall_ms: f64,
    notes_per_sec: f64,
    client_latency: LatencySummary,
    server_avg_total_ms: f64,
    server_p95_total_ms: f64,
    server_cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ColdStartReport {
    py_default_server_startup_ms: Vec<f64>,
    py_onnx_server_startup_ms: Vec<f64>,
    rs_server_startup_ms: Vec<f64>,
    py_cli_cold_ms: Option<f64>,
    rs_cli_cold_ms: f64,
    py_cli_warm: Option<LatencySummary>,
    rs_cli_warm: LatencySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinarySizesReport {
    rust_hifisampler_bytes: u64,
    rust_melos_bytes: Option<u64>,
    rust_melos_wavtool_bytes: Option<u64>,
    rust_hifiserver_bytes: u64,
    csharp_hifisampler_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseModeMetrics {
    note_count: usize,
    wall_ms: f64,
    throughput_nps: f64,
    vocoder_calls: usize,
    server_sum_synthesis_ms: f64,
    server_sum_feature_ms: f64,
    server_sum_postprocess_ms: f64,
    output_samples: usize,
    all_finite: bool,
    lufs: f64,
    overlap_rms_ratio: f64,
    backfilled_stubs_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseScenarioComparison {
    name: String,
    note_count: usize,
    dur_ms: f64,
    ovr_ms: f64,
    expected_samples: usize,
    baseline_note_by_note: PhraseModeMetrics,
    melos_route_b: PhraseModeMetrics,
    wall_speedup: f64,
    vocoder_synthesis_ms_reduction: f64,
    sample_count_diff: isize,
    lufs_delta_db: f64,
    overlap_continuity_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseComparisonReport {
    repo_root: String,
    binary_sizes: BinarySizesReport,
    scenarios: Vec<PhraseScenarioComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryReport {
    py_default_peak_rss_bytes: Option<u64>,
    py_onnx_peak_rss_bytes: Option<u64>,
    rs_server_peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FullBenchmarkReport {
    repo_root: String,
    real_vocal_source: Option<String>,
    binary_sizes: BinarySizesReport,
    cold_start: ColdStartReport,
    cold_cases: Vec<ColdCaseComparison>,
    warm_and_interop: WarmAndInteropResult,
    mixed_batch_py: Option<BatchThroughputResult>,
    mixed_batch_rs: BatchThroughputResult,
    warm_batch_py: Option<BatchThroughputResult>,
    warm_batch_rs: BatchThroughputResult,
    concurrent_mixed_batch_py: Option<BatchThroughputResult>,
    concurrent_mixed_batch_rs: BatchThroughputResult,
    concurrent_warm_batch_py: Option<BatchThroughputResult>,
    concurrent_warm_batch_rs: BatchThroughputResult,
    phrase_route_b: Option<Vec<PhraseScenarioComparison>>,
    memory: MemoryReport,
}

struct ManagedChild {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl ManagedChild {
    fn spawn(
        cmd: &[String],
        cwd: &Path,
        port: u16,
        envs: &[(&str, &str)],
        log_path: PathBuf,
    ) -> Result<Self, String> {
        let log_file = fs::File::create(&log_path)
            .map_err(|e| format!("Failed to create log {}: {e}", log_path.display()))?;
        let log_err = log_file
            .try_clone()
            .map_err(|e| format!("Failed to clone log handle: {e}"))?;

        let mut command = Command::new(&cmd[0]);
        command
            .args(&cmd[1..])
            .current_dir(cwd)
            .stdout(Stdio::from(log_file))
            .stderr(Stdio::from(log_err));
        for (k, v) in envs {
            command.env(k, v);
        }

        let child = command
            .spawn()
            .map_err(|e| format!("Failed to spawn {}: {e}", cmd[0]))?;
        Ok(Self {
            child,
            port,
            log_path,
        })
    }

    fn wait_ready(&mut self, timeout: Duration) -> Result<f64, String> {
        let t0 = Instant::now();
        while t0.elapsed() < timeout {
            if let Ok(Some(status)) = self.child.try_wait() {
                let logs = fs::read_to_string(&self.log_path).unwrap_or_default();
                return Err(format!(
                    "Server exited prematurely with {status}.\nLogs:\n{logs}"
                ));
            }
            if let Ok(resp) = http_request_loopback(
                self.port,
                "GET",
                "/",
                None,
                Duration::from_millis(250),
                Duration::from_secs(2),
            ) {
                if resp.status == 200 && resp.body.contains("Server Ready") {
                    return Ok(t0.elapsed().as_secs_f64() * 1000.0);
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let logs = fs::read_to_string(&self.log_path).unwrap_or_default();
        Err(format!(
            "Timed out waiting for server on port {} after {:?}.\nLogs:\n{logs}",
            self.port, timeout
        ))
    }

    fn peak_rss_bytes(&self) -> Option<u64> {
        process_tree_peak_rss_bytes(self.child.id()).or_else(|| process_peak_rss_bytes(&self.child))
    }

    fn shutdown(&mut self) {
        #[cfg(windows)]
        {
            let pid = self.child.id().to_string();
            let _ = Command::new("taskkill")
                .args(["/F", "/T", "/PID", &pid])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn kill_processes_with_port_arg(port: u16) {
    #[cfg(windows)]
    {
        let script = format!(
            "Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '--port\\s+{port}(\\s|$)' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"
        );
        let _ = Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = port;
    }
}

fn post_note(port: u16, args: &[String]) -> Result<f64, String> {
    let payload = serde_json::json!({ "args": args }).to_string();
    let t0 = Instant::now();
    let resp = http_request_loopback(
        port,
        "POST",
        "/",
        Some((&payload, "application/json")),
        Duration::from_secs(2),
        Duration::from_secs(120),
    )
    .map_err(|e| format!("POST / failed on port {port}: {e}"))?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if resp.status != 200 {
        return Err(format!(
            "POST / returned status {} on port {}: {}",
            resp.status, port, resp.body
        ));
    }
    Ok(wall_ms)
}

fn fetch_stats(port: u16) -> Result<Value, String> {
    let resp = http_request_loopback(
        port,
        "GET",
        "/stats",
        None,
        Duration::from_secs(1),
        Duration::from_secs(5),
    )
    .map_err(|e| format!("GET /stats failed: {e}"))?;
    if resp.status != 200 {
        return Err(format!("GET /stats returned {}", resp.status));
    }
    serde_json::from_str(&resp.body).map_err(|e| format!("Invalid /stats JSON: {e}"))
}

fn latest_stage_timing(port: u16, wall_ms: f64) -> Result<StageTiming, String> {
    let stats = fetch_stats(port)?;
    let recent = stats
        .get("recent")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| "No recent entry in /stats".to_string())?;
    Ok(StageTiming {
        feature_ms: recent["feature_ms"].as_f64().unwrap_or(0.0),
        synthesis_ms: recent["synthesis_ms"].as_f64().unwrap_or(0.0),
        postprocess_ms: recent["postprocess_ms"].as_f64().unwrap_or(0.0),
        total_ms: recent["total_ms"].as_f64().unwrap_or(0.0),
        wall_ms,
        cache_hit: recent["cache_hit"].as_bool().unwrap_or(false),
        hnsep_cache_hit: recent["hnsep_cache_hit"].as_bool().unwrap_or(false),
        input_samples: recent["input_samples"].as_u64().unwrap_or(0) as usize,
        output_samples: recent["output_samples"].as_u64().unwrap_or(0) as usize,
    })
}

fn note_args(in_wav: &Path, out_wav: &Path, case: &NoteCase) -> Vec<String> {
    vec![
        in_wav.to_string_lossy().into_owned(),
        out_wav.to_string_lossy().into_owned(),
        case.pitch.to_string(),
        case.velocity.to_string(),
        case.flags.to_string(),
        case.offset.to_string(),
        case.length.to_string(),
        case.consonant.to_string(),
        case.cutoff.to_string(),
        "100".to_string(),
        "0".to_string(),
        case.tempo.to_string(),
        case.pitch_string.to_string(),
    ]
}

fn resolve_repo_root(explicit: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(p) = explicit {
        return Ok(p.to_path_buf());
    }
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".."),
    ];
    for c in candidates {
        if c.join("pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx")
            .is_file()
        {
            return c
                .canonicalize()
                .map_err(|e| format!("canonicalize failed: {e}"));
        }
    }
    Err("Could not locate repository root containing pc_nsf_hifigan model.onnx".to_string())
}

fn canonical_path_str(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        s.strip_prefix(r"\\?\").unwrap_or(&s).to_string()
    }
    #[cfg(not(windows))]
    {
        s
    }
}

fn spawn_py_default_server(
    repo: &Path,
    port: u16,
    log_path: PathBuf,
) -> Result<ManagedChild, String> {
    let py_exe = canonical_path_str(&repo.join(".venv/Scripts/python.exe"));
    let srv_py = canonical_path_str(&repo.join("hifiserver.py"));
    let port_s = port.to_string();
    let cmd = vec![py_exe, srv_py, "--port".to_string(), port_s.clone()];
    ManagedChild::spawn(&cmd, repo, port, &[("HIFISAMPLER_PORT", &port_s)], log_path)
}

fn spawn_py_onnx_server(repo: &Path, port: u16, log_path: PathBuf) -> Result<ManagedChild, String> {
    let py_exe = canonical_path_str(&repo.join(".venv/Scripts/python.exe"));
    let helper =
        canonical_path_str(&repo.join("rust/crates/hifi_bench/scripts/py_bench_server.py"));
    let port_s = port.to_string();
    let cmd = vec![
        py_exe,
        helper,
        "--port".to_string(),
        port_s.clone(),
        "--hnsep-onnx".to_string(),
    ];
    ManagedChild::spawn(&cmd, repo, port, &[("HIFISAMPLER_PORT", &port_s)], log_path)
}

fn spawn_rs_server(repo: &Path, port: u16, log_path: PathBuf) -> Result<ManagedChild, String> {
    let rs_srv = canonical_path_str(&repo.join("rust/target/release/hifiserver.exe"));
    let port_s = port.to_string();
    let cmd = vec![rs_srv, "--port".to_string(), port_s.clone()];
    ManagedChild::spawn(&cmd, repo, port, &[("HIFISAMPLER_PORT", &port_s)], log_path)
}

fn measure_cli_cold(
    client_exe: &Path,
    repo: &Path,
    in_wav: &Path,
    out_wav: &Path,
    isolated_home: &Path,
) -> Result<f64, String> {
    let port = find_free_port()?;
    let port_s = port.to_string();
    let home_s = isolated_home.to_string_lossy().into_owned();
    let repo_s = canonical_path_str(repo);
    let _ = fs::create_dir_all(isolated_home);

    let t0 = Instant::now();
    let output = Command::new(client_exe)
        .args([
            in_wav.to_string_lossy().as_ref(),
            out_wav.to_string_lossy().as_ref(),
            "A4",
            "100",
            "",
            "0",
            "300",
            "0",
            "0",
            "100",
            "0",
            "!120",
            "AA",
        ])
        .current_dir(repo)
        .env("USERPROFILE", &home_s)
        .env("HOME", &home_s)
        .env("HIFISAMPLER_PORT", &port_s)
        .env("HIFISAMPLER_HOME", &repo_s)
        .output()
        .map_err(|e| format!("Failed to invoke {}: {e}", client_exe.display()))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    kill_processes_with_port_arg(port);

    if !output.status.success() {
        return Err(format!(
            "Cold CLI {} exited with {}:\nstdout={}\nstderr={}",
            client_exe.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    if !out_wav.is_file() {
        return Err(format!(
            "Cold CLI {} did not produce {}",
            client_exe.display(),
            out_wav.display()
        ));
    }
    Ok(elapsed_ms)
}

fn measure_cli_warm(
    client_exe: &Path,
    repo: &Path,
    port: u16,
    in_wav: &Path,
    out_dir: &Path,
    isolated_home: &Path,
    iters: usize,
) -> Result<LatencySummary, String> {
    let port_s = port.to_string();
    let home_s = isolated_home.to_string_lossy().into_owned();
    let repo_s = canonical_path_str(repo);
    let mut times = Vec::with_capacity(iters);

    for i in 0..iters {
        let out_wav = out_dir.join(format!("cli_warm_{i}.wav"));
        let t0 = Instant::now();
        let output = Command::new(client_exe)
            .args([
                in_wav.to_string_lossy().as_ref(),
                out_wav.to_string_lossy().as_ref(),
                "A4",
                "100",
                "",
                "0",
                "300",
                "0",
                "0",
                "100",
                "0",
                "!120",
                "AA",
            ])
            .current_dir(repo)
            .env("USERPROFILE", &home_s)
            .env("HOME", &home_s)
            .env("HIFISAMPLER_PORT", &port_s)
            .env("HIFISAMPLER_HOME", &repo_s)
            .output()
            .map_err(|e| format!("Warm CLI invoke failed: {e}"))?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if !output.status.success() {
            return Err(format!(
                "Warm CLI {} failed: {}",
                client_exe.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        times.push(elapsed_ms);
    }

    Ok(LatencySummary::from_samples(&times))
}

fn build_batch_jobs(in_wav: &Path, out_dir: &Path, mixed: bool) -> Vec<Vec<String>> {
    let pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "G4", "E4"];
    let flag_sets = if mixed {
        vec!["", "P100", "g-10", "A30", "Hb120Hv90"]
    } else {
        vec![""]
    };
    let lengths = ["220", "280", "350", "400"];
    let total_notes = 20usize;

    (0..total_notes)
        .map(|i| {
            let out_wav = out_dir.join(format!("batch_{i}.wav"));
            let pitch = pitches[i % pitches.len()];
            let flags = flag_sets[i % flag_sets.len()];
            let length = lengths[i % lengths.len()];
            vec![
                in_wav.to_string_lossy().into_owned(),
                out_wav.to_string_lossy().into_owned(),
                pitch.to_string(),
                "100".to_string(),
                flags.to_string(),
                "0".to_string(),
                length.to_string(),
                "0".to_string(),
                "0".to_string(),
                "100".to_string(),
                "0".to_string(),
                "!120".to_string(),
                "AA".to_string(),
            ]
        })
        .collect()
}

fn partition_job_indices(total: usize, workers: usize) -> Vec<Vec<usize>> {
    let w = workers.clamp(1, total.max(1));
    let mut buckets = vec![Vec::new(); w];
    for idx in 0..total {
        buckets[idx % w].push(idx);
    }
    buckets
}

fn run_batch(
    port: u16,
    in_wav: &Path,
    out_dir: &Path,
    mixed: bool,
    concurrency: usize,
) -> Result<BatchThroughputResult, String> {
    let jobs = build_batch_jobs(in_wav, out_dir, mixed);
    let total_notes = jobs.len();
    let workers = concurrency.clamp(1, total_notes);

    let stats_before = fetch_stats(port)?;
    let req_before = stats_before["total_requests"].as_u64().unwrap_or(0);
    let hits_before = stats_before["cache_hits"].as_u64().unwrap_or(0);

    let mut client_times = Vec::with_capacity(total_notes);
    let wall_t0 = Instant::now();
    if workers == 1 {
        for args in &jobs {
            client_times.push(post_note(port, args)?);
        }
    } else {
        let buckets = partition_job_indices(total_notes, workers);
        let per_worker: Vec<Result<Vec<f64>, String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = buckets
                .iter()
                .map(|indices| {
                    let jobs_ref = &jobs;
                    scope.spawn(move || -> Result<Vec<f64>, String> {
                        let mut times = Vec::with_capacity(indices.len());
                        for &idx in indices {
                            times.push(post_note(port, &jobs_ref[idx])?);
                        }
                        Ok(times)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("Concurrent batch worker panicked".to_string()))
                })
                .collect()
        });
        for res in per_worker {
            client_times.extend(res?);
        }
    }
    let wall_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;

    for i in 0..total_notes {
        let out_wav = out_dir.join(format!("batch_{i}.wav"));
        if !out_wav.is_file() {
            return Err(format!("Batch output missing: {}", out_wav.display()));
        }
    }

    let stats_after = fetch_stats(port)?;
    let req_delta = stats_after["total_requests"].as_u64().unwrap_or(0) - req_before;
    let hits_delta = stats_after["cache_hits"].as_u64().unwrap_or(0) - hits_before;
    let hit_rate = if req_delta > 0 {
        hits_delta as f64 / req_delta as f64
    } else {
        0.0
    };

    let srv_totals: Vec<f64> = stats_after["recent"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .take(total_notes)
                .filter_map(|e| e["total_ms"].as_f64())
                .collect()
        })
        .unwrap_or_default();
    let batch_srv_summary = LatencySummary::from_samples(&srv_totals);

    Ok(BatchThroughputResult {
        note_count: total_notes,
        concurrency: workers,
        wall_ms,
        notes_per_sec: (total_notes as f64 * 1000.0) / wall_ms.max(1e-6),
        client_latency: LatencySummary::from_samples(&client_times),
        server_avg_total_ms: batch_srv_summary.mean_ms,
        server_p95_total_ms: batch_srv_summary.p95_ms,
        server_cache_hit_rate: hit_rate,
    })
}

fn run_interop_check(
    producer_port: u16,
    consumer_port: u16,
    src_wav_orig: &Path,
    work_dir: &Path,
    tag: &str,
    flags: &'static str,
) -> Result<InteropCheck, String> {
    let dir = work_dir.join(tag);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let wav_path = dir.join("interop_src.wav");
    fs::copy(src_wav_orig, &wav_path).map_err(|e| e.to_string())?;

    let case = NoteCase {
        name: "interop",
        pitch: "A4",
        velocity: "100",
        flags,
        offset: "0",
        length: "350",
        consonant: "0",
        cutoff: "0",
        tempo: "!120",
        pitch_string: "AA",
        use_real_vocal: false,
    };

    let prod_out = dir.join("producer_out.wav");
    let _ = post_note(producer_port, &note_args(&wav_path, &prod_out, &case))?;

    let parsed_flags = parse_flags(flags);
    let cache_name = parsed_flags.feature_cache_filename("interop_src");
    let cache_path = dir.join(&cache_name);
    if !cache_path.is_file() {
        return Err(format!(
            "Producer did not write expected cache file {}",
            cache_path.display()
        ));
    }
    let header = inspect_hifibin(&cache_path)?;

    // Also remove any sibling *_hnsep* file so consumer MUST rely on the .hifi.bin cache hit
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.contains("_hnsep") {
                    let _ = fs::remove_file(&p);
                }
            }
        }
    }

    let cons_out = dir.join("consumer_out.wav");
    let cons_wall = post_note(consumer_port, &note_args(&wav_path, &cons_out, &case))?;
    let cons_timing = latest_stage_timing(consumer_port, cons_wall)?;
    let equiv = compare_wav_files(&prod_out, &cons_out)?;

    Ok(InteropCheck {
        cache_file_name: cache_name,
        header,
        consumer_cache_hit: cons_timing.cache_hit,
        consumer_timing: cons_timing,
        equivalence: equiv,
    })
}

#[derive(Debug, PartialEq, Eq)]
struct CliArgs {
    explicit_root: Option<PathBuf>,
    json_out: Option<PathBuf>,
    rust_only: bool,
    phrase_only: bool,
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, String> {
    let mut explicit_root: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;
    let mut rust_only = false;
    let mut phrase_only = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--root" => {
                i += 1;
                explicit_root = Some(PathBuf::from(args.get(i).ok_or("--root needs value")?));
            }
            "--json-out" => {
                i += 1;
                json_out = Some(PathBuf::from(args.get(i).ok_or("--json-out needs value")?));
            }
            "--rust-only" => {
                rust_only = true;
            }
            "--phrase-only" => {
                phrase_only = true;
            }
            other => return Err(format!("Unknown argument: {other}")),
        }
        i += 1;
    }

    Ok(CliArgs {
        explicit_root,
        json_out,
        rust_only,
        phrase_only,
    })
}

fn run_cli_exe(
    exe: &Path,
    args: &[String],
    cwd: &Path,
    port_s: &str,
    repo_s: &str,
    home_s: &str,
) -> Result<(), String> {
    let output = Command::new(exe)
        .args(args)
        .current_dir(cwd)
        .env("USERPROFILE", home_s)
        .env("HOME", home_s)
        .env("HIFISAMPLER_PORT", port_s)
        .env("HIFISAMPLER_HOME", repo_s)
        .output()
        .map_err(|e| format!("Failed to invoke {}: {e}", exe.display()))?;
    if !output.status.success() {
        return Err(format!(
            "CLI {} exited with {}:\nstdout={}\nstderr={}",
            exe.display(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn compute_rms(slice: &[f32]) -> f64 {
    if slice.is_empty() {
        return 0.0;
    }
    let sq: f64 = slice.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sq / slice.len() as f64).sqrt()
}

fn compute_overlap_rms_ratio(
    samples: &[f32],
    note_count: usize,
    dur_ms: f64,
    ovr_ms: f64,
    sr: u32,
) -> f64 {
    if note_count < 2 || ovr_ms <= 0.0 || dur_ms <= ovr_ms {
        return 1.0;
    }
    let sr_f = sr as f64;
    let step_ms = dur_ms - ovr_ms;
    let half_w = ((15.0 * sr_f / 1000.0).round() as usize).max(64);
    let mut ratios = Vec::with_capacity(note_count - 1);

    for i in 1..note_count {
        let ovr_start_ms = i as f64 * step_ms;
        let ovr_mid_ms = ovr_start_ms + 0.5 * ovr_ms;
        let left_ref_ms = (ovr_start_ms - 32.0).max(20.0);
        let right_ref_ms = ovr_start_ms + ovr_ms + 32.0;

        let mid_idx = (ovr_mid_ms * sr_f / 1000.0).round() as usize;
        let left_idx = (left_ref_ms * sr_f / 1000.0).round() as usize;
        let right_idx = (right_ref_ms * sr_f / 1000.0).round() as usize;

        let window_rms = |center: usize| -> Option<f64> {
            let s0 = center.saturating_sub(half_w);
            let s1 = (center + half_w).min(samples.len());
            if s1 > s0 + 16 {
                Some(compute_rms(&samples[s0..s1]))
            } else {
                None
            }
        };

        if let (Some(rm), Some(rl), Some(rr)) = (
            window_rms(mid_idx),
            window_rms(left_idx),
            window_rms(right_idx),
        ) {
            let ref_rms = 0.5 * (rl + rr);
            if ref_rms > 1e-6 {
                ratios.push(rm / ref_rms);
            }
        }
    }

    if ratios.is_empty() {
        1.0
    } else {
        ratios.iter().sum::<f64>() / ratios.len() as f64
    }
}

fn concat_baseline_wavs(
    note_paths: &[PathBuf],
    dur_ms: f64,
    ovr_ms: f64,
    sr: u32,
) -> Result<Vec<f32>, String> {
    let sr_f = sr as f64;
    let step_ms = dur_ms - ovr_ms;
    let total_ms = if note_paths.is_empty() {
        0.0
    } else {
        (note_paths.len() - 1) as f64 * step_ms + dur_ms
    };
    let total_samples = (total_ms * sr_f / 1000.0).round() as usize;
    let mut out = vec![0.0_f32; total_samples];

    for (i, path) in note_paths.iter().enumerate() {
        let buf = read_wav(path).map_err(|e| format!("Baseline read {}: {e}", path.display()))?;
        let cur_ovr = if i == 0 { 0.0 } else { ovr_ms };
        let start_ms = if i == 0 { 0.0 } else { i as f64 * step_ms };
        let start_sample = (start_ms * sr_f / 1000.0).round() as usize;
        let end_sample =
            (((start_ms + dur_ms) * sr_f / 1000.0).round() as usize).min(total_samples);

        let env = WavtoolEnvelope {
            p1_ms: 0.0,
            p2_ms: if i == 0 { 25.0 } else { cur_ovr },
            p3_ms: if i + 1 == note_paths.len() {
                25.0
            } else {
                ovr_ms
            },
            v1: 0.0,
            v2: 100.0,
            v3: 100.0,
            v4: 0.0,
            ovr_ms: cur_ovr,
            p4_ms: 0.0,
            p5_v5: None,
        };

        for (local_i, (slot, &src)) in out[start_sample..end_sample]
            .iter_mut()
            .zip(buf.samples.iter())
            .enumerate()
        {
            let pos_ms = local_i as f64 * 1000.0 / sr_f;
            let gain = env.evaluate_gain(pos_ms, dur_ms) as f32;
            *slot += src * gain;
        }
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_phrase_scenario(
    name: &str,
    repo: &Path,
    rs_srv_port: u16,
    hifisampler_exe: &Path,
    melos_exe: &Path,
    melos_wavtool_exe: &Path,
    src_wav: &Path,
    work_dir: &Path,
    pitches: &[&'static str],
    flags_cycle: &[&'static str],
    offset_ms: &'static str,
    req_len_ms: &'static str,
    consonant_ms: &'static str,
    cutoff_ms: &'static str,
    dur_ms: f64,
    ovr_ms: f64,
) -> Result<PhraseScenarioComparison, String> {
    let note_count = pitches.len();
    let port_s = rs_srv_port.to_string();
    let repo_s = canonical_path_str(repo);
    let iso_home = work_dir.join(format!("home_{name}"));
    fs::create_dir_all(&iso_home).map_err(|e| e.to_string())?;
    let home_s = iso_home.to_string_lossy().into_owned();

    let base_dir = work_dir.join(format!("phrase_base_{name}"));
    let route_b_dir = work_dir.join(format!("phrase_routeb_{name}"));
    fs::create_dir_all(&base_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&route_b_dir).map_err(|e| e.to_string())?;

    let base_src = base_dir.join("src.wav");
    let rb_src = route_b_dir.join("src.wav");
    fs::copy(src_wav, &base_src).map_err(|e| e.to_string())?;
    fs::copy(src_wav, &rb_src).map_err(|e| e.to_string())?;

    // Pre-warm all flag feature caches identically on both isolated WAVs before timed runs
    for (fi, &fl) in flags_cycle.iter().enumerate() {
        let warm_case = NoteCase {
            name: "warm",
            pitch: "A4",
            velocity: "100",
            flags: fl,
            offset: offset_ms,
            length: req_len_ms,
            consonant: consonant_ms,
            cutoff: cutoff_ms,
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: false,
        };
        let _ = post_note(
            rs_srv_port,
            &note_args(
                &base_src,
                &base_dir.join(format!("pw_{fi}.wav")),
                &warm_case,
            ),
        )?;
        let _ = post_note(
            rs_srv_port,
            &note_args(
                &rb_src,
                &route_b_dir.join(format!("pw_{fi}.wav")),
                &warm_case,
            ),
        )?;
    }

    let stats_sum_from = |s: &Value, field: &str| -> f64 {
        let sc = s["success_count"].as_u64().unwrap_or(0).min(200) as f64;
        s[field].as_f64().unwrap_or(0.0) * sc
    };

    // 1. Baseline: N sequential hifisampler.exe invocations + 5-point WAV crossfade
    let stats_base_before = fetch_stats(rs_srv_port)?;
    let mut base_note_paths = Vec::with_capacity(note_count);
    let t_base0 = Instant::now();
    for i in 0..note_count {
        let out_wav = base_dir.join(format!("res-{i:04}.wav"));
        let args = vec![
            base_src.to_string_lossy().into_owned(),
            out_wav.to_string_lossy().into_owned(),
            pitches[i].to_string(),
            "100".to_string(),
            flags_cycle[i % flags_cycle.len()].to_string(),
            offset_ms.to_string(),
            req_len_ms.to_string(),
            consonant_ms.to_string(),
            cutoff_ms.to_string(),
            "100".to_string(),
            "0".to_string(),
            "!120".to_string(),
            "AA".to_string(),
        ];
        run_cli_exe(hifisampler_exe, &args, repo, &port_s, &repo_s, &home_s)?;
        base_note_paths.push(out_wav);
    }
    let base_samples = concat_baseline_wavs(&base_note_paths, dur_ms, ovr_ms, 44100)?;
    let base_phrase_wav = base_dir.join("cat-baseline.wav");
    write_wav(
        &base_phrase_wav,
        &AudioBuffer::new(base_samples.clone(), 44100, 1),
    )
    .map_err(|e| format!("Baseline write failed: {e}"))?;
    let base_wall_ms = t_base0.elapsed().as_secs_f64() * 1000.0;
    let stats_base_after = fetch_stats(rs_srv_port)?;

    let base_synth_ms = (stats_sum_from(&stats_base_after, "avg_synthesis_ms")
        - stats_sum_from(&stats_base_before, "avg_synthesis_ms"))
    .max(0.0);
    let base_feat_ms = (stats_sum_from(&stats_base_after, "avg_feature_ms")
        - stats_sum_from(&stats_base_before, "avg_feature_ms"))
    .max(0.0);
    let base_post_ms = (stats_sum_from(&stats_base_after, "avg_postprocess_ms")
        - stats_sum_from(&stats_base_before, "avg_postprocess_ms"))
    .max(0.0);

    let baseline_metrics = PhraseModeMetrics {
        note_count,
        wall_ms: base_wall_ms,
        throughput_nps: note_count as f64 * 1000.0 / base_wall_ms.max(1e-6),
        vocoder_calls: note_count,
        server_sum_synthesis_ms: base_synth_ms,
        server_sum_feature_ms: base_feat_ms,
        server_sum_postprocess_ms: base_post_ms,
        output_samples: base_samples.len(),
        all_finite: base_samples.iter().all(|v| v.is_finite()),
        lufs: measure_lufs(&base_samples, 44100).unwrap_or(f64::NEG_INFINITY),
        overlap_rms_ratio: compute_overlap_rms_ratio(
            &base_samples,
            note_count,
            dur_ms,
            ovr_ms,
            44100,
        ),
        backfilled_stubs_count: 0,
    };

    // 2. Melos Route B: authentic OpenUtau temp.bat + melos.exe + melos-wavtool.exe
    let rb_phrase_wav = route_b_dir.join("cat-melos.wav");
    let tool_str = canonical_path_str(melos_wavtool_exe);
    let resamp_str = canonical_path_str(melos_exe);
    let dir_str = canonical_path_str(&route_b_dir);

    let mut bat = format!(
        "@rem project=\r\n\
         @set tempo=120\r\n\
         @set samples=44100\r\n\
         @set oto={dir_str}\r\n\
         @set tool={tool_str}\r\n\
         @set output=cat-melos.wav\r\n\
         @set helper=temp_helper.bat\r\n\
         @set cachedir={dir_str}\r\n"
    );
    let mut rb_note_paths = Vec::with_capacity(note_count);
    let mut rb_resampler_calls = Vec::with_capacity(note_count);
    let mut rb_wavtool_calls = Vec::with_capacity(note_count);
    for i in 0..note_count {
        let note_wav = route_b_dir.join(format!("res-{i:04}.wav"));
        let note_fname = format!("res-{i:04}.wav");
        let fl = flags_cycle[i % flags_cycle.len()];
        let cur_ovr = if i == 0 { 0.0 } else { ovr_ms };
        let p2 = if i == 0 { 25.0 } else { cur_ovr };
        let p3 = if i + 1 == note_count { 25.0 } else { ovr_ms };

        bat.push_str(&format!(
            "@set resamp={resamp_str}\r\n\
             @set params=100 0 !120 AA\r\n\
             @set flag=\"{fl}\"\r\n\
             @set env=0 {p2} {p3} 0 100 100 0 {cur_ovr} 0 0 100\r\n\
             @set stp=0\r\n\
             @set vel=100\r\n\
             @set temp=\"%cachedir%\\{note_fname}\"\r\n\
             @call %helper% \"%oto%\\src.wav\" {} {dur_ms} 0 {offset_ms} {req_len_ms} {consonant_ms} {cutoff_ms} {i}\r\n",
            pitches[i]
        ));

        rb_resampler_calls.push(vec![
            rb_src.to_string_lossy().into_owned(),
            note_wav.to_string_lossy().into_owned(),
            pitches[i].to_string(),
            "100".to_string(),
            fl.to_string(),
            offset_ms.to_string(),
            req_len_ms.to_string(),
            consonant_ms.to_string(),
            cutoff_ms.to_string(),
            "100".to_string(),
            "0".to_string(),
            "!120".to_string(),
            "AA".to_string(),
        ]);

        rb_wavtool_calls.push(vec![
            rb_phrase_wav.to_string_lossy().into_owned(),
            note_wav.to_string_lossy().into_owned(),
            "0".to_string(),
            format!("{dur_ms}"),
            "0".to_string(),
            format!("{p2}"),
            format!("{p3}"),
            "0".to_string(),
            "100".to_string(),
            "100".to_string(),
            "0".to_string(),
            format!("{cur_ovr}"),
            "0".to_string(),
            "0".to_string(),
            "100".to_string(),
        ]);

        rb_note_paths.push(note_wav);
    }
    fs::write(route_b_dir.join("temp.bat"), bat.as_bytes())
        .map_err(|e| format!("Failed to write temp.bat: {e}"))?;

    let stats_rb_before = fetch_stats(rs_srv_port)?;
    let t_rb0 = Instant::now();
    for args in &rb_resampler_calls {
        run_cli_exe(melos_exe, args, &route_b_dir, &port_s, &repo_s, &home_s)?;
    }
    for (idx, args) in rb_wavtool_calls.iter().enumerate() {
        run_cli_exe(
            melos_wavtool_exe,
            args,
            &route_b_dir,
            &port_s,
            &repo_s,
            &home_s,
        )?;
        if idx + 1 < note_count && rb_phrase_wav.exists() {
            return Err(format!(
                "Route B wrote {} prematurely at step {idx}",
                rb_phrase_wav.display()
            ));
        }
    }
    let rb_wall_ms = t_rb0.elapsed().as_secs_f64() * 1000.0;
    let stats_rb_after = fetch_stats(rs_srv_port)?;

    if !rb_phrase_wav.is_file() {
        return Err(format!(
            "Route B did not produce {}",
            rb_phrase_wav.display()
        ));
    }
    let rb_buf = read_wav(&rb_phrase_wav).map_err(|e| format!("Read Route B wav: {e}"))?;
    let rb_samples = &rb_buf.samples;
    if !rb_samples.iter().all(|v| v.is_finite()) {
        return Err("Route B output contains non-finite samples".to_string());
    }

    let rb_recent = stats_rb_after["recent"]
        .as_array()
        .ok_or("Missing /stats recent array")?;
    let recent0_name = rb_recent
        .first()
        .and_then(|v| v["note_name"].as_str())
        .unwrap_or("");
    let recent0_synth = rb_recent
        .first()
        .and_then(|v| v["synthesis_ms"].as_f64())
        .unwrap_or(0.0);
    if !recent0_name.starts_with("phrase:") {
        return Err(format!(
            "Expected latest /stats entry to be phrase:*, got '{recent0_name}'"
        ));
    }

    let rb_synth_ms = (stats_sum_from(&stats_rb_after, "avg_synthesis_ms")
        - stats_sum_from(&stats_rb_before, "avg_synthesis_ms"))
    .max(0.0);
    let rb_feat_ms = (stats_sum_from(&stats_rb_after, "avg_feature_ms")
        - stats_sum_from(&stats_rb_before, "avg_feature_ms"))
    .max(0.0);
    let rb_post_ms = (stats_sum_from(&stats_rb_after, "avg_postprocess_ms")
        - stats_sum_from(&stats_rb_before, "avg_postprocess_ms"))
    .max(0.0);

    // Verify all `prep:*` entries skipped GPU vocoder (sum delta equals single phrase synthesis_ms)
    if (rb_synth_ms - recent0_synth).abs() > 0.5 {
        return Err(format!(
            "Unexpected GPU vocoder synthesis during prepare: sum_delta={rb_synth_ms:.3}, phrase={recent0_synth:.3}"
        ));
    }

    let mut backfilled_count = 0usize;
    for p in &rb_note_paths {
        let nb = read_wav(p).map_err(|e| format!("Read backfill {}: {e}", p.display()))?;
        let peak = nb.samples.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        if nb.samples.len() > 64 && peak > 0.01 {
            backfilled_count += 1;
        }
    }

    let route_b_metrics = PhraseModeMetrics {
        note_count,
        wall_ms: rb_wall_ms,
        throughput_nps: note_count as f64 * 1000.0 / rb_wall_ms.max(1e-6),
        vocoder_calls: 1,
        server_sum_synthesis_ms: rb_synth_ms,
        server_sum_feature_ms: rb_feat_ms,
        server_sum_postprocess_ms: rb_post_ms,
        output_samples: rb_samples.len(),
        all_finite: true,
        lufs: measure_lufs(rb_samples, 44100).unwrap_or(f64::NEG_INFINITY),
        overlap_rms_ratio: compute_overlap_rms_ratio(rb_samples, note_count, dur_ms, ovr_ms, 44100),
        backfilled_stubs_count: backfilled_count,
    };

    let expected_samples =
        (((note_count - 1) as f64 * (dur_ms - ovr_ms) + dur_ms) * 44.1).round() as usize;
    let wall_speedup = baseline_metrics.wall_ms / route_b_metrics.wall_ms.max(1e-6);
    let vocoder_reduction = baseline_metrics.server_sum_synthesis_ms
        / route_b_metrics.server_sum_synthesis_ms.max(1e-6);
    let sample_count_diff =
        route_b_metrics.output_samples as isize - baseline_metrics.output_samples as isize;
    let lufs_delta_db = route_b_metrics.lufs - baseline_metrics.lufs;
    let base_dip_err = (1.0 - baseline_metrics.overlap_rms_ratio).abs();
    let rb_dip_err = (1.0 - route_b_metrics.overlap_rms_ratio).abs();
    let overlap_continuity_improvement = if rb_dip_err > 1e-6 {
        base_dip_err / rb_dip_err
    } else {
        1.0
    };

    Ok(PhraseScenarioComparison {
        name: name.to_string(),
        note_count,
        dur_ms,
        ovr_ms,
        expected_samples,
        baseline_note_by_note: baseline_metrics,
        melos_route_b: route_b_metrics,
        wall_speedup,
        vocoder_synthesis_ms_reduction: vocoder_reduction,
        sample_count_diff,
        lufs_delta_db,
        overlap_continuity_improvement,
    })
}

fn run_phrase_route_b_benchmark(
    repo: &Path,
    rs_srv_port: u16,
    synth_wav: &Path,
    real_vocal_wav: Option<&Path>,
    work_dir: &Path,
) -> Result<Vec<PhraseScenarioComparison>, String> {
    let hifisampler_exe = repo.join("rust/target/release/hifisampler.exe");
    let melos_exe = repo.join("rust/target/release/melos.exe");
    let melos_wavtool_exe = repo.join("rust/target/release/melos-wavtool.exe");
    if !melos_exe.is_file() || !melos_wavtool_exe.is_file() {
        return Err("Missing melos.exe or melos-wavtool.exe under rust/target/release/".into());
    }

    let pitches_20 = [
        "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "G4", "E4", "C4", "D4", "E4", "F4", "G4",
        "A4", "B4", "C5", "G4", "E4",
    ];
    let mixed_flags = ["", "P100", "g-10", "A30", "Hb120Hv90"];
    let mixed_cmp = run_phrase_scenario(
        "mixed_20note_c4_c5",
        repo,
        rs_srv_port,
        &hifisampler_exe,
        &melos_exe,
        &melos_wavtool_exe,
        synth_wav,
        work_dir,
        &pitches_20,
        &mixed_flags,
        "0",
        "300",
        "0",
        "0",
        250.0,
        50.0,
    )?;

    let mut scenarios = vec![mixed_cmp];
    if let Some(real_wav) = real_vocal_wav {
        let pitches_6 = ["A3", "B3", "C4", "D4", "C4", "A3"];
        let real_cmp = run_phrase_scenario(
            "real_vocal_6note_legato",
            repo,
            rs_srv_port,
            &hifisampler_exe,
            &melos_exe,
            &melos_wavtool_exe,
            real_wav,
            work_dir,
            &pitches_6,
            &[""],
            "1200",
            "450",
            "80",
            "-1200",
            350.0,
            60.0,
        )?;
        scenarios.push(real_cmp);
    }

    Ok(scenarios)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let cli = parse_cli_args(&args)?;
    let rust_only = cli.rust_only;

    let repo = resolve_repo_root(cli.explicit_root.as_deref())?;
    eprintln!(
        "[hifi_bench] Repo root: {} (rust_only={})",
        repo.display(),
        rust_only
    );

    let rs_client = repo.join("rust/target/release/hifisampler.exe");
    let rs_melos = repo.join("rust/target/release/melos.exe");
    let rs_melos_wavtool = repo.join("rust/target/release/melos-wavtool.exe");
    let rs_server_bin = repo.join("rust/target/release/hifiserver.exe");
    let cs_client = repo.join("hifisampler.exe");
    if !rs_client.is_file() || !rs_server_bin.is_file() {
        return Err("Missing release Rust binaries under rust/target/release/".to_string());
    }

    let binary_sizes = BinarySizesReport {
        rust_hifisampler_bytes: fs::metadata(&rs_client).map_err(|e| e.to_string())?.len(),
        rust_melos_bytes: fs::metadata(&rs_melos).ok().map(|m| m.len()),
        rust_melos_wavtool_bytes: fs::metadata(&rs_melos_wavtool).ok().map(|m| m.len()),
        rust_hifiserver_bytes: fs::metadata(&rs_server_bin)
            .map_err(|e| e.to_string())?
            .len(),
        csharp_hifisampler_bytes: fs::metadata(&cs_client).ok().map(|m| m.len()),
    };

    let work_dir = std::env::temp_dir().join(format!("hifi_bench_{}", Uuid::new_v4().simple()));
    fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;
    eprintln!("[hifi_bench] Isolated temp dir: {}", work_dir.display());

    // Generate deterministic harmonic test WAV & copy real vocal WAV if present
    let synth_wav = work_dir.join("bench_harmonic_440.wav");
    generate_harmonic_vocal_wav(&synth_wav, 220.0, 1.0, 44100)?;

    let real_vocal_orig =
        repo.join("TestResults/vocal_fry_demo/isolated_vb_cache/A3__ああいあうえあ.wav");
    let real_vocal_copy = work_dir.join("real_vocal_a3.wav");
    let real_vocal_source = if real_vocal_orig.is_file() {
        fs::copy(&real_vocal_orig, &real_vocal_copy).map_err(|e| e.to_string())?;
        Some(real_vocal_orig.to_string_lossy().into_owned())
    } else {
        None
    };

    if cli.phrase_only {
        eprintln!("[hifi_bench] Running Route B phrase synthesis benchmark (--phrase-only)...");
        let port = find_free_port()?;
        let mut srv = spawn_rs_server(&repo, port, work_dir.join("rs_phrase_only.log"))?;
        srv.wait_ready(Duration::from_secs(60))?;
        let real_opt = real_vocal_source
            .as_ref()
            .map(|_| real_vocal_copy.as_path());
        let scenarios = run_phrase_route_b_benchmark(&repo, port, &synth_wav, real_opt, &work_dir)?;
        srv.shutdown();

        let phrase_report = PhraseComparisonReport {
            repo_root: canonical_path_str(&repo),
            binary_sizes,
            scenarios,
        };
        let json_text = serde_json::to_string_pretty(&phrase_report)
            .map_err(|e| format!("JSON serialize failed: {e}"))?;
        if let Some(out_path) = cli.json_out {
            if let Some(parent) = out_path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&out_path, &json_text).map_err(|e| format!("Write JSON failed: {e}"))?;
        } else {
            println!("{json_text}");
        }
        let _ = fs::remove_dir_all(&work_dir);
        return Ok(());
    }

    // 1. Cold-start latency: HTTP server startups (2 repeats each)
    eprintln!("[hifi_bench] 1/6 Measuring server cold-start latency...");
    let mut py_default_startups = Vec::new();
    let mut py_default_peak_rss = None;
    if !rust_only {
        for r in 0..2 {
            let p = find_free_port()?;
            let mut srv =
                spawn_py_default_server(&repo, p, work_dir.join(format!("py_def_{r}.log")))?;
            let ms = srv.wait_ready(Duration::from_secs(60))?;
            py_default_startups.push(ms);
            if r == 1 {
                // Render one note so default server memory reflects real initialized inference
                let out_tmp = work_dir.join("py_def_mem_probe.wav");
                let src_tmp = work_dir.join("py_def_mem_src.wav");
                fs::copy(&synth_wav, &src_tmp).map_err(|e| e.to_string())?;
                let _ = post_note(
                    p,
                    &note_args(
                        &src_tmp,
                        &out_tmp,
                        &NoteCase {
                            name: "probe",
                            pitch: "A4",
                            velocity: "100",
                            flags: "Ht20",
                            offset: "0",
                            length: "300",
                            consonant: "0",
                            cutoff: "0",
                            tempo: "!120",
                            pitch_string: "AA",
                            use_real_vocal: false,
                        },
                    ),
                )?;
                py_default_peak_rss = srv.peak_rss_bytes();
            }
            srv.shutdown();
        }
    }

    // Measure CLI cold-start (auto-spawn server + 1 note) for C#->Py and Rust->Rust
    eprintln!("[hifi_bench] Measuring CLI single-call cold-start...");
    let rs_cli_cold_ms = {
        let src = work_dir.join("cli_cold_rs_in.wav");
        let out = work_dir.join("cli_cold_rs_out.wav");
        fs::copy(&synth_wav, &src).map_err(|e| e.to_string())?;
        measure_cli_cold(&rs_client, &repo, &src, &out, &work_dir.join("home_cli_rs"))?
    };
    let py_cli_cold_ms = if !rust_only && cs_client.is_file() {
        let src = work_dir.join("cli_cold_py_in.wav");
        let out = work_dir.join("cli_cold_py_out.wav");
        fs::copy(&synth_wav, &src).map_err(|e| e.to_string())?;
        Some(measure_cli_cold(
            &cs_client,
            &repo,
            &src,
            &out,
            &work_dir.join("home_cli_py"),
        )?)
    } else {
        None
    };

    // Start long-lived Python (with hnsep/vr/model.onnx) and Rust servers for stages 2..6
    let mut py_onnx_startups = Vec::new();
    let mut rs_startups = Vec::new();
    {
        if !rust_only {
            let p_py = find_free_port()?;
            let mut s_py = spawn_py_onnx_server(&repo, p_py, work_dir.join("py_onnx_boot0.log"))?;
            py_onnx_startups.push(s_py.wait_ready(Duration::from_secs(60))?);
            s_py.shutdown();
        }

        let p_rs = find_free_port()?;
        let mut s_rs = spawn_rs_server(&repo, p_rs, work_dir.join("rs_boot0.log"))?;
        rs_startups.push(s_rs.wait_ready(Duration::from_secs(60))?);
        s_rs.shutdown();
    }

    let mut py_srv = if !rust_only {
        let py_port = find_free_port()?;
        let mut srv = spawn_py_onnx_server(&repo, py_port, work_dir.join("py_onnx_main.log"))?;
        py_onnx_startups.push(srv.wait_ready(Duration::from_secs(60))?);
        Some(srv)
    } else {
        None
    };
    let py_port_opt = py_srv.as_ref().map(|s| s.port);

    let rs_port = find_free_port()?;
    let mut rs_srv = spawn_rs_server(&repo, rs_port, work_dir.join("rs_main.log"))?;
    rs_startups.push(rs_srv.wait_ready(Duration::from_secs(60))?);

    // Warm CLI single call (5 runs each against live servers)
    let cli_warm_src = work_dir.join("cli_warm_src.wav");
    fs::copy(&synth_wav, &cli_warm_src).map_err(|e| e.to_string())?;
    let rs_cli_warm_dir = work_dir.join("cli_warm_rs");
    fs::create_dir_all(&rs_cli_warm_dir).map_err(|e| e.to_string())?;
    let _ = post_note(
        rs_port,
        &note_args(
            &cli_warm_src,
            &rs_cli_warm_dir.join("prewarm.wav"),
            &NoteCase {
                name: "prewarm",
                pitch: "A4",
                velocity: "100",
                flags: "",
                offset: "0",
                length: "300",
                consonant: "0",
                cutoff: "0",
                tempo: "!120",
                pitch_string: "AA",
                use_real_vocal: false,
            },
        ),
    )?;
    let rs_cli_warm = measure_cli_warm(
        &rs_client,
        &repo,
        rs_port,
        &cli_warm_src,
        &rs_cli_warm_dir,
        &work_dir.join("home_warm_rs"),
        5,
    )?;

    let py_cli_warm = if let (Some(py_port), true) = (py_port_opt, cs_client.is_file()) {
        let py_cli_warm_dir = work_dir.join("cli_warm_py");
        fs::create_dir_all(&py_cli_warm_dir).map_err(|e| e.to_string())?;
        let _ = post_note(
            py_port,
            &note_args(
                &cli_warm_src,
                &py_cli_warm_dir.join("prewarm.wav"),
                &NoteCase {
                    name: "prewarm",
                    pitch: "A4",
                    velocity: "100",
                    flags: "",
                    offset: "0",
                    length: "300",
                    consonant: "0",
                    cutoff: "0",
                    tempo: "!120",
                    pitch_string: "AA",
                    use_real_vocal: false,
                },
            ),
        )?;
        Some(measure_cli_warm(
            &cs_client,
            &repo,
            py_port,
            &cli_warm_src,
            &py_cli_warm_dir,
            &work_dir.join("home_warm_py"),
            5,
        )?)
    } else {
        None
    };

    // 2 & 6. Cold Render (cache miss) stage timings & audio equivalence
    eprintln!("[hifi_bench] 2/6 Measuring cold render stage timings & audio equivalence...");
    let mut cases = vec![
        NoteCase {
            name: "default",
            pitch: "A4",
            velocity: "100",
            flags: "",
            offset: "0",
            length: "400",
            consonant: "0",
            cutoff: "0",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: false,
        },
        NoteCase {
            name: "Hb140Hv90",
            pitch: "A4",
            velocity: "100",
            flags: "Hb140Hv90",
            offset: "0",
            length: "400",
            consonant: "0",
            cutoff: "0",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: false,
        },
        NoteCase {
            name: "Ht50Ho40Hr-30",
            pitch: "A4",
            velocity: "100",
            flags: "Ht50Ho40Hr-30",
            offset: "0",
            length: "400",
            consonant: "0",
            cutoff: "0",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: false,
        },
        NoteCase {
            name: "HG60",
            pitch: "A4",
            velocity: "100",
            flags: "HG60",
            offset: "0",
            length: "400",
            consonant: "0",
            cutoff: "0",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: false,
        },
        NoteCase {
            name: "auto_vocal_fry",
            pitch: "A3",
            velocity: "100",
            flags: "",
            offset: "0",
            length: "400",
            consonant: "0",
            cutoff: "0",
            tempo: "!120",
            pitch_string: "AA#20#iU#60#AA",
            use_real_vocal: false,
        },
        NoteCase {
            name: "P100",
            pitch: "A4",
            velocity: "100",
            flags: "P100",
            offset: "0",
            length: "400",
            consonant: "0",
            cutoff: "0",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: false,
        },
    ];
    if real_vocal_source.is_some() {
        cases.push(NoteCase {
            name: "real_vocal_default",
            pitch: "A3",
            velocity: "100",
            flags: "",
            offset: "1200",
            length: "500",
            consonant: "80",
            cutoff: "-1200",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: true,
        });
        cases.push(NoteCase {
            name: "real_vocal_Ht50Ho40Hr-30P100",
            pitch: "C4",
            velocity: "100",
            flags: "Ht50Ho40Hr-30P100",
            offset: "1200",
            length: "500",
            consonant: "80",
            cutoff: "-1200",
            tempo: "!120",
            pitch_string: "AA",
            use_real_vocal: true,
        });
    }

    let mut cold_cases = Vec::new();
    for c in &cases {
        let rs_dir = work_dir.join("cold_rs").join(c.name);
        fs::create_dir_all(&rs_dir).map_err(|e| e.to_string())?;
        let src_template = if c.use_real_vocal {
            &real_vocal_copy
        } else {
            &synth_wav
        };
        let rs_in = rs_dir.join("input.wav");
        fs::copy(src_template, &rs_in).map_err(|e| e.to_string())?;
        let rs_out = rs_dir.join("out.wav");

        let rs_wall = post_note(rs_port, &note_args(&rs_in, &rs_out, c))?;
        let rs_timing = latest_stage_timing(rs_port, rs_wall)?;

        let (py_opt, eq_opt) = if let Some(py_port) = py_port_opt {
            let py_dir = work_dir.join("cold_py").join(c.name);
            fs::create_dir_all(&py_dir).map_err(|e| e.to_string())?;
            let py_in = py_dir.join("input.wav");
            fs::copy(src_template, &py_in).map_err(|e| e.to_string())?;
            let py_out = py_dir.join("out.wav");
            let py_wall = post_note(py_port, &note_args(&py_in, &py_out, c))?;
            let py_timing = latest_stage_timing(py_port, py_wall)?;
            let equivalence = compare_wav_files(&py_out, &rs_out)?;
            (Some(py_timing), Some(equivalence))
        } else {
            (None, None)
        };

        cold_cases.push(ColdCaseComparison {
            name: c.name.to_string(),
            flags: c.flags.to_string(),
            pitch: c.pitch.to_string(),
            length: c.length.to_string(),
            pitch_string: c.pitch_string.to_string(),
            python: py_opt,
            rust: rs_timing,
            equivalence: eq_opt,
        });
    }

    // 3. Warm render (RAM hit + disk hit) + cross-language .hifi.bin interop
    eprintln!("[hifi_bench] 3/6 Measuring warm render & cross-language .hifi.bin interop...");
    let warm_case = NoteCase {
        name: "warm",
        pitch: "A4",
        velocity: "100",
        flags: "Ht50Ho40Hr-30",
        offset: "0",
        length: "400",
        consonant: "0",
        cutoff: "0",
        tempo: "!120",
        pitch_string: "AA",
        use_real_vocal: false,
    };

    // RAM cache hits using the existing cold_py/cold_rs Ht50Ho40Hr-30 directory
    let py_warm_ram = if let Some(py_port) = py_port_opt {
        let py_ram_in = work_dir.join("cold_py/Ht50Ho40Hr-30/input.wav");
        let py_ram_out = work_dir.join("cold_py/Ht50Ho40Hr-30/out_warm_ram.wav");
        let py_ram_wall = post_note(py_port, &note_args(&py_ram_in, &py_ram_out, &warm_case))?;
        Some(latest_stage_timing(py_port, py_ram_wall)?)
    } else {
        None
    };

    let rs_ram_in = work_dir.join("cold_rs/Ht50Ho40Hr-30/input.wav");
    let rs_ram_out = work_dir.join("cold_rs/Ht50Ho40Hr-30/out_warm_ram.wav");
    let rs_ram_wall = post_note(rs_port, &note_args(&rs_ram_in, &rs_ram_out, &warm_case))?;
    let rs_warm_ram = latest_stage_timing(rs_port, rs_ram_wall)?;

    // Disk cache hits using fresh servers (no L1 RAM entry) on a pre-cached directory
    let rs_disk_dir = work_dir.join("warm_disk_rs");
    fs::create_dir_all(&rs_disk_dir).map_err(|e| e.to_string())?;
    let rs_disk_in = rs_disk_dir.join("input.wav");
    fs::copy(&synth_wav, &rs_disk_in).map_err(|e| e.to_string())?;
    let _ = post_note(
        rs_port,
        &note_args(&rs_disk_in, &rs_disk_dir.join("seed.wav"), &warm_case),
    )?;

    let rs_disk_srv_port = find_free_port()?;
    let mut rs_disk_srv = spawn_rs_server(&repo, rs_disk_srv_port, work_dir.join("rs_disk.log"))?;
    let _ = rs_disk_srv.wait_ready(Duration::from_secs(60))?;
    let rs_disk_wall = post_note(
        rs_disk_srv_port,
        &note_args(&rs_disk_in, &rs_disk_dir.join("warm.wav"), &warm_case),
    )?;
    let rs_warm_disk = latest_stage_timing(rs_disk_srv_port, rs_disk_wall)?;

    let (py_warm_disk, py_to_rs_default, py_to_rs_fx, rs_to_py_default, rs_to_py_fx) =
        if let Some(py_port) = py_port_opt {
            let py_disk_dir = work_dir.join("warm_disk_py");
            fs::create_dir_all(&py_disk_dir).map_err(|e| e.to_string())?;
            let py_disk_in = py_disk_dir.join("input.wav");
            fs::copy(&synth_wav, &py_disk_in).map_err(|e| e.to_string())?;
            let _ = post_note(
                py_port,
                &note_args(&py_disk_in, &py_disk_dir.join("seed.wav"), &warm_case),
            )?;

            let py_disk_srv_port = find_free_port()?;
            let mut py_disk_srv =
                spawn_py_onnx_server(&repo, py_disk_srv_port, work_dir.join("py_disk.log"))?;
            let _ = py_disk_srv.wait_ready(Duration::from_secs(60))?;
            let py_disk_wall = post_note(
                py_disk_srv_port,
                &note_args(&py_disk_in, &py_disk_dir.join("warm.wav"), &warm_case),
            )?;
            let py_warm_disk = latest_stage_timing(py_disk_srv_port, py_disk_wall)?;

            let interop_dir = work_dir.join("interop");
            let py_to_rs_default = run_interop_check(
                py_port,
                rs_disk_srv_port,
                &synth_wav,
                &interop_dir,
                "py_to_rs_def",
                "",
            )?;
            let py_to_rs_fx = run_interop_check(
                py_port,
                rs_disk_srv_port,
                &synth_wav,
                &interop_dir,
                "py_to_rs_fx",
                "Ht50Ho40Hr-30",
            )?;
            let rs_to_py_default = run_interop_check(
                rs_port,
                py_disk_srv_port,
                &synth_wav,
                &interop_dir,
                "rs_to_py_def",
                "",
            )?;
            let rs_to_py_fx = run_interop_check(
                rs_port,
                py_disk_srv_port,
                &synth_wav,
                &interop_dir,
                "rs_to_py_fx",
                "Ht50Ho40Hr-30",
            )?;
            py_disk_srv.shutdown();
            (
                Some(py_warm_disk),
                Some(py_to_rs_default),
                Some(py_to_rs_fx),
                Some(rs_to_py_default),
                Some(rs_to_py_fx),
            )
        } else {
            (None, None, None, None, None)
        };

    rs_disk_srv.shutdown();

    let warm_and_interop = WarmAndInteropResult {
        py_warm_disk,
        py_warm_ram,
        rs_warm_disk,
        rs_warm_ram,
        py_to_rs_default,
        py_to_rs_fx,
        rs_to_py_default,
        rs_to_py_fx,
    };

    let prewarm_case = NoteCase {
        name: "pw",
        pitch: "C4",
        velocity: "100",
        flags: "",
        offset: "0",
        length: "250",
        consonant: "0",
        cutoff: "0",
        tempo: "!120",
        pitch_string: "AA",
        use_real_vocal: false,
    };

    // 4. Multi-note batch throughput (sequential + 2-worker concurrent phrase bursts)
    eprintln!("[hifi_bench] 4/6 Measuring 20-note sequential & 2-worker concurrent bursts...");
    let mix_rs_dir = work_dir.join("batch_mix_rs");
    fs::create_dir_all(&mix_rs_dir).map_err(|e| e.to_string())?;
    let mix_rs_in = mix_rs_dir.join("batch_src.wav");
    fs::copy(&synth_wav, &mix_rs_in).map_err(|e| e.to_string())?;
    let mixed_batch_rs = run_batch(rs_port, &mix_rs_in, &mix_rs_dir, true, 1)?;

    let warm_rs_dir = work_dir.join("batch_warm_rs");
    fs::create_dir_all(&warm_rs_dir).map_err(|e| e.to_string())?;
    let warm_rs_in = warm_rs_dir.join("batch_src.wav");
    fs::copy(&synth_wav, &warm_rs_in).map_err(|e| e.to_string())?;
    let _ = post_note(
        rs_port,
        &note_args(&warm_rs_in, &warm_rs_dir.join("prewarm.wav"), &prewarm_case),
    )?;
    let warm_batch_rs = run_batch(rs_port, &warm_rs_in, &warm_rs_dir, false, 1)?;

    let cmix_rs_dir = work_dir.join("batch_concurrent_mix_rs");
    fs::create_dir_all(&cmix_rs_dir).map_err(|e| e.to_string())?;
    let cmix_rs_in = cmix_rs_dir.join("batch_src.wav");
    fs::copy(&synth_wav, &cmix_rs_in).map_err(|e| e.to_string())?;
    let concurrent_mixed_batch_rs = run_batch(rs_port, &cmix_rs_in, &cmix_rs_dir, true, 2)?;

    let cwarm_rs_dir = work_dir.join("batch_concurrent_warm_rs");
    fs::create_dir_all(&cwarm_rs_dir).map_err(|e| e.to_string())?;
    let cwarm_rs_in = cwarm_rs_dir.join("batch_src.wav");
    fs::copy(&synth_wav, &cwarm_rs_in).map_err(|e| e.to_string())?;
    let _ = post_note(
        rs_port,
        &note_args(
            &cwarm_rs_in,
            &cwarm_rs_dir.join("prewarm.wav"),
            &prewarm_case,
        ),
    )?;
    let concurrent_warm_batch_rs = run_batch(rs_port, &cwarm_rs_in, &cwarm_rs_dir, false, 2)?;

    let (mixed_batch_py, warm_batch_py, concurrent_mixed_batch_py, concurrent_warm_batch_py) =
        if let Some(py_port) = py_port_opt {
            let mix_py_dir = work_dir.join("batch_mix_py");
            fs::create_dir_all(&mix_py_dir).map_err(|e| e.to_string())?;
            let mix_py_in = mix_py_dir.join("batch_src.wav");
            fs::copy(&synth_wav, &mix_py_in).map_err(|e| e.to_string())?;
            let seq_mix = run_batch(py_port, &mix_py_in, &mix_py_dir, true, 1)?;

            let warm_py_dir = work_dir.join("batch_warm_py");
            fs::create_dir_all(&warm_py_dir).map_err(|e| e.to_string())?;
            let warm_py_in = warm_py_dir.join("batch_src.wav");
            fs::copy(&synth_wav, &warm_py_in).map_err(|e| e.to_string())?;
            let _ = post_note(
                py_port,
                &note_args(&warm_py_in, &warm_py_dir.join("prewarm.wav"), &prewarm_case),
            )?;
            let seq_warm = run_batch(py_port, &warm_py_in, &warm_py_dir, false, 1)?;

            let cmix_py_dir = work_dir.join("batch_concurrent_mix_py");
            fs::create_dir_all(&cmix_py_dir).map_err(|e| e.to_string())?;
            let cmix_py_in = cmix_py_dir.join("batch_src.wav");
            fs::copy(&synth_wav, &cmix_py_in).map_err(|e| e.to_string())?;
            let conc_mix = run_batch(py_port, &cmix_py_in, &cmix_py_dir, true, 2)?;

            let cwarm_py_dir = work_dir.join("batch_concurrent_warm_py");
            fs::create_dir_all(&cwarm_py_dir).map_err(|e| e.to_string())?;
            let cwarm_py_in = cwarm_py_dir.join("batch_src.wav");
            fs::copy(&synth_wav, &cwarm_py_in).map_err(|e| e.to_string())?;
            let _ = post_note(
                py_port,
                &note_args(
                    &cwarm_py_in,
                    &cwarm_py_dir.join("prewarm.wav"),
                    &prewarm_case,
                ),
            )?;
            let conc_warm = run_batch(py_port, &cwarm_py_in, &cwarm_py_dir, false, 2)?;

            (
                Some(seq_mix),
                Some(seq_warm),
                Some(conc_mix),
                Some(conc_warm),
            )
        } else {
            (None, None, None, None)
        };

    // 5. Route B OpenUtau Phrase Synthesis Comparison + Peak memory RSS
    let phrase_route_b = if rs_melos.is_file() && rs_melos_wavtool.is_file() {
        eprintln!("[hifi_bench] Running Route B phrase synthesis benchmark...");
        let port = find_free_port()?;
        let mut srv = spawn_rs_server(&repo, port, work_dir.join("rs_phrase_main.log"))?;
        srv.wait_ready(Duration::from_secs(60))?;
        let real_opt = real_vocal_source
            .as_ref()
            .map(|_| real_vocal_copy.as_path());
        let res = run_phrase_route_b_benchmark(&repo, port, &synth_wav, real_opt, &work_dir)?;
        srv.shutdown();
        Some(res)
    } else {
        None
    };

    eprintln!("[hifi_bench] 5/6 Capturing peak RSS...");
    let py_onnx_peak_rss = py_srv.as_ref().and_then(|s| s.peak_rss_bytes());
    let rs_server_peak_rss = rs_srv.peak_rss_bytes();
    if let Some(ref mut srv) = py_srv {
        srv.shutdown();
    }
    rs_srv.shutdown();

    let report = FullBenchmarkReport {
        repo_root: canonical_path_str(&repo),
        real_vocal_source,
        binary_sizes,
        cold_start: ColdStartReport {
            py_default_server_startup_ms: py_default_startups,
            py_onnx_server_startup_ms: py_onnx_startups,
            rs_server_startup_ms: rs_startups,
            py_cli_cold_ms,
            rs_cli_cold_ms,
            py_cli_warm,
            rs_cli_warm,
        },
        cold_cases,
        warm_and_interop,
        mixed_batch_py,
        mixed_batch_rs,
        warm_batch_py,
        warm_batch_rs,
        concurrent_mixed_batch_py,
        concurrent_mixed_batch_rs,
        concurrent_warm_batch_py,
        concurrent_warm_batch_rs,
        phrase_route_b,
        memory: MemoryReport {
            py_default_peak_rss_bytes: py_default_peak_rss,
            py_onnx_peak_rss_bytes: py_onnx_peak_rss,
            rs_server_peak_rss_bytes: rs_server_peak_rss,
        },
    };

    let json_text =
        serde_json::to_string_pretty(&report).map_err(|e| format!("JSON serialize failed: {e}"))?;
    if let Some(out_path) = cli.json_out {
        if let Some(parent) = out_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(&out_path, &json_text).map_err(|e| format!("Write JSON failed: {e}"))?;
        eprintln!("[hifi_bench] Wrote JSON report to {}", out_path.display());
    } else {
        println!("{json_text}");
    }

    let _ = fs::remove_dir_all(&work_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_args_parse_flags_cleanly() {
        let args = vec![
            "hifi_bench".to_string(),
            "--rust-only".to_string(),
            "--root".to_string(),
            "C:/Project/hifi".to_string(),
            "--json-out".to_string(),
            "out.json".to_string(),
        ];
        let parsed = parse_cli_args(&args).expect("should parse valid args");
        assert!(parsed.rust_only);
        assert_eq!(parsed.explicit_root, Some(PathBuf::from("C:/Project/hifi")));
        assert_eq!(parsed.json_out, Some(PathBuf::from("out.json")));
    }

    #[test]
    fn batch_jobs_and_partitioning_cover_all_twenty_notes() {
        let jobs = build_batch_jobs(Path::new("in.wav"), Path::new("out_dir"), true);
        assert_eq!(jobs.len(), 20);
        let buckets = partition_job_indices(jobs.len(), 2);
        assert_eq!(buckets.len(), 2);
        assert_eq!(buckets[0].len(), 10);
        assert_eq!(buckets[1].len(), 10);
        let mut seen: Vec<usize> = buckets.into_iter().flatten().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..20).collect::<Vec<usize>>());
    }
}
