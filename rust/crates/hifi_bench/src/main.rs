use hifi_bench::{
    compare_wav_files, find_free_port, generate_harmonic_vocal_wav, http_request_loopback,
    inspect_hifibin, process_peak_rss_bytes, process_tree_peak_rss_bytes, AudioEquivalenceReport,
    HifiBinSummary, LatencySummary,
};
use hifi_core::utau::parse_flags;
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
    python: StageTiming,
    rust: StageTiming,
    equivalence: AudioEquivalenceReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WarmAndInteropResult {
    py_warm_disk: StageTiming,
    py_warm_ram: StageTiming,
    rs_warm_disk: StageTiming,
    rs_warm_ram: StageTiming,
    py_to_rs_default: InteropCheck,
    py_to_rs_fx: InteropCheck,
    rs_to_py_default: InteropCheck,
    rs_to_py_fx: InteropCheck,
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
    py_cli_cold_ms: f64,
    rs_cli_cold_ms: f64,
    py_cli_warm: LatencySummary,
    rs_cli_warm: LatencySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinarySizesReport {
    rust_hifisampler_bytes: u64,
    rust_hifiserver_bytes: u64,
    csharp_hifisampler_bytes: Option<u64>,
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
    mixed_batch_py: BatchThroughputResult,
    mixed_batch_rs: BatchThroughputResult,
    warm_batch_py: BatchThroughputResult,
    warm_batch_rs: BatchThroughputResult,
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

fn run_batch(
    port: u16,
    in_wav: &Path,
    out_dir: &Path,
    mixed: bool,
) -> Result<BatchThroughputResult, String> {
    let pitches = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "G4", "E4"];
    let flag_sets = if mixed {
        vec!["", "P100", "g-10", "A30", "Hb120Hv90"]
    } else {
        vec![""]
    };
    let lengths = ["220", "280", "350", "400"];
    let total_notes = 20usize;

    let stats_before = fetch_stats(port)?;
    let req_before = stats_before["total_requests"].as_u64().unwrap_or(0);
    let hits_before = stats_before["cache_hits"].as_u64().unwrap_or(0);

    let mut client_times = Vec::with_capacity(total_notes);
    let wall_t0 = Instant::now();
    for i in 0..total_notes {
        let out_wav = out_dir.join(format!("batch_{i}.wav"));
        let pitch = pitches[i % pitches.len()];
        let flags = flag_sets[i % flag_sets.len()];
        let length = lengths[i % lengths.len()];
        let args = vec![
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
        ];
        let ms = post_note(port, &args)?;
        client_times.push(ms);
    }
    let wall_ms = wall_t0.elapsed().as_secs_f64() * 1000.0;
    let stats_after = fetch_stats(port)?;
    let req_delta = stats_after["total_requests"].as_u64().unwrap_or(0) - req_before;
    let hits_delta = stats_after["cache_hits"].as_u64().unwrap_or(0) - hits_before;
    let hit_rate = if req_delta > 0 {
        hits_delta as f64 / req_delta as f64
    } else {
        0.0
    };

    let mut srv_totals: Vec<f64> = stats_after["recent"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .take(total_notes)
                .filter_map(|e| e["total_ms"].as_f64())
                .collect()
        })
        .unwrap_or_default();
    let batch_srv_summary = LatencySummary::from_samples(&srv_totals);
    srv_totals.clear();

    Ok(BatchThroughputResult {
        note_count: total_notes,
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

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut explicit_root: Option<PathBuf> = None;
    let mut json_out: Option<PathBuf> = None;

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
            other => return Err(format!("Unknown argument: {other}")),
        }
        i += 1;
    }

    let repo = resolve_repo_root(explicit_root.as_deref())?;
    eprintln!("[hifi_bench] Repo root: {}", repo.display());

    let rs_client = repo.join("rust/target/release/hifisampler.exe");
    let rs_server_bin = repo.join("rust/target/release/hifiserver.exe");
    let cs_client = repo.join("hifisampler.exe");
    if !rs_client.is_file() || !rs_server_bin.is_file() {
        return Err("Missing release Rust binaries under rust/target/release/".to_string());
    }

    let binary_sizes = BinarySizesReport {
        rust_hifisampler_bytes: fs::metadata(&rs_client).map_err(|e| e.to_string())?.len(),
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

    // 1. Cold-start latency: HTTP server startups (2 repeats each)
    eprintln!("[hifi_bench] 1/6 Measuring server cold-start latency...");
    let mut py_default_startups = Vec::new();
    let mut py_default_peak_rss = None;
    for r in 0..2 {
        let p = find_free_port()?;
        let mut srv = spawn_py_default_server(&repo, p, work_dir.join(format!("py_def_{r}.log")))?;
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

    // Measure CLI cold-start (auto-spawn server + 1 note) for C#->Py and Rust->Rust
    eprintln!("[hifi_bench] Measuring CLI single-call cold-start...");
    let rs_cli_cold_ms = {
        let src = work_dir.join("cli_cold_rs_in.wav");
        let out = work_dir.join("cli_cold_rs_out.wav");
        fs::copy(&synth_wav, &src).map_err(|e| e.to_string())?;
        measure_cli_cold(&rs_client, &repo, &src, &out, &work_dir.join("home_cli_rs"))?
    };
    let py_cli_cold_ms = if cs_client.is_file() {
        let src = work_dir.join("cli_cold_py_in.wav");
        let out = work_dir.join("cli_cold_py_out.wav");
        fs::copy(&synth_wav, &src).map_err(|e| e.to_string())?;
        measure_cli_cold(&cs_client, &repo, &src, &out, &work_dir.join("home_cli_py"))?
    } else {
        0.0
    };

    // Start long-lived Python (with hnsep/vr/model.onnx) and Rust servers for stages 2..6
    let mut py_onnx_startups = Vec::new();
    let mut rs_startups = Vec::new();
    {
        let p_py = find_free_port()?;
        let mut s_py = spawn_py_onnx_server(&repo, p_py, work_dir.join("py_onnx_boot0.log"))?;
        py_onnx_startups.push(s_py.wait_ready(Duration::from_secs(60))?);
        s_py.shutdown();

        let p_rs = find_free_port()?;
        let mut s_rs = spawn_rs_server(&repo, p_rs, work_dir.join("rs_boot0.log"))?;
        rs_startups.push(s_rs.wait_ready(Duration::from_secs(60))?);
        s_rs.shutdown();
    }

    let py_port = find_free_port()?;
    let mut py_srv = spawn_py_onnx_server(&repo, py_port, work_dir.join("py_onnx_main.log"))?;
    py_onnx_startups.push(py_srv.wait_ready(Duration::from_secs(60))?);

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

    let py_cli_warm = if cs_client.is_file() {
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
        measure_cli_warm(
            &cs_client,
            &repo,
            py_port,
            &cli_warm_src,
            &py_cli_warm_dir,
            &work_dir.join("home_warm_py"),
            5,
        )?
    } else {
        LatencySummary::from_samples(&[])
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
        let py_dir = work_dir.join("cold_py").join(c.name);
        let rs_dir = work_dir.join("cold_rs").join(c.name);
        fs::create_dir_all(&py_dir).map_err(|e| e.to_string())?;
        fs::create_dir_all(&rs_dir).map_err(|e| e.to_string())?;

        let src_template = if c.use_real_vocal {
            &real_vocal_copy
        } else {
            &synth_wav
        };
        let py_in = py_dir.join("input.wav");
        let rs_in = rs_dir.join("input.wav");
        fs::copy(src_template, &py_in).map_err(|e| e.to_string())?;
        fs::copy(src_template, &rs_in).map_err(|e| e.to_string())?;

        let py_out = py_dir.join("out.wav");
        let rs_out = rs_dir.join("out.wav");

        let py_wall = post_note(py_port, &note_args(&py_in, &py_out, c))?;
        let py_timing = latest_stage_timing(py_port, py_wall)?;

        let rs_wall = post_note(rs_port, &note_args(&rs_in, &rs_out, c))?;
        let rs_timing = latest_stage_timing(rs_port, rs_wall)?;

        let equivalence = compare_wav_files(&py_out, &rs_out)?;
        cold_cases.push(ColdCaseComparison {
            name: c.name.to_string(),
            flags: c.flags.to_string(),
            pitch: c.pitch.to_string(),
            length: c.length.to_string(),
            pitch_string: c.pitch_string.to_string(),
            python: py_timing,
            rust: rs_timing,
            equivalence,
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
    let py_ram_in = work_dir.join("cold_py/Ht50Ho40Hr-30/input.wav");
    let py_ram_out = work_dir.join("cold_py/Ht50Ho40Hr-30/out_warm_ram.wav");
    let py_ram_wall = post_note(py_port, &note_args(&py_ram_in, &py_ram_out, &warm_case))?;
    let py_warm_ram = latest_stage_timing(py_port, py_ram_wall)?;

    let rs_ram_in = work_dir.join("cold_rs/Ht50Ho40Hr-30/input.wav");
    let rs_ram_out = work_dir.join("cold_rs/Ht50Ho40Hr-30/out_warm_ram.wav");
    let rs_ram_wall = post_note(rs_port, &note_args(&rs_ram_in, &rs_ram_out, &warm_case))?;
    let rs_warm_ram = latest_stage_timing(rs_port, rs_ram_wall)?;

    // Disk cache hits using fresh servers (no L1 RAM entry) on a pre-cached directory
    let py_disk_dir = work_dir.join("warm_disk_py");
    let rs_disk_dir = work_dir.join("warm_disk_rs");
    fs::create_dir_all(&py_disk_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&rs_disk_dir).map_err(|e| e.to_string())?;
    let py_disk_in = py_disk_dir.join("input.wav");
    let rs_disk_in = rs_disk_dir.join("input.wav");
    fs::copy(&synth_wav, &py_disk_in).map_err(|e| e.to_string())?;
    fs::copy(&synth_wav, &rs_disk_in).map_err(|e| e.to_string())?;
    let _ = post_note(
        py_port,
        &note_args(&py_disk_in, &py_disk_dir.join("seed.wav"), &warm_case),
    )?;
    let _ = post_note(
        rs_port,
        &note_args(&rs_disk_in, &rs_disk_dir.join("seed.wav"), &warm_case),
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

    let rs_disk_srv_port = find_free_port()?;
    let mut rs_disk_srv = spawn_rs_server(&repo, rs_disk_srv_port, work_dir.join("rs_disk.log"))?;
    let _ = rs_disk_srv.wait_ready(Duration::from_secs(60))?;
    let rs_disk_wall = post_note(
        rs_disk_srv_port,
        &note_args(&rs_disk_in, &rs_disk_dir.join("warm.wav"), &warm_case),
    )?;
    let rs_warm_disk = latest_stage_timing(rs_disk_srv_port, rs_disk_wall)?;

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

    // 4. Multi-note batch throughput (mixed cold+warm & pure warm)
    eprintln!("[hifi_bench] 4/6 Measuring 20-note batch throughput...");
    let mix_py_dir = work_dir.join("batch_mix_py");
    let mix_rs_dir = work_dir.join("batch_mix_rs");
    fs::create_dir_all(&mix_py_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&mix_rs_dir).map_err(|e| e.to_string())?;
    let mix_py_in = mix_py_dir.join("batch_src.wav");
    let mix_rs_in = mix_rs_dir.join("batch_src.wav");
    fs::copy(&synth_wav, &mix_py_in).map_err(|e| e.to_string())?;
    fs::copy(&synth_wav, &mix_rs_in).map_err(|e| e.to_string())?;
    let mixed_batch_py = run_batch(py_port, &mix_py_in, &mix_py_dir, true)?;
    let mixed_batch_rs = run_batch(rs_port, &mix_rs_in, &mix_rs_dir, true)?;

    let warm_py_dir = work_dir.join("batch_warm_py");
    let warm_rs_dir = work_dir.join("batch_warm_rs");
    fs::create_dir_all(&warm_py_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&warm_rs_dir).map_err(|e| e.to_string())?;
    let warm_py_in = warm_py_dir.join("batch_src.wav");
    let warm_rs_in = warm_rs_dir.join("batch_src.wav");
    fs::copy(&synth_wav, &warm_py_in).map_err(|e| e.to_string())?;
    fs::copy(&synth_wav, &warm_rs_in).map_err(|e| e.to_string())?;
    // Pre-warm default cache
    let _ = post_note(
        py_port,
        &note_args(
            &warm_py_in,
            &warm_py_dir.join("prewarm.wav"),
            &NoteCase {
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
            },
        ),
    )?;
    let _ = post_note(
        rs_port,
        &note_args(
            &warm_rs_in,
            &warm_rs_dir.join("prewarm.wav"),
            &NoteCase {
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
            },
        ),
    )?;
    let warm_batch_py = run_batch(py_port, &warm_py_in, &warm_py_dir, false)?;
    let warm_batch_rs = run_batch(rs_port, &warm_rs_in, &warm_rs_dir, false)?;

    // 5. Peak memory RSS
    eprintln!("[hifi_bench] 5/6 Capturing peak RSS...");
    let py_onnx_peak_rss = py_srv.peak_rss_bytes();
    let rs_server_peak_rss = rs_srv.peak_rss_bytes();
    py_srv.shutdown();
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
        memory: MemoryReport {
            py_default_peak_rss_bytes: py_default_peak_rss,
            py_onnx_peak_rss_bytes: py_onnx_peak_rss,
            rs_server_peak_rss_bytes: rs_server_peak_rss,
        },
    };

    let json_text =
        serde_json::to_string_pretty(&report).map_err(|e| format!("JSON serialize failed: {e}"))?;
    if let Some(out_path) = json_out {
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
