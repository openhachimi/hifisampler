use serde_yaml::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Execution provider device preference configured in YAML.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    DirectMl,
    CoreMl,
}

impl DevicePreference {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "rocm" | "hip" => Self::Rocm,
            "dml" | "directml" => Self::DirectMl,
            "coreml" | "mps" => Self::CoreMl,
            other => {
                tracing::warn!("Unknown device preference '{other}', falling back to Auto");
                Self::Auto
            }
        }
    }
}

/// Worker pool size configuration (`int` or `"auto"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxWorkersConfig {
    Auto,
    Fixed(usize),
}

impl MaxWorkersConfig {
    pub fn resolve(&self, directml_active: bool) -> usize {
        if directml_active {
            return 1;
        }
        self.resolve_http_workers()
    }

    /// Resolve HTTP inference worker concurrency decoupled from GPU kernel serialization.
    ///
    /// GPU inference sessions synchronize internally via `OrtRuntime::lock_gpu()`, allowing
    /// multiple HTTP workers to overlap CPU DSP and disk cache work even when DirectML is active.
    pub fn resolve_http_workers(&self) -> usize {
        match self {
            Self::Auto => physical_cpu_count().clamp(1, 4),
            Self::Fixed(n) => (*n).max(1),
        }
    }
}

pub fn physical_cpu_count() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    (logical / 2).max(1)
}

/// Runtime server & DSP configuration loaded from `config.yaml` / `config.default.yaml`.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub sample_rate: u32,
    pub win_size: usize,
    pub hop_size: usize,
    pub origin_hop_size: usize,
    pub n_mels: usize,
    pub n_fft: usize,
    pub mel_fmin: f32,
    pub mel_fmax: f32,
    pub fill: usize,
    pub vocoder_path: String,
    pub model_type: String,
    pub hnsep_model_path: String,
    pub wave_norm: bool,
    pub trim_silence: bool,
    pub silence_threshold: f32,
    pub loop_mode: bool,
    pub auto_vocal_fry: bool,
    pub peak_limit: f32,
    pub max_workers: MaxWorkersConfig,
    pub device: String,
    pub device_preference: DevicePreference,
    pub fp16: bool,
    pub idle_timeout: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            win_size: 2048,
            hop_size: 512,
            origin_hop_size: 128,
            n_mels: 128,
            n_fft: 2048,
            mel_fmin: 40.0,
            mel_fmax: 16000.0,
            fill: 6,
            vocoder_path: "./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx".to_string(),
            model_type: "onnx".to_string(),
            hnsep_model_path: "./hnsep/vr/model_fp16.onnx".to_string(),
            wave_norm: true,
            trim_silence: true,
            silence_threshold: -52.0,
            loop_mode: true,
            auto_vocal_fry: true,
            peak_limit: 1.0,
            max_workers: MaxWorkersConfig::Fixed(2),
            device: "auto".to_string(),
            device_preference: DevicePreference::Auto,
            fp16: false,
            idle_timeout: 0,
        }
    }
}

impl ServerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.sample_rate == 0 {
            return Err("Config 'sample_rate' must be > 0".into());
        }
        if self.win_size == 0 {
            return Err("Config 'win_size' must be > 0".into());
        }
        if self.hop_size == 0 {
            return Err("Config 'hop_size' must be > 0".into());
        }
        if self.origin_hop_size == 0 {
            return Err("Config 'origin_hop_size' must be > 0".into());
        }
        if self.n_mels == 0 {
            return Err("Config 'n_mels' must be > 0".into());
        }
        if self.n_fft == 0 {
            return Err("Config 'n_fft' must be > 0".into());
        }
        Ok(())
    }

    pub fn apply_yaml_str(&mut self, yaml_text: &str) -> Result<(), String> {
        let doc: Value = serde_yaml::from_str(yaml_text)
            .map_err(|e| format!("Failed to parse YAML config: {e}"))?;
        let flat = flatten_yaml_mapping(&doc);
        self.apply_flat_map(&flat);
        self.device_preference = DevicePreference::parse(&self.device);
        self.validate()
    }

    fn apply_flat_map(&mut self, flat: &HashMap<String, Value>) {
        if let Some(v) = get_u64(flat, "sample_rate") {
            self.sample_rate = v as u32;
        }
        if let Some(v) = get_u64(flat, "win_size") {
            self.win_size = v as usize;
        }
        if let Some(v) = get_u64(flat, "hop_size") {
            self.hop_size = v as usize;
        }
        if let Some(v) = get_u64(flat, "origin_hop_size") {
            self.origin_hop_size = v as usize;
        }
        if let Some(v) = get_u64(flat, "n_mels") {
            self.n_mels = v as usize;
        }
        if let Some(v) = get_u64(flat, "n_fft") {
            self.n_fft = v as usize;
        }
        if let Some(v) = get_f64(flat, "mel_fmin") {
            self.mel_fmin = v as f32;
        }
        if let Some(v) = get_f64(flat, "mel_fmax") {
            self.mel_fmax = v as f32;
        }
        if let Some(v) = get_u64(flat, "fill") {
            self.fill = v as usize;
        }
        if let Some(v) = get_str(flat, "vocoder_path") {
            self.vocoder_path = v;
        }
        if let Some(v) = get_str(flat, "model_type") {
            self.model_type = v;
        }
        if let Some(v) = get_str(flat, "hnsep_model_path") {
            self.hnsep_model_path = v;
        }
        if let Some(v) = get_bool(flat, "wave_norm") {
            self.wave_norm = v;
        }
        if let Some(v) = get_bool(flat, "trim_silence") {
            self.trim_silence = v;
        }
        if let Some(v) = get_f64(flat, "silence_threshold") {
            self.silence_threshold = v as f32;
        }
        if let Some(v) = get_bool(flat, "loop_mode") {
            self.loop_mode = v;
        }
        if let Some(v) = get_bool(flat, "auto_vocal_fry") {
            self.auto_vocal_fry = v;
        }
        if let Some(v) = get_f64(flat, "peak_limit") {
            self.peak_limit = v as f32;
        }
        if let Some(val) = flat.get("max_workers") {
            match val {
                Value::String(s) if s.trim().eq_ignore_ascii_case("auto") => {
                    self.max_workers = MaxWorkersConfig::Auto;
                }
                Value::String(s) => {
                    if let Ok(n) = s.trim().parse::<usize>() {
                        self.max_workers = MaxWorkersConfig::Fixed(n.max(1));
                    }
                }
                Value::Number(n) => {
                    if let Some(u) = n.as_u64() {
                        self.max_workers = MaxWorkersConfig::Fixed((u as usize).max(1));
                    }
                }
                _ => {}
            }
        }
        if let Some(v) = get_str(flat, "device") {
            self.device = v;
        }
        if let Some(v) = get_bool(flat, "fp16") {
            self.fp16 = v;
        }
        if let Some(v) = get_u64(flat, "idle_timeout") {
            self.idle_timeout = v;
        }
    }
}

fn flatten_yaml_mapping(doc: &Value) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    if let Value::Mapping(map) = doc {
        for (k, v) in map {
            let Some(key_str) = k.as_str() else { continue };
            if let Value::Mapping(sub) = v {
                for (sk, sv) in sub {
                    if let Some(sk_str) = sk.as_str() {
                        out.insert(sk_str.to_string(), sv.clone());
                    }
                }
            } else {
                out.insert(key_str.to_string(), v.clone());
            }
        }
    }
    out
}

fn get_str(map: &HashMap<String, Value>, key: &str) -> Option<String> {
    match map.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn get_bool(map: &HashMap<String, Value>, key: &str) -> Option<bool> {
    match map.get(key)? {
        Value::Bool(b) => Some(*b),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Some(true),
            "false" | "0" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn get_u64(map: &HashMap<String, Value>, key: &str) -> Option<u64> {
    match map.get(key)? {
        Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_i64().filter(|&x| x >= 0).map(|x| x as u64)),
        Value::String(s) => s.trim().parse::<u64>().ok(),
        _ => None,
    }
}

fn get_f64(map: &HashMap<String, Value>, key: &str) -> Option<f64> {
    match map.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let s = path.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            return PathBuf::from(format!(r"\\{rest}"));
        }
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return PathBuf::from(rest);
        }
    }
    path
}

fn canonicalize_clean(path: &Path) -> PathBuf {
    let p = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    strip_verbatim_prefix(p)
}

/// Resolve the server root directory containing `config.yaml` / `config.default.yaml` / models.
pub fn resolve_server_root(explicit_config: Option<&Path>) -> PathBuf {
    let has_assets = |dir: &Path| -> bool {
        dir.join("config.yaml").is_file()
            || dir.join("config.default.yaml").is_file()
            || dir.join("hifiserver.py").is_file()
            || dir
                .join("pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
                .is_dir()
    };

    for env_key in ["MELOS_HOME", "HIFISAMPLER_HOME"] {
        if let Ok(home) = std::env::var(env_key) {
            let p = PathBuf::from(home.trim());
            if !home.trim().is_empty() && p.is_dir() {
                return canonicalize_clean(&p);
            }
        }
    }

    if let Some(cfg_path) = explicit_config {
        if let Some(parent) = cfg_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            if has_assets(parent) {
                return canonicalize_clean(parent);
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            if has_assets(exe_dir) {
                return canonicalize_clean(exe_dir);
            }
            let mut cursor = exe_dir.parent();
            for _ in 0..4 {
                if let Some(anc) = cursor {
                    if has_assets(anc) {
                        return canonicalize_clean(anc);
                    }
                    cursor = anc.parent();
                } else {
                    break;
                }
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if has_assets(&cwd) {
            return canonicalize_clean(&cwd);
        }
        return strip_verbatim_prefix(cwd);
    }

    PathBuf::from(".")
}

/// Load `ServerConfig` read-only (`config.default.yaml` base + `config.yaml` or `--config` overlay).
/// Never modifies `config.yaml` on disk.
pub fn load_server_config(
    server_root: &Path,
    explicit_config: Option<&Path>,
) -> Result<ServerConfig, String> {
    let mut cfg = ServerConfig::default();

    let default_path = server_root.join("config.default.yaml");
    if default_path.is_file() {
        let txt = fs::read_to_string(&default_path)
            .map_err(|e| format!("Failed to read '{}': {e}", default_path.display()))?;
        cfg.apply_yaml_str(&txt)?;
    }

    if let Some(custom_path) = explicit_config {
        let txt = fs::read_to_string(custom_path)
            .map_err(|e| format!("Failed to read config '{}': {e}", custom_path.display()))?;
        cfg.apply_yaml_str(&txt)?;
        tracing::info!("Loaded config from: {}", custom_path.display());
        return Ok(cfg);
    }

    let config_path = server_root.join("config.yaml");
    if config_path.is_file() {
        let txt = fs::read_to_string(&config_path)
            .map_err(|e| format!("Failed to read '{}': {e}", config_path.display()))?;
        cfg.apply_yaml_str(&txt)?;
        tracing::info!("Loaded config from: {}", config_path.display());
    } else if default_path.is_file() {
        tracing::info!("Loaded config from: {}", default_path.display());
    } else {
        tracing::warn!(
            "Config file {} not found. Using built-in defaults.",
            config_path.display()
        );
    }

    Ok(cfg)
}

/// Resolve a configured model path relative to `server_root` (or CWD if directly found).
pub fn resolve_asset_path(server_root: &Path, configured: &str) -> PathBuf {
    let p = PathBuf::from(configured);
    if p.is_absolute() || p.exists() {
        return p;
    }
    server_root.join(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_yaml_parses_all_sections() {
        let yaml = r#"
version: 0.1.2
model:
  vocoder_path: "./custom/model.onnx"
  hnsep_model_path: "./custom/hnsep.onnx"
audio:
  sample_rate: 44100
  hop_size: 512
processing:
  wave_norm: false
  auto_vocal_fry: true
  peak_limit: 0.95
performance:
  max_workers: auto
  device: "directml"
  idle_timeout: 120
"#;
        let mut cfg = ServerConfig::default();
        cfg.apply_yaml_str(yaml).unwrap();
        assert_eq!(cfg.vocoder_path, "./custom/model.onnx");
        assert!(!cfg.wave_norm);
        assert_eq!(cfg.max_workers, MaxWorkersConfig::Auto);
        assert_eq!(cfg.device_preference, DevicePreference::DirectMl);
        assert_eq!(cfg.max_workers.resolve(true), 1);
        assert!((1..=4).contains(&cfg.max_workers.resolve_http_workers()));
        assert_eq!(MaxWorkersConfig::Fixed(2).resolve_http_workers(), 2);
        assert_eq!(cfg.idle_timeout, 120);
    }

    #[test]
    fn device_preference_variants() {
        assert_eq!(DevicePreference::parse("auto"), DevicePreference::Auto);
        assert_eq!(DevicePreference::parse("CPU"), DevicePreference::Cpu);
        assert_eq!(DevicePreference::parse("cuda"), DevicePreference::Cuda);
        assert_eq!(DevicePreference::parse("rocm"), DevicePreference::Rocm);
        assert_eq!(DevicePreference::parse("dml"), DevicePreference::DirectMl);
        assert_eq!(
            DevicePreference::parse("directml"),
            DevicePreference::DirectMl
        );
        assert_eq!(DevicePreference::parse("coreml"), DevicePreference::CoreMl);
    }
}
