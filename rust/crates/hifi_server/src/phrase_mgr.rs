//! Phrase-level continuous Mel+F0 synthesis manager, `MELOSNT1` note sidecar I/O,
//! bounded L1 `NotePrepCache`, and session-isolated `PhraseSessionStore`.

use hifi_core::audio::{read_wav_with_target_rate, write_wav, AudioBuffer};
use hifi_core::cache::{rename_with_retry, stat_sig, StatSig};
use hifi_core::wavtool::WavtoolEnvelope;
use hifi_dsp::{
    build_spliced_phrase_mel_and_f0, layout_phrase_timeline, PhraseNoteMelSlice, PhraseTimelineItem,
};
use hifi_effects::{
    apply_hg_growl, apply_lf_vocal_fry_in_place, apply_volume_and_peak_limit, gradient_1d_uniform,
    interp_1d_clamped, FryExcitationResult,
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

pub const MELOS_NOTE_MAGIC: &[u8; 8] = b"MELOSNT1";
pub const MELOS_NOTE_VERSION: u32 = 1;
const MAX_SIDECAR_MELS: usize = 512;
const MAX_SIDECAR_FRAMES: usize = 262_144;
const MAX_SIDECAR_MEL_ELEMENTS: usize = 16_777_216;
pub const MAX_PHRASE_DURATION_MS: f64 = 1_800_000.0;

const MAX_NOTE_PREP_RAM_ITEMS: usize = 512;
const MAX_ACTIVE_PHRASE_SESSIONS: usize = 64;
const PHRASE_SESSION_TTL: Duration = Duration::from_secs(600);
const STUB_WAV_SAMPLE_COUNT: usize = 64;

/// Returns sibling `<out_file>.melos` sidecar path (`res-xxxx.wav` -> `res-xxxx.wav.melos`),
/// matching OpenUtau's `DeleteCacheFiles()` prefix glob (`res-xxxx*`).
pub fn note_sidecar_path(out_file: &Path) -> PathBuf {
    let mut os = out_file.as_os_str().to_os_string();
    os.push(".melos");
    PathBuf::from(os)
}

#[inline]
pub fn normalize_path_key(path: &Path) -> String {
    let s = path.to_string_lossy().into_owned();
    if cfg!(windows) {
        s.replace('/', "\\").to_ascii_lowercase()
    } else {
        s
    }
}

/// Prepared acoustic payload for a single note without needing a prior GPU vocoder run.
#[derive(Debug, Clone)]
pub struct PreparedNote {
    /// Post-fry (if any) log-Mel matrix `[n_mels, n_frames]`.
    pub mel_render: Arc<Array2<f32>>,
    /// Post-fry (if any) F0 in Hz `[n_frames]`.
    pub f0_render: Vec<f32>,
    /// Effective MIDI pitch curve corresponding to `f0_render` `[n_frames]`.
    pub pitch_render: Arc<Vec<f64>>,
    /// Pre-fry MIDI pitch curve used for `A`-flag slope modulation `[n_frames]`.
    pub a_mod_pitch: Arc<Vec<f64>>,
    /// Frame timestamps in seconds `[n_frames]`.
    pub t_mel: Vec<f64>,
    pub new_start: f64,
    pub new_end: f64,
    pub cut_l: usize,
    pub cut_r: usize,
    pub scale: f64,
    pub p_gain: f64,
    pub volume: f64,
    pub a_mod: i32,
    pub hg: i32,
    pub fry_ctx: Option<FryExcitationResult>,
}

/// Binary `MELOSNT1` sidecar serialization errors.
#[derive(Debug)]
pub enum SidecarError {
    Io(io::Error),
    Invalid(String),
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SidecarError::Io(e) => write!(f, "Sidecar I/O error: {e}"),
            SidecarError::Invalid(msg) => write!(f, "Invalid MELOSNT1 sidecar: {msg}"),
        }
    }
}

impl From<io::Error> for SidecarError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

fn write_f32_slice<W: Write>(w: &mut W, data: &[f32]) -> io::Result<()> {
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_f64_slice<W: Write>(w: &mut W, data: &[f64]) -> io::Result<()> {
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_u32_le<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32_le<R: Read>(r: &mut R) -> io::Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_u64_le<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64_le<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_f32_vec<R: Read>(r: &mut R, len: usize) -> Result<Vec<f32>, SidecarError> {
    let byte_len = len
        .checked_mul(4)
        .ok_or_else(|| SidecarError::Invalid("f32 slice size overflow".into()))?;
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf)?;
    let mut out = Vec::with_capacity(len);
    for &chunk in buf.as_chunks::<4>().0 {
        let v = f32::from_le_bytes(chunk);
        if !v.is_finite() {
            return Err(SidecarError::Invalid(
                "Non-finite f32 value in sidecar".into(),
            ));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_f64_vec<R: Read>(r: &mut R, len: usize) -> Result<Vec<f64>, SidecarError> {
    let byte_len = len
        .checked_mul(8)
        .ok_or_else(|| SidecarError::Invalid("f64 slice size overflow".into()))?;
    let mut buf = vec![0u8; byte_len];
    r.read_exact(&mut buf)?;
    let mut out = Vec::with_capacity(len);
    for &chunk in buf.as_chunks::<8>().0 {
        let v = f64::from_le_bytes(chunk);
        if !v.is_finite() {
            return Err(SidecarError::Invalid(
                "Non-finite f64 value in sidecar".into(),
            ));
        }
        out.push(v);
    }
    Ok(out)
}

/// Atomically writes `PreparedNote` to `<out_file>.melos` in `MELOSNT1` format.
pub fn save_note_sidecar(path: &Path, note: &PreparedNote) -> Result<(), SidecarError> {
    let n_mels = note.mel_render.nrows();
    let n_frames = note.mel_render.ncols();
    if n_frames != note.f0_render.len()
        || n_frames != note.pitch_render.len()
        || n_frames != note.t_mel.len()
        || n_frames != note.a_mod_pitch.len()
    {
        return Err(SidecarError::Invalid(
            "Inconsistent PreparedNote vector lengths".into(),
        ));
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("note");
    let tmp_name = format!(".{}.{}.tmp", stem, Uuid::new_v4().simple());
    let tmp_path = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(&tmp_name),
        _ => PathBuf::from(&tmp_name),
    };

    struct TmpCleanup<'a>(&'a Path);
    impl<'a> Drop for TmpCleanup<'a> {
        fn drop(&mut self) {
            if self.0.exists() {
                let _ = fs::remove_file(self.0);
            }
        }
    }
    let cleanup = TmpCleanup(&tmp_path);

    {
        let file = File::create(&tmp_path)?;
        let mut w = BufWriter::new(file);
        w.write_all(MELOS_NOTE_MAGIC)?;
        w.write_all(&MELOS_NOTE_VERSION.to_le_bytes())?;
        w.write_all(&(n_mels as u32).to_le_bytes())?;
        w.write_all(&(n_frames as u32).to_le_bytes())?;
        w.write_all(&note.new_start.to_le_bytes())?;
        w.write_all(&note.new_end.to_le_bytes())?;
        w.write_all(&note.scale.to_le_bytes())?;
        w.write_all(&note.p_gain.to_le_bytes())?;
        w.write_all(&note.volume.to_le_bytes())?;
        w.write_all(&(note.cut_l as u64).to_le_bytes())?;
        w.write_all(&(note.cut_r as u64).to_le_bytes())?;
        w.write_all(&note.a_mod.to_le_bytes())?;
        w.write_all(&note.hg.to_le_bytes())?;

        let has_amod: u8 = if note.a_mod != 0 { 1 } else { 0 };
        w.write_all(&[has_amod])?;

        let has_fry: u8 = if note.fry_ctx.is_some() { 1 } else { 0 };
        w.write_all(&[has_fry])?;

        let mel_owned;
        let mel_slice = match note.mel_render.as_slice() {
            Some(s) => s,
            None => {
                mel_owned = note.mel_render.iter().copied().collect::<Vec<f32>>();
                &mel_owned
            }
        };
        write_f32_slice(&mut w, mel_slice)?;
        write_f32_slice(&mut w, &note.f0_render)?;
        write_f64_slice(&mut w, &note.pitch_render)?;
        write_f64_slice(&mut w, &note.t_mel)?;

        if has_amod != 0 {
            write_f64_slice(&mut w, &note.a_mod_pitch)?;
        }

        if let Some(ref fry) = note.fry_ctx {
            if fry.fry_env.len() != n_frames || fry.f0_eff.len() != n_frames {
                return Err(SidecarError::Invalid(
                    "FryExcitationResult vector lengths do not match n_frames".into(),
                ));
            }
            w.write_all(&fry.seed.to_le_bytes())?;
            w.write_all(&fry.new_start.to_le_bytes())?;
            w.write_all(&fry.new_end.to_le_bytes())?;
            write_f64_slice(&mut w, &fry.fry_env)?;
            write_f64_slice(&mut w, &fry.f0_eff)?;
        }
        w.flush()?;
    }

    rename_with_retry(&tmp_path, path)?;
    std::mem::forget(cleanup);
    Ok(())
}

/// Reads and validates a `PreparedNote` from `<out_file>.melos`.
pub fn load_note_sidecar(path: &Path) -> Result<Option<PreparedNote>, SidecarError> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(SidecarError::Io(e)),
    };
    let file_len = file.metadata()?.len();
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    if r.read_exact(&mut magic).is_err() || &magic != MELOS_NOTE_MAGIC {
        return Err(SidecarError::Invalid("Bad magic header".into()));
    }
    let version = read_u32_le(&mut r)?;
    if version != MELOS_NOTE_VERSION {
        return Err(SidecarError::Invalid(format!(
            "Unsupported version {version}"
        )));
    }
    let n_mels = read_u32_le(&mut r)? as usize;
    let n_frames = read_u32_le(&mut r)? as usize;
    if n_mels == 0 || n_mels > MAX_SIDECAR_MELS || n_frames == 0 || n_frames > MAX_SIDECAR_FRAMES {
        return Err(SidecarError::Invalid(format!(
            "Invalid shape ({n_mels}, {n_frames})"
        )));
    }

    let new_start = read_f64_le(&mut r)?;
    let new_end = read_f64_le(&mut r)?;
    let scale = read_f64_le(&mut r)?;
    let p_gain = read_f64_le(&mut r)?;
    let volume = read_f64_le(&mut r)?;
    if !new_start.is_finite()
        || !new_end.is_finite()
        || !scale.is_finite()
        || scale <= 0.0
        || !p_gain.is_finite()
        || p_gain <= 0.0
        || !volume.is_finite()
    {
        return Err(SidecarError::Invalid(
            "Non-finite or invalid scalar metadata".into(),
        ));
    }

    let cut_l = read_u64_le(&mut r)? as usize;
    let cut_r = read_u64_le(&mut r)? as usize;
    let a_mod = read_i32_le(&mut r)?;
    let hg = read_i32_le(&mut r)?;

    let mut flags_buf = [0u8; 2];
    r.read_exact(&mut flags_buf)?;
    let has_amod = flags_buf[0] != 0;
    let has_fry = flags_buf[1] != 0;

    let mel_len = n_mels
        .checked_mul(n_frames)
        .filter(|&len| len <= MAX_SIDECAR_MEL_ELEMENTS)
        .ok_or_else(|| SidecarError::Invalid("Mel shape exceeds safety limit".into()))?;
    let expected_bytes = (86u64)
        .checked_add((mel_len as u64).saturating_mul(4))
        .and_then(|n| n.checked_add((n_frames as u64).saturating_mul(20)))
        .and_then(|n| {
            if has_amod {
                n.checked_add((n_frames as u64).saturating_mul(8))
            } else {
                Some(n)
            }
        })
        .and_then(|n| {
            if has_fry {
                n.checked_add(20)
                    .and_then(|m| m.checked_add((n_frames as u64).saturating_mul(16)))
            } else {
                Some(n)
            }
        })
        .ok_or_else(|| SidecarError::Invalid("Sidecar payload size overflow".into()))?;
    if file_len < expected_bytes {
        return Err(SidecarError::Invalid(format!(
            "Truncated sidecar: expected >= {expected_bytes} bytes, got {file_len}"
        )));
    }
    let mel_raw = read_f32_vec(&mut r, mel_len)?;
    let mel_render = Array2::from_shape_vec((n_mels, n_frames), mel_raw)
        .map_err(|e| SidecarError::Invalid(format!("Mel reshape error: {e}")))?;

    let f0_render = read_f32_vec(&mut r, n_frames)?;
    let pitch_render = Arc::new(read_f64_vec(&mut r, n_frames)?);
    let t_mel = read_f64_vec(&mut r, n_frames)?;

    let a_mod_pitch = if has_amod {
        Arc::new(read_f64_vec(&mut r, n_frames)?)
    } else {
        Arc::clone(&pitch_render)
    };

    let fry_ctx = if has_fry {
        let seed = read_u32_le(&mut r)?;
        let f_start = read_f64_le(&mut r)?;
        let f_end = read_f64_le(&mut r)?;
        if !f_start.is_finite() || !f_end.is_finite() {
            return Err(SidecarError::Invalid("Non-finite fry bounds".into()));
        }
        let fry_env = read_f64_vec(&mut r, n_frames)?;
        let f0_eff = read_f64_vec(&mut r, n_frames)?;
        Some(FryExcitationResult {
            mel_mod: mel_render.clone(),
            f0_mod: f0_render.clone(),
            fry_env,
            f0_eff,
            t_mel: t_mel.clone(),
            new_start: f_start,
            new_end: f_end,
            seed,
        })
    } else {
        None
    };

    Ok(Some(PreparedNote {
        mel_render: Arc::new(mel_render),
        f0_render,
        pitch_render,
        a_mod_pitch,
        t_mel,
        new_start,
        new_end,
        cut_l,
        cut_r,
        scale,
        p_gain,
        volume,
        a_mod,
        hg,
        fry_ctx,
    }))
}

/// Writes a minimal valid 64-sample 16-bit mono PCM WAV stub cleanly for OpenUtau checks.
pub fn write_stub_wav(path: &Path, sample_rate: u32) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let stub = AudioBuffer::new(vec![0.0_f32; STUB_WAV_SAMPLE_COUNT], sample_rate, 1);
    write_wav(path, &stub).map_err(|e| e.to_string())
}

/// Checks that sibling `.melos` sidecar is at least as fresh as `<wav_path>` when `<wav_path>` exists.
pub fn is_sidecar_fresh_for_wav(wav_path: &Path, sidecar_path: &Path) -> bool {
    let Some(w_sig) = stat_sig(wav_path) else {
        return true;
    };
    let Some(s_sig) = stat_sig(sidecar_path) else {
        return false;
    };
    s_sig.mtime_ns >= w_sig.mtime_ns
}

#[derive(Clone)]
struct CachedNoteEntry {
    prepared: Arc<PreparedNote>,
    is_stub: bool,
    wav_sig: Option<StatSig>,
}

/// Bounded L1 RAM LRU tracking prepared note features and whether `<out_file>` is a stub WAV.
pub struct NotePrepCache {
    order: VecDeque<String>,
    map: HashMap<String, CachedNoteEntry>,
}

impl Default for NotePrepCache {
    fn default() -> Self {
        Self {
            order: VecDeque::with_capacity(MAX_NOTE_PREP_RAM_ITEMS),
            map: HashMap::with_capacity(MAX_NOTE_PREP_RAM_ITEMS),
        }
    }
}

impl NotePrepCache {
    pub fn insert(&mut self, out_file: &Path, prepared: Arc<PreparedNote>, is_stub: bool) {
        let key = normalize_path_key(out_file);
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else {
            while self.map.len() >= MAX_NOTE_PREP_RAM_ITEMS {
                if let Some(evicted) = self.order.pop_front() {
                    self.map.remove(&evicted);
                } else {
                    break;
                }
            }
        }
        let wav_sig = stat_sig(out_file);
        self.order.push_back(key.clone());
        self.map.insert(
            key,
            CachedNoteEntry {
                prepared,
                is_stub,
                wav_sig,
            },
        );
    }

    pub fn get(&mut self, out_file: &Path) -> Option<(Arc<PreparedNote>, bool)> {
        let key = normalize_path_key(out_file);
        let cur_sig = stat_sig(out_file);
        if let Some(entry) = self.map.get(&key).cloned() {
            if entry.wav_sig.is_some() && cur_sig != entry.wav_sig {
                self.map.remove(&key);
                self.order.retain(|k| k != &key);
                return None;
            }
            self.order.retain(|k| k != &key);
            self.order.push_back(key);
            return Some((entry.prepared, entry.is_stub));
        }
        None
    }

    pub fn mark_backfilled(&mut self, out_file: &Path) {
        let key = normalize_path_key(out_file);
        let new_sig = stat_sig(out_file);
        if let Some(entry) = self.map.get_mut(&key) {
            entry.is_stub = false;
            entry.wav_sig = new_sig;
        }
    }
}

/// JSON request payload for `POST /phrase/wavtool`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WavtoolStepRequest {
    pub output_file: String,
    pub input_file: String,
    pub stp_ms: f64,
    pub dur_ms: f64,
    #[serde(default)]
    pub envelope: Vec<String>,
    #[serde(default)]
    pub commit: bool,
    #[serde(default)]
    pub batch_id: Option<u64>,
    #[serde(default)]
    pub expected_steps: Option<usize>,
    #[serde(default)]
    pub fallback_resampler_args: Option<Vec<String>>,
}

/// JSON note specification inside `POST /phrase/render`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PhraseBatchNoteSpec {
    #[serde(default)]
    pub resampler_args: Option<Vec<String>>,
    #[serde(default)]
    pub wav_file: Option<String>,
    #[serde(default)]
    pub is_silence: bool,
    pub stp_ms: f64,
    pub dur_ms: f64,
    #[serde(default)]
    pub envelope: Vec<String>,
}

/// JSON request payload for `POST /phrase/render`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PhraseBatchRequest {
    pub output_file: String,
    pub notes: Vec<PhraseBatchNoteSpec>,
}

#[derive(Clone)]
pub enum PhraseStepKind {
    MelNote {
        prepared: Arc<PreparedNote>,
        note_out_file: Option<PathBuf>,
        needs_backfill: bool,
    },
    DirectAudio {
        wav_path: PathBuf,
    },
    Silence,
}

#[derive(Clone)]
pub struct PhraseStepRecord {
    pub kind: PhraseStepKind,
    pub stp_ms: f64,
    pub dur_ms: f64,
    pub envelope: WavtoolEnvelope,
}

pub struct PhraseSession {
    pub batch_id: Option<u64>,
    pub committed: bool,
    pub last_touched: Instant,
    pub steps: Vec<PhraseStepRecord>,
}

impl PhraseSession {
    fn new(batch_id: Option<u64>) -> Self {
        Self {
            batch_id,
            committed: false,
            last_touched: Instant::now(),
            steps: Vec::new(),
        }
    }

    pub fn reset(&mut self, batch_id: Option<u64>) {
        self.batch_id = batch_id;
        self.committed = false;
        self.last_touched = Instant::now();
        self.steps.clear();
    }
}

/// Per-output-path isolated store with bounded session count and TTL pruning.
///
/// Always acquire `PhraseSessionStore` briefly only to obtain the per-output `Arc<Mutex<PhraseSession>>`,
/// then drop the outer store lock before mutating session state or running GPU synthesis.
#[derive(Default)]
pub struct PhraseSessionStore {
    sessions: HashMap<String, Arc<Mutex<PhraseSession>>>,
}

impl PhraseSessionStore {
    pub fn get_or_create(
        &mut self,
        output_file: &Path,
        batch_id: Option<u64>,
    ) -> Arc<Mutex<PhraseSession>> {
        let now = Instant::now();
        // Prune idle sessions safely without locking any session that is currently busy.
        self.sessions.retain(|_, arc| {
            if let Ok(guard) = arc.try_lock() {
                now.duration_since(guard.last_touched) < PHRASE_SESSION_TTL
            } else {
                true
            }
        });

        let key = normalize_path_key(output_file);
        if let Some(existing) = self.sessions.get(&key) {
            return Arc::clone(existing);
        }

        while self.sessions.len() >= MAX_ACTIVE_PHRASE_SESSIONS {
            let mut oldest_key: Option<(String, Instant)> = None;
            for (k, arc) in &self.sessions {
                if let Ok(guard) = arc.try_lock() {
                    if oldest_key
                        .as_ref()
                        .is_none_or(|(_, t)| guard.last_touched < *t)
                    {
                        oldest_key = Some((k.clone(), guard.last_touched));
                    }
                }
            }
            if let Some((evict_k, _)) = oldest_key {
                self.sessions.remove(&evict_k);
            } else {
                break;
            }
        }

        let session = Arc::new(Mutex::new(PhraseSession::new(batch_id)));
        self.sessions.insert(key, Arc::clone(&session));
        session
    }
}

/// Removes legacy `<output>.whd` / `<output>.dat` files left by classic wavtools so OpenUtau
/// does not overwrite the final continuous WAV during `temp.bat` teardown.
pub fn cleanup_legacy_whd_dat(output_file: &Path) {
    for ext in [".whd", ".dat"] {
        let mut os = output_file.as_os_str().to_os_string();
        os.push(ext);
        let p = PathBuf::from(os);
        if p.exists() {
            let _ = fs::remove_file(p);
        }
    }
}

/// Result of synthesizing a continuous phrase waveform, along with pre-envelope note backfill info.
pub struct PhraseRenderResult {
    pub waveform: Vec<f32>,
    pub backfills: Vec<(PathBuf, Vec<f32>)>,
}

/// Evaluates `A`-flag pitch-slope modulation curve over note's `Mel` timestamps.
fn compute_a_mod_frame_gains(
    note: &PreparedNote,
    hop_size: usize,
    sample_rate: u32,
) -> Option<Vec<f64>> {
    if note.a_mod == 0 || note.a_mod_pitch.len() <= 1 || note.t_mel.len() <= 1 {
        return None;
    }
    let thop = (hop_size as f64) / (sample_rate.max(1) as f64);
    if thop <= 0.0 || !thop.is_finite() {
        return None;
    }
    let a_clamped = (note.a_mod as f64).clamp(-100.0, 100.0);
    let dp = gradient_1d_uniform(&note.a_mod_pitch, thop);
    Some(
        dp.iter()
            .map(|&d| 5.0_f64.powf(1.0e-4 * a_clamped * d))
            .collect(),
    )
}

/// Executes single-Vocoder-pass continuous Mel+F0 synthesis and sample-accurate post-processing
/// across a complete sequence of `PhraseStepRecord`s.
pub fn synthesize_phrase_steps<F>(
    steps: &[PhraseStepRecord],
    n_mels: usize,
    hop_size: usize,
    sample_rate: u32,
    peak_limit: f64,
    run_vocoder: F,
) -> Result<PhraseRenderResult, String>
where
    F: FnOnce(&Array2<f32>, &[f32]) -> Result<Vec<f32>, String>,
{
    let mut timeline_items: Vec<PhraseTimelineItem> = Vec::with_capacity(steps.len());
    for step in steps {
        match &step.kind {
            PhraseStepKind::MelNote { prepared, .. } => {
                timeline_items.push(PhraseTimelineItem::MelNote(PhraseNoteMelSlice {
                    mel_render: Arc::clone(&prepared.mel_render),
                    pitch_render: Arc::clone(&prepared.pitch_render),
                    new_start_sec: prepared.new_start,
                    new_end_sec: prepared.new_end,
                    stp_ms: step.stp_ms,
                    dur_ms: step.dur_ms,
                    envelope: step.envelope.clone(),
                    scale: prepared.scale,
                    p_gain: prepared.p_gain,
                    volume: prepared.volume,
                }));
            }
            PhraseStepKind::DirectAudio { .. } => {
                timeline_items.push(PhraseTimelineItem::DirectAudio {
                    dur_ms: step.dur_ms,
                    envelope: step.envelope.clone(),
                });
            }
            PhraseStepKind::Silence => {
                timeline_items.push(PhraseTimelineItem::Silence {
                    dur_ms: step.dur_ms,
                    ovr_ms: step.envelope.ovr_ms,
                });
            }
        }
    }

    let layout = layout_phrase_timeline(&timeline_items, sample_rate);
    if layout.total_duration_ms > MAX_PHRASE_DURATION_MS {
        return Err(format!(
            "Phrase duration {:.1} ms exceeds safety limit ({:.1} ms)",
            layout.total_duration_ms, MAX_PHRASE_DURATION_MS
        ));
    }
    if layout.total_samples == 0 {
        return Ok(PhraseRenderResult {
            waveform: Vec::new(),
            backfills: Vec::new(),
        });
    }

    let total_samples = layout.total_samples;
    let sr_f = sample_rate as f64;
    let mut render = match build_spliced_phrase_mel_and_f0(&layout, n_mels, hop_size, sample_rate) {
        Some(spliced) => {
            let wav_padded = run_vocoder(&spliced.mel, &spliced.f0)?;
            let mut buf = vec![0.0_f32; total_samples];
            if spliced.crop_start < wav_padded.len() {
                let avail = (wav_padded.len() - spliced.crop_start).min(total_samples);
                buf[..avail]
                    .copy_from_slice(&wav_padded[spliced.crop_start..spliced.crop_start + avail]);
            }
            if spliced.phrase_scale != 1.0
                && spliced.phrase_scale > 1e-12
                && spliced.phrase_scale.is_finite()
            {
                let inv = (1.0 / spliced.phrase_scale) as f32;
                for x in &mut buf {
                    *x *= inv;
                }
            }
            buf
        }
        None => vec![0.0_f32; total_samples],
    };

    // Identify first and last MelNote span indices for overlap-weight normalized per-note FX blending.
    let mel_span_indices: Vec<usize> = layout
        .spans
        .iter()
        .enumerate()
        .filter(|(_, sp)| sp.is_mel_note && sp.dur_ms > 0.0)
        .map(|(i, _)| i)
        .collect();
    let first_mel_span = mel_span_indices.first().copied();
    let last_mel_span = mel_span_indices.last().copied();

    // Precompute A-flag modulation frame gains per MelNote span (if non-zero).
    let amod_gains: Vec<Option<Vec<f64>>> = steps
        .iter()
        .map(|step| match &step.kind {
            PhraseStepKind::MelNote { prepared, .. } => {
                compute_a_mod_frame_gains(prepared, hop_size, sample_rate)
            }
            _ => None,
        })
        .collect();

    // Apply per-note HG growl and LF vocal fry smoothly before final sample-level envelope.
    // We snapshot pre-FX `render` only if any note actually activates HG or vocal fry.
    let any_fx = steps.iter().any(|step| match &step.kind {
        PhraseStepKind::MelNote { prepared, .. } => {
            prepared.hg > 0
                || prepared
                    .fry_ctx
                    .as_ref()
                    .is_some_and(|c| c.fry_env.iter().copied().fold(0.0_f64, f64::max) > 1e-4)
        }
        _ => false,
    });

    if any_fx {
        let base_render = render.clone();
        let mut fx_accum = vec![0.0_f64; total_samples];
        let mut weight_accum = vec![0.0_f64; total_samples];

        for (span_idx, span) in layout.spans.iter().enumerate() {
            if !span.is_mel_note || span.end_sample <= span.start_sample {
                continue;
            }
            let step = &steps[span.item_index];
            let PhraseStepKind::MelNote { prepared, .. } = &step.kind else {
                continue;
            };
            let has_hg = prepared.hg > 0;
            let has_fry = prepared
                .fry_ctx
                .as_ref()
                .is_some_and(|c| c.fry_env.iter().copied().fold(0.0_f64, f64::max) > 1e-4);

            let s0 = span.start_sample.min(total_samples);
            let s1 = span.end_sample.min(total_samples);
            if s1 <= s0 {
                continue;
            }

            let mut seg = base_render[s0..s1].to_vec();
            if has_hg {
                seg = apply_hg_growl(&seg, sample_rate, prepared.hg);
            }
            if has_fry {
                if let Some(ref ctx) = prepared.fry_ctx {
                    let stp_sec = step.stp_ms.max(0.0) / 1000.0;
                    let mut shifted_ctx = ctx.clone();
                    shifted_ctx.new_start = ctx.new_start + stp_sec;
                    shifted_ctx.new_end = shifted_ctx.new_start + (seg.len() as f64) / sr_f;
                    apply_lf_vocal_fry_in_place(&mut seg, Some(&shifted_ctx), sample_rate);
                }
            }

            let is_head = first_mel_span == Some(span_idx);
            let is_tail = last_mel_span == Some(span_idx);
            for (local_i, &fx_val) in seg.iter().enumerate() {
                let dst = s0 + local_i;
                let pos_ms = ((dst - span.start_sample) as f64) * 1000.0 / sr_f;
                let w = span
                    .envelope
                    .evaluate_mel_blend_weight(pos_ms, span.dur_ms, is_head, is_tail)
                    .max(1e-6);
                fx_accum[dst] += (fx_val as f64) * w;
                weight_accum[dst] += w;
            }
        }

        for i in 0..total_samples {
            if weight_accum[i] > 0.0 {
                render[i] = (fx_accum[i] / weight_accum[i]) as f32;
            }
        }
    }

    // Collect pre-envelope backfills for any stub WAVs (so switching Wavtool back to `convergence` works).
    let mut backfills: Vec<(PathBuf, Vec<f32>)> = Vec::new();
    for span in &layout.spans {
        let step = &steps[span.item_index];
        if let PhraseStepKind::MelNote {
            prepared,
            note_out_file: Some(ref path),
            needs_backfill: true,
        } = &step.kind
        {
            let s0 = span.start_sample.min(total_samples);
            let s1 = span.end_sample.min(total_samples);
            let stp_samples =
                ((step.stp_ms.max(0.0) * sr_f / 1000.0).round() as isize).max(0) as usize;
            let expected_len = prepared
                .cut_r
                .saturating_sub(prepared.cut_l)
                .max(stp_samples + s1.saturating_sub(s0));
            let mut note_wav = vec![0.0_f32; expected_len];
            // Fill front stp_samples from earlier phrase samples if available
            if stp_samples > 0 && s0 > 0 {
                let copy_front = stp_samples.min(s0);
                let src_front = s0 - copy_front;
                let dst_front = stp_samples - copy_front;
                note_wav[dst_front..stp_samples].copy_from_slice(&render[src_front..s0]);
            }
            let body_copy = s1
                .saturating_sub(s0)
                .min(expected_len.saturating_sub(stp_samples));
            if body_copy > 0 {
                note_wav[stp_samples..stp_samples + body_copy]
                    .copy_from_slice(&render[s0..s0 + body_copy]);
            }
            if let Some(gains) = amod_gains[span.item_index].as_deref() {
                for (idx, sample) in note_wav.iter_mut().enumerate() {
                    let t_audio = prepared.new_start + (idx as f64) / sr_f;
                    let g_amod = interp_1d_clamped(&prepared.t_mel, gains, t_audio) as f32;
                    *sample *= g_amod;
                }
            }
            apply_volume_and_peak_limit(&mut note_wav, 100.0, peak_limit);
            backfills.push((path.clone(), note_wav));
        }
    }

    // Apply sample-accurate UTAU amplitude envelope `E_wave[n] = sum_k g_k(n) * a_mod_k(n)` across MelNotes.
    if !mel_span_indices.is_empty() {
        let mut e_wave = vec![0.0_f64; total_samples];
        for &span_idx in &mel_span_indices {
            let span = &layout.spans[span_idx];
            let step = &steps[span.item_index];
            let PhraseStepKind::MelNote { prepared, .. } = &step.kind else {
                continue;
            };
            let s0 = span.start_sample.min(total_samples);
            let s1 = span.end_sample.min(total_samples);
            let stp_sec = step.stp_ms.max(0.0) / 1000.0;
            let frame_gains = amod_gains[span.item_index].as_deref();

            for (n, e_slot) in e_wave[s0..s1].iter_mut().enumerate() {
                let pos_ms = ((s0 + n - span.start_sample) as f64) * 1000.0 / sr_f;
                let g_env = span.envelope.evaluate_gain(pos_ms, span.dur_ms);
                if g_env <= 0.0 {
                    continue;
                }
                let g_amod = match frame_gains {
                    Some(gains) => {
                        let t_audio = prepared.new_start + stp_sec + (pos_ms / 1000.0);
                        interp_1d_clamped(&prepared.t_mel, gains, t_audio)
                    }
                    None => 1.0,
                };
                *e_slot += g_env * g_amod;
            }
        }
        for (x, &e) in render.iter_mut().zip(e_wave.iter()) {
            *x = ((*x as f64) * e) as f32;
        }
    }

    // Mix DirectAudio items (e.g. OpenUtau $direct=True or external third-party resampler WAVs).
    for span in &layout.spans {
        if !span.is_direct_audio {
            continue;
        }
        let step = &steps[span.item_index];
        let PhraseStepKind::DirectAudio { wav_path } = &step.kind else {
            continue;
        };
        let buf = read_wav_with_target_rate(wav_path, Some(sample_rate)).map_err(|e| {
            format!(
                "Failed to read DirectAudio WAV '{}': {e}",
                wav_path.display()
            )
        })?;
        let stp_samples = (step.stp_ms * sr_f / 1000.0).round() as isize;
        let s0 = span.start_sample.min(total_samples);
        let s1 = span.end_sample.min(total_samples);

        for (offset, dst_slot) in render[s0..s1].iter_mut().enumerate() {
            let local_i = ((s0 + offset) - span.start_sample) as isize;
            let src_i = local_i + stp_samples;
            if src_i >= 0 && (src_i as usize) < buf.samples.len() {
                let pos_ms = (local_i as f64) * 1000.0 / sr_f;
                let gain = span.envelope.evaluate_gain(pos_ms, span.dur_ms);
                *dst_slot += buf.samples[src_i as usize] * (gain as f32);
            }
        }
    }

    apply_volume_and_peak_limit(&mut render, 100.0, peak_limit);

    Ok(PhraseRenderResult {
        waveform: render,
        backfills,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use tempfile::tempdir;

    #[test]
    fn melos_sidecar_roundtrip_and_corruption_rejection() {
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("res-0001.wav");
        let sidecar_path = note_sidecar_path(&wav_path);
        assert_eq!(
            sidecar_path.file_name().unwrap().to_str().unwrap(),
            "res-0001.wav.melos"
        );

        let n_mels = 4;
        let n_frames = 12;
        let mel = Array2::from_elem((n_mels, n_frames), -1.25_f32);
        let f0 = vec![440.0_f32; n_frames];
        let pitch = Arc::new(vec![69.0_f64; n_frames]);
        let a_mod_pitch = Arc::new(vec![70.0_f64; n_frames]);
        let t_mel: Vec<f64> = (0..n_frames)
            .map(|i| i as f64 * (512.0 / 44100.0))
            .collect();

        let note = PreparedNote {
            mel_render: Arc::new(mel),
            f0_render: f0,
            pitch_render: pitch,
            a_mod_pitch,
            t_mel: t_mel.clone(),
            new_start: 0.02,
            new_end: 0.14,
            cut_l: 882,
            cut_r: 6174,
            scale: 0.5,
            p_gain: 1.25,
            volume: 95.0,
            a_mod: 20,
            hg: 15,
            fry_ctx: Some(FryExcitationResult {
                mel_mod: Array2::from_elem((n_mels, n_frames), -1.25_f32),
                f0_mod: vec![440.0_f32; n_frames],
                fry_env: vec![0.3_f64; n_frames],
                f0_eff: vec![430.0_f64; n_frames],
                t_mel,
                new_start: 0.02,
                new_end: 0.14,
                seed: 777,
            }),
        };

        save_note_sidecar(&sidecar_path, &note).unwrap();
        let loaded = load_note_sidecar(&sidecar_path).unwrap().unwrap();
        assert_eq!(loaded.mel_render.shape(), &[n_mels, n_frames]);
        assert_abs_diff_eq!(loaded.scale, 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(loaded.p_gain, 1.25, epsilon = 1e-12);
        assert_eq!(loaded.cut_l, 882);
        assert_eq!(loaded.a_mod, 20);
        assert_eq!(loaded.hg, 15);
        assert_eq!(loaded.fry_ctx.as_ref().unwrap().seed, 777);
        assert_abs_diff_eq!(loaded.a_mod_pitch[0], 70.0, epsilon = 1e-12);

        // Corrupt bytes -> rejected cleanly
        let mut raw = fs::read(&sidecar_path).unwrap();
        raw[0] = b'X';
        fs::write(&sidecar_path, &raw).unwrap();
        assert!(load_note_sidecar(&sidecar_path).is_err());

        // Missing file -> Ok(None)
        let missing = note_sidecar_path(&dir.path().join("nope.wav"));
        assert!(load_note_sidecar(&missing).unwrap().is_none());

        // Truncated file claiming huge (512, 200_000) header must fail immediately without OOM
        let dos_path = dir.path().join("dos.wav.melos");
        let mut fake = Vec::with_capacity(86);
        fake.extend_from_slice(MELOS_NOTE_MAGIC);
        fake.extend_from_slice(&MELOS_NOTE_VERSION.to_le_bytes());
        fake.extend_from_slice(&(512u32).to_le_bytes());
        fake.extend_from_slice(&(200_000u32).to_le_bytes());
        fake.extend_from_slice(&0.0_f64.to_le_bytes());
        fake.extend_from_slice(&1.0_f64.to_le_bytes());
        fake.extend_from_slice(&1.0_f64.to_le_bytes());
        fake.extend_from_slice(&1.0_f64.to_le_bytes());
        fake.extend_from_slice(&100.0_f64.to_le_bytes());
        fake.extend_from_slice(&0u64.to_le_bytes());
        fake.extend_from_slice(&0u64.to_le_bytes());
        fake.extend_from_slice(&0i32.to_le_bytes());
        fake.extend_from_slice(&0i32.to_le_bytes());
        fake.extend_from_slice(&[0u8, 0u8]);
        fs::write(&dos_path, &fake).unwrap();
        assert!(load_note_sidecar(&dos_path).is_err());
    }
}
