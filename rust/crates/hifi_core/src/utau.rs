//! UTAU argument, note, pitchbend, tempo, and flag parsing.
//!
//! Clean-room port of `util/parse_utau.py` and the UTAU parameter parsing
//! helpers from `backend/resampler.py`.

use std::collections::BTreeMap;
use thiserror::Error;

/// Maximum run-length repeat count allowed when decoding `#<count>` in UTAU pitch strings.
pub const MAX_PITCH_RLE_COUNT: usize = 131_072;

/// All supported UTAU flags ordered longest-match-first for deterministic scanning.
pub const ALL_FLAGS: [&str; 26] = [
    "fe", "fl", "fo", "fv", "fp", "ve", "vo", "Hv", "Hb", "Ht", "Ho", "Hr", "He", "HG", "g", "t",
    "A", "B", "G", "P", "S", "p", "R", "D", "C", "Z",
];

/// Errors occurring when parsing UTAU pitch, note, or tempo values.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum UtauError {
    #[error("Invalid Base64 pitchbend character: '{0}'")]
    InvalidBase64Char(char),

    #[error("Truncated Base64 pitchbend pair: '{0}'")]
    TruncatedBase64Pair(String),

    #[error("Invalid MIDI note name: '{0}'")]
    InvalidMidiNote(String),

    #[error("Invalid UTAU tempo value: '{0}'")]
    InvalidTempo(String),
}

/// Converts a single UTAU Base64 character into an unsigned 6-bit integer (`0..=63`).
pub fn to_uint6(ch: char) -> Result<u8, UtauError> {
    let code = ch as u32;
    if (97..=122).contains(&code) {
        Ok((code - 71) as u8)
    } else if (65..=90).contains(&code) {
        Ok((code - 65) as u8)
    } else if (48..=57).contains(&code) {
        Ok((code + 4) as u8)
    } else if code == 43 {
        Ok(62)
    } else if code == 47 {
        Ok(63)
    } else {
        Err(UtauError::InvalidBase64Char(ch))
    }
}

/// Converts a 2-character Base64 token into a signed 12-bit integer (`-2048..=2047`).
pub fn to_int12(b64: &str) -> Result<i32, UtauError> {
    let mut chars = b64.chars();
    let c0 = chars
        .next()
        .ok_or_else(|| UtauError::TruncatedBase64Pair(b64.to_string()))?;
    let c1 = chars
        .next()
        .ok_or_else(|| UtauError::TruncatedBase64Pair(b64.to_string()))?;
    let uint12 = ((to_uint6(c0)? as i32) << 6) | (to_uint6(c1)? as i32);
    if (uint12 >> 11) & 1 == 1 {
        Ok(uint12 - 4096)
    } else {
        Ok(uint12)
    }
}

/// Converts a Base64 pitch stream into a sequence of signed 12-bit integers.
/// Any trailing odd character is ignored, matching `util/parse_utau.py::to_int12_stream`.
pub fn to_int12_stream(b64: &str) -> Result<Vec<i32>, UtauError> {
    let mut res = Vec::with_capacity(b64.len() / 2);
    let mut chars = b64.chars();
    while let Some(c0) = chars.next() {
        let Some(c1) = chars.next() else {
            break;
        };
        let uint12 = ((to_uint6(c0)? as i32) << 6) | (to_uint6(c1)? as i32);
        if (uint12 >> 11) & 1 == 1 {
            res.push(uint12 - 4096);
        } else {
            res.push(uint12);
        }
    }
    Ok(res)
}

/// Decodes UTAU's `#`-RLE Base64 pitchbend string into pitch offsets in cents,
/// appending a trailing `0` endpoint matching `util/parse_utau.py::pitch_string_to_cents`.
pub fn pitch_string_to_cents(x: &str) -> Result<Vec<i32>, UtauError> {
    if x.is_empty() {
        return Ok(vec![0, 0]);
    }

    let parts: Vec<&str> = x.split('#').collect();
    let mut res = Vec::new();

    let mut i = 0;
    while i < parts.len() {
        let pitch_str = parts[i];
        res.extend(to_int12_stream(pitch_str)?);
        if i + 1 < parts.len() {
            let rle = parts[i + 1];
            if !res.is_empty() && !rle.is_empty() && rle.bytes().all(|b| b.is_ascii_digit()) {
                let rle_cnt = rle
                    .parse::<usize>()
                    .unwrap_or(MAX_PITCH_RLE_COUNT)
                    .min(MAX_PITCH_RLE_COUNT);
                if let Some(&last) = res.last() {
                    res.resize(res.len() + rle_cnt, last);
                }
            }
        }
        i += 2;
    }

    if res.is_empty() {
        return Ok(vec![0, 0]);
    }

    res.push(0);
    Ok(res)
}

fn note_semitone_offset(note: &str) -> Option<i32> {
    match note {
        "C" => Some(0),
        "C#" | "Db" => Some(1),
        "D" => Some(2),
        "D#" | "Eb" => Some(3),
        "E" | "Fb" => Some(4),
        "E#" | "F" => Some(5),
        "F#" | "Gb" => Some(6),
        "G" => Some(7),
        "G#" | "Ab" => Some(8),
        "A" => Some(9),
        "A#" | "Bb" => Some(10),
        "B" | "Cb" => Some(11),
        "B#" => Some(0),
        _ => None,
    }
}

/// Converts a scientific pitch note name (e.g. `"A4"`, `"C#4"`, `"C-1"`) or integer MIDI string
/// (e.g. `"60"`) to a MIDI note number.
pub fn note_to_midi(x: &str) -> Result<i32, UtauError> {
    let s = x.trim();
    if s.is_empty() {
        return Err(UtauError::InvalidMidiNote(x.to_string()));
    }

    let unsigned_part = s.strip_prefix(['+', '-']).unwrap_or(s);
    if !unsigned_part.is_empty() && unsigned_part.bytes().all(|b| b.is_ascii_digit()) {
        return s
            .parse::<i32>()
            .map_err(|_| UtauError::InvalidMidiNote(x.to_string()));
    }

    let bytes = s.as_bytes();
    let letter = bytes[0] as char;
    if !('A'..='G').contains(&letter) {
        return Err(UtauError::InvalidMidiNote(x.to_string()));
    }

    let (note_end, accidental_ok) = if bytes.len() >= 2 && (bytes[1] == b'#' || bytes[1] == b'b') {
        (2, true)
    } else {
        (1, false)
    };
    let _ = accidental_ok;

    let note_part = &s[..note_end];
    let octave_part = &s[note_end..];
    if octave_part.is_empty() {
        return Err(UtauError::InvalidMidiNote(x.to_string()));
    }
    let oct_digits = octave_part.strip_prefix('-').unwrap_or(octave_part);
    if oct_digits.is_empty() || !oct_digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(UtauError::InvalidMidiNote(x.to_string()));
    }

    let semitone =
        note_semitone_offset(note_part).ok_or_else(|| UtauError::InvalidMidiNote(x.to_string()))?;
    let octave = octave_part
        .parse::<i32>()
        .map_err(|_| UtauError::InvalidMidiNote(x.to_string()))?;

    Ok((octave + 1) * 12 + semitone)
}

/// Converts a MIDI note number to Hertz using equal temperament (`A4 = 69 = 440.0 Hz`).
#[inline]
pub fn midi_to_hz(midi: f64) -> f64 {
    440.0 * 2.0_f64.powf((midi - 69.0) / 12.0)
}

/// Converts a slice of MIDI note values to `f32` frequencies in Hertz.
pub fn midi_slice_to_hz(midi: &[f64]) -> Vec<f32> {
    midi.iter().map(|&m| midi_to_hz(m) as f32).collect()
}

/// Parses an UTAU tempo string (with optional `'!'` prefix) and clamps to `>= 1.0`.
pub fn parse_tempo(tempo: &str) -> Result<f64, UtauError> {
    let trimmed = tempo.trim();
    let num_str = trimmed.strip_prefix('!').unwrap_or(trimmed);
    let parsed = num_str
        .parse::<f64>()
        .map_err(|_| UtauError::InvalidTempo(tempo.to_string()))?;
    if !parsed.is_finite() {
        return Err(UtauError::InvalidTempo(tempo.to_string()));
    }
    Ok(parsed.max(1.0))
}

/// Scans an UTAU flags string using longest-match-first semantics across all 26 flags
/// (`fe, fl, fo, fv, fp, ve, vo, g, t, A, B, G, P, S, p, R, D, C, Z, Hv, Hb, Ht, Ho, Hr, He, HG`),
/// after stripping `'/'` separators, matching `flag_re.findall(flags.replace('/', ''))`.
pub fn find_flags(raw: &str) -> Vec<(String, Option<i32>)> {
    let sanitized = raw.replace('/', "");
    let bytes = sanitized.as_bytes();
    let mut matches = Vec::new();
    let mut i = 0;

    while i < sanitized.len() {
        // Ensure we start on a valid UTF-8 char boundary.
        let rest = &sanitized[i..];
        let mut matched_flag: Option<&str> = None;
        for &flag in &ALL_FLAGS {
            if rest.starts_with(flag) {
                matched_flag = Some(flag);
                break;
            }
        }

        if let Some(flag) = matched_flag {
            i += flag.len();
            // Match optional ([+-]?\d+)
            let val_start = i;
            let mut val_idx = i;
            if val_idx < bytes.len() && (bytes[val_idx] == b'+' || bytes[val_idx] == b'-') {
                val_idx += 1;
            }
            let digit_start = val_idx;
            while val_idx < bytes.len() && bytes[val_idx].is_ascii_digit() {
                val_idx += 1;
            }

            let val = if val_idx > digit_start {
                let num_slice = &sanitized[val_start..val_idx];
                i = val_idx;
                let parsed = num_slice.parse::<i64>().unwrap_or_else(|_| {
                    if num_slice.starts_with('-') {
                        i64::MIN
                    } else {
                        i64::MAX
                    }
                });
                Some(parsed.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
            } else {
                None
            };

            matches.push((flag.to_string(), val));
        } else {
            // Advance by one Unicode scalar value.
            let ch_len = rest.chars().next().map_or(1, char::len_utf8);
            i += ch_len;
        }
    }

    matches
}

/// Parses an UTAU flags string into a key -> optional integer map where the first occurrence
/// of each flag name wins, matching `backend/resampler.py`.
pub fn parse_flags_map(raw: &str) -> BTreeMap<String, Option<i32>> {
    let mut map = BTreeMap::new();
    for (k, v) in find_flags(raw) {
        map.entry(k).or_insert(v);
    }
    map
}

/// Parsed and normalized UTAU flag container matching `backend/resampler.py` semantics.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UtauFlags {
    raw: BTreeMap<String, Option<i32>>,
}

impl UtauFlags {
    /// Parses a raw UTAU flags string (e.g. `"/g-5/t10Hb100Hv80P"`).
    pub fn parse(raw: &str) -> Self {
        Self {
            raw: parse_flags_map(raw),
        }
    }

    /// Returns the underlying first-match flag map.
    pub fn as_map(&self) -> &BTreeMap<String, Option<i32>> {
        &self.raw
    }

    /// Returns `Some(val)` if `key` was present (`None` inside `Option` means valueless flag like `"G"` or `"P"`),
    /// or `None` if the flag was not present.
    pub fn get(&self, key: &str) -> Option<Option<i32>> {
        self.raw.get(key).copied()
    }

    /// Returns `true` if `key` was specified in the flag string.
    pub fn contains(&self, key: &str) -> bool {
        self.raw.contains_key(key)
    }

    #[inline]
    fn get_clamped(&self, key: &str, default: i32, min: i32, max: i32) -> i32 {
        let v = self.raw.get(key).copied().flatten().unwrap_or(default);
        v.clamp(min, max)
    }

    /// Breathiness `Hb` (default `100`, clamped to `0..=500`).
    pub fn hb(&self) -> i32 {
        self.get_clamped("Hb", 100, 0, 500)
    }

    /// Voicing `Hv` (default `100`, clamped to `0..=150`).
    pub fn hv(&self) -> i32 {
        self.get_clamped("Hv", 100, 0, 150)
    }

    /// Tension `Ht` (default `0`, clamped to `-100..=100`).
    pub fn ht(&self) -> i32 {
        self.get_clamped("Ht", 0, -100, 100)
    }

    /// Openness `Ho` (default `0`, clamped to `-100..=100`).
    pub fn ho(&self) -> i32 {
        self.get_clamped("Ho", 0, -100, 100)
    }

    /// Singer's formant `Hr` (default `0`, clamped to `-100..=100`).
    pub fn hr(&self) -> i32 {
        self.get_clamped("Hr", 0, -100, 100)
    }

    /// Gender `g` (default `0`, clamped to `-100..=100`).
    pub fn gender(&self) -> i32 {
        self.get_clamped("g", 0, -100, 100)
    }

    /// Pitch transpose `t` in tenths of a semitone (default `0`, clamped to `-120..=120`).
    pub fn t_flag(&self) -> i32 {
        self.get_clamped("t", 0, -120, 120)
    }

    /// Amplitude modulation `A` (default `0`, clamped to `-100..=100`).
    pub fn a_mod(&self) -> i32 {
        self.get_clamped("A", 0, -100, 100)
    }

    /// Growl strength `HG` (default `0`, clamped to `0..=100`).
    pub fn hg(&self) -> i32 {
        self.get_clamped("HG", 0, 0, 100)
    }

    /// Force-regeneration `G` flag (`true` if `G` is valueless or non-zero; `false` if absent or `G0`).
    pub fn force_generate(&self) -> bool {
        match self.raw.get("G") {
            None => false,
            Some(None) => true,
            Some(Some(v)) => *v != 0,
        }
    }

    /// Loop override `He` flag (`true` if `He` is valueless or non-zero; `false` if absent or `He0`).
    pub fn he_loop(&self) -> bool {
        match self.raw.get("He") {
            None => false,
            Some(None) => true,
            Some(Some(v)) => *v != 0,
        }
    }

    /// Loudness normalization `P` flag:
    /// - Absent -> `None`
    /// - Valueless `"P"` -> `Some(100)`
    /// - `"P<v>"` -> `Some(v)` (`P0` yields `Some(0)`, which callers treat as skipping norm)
    pub fn p_strength(&self) -> Option<i32> {
        self.raw.get("P").map(|opt| opt.unwrap_or(100))
    }

    /// Builds the exact normalized feature cache suffix matching `backend/resampler.py`:
    /// ASCII-sorted keys `Hb`, (optional non-zero `Ho`, `Hr`), `Ht` (`Ht{v}v3` if non-zero else `Ht0`), `Hv`, `g`.
    /// Example for empty/default flags: `"Hb100_Ht0_Hv100_g0"`.
    pub fn feature_cache_suffix(&self) -> String {
        let mut parts = Vec::with_capacity(6);
        parts.push(format!("Hb{}", self.hb()));
        let ho = self.ho();
        if ho != 0 {
            parts.push(format!("Ho{ho}"));
        }
        let hr = self.hr();
        if hr != 0 {
            parts.push(format!("Hr{hr}"));
        }
        let ht = self.ht();
        if ht != 0 {
            parts.push(format!("Ht{ht}v3"));
        } else {
            parts.push("Ht0".to_string());
        }
        parts.push(format!("Hv{}", self.hv()));
        parts.push(format!("g{}", self.gender()));
        parts.join("_")
    }

    /// Builds the exact `.hifi.bin` filename for a source WAV stem (e.g. `"note_Hb100_Ht0_Hv100_g0.hifi.bin"`).
    pub fn feature_cache_filename(&self, stem: &str) -> String {
        format!("{stem}_{}.hifi.bin", self.feature_cache_suffix())
    }
}

/// Parses an UTAU flags string into a [`UtauFlags`] struct.
pub fn parse_flags(raw: &str) -> UtauFlags {
    UtauFlags::parse(raw)
}
