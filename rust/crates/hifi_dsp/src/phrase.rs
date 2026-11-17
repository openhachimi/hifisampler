//! Continuous phrase-level Mel & F0 splicing with partition-of-unity overlap morphing,
//! silence pitch bridging, and context padding for zero-boundary-artifact vocoding.

use crate::pitch::midi_to_hz;
use hifi_core::wavtool::WavtoolEnvelope;
use ndarray::Array2;
use std::sync::Arc;

/// Number of context frames prepended and appended around the phrase before vocoding.
pub const PHRASE_PAD_FRAMES: usize = 8;

/// Log-Mel floor corresponding to linear amplitude `1e-5`.
const MIN_LINEAR_AMP: f32 = 1.0e-5;
const LOG_SILENCE_FLOOR: f32 = -11.512925_f32; // ln(1e-5)

/// Prepared per-note acoustic slice entering phrase Mel/F0 assembly.
#[derive(Debug, Clone)]
pub struct PhraseNoteMelSlice {
    pub mel_render: Arc<Array2<f32>>,
    pub pitch_render: Arc<Vec<f64>>,
    pub new_start_sec: f64,
    pub new_end_sec: f64,
    pub stp_ms: f64,
    pub dur_ms: f64,
    pub envelope: WavtoolEnvelope,
    pub scale: f64,
    pub p_gain: f64,
    pub volume: f64,
}

/// Timeline item supplied in wavtool execution order.
#[derive(Debug, Clone)]
pub enum PhraseTimelineItem {
    MelNote(PhraseNoteMelSlice),
    DirectAudio {
        dur_ms: f64,
        envelope: WavtoolEnvelope,
    },
    Silence {
        dur_ms: f64,
        ovr_ms: f64,
    },
}

/// Placed timeline span with exact millisecond and sample boundaries.
#[derive(Debug, Clone)]
pub struct PhraseSpan {
    pub item_index: usize,
    pub start_ms: f64,
    pub end_ms: f64,
    pub dur_ms: f64,
    pub start_sample: usize,
    pub end_sample: usize,
    pub envelope: WavtoolEnvelope,
    pub is_mel_note: bool,
    pub is_direct_audio: bool,
}

/// Complete timeline layout for a phrase.
#[derive(Debug, Clone)]
pub struct PhraseLayout {
    pub items: Vec<PhraseTimelineItem>,
    pub spans: Vec<PhraseSpan>,
    pub total_duration_ms: f64,
    pub total_samples: usize,
}

/// Assembled continuous phrase Mel + F0 excitation ready for a single Vocoder call.
#[derive(Debug, Clone)]
pub struct SplicedPhraseExcitation {
    /// Padded Mel spectrogram `[n_mels, total_frames]`.
    pub mel: Array2<f32>,
    /// Padded F0 curve in Hz `[total_frames]`.
    pub f0: Vec<f32>,
    /// Global pre-Vocoder scaling factor (`<= 1.0`) preventing HiFi-GAN ceiling overload.
    /// Multiply post-vocoder waveform by `1.0 / phrase_scale` before note envelopes.
    pub phrase_scale: f64,
    /// Number of active phrase frames (excluding `PHRASE_PAD_FRAMES` front and tail pads).
    pub active_frames: usize,
    /// Sample start index to crop after Vocoder (`PHRASE_PAD_FRAMES * hop_size`).
    pub crop_start: usize,
    /// Sample end index to crop after Vocoder (`crop_start + total_samples`).
    pub crop_end: usize,
}

/// Computes exact OpenUtau/classic UTAU wavtool placement for a sequence of timeline items.
pub fn layout_phrase_timeline(items: &[PhraseTimelineItem], sample_rate: u32) -> PhraseLayout {
    let sr = sample_rate.max(1) as f64;
    let mut spans = Vec::with_capacity(items.len());
    let mut cursor_ms = 0.0_f64;
    let mut max_end_ms = 0.0_f64;

    for (idx, item) in items.iter().enumerate() {
        let (dur_raw, envelope, is_mel_note, is_direct_audio) = match item {
            PhraseTimelineItem::MelNote(s) => (s.dur_ms, s.envelope.clone(), true, false),
            PhraseTimelineItem::DirectAudio { dur_ms, envelope } => {
                (*dur_ms, envelope.clone(), false, true)
            }
            PhraseTimelineItem::Silence { dur_ms, ovr_ms } => (
                *dur_ms,
                WavtoolEnvelope {
                    ovr_ms: *ovr_ms,
                    ..WavtoolEnvelope::default()
                },
                false,
                false,
            ),
        };

        let dur_ms = if dur_raw.is_finite() {
            dur_raw.max(0.0)
        } else {
            0.0
        };
        let ovr_ms = if envelope.ovr_ms.is_finite() {
            envelope.ovr_ms
        } else {
            0.0
        };

        let start_ms = if idx == 0 {
            0.0
        } else {
            (cursor_ms - ovr_ms).max(0.0)
        };
        let end_ms = start_ms + dur_ms;
        cursor_ms = end_ms;
        if end_ms > max_end_ms {
            max_end_ms = end_ms;
        }

        let start_sample = ((start_ms * sr / 1000.0).round() as isize).max(0) as usize;
        let end_sample_raw = ((end_ms * sr / 1000.0).round() as isize).max(0) as usize;
        let end_sample = end_sample_raw.max(start_sample);

        spans.push(PhraseSpan {
            item_index: idx,
            start_ms,
            end_ms,
            dur_ms,
            start_sample,
            end_sample,
            envelope,
            is_mel_note,
            is_direct_audio,
        });
    }

    let max_span_samples = spans.iter().map(|s| s.end_sample).max().unwrap_or(0);
    let total_samples = ((max_end_ms * sr / 1000.0).round() as isize)
        .max(0)
        .max(max_span_samples as isize) as usize;

    PhraseLayout {
        items: items.to_vec(),
        spans,
        total_duration_ms: max_end_ms,
        total_samples,
    }
}

/// Interpolates a single column vector from `[n_mels, n_cols]` Mel spectrogram at fractional frame `pos`.
#[inline]
fn sample_mel_column(mel: &Array2<f32>, pos: f64, out_lin: &mut [f32]) {
    let ncols = mel.ncols();
    if ncols == 0 {
        out_lin.fill(MIN_LINEAR_AMP);
        return;
    }
    if ncols == 1 {
        for (m, dst) in out_lin.iter_mut().enumerate() {
            *dst = mel[[m, 0]].exp();
        }
        return;
    }
    let clamped = pos.clamp(0.0, (ncols - 1) as f64);
    let i0 = (clamped.floor() as usize).min(ncols - 1);
    let i1 = (i0 + 1).min(ncols - 1);
    let frac = (clamped - i0 as f64) as f32;
    let w0 = 1.0 - frac;
    for (m, dst) in out_lin.iter_mut().enumerate() {
        let log_v = mel[[m, i0]] * w0 + mel[[m, i1]] * frac;
        *dst = log_v.exp();
    }
}

/// Interpolates a 1-D MIDI pitch curve at fractional frame `pos`.
#[inline]
fn sample_pitch_at(pitch: &[f64], pos: f64) -> f64 {
    let n = pitch.len();
    if n == 0 {
        return 69.0;
    }
    if n == 1 {
        return pitch[0];
    }
    let clamped = pos.clamp(0.0, (n - 1) as f64);
    let i0 = (clamped.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    let frac = clamped - i0 as f64;
    pitch[i0] * (1.0 - frac) + pitch[i1] * frac
}

/// Bridges unvoiced gap entries in `pitch` linearly between nearest voiced frames.
fn bridge_silence_pitches(pitch: &mut [f64], voiced: &[bool]) {
    let n = pitch.len();
    if n == 0 {
        return;
    }
    let Some(first_v) = voiced.iter().position(|&v| v) else {
        pitch.fill(69.0);
        return;
    };
    let last_v = voiced.iter().rposition(|&v| v).unwrap_or(first_v);

    let head_val = pitch[first_v];
    for p in pitch[..first_v].iter_mut() {
        *p = head_val;
    }
    let tail_val = pitch[last_v];
    for p in pitch[(last_v + 1)..].iter_mut() {
        *p = tail_val;
    }

    let mut i = first_v;
    while i < last_v {
        if voiced[i] {
            i += 1;
            continue;
        }
        let gap_start = i;
        let prev_pitch = pitch[gap_start - 1];
        let mut gap_end = gap_start;
        while gap_end <= last_v && !voiced[gap_end] {
            gap_end += 1;
        }
        let next_pitch = pitch[gap_end];
        let span = (gap_end - gap_start + 1) as f64;
        for (k, slot) in pitch[gap_start..gap_end].iter_mut().enumerate() {
            let alpha = (k + 1) as f64 / span;
            *slot = prev_pitch + (next_pitch - prev_pitch) * alpha;
        }
        i = gap_end;
    }
}

/// Computes `u_k = (1 / scale_k) * p_gain_k * (volume_k / 100)` safely for a `MelNote`.
#[inline]
pub fn mel_note_unscaled_gain(slice: &PhraseNoteMelSlice) -> f64 {
    let inv_scale = if slice.scale > 1e-6 && slice.scale.is_finite() {
        1.0 / slice.scale
    } else {
        1.0
    };
    let p_gain = if slice.p_gain.is_finite() && slice.p_gain > 0.0 {
        slice.p_gain
    } else {
        1.0
    };
    let vol = if slice.volume.is_finite() {
        (slice.volume / 100.0).max(0.0)
    } else {
        1.0
    };
    inv_scale * p_gain * vol
}

/// Builds continuous spliced Mel spectrogram `[n_mels, total_frames]` and F0 curve `[total_frames]`
/// across the phrase timeline, with `PHRASE_PAD_FRAMES` context padding on both ends.
///
/// Returns `None` when the phrase contains zero `MelNote` entries or zero samples.
pub fn build_spliced_phrase_mel_and_f0(
    layout: &PhraseLayout,
    n_mels: usize,
    hop_size: usize,
    sample_rate: u32,
) -> Option<SplicedPhraseExcitation> {
    if n_mels == 0 || hop_size == 0 || sample_rate == 0 || layout.total_samples == 0 {
        return None;
    }

    struct ActiveMelEntry<'a> {
        span_idx: usize,
        slice: &'a PhraseNoteMelSlice,
        u_gain: f64,
        is_head: bool,
        is_tail: bool,
    }

    let mel_indices: Vec<usize> = layout
        .spans
        .iter()
        .enumerate()
        .filter(|(_, sp)| {
            sp.is_mel_note && sp.dur_ms > 0.0 && matches!(&layout.items[sp.item_index], PhraseTimelineItem::MelNote(s) if s.mel_render.ncols() > 0 && s.mel_render.nrows() == n_mels)
        })
        .map(|(idx, _)| idx)
        .collect();

    if mel_indices.is_empty() {
        return None;
    }

    let first_mel_span_idx = mel_indices[0];
    let last_mel_span_idx = *mel_indices.last().unwrap();

    let mut entries: Vec<ActiveMelEntry<'_>> = Vec::with_capacity(mel_indices.len());
    let mut u_max = 0.0_f64;
    for &idx in &mel_indices {
        let sp = &layout.spans[idx];
        if let PhraseTimelineItem::MelNote(ref s) = layout.items[sp.item_index] {
            let u = mel_note_unscaled_gain(s);
            if u > u_max {
                u_max = u;
            }
            entries.push(ActiveMelEntry {
                span_idx: idx,
                slice: s,
                u_gain: u,
                is_head: idx == first_mel_span_idx,
                is_tail: idx == last_mel_span_idx,
            });
        }
    }

    let phrase_scale = if u_max > 1.0 && u_max.is_finite() {
        1.0 / u_max
    } else {
        1.0
    };

    let sr = sample_rate as f64;
    let thop_sec = hop_size as f64 / sr;
    let thop_ms = thop_sec * 1000.0;
    let active_frames = layout.total_samples.div_ceil(hop_size).max(1);
    let pad = PHRASE_PAD_FRAMES;
    let total_frames = pad + active_frames + pad;

    let mut mel_out = Array2::<f32>::from_elem((n_mels, total_frames), LOG_SILENCE_FLOOR);
    let mut pitch_active = vec![69.0_f64; active_frames];
    let mut voiced_active = vec![false; active_frames];

    let mut note_lin_buf = vec![0.0_f32; n_mels];
    let mut power_acc = vec![0.0_f64; n_mels];

    // Temporary collection per frame: (entry_idx, src_pos, raw_weight)
    let mut frame_hits: Vec<(usize, f64, f64)> = Vec::with_capacity(4);

    for f in 0..active_frames {
        let f_start_ms = f as f64 * thop_ms;
        let f_end_ms = (f + 1) as f64 * thop_ms;
        let t_ms = (f as f64 + 0.5) * thop_ms;
        frame_hits.clear();
        let mut raw_sum = 0.0_f64;

        for (ei, entry) in entries.iter().enumerate() {
            let sp = &layout.spans[entry.span_idx];
            if t_ms >= sp.start_ms && t_ms <= sp.end_ms {
                let pos_ms = (t_ms - sp.start_ms).clamp(0.0, sp.dur_ms);
                let w_raw = entry.slice.envelope.evaluate_mel_blend_weight(
                    pos_ms,
                    sp.dur_ms,
                    entry.is_head,
                    entry.is_tail,
                );
                let note_render_sec =
                    entry.slice.new_start_sec + (entry.slice.stp_ms + pos_ms) / 1000.0;
                let src_pos = note_render_sec / thop_sec;
                frame_hits.push((ei, src_pos, w_raw));
                raw_sum += w_raw;
            } else if sp.dur_ms < thop_ms && sp.end_ms > f_start_ms && sp.start_ms < f_end_ms {
                let sample_t_ms = t_ms.clamp(sp.start_ms, sp.end_ms);
                let pos_ms = (sample_t_ms - sp.start_ms).clamp(0.0, sp.dur_ms);
                let overlap_frac = ((sp.end_ms.min(f_end_ms) - sp.start_ms.max(f_start_ms))
                    / thop_ms)
                    .clamp(0.05, 1.0);
                let w_raw = entry.slice.envelope.evaluate_mel_blend_weight(
                    pos_ms,
                    sp.dur_ms,
                    entry.is_head,
                    entry.is_tail,
                ) * overlap_frac;
                let note_render_sec =
                    entry.slice.new_start_sec + (entry.slice.stp_ms + pos_ms) / 1000.0;
                let src_pos = note_render_sec / thop_sec;
                frame_hits.push((ei, src_pos, w_raw));
                raw_sum += w_raw;
            }
        }

        if frame_hits.is_empty() {
            continue;
        }

        voiced_active[f] = true;
        power_acc.fill(0.0);
        let mut pitch_sum = 0.0_f64;
        let use_uniform = raw_sum <= 1e-9;
        let inv_denom = if use_uniform {
            1.0 / (frame_hits.len() as f64)
        } else {
            1.0 / raw_sum
        };

        for &(ei, src_pos, w_raw) in &frame_hits {
            let w = if use_uniform {
                inv_denom
            } else {
                w_raw * inv_denom
            };
            if w <= 1e-12 {
                continue;
            }
            let entry = &entries[ei];
            sample_mel_column(&entry.slice.mel_render, src_pos, &mut note_lin_buf);
            let scaled_u = phrase_scale * entry.u_gain;
            let coeff = w * scaled_u * scaled_u;
            for m in 0..n_mels {
                let lin = note_lin_buf[m] as f64;
                power_acc[m] += coeff * lin * lin;
            }
            pitch_sum += w * sample_pitch_at(&entry.slice.pitch_render, src_pos);
        }

        let dst_col = pad + f;
        for m in 0..n_mels {
            let amp = power_acc[m].max(0.0).sqrt().max(MIN_LINEAR_AMP as f64) as f32;
            mel_out[[m, dst_col]] = amp.ln();
        }
        pitch_active[f] = pitch_sum;
    }

    // 2-frame spectral taper into true silent Rest gaps around voiced islands.
    for f in 0..active_frames {
        if !voiced_active[f] {
            continue;
        }
        let col = pad + f;
        if f > 0 && !voiced_active[f - 1] {
            let gap_left_1 = col - 1;
            for m in 0..n_mels {
                let edge_lin = mel_out[[m, col]].exp() as f64;
                let tapered_1 = (edge_lin * 0.33).max(MIN_LINEAR_AMP as f64) as f32;
                mel_out[[m, gap_left_1]] = mel_out[[m, gap_left_1]].max(tapered_1.ln());
            }
            if f > 1 && !voiced_active[f - 2] {
                let gap_left_2 = col - 2;
                for m in 0..n_mels {
                    let edge_lin = mel_out[[m, col]].exp() as f64;
                    let tapered_2 = (edge_lin * 0.10).max(MIN_LINEAR_AMP as f64) as f32;
                    mel_out[[m, gap_left_2]] = mel_out[[m, gap_left_2]].max(tapered_2.ln());
                }
            }
        }
        if f + 1 < active_frames && !voiced_active[f + 1] {
            let gap_right_1 = col + 1;
            for m in 0..n_mels {
                let edge_lin = mel_out[[m, col]].exp() as f64;
                let tapered_1 = (edge_lin * 0.33).max(MIN_LINEAR_AMP as f64) as f32;
                mel_out[[m, gap_right_1]] = mel_out[[m, gap_right_1]].max(tapered_1.ln());
            }
            if f + 2 < active_frames && !voiced_active[f + 2] {
                let gap_right_2 = col + 2;
                for m in 0..n_mels {
                    let edge_lin = mel_out[[m, col]].exp() as f64;
                    let tapered_2 = (edge_lin * 0.10).max(MIN_LINEAR_AMP as f64) as f32;
                    mel_out[[m, gap_right_2]] = mel_out[[m, gap_right_2]].max(tapered_2.ln());
                }
            }
        }
    }

    bridge_silence_pitches(&mut pitch_active, &voiced_active);

    // Front PHRASE_PAD_FRAMES: pull real pre-onset Mel from first MelNote when available, else edge repeat.
    let head_entry = &entries[0];
    let head_span = &layout.spans[head_entry.span_idx];
    let head_scaled_u = (phrase_scale * head_entry.u_gain).max(0.0);
    for p in 0..pad {
        let t_ms = 0.5 * thop_ms - ((pad - p) as f64) * thop_ms;
        if head_span.start_ms <= thop_ms {
            let pos_ms = t_ms - head_span.start_ms;
            let src_pos = (head_entry.slice.new_start_sec
                + (head_entry.slice.stp_ms + pos_ms) / 1000.0)
                / thop_sec;
            sample_mel_column(&head_entry.slice.mel_render, src_pos, &mut note_lin_buf);
            for m in 0..n_mels {
                let amp =
                    ((note_lin_buf[m] as f64) * head_scaled_u).max(MIN_LINEAR_AMP as f64) as f32;
                mel_out[[m, p]] = amp.ln();
            }
        } else {
            // Leading gap before first MelNote: repeat frame 0 floor
            for m in 0..n_mels {
                mel_out[[m, p]] = mel_out[[m, pad]];
            }
        }
    }

    // Tail PHRASE_PAD_FRAMES: pull real post-offset Mel from last MelNote when available, else edge repeat.
    let tail_entry = entries.last().unwrap();
    let tail_span = &layout.spans[tail_entry.span_idx];
    let tail_scaled_u = (phrase_scale * tail_entry.u_gain).max(0.0);
    for q in 0..pad {
        let f = active_frames + q;
        let dst_col = pad + f;
        let t_ms = (f as f64 + 0.5) * thop_ms;
        if tail_span.end_ms + thop_ms >= layout.total_duration_ms {
            let pos_ms = t_ms - tail_span.start_ms;
            let src_pos = (tail_entry.slice.new_start_sec
                + (tail_entry.slice.stp_ms + pos_ms) / 1000.0)
                / thop_sec;
            sample_mel_column(&tail_entry.slice.mel_render, src_pos, &mut note_lin_buf);
            for m in 0..n_mels {
                let amp =
                    ((note_lin_buf[m] as f64) * tail_scaled_u).max(MIN_LINEAR_AMP as f64) as f32;
                mel_out[[m, dst_col]] = amp.ln();
            }
        } else {
            let last_col = pad + active_frames - 1;
            for m in 0..n_mels {
                mel_out[[m, dst_col]] = mel_out[[m, last_col]];
            }
        }
    }

    // Build full F0 curve with damped tangent extrapolation across PHRASE_PAD_FRAMES.
    let mut f0_out = Vec::with_capacity(total_frames);
    let slope_front = if active_frames >= 2 {
        pitch_active[1] - pitch_active[0]
    } else {
        0.0
    };
    let slope_tail = if active_frames >= 2 {
        pitch_active[active_frames - 1] - pitch_active[active_frames - 2]
    } else {
        0.0
    };

    for p in 0..pad {
        let d = (pad - p) as f64;
        let midi = pitch_active[0] - (0.5 * slope_front * d).clamp(-2.0, 2.0);
        f0_out.push(midi_to_hz(midi) as f32);
    }
    for &midi in &pitch_active {
        f0_out.push(midi_to_hz(midi) as f32);
    }
    let last_pitch = pitch_active[active_frames - 1];
    for q in 0..pad {
        let d = (q + 1) as f64;
        let midi = last_pitch + (0.5 * slope_tail * d).clamp(-2.0, 2.0);
        f0_out.push(midi_to_hz(midi) as f32);
    }

    let crop_start = pad * hop_size;
    let crop_end = crop_start + layout.total_samples;

    Some(SplicedPhraseExcitation {
        mel: mel_out,
        f0: f0_out,
        phrase_scale,
        active_frames,
        crop_start,
        crop_end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_constant_slice(
        n_mels: usize,
        n_frames: usize,
        log_val: f32,
        pitch_midi: f64,
        dur_ms: f64,
        envelope: WavtoolEnvelope,
    ) -> PhraseNoteMelSlice {
        PhraseNoteMelSlice {
            mel_render: Arc::new(Array2::from_elem((n_mels, n_frames), log_val)),
            pitch_render: Arc::new(vec![pitch_midi; n_frames]),
            new_start_sec: 0.0,
            new_end_sec: dur_ms / 1000.0,
            stp_ms: 0.0,
            dur_ms,
            envelope,
            scale: 1.0,
            p_gain: 1.0,
            volume: 100.0,
        }
    }

    #[test]
    fn timeline_layout_tracks_overlap_and_silence_cursor() {
        let sr = 44100;
        let e1 = WavtoolEnvelope::default();
        let e2 = WavtoolEnvelope {
            ovr_ms: 50.0,
            ..WavtoolEnvelope::default()
        };
        let e3 = WavtoolEnvelope {
            ovr_ms: -20.0,
            ..WavtoolEnvelope::default()
        };

        let items = vec![
            PhraseTimelineItem::MelNote(make_constant_slice(4, 50, -1.0, 60.0, 500.0, e1)),
            PhraseTimelineItem::MelNote(make_constant_slice(4, 50, -1.0, 64.0, 400.0, e2)),
            PhraseTimelineItem::Silence {
                dur_ms: 100.0,
                ovr_ms: 0.0,
            },
            PhraseTimelineItem::MelNote(make_constant_slice(4, 50, -1.0, 67.0, 300.0, e3)),
        ];

        let layout = layout_phrase_timeline(&items, sr);
        assert_abs_diff_eq!(layout.spans[0].start_ms, 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(layout.spans[0].end_ms, 500.0, epsilon = 1e-9);
        assert_abs_diff_eq!(layout.spans[1].start_ms, 450.0, epsilon = 1e-9);
        assert_abs_diff_eq!(layout.spans[1].end_ms, 850.0, epsilon = 1e-9);
        assert_abs_diff_eq!(layout.spans[2].start_ms, 850.0, epsilon = 1e-9);
        assert_abs_diff_eq!(layout.spans[2].end_ms, 950.0, epsilon = 1e-9);
        // Negative overlap creates a gap: cursor 950 - (-20) = 970
        assert_abs_diff_eq!(layout.spans[3].start_ms, 970.0, epsilon = 1e-9);
        assert_abs_diff_eq!(layout.total_duration_ms, 1270.0, epsilon = 1e-9);
        assert_eq!(layout.total_samples, (1.270 * sr as f64).round() as usize);
    }

    #[test]
    fn equal_level_crossfade_preserves_constant_mel_without_dip_and_pads_context() {
        let sr = 44100u32;
        let hop = 512usize;
        let n_mels = 4usize;
        let log_v = -2.0_f32;

        let e1 = WavtoolEnvelope {
            p1_ms: 0.0,
            p2_ms: 10.0,
            p3_ms: 60.0,
            v1: 0.0,
            v2: 100.0,
            v3: 100.0,
            v4: 0.0,
            ovr_ms: 0.0,
            p4_ms: 0.0,
            p5_v5: None,
        };
        let e2 = WavtoolEnvelope {
            p1_ms: 0.0,
            p2_ms: 60.0,
            p3_ms: 10.0,
            v1: 0.0,
            v2: 100.0,
            v3: 100.0,
            v4: 0.0,
            ovr_ms: 60.0,
            p4_ms: 0.0,
            p5_v5: None,
        };

        let items = vec![
            PhraseTimelineItem::MelNote(make_constant_slice(n_mels, 60, log_v, 60.0, 300.0, e1)),
            PhraseTimelineItem::MelNote(make_constant_slice(n_mels, 60, log_v, 72.0, 300.0, e2)),
        ];

        let layout = layout_phrase_timeline(&items, sr);
        let spliced = build_spliced_phrase_mel_and_f0(&layout, n_mels, hop, sr).unwrap();

        assert_eq!(spliced.crop_start, PHRASE_PAD_FRAMES * hop);
        assert_eq!(
            spliced.crop_end,
            PHRASE_PAD_FRAMES * hop + layout.total_samples
        );
        assert_eq!(spliced.mel.nrows(), n_mels);
        assert_eq!(
            spliced.mel.ncols(),
            PHRASE_PAD_FRAMES * 2 + spliced.active_frames
        );
        assert_eq!(spliced.f0.len(), spliced.mel.ncols());
        assert!(spliced.mel.iter().all(|v| v.is_finite()));
        assert!(spliced.f0.iter().all(|v| v.is_finite() && *v > 0.0));

        // Overlap is [240ms, 300ms], midpoint 270ms (frame index around 23)
        // Equal-amplitude note crossfade must stay exact at log_v (0.0 dB dip)
        for f in 0..spliced.active_frames {
            let val = spliced.mel[[0, PHRASE_PAD_FRAMES + f]];
            assert_abs_diff_eq!(val, log_v, epsilon = 1e-4);
        }

        // At frame center t_ms within [240ms, 300ms], MIDI pitch linearly morphs between C4 (60) and C5 (72)
        let mid_frame = ((270.0 / 1000.0) * (sr as f64) / (hop as f64)).floor() as usize;
        let t_mid_ms = (mid_frame as f64 + 0.5) * (hop as f64) * 1000.0 / (sr as f64);
        let alpha = (t_mid_ms - 240.0) / 60.0;
        let mid_hz = spliced.f0[PHRASE_PAD_FRAMES + mid_frame] as f64;
        let expected_mid_hz = midi_to_hz(60.0 + 12.0 * alpha);
        assert!(
            (mid_hz - expected_mid_hz).abs() < 1e-3,
            "mid_hz={mid_hz}, expected={expected_mid_hz}"
        );
    }

    #[test]
    fn real_front_and_tail_context_and_damped_f0_extrapolation_work() {
        let sr = 44100u32;
        let hop = 512usize;
        let n_mels = 4usize;
        let n_frames = 80usize;

        // Distinct Mel markers in front pre-onset (< frame 6) and post-end frames so we can verify real context reuse
        let mut mel = Array2::<f32>::from_elem((n_mels, n_frames), -1.0_f32);
        for j in 0..6 {
            mel.column_mut(j).fill(-0.5_f32);
        }
        let mut pitch = vec![60.0_f64; n_frames];
        // Rising pitch inside active region so front extrapolation dips gently below C4
        for (k, slot) in pitch[6..40].iter_mut().enumerate() {
            *slot = 60.0 + (k as f64) * 0.5;
        }
        pitch[40..].fill(77.0);

        let new_start_sec = 6.0 * (hop as f64) / (sr as f64);
        let dur_ms = 250.0;
        let slice = PhraseNoteMelSlice {
            mel_render: Arc::new(mel),
            pitch_render: Arc::new(pitch),
            new_start_sec,
            new_end_sec: new_start_sec + dur_ms / 1000.0,
            stp_ms: 0.0,
            dur_ms,
            envelope: WavtoolEnvelope::default(),
            scale: 0.5,
            p_gain: 1.0,
            volume: 100.0,
        };

        let layout = layout_phrase_timeline(&[PhraseTimelineItem::MelNote(slice)], sr);
        let spliced = build_spliced_phrase_mel_and_f0(&layout, n_mels, hop, sr).unwrap();

        // u = 1/0.5 = 2.0 -> phrase_scale = 0.5, so Mel magnitude stays unscaled (-0.5 on real pre-context frame 7)
        assert_abs_diff_eq!(spliced.phrase_scale, 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(
            spliced.mel[[0, PHRASE_PAD_FRAMES - 2]],
            -0.5,
            epsilon = 1e-4
        );

        // Front padded F0 slopes downward into C4 without crashing or diverging
        assert!(spliced.f0[0] < spliced.f0[PHRASE_PAD_FRAMES]);
        assert!(spliced.f0[0] as f64 >= midi_to_hz(58.0) - 1.0);
    }

    #[test]
    fn sub_hop_short_grace_note_is_voiced_not_silenced() {
        let sr = 44100u32;
        let hop = 512usize;
        let n_mels = 4usize;
        let log_v = -1.5_f32;
        // 4.0 ms sub-hop note sits completely inside [0, 11.61 ms] before frame 0 center (5.805 ms)
        let slice = make_constant_slice(n_mels, 20, log_v, 64.0, 4.0, WavtoolEnvelope::default());
        let layout = layout_phrase_timeline(&[PhraseTimelineItem::MelNote(slice)], sr);
        let spliced = build_spliced_phrase_mel_and_f0(&layout, n_mels, hop, sr).unwrap();

        let frame0_mel = spliced.mel[[0, PHRASE_PAD_FRAMES]];
        assert!(
            frame0_mel > LOG_SILENCE_FLOOR + 1.0,
            "sub-hop 4ms note was silenced: mel={frame0_mel}"
        );
        assert_abs_diff_eq!(frame0_mel, log_v, epsilon = 1e-4);
        assert_abs_diff_eq!(
            spliced.f0[PHRASE_PAD_FRAMES] as f64,
            midi_to_hz(64.0),
            epsilon = 1e-2
        );
    }
}
