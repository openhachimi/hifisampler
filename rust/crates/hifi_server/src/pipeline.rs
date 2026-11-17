use crate::args::is_null_output;
use crate::cache_locks::{hnsep_disk_cache_path, HnsepCache};
use crate::config::ServerConfig;
use crate::phrase_mgr::{
    cleanup_legacy_whd_dat, is_sidecar_fresh_for_wav, load_note_sidecar, normalize_path_key,
    note_sidecar_path, save_note_sidecar, synthesize_phrase_steps, write_stub_wav, NotePrepCache,
    PhraseBatchRequest, PhraseSessionStore, PhraseStepKind, PhraseStepRecord, PreparedNote,
    WavtoolStepRequest, MAX_PHRASE_DURATION_MS,
};
use crate::resampler::{
    build_feature_cache_path, OnnxRunner, ResampleEngine, ResampleError, ResampleJob,
};
use crate::stats::{ResampleStats, StatsCollector};
use hifi_core::audio::{
    read_wav_with_target_rate, resolve_audio_path, write_wav, AudioBuffer, AudioError,
};
use hifi_core::cache::{CacheManager, CachedFeatures};
use hifi_core::utau::{pitch_string_to_cents, UtauFlags};
use hifi_core::wavtool::WavtoolEnvelope;
use hifi_dsp::{
    apply_tension_and_coupling_with_config, resample_mel_and_pitch, HnsepStft, MelConfig,
    MelSpectrogram, ResampleTimeParams, TensionConfig,
};
use hifi_effects::{
    apply_amplitude_modulation, apply_hg_growl, apply_lf_vocal_fry_in_place, apply_scale_inverse,
    apply_volume_and_peak_limit, peak_abs, prepare_vocal_fry_excitation, VocalFryInput,
};
use hifi_lufs::{compute_lufs_target_gain, loudness_norm, LoudnessNormOptions};
use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

const MAX_STEP_DURATION_MS: f64 = 600_000.0;
const MAX_PHRASE_STEPS: usize = 4_096;

#[inline]
fn rms_energy(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

fn validate_phrase_output_path(raw: &str) -> Result<PathBuf, ResampleError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ResampleError::BadInput(
            "Empty phrase output_file path".into(),
        ));
    }
    if is_null_output(trimmed) {
        return Ok(PathBuf::from(trimmed));
    }
    let path = PathBuf::from(trimmed);
    let is_wav = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
    if !is_wav {
        return Err(ResampleError::BadInput(format!(
            "Phrase output_file must have a .wav extension or be a null device, got '{trimmed}'"
        )));
    }
    Ok(path)
}

fn validate_step_timing(stp_ms: f64, dur_ms: f64) -> Result<(), ResampleError> {
    if !stp_ms.is_finite() || !dur_ms.is_finite() {
        return Err(ResampleError::BadInput(
            "stp_ms and dur_ms must be finite numbers".into(),
        ));
    }
    if !(0.0..=MAX_STEP_DURATION_MS).contains(&dur_ms) {
        return Err(ResampleError::BadInput(format!(
            "dur_ms out of bounds [0, {MAX_STEP_DURATION_MS}]: {dur_ms}"
        )));
    }
    if !(-60_000.0..=MAX_STEP_DURATION_MS).contains(&stp_ms) {
        return Err(ResampleError::BadInput(format!(
            "stp_ms out of bounds [-60000, {MAX_STEP_DURATION_MS}]: {stp_ms}"
        )));
    }
    Ok(())
}

/// Full synthesis pipeline orchestrator backed by `hifi_core`, `hifi_lufs`, `hifi_dsp`,
/// `hifi_effects`, and an `OnnxRunner`.
pub struct RealResampleEngine {
    pub config: ServerConfig,
    pub cache_manager: CacheManager,
    pub hnsep_cache: HnsepCache,
    pub mel_analyzer: MelSpectrogram,
    pub hnsep_stft: HnsepStft,
    pub onnx: Arc<dyn OnnxRunner>,
    pub note_prep_cache: Mutex<NotePrepCache>,
    pub phrase_sessions: Mutex<PhraseSessionStore>,
}

impl RealResampleEngine {
    pub fn new(config: ServerConfig, onnx: Arc<dyn OnnxRunner>) -> Result<Self, String> {
        let mel_analyzer = MelSpectrogram::new(MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            win_size: config.win_size,
            hop_length: config.origin_hop_size,
            f_min: config.mel_fmin as f64,
            f_max: config.mel_fmax as f64,
            n_mels: config.n_mels,
            center: false,
        })
        .map_err(|e| format!("Failed to initialize MelSpectrogram: {e}"))?;

        let hnsep_stft = HnsepStft::new(config.n_fft, config.hop_size)
            .map_err(|e| format!("Failed to initialize HnsepStft: {e}"))?;

        Ok(Self {
            config,
            cache_manager: CacheManager::new(),
            hnsep_cache: HnsepCache::default(),
            mel_analyzer,
            hnsep_stft,
            onnx,
            note_prep_cache: Mutex::new(NotePrepCache::default()),
            phrase_sessions: Mutex::new(PhraseSessionStore::default()),
        })
    }

    fn needs_hnsep(flags: &UtauFlags) -> bool {
        flags.ht() != 0 || flags.ho() != 0 || flags.hr() != 0 || flags.hb() != flags.hv()
    }

    fn read_wave_cached(&self, in_file: &Path) -> Result<Arc<AudioBuffer>, ResampleError> {
        let resolved = resolve_audio_path(in_file).map_err(|e| match e {
            AudioError::FileNotFound(_) => {
                ResampleError::InputNotFound(in_file.display().to_string())
            }
            other => ResampleError::Internal(other.to_string()),
        })?;

        if let Some(cached) = self.cache_manager.load_wav_cache_arc(&resolved) {
            return Ok(cached);
        }

        let buf = Arc::new(
            read_wav_with_target_rate(&resolved, Some(self.config.sample_rate)).map_err(
                |e| match e {
                    AudioError::FileNotFound(_) => {
                        ResampleError::InputNotFound(resolved.display().to_string())
                    }
                    other => ResampleError::Internal(other.to_string()),
                },
            )?,
        );
        self.cache_manager
            .save_wav_cache_arc(&resolved, Arc::clone(&buf));
        Ok(buf)
    }

    fn get_features(
        &self,
        job: &ResampleJob,
    ) -> Result<(Arc<CachedFeatures>, bool, bool, usize), ResampleError> {
        let features_path = build_feature_cache_path(&job.in_file, &job.flags);
        let force_generate = job.flags.force_generate();
        let lock = self.cache_manager.get_lock(&features_path);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());

        if let Some(hit) = self.cache_manager.load_features_cache_arc(
            &features_path,
            force_generate,
            Some(&job.in_file),
        ) {
            return Ok((hit, true, false, 0));
        }

        tracing::info!(
            "{} not found or forcing generation. Generating features.",
            features_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("cache")
        );

        let wave_buf = self.read_wave_cached(&job.in_file)?;
        let raw_wave: &[f32] = &wave_buf.samples;
        let input_samples = raw_wave.len();
        let mut wave: Cow<'_, [f32]> = Cow::Borrowed(raw_wave);

        let breath = job.flags.hb();
        let voicing = job.flags.hv();
        let tension = job.flags.ht();
        let openness = job.flags.ho();
        let singer_formant = job.flags.hr();
        let gender = job.flags.gender();
        let mut hnsep_cache_hit = false;

        if Self::needs_hnsep(&job.flags) {
            let hnsep_path = hnsep_disk_cache_path(&job.in_file);
            let hlock = self.cache_manager.get_lock(&hnsep_path);
            let harmonic: Arc<Vec<f32>> = {
                let _hguard = hlock.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(cached_h) =
                    self.hnsep_cache
                        .load(&hnsep_path, &job.in_file, force_generate)
                {
                    hnsep_cache_hit = true;
                    cached_h
                } else {
                    let prepared = self
                        .hnsep_stft
                        .prepare_input(raw_wave)
                        .map_err(|e| ResampleError::Internal(e.to_string()))?;
                    let mask = self.onnx.run_hnsep_mask(&prepared.tensor)?;
                    let separated = self
                        .hnsep_stft
                        .reconstruct_from_mask(&prepared, mask.view())
                        .map_err(|e| ResampleError::Internal(e.to_string()))?;
                    self.hnsep_cache.save(&hnsep_path, &job.in_file, separated)
                }
            };

            let harmonic_slice: &[f32] = &harmonic;
            let aperiodic: Vec<f32> = raw_wave
                .iter()
                .zip(harmonic_slice.iter())
                .map(|(&w, &h)| w - h)
                .collect();

            if tension != 0 || openness != 0 || singer_formant != 0 {
                let b_tension = -(tension as f64) / 50.0;
                let tension_cfg = TensionConfig {
                    sample_rate: self.config.sample_rate,
                    n_fft: self.config.n_fft,
                    hop_size: self.config.hop_size,
                    win_size: self.config.win_size,
                };
                let (seg_mod, ap_mod) = apply_tension_and_coupling_with_config(
                    harmonic_slice,
                    &aperiodic,
                    b_tension,
                    openness as f64,
                    singer_formant as f64,
                    voicing > 0,
                    breath > 0,
                    &tension_cfg,
                )
                .map_err(|e| ResampleError::Internal(e.to_string()))?;
                let b_scale = breath as f32 / 100.0;
                let v_scale = voicing as f32 / 100.0;
                wave = Cow::Owned(
                    ap_mod
                        .iter()
                        .zip(seg_mod.iter())
                        .map(|(&a, &s)| b_scale * a + v_scale * s)
                        .collect(),
                );
            } else {
                let b_scale = breath as f32 / 100.0;
                let v_scale = voicing as f32 / 100.0;
                wave = Cow::Owned(
                    aperiodic
                        .iter()
                        .zip(harmonic_slice.iter())
                        .map(|(&a, &h)| b_scale * a + v_scale * h)
                        .collect(),
                );
            }
        } else if breath != 100 || voicing != 100 {
            let scale_factor = breath as f32 / 100.0;
            wave = Cow::Owned(raw_wave.iter().map(|&x| x * scale_factor).collect());
        }

        let wave_max = peak_abs(&wave);
        let scale = if wave_max >= 0.5 {
            let s = 0.5 / wave_max;
            let s32 = s as f32;
            for x in wave.to_mut().iter_mut() {
                *x *= s32;
            }
            s
        } else {
            1.0
        };

        let mel_origin = self
            .mel_analyzer
            .compute_compressed_with_gender(&wave, gender as f64, 1.0)
            .map_err(|e| ResampleError::Internal(e.to_string()))?;

        let saved = self
            .cache_manager
            .save_features_cache_arc(
                &features_path,
                &mel_origin,
                scale,
                force_generate,
                Some(&job.in_file),
            )
            .map_err(|e| ResampleError::Internal(e.to_string()))?;

        Ok((saved, false, hnsep_cache_hit, input_samples))
    }

    fn build_prepared_note(
        &self,
        job: &ResampleJob,
        features: &CachedFeatures,
        compute_p_gain_from_source: bool,
    ) -> Result<PreparedNote, ResampleError> {
        let pitchbend_i32 = pitch_string_to_cents(&job.pitch_string)
            .map_err(|e| ResampleError::BadInput(e.to_string()))?;

        let time_params = ResampleTimeParams {
            sample_rate: self.config.sample_rate,
            hop_size: self.config.hop_size,
            origin_hop_size: self.config.origin_hop_size,
            fill: self.config.fill,
            loop_mode: self.config.loop_mode || job.flags.he_loop(),
            base_pitch_midi: job.pitch_midi,
            velocity: job.velocity,
            offset_ms: job.offset_ms,
            length_ms: job.length_ms as f64,
            consonant_ms: job.consonant_ms,
            cutoff_ms: job.cutoff_ms,
            t_flag: job.flags.t_flag() as f64,
            tempo: job.tempo,
            modulation: job.modulation,
        };

        let resampled =
            resample_mel_and_pitch(features.mel_origin.view(), &pitchbend_i32, &time_params)
                .map_err(|e| ResampleError::Internal(e.to_string()))?;

        let hg_active = job.flags.hg() > 0;
        let fry_ctx = if self.config.auto_vocal_fry && !hg_active {
            let pb_f64: Vec<f64> = pitchbend_i32.iter().map(|&c| c as f64).collect();
            prepare_vocal_fry_excitation(&VocalFryInput {
                pitchbend: &pb_f64,
                base_pitch_midi: job.pitch_midi,
                t_flag_val: job.flags.t_flag() as f64,
                tempo: job.tempo,
                new_start: resampled.new_start,
                new_end: resampled.new_end,
                t_mel: &resampled.t_mel,
                f0_render: &resampled.f0_render,
                mel_render: resampled.mel_render.view(),
                sr: self.config.sample_rate,
                n_fft: self.config.n_fft,
                mel_fmin: self.config.mel_fmin as f64,
                mel_fmax: self.config.mel_fmax as f64,
                new_con: Some(resampled.new_con),
            })
        } else {
            None
        };

        let p_gain = if compute_p_gain_from_source && self.config.wave_norm {
            if let Some(p_strength) = job.flags.p_strength() {
                if p_strength > 0 {
                    let wave_buf = self.read_wave_cached(&job.in_file)?;
                    let sr = self.config.sample_rate as f64;
                    let total_sec = wave_buf.samples.len() as f64 / sr;
                    let offset_sec = job.offset_ms / 1000.0;
                    let cutoff_sec = job.cutoff_ms / 1000.0;
                    let end_sec = if job.cutoff_ms < 0.0 {
                        offset_sec - cutoff_sec
                    } else {
                        total_sec - cutoff_sec
                    };
                    let start_idx = ((offset_sec * sr).round() as isize)
                        .clamp(0, wave_buf.samples.len() as isize)
                        as usize;
                    let end_idx = ((end_sec * sr).round() as isize)
                        .clamp(start_idx as isize, wave_buf.samples.len() as isize)
                        as usize;
                    if end_idx > start_idx {
                        let opts = LoudnessNormOptions {
                            target_loudness: -16.0,
                            strength: p_strength as f64,
                            trim_silence: self.config.trim_silence,
                            silence_threshold: self.config.silence_threshold as f64,
                            ..LoudnessNormOptions::default()
                        };
                        compute_lufs_target_gain(
                            &wave_buf.samples[start_idx..end_idx],
                            self.config.sample_rate,
                            opts,
                        )
                        .map_err(|e| ResampleError::Internal(e.to_string()))?
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        let a_mod_pitch = Arc::new(resampled.pitch_render);
        let (mel_render, f0_render, pitch_render) = match fry_ctx {
            Some(ref ctx) => {
                let effective_midi: Vec<f64> = ctx
                    .f0_mod
                    .iter()
                    .map(|&hz| 69.0 + 12.0 * (hz.max(1.0) as f64 / 440.0).log2())
                    .collect();
                (
                    Arc::new(ctx.mel_mod.clone()),
                    ctx.f0_mod.clone(),
                    Arc::new(effective_midi),
                )
            }
            None => (
                Arc::new(resampled.mel_render),
                resampled.f0_render,
                Arc::clone(&a_mod_pitch),
            ),
        };

        Ok(PreparedNote {
            mel_render,
            f0_render,
            pitch_render,
            a_mod_pitch,
            t_mel: resampled.t_mel,
            new_start: resampled.new_start,
            new_end: resampled.new_end,
            cut_l: resampled.cut_l,
            cut_r: resampled.cut_r,
            scale: features.scale,
            p_gain,
            volume: job.volume,
            a_mod: job.flags.a_mod(),
            hg: job.flags.hg(),
            fry_ctx,
        })
    }

    fn synthesize_and_save(
        &self,
        job: &ResampleJob,
        features: &CachedFeatures,
    ) -> Result<(usize, f64, f64), ResampleError> {
        let mut prepared = self.build_prepared_note(job, features, false)?;

        let t_synth0 = Instant::now();
        let wav_con = self
            .onnx
            .run_vocoder(prepared.mel_render.view(), &prepared.f0_render)?;

        let mut render = if wav_con.is_empty() {
            Vec::new()
        } else {
            let cut_l = prepared.cut_l.min(wav_con.len().saturating_sub(1));
            let cut_r = prepared.cut_r.max(cut_l + 1).min(wav_con.len());
            wav_con[cut_l..cut_r].to_vec()
        };
        let synthesis_ms = t_synth0.elapsed().as_secs_f64() * 1000.0;

        let t_post0 = Instant::now();
        if prepared.a_mod != 0 {
            let thop = self.config.hop_size as f64 / self.config.sample_rate as f64;
            apply_amplitude_modulation(
                &mut render,
                prepared.a_mod,
                &prepared.a_mod_pitch,
                &prepared.t_mel,
                thop,
                prepared.new_start,
                prepared.new_end,
            );
        }

        apply_scale_inverse(&mut render, features.scale);

        if prepared.hg > 0 {
            render = apply_hg_growl(&render, self.config.sample_rate, prepared.hg);
        }

        if self.config.wave_norm {
            if let Some(p_strength) = job.flags.p_strength() {
                if p_strength > 0 {
                    let rms_before = rms_energy(&render);
                    render = loudness_norm(
                        &render,
                        self.config.sample_rate,
                        -16.0,
                        p_strength as f64,
                        self.config.trim_silence,
                        self.config.silence_threshold as f64,
                    )
                    .map_err(|e| ResampleError::Internal(e.to_string()))?;
                    let rms_after = rms_energy(&render);
                    if rms_before > 1e-9 {
                        prepared.p_gain = (rms_after / rms_before).clamp(0.01, 100.0);
                    }
                }
            }
        }

        if let Some(ref ctx) = prepared.fry_ctx {
            apply_lf_vocal_fry_in_place(&mut render, Some(ctx), self.config.sample_rate);
        }

        apply_volume_and_peak_limit(&mut render, job.volume, self.config.peak_limit as f64);

        let output_samples = render.len();
        let out_buf = AudioBuffer::new(render, self.config.sample_rate, 1);
        write_wav(Path::new(&job.out_file), &out_buf)
            .map_err(|e| ResampleError::Internal(e.to_string()))?;

        // Keep prepared note in L1 RAM (never write .melos sidecar on single-note POST /) so
        // POSIX UnixWavtool can still splice in Mel domain without polluting E2E output folders.
        {
            let mut cache = self
                .note_prep_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(Path::new(&job.out_file), Arc::new(prepared), false);
        }

        let postprocess_ms = t_post0.elapsed().as_secs_f64() * 1000.0;
        Ok((output_samples, synthesis_ms, postprocess_ms))
    }

    fn resolve_wavtool_step_kind(
        &self,
        req: &WavtoolStepRequest,
    ) -> Result<PhraseStepKind, ResampleError> {
        let trimmed_in = req.input_file.trim();
        if trimmed_in.is_empty() || is_null_output(trimmed_in) {
            return Ok(PhraseStepKind::Silence);
        }

        let input_path = Path::new(trimmed_in);

        // 1. Check L1 RAM NotePrepCache
        {
            let mut cache = self
                .note_prep_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some((prepared, is_stub)) = cache.get(input_path) {
                return Ok(PhraseStepKind::MelNote {
                    prepared,
                    note_out_file: is_stub.then(|| input_path.to_path_buf()),
                    needs_backfill: is_stub,
                });
            }
        }

        // 2. Check sibling `<out_file>.melos` binary sidecar on disk (rejecting stale sidecars)
        let sidecar = note_sidecar_path(input_path);
        if is_sidecar_fresh_for_wav(input_path, &sidecar) {
            match load_note_sidecar(&sidecar) {
                Ok(Some(note)) => {
                    let prepared = Arc::new(note);
                    let is_stub = fs::metadata(input_path)
                        .map(|m| m.len() <= 256)
                        .unwrap_or(true);
                    let mut cache = self
                        .note_prep_cache
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    cache.insert(input_path, Arc::clone(&prepared), is_stub);
                    return Ok(PhraseStepKind::MelNote {
                        prepared,
                        note_out_file: is_stub.then(|| input_path.to_path_buf()),
                        needs_backfill: is_stub,
                    });
                }
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!("Ignoring corrupt sidecar '{}': {err}", sidecar.display());
                }
            }
        } else if sidecar.exists() {
            tracing::info!(
                "Removing stale .melos sidecar '{}' because '{}' is newer.",
                sidecar.display(),
                input_path.display()
            );
            let _ = fs::remove_file(&sidecar);
        }

        // 3. Fallback: recover Mel+F0 directly from temp.bat resampler args only if WAV is missing or stub
        let is_missing_or_stub = !input_path.is_file()
            || fs::metadata(input_path)
                .map(|m| m.len() <= 256)
                .unwrap_or(false);
        if is_missing_or_stub {
            if let Some(ref fallback_args) = req.fallback_resampler_args {
                let job = ResampleJob::from_raw_args(fallback_args)?;
                let (features, _, _, _) = self.get_features(&job)?;
                let prepared = Arc::new(self.build_prepared_note(&job, &features, true)?);
                let is_stub = fs::metadata(input_path)
                    .map(|m| m.len() <= 256)
                    .unwrap_or(false);
                let _ = save_note_sidecar(&sidecar, &prepared);
                let mut cache = self
                    .note_prep_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                cache.insert(input_path, Arc::clone(&prepared), is_stub);
                return Ok(PhraseStepKind::MelNote {
                    prepared,
                    note_out_file: is_stub.then(|| input_path.to_path_buf()),
                    needs_backfill: is_stub,
                });
            }
        }

        // 4. External resampler WAV or OpenUtau `$direct=True` source WAV
        let resolved = resolve_audio_path(input_path).map_err(|e| match e {
            AudioError::FileNotFound(_) => {
                ResampleError::InputNotFound(input_path.display().to_string())
            }
            other => ResampleError::Internal(other.to_string()),
        })?;
        Ok(PhraseStepKind::DirectAudio { wav_path: resolved })
    }

    fn execute_phrase_synthesis(
        &self,
        output_path: &Path,
        steps: &[PhraseStepRecord],
    ) -> Result<(usize, f64, f64), ResampleError> {
        let mut synthesis_ms = 0.0_f64;
        let t_post_total0 = Instant::now();

        let rendered = synthesize_phrase_steps(
            steps,
            self.config.n_mels,
            self.config.hop_size,
            self.config.sample_rate,
            self.config.peak_limit as f64,
            |mel, f0| {
                let t0 = Instant::now();
                let wav = self
                    .onnx
                    .run_vocoder(mel.view(), f0)
                    .map_err(|e| e.to_string())?;
                synthesis_ms = t0.elapsed().as_secs_f64() * 1000.0;
                Ok(wav)
            },
        )
        .map_err(ResampleError::Internal)?;

        let total_elapsed_ms = t_post_total0.elapsed().as_secs_f64() * 1000.0;
        let postprocess_ms = (total_elapsed_ms - synthesis_ms).max(0.0);

        let out_lock = self.cache_manager.get_lock(output_path);
        let _out_guard = out_lock.lock().unwrap_or_else(|e| e.into_inner());

        cleanup_legacy_whd_dat(output_path);
        if !is_null_output(&output_path.to_string_lossy()) {
            let out_buf = AudioBuffer::new(rendered.waveform.clone(), self.config.sample_rate, 1);
            write_wav(output_path, &out_buf).map_err(|e| ResampleError::Internal(e.to_string()))?;
        }

        if !rendered.backfills.is_empty() {
            let mut seen = std::collections::HashSet::with_capacity(rendered.backfills.len());
            for (note_path, samples) in rendered.backfills {
                if !seen.insert(normalize_path_key(&note_path)) {
                    continue;
                }
                let is_wav = note_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
                if !is_wav {
                    continue;
                }
                let note_buf = AudioBuffer::new(samples, self.config.sample_rate, 1);
                write_wav(&note_path, &note_buf).map_err(|e| {
                    ResampleError::Internal(format!(
                        "Failed to backfill stub WAV '{}': {e}",
                        note_path.display()
                    ))
                })?;
                let sidecar = note_sidecar_path(&note_path);
                if sidecar.is_file() {
                    let touch_time = fs::metadata(&note_path)
                        .and_then(|m| m.modified())
                        .map(|mt| mt.max(SystemTime::now()))
                        .unwrap_or_else(|_| SystemTime::now());
                    if let Ok(f) = fs::File::options().write(true).open(&sidecar) {
                        let _ = f.set_modified(touch_time);
                    }
                }
                let mut cache = self
                    .note_prep_cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                cache.mark_backfilled(&note_path);
            }
        }

        Ok((rendered.waveform.len(), synthesis_ms, postprocess_ms))
    }
}

impl ResampleEngine for RealResampleEngine {
    fn resample(&self, raw_args: &[String], stats: &StatsCollector) -> Result<(), ResampleError> {
        let t0 = Instant::now();
        let job = ResampleJob::from_raw_args(raw_args)?;

        let t_feat0 = Instant::now();
        let (features, cache_hit, hnsep_cache_hit, input_samples) = self.get_features(&job)?;
        let feature_ms = t_feat0.elapsed().as_secs_f64() * 1000.0;

        if is_null_output(&job.out_file) {
            tracing::info!("Null output file. Skipping...");
            return Ok(());
        }

        let (output_samples, synthesis_ms, postprocess_ms) =
            self.synthesize_and_save(&job, &features)?;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let note_name = job
            .in_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("note")
            .to_string();

        stats.record(ResampleStats::success(
            note_name,
            feature_ms,
            synthesis_ms,
            postprocess_ms,
            total_ms,
            cache_hit,
            hnsep_cache_hit,
            input_samples,
            output_samples,
        ));
        Ok(())
    }

    fn prepare_phrase_note(
        &self,
        raw_args: &[String],
        stats: &StatsCollector,
    ) -> Result<(), ResampleError> {
        let t0 = Instant::now();
        let job = ResampleJob::from_raw_args(raw_args)?;

        let t_feat0 = Instant::now();
        let (features, cache_hit, hnsep_cache_hit, input_samples) = self.get_features(&job)?;
        let feature_ms = t_feat0.elapsed().as_secs_f64() * 1000.0;

        if is_null_output(&job.out_file) {
            tracing::info!("Null output file in /phrase/prepare. Skipping...");
            return Ok(());
        }

        let t_prep0 = Instant::now();
        let prepared = Arc::new(self.build_prepared_note(&job, &features, true)?);
        let out_path = Path::new(&job.out_file);
        let sidecar = note_sidecar_path(out_path);
        write_stub_wav(out_path, self.config.sample_rate).map_err(ResampleError::Internal)?;
        save_note_sidecar(&sidecar, &prepared)
            .map_err(|e| ResampleError::Internal(e.to_string()))?;

        {
            let mut cache = self
                .note_prep_cache
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(out_path, prepared, true);
        }

        let postprocess_ms = t_prep0.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let note_name = job
            .in_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("note");
        stats.record(ResampleStats::success(
            format!("prep:{note_name}"),
            feature_ms,
            0.0,
            postprocess_ms,
            total_ms,
            cache_hit,
            hnsep_cache_hit,
            input_samples,
            0,
        ));
        Ok(())
    }

    fn process_wavtool_step(
        &self,
        req: WavtoolStepRequest,
        stats: &StatsCollector,
    ) -> Result<(), ResampleError> {
        let t0 = Instant::now();
        let output_path = validate_phrase_output_path(&req.output_file)?;
        validate_step_timing(req.stp_ms, req.dur_ms)?;
        let envelope = WavtoolEnvelope::parse(&req.envelope).map_err(ResampleError::BadInput)?;
        let kind = self.resolve_wavtool_step_kind(&req)?;
        let step = PhraseStepRecord {
            kind,
            stp_ms: req.stp_ms,
            dur_ms: req.dur_ms,
            envelope,
        };

        let session_arc = {
            let mut store = self
                .phrase_sessions
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            store.get_or_create(&output_path, req.batch_id)
        };

        let (snapshot_steps, should_synthesize) = {
            let mut session = session_arc.lock().unwrap_or_else(|e| e.into_inner());
            let batch_changed = req.batch_id.is_some() && session.batch_id != req.batch_id;
            let stale_uncommitted = !session.steps.is_empty()
                && !session.committed
                && session.last_touched.elapsed() > Duration::from_secs(30);
            let repeat_committed_batch = session.committed
                && session.steps.is_empty()
                && req.batch_id.is_some()
                && session.batch_id == req.batch_id;
            let output_missing_restart = if req.batch_id.is_none() {
                !session.steps.is_empty() && !output_path.exists()
            } else {
                session.committed && !output_path.exists()
            };
            if batch_changed
                || stale_uncommitted
                || repeat_committed_batch
                || output_missing_restart
            {
                session.reset(req.batch_id);
            }
            if session.steps.len() >= MAX_PHRASE_STEPS {
                return Err(ResampleError::BadInput(format!(
                    "Phrase step count exceeds limit ({MAX_PHRASE_STEPS})"
                )));
            }
            if session.batch_id.is_none() {
                session.batch_id = req.batch_id;
            }
            session.last_touched = Instant::now();
            session.steps.push(step);

            let should_synth = req.commit
                || req
                    .expected_steps
                    .is_none_or(|exp| session.steps.len() >= exp);
            if should_synth {
                session.committed = true;
            }
            let cloned = if should_synth {
                let out = session.steps.clone();
                if req.commit || req.expected_steps.is_some() {
                    session.steps.clear();
                }
                out
            } else {
                Vec::new()
            };
            (cloned, should_synth)
        };

        if should_synthesize {
            let (output_samples, synthesis_ms, postprocess_ms) =
                self.execute_phrase_synthesis(&output_path, &snapshot_steps)?;
            let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let out_name = output_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("phrase");
            stats.record(ResampleStats::success(
                format!("phrase:{out_name}"),
                0.0,
                synthesis_ms,
                postprocess_ms,
                total_ms,
                false,
                false,
                snapshot_steps.len(),
                output_samples,
            ));
        }

        Ok(())
    }

    fn render_phrase_batch(
        &self,
        req: PhraseBatchRequest,
        stats: &StatsCollector,
    ) -> Result<(), ResampleError> {
        let t0 = Instant::now();
        let output_path = validate_phrase_output_path(&req.output_file)?;
        if req.notes.len() > MAX_PHRASE_STEPS {
            return Err(ResampleError::BadInput(format!(
                "Phrase batch note count {} exceeds limit ({MAX_PHRASE_STEPS})",
                req.notes.len()
            )));
        }
        let mut steps = Vec::with_capacity(req.notes.len());
        let mut cache_hits = 0usize;
        let mut cursor_ms = 0.0_f64;
        let mut max_end_ms = 0.0_f64;

        let t_feat0 = Instant::now();
        for (idx, note) in req.notes.into_iter().enumerate() {
            validate_step_timing(note.stp_ms, note.dur_ms)?;
            let envelope =
                WavtoolEnvelope::parse(&note.envelope).map_err(ResampleError::BadInput)?;
            let ovr = if envelope.ovr_ms.is_finite() {
                envelope.ovr_ms
            } else {
                0.0
            };
            let start_ms = if idx == 0 {
                0.0
            } else {
                (cursor_ms - ovr).max(0.0)
            };
            cursor_ms = start_ms + note.dur_ms;
            if cursor_ms > max_end_ms {
                max_end_ms = cursor_ms;
            }
            if max_end_ms > MAX_PHRASE_DURATION_MS {
                return Err(ResampleError::BadInput(format!(
                    "Phrase duration {:.1} ms exceeds safety limit ({:.1} ms)",
                    max_end_ms, MAX_PHRASE_DURATION_MS
                )));
            }
            let kind = if note.is_silence {
                PhraseStepKind::Silence
            } else if let Some(ref raw_args) = note.resampler_args {
                let job = ResampleJob::from_raw_args(raw_args)?;
                let (features, hit, _, _) = self.get_features(&job)?;
                if hit {
                    cache_hits += 1;
                }
                let prepared = Arc::new(self.build_prepared_note(&job, &features, true)?);
                PhraseStepKind::MelNote {
                    prepared,
                    note_out_file: None,
                    needs_backfill: false,
                }
            } else if let Some(ref wav_file) = note.wav_file {
                let resolved = resolve_audio_path(Path::new(wav_file)).map_err(|e| match e {
                    AudioError::FileNotFound(_) => ResampleError::InputNotFound(wav_file.clone()),
                    other => ResampleError::Internal(other.to_string()),
                })?;
                PhraseStepKind::DirectAudio { wav_path: resolved }
            } else {
                PhraseStepKind::Silence
            };
            steps.push(PhraseStepRecord {
                kind,
                stp_ms: note.stp_ms,
                dur_ms: note.dur_ms,
                envelope,
            });
        }
        let feature_ms = t_feat0.elapsed().as_secs_f64() * 1000.0;

        let (output_samples, synthesis_ms, postprocess_ms) =
            self.execute_phrase_synthesis(&output_path, &steps)?;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let out_name = output_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("phrase");
        stats.record(ResampleStats::success(
            format!("batch:{out_name}"),
            feature_ms,
            synthesis_ms,
            postprocess_ms,
            total_ms,
            cache_hits == steps.len() && !steps.is_empty(),
            false,
            steps.len(),
            output_samples,
        ));
        Ok(())
    }
}
