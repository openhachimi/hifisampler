use crate::args::is_null_output;
use crate::cache_locks::{hnsep_disk_cache_path, HnsepCache};
use crate::config::ServerConfig;
use crate::resampler::{
    build_feature_cache_path, OnnxRunner, ResampleEngine, ResampleError, ResampleJob,
};
use crate::stats::{ResampleStats, StatsCollector};
use hifi_core::audio::{
    read_wav_with_target_rate, resolve_audio_path, write_wav, AudioBuffer, AudioError,
};
use hifi_core::cache::{CacheManager, CachedFeatures};
use hifi_core::utau::{pitch_string_to_cents, UtauFlags};
use hifi_dsp::{
    apply_tension_and_coupling_with_config, resample_mel_and_pitch, HnsepStft, MelConfig,
    MelSpectrogram, ResampleTimeParams, TensionConfig,
};
use hifi_effects::{
    apply_amplitude_modulation, apply_hg_growl, apply_lf_vocal_fry_in_place, apply_scale_inverse,
    apply_volume_and_peak_limit, peak_abs, prepare_vocal_fry_excitation, VocalFryInput,
};
use hifi_lufs::loudness_norm;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Full synthesis pipeline orchestrator backed by `hifi_core`, `hifi_lufs`, `hifi_dsp`,
/// `hifi_effects`, and an `OnnxRunner`.
pub struct RealResampleEngine {
    pub config: ServerConfig,
    pub cache_manager: CacheManager,
    pub hnsep_cache: HnsepCache,
    pub mel_analyzer: MelSpectrogram,
    pub hnsep_stft: HnsepStft,
    pub onnx: Arc<dyn OnnxRunner>,
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

    fn synthesize_and_save(
        &self,
        job: &ResampleJob,
        features: &CachedFeatures,
    ) -> Result<(usize, f64, f64), ResampleError> {
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

        let (mel_for_vocoder, f0_for_vocoder) = match fry_ctx {
            Some(ref ctx) => (ctx.mel_mod.view(), ctx.f0_mod.as_slice()),
            None => (resampled.mel_render.view(), resampled.f0_render.as_slice()),
        };

        let t_synth0 = Instant::now();
        let wav_con = self.onnx.run_vocoder(mel_for_vocoder, f0_for_vocoder)?;

        let mut render = if wav_con.is_empty() {
            Vec::new()
        } else {
            let cut_l = resampled.cut_l.min(wav_con.len().saturating_sub(1));
            let cut_r = resampled.cut_r.max(cut_l + 1).min(wav_con.len());
            wav_con[cut_l..cut_r].to_vec()
        };
        let synthesis_ms = t_synth0.elapsed().as_secs_f64() * 1000.0;

        let t_post0 = Instant::now();
        let a_flag = job.flags.a_mod();
        if a_flag != 0 {
            let thop = self.config.hop_size as f64 / self.config.sample_rate as f64;
            apply_amplitude_modulation(
                &mut render,
                a_flag,
                &resampled.pitch_render,
                &resampled.t_mel,
                thop,
                resampled.new_start,
                resampled.new_end,
            );
        }

        apply_scale_inverse(&mut render, features.scale);

        let hg = job.flags.hg();
        if hg > 0 {
            render = apply_hg_growl(&render, self.config.sample_rate, hg);
        }

        if self.config.wave_norm {
            if let Some(p_strength) = job.flags.p_strength() {
                if p_strength > 0 {
                    render = loudness_norm(
                        &render,
                        self.config.sample_rate,
                        -16.0,
                        p_strength as f64,
                        self.config.trim_silence,
                        self.config.silence_threshold as f64,
                    )
                    .map_err(|e| ResampleError::Internal(e.to_string()))?;
                }
            }
        }

        if let Some(ref ctx) = fry_ctx {
            apply_lf_vocal_fry_in_place(&mut render, Some(ctx), self.config.sample_rate);
        }

        apply_volume_and_peak_limit(&mut render, job.volume, self.config.peak_limit as f64);

        let output_samples = render.len();
        let out_buf = AudioBuffer::new(render, self.config.sample_rate, 1);
        write_wav(Path::new(&job.out_file), &out_buf)
            .map_err(|e| ResampleError::Internal(e.to_string()))?;

        let postprocess_ms = t_post0.elapsed().as_secs_f64() * 1000.0;
        Ok((output_samples, synthesis_ms, postprocess_ms))
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
}
