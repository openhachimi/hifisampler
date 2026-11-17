use hifi_onnx::{
    DevicePreference as OnnxDevicePreference, ExecutionProvider, HnsepSession, OnnxError,
    SessionConfig, VocoderSession, DEFAULT_HNSEP_ONNX_PATHS, DEFAULT_VOCODER_ONNX_PATH,
};
use hifi_server::config::{
    resolve_asset_path, DevicePreference as ServerDevicePreference, ServerConfig,
};
use hifi_server::pipeline::RealResampleEngine;
use hifi_server::resampler::{OnnxRunner, ResampleError};
use hifi_server::run_server_with_factory;
use ndarray::{Array4, ArrayView2};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

struct RealOnnxRunner {
    vocoder: VocoderSession,
    hnsep: Option<HnsepSession>,
}

impl RealOnnxRunner {
    fn load_and_warmup(
        cfg: &ServerConfig,
        server_root: &Path,
        http_workers: usize,
    ) -> Result<(Self, bool), String> {
        let session_cfg = SessionConfig {
            device: map_device_preference(cfg.device_preference),
            workers_hint: http_workers,
            ..SessionConfig::default()
        };

        let t0 = Instant::now();
        let vocoder = load_session_with_fallbacks(
            "vocoder",
            server_root,
            &cfg.vocoder_path,
            &[DEFAULT_VOCODER_ONNX_PATH],
            |p| VocoderSession::load_with_config(p, &session_cfg),
        )?;
        if let Some(reason) = vocoder.fallback_reason() {
            tracing::warn!("Vocoder EP fallback: {reason}");
        }
        vocoder
            .warmup()
            .map_err(|e| format!("Vocoder warmup failed: {e}"))?;
        tracing::info!(
            "Loaded vocoder '{}' on {} (gpu={}, {:.1}ms)",
            vocoder.model_path().display(),
            vocoder.execution_provider(),
            vocoder.is_gpu_ep(),
            t0.elapsed().as_secs_f64() * 1000.0
        );

        let hnsep = if cfg.hnsep_model_path.trim().is_empty() {
            None
        } else {
            let t_hnsep = Instant::now();
            let sep = load_session_with_fallbacks(
                "HN-SEP",
                server_root,
                &cfg.hnsep_model_path,
                DEFAULT_HNSEP_ONNX_PATHS,
                |p| HnsepSession::load_with_config(p, &session_cfg),
            )?;
            if let Some(reason) = sep.fallback_reason() {
                tracing::warn!("HN-SEP EP fallback: {reason}");
            }
            sep.warmup()
                .map_err(|e| format!("HN-SEP warmup failed: {e}"))?;
            tracing::info!(
                "Loaded HN-SEP '{}' on {} (gpu={}, {:.1}ms)",
                sep.model_path().display(),
                sep.execution_provider(),
                sep.is_gpu_ep(),
                t_hnsep.elapsed().as_secs_f64() * 1000.0
            );
            Some(sep)
        };

        let directml_active = matches!(vocoder.execution_provider(), ExecutionProvider::DirectMl)
            || hnsep
                .as_ref()
                .is_some_and(|h| matches!(h.execution_provider(), ExecutionProvider::DirectMl));

        Ok((Self { vocoder, hnsep }, directml_active))
    }
}

impl OnnxRunner for RealOnnxRunner {
    fn hnsep_uses_gpu(&self) -> bool {
        self.hnsep.as_ref().is_some_and(HnsepSession::is_gpu_ep)
    }

    fn vocoder_uses_gpu(&self) -> bool {
        self.vocoder.is_gpu_ep()
    }

    fn run_hnsep_mask(&self, stft_tensor: &Array4<f32>) -> Result<Array4<f32>, ResampleError> {
        let Some(hnsep) = self.hnsep.as_ref() else {
            return Err(ResampleError::Internal(
                "HN-SEP model is disabled or unavailable".into(),
            ));
        };
        hnsep
            .forward(stft_tensor.view())
            .map_err(|e| ResampleError::Internal(format!("HN-SEP inference failed: {e}")))
    }

    fn run_vocoder(
        &self,
        mel_n_mels_by_t: ArrayView2<f32>,
        f0: &[f32],
    ) -> Result<Vec<f32>, ResampleError> {
        self.vocoder
            .synthesize_mels_by_frames(mel_n_mels_by_t, f0)
            .map_err(|e| ResampleError::Internal(format!("Vocoder inference failed: {e}")))
    }
}

fn map_device_preference(pref: ServerDevicePreference) -> OnnxDevicePreference {
    match pref {
        ServerDevicePreference::Auto => OnnxDevicePreference::Auto,
        ServerDevicePreference::Cpu => OnnxDevicePreference::Cpu,
        ServerDevicePreference::Cuda => OnnxDevicePreference::Cuda,
        ServerDevicePreference::Rocm => OnnxDevicePreference::Rocm,
        ServerDevicePreference::DirectMl => OnnxDevicePreference::DirectMl,
        ServerDevicePreference::CoreMl => OnnxDevicePreference::CoreMl,
    }
}

fn load_session_with_fallbacks<T>(
    kind: &str,
    server_root: &Path,
    configured: &str,
    default_rels: &[&str],
    mut load_fn: impl FnMut(&Path) -> Result<T, OnnxError>,
) -> Result<T, String> {
    let primary = resolve_asset_path(server_root, configured);
    match load_fn(&primary) {
        Ok(session) => Ok(session),
        Err(primary_err) => {
            let mut last_err = format!(
                "Failed to load {kind} from '{}': {primary_err}",
                primary.display()
            );
            for rel in default_rels {
                let candidate = server_root.join(rel);
                if candidate.is_file() && candidate != primary {
                    tracing::warn!(
                        "Configured {kind} '{}' failed ({primary_err}); trying '{}'",
                        primary.display(),
                        candidate.display()
                    );
                    match load_fn(&candidate) {
                        Ok(session) => return Ok(session),
                        Err(e) => {
                            last_err = format!(
                                "Failed to load {kind} from '{}' or '{}': {e}",
                                primary.display(),
                                candidate.display()
                            );
                        }
                    }
                }
            }
            Err(last_err)
        }
    }
}

fn main() {
    let code = run_server_with_factory(Box::new(|cfg, root| {
        let workers = cfg.max_workers.resolve_http_workers();
        let (runner, directml_active) = RealOnnxRunner::load_and_warmup(cfg, root, workers)?;
        tracing::info!(
            "HTTP inference concurrency configured: workers={workers}, directml_active={directml_active}"
        );
        let engine = Arc::new(RealResampleEngine::new(cfg.clone(), Arc::new(runner))?);
        Ok((engine, workers))
    }));
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_preference_maps_all_variants() {
        assert_eq!(
            map_device_preference(ServerDevicePreference::Auto),
            OnnxDevicePreference::Auto
        );
        assert_eq!(
            map_device_preference(ServerDevicePreference::Cpu),
            OnnxDevicePreference::Cpu
        );
        assert_eq!(
            map_device_preference(ServerDevicePreference::Cuda),
            OnnxDevicePreference::Cuda
        );
        assert_eq!(
            map_device_preference(ServerDevicePreference::Rocm),
            OnnxDevicePreference::Rocm
        );
        assert_eq!(
            map_device_preference(ServerDevicePreference::DirectMl),
            OnnxDevicePreference::DirectMl
        );
        assert_eq!(
            map_device_preference(ServerDevicePreference::CoreMl),
            OnnxDevicePreference::CoreMl
        );
    }
}
