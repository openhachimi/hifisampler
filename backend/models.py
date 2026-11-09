import logging
import threading
import time
import numpy as np
import torch

from config import CONFIG
from util.wav2mel import PitchAdjustableMelSpectrogram
from util.model_loader import HifiGANLoader, HNSEPLoader

vocoder = None
ort_session = None
hnsep_model = None
mel_analyzer = None
vocoder_type = None
hnsep_type = None

# Serialize GPU forward passes across worker threads to prevent MIOpen / CUDA stream / DML contention
gpu_infer_lock = threading.Lock()

logging.basicConfig(format='%(message)s', level=logging.INFO)


def warmup_models():
    """Run a short dummy forward pass to pre-compile MIOpen / cuDNN / ORT kernels at server startup."""
    t0 = time.perf_counter()
    try:
        with gpu_infer_lock, torch.inference_mode():
            dummy_wav = torch.zeros((1, 1, CONFIG.sample_rate // 4), dtype=torch.float32, device=CONFIG.device)
            if hnsep_model is not None and (hnsep_type == "pytorch" or CONFIG.device.type != "cpu"):
                _ = hnsep_model.predict_fromaudio(dummy_wav)
            if mel_analyzer is not None:
                _ = mel_analyzer(dummy_wav.squeeze(1), 0, 1)
            if vocoder is not None and vocoder_type == "pytorch":
                dummy_mel = torch.zeros((1, CONFIG.n_mels, 20), dtype=torch.float32, device=CONFIG.device)
                dummy_f0 = torch.full((1, 20), 440.0, dtype=torch.float32, device=CONFIG.device)
                _ = vocoder.spec2wav_torch(dummy_mel, f0=dummy_f0)
            elif ort_session is not None and CONFIG.device.type != "cpu":
                dummy_mel_np = np.zeros((1, 20, CONFIG.n_mels), dtype=np.float32)
                dummy_f0_np = np.full((1, 20), 440.0, dtype=np.float32)
                _ = ort_session.run(["waveform"], {"mel": dummy_mel_np, "f0": dummy_f0_np})
            if CONFIG.device.type == "cuda":
                torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logging.info(f"Model warmup completed in {elapsed_ms:.1f} ms.")
    except Exception as e:
        logging.warning(f"Model warmup skipped or encountered non-fatal error: {e}")


def initialize_models():
    global vocoder, ort_session, hnsep_model, mel_analyzer, vocoder_type, hnsep_type

    logging.info("Initializing models...")

    hifigan_loader = HifiGANLoader(CONFIG.vocoder_path, CONFIG.device)
    hnsep_loader = HNSEPLoader(CONFIG.hnsep_model_path, CONFIG.device)

    vocoder_result = hifigan_loader.load_model()
    model_or_session, vocoder_type = vocoder_result
    if vocoder_type == "onnx":
        ort_session = model_or_session
        CONFIG.model_type = "onnx"
    else:
        vocoder = model_or_session
        CONFIG.model_type = "ckpt"

    hnsep_result = hnsep_loader.load_model()
    hnsep_model, hnsep_type, _ = hnsep_result

    mel_analyzer = PitchAdjustableMelSpectrogram(
        sample_rate=CONFIG.sample_rate,
        n_fft=CONFIG.n_fft,
        win_length=CONFIG.win_size,
        hop_length=CONFIG.origin_hop_size,
        f_min=CONFIG.mel_fmin,
        f_max=CONFIG.mel_fmax,
        n_mels=CONFIG.n_mels,
    )
    logging.info(f"Initialized Mel Analysis with hop_size={CONFIG.origin_hop_size}.")

    warmup_models()
    logging.info("Models initialized successfully.")
