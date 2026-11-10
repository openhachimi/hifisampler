import dataclasses
import logging
import os
from pathlib import Path
from typing import Any
import torch
from util.load_config_from_yaml import load_config_from_yaml


def _resolve_device(device_pref: Any) -> torch.device:
    pref = str(device_pref or "auto").strip().lower()
    if pref in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref in ("dml", "directml"):
        # DirectML is handled by ONNX Runtime (DmlExecutionProvider); PyTorch tensors stay on CPU
        return torch.device("cpu")
    if pref in ("rocm", "hip", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        logging.warning(f"Requested device '{pref}' is not available in PyTorch; falling back to CPU.")
        return torch.device("cpu")
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logging.warning("Requested device 'mps' is not available; falling back to CPU.")
        return torch.device("cpu")
    try:
        return torch.device(pref)
    except Exception:
        logging.warning(f"Invalid device '{pref}' in config; falling back to CPU.")
        return torch.device("cpu")


@dataclasses.dataclass
@load_config_from_yaml(script_path=Path(__file__))
class Config:
    sample_rate: int = 44100  # UTAU only really likes 44.1khz
    win_size: int = 2048     # 必须和vocoder训练时一致
    hop_size: int = 512      # 必须和vocoder训练时一致
    origin_hop_size: int = 128  # 插值前的hopsize,可以适当调小改善长音的电音
    n_mels: int = 128        # 必须和vocoder训练时一致
    n_fft: int = 2048        # 必须和vocoder训练时一致
    mel_fmin: float = 40     # 必须和vocoder训练时一致
    mel_fmax: float = 16000  # 必须和vocoder训练时一致
    fill: int = 6
    vocoder_path: str = "./pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx"
    model_type: str = 'onnx'  # or 'ckpt'
    hnsep_model_path: str = "./hnsep/vr/model_fp16.onnx"
    wave_norm: bool = True
    trim_silence: bool = True  # 是否在响度标准化前截取无声部分
    silence_threshold: float = -52.0
    loop_mode: bool = True
    peak_limit: float = 1.0
    # max_workers can be an int or 'auto' (resolved to physical cores at runtime)
    max_workers: Any = 2
    device: Any = "auto"
    fp16: bool = False
    idle_timeout: int = 0
    device_preference: str = dataclasses.field(default="auto", init=False)

    def __post_init__(self):
        for field_name in ("sample_rate", "win_size", "hop_size", "origin_hop_size", "n_mels", "n_fft"):
            if int(getattr(self, field_name, 0)) <= 0:
                raise ValueError(f"Config '{field_name}' must be > 0, got {getattr(self, field_name)!r}")
        self.device_preference = str(self.device or "auto").strip().lower()
        self.device = _resolve_device(self.device_preference)
        is_rocm = bool(getattr(torch.version, "hip", None))
        if self.device.type == "cuda":
            if is_rocm:
                # Avoid slow MIOpen exhaustive kernel search on dynamic note lengths
                os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
                torch.backends.cudnn.benchmark = False
                try:
                    dev_name = torch.cuda.get_device_name(0)
                except Exception:
                    dev_name = "AMD ROCm GPU"
                logging.info(
                    f"ROCm backend active: {dev_name} (HIP {torch.version.hip}, MIOPEN_FIND_MODE={os.environ.get('MIOPEN_FIND_MODE')})"
                )
            else:
                torch.backends.cudnn.benchmark = False
                try:
                    dev_name = torch.cuda.get_device_name(0)
                except Exception:
                    dev_name = "NVIDIA CUDA GPU"
                logging.info(f"CUDA backend active: {dev_name}")
        elif self.device.type == "cpu" and self.fp16:
            logging.warning("fp16=true is not recommended on CPU; disabling fp16.")
            self.fp16 = False


CONFIG = Config()
