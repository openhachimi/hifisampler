import logging
import os
from pathlib import Path
import torch
import yaml

from config import CONFIG


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_onnx_providers():
    """Get optimal ONNX ExecutionProviders in priority order."""
    import onnxruntime
    available_providers = onnxruntime.get_available_providers()

    # Only force CPUExecutionProvider when the user explicitly set device: "cpu" in config.yaml.
    # When device is "auto" on Windows AMD/Intel GPUs, CONFIG.device is cpu for PyTorch,
    # but ONNX Runtime should still use DmlExecutionProvider!
    if getattr(CONFIG, "device_preference", "auto") == "cpu":
        return ["CPUExecutionProvider"]

    preferred_providers = []
    gpu_candidates = [
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
    ]
    for candidate in gpu_candidates:
        if candidate in available_providers:
            preferred_providers.append(candidate)
    preferred_providers.append("CPUExecutionProvider")
    return preferred_providers


def create_optimized_session_options(preferred_providers=None):
    """Create optimized ONNX session options."""
    import onnxruntime

    logical_cpus = os.cpu_count() or 2
    physical_cpus = max(1, logical_cpus // 2)
    uses_dml = bool(preferred_providers and "DmlExecutionProvider" in preferred_providers)
    raw_workers = getattr(CONFIG, "max_workers", 2)
    if str(raw_workers).strip().lower() == "auto":
        workers = max(1, min(4, physical_cpus))
    else:
        try:
            workers = max(1, int(raw_workers))
        except (ValueError, TypeError):
            workers = 2
    if uses_dml:
        workers = 1

    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = max(1, physical_cpus // workers)
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = not uses_dml
    session_options.enable_mem_reuse = True
    session_options.enable_cpu_mem_arena = True

    logging.debug(
        f"ONNX Session configured with intra_op_threads={session_options.intra_op_num_threads}, "
        f"inter_op_threads={session_options.inter_op_num_threads}, dml={uses_dml}"
    )
    return session_options


def get_cpu_provider_options():
    """Get valid CPUExecutionProvider options."""
    return {
        "arena_extend_strategy": "kSameAsRequested",
    }


def create_onnx_session(actual_path: Path, model_label: str):
    """Create an ONNX Runtime InferenceSession with unified provider configuration."""
    import onnxruntime

    preferred_providers = get_onnx_providers()
    session_options = create_optimized_session_options(preferred_providers)

    provider_options = []
    for provider in preferred_providers:
        if provider == "CPUExecutionProvider":
            provider_options.append(("CPUExecutionProvider", get_cpu_provider_options()))
        elif provider == "DmlExecutionProvider":
            # Disable DirectML MetaCommands so dynamic Mel frame lengths T do not trigger
            # 20-85s per-shape D3D12 driver shader recompilations on every new note duration.
            provider_options.append(("DmlExecutionProvider", {"disable_metacommands": "true"}))
        else:
            provider_options.append((provider, {}))

    session = onnxruntime.InferenceSession(
        str(actual_path),
        providers=provider_options,
        sess_options=session_options,
    )
    used_provider = session.get_providers()[0]

    logging.info(f"Loaded {model_label} (ONNX): {actual_path} using provider {used_provider}")

    if used_provider == "DmlExecutionProvider" and CONFIG.max_workers != 1:
        logging.info("DirectML detected: forcing max_workers=1 to avoid DML multi-thread bug.")
        CONFIG.max_workers = 1

    return session


def resolve_model_path(configured_path, default_paths):
    """Resolve actual model path from configured path and defaults."""
    model_path = Path(configured_path)
    if model_path.exists():
        return model_path

    for default_path in default_paths:
        if default_path.exists():
            logging.warning(f"Configured path '{model_path}' not found, using default: {default_path}")
            return default_path

    raise FileNotFoundError(f"No model found. Checked {model_path} and defaults: {default_paths}")


class HifiGANLoader:
    """HifiGAN model loader"""

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device

    def get_default_paths(self):
        return [
            Path("pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.ckpt"),
            Path("pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/model.onnx"),
        ]

    def load_model(self):
        actual_path = resolve_model_path(self.model_path, self.get_default_paths())

        if actual_path.suffix == ".onnx":
            return self.load_onnx_model(actual_path)
        else:
            return self.load_pytorch_model(actual_path)

    def load_pytorch_model(self, actual_path):
        from util.nsf_hifigan import NsfHifiGAN

        use_fp16 = bool(CONFIG.fp16 and self.device.type != "cpu")
        vocoder = NsfHifiGAN(model_path=actual_path)
        vocoder.to_device(self.device, fp16=use_fp16)
        precision_str = "FP16" if use_fp16 else "FP32"
        logging.info(f"Loaded HifiGAN (PyTorch {precision_str}): {actual_path} on {self.device}")
        return vocoder, "pytorch"

    def load_onnx_model(self, actual_path):
        session = create_onnx_session(actual_path, "HifiGAN")
        return session, "onnx"


class HNSEPLoader:
    """HN-SEP model loader"""

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device

    def get_default_paths(self):
        return [
            Path("hnsep/vr/model_fp16.onnx"),
            Path("hnsep/vr/model.onnx"),
            Path("hnsep/vr/model.pt"),
        ]

    def get_model_config(self, model_path):
        from util.audio import DotDict

        model_dir = os.path.dirname(os.path.abspath(model_path))
        config_file = os.path.join(model_dir, "config.yaml")

        with open(config_file, "r", encoding="utf-8") as config:
            args_dict = yaml.safe_load(config)
        return DotDict(args_dict), args_dict

    def load_model(self):
        actual_path = resolve_model_path(self.model_path, self.get_default_paths())

        if actual_path.suffix == ".onnx":
            return self.load_onnx_model(actual_path)
        else:
            return self.load_pytorch_model(actual_path)

    def load_pytorch_model(self, actual_path):
        from hnsep.nets import CascadedNet

        args, args_dict = self.get_model_config(actual_path)

        model = CascadedNet(
            args_dict["n_fft"],
            args_dict["hop_length"],
            args_dict["n_out"],
            args_dict["n_out_lstm"],
            True,
            is_mono=args_dict["is_mono"],
            fixed_length=True if args_dict.get("fixed_length", None) is None else args_dict["fixed_length"],
        )
        state_dict = _torch_load_compat(actual_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        use_fp16 = bool(CONFIG.fp16 and self.device.type != "cpu")
        if use_fp16:
            model.half()
        model.eval()

        precision_str = "FP16" if use_fp16 else "FP32"
        logging.info(f"Loaded HN-SEP (PyTorch {precision_str}): {actual_path} on {self.device}")
        return model, "pytorch", args

    def load_onnx_model(self, actual_path):
        from util.hnsep import HnsepModel

        args, args_dict = self.get_model_config(actual_path)
        session = create_onnx_session(actual_path, "HN-SEP")
        model = HnsepModel(session, args_dict)
        return model, "onnx", args
