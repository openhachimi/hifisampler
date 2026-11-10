import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from config import CONFIG
from util.model_loader import (
    HNSEPLoader,
    HifiGANLoader,
    create_optimized_session_options,
    get_cpu_provider_options,
    get_onnx_providers,
    resolve_model_path,
)


class TestProviderPriorityResolution:
    """Test ONNX Runtime ExecutionProvider priority ordering and constraints."""

    def test_explicit_cpu_preference_forces_cpu(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "device_preference", "cpu")
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            providers = get_onnx_providers()
            assert providers == ["CPUExecutionProvider"]

    def test_cuda_preferred_when_available(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "device_preference", "auto")
        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]):
            providers = get_onnx_providers()
            assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_dml_preferred_when_available(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "device_preference", "auto")
        with patch("onnxruntime.get_available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            providers = get_onnx_providers()
            assert providers == ["DmlExecutionProvider", "CPUExecutionProvider"]

    def test_rocm_preferred_when_available(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "device_preference", "auto")
        with patch("onnxruntime.get_available_providers", return_value=["ROCMExecutionProvider", "CPUExecutionProvider"]):
            providers = get_onnx_providers()
            assert providers == ["ROCMExecutionProvider", "CPUExecutionProvider"]

    def test_provider_priority_hierarchy(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "device_preference", "auto")
        # All providers available: CUDA > MIGraphX > ROCm > DirectML > CoreML > CPU
        all_providers = [
            "CoreMLExecutionProvider",
            "DmlExecutionProvider",
            "ROCMExecutionProvider",
            "MIGraphXExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        with patch("onnxruntime.get_available_providers", return_value=all_providers):
            providers = get_onnx_providers()
            assert providers == [
                "CUDAExecutionProvider",
                "MIGraphXExecutionProvider",
                "ROCMExecutionProvider",
                "DmlExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]


class TestSessionOptions:
    """Test optimized ONNX Runtime session options creation."""

    def test_dml_disables_mem_pattern(self):
        import onnxruntime
        opts = create_optimized_session_options(preferred_providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        assert opts.enable_mem_pattern is False
        assert opts.inter_op_num_threads == 1
        assert opts.execution_mode == onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    def test_cpu_or_cuda_enables_mem_pattern(self):
        opts = create_optimized_session_options(preferred_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert opts.enable_mem_pattern is True

    def test_cpu_provider_options(self):
        opts = get_cpu_provider_options()
        assert opts.get("arena_extend_strategy") == "kSameAsRequested"


class TestModelPathResolution:
    """Test model file path resolution and missing fallback handling."""

    def test_configured_path_exists(self, tmp_path):
        configured = tmp_path / "custom_model.onnx"
        configured.write_bytes(b"dummy")
        default = tmp_path / "default_model.onnx"

        resolved = resolve_model_path(configured, [default])
        assert resolved == configured

    def test_fallback_to_default_path(self, tmp_path):
        configured = tmp_path / "nonexistent_model.onnx"
        default = tmp_path / "default_model.onnx"
        default.write_bytes(b"dummy")

        resolved = resolve_model_path(configured, [default])
        assert resolved == default

    def test_all_missing_raises_file_not_found_error(self, tmp_path):
        configured = tmp_path / "nonexistent.onnx"
        default1 = tmp_path / "nonexistent_def1.onnx"
        default2 = tmp_path / "nonexistent_def2.onnx"

        with pytest.raises(FileNotFoundError, match="No model found"):
            resolve_model_path(configured, [default1, default2])


class TestLoaderClasses:
    """Test HifiGANLoader and HNSEPLoader dispatch and configurations."""

    def test_hifigan_loader_default_paths(self):
        loader = HifiGANLoader("some_path.onnx", CONFIG.device)
        defaults = loader.get_default_paths()
        assert len(defaults) == 2
        assert defaults[0].name == "model.ckpt"
        assert defaults[1].name == "model.onnx"

    def test_hnsep_loader_default_paths(self):
        loader = HNSEPLoader("some_path.pt", CONFIG.device)
        defaults = loader.get_default_paths()
        assert len(defaults) == 3
        assert defaults[0].name == "model_fp16.onnx"
        assert defaults[1].name == "model.onnx"
        assert defaults[2].name == "model.pt"

    def test_hnsep_get_model_config(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"dummy")
        config_file = tmp_path / "config.yaml"
        sample_config = {
            "n_fft": 2048,
            "hop_length": 512,
            "n_out": 2,
            "n_out_lstm": 128,
            "is_mono": True,
            "fixed_length": None,
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(sample_config, f)

        loader = HNSEPLoader(model_file, CONFIG.device)
        dot_dict, raw_dict = loader.get_model_config(model_file)
        assert dot_dict.n_fft == 2048
        assert dot_dict.is_mono is True
        assert raw_dict["hop_length"] == 512

    def test_hifigan_loader_dispatches_onnx(self, tmp_path):
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"dummy_onnx")

        loader = HifiGANLoader(model_path, CONFIG.device)
        with patch.object(loader, "load_onnx_model", return_value=("fake_session", "onnx")) as mock_onnx:
            session, model_type = loader.load_model()
            assert model_type == "onnx"
            mock_onnx.assert_called_once_with(model_path)

    def test_hifigan_loader_dispatches_pytorch(self, tmp_path):
        model_path = tmp_path / "model.ckpt"
        model_path.write_bytes(b"dummy_ckpt")

        loader = HifiGANLoader(model_path, CONFIG.device)
        with patch.object(loader, "load_pytorch_model", return_value=("fake_vocoder", "pytorch")) as mock_pt:
            vocoder, model_type = loader.load_model()
            assert model_type == "pytorch"
            mock_pt.assert_called_once_with(model_path)
