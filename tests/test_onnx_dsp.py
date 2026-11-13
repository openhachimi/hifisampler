from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from util.audio import dynamic_range_compression_torch
from util.hnsep import HnsepModel
from util.onnx_dsp import (
    MAX_WINDOW_CACHE_SIZE,
    NumpyHnsepModel,
    NumpyMelSpectrogram,
    dynamic_range_compression_np,
)
from util.wav2mel import PitchAdjustableMelSpectrogram


class _MockOnnxSession:
    """Deterministic mock ONNX Runtime session producing a complex mask from input spectrogram."""

    def __init__(self, input_type: str = "tensor(float32)") -> None:
        self._input_type = input_type
        self.calls: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", type=self._input_type)]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, input_feed):
        x = input_feed["input"]
        self.calls.append(x)
        x_f32 = x.astype(np.float32, copy=False)
        # Produce non-trivial real and imaginary mask components: [B, 2, F, T_f]
        real_in = x_f32[:, :1, :, :]
        imag_in = x_f32[:, 1:, :, :]
        mag = np.sqrt(real_in * real_in + imag_in * imag_in + 1e-6)
        mask_real = 0.75 * (mag / (mag + 0.1))
        mask_imag = 0.15 * np.tanh(real_in - imag_in)
        out = np.concatenate([mask_real, mask_imag], axis=1).astype(np.float32, copy=False)
        return [out]


def _make_test_audio(length: int = 44100, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32) / 44100.0
    harmonic = (
        0.35 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 660.0 * t)
        + 0.10 * np.cos(2.0 * np.pi * 1320.0 * t)
    )
    noise = rng.normal(0.0, 0.03, size=length).astype(np.float32)
    return (harmonic + noise).astype(np.float32)


def test_dynamic_range_compression_equivalence():
    rng = np.random.default_rng(7)
    arr = rng.uniform(-0.5, 5.0, size=(2, 128, 87)).astype(np.float32)
    arr[0, 0, 0] = 0.0
    arr[0, 1, 0] = -1.0

    np_out = dynamic_range_compression_np(arr, C=1.5, clip_val=1e-7)
    torch_out = dynamic_range_compression_torch(torch.from_numpy(arr), C=1.5, clip_val=1e-7).numpy()

    assert np_out.dtype == np.float32
    assert np.max(np.abs(np_out - torch_out)) < 1e-6


@pytest.mark.parametrize("key_shift", [0.0, 1.2, -0.8])
@pytest.mark.parametrize("speed", [1.0, 0.9])
def test_numpy_mel_spectrogram_equivalence(key_shift: float, speed: float):
    torch_mel = PitchAdjustableMelSpectrogram()
    numpy_mel = NumpyMelSpectrogram()

    audio_1d = _make_test_audio(length=44100, seed=42)
    audio_2d = np.stack([audio_1d, _make_test_audio(length=44100, seed=99)], axis=0)

    expected_2d = torch_mel(torch.from_numpy(audio_2d), key_shift=key_shift, speed=speed).numpy()
    actual_2d = numpy_mel(audio_2d, key_shift=key_shift, speed=speed)
    actual_1d = numpy_mel(audio_1d, key_shift=key_shift, speed=speed)

    assert actual_2d.dtype == np.float32
    assert actual_1d.dtype == np.float32
    assert actual_2d.shape == expected_2d.shape
    assert actual_1d.shape == expected_2d[0].shape

    max_diff_2d = float(np.max(np.abs(actual_2d - expected_2d)))
    max_diff_1d = float(np.max(np.abs(actual_1d - expected_2d[0])))
    assert max_diff_2d < 1e-4, f"2D max_abs_diff={max_diff_2d} >= 1e-4"
    assert max_diff_1d < 1e-4, f"1D max_abs_diff={max_diff_1d} >= 1e-4"


def test_numpy_mel_spectrogram_short_audio_and_center():
    for center in (False, True):
        torch_mel = PitchAdjustableMelSpectrogram(center=center)
        numpy_mel = NumpyMelSpectrogram(center=center)
        short_audio = _make_test_audio(length=120, seed=11)[np.newaxis, :]

        expected = torch_mel(torch.from_numpy(short_audio), key_shift=-0.8, speed=1.0).numpy()
        actual = numpy_mel(short_audio, key_shift=-0.8, speed=1.0)
        assert actual.shape == expected.shape
        assert float(np.max(np.abs(actual - expected))) < 1e-4


def test_numpy_mel_lru_cache_and_thread_safety():
    mel = NumpyMelSpectrogram()
    for i in range(MAX_WINDOW_CACHE_SIZE + 10):
        _ = mel(_make_test_audio(length=4096, seed=i), key_shift=-5.0 + 0.25 * i)
    assert len(mel.hann_window) == MAX_WINDOW_CACHE_SIZE

    audio = _make_test_audio(length=22050, seed=55)
    ref = mel(audio, key_shift=1.2, speed=1.0)

    def _worker(shift: float) -> np.ndarray:
        return mel(audio, key_shift=shift, speed=1.0)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_worker, [1.2] * 12))

    for res in results:
        assert np.allclose(res, ref, atol=1e-6)


@pytest.mark.parametrize("input_type", ["tensor(float32)", "tensor(float16)"])
@pytest.mark.parametrize("length", [100, 15872, 44100, 44101])
def test_numpy_hnsep_model_equivalence(input_type: str, length: int):
    config_args = {"n_fft": 2048, "hop_length": 512}
    mock_session = _MockOnnxSession(input_type=input_type)

    torch_model = HnsepModel(mock_session, config_args)
    numpy_model = NumpyHnsepModel(mock_session, config_args)

    wave_1d = _make_test_audio(length=length, seed=77)
    wave_3d = torch.from_numpy(wave_1d).reshape(1, 1, length)

    expected = torch_model.predict_fromaudio(wave_3d).numpy()[0, 0]
    actual = numpy_model.predict_from_audio(wave_1d)

    assert actual.shape == (length,)
    assert actual.dtype == np.float32
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff < 1e-4, f"max_abs_diff={max_diff} >= 1e-4 for {input_type}, T={length}"

    # Verify the ONNX session received the exact expected input dtype
    expected_np_dtype = np.float16 if input_type == "tensor(float16)" else np.float32
    assert mock_session.calls[-1].dtype == expected_np_dtype


def test_zero_torch_tensor_allocations(monkeypatch):
    """Ensure NumpyMelSpectrogram and NumpyHnsepModel allocate zero torch.Tensors."""
    import util.onnx_dsp as onnx_dsp

    assert not hasattr(onnx_dsp, "torch")

    def _forbid(*args, **kwargs):
        raise AssertionError("torch tensor allocation is forbidden in util.onnx_dsp")

    monkeypatch.setattr(torch, "tensor", _forbid)
    monkeypatch.setattr(torch, "from_numpy", _forbid)
    monkeypatch.setattr(torch, "zeros", _forbid)
    monkeypatch.setattr(torch, "hann_window", _forbid)
    monkeypatch.setattr(torch, "stft", _forbid)
    monkeypatch.setattr(torch, "istft", _forbid)

    mel = NumpyMelSpectrogram()
    hnsep = NumpyHnsepModel(_MockOnnxSession(), {"n_fft": 2048, "hop_length": 512})
    audio = _make_test_audio(length=16000, seed=5)

    sep = hnsep.predict_from_audio(audio)
    m = mel(sep, key_shift=1.2)
    comp = dynamic_range_compression_np(m)
    assert sep.shape == audio.shape
    assert comp.ndim == 2
