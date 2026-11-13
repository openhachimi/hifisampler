"""Pure NumPy/SciPy DSP pipeline without PyTorch tensor allocation.

Provides thread-safe, CPU-only equivalents of:
- ``dynamic_range_compression_np`` (``dynamic_range_compression_torch``)
- ``NumpyMelSpectrogram`` (``PitchAdjustableMelSpectrogram``)
- ``NumpyHnsepModel`` (``HnsepModel``)
"""

from collections import OrderedDict
import threading
import numpy as np
import scipy.fft
import scipy.signal.windows

from util.utils import slaney_mel_filterbank

MAX_WINDOW_CACHE_SIZE = 32


def dynamic_range_compression_np(
    x: np.ndarray,
    C: float = 1.0,
    clip_val: float = 1e-9,
) -> np.ndarray:
    """Apply natural-log dynamic range compression in pure NumPy.

    Args:
        x: Input magnitude/mel array.
        C: Multiplicative scale factor applied after clipping.
        clip_val: Minimum floor value before taking logarithm.

    Returns:
        Compressed ``float32`` NumPy array.
    """
    return np.log(np.maximum(x, clip_val) * C).astype(np.float32, copy=False)


def _reflect_pad_1d_batch(y: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
    """Apply 1D reflect padding (or cropping if negative) along the last axis."""
    if pad_left < 0:
        y = y[:, -pad_left:]
        pad_left = 0
    if pad_right < 0:
        y = y[:, : y.shape[-1] + pad_right]
        pad_right = 0

    if pad_left == 0 and pad_right == 0:
        return y

    max_pad = max(pad_left, pad_right)
    if y.shape[-1] <= max_pad:
        y = np.pad(
            y,
            ((0, 0), (0, max_pad - y.shape[-1] + 1)),
            mode="constant",
            constant_values=0.0,
        )
    return np.pad(y, ((0, 0), (pad_left, pad_right)), mode="reflect")


def _stft_np(
    x: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    center: bool,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Compute onesided complex64 STFT matching ``torch.stft(..., normalized=False, onesided=True)``.

    Args:
        x: 2D array of shape ``[B, L]`` in ``float32``.
        n_fft: FFT size.
        hop_length: Hop size between frames.
        window: 1D ``float32`` window of length ``win_length <= n_fft``.
        center: Whether to pad by ``n_fft // 2`` on both ends.
        pad_mode: ``'reflect'`` or ``'constant'`` (used when ``center=True``).

    Returns:
        Complex64 spectrogram of shape ``[B, n_fft // 2 + 1, n_frames]``.
    """
    win_length = window.shape[0]
    if win_length > n_fft:
        raise ValueError(f"win_length ({win_length}) must be <= n_fft ({n_fft})")

    if win_length < n_fft:
        pad_l = (n_fft - win_length) // 2
        pad_r = n_fft - win_length - pad_l
        window_padded = np.pad(window, (pad_l, pad_r), mode="constant")
    else:
        window_padded = window

    if center:
        pad_c = n_fft // 2
        if pad_mode == "reflect":
            x = _reflect_pad_1d_batch(x, pad_c, pad_c)
        elif pad_mode == "constant":
            x = np.pad(x, ((0, 0), (pad_c, pad_c)), mode="constant", constant_values=0.0)
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    if x.shape[-1] < n_fft:
        raise ValueError(f"Padded signal length ({x.shape[-1]}) is smaller than n_fft ({n_fft})")

    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft, axis=-1)[:, ::hop_length, :]
    spec = scipy.fft.rfft(frames * window_padded, n=n_fft, axis=-1)
    return np.swapaxes(spec, -1, -2).astype(np.complex64, copy=False)


def _istft_ola_np(
    spec: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    center: bool = True,
) -> np.ndarray:
    """Compute overlap-add inverse STFT matching ``torch.istft(..., normalized=False, onesided=True)``.

    Args:
        spec: Complex spectrogram of shape ``[B, n_fft // 2 + 1, n_frames]``.
        n_fft: FFT size.
        hop_length: Hop size between frames.
        window: 1D ``float32`` window of length ``n_fft``.
        center: Whether to trim ``n_fft // 2`` samples from both ends.

    Returns:
        Reconstructed ``float32`` waveform of shape ``[B, L]``.
    """
    batch_size = spec.shape[0]
    sp_frames = np.swapaxes(spec, -1, -2)
    inv_frames = scipy.fft.irfft(sp_frames, n=n_fft, axis=-1) * window
    n_frames = inv_frames.shape[1]
    out_len = (n_frames - 1) * hop_length + n_fft

    out = np.zeros((batch_size, out_len), dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    win_sq = window * window

    if n_fft % hop_length == 0:
        stride_groups = n_fft // hop_length
        for j in range(stride_groups):
            sub = inv_frames[:, j::stride_groups, :]
            m = sub.shape[1]
            if m == 0:
                continue
            start = j * hop_length
            end = start + m * n_fft
            out[:, start:end] += sub.reshape(batch_size, m * n_fft)
            win_sum[start:end] += np.tile(win_sq, m)
    else:
        for i in range(n_frames):
            start = i * hop_length
            out[:, start : start + n_fft] += inv_frames[:, i, :]
            win_sum[start : start + n_fft] += win_sq

    if center:
        pad_c = n_fft // 2
        out = out[:, pad_c : out_len - pad_c]
        win_sum = win_sum[pad_c : out_len - pad_c]

    return (out / np.maximum(win_sum, np.float32(1e-11))).astype(np.float32, copy=False)


class NumpyMelSpectrogram:
    """Pure NumPy/SciPy pitch-adjustable Mel spectrogram matching ``PitchAdjustableMelSpectrogram``.

    Input / Output shape convention:
        - 1D input ``[T]`` -> returns ``[n_mels, n_frames]`` (``float32``).
        - 2D input ``[B, T]`` -> returns ``[B, n_mels, n_frames]`` (``float32``).
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: float = 40.0,
        f_max: float = 16000.0,
        n_mels: int = 128,
        center: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center

        self.mel_basis: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hann_window: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _get_mel_and_window(self, win_size_new: int) -> tuple[np.ndarray, np.ndarray]:
        mel_key = f"{self.sample_rate}_{self.n_fft}_{self.n_mels}_{self.f_min}_{self.f_max}"
        with self._cache_lock:
            mel_basis = self.mel_basis.get(mel_key)
            if mel_basis is None:
                if len(self.mel_basis) >= MAX_WINDOW_CACHE_SIZE:
                    self.mel_basis.popitem(last=False)
                mel_basis = slaney_mel_filterbank(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels,
                    fmin=self.f_min,
                    fmax=self.f_max,
                ).astype(np.float32, copy=False)
                self.mel_basis[mel_key] = mel_basis
            else:
                self.mel_basis.move_to_end(mel_key)

            window = self.hann_window.get(win_size_new)
            if window is None:
                if len(self.hann_window) >= MAX_WINDOW_CACHE_SIZE:
                    self.hann_window.popitem(last=False)
                window = scipy.signal.windows.hann(win_size_new, sym=False).astype(
                    np.float32, copy=False
                )
                self.hann_window[win_size_new] = window
            else:
                self.hann_window.move_to_end(win_size_new)

        return mel_basis, window

    def __call__(
        self,
        y_np: np.ndarray,
        key_shift: float = 0.0,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Compute pitch- and speed-adjusted Mel spectrogram in pure NumPy/SciPy.

        Args:
            y_np: Audio waveform of shape ``[T]`` (1D) or ``[B, T]`` (2D).
            key_shift: Semitone shift for STFT frequency-axis scaling.
            speed: Speed factor scaling hop length.

        Returns:
            Mel spectrogram of shape ``[n_mels, n_frames]`` for 1D input, or
            ``[B, n_mels, n_frames]`` for 2D input.
        """
        y = np.asarray(y_np, dtype=np.float32)
        if y.ndim == 1:
            is_1d = True
            y = y[np.newaxis, :]
        elif y.ndim == 2:
            is_1d = False
        else:
            raise ValueError(f"Expected 1D [T] or 2D [B, T] audio array, got shape {y.shape}")

        factor = 2.0 ** (float(key_shift) / 12.0)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * float(speed)))
        if n_fft_new <= 0 or win_size_new <= 0 or hop_length <= 0:
            raise ValueError(
                f"Invalid STFT sizes: n_fft_new={n_fft_new}, "
                f"win_size_new={win_size_new}, hop_length={hop_length}"
            )

        mel_basis, window = self._get_mel_and_window(win_size_new)

        pad_left = int((win_size_new - hop_length) // 2)
        pad_right = int((win_size_new - hop_length + 1) // 2)
        y_padded = _reflect_pad_1d_batch(y, pad_left, pad_right)

        spec = _stft_np(
            y_padded,
            n_fft=n_fft_new,
            hop_length=hop_length,
            window=window,
            center=self.center,
            pad_mode="reflect",
        )

        mag = np.abs(spec)
        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = mag.shape[1]
            if resize < size:
                mag = np.pad(mag, ((0, 0), (0, size - resize), (0, 0)), mode="constant")
            mag = mag[:, :size, :] * np.float32(self.win_size / win_size_new)

        mel_spec = np.matmul(mel_basis, mag).astype(np.float32, copy=False)
        return mel_spec[0] if is_1d else mel_spec


class NumpyHnsepModel:
    """Pure NumPy/SciPy HN-SEP ONNX inference wrapper with zero PyTorch tensor allocation."""

    def __init__(self, onnx_session, config_args: dict) -> None:
        self.session = onnx_session
        self.config = config_args
        self.n_fft = int(config_args["n_fft"])
        self.hop_length = int(config_args["hop_length"])
        self.max_bin = self.n_fft // 2
        self.output_bin = self.n_fft // 2 + 1
        self.offset = 64
        self._window_cache: np.ndarray | None = None
        self._window_lock = threading.Lock()

        inputs = onnx_session.get_inputs()
        outputs = onnx_session.get_outputs()
        self.input_name = inputs[0].name if inputs else "input"
        self.input_dtype = (
            np.float16
            if (inputs and getattr(inputs[0], "type", "") == "tensor(float16)")
            else np.float32
        )
        self.output_name = outputs[0].name if outputs else "output"

    def _get_window(self) -> np.ndarray:
        with self._window_lock:
            if self._window_cache is None:
                self._window_cache = scipy.signal.windows.hann(
                    self.n_fft, sym=False
                ).astype(np.float32, copy=False)
            return self._window_cache

    def forward(self, spec: np.ndarray) -> np.ndarray:
        """Run ONNX mask inference on complex spectrogram ``[B, C, F, T_f]``."""
        if np.iscomplexobj(spec):
            x_input = np.concatenate([spec.real, spec.imag], axis=1)
        else:
            x_input = spec

        x_numpy = np.ascontiguousarray(x_input, dtype=self.input_dtype)
        output = self.session.run([self.output_name], {self.input_name: x_numpy})[0]
        output = np.asarray(output, dtype=np.float32)

        if output.shape[1] == 2:
            mask = np.empty(
                (output.shape[0], 1, output.shape[2], output.shape[3]),
                dtype=np.complex64,
            )
            mask.real = output[:, :1]
            mask.imag = output[:, 1:]
            return mask

        return output

    def predict_from_audio(self, wave_1d: np.ndarray) -> np.ndarray:
        """Predict separated harmonic waveform without allocating any ``torch.Tensor``.

        Supports:
            - 1D input ``[T]`` -> returns ``[T]``
            - 2D input ``[B, T]`` -> returns ``[B, T]``
            - 3D input ``[B, C, T]`` -> returns ``[B, C, T]``
        """
        x = np.asarray(wave_1d)
        orig_dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
        orig_ndim = x.ndim
        if orig_ndim == 1:
            B, C, T = 1, 1, x.shape[0]
        elif orig_ndim == 2:
            B, C, T = x.shape[0], 1, x.shape[1]
        elif orig_ndim == 3:
            B, C, T = x.shape
        else:
            raise ValueError(
                f"Expected 1D [T], 2D [B, T], or 3D [B, C, T] audio array, got shape {x.shape}"
            )

        x_2d = x.astype(np.float32, copy=False).reshape(B * C, T)
        T1 = T + self.hop_length
        seg_length = 32 * self.hop_length
        T_pad = seg_length * ((T1 - 1) // seg_length + 1) - T1
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length
        x_padded = np.pad(x_2d, ((0, 0), (Tl_pad, T_pad - Tl_pad)), mode="constant")

        window = self._get_window()
        spec_2d = _stft_np(
            x_padded,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            pad_mode="constant",
        )
        n_freq, n_frames = spec_2d.shape[-2], spec_2d.shape[-1]
        spec = spec_2d.reshape(B, C, n_freq, n_frames)

        mask = self.forward(spec)
        spec_pred = (spec * mask).reshape(B * C, n_freq, n_frames)

        x_pred = _istft_ola_np(
            spec_pred,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
        )
        x_pred = x_pred[:, Tl_pad : Tl_pad + T]
        x_pred = x_pred.reshape(B, C, T).astype(orig_dtype, copy=False)

        if orig_ndim == 1:
            return x_pred[0, 0]
        if orig_ndim == 2:
            return x_pred[:, 0, :]
        return x_pred
