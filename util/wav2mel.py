# 从SingingVocoders项目复制
# https://github.com/openvpi/SingingVocoders/blob/main/utils/wav2mel.py

import threading
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from librosa.filters import mel as librosa_mel_fn

MAX_WINDOW_CACHE_SIZE = 32


class PitchAdjustableMelSpectrogram:
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=40.0,
        f_max=16000.0,
        n_mels=128,
        center=False,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center

        self.mel_basis = {}
        self.hann_window = {}
        self._cache_lock = threading.Lock()

    def __call__(self, y, key_shift=0, speed=1.0):
        y = y.float()
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * speed))

        mel_basis_key = f"{self.f_max}_{y.device}"
        hann_window_key = f"{win_size_new}_{y.device}"

        with self._cache_lock:
            mel_basis = self.mel_basis.get(mel_basis_key)
            if mel_basis is None:
                mel = librosa_mel_fn(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels,
                    fmin=self.f_min,
                    fmax=self.f_max,
                )
                mel_basis = torch.from_numpy(mel).float().to(y.device)
                self.mel_basis[mel_basis_key] = mel_basis

            window = self.hann_window.get(hann_window_key)
            if window is None:
                if len(self.hann_window) >= MAX_WINDOW_CACHE_SIZE:
                    oldest_key = next(iter(self.hann_window))
                    del self.hann_window[oldest_key]
                window = torch.hann_window(win_size_new, device=y.device, dtype=torch.float32)
                self.hann_window[hann_window_key] = window

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((win_size_new - hop_length) // 2),
                int((win_size_new - hop_length + 1) // 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft_new,
            hop_length=hop_length,
            win_length=win_size_new,
            window=window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()

        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = F.pad(spec, (0, 0, 0, size - resize))

            spec = spec[:, :size, :] * self.win_size / win_size_new

        spec = torch.matmul(mel_basis, spec)

        return spec

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
