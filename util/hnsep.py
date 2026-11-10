import logging
import os
from pathlib import Path
import numpy as np
import yaml
import torch
from hnsep.nets import CascadedNet
from util.audio import DotDict


class HnsepModel:
    """HN-SEP ONNX inference wrapper that receives pre-created session"""

    def __init__(self, onnx_session, config_args):
        self.session = onnx_session
        self.config = config_args
        self.n_fft = config_args['n_fft']
        self.hop_length = config_args['hop_length']
        self.max_bin = self.n_fft // 2
        self.output_bin = self.n_fft // 2 + 1
        self.offset = 64
        self._window_cache = {}
        inputs = onnx_session.get_inputs()
        outputs = onnx_session.get_outputs()
        self.input_name = inputs[0].name if inputs else 'input'
        self.input_dtype = np.float16 if (inputs and getattr(inputs[0], 'type', '') == 'tensor(float16)') else np.float32
        self.output_name = outputs[0].name if outputs else 'output'

    def _get_window(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        win = self._window_cache.get(key)
        if win is None:
            win = torch.hann_window(self.n_fft, device=device, dtype=torch.float32)
            self._window_cache[key] = win
        return win

    def forward(self, x):
        """
        Forward pass with complex spectrogram input
        x: complex tensor [batch, channels, freq, time]
        """
        original_device = x.device

        if torch.is_complex(x):
            real_part = x.real
            imag_part = x.imag
            x_input = torch.cat([real_part, imag_part], dim=1)
        else:
            x_input = x

        x_numpy = x_input.detach().cpu().numpy().astype(self.input_dtype, copy=False)

        output = self.session.run([self.output_name], {self.input_name: x_numpy})[0]

        output_tensor = torch.from_numpy(output).to(device=original_device, dtype=torch.float32)

        if output_tensor.shape[1] == 2:
            real_part = output_tensor[:, :1]
            imag_part = output_tensor[:, 1:]
            output_tensor = torch.complex(real_part, imag_part)

        return output_tensor

    @torch.inference_mode()
    def predict_fromaudio(self, x):
        """Predict from audio input"""
        orig_dtype = x.dtype
        x = x.float()
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        T1 = T + self.hop_length
        seg_length = 32 * self.hop_length
        T_pad = seg_length * ((T1 - 1) // seg_length + 1) - T1
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length
        x = torch.nn.functional.pad(x, (Tl_pad, T_pad - Tl_pad))

        window = self._get_window(x.device)

        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=window,
            pad_mode='constant'
        )
        spec = spec.reshape(B, C, spec.shape[-2], spec.shape[-1])

        mask = self.forward(spec)
        spec_pred = spec * mask
        spec_pred = spec_pred.reshape(B * C, spec.shape[-2], spec.shape[-1])

        x_pred = torch.istft(spec_pred, self.n_fft, self.hop_length, window=window)
        x_pred = x_pred[:, Tl_pad: Tl_pad + T]
        x_pred = x_pred.reshape(B, C, T)
        return x_pred.to(dtype=orig_dtype)


def load_sep_model(model_path, device=torch.device('cpu')):
    """Load HN-SEP model from checkpoint. Use HNSEPLoader for unified loading."""
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_file = os.path.join(model_dir, 'config.yaml')

    with open(config_file, "r", encoding="utf-8") as config:
        args_dict = yaml.safe_load(config)
    args = DotDict(args_dict)

    model_path_obj = Path(model_path)

    if model_path_obj.suffix == '.onnx':
        raise ValueError("ONNX models should be loaded through HNSEPLoader for unified session management")

    model = CascadedNet(
        args_dict['n_fft'],
        args_dict['hop_length'],
        args_dict['n_out'],
        args_dict['n_out_lstm'],
        True,
        is_mono=args_dict['is_mono'],
        fixed_length=True if args_dict.get('fixed_length', None) is None else args_dict['fixed_length']
    )
    model.to(device)
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Loaded HN-SEP model (PyTorch): {model_path}")
    return model, args
