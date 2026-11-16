"""Generate deterministic Python reference fixtures for hifi_dsp numerical parity tests.

Run from repository root:
    C:\\Project\\hifi\\.venv\\Scripts\\python.exe rust/crates/hifi_dsp/tests/gen_parity_fixtures.py
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.interpolate as interp
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from config import CONFIG  # noqa: E402
from util.audio import (  # noqa: E402
    _solve_lf_unit_period_params,
    couple_aperiodic_to_tension,
    pre_emphasis_base_tension,
)
from util.onnx_dsp import (  # noqa: E402
    NumpyHnsepModel,
    NumpyMelSpectrogram,
    dynamic_range_compression_np,
)
from util.parse_utau import midi_to_hz, pitch_string_to_cents  # noqa: E402
from util.utils import slaney_mel_filterbank  # noqa: E402


class _MockOnnxSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input", type="tensor(float32)")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, _output_names, input_feed):
        x = input_feed["input"].astype(np.float32, copy=False)
        real_in = x[:, :1, :, :]
        imag_in = x[:, 1:, :, :]
        mag = np.sqrt(real_in * real_in + imag_in * imag_in + 1e-6)
        mask_real = 0.75 * (mag / (mag + 0.1))
        mask_imag = 0.15 * np.tanh(real_in - imag_in)
        out = np.concatenate([mask_real, mask_imag], axis=1).astype(np.float32, copy=False)
        self.last_input = x
        self.last_output = out
        return [out]


def write_record(fh, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.float32:
        kind = 0
    elif arr.dtype == np.float64:
        kind = 1
    elif arr.dtype == np.int32:
        kind = 2
    else:
        raise TypeError(f"Unsupported dtype for {name}: {arr.dtype}")

    name_bytes = name.encode("utf-8")
    fh.write(struct.pack("<IBI", len(name_bytes), kind, arr.ndim))
    fh.write(name_bytes)
    for d in arr.shape:
        fh.write(struct.pack("<I", d))
    fh.write(arr.tobytes(order="C"))


def resample_time_reference(
    mel_origin: np.ndarray,
    pitchbend: np.ndarray,
    *,
    base_pitch_midi: float,
    velocity: float,
    offset_ms: float,
    length_ms: float,
    consonant_ms: float,
    cutoff_ms: float,
    t_flag: float,
    tempo: float,
    loop_mode: bool,
    fill: int = 6,
    sr: int = 44100,
    hop_size: int = 512,
    origin_hop_size: int = 128,
):
    thop_origin = origin_hop_size / sr
    thop = hop_size / sr
    t_area_origin = np.arange(mel_origin.shape[1]) * thop_origin + thop_origin / 2
    total_time = t_area_origin[-1] + thop_origin / 2

    vel = np.exp2(1 - velocity / 100)
    offset = offset_ms / 1000
    cutoff = cutoff_ms / 1000
    start = offset
    end = (start - cutoff) if cutoff_ms < 0 else (total_time - cutoff)
    con = start + consonant_ms / 1000
    length_req = max(1, int(round(float(length_ms)))) / 1000
    stretch_length = end - con

    if loop_mode:
        con_frame = max(0, min(mel_origin.shape[1] - 1, int((con + thop_origin / 2) // thop_origin)))
        end_frame = max(con_frame + 1, min(mel_origin.shape[1], int((end + thop_origin / 2) // thop_origin)))
        mel_loop = mel_origin[:, con_frame:end_frame]
        if mel_loop.shape[1] > 0:
            pad_loop_size = int(max(1, length_req // thop_origin + 1))
            padded_mel = np.pad(mel_loop, pad_width=((0, 0), (0, pad_loop_size)), mode="reflect")
            mel_origin = np.concatenate((mel_origin[:, :con_frame], padded_mel), axis=1)
            stretch_length = pad_loop_size * thop_origin
            t_area_origin = np.arange(mel_origin.shape[1]) * thop_origin + thop_origin / 2
            total_time = t_area_origin[-1] + thop_origin / 2

    if stretch_length <= 1e-6:
        scaling_ratio = 1.0
    elif stretch_length < length_req:
        scaling_ratio = length_req / stretch_length
    else:
        scaling_ratio = 1.0

    def stretch(t_arr, con_t, sr_ratio):
        return np.where(t_arr < vel * con_t, t_arr / vel, con_t + (t_arr - vel * con_t) / sr_ratio)

    stretched_n_frames = max(1, int((con * vel + max(0.0, total_time - con) * scaling_ratio) // thop + 1))
    stretched_t_mel = np.arange(stretched_n_frames) * thop + thop / 2

    start_left_mel_frames = (start * vel + thop / 2) // thop
    cut_left_mel_frames = (start_left_mel_frames - fill) if start_left_mel_frames > fill else 0

    end_right_mel_frames = stretched_n_frames - (length_req + con * vel + thop / 2) // thop
    cut_right_mel_frames = (end_right_mel_frames - fill) if end_right_mel_frames > fill else 0

    slice_start = max(0, int(cut_left_mel_frames))
    slice_end = max(slice_start + 1, int(stretched_n_frames - cut_right_mel_frames))
    sliced_t_mel = stretched_t_mel[slice_start:slice_end]
    if len(sliced_t_mel) > 0:
        stretched_t_mel = sliced_t_mel
    else:
        cut_left_mel_frames = 0

    stretch_t_mel = np.clip(stretch(stretched_t_mel, con, scaling_ratio), t_area_origin[0], t_area_origin[-1])
    new_start = start * vel - cut_left_mel_frames * thop
    new_end = (length_req + con * vel) - cut_left_mel_frames * thop
    new_con = new_end - length_req

    n_origin = mel_origin.shape[1]
    pos = np.clip((stretch_t_mel - t_area_origin[0]) / thop_origin, 0.0, float(n_origin - 1))
    idx0 = np.floor(pos).astype(np.intp)
    idx1 = np.minimum(idx0 + 1, n_origin - 1)
    frac = (pos - idx0).astype(mel_origin.dtype)[np.newaxis, :]
    mel_render = (mel_origin[:, idx0] * (1.0 - frac) + mel_origin[:, idx1] * frac).astype(np.float32)

    t = np.arange(mel_render.shape[1]) * thop
    pitch = pitchbend / 100 + base_pitch_midi
    if t_flag:
        pitch = pitch + np.clip(t_flag, -120, 120) / 10
    t_pitch = 60 * np.arange(len(pitch)) / (tempo * 96) + new_start
    if len(pitch) < 2 or np.all(pitch == pitch[0]):
        const_pitch = float(pitch[0]) if len(pitch) else float(base_pitch_midi)
        pitch_render = np.full(len(t), const_pitch, dtype=np.float64)
        f0_render = np.full(len(t), float(midi_to_hz(const_pitch)), dtype=np.float32)
    else:
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(np.clip(t, new_start, t_pitch[-1])).astype(np.float64)
        f0_render = midi_to_hz(pitch_render).astype(np.float32)

    cut_l = max(0, int(new_start * sr))
    cut_r = max(cut_l + 1, int(new_end * sr))
    return mel_render, f0_render, pitch_render, t, np.array([new_start, new_end, new_con, float(cut_l), float(cut_r)], dtype=np.float64)


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parity_fixtures.bin"

    rng = np.random.default_rng(42)
    n_main = 16000
    t_main = np.arange(n_main, dtype=np.float32) / 44100.0
    audio_main = (
        0.35 * np.sin(2.0 * np.pi * 220.0 * t_main)
        + 0.20 * np.sin(2.0 * np.pi * 660.0 * t_main)
        + 0.10 * np.cos(2.0 * np.pi * 1320.0 * t_main)
        + rng.normal(0.0, 0.03, size=n_main).astype(np.float32)
    ).astype(np.float32)

    slaney_fb = slaney_mel_filterbank(44100, 2048, 128, 40.0, 16000.0).astype(np.float32)
    mel_calc = NumpyMelSpectrogram()
    mel_center_calc = NumpyMelSpectrogram(center=True)

    mel_default = mel_calc(audio_main, key_shift=0.0, speed=1.0)
    mel_default_drc = dynamic_range_compression_np(mel_default)
    mel_shift_up = mel_calc(audio_main, key_shift=1.2, speed=0.9)
    mel_shift_odd = mel_calc(audio_main, key_shift=-0.06, speed=1.0)
    mel_short_center = mel_center_calc(audio_main[:120], key_shift=-0.8, speed=1.0)

    # HN-SEP fixtures
    mock_sess = _MockOnnxSession()
    hnsep = NumpyHnsepModel(mock_sess, {"n_fft": 2048, "hop_length": 512})
    hnsep_out_100 = hnsep.predict_from_audio(audio_main[:100])
    hnsep_in_100 = mock_sess.last_input.copy()
    hnsep_mask_100 = mock_sess.last_output.copy()

    hnsep_out_main = hnsep.predict_from_audio(audio_main)
    hnsep_in_main = mock_sess.last_input.copy()
    hnsep_mask_main = mock_sess.last_output.copy()

    # Akima fixtures
    akima_step_x = np.linspace(1.0, 7.0, 7, dtype=np.float64)
    akima_step_y = np.array([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    akima_step_q = np.linspace(1.0, 7.0, 61, dtype=np.float64)
    akima_step_out = interp.Akima1DInterpolator(akima_step_x, akima_step_y)(akima_step_q)

    akima_nonunif_x = np.array([0.0, 0.05, 0.12, 0.30, 0.31, 0.60, 1.0], dtype=np.float64)
    akima_nonunif_y = np.array([60.0, 62.5, 59.0, 64.0, 64.2, 58.0, 60.0], dtype=np.float64)
    akima_nonunif_q = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    akima_nonunif_out = interp.Akima1DInterpolator(akima_nonunif_x, akima_nonunif_y)(akima_nonunif_q)

    akima_2_x = np.array([0.1, 0.9], dtype=np.float64)
    akima_2_y = np.array([60.0, 65.0], dtype=np.float64)
    akima_2_q = np.linspace(0.1, 0.9, 11, dtype=np.float64)
    akima_2_out = interp.Akima1DInterpolator(akima_2_x, akima_2_y)(akima_2_q)

    akima_3_x = np.array([0.0, 0.2, 0.8], dtype=np.float64)
    akima_3_y = np.array([60.0, 63.0, 61.0], dtype=np.float64)
    akima_3_q = np.linspace(0.0, 0.8, 17, dtype=np.float64)
    akima_3_out = interp.Akima1DInterpolator(akima_3_x, akima_3_y)(akima_3_q)

    # Time-axis & pitch fixtures
    mel_origin_synth = NumpyMelSpectrogram(hop_length=128)(audio_main, key_shift=0.0, speed=1.0)
    mel_origin_synth = dynamic_range_compression_np(mel_origin_synth)
    pb_cents = pitch_string_to_cents("AA#5#AB#5#AC").astype(np.int32)
    mel_r_loop, f0_r_loop, pitch_r_loop, t_mel_loop, scalars_loop = resample_time_reference(
        mel_origin_synth,
        pb_cents,
        base_pitch_midi=69.0,
        velocity=85.0,
        offset_ms=40.0,
        length_ms=350.0,
        consonant_ms=30.0,
        cutoff_ms=-60.0,
        t_flag=5.0,
        tempo=135.0,
        loop_mode=True,
    )
    mel_r_noloop, f0_r_noloop, pitch_r_noloop, t_mel_noloop, scalars_noloop = resample_time_reference(
        mel_origin_synth,
        pb_cents,
        base_pitch_midi=69.0,
        velocity=120.0,
        offset_ms=20.0,
        length_ms=420.0,
        consonant_ms=25.0,
        cutoff_ms=30.0,
        t_flag=-3.0,
        tempo=110.0,
        loop_mode=False,
    )

    # Tension & LF fixtures
    rds = np.array([0.07, 0.21, 0.45, 1.0, 1.8, 3.0], dtype=np.float64)
    lf_params = np.array([_solve_lf_unit_period_params(float(r)) for r in rds], dtype=np.float64)

    sr = CONFIG.sample_rate
    n_harm = sr // 2
    t_h = np.arange(n_harm, dtype=np.float32) / sr
    harm_sig = np.zeros(n_harm, dtype=np.float32)
    for k in range(1, 20):
        harm_sig += (1.0 / (k ** 1.3)) * np.sin(2.0 * np.pi * 220.0 * k * t_h)
    harm_sig = (0.2 * harm_sig).astype(np.float32)
    harm_t = torch.from_numpy(harm_sig).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        tension_pos = pre_emphasis_base_tension(harm_t, -1.0, ho=0.0, hr=0.0).squeeze().numpy()
        tension_neg = pre_emphasis_base_tension(harm_t, 1.5, ho=0.0, hr=0.0).squeeze().numpy()
        tension_hohr = pre_emphasis_base_tension(harm_t, -1.2, ho=45.0, hr=-35.0).squeeze().numpy()
        tension_short = pre_emphasis_base_tension(harm_t[:, :, :128], 0.5, ho=0.0, hr=0.0).squeeze().numpy()

        consonant_end = int(0.12 * sr)
        ap_noise_1d = (0.05 * rng.standard_normal(n_harm)).astype(np.float32)
        ap_t = torch.from_numpy(ap_noise_1d).unsqueeze(0).unsqueeze(0)
        voiced_part = torch.zeros_like(harm_t)
        voiced_part[:, :, consonant_end + 2048 :] = harm_t[:, :, consonant_end + 2048 :]
        voiced_soft = pre_emphasis_base_tension(voiced_part, 2.0, ho=-40.0, hr=30.0)
        ap_coupled = couple_aperiodic_to_tension(
            ap_t, voiced_part, voiced_soft, 2.0, ho=-40.0, hr=30.0
        ).squeeze().numpy()
        voiced_part_1d = voiced_part.squeeze().numpy()
        voiced_soft_1d = voiced_soft.squeeze().numpy()

    records = [
        ("audio_main", audio_main),
        ("slaney_fb", slaney_fb),
        ("mel_default", mel_default),
        ("mel_default_drc", mel_default_drc),
        ("mel_shift_up", mel_shift_up),
        ("mel_shift_odd", mel_shift_odd),
        ("mel_short_center", mel_short_center),
        ("hnsep_in_100", hnsep_in_100),
        ("hnsep_mask_100", hnsep_mask_100),
        ("hnsep_out_100", hnsep_out_100),
        ("hnsep_in_main", hnsep_in_main),
        ("hnsep_mask_main", hnsep_mask_main),
        ("hnsep_out_main", hnsep_out_main),
        ("akima_step_x", akima_step_x),
        ("akima_step_y", akima_step_y),
        ("akima_step_q", akima_step_q),
        ("akima_step_out", akima_step_out),
        ("akima_nonunif_x", akima_nonunif_x),
        ("akima_nonunif_y", akima_nonunif_y),
        ("akima_nonunif_q", akima_nonunif_q),
        ("akima_nonunif_out", akima_nonunif_out),
        ("akima_2_x", akima_2_x),
        ("akima_2_y", akima_2_y),
        ("akima_2_q", akima_2_q),
        ("akima_2_out", akima_2_out),
        ("akima_3_x", akima_3_x),
        ("akima_3_y", akima_3_y),
        ("akima_3_q", akima_3_q),
        ("akima_3_out", akima_3_out),
        ("mel_origin_synth", mel_origin_synth),
        ("pb_cents", pb_cents),
        ("mel_r_loop", mel_r_loop),
        ("f0_r_loop", f0_r_loop),
        ("pitch_r_loop", pitch_r_loop),
        ("t_mel_loop", t_mel_loop),
        ("scalars_loop", scalars_loop),
        ("mel_r_noloop", mel_r_noloop),
        ("f0_r_noloop", f0_r_noloop),
        ("pitch_r_noloop", pitch_r_noloop),
        ("t_mel_noloop", t_mel_noloop),
        ("scalars_noloop", scalars_noloop),
        ("lf_rds", rds),
        ("lf_params", lf_params),
        ("harm_sig", harm_sig),
        ("tension_pos", tension_pos),
        ("tension_neg", tension_neg),
        ("tension_hohr", tension_hohr),
        ("tension_short", tension_short),
        ("ap_noise_1d", ap_noise_1d),
        ("voiced_part_1d", voiced_part_1d),
        ("voiced_soft_1d", voiced_soft_1d),
        ("ap_coupled", ap_coupled),
    ]

    with open(out_path, "wb") as fh:
        fh.write(b"PARITY01")
        fh.write(struct.pack("<I", len(records)))
        for name, arr in records:
            write_record(fh, name, arr)

    print(f"Wrote {len(records)} parity records to {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
