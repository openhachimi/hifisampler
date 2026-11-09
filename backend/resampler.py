import logging
import time
import numpy as np
import torch
import scipy.interpolate as interp
from pathlib import Path

from config import CONFIG
from backend import models
from util.growl import growl
from util.audio import dynamic_range_compression_torch, loudness_norm, pre_emphasis_base_tension, read_wav, save_wav
from util.parse_utau import flag_re, midi_to_hz, note_to_midi, pitch_string_to_cents
from util.cache_manager import cache_manager

cache_ext = '.hifi.npz'  # cache file extension


class Resampler:
    def __init__(self, in_file, out_file, pitch, velocity, flags='', offset=0, length=1000, consonant=0, cutoff=0, volume=100, modulation=0, tempo='!100', pitch_string='AA'):
        self.in_file = Path(in_file)
        self.out_file = out_file
        self.pitch = note_to_midi(pitch)
        self.velocity = float(velocity)
        self.flags = {}
        for k, v in flag_re.findall(flags.replace('/', '')):
            if k not in self.flags:
                self.flags[k] = int(v) if v else None
        self.offset = float(offset)
        self.length = int(length)
        self.consonant = float(consonant)
        self.cutoff = float(cutoff)
        self.volume = float(volume)
        self.modulation = float(modulation)
        self.tempo = float(tempo[1:]) if str(tempo).startswith('!') else float(tempo)
        self.pitchbend = pitch_string_to_cents(pitch_string)

        self.render()

    def render(self):
        t0 = time.perf_counter()
        features = self.get_features()
        self.resample(features)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logging.info(f"Render finished in {elapsed_ms:.1f} ms: {self.in_file.name} -> {Path(self.out_file).name}")

    def get_features(self):
        features_path = self.in_file.with_suffix(cache_ext)

        hb_val = self.flags.get('Hb', 100)
        hv_val = self.flags.get('Hv', 100)
        ht_val = self.flags.get('Ht', 0)
        g_val = self.flags.get('g', 0)
        self.flags['Hb'] = int(np.clip(100 if hb_val is None else hb_val, 0, 500))
        self.flags['Hv'] = int(np.clip(100 if hv_val is None else hv_val, 0, 150))
        self.flags['Ht'] = int(np.clip(0 if ht_val is None else ht_val, -100, 100))
        self.flags['g'] = int(np.clip(0 if g_val is None else g_val, -100, 100))

        flag_suffix = '_'.join(f"{k}{v if v is not None else ''}" for k, v in sorted(
            self.flags.items()) if k in ['Hb', 'Hv', 'Ht', 'g'])
        if flag_suffix:
            features_path = features_path.with_name(
                f'{self.in_file.stem}_{flag_suffix}{cache_ext}')
        else:
            features_path = features_path.with_name(
                f'{self.in_file.stem}{cache_ext}')

        force_generate = 'G' in self.flags.keys()
        with cache_manager.get_lock(features_path):
            features = cache_manager.load_features_cache(features_path, force_generate)
            if features is not None:
                return features
            logging.info(f'{features_path.name} not found or forcing generation. Generating features.')
            features = self.generate_features(features_path)
            return cache_manager.save_features_cache(features_path, features, force_generate=force_generate)

    def _needs_hnsep_separation(self, breath, voicing, tension):
        if tension != 0:
            return True
        elif breath == voicing:
            return False
        return True

    def _apply_simple_scaling(self, wave, breath):
        breath = np.clip(breath, 0, 500)
        scale_factor = breath / 100
        logging.debug(f'Applying simple scaling with factor: {scale_factor}')
        return wave * scale_factor

    def generate_features(self, features_path):
        wave = read_wav(self.in_file)
        wave = torch.from_numpy(wave).to(
            dtype=torch.float32, device=CONFIG.device).unsqueeze(0).unsqueeze(0)
        logging.debug(f"wave shape: {wave.shape}")

        breath = self.flags.get("Hb", 100)
        voicing = self.flags.get("Hv", 100)
        tension = self.flags.get("Ht", 0)
        logging.debug(f'breath: {breath}, voicing: {voicing}, tension: {tension}')

        if self._needs_hnsep_separation(breath, voicing, tension):
            logging.debug('Hb, Hv, or Ht flag requires hnsep separation. Split audio into breath, voicing')

            hnsep_cache_path = self.in_file.with_name(
                f'{self.in_file.stem}_hnsep')

            force_generate = 'G' in self.flags.keys()

            with cache_manager.get_lock(hnsep_cache_path):
                seg_output = cache_manager.load_hnsep_cache(hnsep_cache_path, str(CONFIG.device), force_generate)

                if seg_output is None:
                    logging.info(f'Generating hnsep features for {hnsep_cache_path.name}')
                    with models.gpu_infer_lock, torch.inference_mode():
                        seg_output = models.hnsep_model.predict_fromaudio(wave)
                    seg_output = cache_manager.save_hnsep_cache(
                        hnsep_cache_path, seg_output, device=str(CONFIG.device), force_generate=force_generate
                    )

            breath = np.clip(breath, 0, 500)
            voicing = np.clip(voicing, 0, 150)
            if tension != 0:
                tension = np.clip(tension, -100, 100)
                wave = (breath/100)*(wave - seg_output) + \
                    pre_emphasis_base_tension(
                        (voicing/100)*seg_output, -tension/50)
            else:
                wave = (breath/100)*(wave - seg_output) + \
                    (voicing/100)*seg_output
        elif breath != 100 or voicing != 100:
            # 当 breath == voicing 且不等于 100 时，应用简单缩放优化
            logging.debug(f'Hb == Hv ({breath}), applying optimized simple scaling instead of hnsep separation')
            wave = self._apply_simple_scaling(wave, breath)

        wave = wave.squeeze(1)  # [1, 1, T] -> [1, T], keep on target device
        wave_max = torch.max(torch.abs(wave))
        if wave_max >= 0.5:
            logging.debug('The audio volume is too high. Scaling down to 0.5')
            # 先缩小到最大0.5
            scale = 0.5 / wave_max
            wave = wave * scale
            scale = scale.item()
        else:
            logging.debug('The audio volume is already low enough')
            scale = 1.0

        gender = self.flags.get("g", 0)
        if gender is None:
            gender = 0
        gender = np.clip(gender, -100, 100)
        logging.debug(f'gender: {gender}')

        with models.gpu_infer_lock, torch.inference_mode():
            mel_origin = models.mel_analyzer(
                wave,
                -gender * 0.06, 1).squeeze()
            logging.debug(f'mel_origin: {mel_origin.shape}')
            mel_origin = dynamic_range_compression_torch(mel_origin).cpu().numpy()
        logging.debug('Features generation completed.')

        features = {'mel_origin': mel_origin, 'scale': scale}
        return features

    def resample(self, features):
        if str(self.out_file).strip().lower() in ('nul', '/dev/null'):
            logging.info('Null output file. Skipping...')
            return

        self.out_file = Path(self.out_file)

        scale = features['scale']
        logging.debug(f'scale: {scale}')

        mel_origin = features['mel_origin']
        logging.debug(f'mel_origin: {mel_origin.shape}')

        thop_origin = CONFIG.origin_hop_size / CONFIG.sample_rate
        thop = CONFIG.hop_size / CONFIG.sample_rate
        logging.debug(f'thop_origin: {thop_origin}')
        logging.debug(f'thop: {thop}')

        t_area_origin = np.arange(
            mel_origin.shape[1]) * thop_origin + thop_origin / 2
        total_time = t_area_origin[-1] + thop_origin/2
        logging.debug(f"t_area_mel_origin: {t_area_origin.shape}")
        logging.debug(f"total_time: {total_time}")

        vel = np.exp2(1 - self.velocity / 100)
        offset = self.offset / 1000  # start time
        cutoff = self.cutoff / 1000  # end time
        start = offset
        logging.debug(f'vel:{vel}')
        logging.debug(f'offset:{offset}')
        logging.debug(f'cutoff:{cutoff}')

        logging.debug('Calculating timing.')
        if self.cutoff < 0:  # deal with relative end time
            end = start - cutoff
        else:
            end = total_time - cutoff
        con = start + self.consonant / 1000
        logging.debug(f'start:{start}')
        logging.debug(f'end:{end}')
        logging.debug(f'con:{con}')

        logging.debug('Preparing interpolators.')

        length_req = self.length / 1000
        stretch_length = end - con
        logging.debug(f'length_req: {length_req}')
        logging.debug(f'stretch_length: {stretch_length}')

        if CONFIG.loop_mode or "He" in self.flags.keys():
            # 添加循环拼接模式
            logging.debug('Looping.')
            con_frame = max(0, min(mel_origin.shape[1] - 1, int((con + thop_origin/2)//thop_origin)))
            end_frame = max(con_frame + 1, min(mel_origin.shape[1], int((end + thop_origin/2)//thop_origin)))
            mel_loop = mel_origin[:, con_frame:end_frame]
            logging.debug(f'mel_loop: {mel_loop.shape}')
            if mel_loop.shape[1] > 0:
                pad_loop_size = int(max(1, length_req//thop_origin + 1))
                logging.debug(f'pad_loop_size: {pad_loop_size}')
                padded_mel = np.pad(mel_loop, pad_width=(
                    (0, 0), (0, pad_loop_size)), mode='reflect')  # 多pad一点
                logging.debug(f'padded_mel: {padded_mel.shape}')
                mel_origin = np.concatenate(
                    (mel_origin[:, :con_frame], padded_mel), axis=1)
                logging.debug(f'mel_origin: {mel_origin.shape}')
                stretch_length = pad_loop_size * thop_origin
                t_area_origin = np.arange(
                    mel_origin.shape[1]) * thop_origin + thop_origin / 2
                total_time = t_area_origin[-1] + thop_origin/2
                logging.debug(f'new_total_time: {total_time}')

        # Make interpolators to render new areas
        mel_interp = interp.interp1d(t_area_origin, mel_origin, axis=1)

        if stretch_length <= 1e-6:
            logging.warning(f'stretch_length ({stretch_length:.4f}s) <= 0, clamping scaling_ratio to 1.0')
            scaling_ratio = 1.0
        elif stretch_length < length_req:
            logging.debug('stretch_length < length_req')
            scaling_ratio = length_req / stretch_length
        else:
            logging.debug('stretch_length >= length_req, no stretching needed.')
            scaling_ratio = 1.0

        def stretch(t, con, scaling_ratio):
            return np.where(t < vel*con, t/vel, con + (t - vel*con) / scaling_ratio)

        stretched_n_frames = max(1, int((con*vel + max(0.0, total_time - con)
                              * scaling_ratio) // thop + 1))
        stretched_t_mel = np.arange(stretched_n_frames) * thop + thop / 2
        logging.debug(f'stretched_n_frames: {stretched_n_frames}')
        logging.debug(f'stretched_t_mel: {stretched_t_mel.shape}')

        # 在start左边的mel帧数
        start_left_mel_frames = (start*vel + thop/2)//thop
        if start_left_mel_frames > CONFIG.fill:
            cut_left_mel_frames = start_left_mel_frames - CONFIG.fill
        else:
            cut_left_mel_frames = 0
        logging.debug(f'start_left_mel_frames: {start_left_mel_frames}')
        logging.debug(f'cut_left_mel_frames: {cut_left_mel_frames}')

        # 在length_req+con右边的mel帧数
        end_right_mel_frames = stretched_n_frames - \
            (length_req+con*vel + thop/2)//thop
        if end_right_mel_frames > CONFIG.fill:
            cut_right_mel_frames = end_right_mel_frames - CONFIG.fill
        else:
            cut_right_mel_frames = 0
        logging.debug(f'end_right_mel_frames: {end_right_mel_frames}')
        logging.debug(f'cut_right_mel_frames: {cut_right_mel_frames}')

        logging.debug(f'length_req: {length_req}')
        logging.debug(f'stretch_length: {stretch_length}')
        logging.debug(
            f'(length_req+con*vel + thop/2)//thop: {(length_req+con*vel + thop/2)//thop}')

        slice_start = max(0, int(cut_left_mel_frames))
        slice_end = max(slice_start + 1, int(stretched_n_frames - cut_right_mel_frames))
        sliced_t_mel = stretched_t_mel[slice_start:slice_end]
        if len(sliced_t_mel) > 0:
            stretched_t_mel = sliced_t_mel
        else:
            cut_left_mel_frames = 0
        logging.debug(f'stretched_t_mel: {stretched_t_mel.shape}')

        stretch_t_mel = np.clip(
            stretch(stretched_t_mel, con, scaling_ratio), t_area_origin[0], t_area_origin[-1])
        logging.debug(f'stretch_t_mel: {stretch_t_mel.shape}')

        new_start = start*vel - cut_left_mel_frames * thop
        new_end = (length_req+con*vel) - cut_left_mel_frames * thop
        logging.debug(f'new_start: {new_start}')
        logging.debug(f'new_end: {new_end}')
        logging.debug(f'stretched_t_mel[0]: {stretched_t_mel[0]}')
        logging.debug(f'stretched_t_mel[-1]: {stretched_t_mel[-1]}')

        mel_render = mel_interp(stretch_t_mel)
        logging.debug(f'mel_render: {mel_render.shape}')

        t = np.arange(mel_render.shape[1]) * thop
        logging.debug(f't: {t.shape}')
        logging.debug('Calculating pitch.')
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        if "t" in self.flags.keys() and self.flags["t"]:
            t_val = np.clip(self.flags["t"], -120, 120)
            pitch = pitch + t_val / 10
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96) + new_start
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(np.clip(t, new_start, t_pitch[-1]))
        f0_render = midi_to_hz(pitch_render)
        logging.debug(f'f0_render: {f0_render.shape}')

        logging.debug('Cutting mel and f0.')

        if CONFIG.model_type == "ckpt":

            mel_render = torch.from_numpy(
                mel_render).unsqueeze(0).to(dtype=torch.float32)
            f0_render = torch.from_numpy(f0_render).unsqueeze(
                0).to(dtype=torch.float32)
            logging.debug(f'mel_render: {mel_render.shape}')
            logging.debug(f'f0_render: {f0_render.shape}')

            logging.debug('Rendering audio.')

            with models.gpu_infer_lock, torch.inference_mode():
                wav_con = models.vocoder.spec2wav_torch(mel_render.to(
                    CONFIG.device), f0=f0_render.to(CONFIG.device))
            cut_l = max(0, int(new_start * CONFIG.sample_rate))
            cut_r = max(cut_l + 1, int(new_end * CONFIG.sample_rate))
            render = wav_con[cut_l:cut_r].to('cpu').numpy()
            logging.debug(f'cut_l:{cut_l}')
            logging.debug(
                f'cut_r:{len(wav_con)-cut_r}')
            logging.debug(
                f'mel_l:{(cut_l+256)//CONFIG.hop_size}')
            logging.debug(
                f'mel_r:{(len(wav_con)-cut_r+256)//CONFIG.hop_size}')

            logging.debug(f'wav_con: {wav_con.shape}')
            logging.debug(f'render: {render.shape}')
        elif CONFIG.model_type == "onnx":
            logging.debug('Rendering audio.')
            f0 = f0_render.astype(np.float32)
            mel = mel_render.astype(np.float32)
            # 给mel和f0添加batched维度
            mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
            f0 = np.expand_dims(f0, axis=0)
            input_data = {'mel': mel, 'f0': f0, }
            if CONFIG.device.type != 'cpu' or getattr(models, 'ort_uses_gpu', False):
                with models.gpu_infer_lock:
                    output = models.ort_session.run(['waveform'], input_data)[0]
            else:
                output = models.ort_session.run(['waveform'], input_data)[0]
            wav_con = output[0]

            cut_l = max(0, int(new_start * CONFIG.sample_rate))
            cut_r = max(cut_l + 1, int(new_end * CONFIG.sample_rate))
            render = wav_con[cut_l:cut_r]
            logging.debug(f'cut_l:{cut_l}')
            logging.debug(
                f'cut_r:{len(wav_con)-cut_r}')
            logging.debug(
                f'mel_l:{(cut_l+256)//CONFIG.hop_size}')
            logging.debug(
                f'mel_r:{(len(wav_con)-cut_r+256)//CONFIG.hop_size}')

            logging.debug(f'wav_con: {wav_con.shape}')
            logging.debug(f'render: {render.shape}')
        else:
            raise ValueError(f"Unsupported model type: {CONFIG.model_type}")

        # 添加幅度调制
        A_flag = self.flags.get('A', 0)
        if A_flag != 0:
            logging.debug(f'Applying Amplitude Modulation A={A_flag}')
            A_clamped = np.clip(A_flag, -100, 100)

            if len(pitch_render) > 1 and len(t) > 1 and len(render) > 0:
                pitch_derivative = np.gradient(pitch_render, t)
                gain_at_mel_frames = 5**((10**-4) *
                                         A_clamped * pitch_derivative)
                num_samples = len(render)
                audio_time_vector = np.linspace(
                    new_start, new_end, num=num_samples, endpoint=False)

                interpolated_gain = np.interp(audio_time_vector,
                                              t,  # Time points for gain_at_mel_frames
                                              gain_at_mel_frames,
                                              # Value for time < t[0]
                                              left=gain_at_mel_frames[0],
                                              right=gain_at_mel_frames[-1])  # Value for time > t[-1]

                render = render * interpolated_gain
                logging.debug('Amplitude modulation applied.')
            else:
                logging.warning(
                    "Not enough pitch points (>1) to calculate derivative for Amplitude Modulation.")

        render = render / scale

        if "HG" in self.flags.keys():
            hg_strength = self.flags['HG']
            if hg_strength is not None and hg_strength > 0:
                render = growl(
                    render, CONFIG.sample_rate, frequency=80.0, strength=hg_strength/100)

        # normalize using loudness_norm
        if CONFIG.wave_norm:
            if "P" in self.flags.keys():
                p_strength = self.flags['P']
                if p_strength is not None:
                    if p_strength > 0:
                        render = loudness_norm(
                            render, CONFIG.sample_rate, peak=-1, loudness=-16.0, block_size=0.400, strength=p_strength)
                else:
                    render = loudness_norm(
                        render, CONFIG.sample_rate, peak=-1, loudness=-16.0, block_size=0.400)

        volume_scale = self.volume / 100.0
        render = render * volume_scale

        new_max = np.max(np.abs(render)) if len(render) > 0 else 0.0
        if new_max > CONFIG.peak_limit and new_max > 1e-8:
            render = render * (CONFIG.peak_limit / new_max)

        save_wav(self.out_file, render)
