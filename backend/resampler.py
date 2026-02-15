import logging
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
        self.tempo = float(tempo[1:])
        self.pitchbend = pitch_string_to_cents(pitch_string)

        self.render()

    def render(self):
        features = self.get_features()
        self.resample(features)

    def get_features(self):
        features_path = self.in_file.with_suffix(cache_ext)

        self.flags['Hb'] = self.flags.get('Hb', 100)
        self.flags['Hv'] = self.flags.get('Hv', 100)
        self.flags['Ht'] = self.flags.get('Ht', 0)
        self.flags['g'] = self.flags.get('g', 0)

        flag_suffix = '_'.join(f"{k}{v if v is not None else ''}" for k, v in sorted(
            self.flags.items()) if k in ['Hb', 'Hv', 'Ht', 'g'])
        if flag_suffix:
            features_path = features_path.with_name(
                f'{self.in_file.stem}_{flag_suffix}{cache_ext}')
        else:
            features_path = features_path.with_name(
                f'{self.in_file.stem}{cache_ext}')

        force_generate = 'G' in self.flags.keys()
        features = cache_manager.load_features_cache(features_path, force_generate)

        if features is not None:
            return features
        logging.info(f'{features_path} not found or forcing generation. Generating features.')
        features = self.generate_features(features_path)
        return cache_manager.save_features_cache(features_path, features)

    def _needs_hnsep_separation(self, breath, voicing, tension):
        if tension != 0:
            return True
        elif breath == voicing:
            return False
        return True

    def _apply_simple_scaling(self, wave, breath):
        breath = np.clip(breath, 0, 500)
        scale_factor = breath / 100
        logging.info(f'Applying simple scaling with factor: {scale_factor}')
        return wave * scale_factor

    def generate_features(self, features_path):
        wave = read_wav(self.in_file)
        wave = torch.from_numpy(wave).to(
            dtype=torch.float32, device=CONFIG.device).unsqueeze(0).unsqueeze(0)
        logging.info(wave.shape)

        breath = self.flags.get("Hb", 100)
        voicing = self.flags.get("Hv", 100)
        tension = self.flags.get("Ht", 0)
        logging.info(f'breath: {breath}, voicing: {voicing}, tension: {tension}')
        
        if self._needs_hnsep_separation(breath, voicing, tension):
            logging.info('Hb, Hv, or Ht flag requires hnsep separation. Split audio into breath, voicing')

            hnsep_cache_path = self.in_file.with_name(
                f'{self.in_file.stem}_hnsep')
            
            force_generate = 'G' in self.flags.keys()
            # 缓存优先

            seg_output = cache_manager.load_hnsep_cache(hnsep_cache_path, str(CONFIG.device), force_generate)
            
            if seg_output is None:
                logging.info(f'Generating hnsep features for {hnsep_cache_path}')
                with torch.inference_mode():
                    seg_output = models.hnsep_model.predict_fromaudio(wave)
                seg_output = cache_manager.save_hnsep_cache(hnsep_cache_path, seg_output)

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
            logging.info(f'Hb == Hv ({breath}), applying optimized simple scaling instead of hnsep separation')
            wave = self._apply_simple_scaling(wave, breath)
        wave = wave.squeeze(0).squeeze(0).cpu().numpy()
        wave = torch.from_numpy(wave).to(
            dtype=torch.float32, device=CONFIG.device).unsqueeze(0)  # 默认不缩放
        wave_max = torch.max(torch.abs(wave))
        if wave_max >= 0.5:
            logging.info('The audio volume is too high. Scaling down to 0.5')
            # 先缩小到最大0.5
            scale = 0.5 / wave_max
            wave = wave * scale
            scale = scale.item()
        else:
            logging.info('The audio volume is already low enough')
            scale = 1.0

        gender = self.flags.get("g", 0)
        gender = np.clip(gender, -600, 600)
        logging.info(f'gender: {gender}')

        mel_origin = models.mel_analyzer(
            wave,
            gender/100, 1).squeeze()
        logging.info(f'mel_origin: {mel_origin.shape}')
        mel_origin = dynamic_range_compression_torch(mel_origin).cpu().numpy()
        logging.info('Features generation completed.')

        features = {'mel_origin': mel_origin, 'scale': scale}
        return features

    def resample(self, features):
        if self.out_file == 'nul':
            logging.info('Null output file. Skipping...')
            return

        mod = self.modulation / 100
        logging.info(f"mod: {mod}")

        self.out_file = Path(self.out_file)
        wave = read_wav(Path(self.in_file))
        logging.info(f'wave: {wave.shape}')

        scale = features['scale']
        logging.info(f'scale: {scale}')

        mel_origin = features['mel_origin']
        logging.info(f'mel_origin: {mel_origin.shape}')

        thop_origin = CONFIG.origin_hop_size / CONFIG.sample_rate
        thop = CONFIG.hop_size / CONFIG.sample_rate
        logging.info(f'thop_origin: {thop_origin}')
        logging.info(f'thop: {thop}')

        t_area_origin = np.arange(
            mel_origin.shape[1]) * thop_origin + thop_origin / 2
        total_time = t_area_origin[-1] + thop_origin/2
        logging.info(f"t_area_mel_origin: {t_area_origin.shape}")
        logging.info(f"total_time: {total_time}")

        vel = np.exp2(1 - self.velocity / 100)
        offset = self.offset / 1000  # start time
        cutoff = self.cutoff / 1000  # end time
        start = offset
        logging.info(f'vel:{vel}')
        logging.info(f'offset:{offset}')
        logging.info(f'cutoff:{cutoff}')

        logging.info('Calculating timing.')
        if self.cutoff < 0:  # deal with relative end time
            end = start - cutoff  # ???
        else:
            end = total_time - cutoff
        con = start + self.consonant / 1000
        logging.info(f'start:{start}')
        logging.info(f'end:{end}')
        logging.info(f'con:{con}')

        logging.info('Preparing interpolators.')

        length_req = self.length / 1000
        stretch_length = end - con
        logging.info(f'length_req: {length_req}')
        logging.info(f'stretch_length: {stretch_length}')

        # `He` toggles extension strategy against config default:
        # - loop_mode=False + no He => stretch
        # - loop_mode=False + He    => loop
        # - loop_mode=True  + no He => loop
        # - loop_mode=True  + He    => stretch
        use_loop_mode = bool(CONFIG.loop_mode) ^ ("He" in self.flags.keys())
        if use_loop_mode:
            logging.info('Looping.')
            logging.info(
                f'con_mel_frame: {int((con + thop_origin/2)//thop_origin)}')
            mel_loop = mel_origin[:, int(
                (con + thop_origin/2)//thop_origin):int((end + thop_origin/2)//thop_origin)]
            logging.info(f'mel_loop: {mel_loop.shape}')
            pad_loop_size = length_req//thop_origin + 1
            logging.info(f'pad_loop_size: {pad_loop_size}')
            padded_mel = np.pad(mel_loop, pad_width=(
                (0, 0), (0, int(pad_loop_size))), mode='reflect')  # 多pad一点
            logging.info(f'padded_mel: {padded_mel.shape}')
            mel_origin = np.concatenate(
                (mel_origin[:, :int((con + thop_origin/2)//thop_origin)], padded_mel), axis=1)
            logging.info(f'mel_origin: {mel_origin.shape}')
            stretch_length = pad_loop_size*thop_origin
            t_area_origin = np.arange(
                mel_origin.shape[1]) * thop_origin + thop_origin / 2
            total_time = t_area_origin[-1] + thop_origin/2
            logging.info(f'new_total_time: {total_time}')

        # Make interpolators to render new areas
        mel_interp = interp.interp1d(t_area_origin, mel_origin, axis=1)

        if stretch_length < length_req:
            logging.info('stretch_length < length_req')
            scaling_ratio = length_req / stretch_length
        else:
            logging.info('stretch_length >= length_req, no stretching needed.')
            scaling_ratio = 1

        def stretch(t, con, scaling_ratio):
            return np.where(t < vel*con, t/vel, con + (t - vel*con) / scaling_ratio)

        stretched_n_frames = (con*vel + (total_time - con)
                              * scaling_ratio) // thop + 1
        stretched_t_mel = np.arange(stretched_n_frames) * thop + thop / 2
        logging.info(f'stretched_n_frames: {stretched_n_frames}')
        logging.info(f'stretched_t_mel: {stretched_t_mel.shape}')

        # 在start左边的mel帧数
        start_left_mel_frames = (start*vel + thop/2)//thop
        if start_left_mel_frames > CONFIG.fill:
            cut_left_mel_frames = start_left_mel_frames - CONFIG.fill
        else:
            cut_left_mel_frames = 0
        logging.info(f'start_left_mel_frames: {start_left_mel_frames}')
        logging.info(f'cut_left_mel_frames: {cut_left_mel_frames}')

        # 在length_req+con右边的mel帧数
        end_right_mel_frames = stretched_n_frames - \
            (length_req+con*vel + thop/2)//thop
        if end_right_mel_frames > CONFIG.fill:
            cut_right_mel_frames = end_right_mel_frames - CONFIG.fill
        else:
            cut_right_mel_frames = 0
        logging.info(f'end_right_mel_frames: {end_right_mel_frames}')
        logging.info(f'cut_right_mel_frames: {cut_right_mel_frames}')

        logging.info(f'length_req: {length_req}')
        logging.info(f'stretch_length: {stretch_length}')
        logging.info(
            f'(length_req+con*vel + thop/2)//thop: {(length_req+con*vel + thop/2)//thop}')

        stretched_t_mel = stretched_t_mel[int(cut_left_mel_frames):int(
            stretched_n_frames-cut_right_mel_frames)]
        logging.info(f'stretched_t_mel: {stretched_t_mel.shape}')

        stretch_t_mel = np.clip(
            stretch(stretched_t_mel, con, scaling_ratio), 0, t_area_origin[-1])
        logging.info(f'stretch_t_mel: {stretch_t_mel.shape}')

        new_start = start*vel - cut_left_mel_frames * thop
        new_end = (length_req+con*vel) - cut_left_mel_frames * thop
        logging.info(f'new_start: {new_start}')
        logging.info(f'new_end: {new_end}')
        logging.info(f'stretched_t_mel[0]: {stretched_t_mel[0]}')
        logging.info(f'stretched_t_mel[-1]: {stretched_t_mel[-1]}')

        mel_render = mel_interp(stretch_t_mel)
        logging.info(f'mel_render: {mel_render.shape}')

        t = np.arange(mel_render.shape[1]) * thop
        logging.info(f't: {t.shape}')
        logging.info('Calculating pitch.')
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        if "t" in self.flags.keys() and self.flags["t"]:
            pitch = pitch + self.flags["t"] / 100
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96) + new_start
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(np.clip(t, new_start, t_pitch[-1]))
        f0_render = midi_to_hz(pitch_render)
        logging.info(f'f0_render: {f0_render.shape}')

        logging.info('Cutting mel and f0.')

        if CONFIG.model_type == "ckpt":

            mel_render = torch.from_numpy(
                mel_render).unsqueeze(0).to(dtype=torch.float32)
            f0_render = torch.from_numpy(f0_render).unsqueeze(
                0).to(dtype=torch.float32)
            logging.info(f'mel_render: {mel_render.shape}')
            logging.info(f'f0_render: {f0_render.shape}')

            logging.info('Rendering audio.')

            with torch.inference_mode():
                wav_con = models.vocoder.spec2wav_torch(mel_render.to(
                    CONFIG.device), f0=f0_render.to(CONFIG.device))
            render = wav_con[int(new_start * CONFIG.sample_rate):int(new_end * CONFIG.sample_rate)].to('cpu').numpy()
            logging.info(f'cut_l:{int(new_start * CONFIG.sample_rate)}')
            logging.info(
                f'cut_r:{len(wav_con)-int(new_end * CONFIG.sample_rate)}')
            logging.info(
                f'mel_l:{(int(new_start * CONFIG.sample_rate)+256)//CONFIG.hop_size}')
            logging.info(
                f'mel_r:{(len(wav_con)-int(new_end * CONFIG.sample_rate)+256)//CONFIG.hop_size}')

            logging.info(f'wav_con: {wav_con.shape}')
            logging.info(f'render: {render.shape}')
        elif CONFIG.model_type == "onnx":
            logging.info('Rendering audio.')
            f0 = f0_render.astype(np.float32)
            mel = mel_render.astype(np.float32)
            # 给mel和f0添加batched维度
            mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
            f0 = np.expand_dims(f0, axis=0)
            input_data = {'mel': mel, 'f0': f0, }
            output = models.ort_session.run(['waveform'], input_data)[0]
            wav_con = output[0]

            render = wav_con[int(new_start * CONFIG.sample_rate):int(new_end * CONFIG.sample_rate)]
            logging.info(f'cut_l:{int(new_start * CONFIG.sample_rate)}')
            logging.info(
                f'cut_r:{len(wav_con)-int(new_end * CONFIG.sample_rate)}')
            logging.info(
                f'mel_l:{(int(new_start * CONFIG.sample_rate)+256)//CONFIG.hop_size}')
            logging.info(
                f'mel_r:{(len(wav_con)-int(new_end * CONFIG.sample_rate)+256)//CONFIG.hop_size}')

            logging.info(f'wav_con: {wav_con.shape}')
            logging.info(f'render: {render.shape}')
        else:
            raise ValueError(f"Unsupported model type: {CONFIG.model_type}")

        # 添加幅度调制
        A_flag = self.flags.get('A', 0)
        if A_flag != 0:
            logging.info(f'Applying Amplitude Modulation A={A_flag}')
            A_clamped = np.clip(A_flag, -100, 100)

            if len(pitch_render) > 1 and len(t) > 1:
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
                logging.info('Amplitude modulation applied.')
            else:
                logging.warning(
                    "Not enough pitch points (>1) to calculate derivative for Amplitude Modulation.")

        render = render / scale
        new_max = np.max(np.abs(render))

        if "HG" in self.flags.keys():
            hg_strength = self.flags['HG']
            if hg_strength is not None:
                render = growl(
                    render, CONFIG.sample_rate, frequency=80.0, strength=hg_strength/100)

        # normalize using loudness_norm
        if CONFIG.wave_norm:
            if "P" in self.flags.keys():
                p_strength = self.flags['P']
                if p_strength is not None:
                    render = loudness_norm(
                        render, CONFIG.sample_rate, peak=-1, loudness=-16.0, block_size=0.400, strength=p_strength)
                else:
                    render = loudness_norm(
                        render, CONFIG.sample_rate, peak=-1, loudness=-16.0, block_size=0.400)

        if new_max > CONFIG.peak_limit:
            render = render / new_max

        volume_scale = self.volume / 100.0
        render = render * volume_scale

        save_wav(self.out_file, render)
