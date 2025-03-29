import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)
import os
import re
from pathlib import Path # path manipulation
import dataclasses
import yaml

import numpy as np # Numpy <3
import torch
import soundfile as sf # WAV read + write
import scipy.interpolate as interp # Interpolator for feats
import resampy # Resampler (as in sampling rate stuff)
from http.server import BaseHTTPRequestHandler, HTTPServer

from wav2mel import PitchAdjustableMelSpectrogram
from hnsep.nets import CascadedNet

version = '0.0.3-hifisampler'
help_string = '''usage: hifisampler in_file out_file pitch velocity [flags] [offset] [length] [consonant] [cutoff] [volume] [modulation] [tempo] [pitch_string]

Resamples using the PC-NSF-HIFIGAN Vocoder.

arguments:
\tin_file\t\tPath to input file.
\tout_file\tPath to output file.
\tpitch\t\tThe pitch to render on.
\tvelocity\tThe consonant velocity of the render.

optional arguments:
\tflags\t\tThe flags of the render. But now, it's not implemented yet. 
\toffset\t\tThe offset from the start of the render area of the sample. (default: 0)
\tlength\t\tThe length of the stretched area in milliseconds. (default: 1000)
\tconsonant\tThe unstretched area of the render in milliseconds. (default: 0)
\tcutoff\t\tThe cutoff from the end or from the offset for the render area of the sample. (default: 0)
\tvolume\t\tThe volume of the render in percentage. (default: 100)
\tmodulation\tThe pitch modulation of the render in percentage. (default: 0)
\ttempo\t\tThe tempo of the render. Needs to have a ! at the start. (default: !100)
\tpitch_string\tThe UTAU pitchbend parameter written in Base64 with RLE encoding. (default: AA)'''

notes = {'C' : 0, 'C#' : 1, 'D' : 2, 'D#' : 3, 'E' : 4, 'F' : 5, 'F#' : 6, 'G' : 7, 'G#' : 8, 'A' : 9, 'A#' : 10, 'B' : 11} # Note names lol
note_re = re.compile(r'([A-G]#?)(-?\d+)') # Note Regex for conversion
cache_ext = '.hifi.npz' # cache file extension

# Flags
flags = ['fe', 'fl', 'fo', 'fv', 'fp', 've', 'vo', 'g', 't', 'A', 'B', 'G', 'P', 'S', 'p', 'R', 'D', 'C', 'Z', 'Me']
flag_re = '|'.join(flags)
flag_re = f'({flag_re})([+-]?\\d+)?'
flag_re = re.compile(flag_re)

@dataclasses.dataclass
class Config:
    sample_rate: int = 44100  # UTAU only really likes 44.1khz
    win_size: int = 2048     # 必须和vocoder训练时一致   
    hop_size: int = 512      # 必须和vocoder训练时一致     
    origin_hop_size: int = 128 # 插值前的hopsize,可以适当调小改善长音的电音
    n_mels: int = 128        # 必须和vocoder训练时一致 
    n_fft: int = 2048        # 必须和vocoder训练时一致 
    mel_fmin: float = 40     # 必须和vocoder训练时一致 
    mel_fmax: float = 16000  # 必须和vocoder训练时一致 
    fill: int = 6
    vocoder_path: str = r"path\to\your\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt"
    model_type: str = 'ckpt' # or 'onnx'
    hnsep_model_path: str = r"path\to\your\hnsep_240512\vr\model.pt"
    wave_norm: bool = True
    loop_mode: bool = False
    peak_limit: float = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400, strength=100
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)
        strength: strength of the normalization. Defaults to 100.

    Returns:
        loudness normalized audio
    """

    # peak normalize audio to [peak] dB
    original_length = len(audio)
    # Check if the audio is shorter than block_size and pad if necessary
    if original_length < int(rate * block_size):
        padding_length = int(rate * block_size) - original_length
        audio = np.pad(audio, (0, padding_length), mode='reflect')
    
    # Peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # Measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    # Apply strength to calculate the target loudness
    final_loudness = _loudness + (loudness - _loudness) * strength / 100

    # Loudness normalize audio to [loudness] LUFS
    audio = pyln.normalize.loudness(audio, _loudness, final_loudness)
    
    # If original audio was shorter than block_size, crop it back to its original length
    if original_length < int(rate * block_size):
        audio = audio[:original_length]
    
    return audio

def load_sep_model(model_path, device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(
                args.n_fft, 
                args.hop_length, 
                args.n_out, 
                args.n_out_lstm, 
                True, 
                is_mono=args.is_mono,
                fixed_length = True if args.fixed_length is None else args.fixed_length)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, args

# Pitch string interpreter
def to_uint6(b64):
    """Convert one Base64 character to an unsigned integer.

    Parameters
    ----------
    b64 : str
        The Base64 character.

    Returns
    -------
    int
        The equivalent of the Base64 character as an integer.
    """
    c = ord(b64) # Convert based on ASCII mapping
    if c >= 97:
        return c - 71
    elif c >= 65:
        return c - 65
    elif c >= 48:
        return c + 4
    elif c == 43:
        return 62
    elif c == 47:
        return 63
    else:
        raise Exception

def to_int12(b64):
    """Converts two Base64 characters to a signed 12-bit integer.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    int
        The equivalent of the Base64 characters as a signed 12-bit integer (-2047 to 2048)
    """
    uint12 = to_uint6(b64[0]) << 6 | to_uint6(b64[1]) # Combined uint6 to uint12
    if uint12 >> 11 & 1 == 1: # Check most significant bit to simulate two's complement
        return uint12 - 4096
    else:
        return uint12

def to_int12_stream(b64):
    """Converts a Base64 string to a list of integers.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    list
        The equivalent of the Base64 string if split every 12-bits and interpreted as a signed 12-bit integer.
    """
    res = []
    for i in range(0, len(b64), 2):
        res.append(to_int12(b64[i:i+2]))
    return res

def pitch_string_to_cents(x):
    """Converts UTAU's pitchbend argument to an ndarray representing the pitch offset in cents.

    Parameters
    ----------
    x : str
        The pitchbend argument.

    Returns
    -------
    ndarray
        The pitchbend argument as pitch offset in cents.
    """
    pitch = x.split('#') # Split RLE Encoding
    res = []
    for i in range(0, len(pitch), 2):
        # Go through each pair
        p = pitch[i:i+2]
        if len(p) == 2:
            # Decode pitch string and extend RLE
            pitch_str, rle = p
            res.extend(to_int12_stream(pitch_str))
            res.extend([res[-1]] * int(rle))
        else:
            # Decode last pitch string without RLE if it exists
            res.extend(to_int12_stream(p[0]))
    res = np.array(res, dtype=np.int32)
    if np.all(res == res[0]):
        return np.zeros(res.shape)
    else:
        return np.concatenate([res, np.zeros(1)])

# Pitch conversion
def note_to_midi(x):
    """Note name to MIDI note number."""
    note, octave = note_re.match(x).group(1, 2)
    octave = int(octave) + 1
    return octave * 12 + notes[note]

def midi_to_hz(x):
    """MIDI note number to Hertz using equal temperament. A4 = 440 Hz."""
    return 440 * np.exp2((x - 69) / 12)

# WAV read/write
def read_wav(loc):
    """Read audio files supported by soundfile and resample to 44.1kHz if needed. Mixes down to mono if needed.

    Parameters
    ----------
    loc : str or file
        Input audio file.

    Returns
    -------
    ndarray
        Data read from WAV file remapped to [-1, 1] and in 44.1kHz
    """
    if type(loc) == str: # make sure input is Path
        loc = Path(loc)

    exists = loc.exists()
    if not exists: # check for alternative files
        for ext in sf.available_formats().keys():
            loc = loc.with_suffix('.' + ext.lower())
            exists = loc.exists()
            if exists:
                break

    if not exists:
        raise FileNotFoundError("No supported audio file was found.")
    
    x, fs = sf.read(str(loc))
    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than stereo
        x = np.mean(x, axis=1)

    if fs != Config.sample_rate:
        x = resampy.resample(x, fs, Config.sample_rate)

    return x

def save_wav(loc, x):
    """Save data into a WAV file.

    Parameters
    ----------
    loc : str or file
        Output WAV file.

    x : ndarray
        Audio data in 44.1kHz within [-1, 1].

    Returns
    -------
    None
    """
    try:
        sf.write(str(loc), x, Config.sample_rate, 'PCM_16')
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")


class Resampler:
    """
    A class for the UTAU resampling process.

    Attributes
    ----------
    in_file : str
        Path to input file.

    out_file : str
        Path to output file.

    pitch : str
        The pitch of the note.

    velocity : str or float
        The consonant velocity of the note.

    flags : str
        The flags of the note.

    offset : str or float
        The offset from the start for the render area of the sample.

    length : str or int
        The length of the stretched area in milliseconds.

    consonant : str or float
        The unstretched area of the render.

    cutoff : str or float
        The cutoff from the end or from the offset for the render area of the sample.

    volume : str or float
        The volume of the note in percentage.

    modulation : str or float
        The modulation of the note in percentage.

    tempo : str
        The tempo of the note.

    pitch_string : str
        The UTAU pitchbend parameter.

    Methods
    -------    
    render(self):
        The rendering workflow. Immediately starts when class is initialized.

    get_features(self):
        Gets the MEL features either from a cached file or generating it if it doesn't exist.

    generate_features(self, features_path):
        Generates MEL features and saves it for later.

    resample(self, features):
        Renders a WAV file using the passed MEL features.
    """
    def __init__(self, in_file, out_file, pitch, velocity, flags='', offset=0, length=1000, consonant=0, cutoff=0, volume=100, modulation=0, tempo='!100', pitch_string='AA'):
        """Initializes the renderer and immediately starts it.

        Parameters
        ---------
        in_file : str
            Path to input file.

        out_file : str
            Path to output file.

        pitch : str
            The pitch of the note.

        velocity : str or float
            The consonant velocity of the note.

        flags : str
            The flags of the note.

        offset : str or float
            The offset from the start for the render area of the sample.

        length : str or int
            The length of the stretched area in milliseconds.

        consonant : str or float
            The unstretched area of the render.

        cutoff : str or float
            The cutoff from the end or from the offset for the render area of the sample.

        volume : str or float
            The volume of the note in percentage.

        modulation : str or float
            The modulation of the note in percentage.

        tempo : str
            The tempo of the note.

        pitch_string : str
            The UTAU pitchbend parameter.
        """
        self.in_file = Path(in_file)
        self.out_file = out_file
        self.pitch = note_to_midi(pitch)
        self.velocity = float(velocity)
        self.flags = {k : int(v) if v else None for k, v in flag_re.findall(flags.replace('/', ''))}
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
        """The rendering workflow. Immediately starts when class is initialized.

        Parameters
        ----------
        None
        """
        features = self.get_features()
        self.resample(features)

    def get_features(self):
        """Gets the MEL features either from a cached file or generating it if it doesn't exist.

        Parameters
        ----------
        None

        Returns
        -------
        features : dict
            A dictionary of the MEL.
        """
        # Setup cache path file
        fname = self.in_file.name
        features_path = self.in_file.with_suffix(cache_ext)
        features = None
        if "B" in self.flags.keys():
            #把B的数值加入Cache path里来区分
            features_path = features_path.with_name(f'{fname}_B{self.flags["B"]}{features_path.suffix}')
        logging.info(f'Cache path: {features_path}')

        if 'G' in self.flags.keys():
            logging.info('G flag exists. Forcing feature generation.')
            features = self.generate_features(features_path)
        elif os.path.exists(features_path):
            # Load if it exists
            logging.info(f'Reading {fname}{cache_ext}.')
            features = np.load(features_path)
        else:
            # Generate if not
            logging.info(f'{fname}{cache_ext} not found. Generating features.')
            features = self.generate_features(features_path)

        return features
    def generate_features(self, features_path):
        """Generates PC-NSF-hifigan features and saves it for later.

        Parameters
        ----------
        features_path : str or file
            The path for caching the features.

        Returns
        -------
        features : dict
            A dictionary of the MEL.
        """
        wave = read_wav(self.in_file)
        wave = torch.from_numpy(wave).to(dtype=torch.float32, device=Config.device).unsqueeze(0).unsqueeze(0)
        print(wave.shape)

        if "B" in self.flags.keys():
            breath = self.flags['B']
            if breath != 50:
                logging.info('B flag exists. Breathing.')
                with torch.no_grad():
                    seg_output = hnsep_model.predict_fromaudio(wave) 
                print(seg_output.shape)
                breath = np.clip(breath, 0, 100)
                if breath < 50:
                    wave = (breath/50)*(wave - seg_output) + seg_output
                else:    
                    wave = (wave - seg_output) + seg_output*((100-breath)/50)

        wave = wave.squeeze(0).squeeze(0).cpu().numpy()
        wave = torch.from_numpy(wave).to(dtype=torch.float32, device=Config.device).unsqueeze(0) # 默认不缩放
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

        mel_origin = melAnalysis(
            wave,
            0, 1).squeeze()
        logging.info(f'mel_origin: {mel_origin.shape}')
        mel_origin = dynamic_range_compression_torch(mel_origin).cpu().numpy()
        logging.info('Saving features.')
        
        features = {'mel_origin' : mel_origin, 'scale' : scale}
        np.savez_compressed(features_path, **features)

        return features
    def resample(self, features):
        """
        Renders a WAV file using the passed MEL features.

        Parameters
        ----------
        features : dict
            A dictionary of the mel.
 
        Returns
        -------
        None
        """
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

        thop_origin = Config.origin_hop_size / Config.sample_rate
        thop = Config.hop_size / Config.sample_rate
        logging.info(f'thop_origin: {thop_origin}')
        logging.info(f'thop: {thop}')

        t_area_origin = np.arange(mel_origin.shape[1]) * thop_origin + thop_origin / 2
        total_time = t_area_origin[-1] + thop_origin/2
        logging.info(f"t_area_mel_origin: {t_area_origin.shape}")
        logging.info(f"total_time: {total_time}")

        vel = np.exp2(1 - self.velocity / 100)
        offset = self.offset / 1000 # start time
        cutoff = self.cutoff / 1000 # end time
        start = offset
        logging.info(f'vel:{vel}')
        logging.info(f'offset:{offset}')
        logging.info(f'cutoff:{cutoff}')

        logging.info('Calculating timing.') 
        if self.cutoff < 0: # deal with relative end time
            end = start - cutoff       #???
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

        if Config.loop_mode or "Me" in self.flags.keys():
            # 添加循环拼接模式
            logging.info('Looping.')
            logging.info(f'con_mel_frame: {int((con + thop_origin/2)//thop_origin)}')
            mel_loop = mel_origin[:, int((con + thop_origin/2)//thop_origin):int((end + thop_origin/2)//thop_origin)]
            logging.info(f'mel_loop: {mel_loop.shape}')
            pad_loop_size = length_req//thop_origin + 1
            logging.info(f'pad_loop_size: {pad_loop_size}')
            padded_mel = np.pad(mel_loop, pad_width=((0,0),(0, int(pad_loop_size))), mode='reflect') #多pad一点
            logging.info(f'padded_mel: {padded_mel.shape}')
            mel_origin = np.concatenate((mel_origin[:,:int((con + thop_origin/2)//thop_origin)], padded_mel), axis=1)
            logging.info(f'mel_origin: {mel_origin.shape}')
            stretch_length = pad_loop_size*thop_origin
            t_area_origin = np.arange(mel_origin.shape[1]) * thop_origin + thop_origin / 2
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
        
        stretched_n_frames = (con*vel + (total_time - con)*scaling_ratio) // thop + 1
        stretched_t_mel = np.arange(stretched_n_frames) * thop + thop / 2
        logging.info(f'stretched_n_frames: {stretched_n_frames}')
        logging.info(f'stretched_t_mel: {stretched_t_mel.shape}')

        # 在start左边的mel帧数
        start_left_mel_frames = (start*vel + thop/2)//thop
        if start_left_mel_frames > Config.fill:
            cut_left_mel_frames = start_left_mel_frames - Config.fill
        else:
            cut_left_mel_frames = 0
        logging.info(f'start_left_mel_frames: {start_left_mel_frames}')
        logging.info(f'cut_left_mel_frames: {cut_left_mel_frames}')

        # 在length_req+con右边的mel帧数
        end_right_mel_frames = stretched_n_frames - (length_req+con*vel + thop/2)//thop
        if end_right_mel_frames > Config.fill:
            cut_right_mel_frames = end_right_mel_frames - Config.fill
        else:
            cut_right_mel_frames = 0
        logging.info(f'end_right_mel_frames: {end_right_mel_frames}')
        logging.info(f'cut_right_mel_frames: {cut_right_mel_frames}')

        logging.info(f'length_req: {length_req}')
        logging.info(f'stretch_length: {stretch_length}')
        logging.info(f'(length_req+con*vel + thop/2)//thop: {(length_req+con*vel + thop/2)//thop}')

        stretched_t_mel = stretched_t_mel[int(cut_left_mel_frames):int(stretched_n_frames-cut_right_mel_frames)]
        logging.info(f'stretched_t_mel: {stretched_t_mel.shape}')

        stretch_t_mel = np.clip(stretch(stretched_t_mel, con, scaling_ratio),0,t_area_origin[-1])
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
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96) + new_start
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(np.clip(t, new_start, t_pitch[-1]))
        f0_render = midi_to_hz(pitch_render)
        logging.info(f'f0_render: {f0_render.shape}')

        logging.info('Cutting mel and f0.')

        if Config.model_type == "ckpt":

            mel_render = torch.from_numpy(mel_render).unsqueeze(0).to(dtype=torch.float32)
            f0_render = torch.from_numpy(f0_render).unsqueeze(0).to(dtype=torch.float32)
            logging.info(f'mel_render: {mel_render.shape}')
            logging.info(f'f0_render: {f0_render.shape}')

            logging.info('Rendering audio.')

            wav_con = vocoder.spec2wav_torch(mel_render.to(Config.device), f0 = f0_render.to(Config.device))
            render = wav_con[int(new_start * Config.sample_rate):int(new_end * Config.sample_rate)].to('cpu').numpy()
            logging.info(f'cut_l:{int(new_start * Config.sample_rate)}')
            logging.info(f'cut_r:{len(wav_con)-int(new_end * Config.sample_rate)}')
            logging.info(f'mel_l:{(int(new_start * Config.sample_rate)+256)//Config.hop_size}')
            logging.info(f'mel_r:{(len(wav_con)-int(new_end * Config.sample_rate)+256)//Config.hop_size}')

            logging.info(f'wav_con: {wav_con.shape}')
            logging.info(f'render: {render.shape}')
        elif Config.model_type == "onnx":
            logging.info('Rendering audio.')
            f0 = f0_render.astype(np.float32)
            mel = mel_render.astype(np.float32)
            #给mel和f0添加batched维度
            mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
            f0 = np.expand_dims(f0, axis=0)
            input_data = {'mel': mel,'f0': f0,}
            output = ort_session.run(['waveform'], input_data)[0]
            wav_con = output[0]

            render = wav_con[int(new_start * Config.sample_rate):int(new_end * Config.sample_rate)]
            logging.info(f'cut_l:{int(new_start * Config.sample_rate)}')
            logging.info(f'cut_r:{len(wav_con)-int(new_end * Config.sample_rate)}')
            logging.info(f'mel_l:{(int(new_start * Config.sample_rate)+256)//Config.hop_size}')
            logging.info(f'mel_r:{(len(wav_con)-int(new_end * Config.sample_rate)+256)//Config.hop_size}')

            logging.info(f'wav_con: {wav_con.shape}')
            logging.info(f'render: {render.shape}')
        else:
            raise ValueError(f"Unsupported model type: {Config.model_type}")
        

        render = render / scale
        new_max = np.max(np.abs(render))

        # normalize using loudness_norm
        if Config.wave_norm:
            if "P" in self.flags.keys():
                p_strength = self.flags['P']
                if p_strength is not None:
                    render = loudness_norm(render, Config.sample_rate, peak = -1, loudness=-16.0, block_size=0.400, strength=p_strength)
                else:
                    render = loudness_norm(render, Config.sample_rate, peak = -1, loudness=-16.0, block_size=0.400)

        if new_max > Config.peak_limit:
            render = render / new_max
        save_wav(self.out_file, render)

def split_arguments(input_string):
    # Regular expression to match two file paths at the beginning
    otherargs = input_string.split(' ')[-11:]
    file_path_strings = ' '.join(input_string.split(' ')[:-11])

    first_file, second_file = file_path_strings.split('.wav ')
    return [first_file+".wav", second_file] + otherargs

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        logging.info(self.requestline)
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data_string = post_data.decode('utf-8')
        logging.info(f"post_data_string: {post_data_string}")
        #try:
        sliced = split_arguments(post_data_string)

        Resampler(*sliced)
        '''
        except Exception as e:
            trcbk = traceback.format_exc()
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"An error occurred.\n{trcbk}".encode('utf-8'))
        self.send_response(200)
        self.end_headers()
        '''
def run(server_class=HTTPServer, handler_class=RequestHandler, port=8572):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info(f'Starting http server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    if Config.wave_norm:
        import pyloudnorm as pyln
    logging.info(f'hachisampler {version}')

    # Load HifiGAN
    vocoder_path = Path(Config.vocoder_path)
    onnx_default_path = Path(r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")
    ckpt_default_path = Path(r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt")
    if not vocoder_path.exists():
        if ckpt_default_path.exists():
            vocoder_path = ckpt_default_path
        elif onnx_default_path.exists():
            vocoder_path = onnx_default_path
        else:
            raise FileNotFoundError("No HifiGAN model found.")

    if vocoder_path.suffix == '.ckpt':
        from nsf_hifigan import NsfHifiGAN
        Config.model_type = 'ckpt'
        vocoder = NsfHifiGAN(model_path=vocoder_path)
        vocoder.to_device(Config.device)
        logging.info(f'Loaded HifiGAN: {vocoder}')
    elif vocoder_path.suffix == '.onnx':
        import onnxruntime
        Config.model_type = 'onnx'
        ort_session = onnxruntime.InferenceSession(str(vocoder_path),providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        logging.info(f'Loaded HifiGAN: {vocoder_path}')
    else:
        Config.model_type = vocoder_path.suffix
        raise ValueError(f'Invalid model type: {Config.model_type}')
    
    hnsep_model, hnsep_model_args = load_sep_model(Config.hnsep_model_path, Config.device)
    logging.info(f'Loaded HN-SEP: {Config.hnsep_model_path}')

    melAnalysis = PitchAdjustableMelSpectrogram(
        sample_rate=Config.sample_rate, 
        n_fft=Config.n_fft, 
        win_length=Config.win_size, 
        hop_length=Config.origin_hop_size, 
        f_min=Config.mel_fmin, 
        f_max=Config.mel_fmax,
        n_mels=Config.n_mels
        )
    # Start server
    try:
        run()
    except Exception as e:
        name = e.__class__.__name__
        if name == 'TypeError':
            logging.info(help_string)
        else:
            raise e
