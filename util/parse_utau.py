import re
import numpy as np

notes = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'E#': 5,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11, 'B#': 0,
}  # Note names lol
note_re = re.compile(r'^([A-G][#b]?)(-?\d+)$')  # Note Regex for conversion
flags = ['fe', 'fl', 'fo', 'fv', 'fp', 've', 'vo', 'g', 't',
         'A', 'B', 'G', 'P', 'S', 'p', 'R', 'D', 'C', 'Z', 'Hv', 'Hb', 'Ht', 'He', 'HG']
flag_re = re.compile(f"({'|'.join(sorted(flags, key=len, reverse=True))})([+-]?\\d+)?")


def to_uint6(b64):
    """Convert one Base64 character to an unsigned integer."""
    c = ord(b64)
    if 97 <= c <= 122:
        return c - 71
    elif 65 <= c <= 90:
        return c - 65
    elif 48 <= c <= 57:
        return c + 4
    elif c == 43:
        return 62
    elif c == 47:
        return 63
    else:
        raise ValueError(f"Invalid Base64 pitchbend character: {b64!r}")


def to_int12(b64):
    """Converts two Base64 characters to a signed 12-bit integer (-2047 to 2048)."""
    uint12 = to_uint6(b64[0]) << 6 | to_uint6(b64[1])
    if uint12 >> 11 & 1 == 1:
        return uint12 - 4096
    else:
        return uint12


def to_int12_stream(b64):
    """Converts a Base64 string to a list of integers."""
    res = []
    for i in range(0, len(b64) - 1, 2):
        res.append(to_int12(b64[i:i+2]))
    return res


def pitch_string_to_cents(x):
    """Converts UTAU's pitchbend argument to an ndarray representing the pitch offset in cents."""
    if not x:
        return np.zeros(2, dtype=np.int32)
    pitch = x.split('#')
    res = []
    for i in range(0, len(pitch), 2):
        p = pitch[i:i+2]
        if len(p) == 2:
            pitch_str, rle = p
            res.extend(to_int12_stream(pitch_str))
            if res and rle.isdigit():
                rle_cnt = min(int(rle), 131072)
                res.extend([res[-1]] * rle_cnt)
        else:
            res.extend(to_int12_stream(p[0]))
    if not res:
        return np.zeros(2, dtype=np.int32)
    res = np.array(res, dtype=np.int32)
    return np.concatenate([res, np.zeros(1)])


def note_to_midi(x):
    """Note name (or MIDI number string) to MIDI note number."""
    s = str(x).strip()
    if s.lstrip('+-').isdigit():
        return int(s)
    m = note_re.match(s)
    if not m:
        raise ValueError(f"Invalid MIDI note name: {x!r}")
    note, octave = m.group(1, 2)
    octave = int(octave) + 1
    return octave * 12 + notes[note]


def midi_to_hz(x):
    """MIDI note number to Hertz using equal temperament. A4 = 440 Hz."""
    return 440 * np.exp2((x - 69) / 12)
