import math
import numpy as np
import pytest

from util.parse_utau import (
    flags,
    flag_re,
    midi_to_hz,
    note_re,
    note_to_midi,
    pitch_string_to_cents,
    to_int12,
    to_int12_stream,
    to_uint6,
)


class TestBase64Decoding:
    """Test Base64 character and 12-bit signed integer conversion."""

    def test_to_uint6_valid_ranges(self):
        # A-Z -> 0..25
        assert to_uint6('A') == 0
        assert to_uint6('B') == 1
        assert to_uint6('Z') == 25

        # a-z -> 26..51
        assert to_uint6('a') == 26
        assert to_uint6('b') == 27
        assert to_uint6('z') == 51

        # 0-9 -> 52..61
        assert to_uint6('0') == 52
        assert to_uint6('1') == 53
        assert to_uint6('9') == 61

        # Special chars
        assert to_uint6('+') == 62
        assert to_uint6('/') == 63

    @pytest.mark.parametrize("invalid_char", [' ', '!', '#', '$', '%', '&', '*', '-', ',', '\n', '\0'])
    def test_to_uint6_invalid_characters_raise_value_error(self, invalid_char):
        with pytest.raises(ValueError, match="Invalid Base64 pitchbend character"):
            to_uint6(invalid_char)

    @pytest.mark.parametrize("out_of_bounds_char", ['=', '@', ':', ';', '<', '>', '?', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'])
    def test_to_uint6_strict_bounds_reject_out_of_range(self, out_of_bounds_char):
        with pytest.raises(ValueError, match="Invalid Base64 pitchbend character"):
            to_uint6(out_of_bounds_char)

    def test_to_int12_positive_and_zero(self):
        assert to_int12('AA') == 0
        assert to_int12('AB') == 1
        assert to_int12('A/') == 63
        assert to_int12('BA') == 64
        # Max 11-bit positive is 2047: 'f/' -> (31 << 6) | 63 = 2047
        assert to_int12('f/') == 2047

    def test_to_int12_negative_numbers(self):
        # '//' -> 4095 - 4096 = -1
        assert to_int12('//') == -1
        # '/+' -> 4094 - 4096 = -2
        assert to_int12('/+') == -2
        # Min 12-bit signed is -2048: 'gA' -> (32 << 6) | 0 = 2048 - 4096 = -2048
        assert to_int12('gA') == -2048

    def test_to_int12_stream(self):
        assert to_int12_stream("") == []
        assert to_int12_stream("AA") == [0]
        assert to_int12_stream("AAAB") == [0, 1]
        assert to_int12_stream("AAAB//") == [0, 1, -1]
        # Odd-length string ignores the last trailing odd byte
        assert to_int12_stream("AAABX") == [0, 1]


class TestPitchStringToCents:
    """Test UTAU pitchbend string to pitch offset in cents."""

    def test_empty_or_none(self):
        empty_res = pitch_string_to_cents("")
        assert isinstance(empty_res, np.ndarray)
        assert np.array_equal(empty_res, np.zeros(2, dtype=np.int32))

        none_res = pitch_string_to_cents(None)
        assert np.array_equal(none_res, np.zeros(2, dtype=np.int32))

    def test_simple_pitch_stream(self):
        # 'AA' -> [0] + trailing [0] = [0, 0]
        cents = pitch_string_to_cents("AA")
        assert np.array_equal(cents, np.array([0, 0], dtype=np.int32))

    def test_run_length_encoded_pitch(self):
        # 'AA#5' -> [0] followed by 5 repeats of 0, plus trailing zero -> length 7
        cents = pitch_string_to_cents("AA#5")
        assert len(cents) == 1 + 5 + 1
        assert np.all(cents == 0)

        # 'AB#3' -> [1] followed by 3 repeats of 1, plus trailing zero -> [1, 1, 1, 1, 0]
        cents = pitch_string_to_cents("AB#3")
        assert np.array_equal(cents, np.array([1, 1, 1, 1, 0], dtype=np.int32))

    def test_multiple_rle_segments(self):
        # 'AA#2#AB#3'
        cents = pitch_string_to_cents("AA#2#AB#3")
        # Segment 1: AA + 2 repeats = [0, 0, 0]
        # Segment 2: AB + 3 repeats = [1, 1, 1, 1]
        # Trailing zero: [0]
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int32)
        assert np.array_equal(cents, expected)

    def test_malformed_rle_graceful_handling(self):
        # Missing RLE count or non-digits
        cents = pitch_string_to_cents("AA#abc")
        # RLE abc is not isdigit(), so only AA is added + trailing zero
        assert np.array_equal(cents, np.array([0, 0], dtype=np.int32))

    def test_rle_dos_cap(self):
        cents = pitch_string_to_cents("AB#999999999")
        assert len(cents) == 1 + 131072 + 1


class TestMidiAndHzConversion:
    """Test note name, MIDI number, and Hertz conversions."""

    @pytest.mark.parametrize(
        "note_name,expected_midi",
        [
            ("C-1", 0),
            ("B-1", 11),
            ("C0", 12),
            ("C4", 60),
            ("C#4", 61),
            ("Db4", 61),
            ("D4", 62),
            ("D#4", 63),
            ("Eb4", 63),
            ("E4", 64),
            ("F4", 65),
            ("F#4", 66),
            ("Gb4", 66),
            ("G4", 67),
            ("G#4", 68),
            ("Ab4", 68),
            ("A4", 69),
            ("A#4", 70),
            ("Bb4", 70),
            ("B4", 71),
            ("C5", 72),
            ("G9", 127),
            ("60", 60),
            (" A4 ", 69),  # Leading/trailing whitespace
        ],
    )
    def test_note_to_midi_valid(self, note_name, expected_midi):
        assert note_to_midi(note_name) == expected_midi

    @pytest.mark.parametrize("invalid_note", ["H4", "C", "", "C#", "b4", "!A4", "XYZ1", "A4b", "C4_extra"])
    def test_note_to_midi_invalid_raises_value_error(self, invalid_note):
        with pytest.raises(ValueError, match="Invalid MIDI note name"):
            note_to_midi(invalid_note)

    def test_midi_to_hz_standards(self):
        # A4 = 440.0 Hz
        assert math.isclose(midi_to_hz(69), 440.0, rel_tol=1e-6)
        # A3 = 220.0 Hz
        assert math.isclose(midi_to_hz(57), 220.0, rel_tol=1e-6)
        # A5 = 880.0 Hz
        assert math.isclose(midi_to_hz(81), 880.0, rel_tol=1e-6)
        # C4 ≈ 261.625565 Hz
        assert math.isclose(midi_to_hz(60), 261.6255653, rel_tol=1e-5)

    def test_midi_to_hz_numpy_array(self):
        midi_arr = np.array([57, 69, 81], dtype=np.float32)
        hz_arr = midi_to_hz(midi_arr)
        assert isinstance(hz_arr, np.ndarray)
        assert np.allclose(hz_arr, [220.0, 440.0, 880.0], rtol=1e-5)


class TestUtauFlagsRegex:
    """Test UTAU expression flags matching and extraction."""

    def test_flag_definitions_exist(self):
        assert "g" in flags
        assert "t" in flags
        assert "Hb" in flags
        assert "Hv" in flags
        assert "Ht" in flags
        assert "He" in flags
        assert "HG" in flags
        assert "P" in flags

    def test_single_flag_matching(self):
        matches = flag_re.findall("g-5")
        assert matches == [("g", "-5")]

        matches_pos = flag_re.findall("t+10")
        assert matches_pos == [("t", "+10")]

        matches_no_val = flag_re.findall("G")
        assert matches_no_val == [("G", "")]

    def test_compound_flag_matching(self):
        flag_str = "g-5t10Hb100Hv80Ht-20HG30P90"
        matches = dict(flag_re.findall(flag_str))
        assert matches["g"] == "-5"
        assert matches["t"] == "10"
        assert matches["Hb"] == "100"
        assert matches["Hv"] == "80"
        assert matches["Ht"] == "-20"
        assert matches["HG"] == "30"
        assert matches["P"] == "90"

    def test_slash_separated_flags(self):
        # OpenUtau often sends /g-5/t10
        raw = "/g-5/t10/P90"
        sanitized = raw.replace("/", "")
        matches = dict(flag_re.findall(sanitized))
        assert matches == {"g": "-5", "t": "10", "P": "90"}

    def test_multi_char_priority(self):
        # 'HG' vs 'H' or 'G': 'HG' should match as a single flag, not 'H' and 'G'
        matches = flag_re.findall("HG50")
        assert matches == [("HG", "50")]

        # 'Hb' should match 'Hb', not 'H' + 'b'
        matches_hb = flag_re.findall("Hb120")
        assert matches_hb == [("Hb", "120")]
