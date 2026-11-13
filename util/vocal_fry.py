"""
Clean-room TIPS + LF Glottal Vocal Fry ("气泡音" / Creaky Voice) synthesis
triggered automatically by sustained deep relative pitchbend ('pit') drops inside
a note's vowel body.

Core design principles:
1. Zero new flags required; zero impact on .hifi.npz feature caches.
2. O(1) early-exit guard: when min(pitchbend) >= FRY_ONSET_CENTS (-1250 cents),
   returns None immediately so normal notes, vibratos, and <=1-octave portamentos
   follow the 100% untouched bit-identical synthesis path.
3. Melodic-leap immunity (Two-Pass Stationary Basin Detector):
   - Excludes pre-consonant / VCV preutterance region [new_start .. t_vowel_start]
     so >1-octave legato upward leaps ('nian C3 -> xiang F4') never mis-trigger.
   - Suppresses high-velocity inter-note glides via a Slew-Rate Gate (|d(pit)/dt|).
   - Requires stationary deep-pit dwell inside the vowel body.
4. Continuous Single-Pass Neural Glide + Sliding-Phase 3-Band LF Bubble Shaper:
   - Once a fry basin is confirmed, glides F0 continuously across a wide [-1720c .. -560c]
     transition runway from true creaky bubble rates (27..36 Hz with band-limited
     stochastic period jitter) smoothly up into modal singing — zero subharmonic
     frequency jump and zero resynthesis seam!
   - Splits vocoder audio into 3 zero-phase complementary bands (sub <155 Hz, F1 body
     155..900 Hz, F2-F4 crisp >900 Hz), attenuating neural SineGen sub-bass hum while
     phase-locking crisp full-band LF closure envelopes onto every local glottal burst.
"""

from functools import lru_cache
import math
import threading
import numpy as np
import scipy.interpolate as interp
from scipy.signal import butter, sosfiltfilt

from util.parse_utau import midi_to_hz
from util.utils import slaney_mel_filterbank

# Strict detection thresholds (cents below current note):
# Normal vibrato (+/-150c), glides (-600c), and 1-octave portamentos (-1200c) stay 100% untouched.
FRY_ONSET_CENTS = -1250.0
FRY_FULL_CENTS = -1680.0

# Wide morphing transition bounds INSIDE confirmed vocal-fry basins so entry & exit
# between vocal fry and modal phonation have a silky-smooth, gradual transition!
BASIN_MORPH_START_CENTS = -560.0
BASIN_MORPH_FULL_CENTS = -1700.0

# Portamento slew-rate gate bounds (cents / second) for detection pass:
SLEW_PASS_CPS = 2400.0
SLEW_STOP_CPS = 5600.0

# Minimum contiguous stationary dwell durations (seconds) inside vowel region:
MIN_DWELL_INTERIOR_SEC = 0.085
MIN_DWELL_BOUNDARY_SEC = 0.115

_MEL_PROFILE_LOCK = threading.Lock()
_MEL_PROFILE_CACHE: dict[tuple[int, int, int, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


@lru_cache(maxsize=16)
def _get_crossover_sos(sr: int, cutoff_hz: float, btype: str = "high"):
    nyq = 0.5 * float(sr)
    norm_freq = float(np.clip(cutoff_hz / nyq, 0.003, 0.95))
    return butter(2, norm_freq, btype=btype, output="sos")


@lru_cache(maxsize=32)
def _get_hanning_kernel(win: int) -> np.ndarray:
    kern = np.hanning(win + 2)[1:-1].astype(np.float64)
    kern /= np.sum(kern)
    kern.setflags(write=False)
    return kern


def _get_mel_fry_profiles(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute and cache:
      - low_voiced_bins: boolean mask for voiced fundamental/F1 energy [80..1050 Hz]
      - high_fric_bins: boolean mask for unvoiced fricative energy [4200..14500 Hz]
      - lf_fry_mel_tilt: [n_mels] log-magnitude delta (Nepers) suppressing <165 Hz
        neural SineGen sub-hum while lifting 550..3600 Hz glottal closure crispness.
    """
    key = (int(sr), int(n_fft), int(n_mels), float(fmin), float(fmax))
    with _MEL_PROFILE_LOCK:
        cached = _MEL_PROFILE_CACHE.get(key)
        if cached is not None:
            return cached

        fb = slaney_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        fft_freqs = np.linspace(0.0, 0.5 * sr, n_fft // 2 + 1, dtype=np.float64)
        fb_sum = np.maximum(np.sum(fb, axis=1), 1e-12)
        mel_centers = (fb @ fft_freqs) / fb_sum

        low_voiced_bins = (mel_centers >= 80.0) & (mel_centers <= 1050.0)
        high_fric_bins = (mel_centers >= 4200.0) & (mel_centers <= 14500.0)

        f_hz = np.maximum(mel_centers, 35.0)
        # Mildly trim <165 Hz SineGen fundamental hum (-2.8 dB) and lift warm F1-F3 tap (+3.6 dB)
        low_cut_db = -2.8 / (1.0 + (f_hz / 165.0) ** 2)
        mid_snap_db = 3.6 * (((f_hz / 650.0) ** 2) / (1.0 + (f_hz / 650.0) ** 2)) / (
            1.0 + (f_hz / 5200.0) ** 2
        )
        lf_fry_mel_tilt = ((low_cut_db + mid_snap_db) * (math.log(10.0) / 20.0)).astype(np.float32)

        res = (low_voiced_bins, high_fric_bins, lf_fry_mel_tilt)
        _MEL_PROFILE_CACHE[key] = res
        return res


def _compute_voiced_mel_gate(
    mel_render: np.ndarray,
    sr: int,
    n_fft: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Smooth [0, 1] per-frame voicing gate protecting unvoiced consonants (s, sh, k, t) and silence."""
    n_mels, n_frames = mel_render.shape
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    low_bins, fric_bins, _ = _get_mel_fry_profiles(sr, n_fft, n_mels, fmin, fmax)

    if np.any(low_bins):
        low_slice = np.clip(mel_render[low_bins, :].astype(np.float64, copy=False), -12.0, 6.0)
        low_slice *= 2.0
        np.exp(low_slice, out=low_slice)
        e_low = np.sum(low_slice, axis=0)
    else:
        full_lin = np.clip(mel_render.astype(np.float64, copy=False), -12.0, 6.0)
        full_lin *= 2.0
        np.exp(full_lin, out=full_lin)
        e_low = np.sum(full_lin, axis=0)

    if np.any(fric_bins):
        fric_slice = np.clip(mel_render[fric_bins, :].astype(np.float64, copy=False), -12.0, 6.0)
        fric_slice *= 2.0
        np.exp(fric_slice, out=fric_slice)
        e_fric = np.sum(fric_slice, axis=0)
    else:
        e_fric = np.zeros(n_frames, dtype=np.float64)

    peak_low = float(np.max(e_low))
    if peak_low <= 1e-9:
        return np.zeros(n_frames, dtype=np.float64)

    rel_db = 10.0 * np.log10(np.maximum(e_low, 1e-18) / peak_low)
    gate_energy = np.clip((rel_db + 28.0) / 10.0, 0.0, 1.0)

    vh_db = 10.0 * np.log10(np.maximum(e_low, 1e-18) / np.maximum(e_fric, 1e-18))
    gate_vh = np.clip((vh_db + 4.5) / 8.0, 0.0, 1.0)

    raw_gate = gate_energy
    raw_gate *= gate_vh
    if n_frames >= 3:
        smoothed = 0.5 * raw_gate
        smoothed[1:] += 0.25 * raw_gate[:-1]
        smoothed[0] += 0.25 * raw_gate[0]
        smoothed[:-1] += 0.25 * raw_gate[1:]
        smoothed[-1] += 0.25 * raw_gate[-1]
        return smoothed
    return raw_gate


def _smoothstep01(u: np.ndarray) -> np.ndarray:
    u_c = np.clip(u, 0.0, 1.0)
    return (u_c * u_c) * (3.0 - 2.0 * u_c)


def _confirm_and_build_fry_basins(
    w_cand: np.ndarray,
    ramp_region: np.ndarray,
    voiced_gate: np.ndarray,
    rel_cents: np.ndarray,
    t_mel: np.ndarray,
    new_end: float,
    t_vowel_start: float,
    thop: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-Pass Detector & Wide Transition Basin Shaper:
      - Pass 1: Strictly validate stationary deep-pit islands (rel_cents < -1250c, low slew,
        sufficient dwell inside vowel body) so melodic leaps NEVER trigger.
      - Pass 2: Expand each confirmed island out to rel_cents < -180c and evaluate a wide,
        gradual morph envelope across [-1700c .. -560c] so the transition between vocal fry
        and normal singing is completely continuous and natural!
    """
    zeros = np.zeros_like(w_cand)
    n_frames = len(w_cand)
    if n_frames == 0:
        return zeros, zeros

    active = w_cand > 0.05
    if not np.any(active):
        return zeros, zeros

    padded = np.concatenate(([False], active, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)

    # Precompute contiguous runs of rel_cents < -180c once to avoid inner per-frame while loops.
    below_180 = rel_cents < -180.0
    pad_b = np.concatenate(([False], below_180, [False]))
    diff_b = np.diff(pad_b.astype(np.int8))
    run_starts = np.flatnonzero(diff_b == 1)
    run_ends = np.flatnonzero(diff_b == -1)

    basin_mask = np.zeros(n_frames, dtype=np.float64)
    left_entered_low = float(rel_cents[0]) <= -850.0
    vowel_start_idx = int(np.searchsorted(t_mel, t_vowel_start, side="left"))

    confirmed_any = False
    for i0, i1 in zip(starts, ends):
        i0_int = int(i0)
        i1_int = int(i1)
        dwell_sec = (i1_int - i0_int) * thop
        touches_right = t_mel[min(i1_int - 1, n_frames - 1)] >= new_end - max(2.0 * thop, 0.025)
        touches_left_vowel = i0_int <= vowel_start_idx + 2

        # 1. Left-boundary melodic upward leap check ('nian C3 -> xiang F4')
        if touches_left_vowel and left_entered_low:
            post_max = float(np.max(rel_cents[i1_int:])) if i1_int < n_frames else float(rel_cents[-1])
            island_min = float(np.min(rel_cents[i0_int:i1_int]))
            dipped_locally = island_min < (float(rel_cents[0]) - 180.0)
            if post_max >= -350.0 and not dipped_locally and dwell_sec < MIN_DWELL_BOUNDARY_SEC:
                continue

        # 2. Minimum stationary dwell check
        min_req = MIN_DWELL_BOUNDARY_SEC if touches_right else MIN_DWELL_INTERIOR_SEC
        if dwell_sec < min_req:
            continue

        confirmed_any = True
        min_lo = vowel_start_idx if left_entered_low else 0
        r = int(np.searchsorted(run_starts, i0_int, side="right")) - 1
        if r >= 0 and int(run_starts[r]) <= i0_int < int(run_ends[r]):
            b_lo = i0_int if i0_int <= min_lo else max(int(run_starts[r]), min_lo)
            b_hi = max(i1_int, int(run_ends[r]))
        else:
            b_lo, b_hi = i0_int, i1_int
        basin_mask[b_lo:b_hi] = 1.0

    if not confirmed_any:
        return zeros, zeros

    # Broad ~65 ms temporal smoothing on basin_mask & morph envelope so even steep hand-drawn
    # pit exits glide smoothly through intermediate pulse rates instead of snapping!
    win_b = max(3, int(round(0.065 / max(thop, 1e-4))) | 1)
    if n_frames >= win_b:
        kern_b = _get_hanning_kernel(win_b)
        m_basin = np.convolve(np.pad(basin_mask, win_b // 2, mode="edge"), kern_b, mode="valid")[:n_frames]
    else:
        kern_b = None
        m_basin = basin_mask.copy()

    if left_entered_low:
        m_basin = m_basin * ramp_region
    np.clip(m_basin, 0.0, 1.0, out=m_basin)

    # Wide continuous morph weight across [-1700c .. -560c] (1140-cent gradual transition runway!)
    morph_span = max(BASIN_MORPH_START_CENTS - BASIN_MORPH_FULL_CENTS, 1.0)
    w_wide = _smoothstep01((BASIN_MORPH_START_CENTS - rel_cents) / morph_span)
    w_raw = m_basin * w_wide
    w_raw *= ramp_region
    w_raw *= voiced_gate

    if n_frames >= win_b:
        w_out = np.convolve(np.pad(w_raw, win_b // 2, mode="edge"), kern_b, mode="valid")[:n_frames]
    else:
        w_out = w_raw
    w_out *= ramp_region
    np.clip(w_out, 0.0, 1.0, out=w_out)
    return w_out, m_basin


def _bandlimited_fry_jitter(n_frames: int, thop: float, seed: int) -> np.ndarray:
    """
    Generate smooth band-limited (~6..18 Hz) zero-mean unit-std stochastic process so
    each ~30 ms glottal bubble has natural physiological timing irregularity (+/-15%)
    instead of robotic 32 Hz periodicity.
    """
    if n_frames < 4:
        return np.zeros(n_frames, dtype=np.float64)
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n_frames)
    # Smooth over ~38 ms (~1 glottal fry period) so F0 stays C^1 smooth for HiFi-GAN SineGen
    # while varying randomly from bubble to bubble!
    win = max(3, int(round(0.038 / max(thop, 1e-4))) | 1)
    if n_frames >= win:
        kern = _get_hanning_kernel(win)
        sm = np.convolve(np.pad(raw, win // 2, mode="reflect"), kern, mode="valid")[:n_frames]
    else:
        sm = raw
    sm -= float(np.mean(sm))
    std = float(np.std(sm))
    if std > 1e-8:
        sm /= std
    return np.clip(sm, -2.2, 2.2)


def prepare_vocal_fry_excitation(
    pitchbend: np.ndarray,
    base_pitch_midi: float,
    t_flag_val: float,
    tempo: float,
    new_start: float,
    new_end: float,
    t_mel: np.ndarray,
    f0_render: np.ndarray,
    mel_render: np.ndarray,
    sr: int,
    n_fft: int,
    mel_fmin: float,
    mel_fmax: float,
    new_con: float | None = None,
) -> dict | None:
    """
    O(1)-guarded leap-safe detector & continuous single-pass F0/mel excitation shaper.
    Returns None whenever vocal fry is not triggered (100% bit-identical to normal synthesis).
    """
    if pitchbend is None or len(pitchbend) < 2:
        return None
    if t_mel is None or t_mel.size == 0 or mel_render is None or mel_render.ndim != 2 or mel_render.shape[1] == 0:
        return None
    if float(np.min(pitchbend)) >= FRY_ONSET_CENTS:
        return None

    n_frames = len(t_mel)
    thop = float(t_mel[1] - t_mel[0]) if n_frames >= 2 else 512.0 / max(float(sr), 1.0)

    pb_in = np.asarray(pitchbend, dtype=np.float64)
    fix_tail = len(pb_in) >= 2 and pb_in[-1] == 0.0 and pb_in[-2] <= FRY_ONSET_CENTS
    if fix_tail:
        pb = pb_in.copy()
        pb[-1] = pb[-2]
    else:
        pb = pb_in

    safe_tempo = max(float(tempo), 1e-3)
    dt_pitch = 60.0 / (safe_tempo * 96.0)
    u_pitch = (t_mel - float(new_start)) / dt_pitch
    np.clip(u_pitch, 0.0, float(len(pb) - 1), out=u_pitch)
    p_i0 = u_pitch.astype(np.intp)
    u_pitch -= p_i0
    p_i1 = np.minimum(p_i0 + 1, len(pb) - 1)
    rel_cents = pb[p_i0] + u_pitch * (pb[p_i1] - pb[p_i0])

    # Strict detection depth [-1250c .. -1680c]
    det_span = max(FRY_ONSET_CENTS - FRY_FULL_CENTS, 1.0)
    w_det_depth = _smoothstep01((FRY_ONSET_CENTS - rel_cents) / det_span)
    if float(np.max(w_det_depth)) <= 1e-4:
        return None

    # Pre-vowel / VCV preutterance exclusion [new_start .. t_vowel_start]
    note_span = max(float(new_end) - float(new_start), thop)
    raw_con_dur = max(float(new_con) - float(new_start), 0.0) if new_con is not None else 0.0
    min_left_guard = 0.030 if float(rel_cents[0]) <= -850.0 else 0.0
    guard_dur = min(max(raw_con_dur, min_left_guard), 0.55 * note_span)
    t_vowel_start = float(new_start) + guard_dur
    ramp_region = _smoothstep01((t_mel - t_vowel_start) / max(0.025, 2.0 * thop))

    # Slew-rate gate for detection pass (rejects >1-octave melodic leaps)
    if n_frames >= 3:
        sm_w = max(3, int(round(0.025 / max(thop, 1e-4))) | 1)
        if n_frames >= sm_w:
            k_sm = np.ones(sm_w, dtype=np.float64) / float(sm_w)
            rc_smooth = np.convolve(np.pad(rel_cents, sm_w // 2, mode="edge"), k_sm, mode="valid")[:n_frames]
        else:
            rc_smooth = rel_cents
        abs_slope_cps = np.abs(np.gradient(rc_smooth, max(thop, 1e-5)))
    else:
        abs_slope_cps = np.zeros(n_frames, dtype=np.float64)

    g_slew = _smoothstep01((SLEW_STOP_CPS - abs_slope_cps) / max(SLEW_STOP_CPS - SLEW_PASS_CPS, 1.0))
    voiced_gate = _compute_voiced_mel_gate(mel_render, sr=sr, n_fft=n_fft, fmin=mel_fmin, fmax=mel_fmax)

    w_cand = w_det_depth * ramp_region
    w_cand *= g_slew
    w_cand *= voiced_gate
    w_fry, m_basin = _confirm_and_build_fry_basins(
        w_cand=w_cand,
        ramp_region=ramp_region,
        voiced_gate=voiced_gate,
        rel_cents=rel_cents,
        t_mel=t_mel,
        new_end=float(new_end),
        t_vowel_start=t_vowel_start,
        thop=thop,
    )
    if float(np.max(w_fry)) <= 1e-3:
        return None

    if fix_tail:
        pitch_full = pb / 100.0 + float(base_pitch_midi) + float(t_flag_val) / 10.0
        if np.all(pitch_full == pitch_full[0]):
            f0_base = np.full(n_frames, float(midi_to_hz(pitch_full[0])), dtype=np.float64)
        else:
            t_pitch = dt_pitch * np.arange(len(pb), dtype=np.float64) + float(new_start)
            t_clamped = np.clip(t_mel, float(new_start), float(t_pitch[-1]))
            akima = interp.Akima1DInterpolator(t_pitch, pitch_full)
            f0_base = midi_to_hz(akima(t_clamped)).astype(np.float64, copy=False)
    else:
        f0_base = f0_render.astype(np.float64, copy=False)

    # Continuous Physiological Creak Acceleration Curve [27..135 Hz]:
    # Deep pit (-1900c) sits in authentic discrete vocal-fry bubble band (~28..35 Hz), then
    # accelerates smoothly through 45 -> 68 -> 98 -> 135 Hz across the upward glide so
    # pulses seamlessly compress into modal phonation on both onset scoops and tail drops!
    seed_val = (int(round(float(base_pitch_midi) * 100.0)) * 1000003 + len(pitchbend) * 9176) & 0xFFFFFFFF
    jit_proc = _bandlimited_fry_jitter(n_frames, thop, int(seed_val))

    f_fry_target = np.clip(34.5 * np.exp2((rel_cents - (-1750.0)) / 520.0), 27.0, 135.0)
    # Organic cycle-to-cycle rate irregularity (+/- 12% in deep <48 Hz fry, fading out as rate accelerates!)
    slow_fry_w = w_fry * _smoothstep01((62.0 - f_fry_target) / 26.0)
    jitter_mod = np.exp(0.115 * slow_fry_w * jit_proc)
    f_fry_inst = np.minimum(f0_base, f_fry_target) * jitter_mod

    # Continuous log-frequency glide: 30 Hz -> 50 Hz -> 85 Hz -> 150 Hz -> f0_base with ZERO jump!
    log_f0_eff = (1.0 - w_fry) * np.log(np.maximum(f0_base, 1.0)) + w_fry * np.log(np.maximum(f_fry_inst, 1.0))
    f0_eff = np.exp(log_f0_eff).astype(np.float32)

    _, _, lf_mel_tilt = _get_mel_fry_profiles(sr, n_fft, mel_render.shape[0], mel_fmin, mel_fmax)
    w_fry_32 = w_fry.astype(np.float32, copy=False)
    mel_mod = mel_render.astype(mel_render.dtype, copy=True)
    mel_mod += lf_mel_tilt[:, np.newaxis] * w_fry_32[np.newaxis, :]

    return {
        "mel_render": mel_mod,
        "f0_render": f0_eff,
        "w_fry": w_fry,
        "f0_eff": f0_eff.astype(np.float64),
        "t_mel": t_mel,
        "new_start": float(new_start),
        "new_end": float(new_end),
        "seed": int(seed_val),
    }


def _pulse_rms_integral(tau_att: float, tau_dec: float) -> float:
    """Analytical integral int_0^inf ((1 - exp(-t/tau_a)) * exp(-t/tau_d))^2 dt."""
    r1 = 2.0 / tau_dec
    r2 = (1.0 / tau_att) + (2.0 / tau_dec)
    r3 = (2.0 / tau_att) + (2.0 / tau_dec)
    val = (1.0 / r1) - (2.0 / r2) + (1.0 / r3)
    return max(val, 1e-8)


def _interp_control_curves(
    t_mel: np.ndarray,
    new_start: float,
    new_end: float,
    n_samples: int,
    w_fry: np.ndarray,
    f0_eff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Interpolate w_fry and f0_eff to sample rate, using O(N) uniform indexing when t_mel is uniform."""
    w_arr = np.asarray(w_fry, dtype=np.float64)
    f0_arr = np.asarray(f0_eff, dtype=np.float64)
    n_mel = len(t_mel)

    if n_mel == 1:
        w0 = float(w_arr[0])
        if w0 <= 1e-4:
            return None, None
        w_s = np.full(n_samples, w0, dtype=np.float64)
        f0_s = np.full(n_samples, float(np.clip(f0_arr[0], 20.0, 1200.0)), dtype=np.float64)
        return w_s, f0_s

    if n_mel >= 2:
        t0 = float(t_mel[0])
        dt_mel = (float(t_mel[-1]) - t0) / (n_mel - 1)
        if dt_mel > 0.0 and float(np.max(np.abs(np.diff(t_mel) - dt_mel))) <= 1e-9 * max(dt_mel, 1.0):
            u = np.linspace(
                (float(new_start) - t0) / dt_mel,
                (float(new_end) - t0) / dt_mel,
                num=n_samples,
                endpoint=False,
                dtype=np.float64,
            )
            np.clip(u, 0.0, float(n_mel - 1), out=u)
            i0 = u.astype(np.intp)
            u -= i0
            i1 = np.minimum(i0 + 1, n_mel - 1)
            w_s = w_arr[i0]
            w_s += u * (w_arr[i1] - w_s)
            if float(np.max(w_s)) <= 1e-4:
                return None, None
            f0_s = f0_arr[i0]
            f0_s += u * (f0_arr[i1] - f0_s)
            np.clip(f0_s, 20.0, 1200.0, out=f0_s)
            return w_s, f0_s

    audio_t = np.linspace(new_start, new_end, num=n_samples, endpoint=False, dtype=np.float64)
    w_s = np.interp(audio_t, t_mel, w_arr, left=w_arr[0], right=w_arr[-1])
    if float(np.max(w_s)) <= 1e-4:
        return None, None
    f0_s = np.interp(audio_t, t_mel, f0_arr, left=f0_arr[0], right=f0_arr[-1])
    np.clip(f0_s, 20.0, 1200.0, out=f0_s)
    return w_s, f0_s


def _compute_block_epoch_phasors(
    w_s: np.ndarray,
    mid_band: np.ndarray,
    high_band: np.ndarray,
    phase: np.ndarray,
    step: int = 1024,
) -> tuple[np.ndarray, complex]:
    """Compute block-averaged complex epoch phasors without full-length complex128/energy allocations."""
    n_samples = len(phase)
    n_blk = max(1, int(math.ceil(n_samples / step)))
    c_blk = np.zeros(n_blk, dtype=np.complex128)

    blocks_per_chunk = 16
    chunk_len = min(n_samples, blocks_per_chunk * step)
    e_buf = np.empty(chunk_len, dtype=np.float64)
    tmp_buf = np.empty(chunk_len, dtype=np.float64)
    theta_buf = np.empty(chunk_len, dtype=np.float64)
    cos_buf = np.empty(chunk_len, dtype=np.float64)
    sin_buf = np.empty(chunk_len, dtype=np.float64)

    inv_step = 1.0 / float(step)
    two_pi_neg = -2.0 * np.pi

    for b0 in range(0, n_blk, blocks_per_chunk):
        b1 = min(b0 + blocks_per_chunk, n_blk)
        s0 = b0 * step
        s1 = min(b1 * step, n_samples)
        m = s1 - s0

        np.square(mid_band[s0:s1], out=e_buf[:m])
        e_buf[:m] *= 0.35
        np.square(high_band[s0:s1], out=tmp_buf[:m])
        tmp_buf[:m] *= 0.65
        e_buf[:m] += tmp_buf[:m]
        e_buf[:m] *= w_s[s0:s1]

        np.remainder(phase[s0:s1], 1.0, out=theta_buf[:m])
        theta_buf[:m] *= two_pi_neg
        np.cos(theta_buf[:m], out=cos_buf[:m])
        np.sin(theta_buf[:m], out=sin_buf[:m])
        cos_buf[:m] *= e_buf[:m]
        sin_buf[:m] *= e_buf[:m]

        nb = b1 - b0
        full_m = nb * step
        if m == full_m:
            re_sum = np.sum(cos_buf[:m].reshape(nb, step), axis=1)
            im_sum = np.sum(sin_buf[:m].reshape(nb, step), axis=1)
            c_blk[b0:b1] = (re_sum + 1j * im_sum) * inv_step
        else:
            full_blocks = m // step
            if full_blocks > 0:
                fm = full_blocks * step
                re_sum = np.sum(cos_buf[:fm].reshape(full_blocks, step), axis=1)
                im_sum = np.sum(sin_buf[:fm].reshape(full_blocks, step), axis=1)
                c_blk[b0:b0 + full_blocks] = (re_sum + 1j * im_sum) * inv_step
            tail_s = full_blocks * step
            if tail_s < m:
                c_blk[b0 + full_blocks] = complex(
                    float(np.sum(cos_buf[tail_s:m])),
                    float(np.sum(sin_buf[tail_s:m])),
                ) * inv_step

    z_glob = complex(np.sum(c_blk)) * float(step)
    return c_blk, z_glob


def _eval_lf_env_chunk(
    tau: np.ndarray,
    tau_prev: np.ndarray,
    amp_cur: np.ndarray,
    amp_prev: np.ndarray,
    inv_tau_a: float,
    inv_tau_d: float,
    buf_a: np.ndarray,
    buf_b: np.ndarray,
    out: np.ndarray,
) -> None:
    """Evaluate two-pulse LF envelope in-place using preallocated chunk buffers."""
    np.multiply(tau, -inv_tau_a, out=buf_a)
    np.exp(buf_a, out=buf_a)
    np.subtract(1.0, buf_a, out=buf_a)
    np.multiply(tau, -inv_tau_d, out=buf_b)
    np.exp(buf_b, out=buf_b)
    buf_a *= buf_b
    buf_a *= amp_cur
    out[:] = buf_a

    np.multiply(tau_prev, -inv_tau_a, out=buf_a)
    np.exp(buf_a, out=buf_a)
    np.subtract(1.0, buf_a, out=buf_a)
    np.multiply(tau_prev, -inv_tau_d, out=buf_b)
    np.exp(buf_b, out=buf_b)
    buf_a *= buf_b
    buf_a *= amp_prev
    out += buf_a


def apply_lf_vocal_fry(render: np.ndarray, fry_ctx: dict | None, sr: int) -> np.ndarray:
    """
    Post-vocoder sliding-phase-locked 3-band LF glottal closure shaper:
      - Band 1 (<155 Hz): attenuates neural SineGen sub-bass fundamental sine hum
      - Band 2 (155..900 Hz, F1 chest/pharynx): warm damped vocal-cord bubble (tau_d = 3.6 ms)
      - Band 3 (>900 Hz, F2-F4 & crisp closure): sharp glottal closure pop (tau_d = 1.85 ms)
    As F0 accelerates out of fry into modal singing, adjacent pulse tails naturally overlap
    and w_s -> 0, yielding a 100% continuous, seamless transition!
    """
    if fry_ctx is None or len(render) < 32:
        return render

    t_mel = fry_ctx["t_mel"]
    w_fry = fry_ctx["w_fry"]
    f0_eff = fry_ctx.get("f0_eff", fry_ctx.get("f0_carrier"))
    new_start = fry_ctx["new_start"]
    new_end = fry_ctx["new_end"]
    seed = int(fry_ctx.get("seed", 20261111))

    n_samples = len(render)
    w_s, f0_s = _interp_control_curves(t_mel, new_start, new_end, n_samples, w_fry, f0_eff)
    if w_s is None:
        return render

    # Exact 3-band zero-phase complementary crossover: sub_band + mid_band + high_band == x64 identically!
    x64 = render.astype(np.float64, copy=False)
    if n_samples > 64:
        sos_hp_mid = _get_crossover_sos(int(sr), 155.0, "high")
        sos_hp_hi = _get_crossover_sos(int(sr), 900.0, "high")
        above_sub = sosfiltfilt(sos_hp_mid, x64)
        sub_band = x64 - above_sub
        high_band = sosfiltfilt(sos_hp_hi, above_sub)
        above_sub -= high_band
        mid_band = above_sub
    else:
        sub_band = 0.2 * x64
        mid_band = 0.4 * x64
        high_band = 0.4 * x64

    # Slew-limited sliding complex Fourier epoch tracker: locks onto HiFi-GAN's local glottal
    # bursts while bounding |d(phi_peak)/dn| <= 0.18 * d(phase)/dn so instantaneous phase
    # velocity stays >= 82% of f0_s everywhere (zero stalled/dropped pulses!)
    period_s = 1.0 / f0_s
    dphase = f0_s / float(sr)
    phase = np.cumsum(dphase, dtype=np.float64)

    step = 1024
    c_blk, z_glob = _compute_block_epoch_phasors(w_s, mid_band, high_band, phase, step=step)

    chunk_sz = min(n_samples, 8192)
    if abs(z_glob) > 1e-16:
        z_fb = z_glob / abs(z_glob)
        n_blk = len(c_blk)
        win_l = min(11, max(3, (n_blk // 2) * 2 + 1))
        kern = _get_hanning_kernel(win_l)
        c_re = np.convolve(np.pad(c_blk.real, win_l // 2, mode="edge"), kern, mode="valid")[:n_blk]
        c_im = np.convolve(np.pad(c_blk.imag, win_l // 2, mode="edge"), kern, mode="valid")[:n_blk]
        c_sm = (c_re + 1j * c_im) + 0.08 * float(np.max(np.abs(c_blk)) + 1e-12) * z_fb
        phi_blk = np.unwrap(-np.angle(c_sm)) / (2.0 * np.pi)
        blk_t = (np.arange(n_blk, dtype=np.float64) + 0.5) * step

        # Integrate slew-limited phi_peak in chunks and overwrite `phase` in-place as `aligned`.
        base_idx = np.arange(chunk_sz, dtype=np.float64)
        idx_buf = np.empty(chunk_sz, dtype=np.float64)
        max_d_buf = np.empty(chunk_sz, dtype=np.float64)
        neg_max_buf = np.empty(chunk_sz, dtype=np.float64)
        dphi_buf = np.empty(chunk_sz, dtype=np.float64)
        phi_peak_buf = np.empty(chunk_sz, dtype=np.float64)

        prev_phi_raw = 0.0
        running_phi_peak = 0.0
        for s0 in range(0, n_samples, chunk_sz):
            s1 = min(s0 + chunk_sz, n_samples)
            m = s1 - s0
            np.add(base_idx[:m], float(s0), out=idx_buf[:m])
            phi_raw_c = np.interp(idx_buf[:m], blk_t, phi_blk)
            if s0 == 0:
                prev_phi_raw = float(phi_raw_c[0])
                running_phi_peak = prev_phi_raw

            dphi_buf[0] = float(phi_raw_c[0]) - prev_phi_raw
            if m > 1:
                np.subtract(phi_raw_c[1:], phi_raw_c[:-1], out=dphi_buf[1:m])
            prev_phi_raw = float(phi_raw_c[-1])

            np.multiply(dphase[s0:s1], 0.18, out=max_d_buf[:m])
            np.negative(max_d_buf[:m], out=neg_max_buf[:m])
            np.clip(dphi_buf[:m], neg_max_buf[:m], max_d_buf[:m], out=dphi_buf[:m])
            np.cumsum(dphi_buf[:m], out=phi_peak_buf[:m])
            phi_peak_buf[:m] += running_phi_peak
            running_phi_peak = float(phi_peak_buf[m - 1])

            phase[s0:s1] -= phi_peak_buf[:m]
            phase[s0:s1] += 0.050
    else:
        phase += 0.050

    aligned = phase

    # Stochastic + dichrotic per-bubble amplitude shimmer (strong/weak glottal alternation)
    c_min = int(math.floor(float(np.min(aligned)))) - 2
    c_max = int(math.floor(float(np.max(aligned)))) + 2
    n_cyc = max(4, c_max - c_min + 1)
    rng = np.random.default_rng(seed ^ 0x5A3C91)
    shim_table = rng.standard_normal(n_cyc)
    cyc_grid = np.arange(c_min, c_min + n_cyc, dtype=np.int64)
    parity = (cyc_grid & 1).astype(np.float64)
    amp_table = np.clip(1.0 - 0.34 * parity + 0.11 * shim_table, 0.45, 1.18)

    # Band-specific physiological glottal-bubble damping time constants:
    # - Mid band (155..900 Hz, F1 body): tau_att = 0.75 ms, tau_dec = 4.4 ms (full woody throat body)
    # - High band (>900 Hz, F2-F4 snap): tau_att = 0.40 ms, tau_dec = 2.35 ms (crisp bubble pop)
    tau_a_mid, tau_d_mid = 0.00075, 0.0044
    tau_a_hi, tau_d_hi = 0.00040, 0.00235
    inv_tau_a_mid, inv_tau_d_mid = 1.0 / tau_a_mid, 1.0 / tau_d_mid
    inv_tau_a_hi, inv_tau_d_hi = 1.0 / tau_a_hi, 1.0 / tau_d_hi
    amp_rms = 0.86
    int_mid = _pulse_rms_integral(tau_a_mid, tau_d_mid)
    int_hi = _pulse_rms_integral(tau_a_hi, tau_d_hi)

    orig_peak = max(float(np.max(x64)), -float(np.min(x64)))
    ceil_peak = max(0.96 * orig_peak, 0.12)
    inv_ceil = 1.0 / ceil_peak

    # Preallocated L1/L2-resident chunk buffers for envelope & 3-band blend
    buf_cyc_f = np.empty(chunk_sz, dtype=np.float64)
    buf_tau = np.empty(chunk_sz, dtype=np.float64)
    buf_tau_p = np.empty(chunk_sz, dtype=np.float64)
    buf_idx = np.empty(chunk_sz, dtype=np.int64)
    buf_a = np.empty(chunk_sz, dtype=np.float64)
    buf_b = np.empty(chunk_sz, dtype=np.float64)
    buf_env_m = np.empty(chunk_sz, dtype=np.float64)
    buf_env_h = np.empty(chunk_sz, dtype=np.float64)
    buf_rms = np.empty(chunk_sz, dtype=np.float64)
    buf_gain = np.empty(chunk_sz, dtype=np.float64)
    buf_fry = np.empty(chunk_sz, dtype=np.float64)

    out = np.empty(n_samples, dtype=render.dtype)
    for s0 in range(0, n_samples, chunk_sz):
        s1 = min(s0 + chunk_sz, n_samples)
        m = s1 - s0
        al_c = aligned[s0:s1]
        per_c = period_s[s0:s1]

        np.floor(al_c, out=buf_cyc_f[:m])
        cyc_i = buf_cyc_f[:m].astype(np.int64)
        np.subtract(al_c, buf_cyc_f[:m], out=buf_tau[:m])
        buf_tau[:m] *= per_c
        np.add(buf_tau[:m], per_c, out=buf_tau_p[:m])

        np.subtract(cyc_i, c_min, out=buf_idx[:m])
        np.clip(buf_idx[:m], 0, n_cyc - 1, out=buf_idx[:m])
        amp_cur = amp_table[buf_idx[:m]]

        np.subtract(cyc_i, 1 + c_min, out=buf_idx[:m])
        np.clip(buf_idx[:m], 0, n_cyc - 1, out=buf_idx[:m])
        amp_prev = amp_table[buf_idx[:m]]

        _eval_lf_env_chunk(
            buf_tau[:m], buf_tau_p[:m], amp_cur, amp_prev,
            inv_tau_a_mid, inv_tau_d_mid, buf_a[:m], buf_b[:m], buf_env_m[:m],
        )
        _eval_lf_env_chunk(
            buf_tau[:m], buf_tau_p[:m], amp_cur, amp_prev,
            inv_tau_a_hi, inv_tau_d_hi, buf_a[:m], buf_b[:m], buf_env_h[:m],
        )

        # Mid & sub bands share rms_mid
        np.divide(int_mid, per_c, out=buf_rms[:m])
        np.sqrt(buf_rms[:m], out=buf_rms[:m])
        buf_rms[:m] *= amp_rms
        np.maximum(buf_rms[:m], 1e-4, out=buf_rms[:m])

        # Trim <155 Hz SineGen fundamental hum while keeping natural chest warmth
        np.divide(0.85, buf_rms[:m], out=buf_gain[:m])
        np.minimum(buf_gain[:m], 1.9, out=buf_gain[:m])
        buf_gain[:m] *= buf_env_m[:m]
        buf_gain[:m] *= 0.42
        np.multiply(sub_band[s0:s1], buf_gain[:m], out=buf_fry[:m])

        np.divide(1.28, buf_rms[:m], out=buf_gain[:m])
        np.minimum(buf_gain[:m], 3.10, out=buf_gain[:m])
        buf_gain[:m] *= buf_env_m[:m]
        buf_gain[:m] *= mid_band[s0:s1]
        buf_fry[:m] += buf_gain[:m]

        # High band (>900 Hz)
        np.divide(int_hi, per_c, out=buf_rms[:m])
        np.sqrt(buf_rms[:m], out=buf_rms[:m])
        buf_rms[:m] *= amp_rms
        np.maximum(buf_rms[:m], 1e-4, out=buf_rms[:m])

        np.divide(1.52, buf_rms[:m], out=buf_gain[:m])
        np.minimum(buf_gain[:m], 3.65, out=buf_gain[:m])
        buf_gain[:m] *= buf_env_h[:m]
        buf_gain[:m] *= high_band[s0:s1]
        buf_fry[:m] += buf_gain[:m]

        # Soft limiter + wet/dry blend: x64 + w_s * (fry_wave - x64)
        buf_fry[:m] *= inv_ceil
        np.tanh(buf_fry[:m], out=buf_fry[:m])
        buf_fry[:m] *= ceil_peak
        buf_fry[:m] -= x64[s0:s1]
        buf_fry[:m] *= w_s[s0:s1]
        buf_fry[:m] += x64[s0:s1]
        out[s0:s1] = buf_fry[:m]

    return out
