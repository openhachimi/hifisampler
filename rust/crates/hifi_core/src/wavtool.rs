//! UTAU Wavtool CLI argument, duration (`ticks@tempo+corr`), and 5-point envelope parsing.

/// Parsed UTAU wavtool amplitude envelope (`p1 p2 p3 v1 v2 v3 v4 [ovr] [p4] [p5 v5]`).
#[derive(Debug, Clone, PartialEq)]
pub struct WavtoolEnvelope {
    pub p1_ms: f64,
    pub p2_ms: f64,
    pub p3_ms: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
    pub ovr_ms: f64,
    pub p4_ms: f64,
    pub p5_v5: Option<(f64, f64)>,
}

impl Default for WavtoolEnvelope {
    fn default() -> Self {
        Self {
            p1_ms: 0.0,
            p2_ms: 0.0,
            p3_ms: 0.0,
            v1: 100.0,
            v2: 100.0,
            v3: 100.0,
            v4: 100.0,
            ovr_ms: 0.0,
            p4_ms: 0.0,
            p5_v5: None,
        }
    }
}

impl WavtoolEnvelope {
    /// Parses UTAU wavtool envelope tokens (`0`, `2`, or `7..` tokens).
    pub fn parse(tokens: &[String]) -> Result<Self, String> {
        let parse_f = |s: &str, name: &str| -> Result<f64, String> {
            let v: f64 = s
                .trim()
                .parse()
                .map_err(|_| format!("Invalid wavtool envelope {name}: '{s}'"))?;
            if !v.is_finite() {
                return Err(format!("Non-finite wavtool envelope {name}: '{s}'"));
            }
            Ok(v)
        };

        match tokens.len() {
            0 => Ok(Self::default()),
            2 => Ok(Self {
                p1_ms: parse_f(&tokens[0], "p1")?,
                p2_ms: parse_f(&tokens[1], "p2")?,
                ..Self::default()
            }),
            n if n >= 7 => {
                let p1_ms = parse_f(&tokens[0], "p1")?;
                let p2_ms = parse_f(&tokens[1], "p2")?;
                let p3_ms = parse_f(&tokens[2], "p3")?;
                let v1 = parse_f(&tokens[3], "v1")?;
                let v2 = parse_f(&tokens[4], "v2")?;
                let v3 = parse_f(&tokens[5], "v3")?;
                let v4 = parse_f(&tokens[6], "v4")?;
                let ovr_ms = if n >= 8 {
                    parse_f(&tokens[7], "ovr")?
                } else {
                    0.0
                };
                let p4_ms = if n >= 9 {
                    parse_f(&tokens[8], "p4")?
                } else {
                    0.0
                };
                if n == 10 {
                    let _ = parse_f(&tokens[9], "p5")?;
                }
                let p5_v5 = if n >= 11 {
                    Some((parse_f(&tokens[9], "p5")?, parse_f(&tokens[10], "v5")?))
                } else {
                    None
                };
                Ok(Self {
                    p1_ms,
                    p2_ms,
                    p3_ms,
                    v1,
                    v2,
                    v3,
                    v4,
                    ovr_ms,
                    p4_ms,
                    p5_v5,
                })
            }
            other => Err(format!(
                "Invalid wavtool envelope token count: expected 0, 2, or >=7, got {other}"
            )),
        }
    }

    /// Computes `(x, y)` gain breakpoints over `[0, dur_ms]`, matching OpenUtau `EnvelopeMsToSamples`.
    fn resolved_points(&self, dur_ms: f64) -> ([f64; 5], [f64; 5], usize) {
        if !dur_ms.is_finite() || dur_ms <= 0.0 {
            return ([0.0; 5], [0.0; 5], 0);
        }
        let x0 = self.p1_ms.clamp(0.0, dur_ms);
        let x4 = (dur_ms - self.p4_ms).clamp(x0, dur_ms);
        let span = (x4 - x0).max(0.0);

        let attack = self.p2_ms.max(0.0);
        let release = self.p3_ms.max(0.0);
        let (x1, x3) = if attack + release > span && (attack + release) > 0.0 {
            let scale = span / (attack + release);
            (x0 + attack * scale, x4 - release * scale)
        } else {
            (x0 + attack, x4 - release)
        };

        let y0 = (self.v1 / 100.0).max(0.0);
        let y1 = (self.v2 / 100.0).max(0.0);
        let y3 = (self.v3 / 100.0).max(0.0);
        let y4 = (self.v4 / 100.0).max(0.0);

        if let Some((p5_ms, v5)) = self.p5_v5 {
            let x2 = (x1 + p5_ms.max(0.0)).clamp(x1, x3.max(x1));
            let y2 = (v5 / 100.0).max(0.0);
            ([x0, x1, x2, x3, x4], [y0, y1, y2, y3, y4], 5)
        } else {
            ([x0, x1, x3, x4, x4], [y0, y1, y3, y4, y4], 4)
        }
    }

    /// Evaluates the linear UTAU amplitude envelope at `pos_ms` within `[0, dur_ms]`.
    pub fn evaluate_gain(&self, pos_ms: f64, dur_ms: f64) -> f64 {
        if !dur_ms.is_finite()
            || dur_ms <= 0.0
            || !pos_ms.is_finite()
            || pos_ms < 0.0
            || pos_ms > dur_ms
        {
            return 0.0;
        }
        let (xs, ys, count) = self.resolved_points(dur_ms);
        interp_breakpoints(&xs[..count], &ys[..count], pos_ms).max(0.0)
    }

    /// Evaluates Mel/F0 blending weight at `pos_ms` within `[0, dur_ms]`.
    ///
    /// On phrase outer edges (`is_head` attack `[0, x1]` and `is_tail` release `[x3, x4]`),
    /// sustains the plateau level (`v2/100` and `v3/100`) so HiFi-GAN does not receive
    /// starved near-zero Mel frames while outer sample fades are applied post-vocoder.
    pub fn evaluate_mel_blend_weight(
        &self,
        pos_ms: f64,
        dur_ms: f64,
        is_head: bool,
        is_tail: bool,
    ) -> f64 {
        if !dur_ms.is_finite()
            || dur_ms <= 0.0
            || !pos_ms.is_finite()
            || pos_ms < 0.0
            || pos_ms > dur_ms
        {
            return 0.0;
        }
        let (xs, mut ys, count) = self.resolved_points(dur_ms);
        if count == 0 {
            return 0.0;
        }
        let tail_idx = count - 1;
        if is_head {
            ys[0] = ys[1];
        }
        if is_tail && tail_idx >= 1 {
            ys[tail_idx] = ys[tail_idx - 1];
        }
        interp_breakpoints(&xs[..count], &ys[..count], pos_ms).max(0.0)
    }
}

#[inline]
fn interp_breakpoints(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 || x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    for i in 0..(n - 1) {
        if x <= xs[i + 1] {
            let span = xs[i + 1] - xs[i];
            if span <= 1e-12 {
                return ys[i + 1];
            }
            let frac = (x - xs[i]) / span;
            return ys[i] + frac * (ys[i + 1] - ys[i]);
        }
    }
    ys[n - 1]
}

/// Parses an UTAU wavtool duration argument in milliseconds:
/// - `<ticks>@<tempo>[+|-<corr_ms>]` (with optional `!` before tempo)
/// - plain `<ms>` float (`$direct` notes and `stp` arguments)
pub fn parse_wavtool_duration_ms(raw: &str) -> Result<f64, String> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("Empty wavtool duration argument".to_string());
    }

    if let Some((ticks_raw, rest_raw)) = s.split_once('@') {
        let ticks: f64 = ticks_raw
            .trim()
            .parse()
            .map_err(|_| format!("Invalid wavtool tick count in '{s}'"))?;
        if !ticks.is_finite() {
            return Err(format!("Non-finite wavtool tick count in '{s}'"));
        }

        let rest = rest_raw.trim().strip_prefix('!').unwrap_or(rest_raw.trim());
        if rest.is_empty() {
            return Err(format!("Missing tempo after '@' in '{s}'"));
        }

        let bytes = rest.as_bytes();
        let mut sep_idx = None;
        for i in 1..bytes.len() {
            if (bytes[i] == b'+' || bytes[i] == b'-')
                && bytes[i - 1] != b'e'
                && bytes[i - 1] != b'E'
            {
                sep_idx = Some(i);
                break;
            }
        }

        let (tempo_raw, corr) = match sep_idx {
            Some(idx) => {
                let (t_part, c_part) = rest.split_at(idx);
                let c_val: f64 = c_part
                    .parse()
                    .map_err(|_| format!("Invalid wavtool duration correction in '{s}'"))?;
                if !c_val.is_finite() {
                    return Err(format!("Non-finite wavtool correction in '{s}'"));
                }
                (t_part, c_val)
            }
            None => (rest, 0.0),
        };

        let tempo: f64 = tempo_raw
            .parse()
            .map_err(|_| format!("Invalid wavtool tempo in '{s}'"))?;
        if !tempo.is_finite() {
            return Err(format!("Non-finite wavtool tempo in '{s}'"));
        }

        let ms = ticks * 125.0 / tempo.max(1.0) + corr;
        if !ms.is_finite() {
            return Err(format!("Non-finite wavtool duration in '{s}'"));
        }
        return Ok(ms);
    }

    let ms: f64 = s
        .parse()
        .map_err(|_| format!("Invalid wavtool duration '{s}'"))?;
    if !ms.is_finite() {
        return Err(format!("Non-finite wavtool duration '{s}'"));
    }
    Ok(ms)
}

/// Parsed CLI invocation for `melos-wavtool`.
#[derive(Debug, Clone, PartialEq)]
pub struct WavtoolCliArgs {
    pub output_file: String,
    pub input_file: String,
    pub stp_ms: f64,
    pub dur_ms: f64,
    pub envelope: WavtoolEnvelope,
    pub env_tokens: Vec<String>,
    pub commit: bool,
    pub batch_id: Option<u64>,
    pub expected_steps: Option<usize>,
}

/// Parses `melos-wavtool` CLI arguments (`<output> <input> <stp> <dur> [env...]`, plus optional
/// `--commit`, `--batch-id <u64>`, `--expected-steps <usize>` flags).
pub fn parse_wavtool_cli_args(raw_args: &[String]) -> Result<WavtoolCliArgs, String> {
    let mut positional: Vec<String> = Vec::with_capacity(raw_args.len());
    let mut commit = false;
    let mut batch_id: Option<u64> = None;
    let mut expected_steps: Option<usize> = None;

    let mut i = 0;
    while i < raw_args.len() {
        let arg = raw_args[i].as_str();
        if arg == "--commit" {
            commit = true;
            i += 1;
        } else if let Some(val) = arg.strip_prefix("--batch-id=") {
            let parsed = val
                .trim()
                .parse::<u64>()
                .map_err(|_| format!("Invalid --batch-id value: '{val}'"))?;
            batch_id = Some(parsed);
            i += 1;
        } else if arg == "--batch-id" {
            let next = raw_args
                .get(i + 1)
                .ok_or_else(|| "Missing value for --batch-id".to_string())?;
            let parsed = next
                .trim()
                .parse::<u64>()
                .map_err(|_| format!("Invalid --batch-id value: '{next}'"))?;
            batch_id = Some(parsed);
            i += 2;
        } else if let Some(val) = arg.strip_prefix("--expected-steps=") {
            let parsed = val
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid --expected-steps value: '{val}'"))?;
            expected_steps = Some(parsed);
            i += 1;
        } else if arg == "--expected-steps" {
            let next = raw_args
                .get(i + 1)
                .ok_or_else(|| "Missing value for --expected-steps".to_string())?;
            let parsed = next
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("Invalid --expected-steps value: '{next}'"))?;
            expected_steps = Some(parsed);
            i += 2;
        } else {
            positional.push(raw_args[i].clone());
            i += 1;
        }
    }

    if positional.len() < 4 {
        return Err(format!(
            "Expected at least 4 positional wavtool arguments (output, input, stp, dur), got {}",
            positional.len()
        ));
    }

    let stp_ms = parse_wavtool_duration_ms(&positional[2])?;
    let dur_ms = parse_wavtool_duration_ms(&positional[3])?;
    let env_tokens = positional[4..].to_vec();
    let envelope = WavtoolEnvelope::parse(&env_tokens)?;

    Ok(WavtoolCliArgs {
        output_file: positional[0].clone(),
        input_file: positional[1].clone(),
        stp_ms,
        dur_ms,
        envelope,
        env_tokens,
        commit,
        batch_id,
        expected_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn duration_parsing_handles_ticks_tempo_corr_and_direct_ms() {
        // 480 ticks @ 120 BPM = 500 ms
        assert_abs_diff_eq!(
            parse_wavtool_duration_ms("480@120").unwrap(),
            500.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            parse_wavtool_duration_ms("480@!120+25.5").unwrap(),
            525.5,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            parse_wavtool_duration_ms("480@120-15.25").unwrap(),
            484.75,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            parse_wavtool_duration_ms("240@120+1.5e-2").unwrap(),
            250.015,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            parse_wavtool_duration_ms("350.5").unwrap(),
            350.5,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(parse_wavtool_duration_ms("0").unwrap(), 0.0, epsilon = 1e-9);

        assert!(parse_wavtool_duration_ms("").is_err());
        assert!(parse_wavtool_duration_ms("NaN").is_err());
        assert!(parse_wavtool_duration_ms("480@inf").is_err());
    }

    #[test]
    fn envelope_parses_zero_two_seven_and_eleven_tokens() {
        let e0 = WavtoolEnvelope::parse(&[]).unwrap();
        assert_abs_diff_eq!(e0.evaluate_gain(120.0, 400.0), 1.0, epsilon = 1e-9);

        let e2 = WavtoolEnvelope::parse(&["0".into(), "10".into()]).unwrap();
        assert_abs_diff_eq!(e2.p2_ms, 10.0, epsilon = 1e-9);
        assert_abs_diff_eq!(e2.evaluate_gain(50.0, 400.0), 1.0, epsilon = 1e-9);

        let e4: Vec<String> = ["0", "5", "35", "0", "100", "100", "0"]
            .into_iter()
            .map(String::from)
            .collect();
        let env4 = WavtoolEnvelope::parse(&e4).unwrap();
        assert_abs_diff_eq!(env4.evaluate_gain(0.0, 500.0), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(env4.evaluate_gain(2.5, 500.0), 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(env4.evaluate_gain(250.0, 500.0), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(env4.evaluate_gain(482.5, 500.0), 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(env4.evaluate_gain(500.0, 500.0), 0.0, epsilon = 1e-9);

        let e5: Vec<String> = [
            "0", "10", "40", "0", "100", "80", "0", "30", "0", "90", "50",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let env5 = WavtoolEnvelope::parse(&e5).unwrap();
        assert_abs_diff_eq!(env5.ovr_ms, 30.0, epsilon = 1e-9);
        assert_abs_diff_eq!(env5.evaluate_gain(100.0, 500.0), 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(env5.evaluate_gain(480.0, 500.0), 0.4, epsilon = 1e-9);

        assert!(WavtoolEnvelope::parse(&["0".into(), "5".into(), "10".into()]).is_err());
    }

    #[test]
    fn crossfade_midpoint_unity_and_outer_edge_mel_blend_weights() {
        let e1 = WavtoolEnvelope {
            p1_ms: 0.0,
            p2_ms: 5.0,
            p3_ms: 50.0,
            v1: 0.0,
            v2: 100.0,
            v3: 100.0,
            v4: 0.0,
            ovr_ms: 0.0,
            p4_ms: 0.0,
            p5_v5: None,
        };
        let e2 = WavtoolEnvelope {
            p1_ms: 0.0,
            p2_ms: 50.0,
            p3_ms: 35.0,
            v1: 0.0,
            v2: 100.0,
            v3: 100.0,
            v4: 0.0,
            ovr_ms: 50.0,
            p4_ms: 0.0,
            p5_v5: None,
        };

        // At midpoint of 50 ms overlap: note 1 at 475/500, note 2 at 25/500
        let g1 = e1.evaluate_gain(475.0, 500.0);
        let g2 = e2.evaluate_gain(25.0, 500.0);
        assert_abs_diff_eq!(g1 + g2, 1.0, epsilon = 1e-9);

        // Phrase head attack holds v2=1.0 for Mel blend while sample gain ramps 0 -> 1
        assert_abs_diff_eq!(
            e2.evaluate_mel_blend_weight(0.0, 500.0, true, false),
            1.0,
            epsilon = 1e-9
        );
        // Phrase tail release holds v3=1.0 for Mel blend while sample gain fades 1 -> 0
        assert_abs_diff_eq!(
            e1.evaluate_mel_blend_weight(500.0, 500.0, false, true),
            1.0,
            epsilon = 1e-9
        );
        // Internal boundary follows envelope ramp
        assert_abs_diff_eq!(
            e2.evaluate_mel_blend_weight(25.0, 500.0, false, false),
            0.5,
            epsilon = 1e-9
        );
    }

    #[test]
    fn cli_args_parse_flags_and_positionals() {
        let raw: Vec<String> = [
            "out.wav",
            "in.wav",
            "12.5",
            "480@120-5",
            "0",
            "5",
            "35",
            "0",
            "100",
            "100",
            "0",
            "20",
            "0",
            "100",
            "100",
            "--batch-id",
            "42",
            "--expected-steps=3",
            "--commit",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let parsed = parse_wavtool_cli_args(&raw).unwrap();
        assert_eq!(parsed.output_file, "out.wav");
        assert_eq!(parsed.input_file, "in.wav");
        assert_abs_diff_eq!(parsed.stp_ms, 12.5, epsilon = 1e-9);
        assert_abs_diff_eq!(parsed.dur_ms, 495.0, epsilon = 1e-9);
        assert_eq!(parsed.batch_id, Some(42));
        assert_eq!(parsed.expected_steps, Some(3));
        assert!(parsed.commit);
        assert_abs_diff_eq!(parsed.envelope.ovr_ms, 20.0, epsilon = 1e-9);
    }

    #[test]
    fn non_finite_dur_and_pos_return_zero_without_panic() {
        let env = WavtoolEnvelope::default();
        for bad_dur in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -5.0] {
            assert_eq!(env.evaluate_gain(10.0, bad_dur), 0.0);
            assert_eq!(
                env.evaluate_mel_blend_weight(10.0, bad_dur, true, false),
                0.0
            );
        }
        for bad_pos in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 600.0] {
            assert_eq!(env.evaluate_gain(bad_pos, 500.0), 0.0);
            assert_eq!(
                env.evaluate_mel_blend_weight(bad_pos, 500.0, false, true),
                0.0
            );
        }
    }
}
