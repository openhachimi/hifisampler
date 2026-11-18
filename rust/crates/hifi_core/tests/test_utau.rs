use approx::assert_relative_eq;
use hifi_core::utau::{
    find_flags, midi_slice_to_hz, midi_to_hz, note_to_midi, parse_flags, parse_flags_map,
    parse_tempo, pitch_string_to_cents, to_int12, to_int12_stream, to_uint6, UtauError, ALL_FLAGS,
    MAX_PITCH_RLE_COUNT,
};
use std::path::PathBuf;
use std::process::Command;

fn python_exe() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in [
        "../../../.venv/Scripts/python.exe",
        "../../../.venv/bin/python",
    ] {
        let cand = manifest_dir.join(rel);
        if cand.exists() {
            return cand.canonicalize().unwrap_or(cand);
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.join("../../..");
    root.canonicalize().unwrap_or(root)
}

#[test]
fn test_to_uint6_valid_and_invalid() {
    assert_eq!(to_uint6('A').unwrap(), 0);
    assert_eq!(to_uint6('B').unwrap(), 1);
    assert_eq!(to_uint6('Z').unwrap(), 25);
    assert_eq!(to_uint6('a').unwrap(), 26);
    assert_eq!(to_uint6('b').unwrap(), 27);
    assert_eq!(to_uint6('z').unwrap(), 51);
    assert_eq!(to_uint6('0').unwrap(), 52);
    assert_eq!(to_uint6('1').unwrap(), 53);
    assert_eq!(to_uint6('9').unwrap(), 61);
    assert_eq!(to_uint6('+').unwrap(), 62);
    assert_eq!(to_uint6('/').unwrap(), 63);

    for bad in [
        ' ', '!', '#', '$', '%', '&', '*', '-', ',', '\n', '\0', '=', '@', ':', ';', '<', '>', '?',
        '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~',
    ] {
        assert!(matches!(
            to_uint6(bad),
            Err(UtauError::InvalidBase64Char(c)) if c == bad
        ));
    }
}

#[test]
fn test_to_int12_and_stream() {
    assert_eq!(to_int12("AA").unwrap(), 0);
    assert_eq!(to_int12("AB").unwrap(), 1);
    assert_eq!(to_int12("A/").unwrap(), 63);
    assert_eq!(to_int12("BA").unwrap(), 64);
    assert_eq!(to_int12("f/").unwrap(), 2047);
    assert_eq!(to_int12("//").unwrap(), -1);
    assert_eq!(to_int12("/+").unwrap(), -2);
    assert_eq!(to_int12("gA").unwrap(), -2048);

    assert_eq!(to_int12_stream("").unwrap(), Vec::<i32>::new());
    assert_eq!(to_int12_stream("AA").unwrap(), vec![0]);
    assert_eq!(to_int12_stream("AAAB").unwrap(), vec![0, 1]);
    assert_eq!(to_int12_stream("AAAB//").unwrap(), vec![0, 1, -1]);
    // Odd trailing char ignored
    assert_eq!(to_int12_stream("AAABX").unwrap(), vec![0, 1]);
}

#[test]
fn test_pitch_string_to_cents_rle_and_dos_cap() {
    assert_eq!(pitch_string_to_cents("").unwrap(), vec![0, 0]);
    assert_eq!(pitch_string_to_cents("AA").unwrap(), vec![0, 0]);
    assert_eq!(pitch_string_to_cents("AAAA").unwrap(), vec![0, 0, 0]);
    assert_eq!(pitch_string_to_cents("AA#5").unwrap(), vec![0; 7]);
    assert_eq!(pitch_string_to_cents("AB#3").unwrap(), vec![1, 1, 1, 1, 0]);
    assert_eq!(
        pitch_string_to_cents("AA#2#AB#3").unwrap(),
        vec![0, 0, 0, 1, 1, 1, 1, 0]
    );
    assert_eq!(pitch_string_to_cents("AA#abc").unwrap(), vec![0, 0]);
    assert_eq!(pitch_string_to_cents("AA#0").unwrap(), vec![0, 0]);
    assert_eq!(pitch_string_to_cents("#5").unwrap(), vec![0, 0]);

    let dos = pitch_string_to_cents("AB#999999999").unwrap();
    assert_eq!(dos.len(), 1 + MAX_PITCH_RLE_COUNT + 1);
    assert_eq!(dos[0], 1);
    assert_eq!(dos[MAX_PITCH_RLE_COUNT], 1);
    assert_eq!(dos[MAX_PITCH_RLE_COUNT + 1], 0);
}

#[test]
fn test_note_to_midi_and_midi_to_hz() {
    let cases = [
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
        ("Fb4", 64),
        ("E#4", 65),
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
        ("Cb4", 71),
        ("B#3", 48),
        ("C5", 72),
        ("G9", 127),
        ("60", 60),
        (" A4 ", 69),
    ];
    for (name, expected) in cases {
        assert_eq!(note_to_midi(name).unwrap(), expected, "note={name}");
    }

    for invalid in ["H4", "C", "", "C#", "b4", "!A4", "XYZ1", "A4b", "C4_extra"] {
        assert!(
            matches!(note_to_midi(invalid), Err(UtauError::InvalidMidiNote(_))),
            "expected error for {invalid}"
        );
    }

    assert_relative_eq!(midi_to_hz(69.0), 440.0, epsilon = 1e-6);
    assert_relative_eq!(midi_to_hz(57.0), 220.0, epsilon = 1e-6);
    assert_relative_eq!(midi_to_hz(81.0), 880.0, epsilon = 1e-6);
    assert_relative_eq!(midi_to_hz(60.0), 261.6255653, epsilon = 1e-5);
    assert!(midi_to_hz(0.0) > 8.0 && midi_to_hz(0.0) < 8.5);
    assert!(midi_to_hz(127.0) > 12500.0);

    let hz_vec = midi_slice_to_hz(&[57.0, 69.0, 81.0]);
    assert_relative_eq!(hz_vec[0], 220.0, epsilon = 1e-4);
    assert_relative_eq!(hz_vec[1], 440.0, epsilon = 1e-4);
    assert_relative_eq!(hz_vec[2], 880.0, epsilon = 1e-4);
}

#[test]
fn test_parse_tempo() {
    assert_relative_eq!(parse_tempo("!120").unwrap(), 120.0, epsilon = 1e-9);
    assert_relative_eq!(parse_tempo("!60.5").unwrap(), 60.5, epsilon = 1e-9);
    assert_relative_eq!(parse_tempo("140").unwrap(), 140.0, epsilon = 1e-9);
    assert_relative_eq!(parse_tempo("!0").unwrap(), 1.0, epsilon = 1e-9);
    assert_relative_eq!(parse_tempo("-50").unwrap(), 1.0, epsilon = 1e-9);
    assert!(parse_tempo("not_a_tempo").is_err());
    assert!(parse_tempo("!nan").is_err());
}

#[test]
fn test_all_26_flags_and_longest_match_parity_with_python() {
    assert_eq!(ALL_FLAGS.len(), 26);

    let raw = "/fe1/fl2/fo3/fv4/fp5/ve6/vo7/g-5/t+10/A40/B20/G/P90/S1/p-2/R3/D4/C5/Z6/Hv80/Hb120/Ht-20/Ho+35/Hr-25/He1/HG30";
    let parsed_map = parse_flags_map(raw);
    assert_eq!(parsed_map.len(), 26);
    assert_eq!(parsed_map.get("fe"), Some(&Some(1)));
    assert_eq!(parsed_map.get("fl"), Some(&Some(2)));
    assert_eq!(parsed_map.get("fo"), Some(&Some(3)));
    assert_eq!(parsed_map.get("fv"), Some(&Some(4)));
    assert_eq!(parsed_map.get("fp"), Some(&Some(5)));
    assert_eq!(parsed_map.get("ve"), Some(&Some(6)));
    assert_eq!(parsed_map.get("vo"), Some(&Some(7)));
    assert_eq!(parsed_map.get("g"), Some(&Some(-5)));
    assert_eq!(parsed_map.get("t"), Some(&Some(10)));
    assert_eq!(parsed_map.get("A"), Some(&Some(40)));
    assert_eq!(parsed_map.get("B"), Some(&Some(20)));
    assert_eq!(parsed_map.get("G"), Some(&None));
    assert_eq!(parsed_map.get("P"), Some(&Some(90)));
    assert_eq!(parsed_map.get("S"), Some(&Some(1)));
    assert_eq!(parsed_map.get("p"), Some(&Some(-2)));
    assert_eq!(parsed_map.get("R"), Some(&Some(3)));
    assert_eq!(parsed_map.get("D"), Some(&Some(4)));
    assert_eq!(parsed_map.get("C"), Some(&Some(5)));
    assert_eq!(parsed_map.get("Z"), Some(&Some(6)));
    assert_eq!(parsed_map.get("Hv"), Some(&Some(80)));
    assert_eq!(parsed_map.get("Hb"), Some(&Some(120)));
    assert_eq!(parsed_map.get("Ht"), Some(&Some(-20)));
    assert_eq!(parsed_map.get("Ho"), Some(&Some(35)));
    assert_eq!(parsed_map.get("Hr"), Some(&Some(-25)));
    assert_eq!(parsed_map.get("He"), Some(&Some(1)));
    assert_eq!(parsed_map.get("HG"), Some(&Some(30)));

    // Longest-match priority: HG50 is HG=50, not G; Hb120 is Hb=120
    assert_eq!(find_flags("HG50"), vec![("HG".to_string(), Some(50))]);
    assert_eq!(find_flags("Hb120"), vec![("Hb".to_string(), Some(120))]);
    // First-match-wins per key
    assert_eq!(parse_flags("g-5g15").gender(), -5);

    // Valueless flags & defaults
    let default_flags = parse_flags("");
    assert_eq!(default_flags.hb(), 100);
    assert_eq!(default_flags.hv(), 100);
    assert_eq!(default_flags.ht(), 0);
    assert_eq!(default_flags.ho(), 0);
    assert_eq!(default_flags.hr(), 0);
    assert_eq!(default_flags.gender(), 0);
    assert!(!default_flags.force_generate());
    assert!(!default_flags.he_loop());
    assert_eq!(default_flags.p_strength(), None);
    assert_eq!(
        default_flags.feature_cache_filename("note"),
        "note_Hb100_Ht0_Hv100_g0.hifi.bin"
    );

    let custom_flags = parse_flags("G0He0AP");
    assert!(!custom_flags.force_generate());
    assert!(!custom_flags.he_loop());
    assert_eq!(custom_flags.a_mod(), 0);
    assert_eq!(custom_flags.p_strength(), Some(100));

    let active = parse_flags("GHt50Ho40Hr-30Hb999Hv-5g150P0");
    assert!(active.force_generate());
    assert_eq!(active.hb(), 500);
    assert_eq!(active.hv(), 0);
    assert_eq!(active.ht(), 50);
    assert_eq!(active.ho(), 40);
    assert_eq!(active.hr(), -30);
    assert_eq!(active.gender(), 100);
    assert_eq!(active.p_strength(), Some(0));
    assert_eq!(
        active.feature_cache_filename("sample"),
        "sample_Hb500_Ho40_Hr-30_Ht50v3_Hv0_g100.hifi.bin"
    );

    // Verify exact parity with Python util.parse_utau.flag_re on tricky inputs
    let tricky = "G-g+5t-10Hbfp20P";
    let rust_pairs: Vec<(String, String)> = find_flags(tricky)
        .into_iter()
        .map(|(k, v)| (k, v.map_or_else(String::new, |n| n.to_string())))
        .collect();
    let py_code = format!(
        "import sys, json; sys.path.insert(0, r'{}'); \
         from util.parse_utau import flag_re; \
         m = flag_re.findall({:?}.replace('/', '')); \
         print(json.dumps([[k, str(int(v)) if v else ''] for k, v in m]))",
        repo_root().display(),
        tricky
    );
    let out = Command::new(python_exe())
        .args(["-c", &py_code])
        .output()
        .unwrap();
    assert!(out.status.success());
    let expected_json = String::from_utf8(out.stdout).unwrap();
    let expected_formatted = format!(
        "[{}]",
        rust_pairs
            .iter()
            .map(|(k, v)| format!("[\"{k}\", \"{v}\"]"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    assert_eq!(expected_json.trim(), expected_formatted);
}
