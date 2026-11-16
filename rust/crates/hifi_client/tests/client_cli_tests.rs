use std::process::Command;
use uuid::Uuid;

fn client_bin() -> &'static str {
    env!("CARGO_BIN_EXE_hifisampler")
}

#[test]
fn cli_exits_1_with_no_args_or_fewer_than_4_args() {
    let out0 = Command::new(client_bin()).output().unwrap();
    assert_eq!(out0.status.code(), Some(1));
    let stderr0 = String::from_utf8_lossy(&out0.stderr);
    assert!(stderr0.contains("No arguments provided"));

    let out2 = Command::new(client_bin())
        .args(["arg1.wav", "arg2.wav"])
        .output()
        .unwrap();
    assert_eq!(out2.status.code(), Some(1));
}

#[test]
fn cli_rejects_invalid_output_extension_with_exit_1() {
    let out = Command::new(client_bin())
        .args(["in.wav", "bad_out.mp3", "C4", "100"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn cli_succeeds_against_ephemeral_loopback_http_server() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();

    let server_thread = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut raw = Vec::new();
        let mut tmp_buf = [0u8; 4096];
        loop {
            let n = stream.read(&mut tmp_buf).unwrap();
            if n == 0 {
                break;
            }
            raw.extend_from_slice(&tmp_buf[..n]);
            if let Some(hdr_end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                let hdr_str = String::from_utf8_lossy(&raw[..hdr_end]);
                let content_len: usize = hdr_str
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                if raw.len() >= hdr_end + 4 + content_len {
                    break;
                }
            }
        }
        let req = String::from_utf8_lossy(&raw).into_owned();
        assert!(req.starts_with("POST / HTTP/1.1"));
        assert!(req.contains("special (voice-1) [take2].wav"));

        let body = "Success: 'special' -> 'out'";
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(resp.as_bytes()).unwrap();
        stream.flush().unwrap();
    });

    let tmp = std::env::temp_dir().join(format!("hifi_cli_bin_{}", Uuid::new_v4().simple()));
    std::fs::create_dir_all(&tmp).unwrap();
    let in_path = tmp.join("special (voice-1) [take2].wav");
    let out_path = tmp.join("out (rendered) [final].wav");

    let out = Command::new(client_bin())
        .args([
            in_path.to_str().unwrap(),
            out_path.to_str().unwrap(),
            "A4",
            "100",
        ])
        .env("HIFISAMPLER_PORT", port.to_string())
        .output()
        .unwrap();

    assert_eq!(out.status.code(), Some(0));
    assert!(out.stdout.is_empty(), "fast-path should emit zero stdout");
    assert!(out.stderr.is_empty(), "fast-path should emit zero stderr");

    server_thread.join().unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
}
