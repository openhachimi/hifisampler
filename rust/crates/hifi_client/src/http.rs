use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub reason: String,
    pub body: String,
}

#[derive(Debug)]
pub enum HttpError {
    ConnectRefusedOrUnreachable(String),
    ConnectTimeout,
    ReadTimeout,
    Io(std::io::Error),
    Protocol(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::ConnectRefusedOrUnreachable(msg) => write!(f, "Connection failed: {msg}"),
            HttpError::ConnectTimeout => write!(f, "Connection timed out"),
            HttpError::ReadTimeout => write!(f, "Request timed out"),
            HttpError::Io(e) => write!(f, "I/O error: {e}"),
            HttpError::Protocol(msg) => write!(f, "HTTP protocol error: {msg}"),
        }
    }
}

/// Returns true when all args can safely be transmitted as space-separated `text/plain`.
pub fn can_send_as_plain_text(args: &[String]) -> bool {
    let Some(first) = args.first() else {
        return false;
    };
    if first.is_empty() || first.starts_with('{') || first.starts_with('[') {
        return false;
    }
    for raw in args {
        if raw.is_empty() {
            return false;
        }
        for c in raw.chars() {
            if c == '\r'
                || c == '\n'
                || c == '\t'
                || c == '"'
                || c == '\''
                || c.is_control()
                || c.is_whitespace()
            {
                return false;
            }
        }
    }
    true
}

/// Build HTTP request `(payload, media_type)` respecting `HIFISAMPLER_PROTOCOL`.
pub fn build_request_payload(args: &[String]) -> (String, &'static str) {
    let proto = std::env::var("HIFISAMPLER_PROTOCOL")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    build_request_payload_with_protocol(args, &proto)
}

pub fn build_request_payload_with_protocol(
    args: &[String],
    protocol: &str,
) -> (String, &'static str) {
    if (protocol == "text" || protocol == "auto") && can_send_as_plain_text(args) {
        return (args.join(" "), "text/plain");
    }
    let body = serde_json::json!({ "args": args }).to_string();
    (body, "application/json")
}

/// Send an HTTP/1.1 request directly over loopback TCP (`127.0.0.1:{port}`), bypassing all proxies.
pub fn http_request_loopback(
    port: u16,
    method: &str,
    path: &str,
    body: Option<(&str, &str)>,
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<HttpResponse, HttpError> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, connect_timeout).map_err(|e| {
        if e.kind() == std::io::ErrorKind::TimedOut {
            HttpError::ConnectTimeout
        } else {
            HttpError::ConnectRefusedOrUnreachable(e.to_string())
        }
    })?;

    let _ = stream.set_nodelay(true);
    stream
        .set_read_timeout(Some(read_timeout))
        .map_err(HttpError::Io)?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(HttpError::Io)?;

    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nUser-Agent: hifisampler-rs\r\nConnection: close\r\n"
    );
    if let Some((payload, media_type)) = body {
        req.push_str(&format!(
            "Content-Type: {media_type}; charset=utf-8\r\nContent-Length: {}\r\n\r\n{payload}",
            payload.len()
        ));
    } else {
        req.push_str("\r\n");
    }
    stream.write_all(req.as_bytes()).map_err(HttpError::Io)?;
    stream.flush().map_err(HttpError::Io)?;

    read_http_response(stream)
}

fn read_http_response<R: Read>(stream: R) -> Result<HttpResponse, HttpError> {
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    let n = reader.read_line(&mut status_line).map_err(map_read_error)?;
    if n == 0 {
        return Err(HttpError::Protocol("Empty HTTP response".to_string()));
    }
    let status_trimmed = status_line.trim_end_matches(&['\r', '\n'][..]);
    let mut parts = status_trimmed.splitn(3, ' ');
    let _http_ver = parts.next().unwrap_or("");
    let status: u16 = parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| HttpError::Protocol(format!("Malformed status line: {status_trimmed}")))?;
    let reason = parts.next().unwrap_or("").to_string();

    let mut content_length: Option<usize> = None;
    let mut is_chunked = false;

    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line).map_err(map_read_error)?;
        if read == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, val)) = trimmed.split_once(':') {
            let name_lower = name.trim().to_ascii_lowercase();
            let val_trimmed = val.trim();
            if name_lower == "content-length" {
                content_length = val_trimmed.parse::<usize>().ok();
            } else if name_lower == "transfer-encoding"
                && val_trimmed.to_ascii_lowercase().contains("chunked")
            {
                is_chunked = true;
            }
        }
    }

    let body_bytes = if is_chunked {
        read_chunked_body(&mut reader)?
    } else if let Some(len) = content_length {
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf).map_err(map_read_error)?;
        buf
    } else {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(map_read_error)?;
        buf
    };

    Ok(HttpResponse {
        status,
        reason,
        body: String::from_utf8_lossy(&body_bytes).into_owned(),
    })
}

fn read_chunked_body<R: BufRead>(reader: &mut R) -> Result<Vec<u8>, HttpError> {
    let mut out = Vec::new();
    loop {
        let mut size_line = String::new();
        if reader.read_line(&mut size_line).map_err(map_read_error)? == 0 {
            break;
        }
        let size_hex = size_line.trim().split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| HttpError::Protocol(format!("Invalid chunk size: {size_hex}")))?;
        if size == 0 {
            // Drain trailer lines
            loop {
                let mut trailer = String::new();
                if reader.read_line(&mut trailer).map_err(map_read_error)? == 0 {
                    break;
                }
                if trailer.trim().is_empty() {
                    break;
                }
            }
            break;
        }
        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk).map_err(map_read_error)?;
        out.extend_from_slice(&chunk);
        // Trailing CRLF after chunk data
        let mut crlf = [0u8; 2];
        reader.read_exact(&mut crlf).map_err(map_read_error)?;
    }
    Ok(out)
}

fn map_read_error(err: std::io::Error) -> HttpError {
    if err.kind() == std::io::ErrorKind::TimedOut || err.kind() == std::io::ErrorKind::WouldBlock {
        HttpError::ReadTimeout
    } else {
        HttpError::Io(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn payload_selection_matches_csharp_rules() {
        let simple = vec!["in.wav".into(), "out.wav".into(), "C4".into(), "100".into()];
        let (p_json, m_json) = build_request_payload_with_protocol(&simple, "");
        assert_eq!(m_json, "application/json");
        assert!(p_json.contains(r#""args""#));

        let (p_txt, m_txt) = build_request_payload_with_protocol(&simple, "text");
        assert_eq!(m_txt, "text/plain");
        assert_eq!(p_txt, "in.wav out.wav C4 100");

        let with_spaces = vec![
            "my file.wav".into(),
            "out.wav".into(),
            "C4".into(),
            "100".into(),
        ];
        let (_p_auto, m_auto) = build_request_payload_with_protocol(&with_spaces, "auto");
        assert_eq!(m_auto, "application/json");
    }

    #[test]
    fn parse_chunked_and_content_length_responses() {
        let raw_len = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\nServer Ready";
        let resp = read_http_response(Cursor::new(&raw_len[..])).unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "Server Ready");

        let raw_chunk =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nHello\r\n0\r\n\r\n";
        let resp2 = read_http_response(Cursor::new(&raw_chunk[..])).unwrap();
        assert_eq!(resp2.status, 200);
        assert_eq!(resp2.body, "Hello");
    }
}
