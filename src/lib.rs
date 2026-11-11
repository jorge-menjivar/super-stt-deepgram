// SPDX-License-Identifier: GPL-3.0-only
//! Deepgram speech-to-text backend as a `wasi:http` proxy component.
//!
//! Implements the Super Voice `/v1` backend contract
//! (`docs/protocol/backend/contract.md`) by exporting
//! `wasi:http/incoming-handler` and dispatching on method + path. The
//! component is stateless: the daemon injects the API key as the
//! `x-voice-secret-deepgram_api_key` request header and the model as
//! `x-voice-model`, and the component forwards audio to the Deepgram
//! transcription API over `wasi:http/outgoing-handler`.
//!
//! The `wasi:http` handler compiles only for `wasm32`; the pure helpers
//! (`header`, `encode_wav`, `parse_base`, `parse_transcript`) build for the
//! host too, so they are unit-tested natively while the component as a whole is
//! exercised by the wasmtime harness in `tests/`.

// Casts are intentional in audio/WAV encoding; doc lint trips on brand names.
#![allow(clippy::cast_possible_truncation, clippy::doc_markdown)]

use std::fmt::Write as _;

// ── pure helpers (host-testable) ────────────────────────────────────────────
// `pub` so the non-test host build that backs the integration harness does not
// flag them as dead code (the wasm handler that calls them is cfg'd out there).

/// Case-insensitive header lookup over `Fields::entries()`.
///
/// `want` is the Super Voice name, `x-voice-…`. A request without it is looked
/// up under its Super STT name, `x-stt-…`: a Super Voice daemon sends every
/// header under both, and a Super STT daemon only under the old one.
#[must_use]
pub fn header(entries: &[(String, Vec<u8>)], want: &str) -> Option<String> {
    let find = |name: &str| {
        entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| String::from_utf8_lossy(v).into_owned())
    };
    find(want).or_else(|| {
        let rest = want.strip_prefix("x-voice-")?;
        find(&format!("x-stt-{rest}"))
    })
}

/// The most vocabulary terms sent as keyterms, the number Deepgram's keyterm
/// guide names.
pub const MAX_KEYTERMS: usize = 100;

/// The most bytes of term text sent as keyterms.
///
/// Deepgram's limit is 500 tokens across all keyterms, counted by its own
/// tokenizer, which a client cannot run. Names and rare words come close to
/// two bytes a token, so 1000 bytes keeps even a vocabulary of them inside the
/// limit; ordinary words take fewer tokens, so most vocabularies are well
/// inside. The protocol asks a backend to truncate rather than let the
/// request fail, and this is that truncation: a vocabulary past the budget
/// loses its last terms, not all of them.
pub const MAX_KEYTERM_BYTES: usize = 1000;

/// The user's vocabulary as Deepgram keyterms.
///
/// `raw_vocabulary` is `x-voice-vocabulary`, a JSON array of strings that the
/// daemon escapes to pure ASCII, so `serde_json` is the whole decode. A
/// keyterm may be a phrase, so each term keeps its words together, with its
/// whitespace collapsed; empty and repeated terms are dropped. The list stops
/// at the first term past [`MAX_KEYTERMS`] or [`MAX_KEYTERM_BYTES`], keeping
/// the user's order. A header that will not parse gives no keyterms rather
/// than failing the request: losing the vocabulary should not lose the
/// recording.
#[must_use]
pub fn keyterms(raw_vocabulary: Option<&str>) -> Vec<String> {
    let terms: Vec<String> = raw_vocabulary
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default();
    let mut out: Vec<String> = Vec::new();
    let mut bytes = 0;
    for term in &terms {
        let term = term.split_whitespace().collect::<Vec<_>>().join(" ");
        if term.is_empty() || out.contains(&term) {
            continue;
        }
        if out.len() == MAX_KEYTERMS || bytes + term.len() > MAX_KEYTERM_BYTES {
            break;
        }
        bytes += term.len();
        out.push(term);
    }
    out
}

/// Percent-encode a query value: every byte but RFC 3986's unreserved
/// characters, so a space is `%20` and a non-ASCII letter its UTF-8 bytes.
#[must_use]
pub fn query_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// The `/v1/listen` path and query for one request.
///
/// The daemon's reserved `auto` language means "detect"; Deepgram's
/// equivalent is its multilingual code-switching mode, `multi`. Other tags,
/// region variants like `es-419` among them, pass through. Each keyterm is a
/// `keyterm` parameter of its own: Deepgram reads a comma inside one as part
/// of the term.
#[must_use]
pub fn listen_path(model: &str, language: Option<&str>, keyterms: &[String]) -> String {
    let mut path = format!("/v1/listen?model={}&smart_format=true", query_escape(model));
    if let Some(lang) = language {
        let lang = if lang == "auto" { "multi" } else { lang };
        let _ = write!(path, "&language={}", query_escape(lang));
    }
    for term in keyterms {
        let _ = write!(path, "&keyterm={}", query_escape(term));
    }
    path
}

/// Split a base URL into `(is_https, authority)` where authority is
/// `host[:port]`. A bare host (no scheme) is treated as HTTPS.
#[must_use]
pub fn parse_base(base: &str) -> (bool, String) {
    if let Some(rest) = base.strip_prefix("https://") {
        (true, rest.trim_end_matches('/').to_string())
    } else if let Some(rest) = base.strip_prefix("http://") {
        (false, rest.trim_end_matches('/').to_string())
    } else {
        (true, base.trim_end_matches('/').to_string())
    }
}

/// Extract the transcript from a Deepgram listen response
/// (`results.channels[0].alternatives[0].transcript`).
///
/// # Errors
/// Returns an error string if the body is not valid JSON or lacks the field.
pub fn parse_transcript(bytes: &[u8]) -> Result<String, String> {
    let json: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("parse: {e}"))?;
    json["results"]["channels"]
        .get(0)
        .and_then(|ch| ch["alternatives"].get(0))
        .and_then(|alt| alt["transcript"].as_str())
        .map(String::from)
        .ok_or_else(|| "no_transcript_field".to_string())
}

/// Encode f32 samples as a 16-bit PCM mono WAV.
#[must_use]
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample: u32 = 2;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes()); // byte rate
    buf.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)) as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

// ── wasi:http component (wasm32 only) ───────────────────────────────────────
#[cfg(target_arch = "wasm32")]
mod component {
    use super::{encode_wav, header, keyterms, listen_path, parse_base, parse_transcript};

    use wasi::exports::http::incoming_handler::Guest;
    use wasi::http::types::{
        Fields, IncomingBody, IncomingRequest, Method, OutgoingBody, OutgoingRequest,
        OutgoingResponse, ResponseOutparam, Scheme,
    };
    use wasi::io::streams::StreamError;

    struct Component;

    impl Guest for Component {
        fn handle(request: IncomingRequest, outparam: ResponseOutparam) {
            let (status, body) = route(&request);
            send_response(outparam, status, &body);
        }
    }

    wasi::http::proxy::export!(Component);

    /// Dispatch a `/v1` request to its handler, returning `(status, json_bytes)`.
    fn route(request: &IncomingRequest) -> (u16, Vec<u8>) {
        let method = request.method();
        let full = request.path_with_query().unwrap_or_default();
        let path = full.split('?').next().unwrap_or("");

        match (&method, path) {
            (Method::Get, "/v1/ping") => ok(&serde_json::json!({
                "status": "success", "message": "pong"
            })),
            (Method::Get, "/v1/status") => ok(&serde_json::json!({
                "status": "success", "state": "ready", "device": "cpu"
            })),
            (Method::Post, "/v1/load") => (
                202,
                to_vec(&serde_json::json!({ "status": "success", "message": "Loading started" })),
            ),
            (Method::Post, "/v1/cancel") => ok(&serde_json::json!({
                "status": "success", "message": "Cancelled"
            })),
            (Method::Post, "/v1/transcribe") => transcribe(request),
            _ => err(404, "not_found"),
        }
    }

    /// Handle `POST /v1/transcribe`: read the injected secret/option headers and
    /// the audio body, forward to Deepgram, and return the transcription.
    fn transcribe(request: &IncomingRequest) -> (u16, Vec<u8>) {
        let entries = request.headers().entries();
        let Some(api_key) = header(&entries, "x-voice-secret-deepgram_api_key") else {
            return (
                400,
                to_vec(&serde_json::json!({
                    "status": "error",
                    "message": "missing_secret_deepgram_api_key",
                    "detail": "Deepgram API key not set. Add it in Settings \u{2192} Models \u{2192} Deepgram.",
                })),
            );
        };
        // Internal test seam: `base_url` is not a declared manifest option, so
        // the daemon never sends this header in production — the default is always
        // used. The round-trip tests inject it to reach a mock upstream.
        let base_url = header(&entries, "x-voice-option-base_url")
            .unwrap_or_else(|| "https://api.deepgram.com".to_string());
        let model = header(&entries, "x-voice-model").unwrap_or_else(|| "nova-3".to_string());

        let Ok(body) = request.consume() else {
            return err(400, "no_body");
        };
        let raw = match read_all(body) {
            Ok(r) => r,
            Err(e) => return err(500, &e),
        };
        let req: serde_json::Value = match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(_) => return err(400, "invalid_json"),
        };

        let Some(audio) = req.get("audio_data").and_then(|v| v.as_array()) else {
            return err(400, "invalid_audio");
        };
        let audio: Vec<f32> = audio
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        let sample_rate = u32::try_from(
            req.get("sample_rate")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(16000),
        )
        .unwrap_or(16000);
        let language = req
            .get("language")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // The user's vocabulary, as keyterms. Read from the headers rather than
        // the body: it is the daemon's, not the caller's. Deepgram takes terms,
        // not instructions, so `x-voice-prompt` has nowhere to go.
        let keyterms = keyterms(header(&entries, "x-voice-vocabulary").as_deref());

        match call_deepgram(
            &base_url,
            &api_key,
            &model,
            language.as_deref(),
            &keyterms,
            &audio,
            sample_rate,
        ) {
            Ok(text) => (
                200,
                to_vec(&serde_json::json!({ "status": "success", "transcription": text })),
            ),
            Err(detail) => (
                502,
                to_vec(&serde_json::json!({
                    "status": "error", "message": "upstream_error", "detail": detail
                })),
            ),
        }
    }

    /// Send the audio to Deepgram's listen API and return the text.
    ///
    /// A request carrying keyterms that Deepgram answers with `400` is sent
    /// again without them. Deepgram refuses the whole request over its
    /// keyterms, when they run past its 500-token limit for one, which the
    /// budget in [`keyterms`] makes rare but cannot rule out. A vocabulary it
    /// will not take should cost the vocabulary, not the recording.
    fn call_deepgram(
        base_url: &str,
        api_key: &str,
        model: &str,
        language: Option<&str>,
        keyterms: &[String],
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<String, String> {
        let wav = encode_wav(audio, sample_rate);
        let send = |keyterms: &[String]| {
            post_listen(
                base_url,
                api_key,
                &listen_path(model, language, keyterms),
                &wav,
            )
        };
        let (mut status, mut bytes) = send(keyterms)?;
        if status == 400 && !keyterms.is_empty() {
            (status, bytes) = send(&[])?;
        }
        if !(200..300).contains(&status) {
            return Err(format!(
                "status {status}: {}",
                String::from_utf8_lossy(&bytes)
            ));
        }

        // Deepgram response: results.channels[0].alternatives[0].transcript
        parse_transcript(&bytes)
    }

    /// One `POST` of the audio to `path` on the listen API, answered with its
    /// status and body.
    fn post_listen(
        base_url: &str,
        api_key: &str,
        path: &str,
        wav: &[u8],
    ) -> Result<(u16, Vec<u8>), String> {
        let (https, authority) = parse_base(base_url);
        let scheme = if https { Scheme::Https } else { Scheme::Http };

        let headers = Fields::new();
        headers
            .append("authorization", format!("Token {api_key}").as_bytes())
            .map_err(|e| format!("header: {e:?}"))?;
        headers
            .append("content-type", b"audio/wav")
            .map_err(|e| format!("header: {e:?}"))?;

        let request = OutgoingRequest::new(headers);
        request
            .set_method(&Method::Post)
            .map_err(|()| "set_method")?;
        request
            .set_scheme(Some(&scheme))
            .map_err(|()| "set_scheme")?;
        request
            .set_authority(Some(&authority))
            .map_err(|()| "set_authority")?;
        request
            .set_path_with_query(Some(path))
            .map_err(|()| "set_path")?;

        let out_body = request.body().map_err(|()| "request_body")?;
        let future = wasi::http::outgoing_handler::handle(request, None)
            .map_err(|e| format!("handle: {e:?}"))?;
        write_all(&out_body, wav)?;
        OutgoingBody::finish(out_body, None).map_err(|e| format!("finish: {e:?}"))?;

        let pollable = future.subscribe();
        pollable.block();
        let response = future
            .get()
            .ok_or("no_response")?
            .map_err(|()| "future_taken")?
            .map_err(|e| format!("http: {e:?}"))?;

        let status = response.status();
        let body = response.consume().map_err(|()| "response_consume")?;
        Ok((status, read_all(body)?))
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    fn ok(value: &serde_json::Value) -> (u16, Vec<u8>) {
        (200, to_vec(value))
    }

    fn err(status: u16, message: &str) -> (u16, Vec<u8>) {
        (
            status,
            to_vec(&serde_json::json!({ "status": "error", "message": message })),
        )
    }

    fn to_vec(value: &serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(value).unwrap_or_default()
    }

    /// Drain an incoming body to bytes.
    fn read_all(body: IncomingBody) -> Result<Vec<u8>, String> {
        let stream = body.stream().map_err(|()| "no_stream".to_string())?;
        let mut out = Vec::new();
        loop {
            match stream.blocking_read(65536) {
                Ok(chunk) => out.extend_from_slice(&chunk),
                Err(StreamError::Closed) => break,
                Err(StreamError::LastOperationFailed(_)) => return Err("read_failed".to_string()),
            }
        }
        drop(stream);
        let _ = IncomingBody::finish(body);
        Ok(out)
    }

    /// Write all bytes to an outgoing body in ≤4096-byte flushes.
    fn write_all(body: &OutgoingBody, data: &[u8]) -> Result<(), String> {
        let stream = body.write().map_err(|()| "write_stream".to_string())?;
        for chunk in data.chunks(4096) {
            stream
                .blocking_write_and_flush(chunk)
                .map_err(|_| "write_failed".to_string())?;
        }
        drop(stream);
        Ok(())
    }

    /// Build the response and hand it to the outparam.
    fn send_response(outparam: ResponseOutparam, status: u16, body_bytes: &[u8]) {
        let headers = Fields::new();
        let _ = headers.append("content-type", b"application/json");
        let response = OutgoingResponse::new(headers);
        let _ = response.set_status_code(status);
        let Ok(body) = response.body() else {
            ResponseOutparam::set(outparam, Ok(response));
            return;
        };
        ResponseOutparam::set(outparam, Ok(response));
        if let Ok(stream) = body.write() {
            for chunk in body_bytes.chunks(4096) {
                let _ = stream.blocking_write_and_flush(chunk);
            }
            drop(stream);
        }
        let _ = OutgoingBody::finish(body, None);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_KEYTERM_BYTES, MAX_KEYTERMS, encode_wav, header, keyterms, listen_path, parse_base,
        parse_transcript, query_escape,
    };

    /// A header is found under its `x-voice-` name first, and then under its
    /// `x-stt-` one: a Super Voice daemon sends both, a Super STT daemon only
    /// the old one.
    #[test]
    fn a_header_is_read_under_its_super_voice_name_or_its_super_stt_one() {
        let both = vec![
            ("X-Voice-Model".to_string(), b"nova-3".to_vec()),
            ("x-stt-model".to_string(), b"nova-2".to_vec()),
        ];
        assert_eq!(header(&both, "x-voice-model").as_deref(), Some("nova-3"));

        let old = vec![("x-stt-model".to_string(), b"nova-2".to_vec())];
        assert_eq!(header(&old, "x-voice-model").as_deref(), Some("nova-2"));

        assert_eq!(header(&old, "x-voice-prompt"), None);
        // Only a Super Voice name falls back; any other name is looked up as is.
        assert_eq!(header(&old, "authorization"), None);
    }

    /// The vocabulary becomes keyterms: a phrase kept whole, its whitespace
    /// collapsed, an escaped term decoded, and empty or repeated terms dropped.
    #[test]
    fn the_vocabulary_becomes_keyterms() {
        let raw = r#"["Super Libre", "  Kokoro ", "Super   Libre", "", "caf\u00e9"]"#;
        assert_eq!(keyterms(Some(raw)), ["Super Libre", "Kokoro", "caf\u{e9}"]);
    }

    /// No vocabulary, an empty one and one that will not parse all give no
    /// keyterms, and the request goes ahead without them.
    #[test]
    fn no_readable_vocabulary_means_no_keyterms() {
        assert_eq!(keyterms(None), Vec::<String>::new());
        assert_eq!(keyterms(Some("[]")), Vec::<String>::new());
        assert_eq!(keyterms(Some("not json")), Vec::<String>::new());
        assert_eq!(keyterms(Some(r#""a string""#)), Vec::<String>::new());
    }

    #[test]
    fn the_keyterms_stop_at_their_limit() {
        let terms: Vec<String> = (0..MAX_KEYTERMS + 50).map(|i| format!("term{i}")).collect();
        let raw = serde_json::to_string(&terms).unwrap();
        let kept = keyterms(Some(&raw));
        assert_eq!(kept.len(), MAX_KEYTERMS);
        assert_eq!(kept[..], terms[..MAX_KEYTERMS]);
    }

    /// A vocabulary past the byte budget loses its last terms, not all of
    /// them, and keeps the user's order.
    #[test]
    fn the_keyterms_stop_at_their_byte_budget() {
        let long = "x".repeat(300);
        let terms = [
            long.clone(),
            format!("{long}a"),
            format!("{long}b"),
            format!("{long}c"),
            "short".to_string(),
        ];
        let kept = keyterms(Some(&serde_json::to_string(&terms).unwrap()));
        assert_eq!(kept[..], terms[..3]);
        assert!(kept.iter().map(String::len).sum::<usize>() <= MAX_KEYTERM_BYTES);
    }

    /// Everything but the unreserved characters is escaped, so a term can
    /// carry a space, a separator or a non-ASCII letter into the query.
    #[test]
    fn query_values_are_percent_encoded() {
        assert_eq!(query_escape("nova-3"), "nova-3");
        assert_eq!(query_escape("Super Libre"), "Super%20Libre");
        assert_eq!(query_escape("caf\u{e9}"), "caf%C3%A9");
        assert_eq!(query_escape("a,b&c=d+e"), "a%2Cb%26c%3Dd%2Be");
    }

    #[test]
    fn the_listen_path_carries_model_language_and_keyterms() {
        assert_eq!(
            listen_path("nova-3", None, &[]),
            "/v1/listen?model=nova-3&smart_format=true"
        );
        // The reserved `auto` is Deepgram's `multi`; other tags pass through.
        assert_eq!(
            listen_path("nova-3", Some("auto"), &[]),
            "/v1/listen?model=nova-3&smart_format=true&language=multi"
        );
        assert_eq!(
            listen_path("nova-3", Some("es-419"), &[]),
            "/v1/listen?model=nova-3&smart_format=true&language=es-419"
        );
        // One `keyterm` per term, never one joined with commas.
        assert_eq!(
            listen_path(
                "nova-3",
                Some("en"),
                &["Super Libre".to_string(), "Kokoro".to_string()]
            ),
            "/v1/listen?model=nova-3&smart_format=true&language=en&keyterm=Super%20Libre&keyterm=Kokoro"
        );
    }

    #[test]
    fn parse_base_splits_scheme_and_authority() {
        assert_eq!(
            parse_base("https://api.deepgram.com"),
            (true, "api.deepgram.com".to_string())
        );
        assert_eq!(
            parse_base("http://localhost:8080/"),
            (false, "localhost:8080".to_string())
        );
        // A bare host (no scheme) defaults to HTTPS.
        assert_eq!(
            parse_base("api.deepgram.com"),
            (true, "api.deepgram.com".to_string())
        );
    }

    #[test]
    fn parse_transcript_reads_nested_field() {
        let body = serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "hello world" }] }] }
        });
        let bytes = serde_json::to_vec(&body).unwrap();
        assert_eq!(parse_transcript(&bytes).unwrap(), "hello world");
    }

    #[test]
    fn parse_transcript_errors_on_missing_field() {
        let bytes = serde_json::to_vec(&serde_json::json!({ "results": {} })).unwrap();
        assert!(parse_transcript(&bytes).is_err());
        assert!(parse_transcript(b"not json").is_err());
    }

    #[test]
    fn encode_wav_writes_a_valid_header() {
        let wav = encode_wav(&[0.0, 1.0, -1.0], 16000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 3 samples * 2 bytes.
        assert_eq!(wav.len(), 44 + 6);
        // Full-scale samples clamp to i16::MAX / -i16::MAX.
        let max = i16::from_le_bytes([wav[46], wav[47]]);
        let min = i16::from_le_bytes([wav[48], wav[49]]);
        assert_eq!(max, i16::MAX);
        assert_eq!(min, -i16::MAX);
    }
}
