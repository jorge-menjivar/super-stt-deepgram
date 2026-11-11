// SPDX-License-Identifier: GPL-3.0-only
//! End-to-end against a mock Deepgram upstream: the component shapes the request
//! (Token auth, raw WAV body, model in the query string), the host enforces the
//! egress allowlist + SSRF guard, and the transcript is parsed back out. This is
//! the standalone port of the daemon's `wasm_deepgram.rs` — daemon and upstream
//! are both mocked, the component is real.
#![allow(clippy::doc_markdown)]

mod common;

use common::{WasmBackend, component_path};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Match, Mock, MockServer, Request, ResponseTemplate};

const SECRET: &str = "x-voice-secret-deepgram_api_key";
const BASE_URL: &str = "x-voice-option-base_url";
const PROMPT: &str = "x-voice-prompt";
const VOCABULARY: &str = "x-voice-vocabulary";

/// Matches a request by whether its query carries any `keyterm` at all.
struct HasKeyterms(bool);
impl Match for HasKeyterms {
    fn matches(&self, req: &Request) -> bool {
        req.url.query_pairs().any(|(k, _)| k == "keyterm") == self.0
    }
}

/// Happy path: Token auth + `audio/wav` body + model in the query reach the
/// allowlisted upstream, and `results.channels[0].alternatives[0].transcript`
/// comes back as the transcription.
#[tokio::test]
async fn transcribe_round_trip() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        // Deepgram uses `Token` auth, not `Bearer`.
        .and(header("authorization", "Token test-key"))
        .and(header("content-type", "audio/wav"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "hello world" }] }] }
        })))
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    // The mock upstream is on loopback (wiremock binds 127.0.0.1); the SSRF guard
    // blocks loopback for untrusted backends, so opt in for the test.
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "nova-3".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let audio = vec![0.0_f32; 1600];
    let text = backend
        .transcribe_audio(&audio, 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hello world");
}

/// A Super STT daemon sends its headers only under their `x-stt-` names: the
/// key and the base URL still reach the component, which falls back to them.
#[tokio::test]
async fn the_super_stt_header_names_still_work() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(header("authorization", "Token test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "hello world" }] }] }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "nova-3".to_string(),
        vec![
            (
                "x-stt-secret-deepgram_api_key".to_string(),
                "test-key".to_string(),
            ),
            (
                "x-stt-option-base_url".to_string(),
                format!("http://{authority}"),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hello world");
}

/// The vocabulary reaches Deepgram as one `keyterm` per term: a phrase kept
/// whole and an escaped term decoded. The prompt has no Deepgram field and is
/// not sent.
#[tokio::test]
async fn the_vocabulary_is_sent_as_keyterms() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(query_param("keyterm", "Super Libre"))
        .and(query_param("keyterm", "caf\u{e9}"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "The Super Libre team shipped it." }] }] }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "nova-3".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
            // Exactly as the daemon injects them: JSON, escaped to pure ASCII.
            (
                PROMPT.to_string(),
                r#""A product announcement.""#.to_string(),
            ),
            (
                VOCABULARY.to_string(),
                r#"["Super Libre","caf\u00e9"]"#.to_string(),
            ),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "The Super Libre team shipped it.");

    let requests = server.received_requests().await.unwrap();
    let names: Vec<String> = requests[0]
        .url
        .query_pairs()
        .map(|(k, _)| k.into_owned())
        .collect();
    assert_eq!(
        names,
        ["model", "smart_format", "keyterm", "keyterm"],
        "only the vocabulary is added; the prompt has nowhere to go"
    );
}

/// A vocabulary Deepgram refuses costs the vocabulary, not the recording: the
/// request goes again without keyterms.
#[tokio::test]
async fn a_refused_vocabulary_costs_the_vocabulary_not_the_recording() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(HasKeyterms(true))
        .respond_with(ResponseTemplate::new(400).set_body_json(serde_json::json!({
            "err_code": "Bad Request",
            "err_msg": "Keyterm limit exceeded."
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(HasKeyterms(false))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "hello world" }] }] }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "nova-3".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
            (VOCABULARY.to_string(), r#"["Kokoro"]"#.to_string()),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let text = backend
        .transcribe_audio(&[0.0_f32; 1600], 16000)
        .await
        .expect("the recording must still be transcribed");
    assert_eq!(text, "hello world");
}

/// The daemon's reserved `auto` (detect) maps to Deepgram's `multi` multilingual
/// code-switching mode in the upstream query string; other tags pass through.
#[tokio::test]
async fn auto_language_maps_to_multi() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/listen"))
        .and(query_param("language", "multi"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "hola mundo" }] }] }
        })))
        .mount(&server)
        .await;

    let authority = server.address().to_string();
    let mut backend = WasmBackend::new(
        &component,
        vec![authority.clone()],
        "nova-3".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://{authority}")),
        ],
    )
    .expect("load backend")
    .permit_loopback_egress();

    let audio = vec![0.0_f32; 1600];
    let text = backend
        .transcribe_with_language(&audio, 16000, Some("auto"))
        .await
        .expect("transcription should succeed");
    assert_eq!(text, "hola mundo");
}

/// The allowlist blocks egress to a host the configuration does not permit, even
/// though a server is listening there.
#[tokio::test]
async fn allowlist_blocks_disallowed_host() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "nope" }] }] }
        })))
        .mount(&server)
        .await;

    let mut backend = WasmBackend::new(
        &component,
        // Allowlist a different host than the mock is listening on.
        vec!["api.deepgram.com".to_string()],
        "nova-3".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), server.uri()),
        ],
    )
    .expect("load backend");

    let result = backend.transcribe_audio(&[0.0_f32; 100], 16000).await;
    assert!(
        result.is_err(),
        "outbound call to a non-allowlisted host must be blocked"
    );
}

/// SSRF guard: an allowlisted *hostname* that resolves to loopback is blocked,
/// even though the host string is on the allowlist.
#[tokio::test]
async fn ssrf_blocks_hostname_resolving_to_loopback() {
    let Some(component) = component_path() else {
        eprintln!("skipping: component not built (run `just build-component`)");
        return;
    };

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": { "channels": [{ "alternatives": [{ "transcript": "nope" }] }] }
        })))
        .mount(&server)
        .await;

    let port = server.address().port();
    let mut backend = WasmBackend::new(
        &component,
        // `localhost` is allowlisted by name, but resolves to 127.0.0.1 / ::1.
        vec!["localhost".to_string()],
        "nova-3".to_string(),
        vec![
            (SECRET.to_string(), "test-key".to_string()),
            (BASE_URL.to_string(), format!("http://localhost:{port}")),
        ],
    )
    .expect("load backend");

    let result = backend.transcribe_audio(&[0.0_f32; 100], 16000).await;
    assert!(
        result.is_err(),
        "a hostname resolving to loopback must be blocked by the SSRF guard"
    );
}
