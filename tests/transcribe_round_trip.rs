// SPDX-License-Identifier: GPL-3.0-only
//! End-to-end against a mock Deepgram upstream: the component shapes the request
//! (Token auth, raw WAV body, model in the query string), the host enforces the
//! egress allowlist + SSRF guard, and the transcript is parsed back out. This is
//! the standalone port of the daemon's `wasm_deepgram.rs` — daemon and upstream
//! are both mocked, the component is real.
#![allow(clippy::doc_markdown)]

mod common;

use common::{WasmBackend, component_path};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const SECRET: &str = "x-stt-secret-deepgram_api_key";
const BASE_URL: &str = "x-stt-option-base_url";

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
