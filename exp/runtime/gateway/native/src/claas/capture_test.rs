//! Real HTTP delivery checks for bounded JSON and streaming capture.

use super::*;
use axum::{routing::get, Router};
use rusqlite::Connection;

fn configuration() -> CaptureConfiguration {
    static NEXT_PATH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "claas-http-{}-{}-{}.db",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    CaptureConfiguration {
        database_path: path.to_string_lossy().into(),
        queue_capacity: 4,
        bindings: vec![Binding {
            alias: "model".into(),
            policy: Policy {
                scope: Scope {
                    user_id: "user".into(),
                    application_id: "app".into(),
                },
                enabled: true,
                maximum_experiences: 10,
                maximum_storage_bytes: 100000,
                maximum_experience_bytes: 10000,
                retention_seconds: 60,
            },
        }],
    }
}

fn session(store: Arc<CaptureStore>, id: &str, protocol: &'static str) -> CaptureSession {
    CaptureSession {
        policy: store.policy("user", "model").unwrap().clone(),
        store,
        request_id: id.into(),
        model_id: "base-model".into(),
        protocol,
        request: json!({"messages":[{"role":"user","content":"hello"}]}),
        deployment_id: None,
    }
}

#[test]
fn authenticated_identity_controls_capture_and_transport_metadata_is_excluded() {
    let store = CaptureStore::open(configuration()).unwrap().unwrap();
    let mut admission: Admission = serde_json::from_value(json!({
        "request_id":"request", "alias":"model", "alias_revision_id":"alias-v1",
        "stream":false, "include_usage":false, "exact_model_id":"model-v1",
        "route_reason":"direct", "route":[], "maximum_total_attempts":1,
        "maximum_same_deployment_attempts":1, "caller_scope":"organization:other-user", "caller_identity_id":"other-user"
    }))
    .unwrap();
    let raw = json!({"messages":[{"role":"user","content":"task"}],
        "metadata":{"user_id":"user"}, "headers":{"Authorization":"canary"},
        "api_key":"canary"})
    .to_string();
    assert!(
        CaptureSession::begin(&Some(store.clone()), &admission, &raw, "chat_completions").is_none()
    );
    admission.caller_scope = Some("organization:with:colons:user".into());
    admission.caller_identity_id = Some("user".into());
    let captured =
        CaptureSession::begin(&Some(store.clone()), &admission, &raw, "chat_completions").unwrap();
    assert_eq!(
        captured.request,
        json!({"messages":[{"role":"user","content":"task"}]})
    );
    store.close();
    std::fs::remove_file(store.database_path()).unwrap();
}

#[tokio::test]
async fn captures_real_http_known_length_json_and_streamed_terminal() {
    let store = CaptureStore::open(configuration()).unwrap().unwrap();
    let json_store = store.clone();
    let stream_store = store.clone();
    let app = Router::new().route("/json", get(move || {
        let store = json_store.clone();
        async move {
            let body = json!({"id":"response-json","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}]}).to_string();
            capture_response(Some(session(store, "json", "chat_completions")),
                Response::builder().header("content-type", "application/json")
                    .header("content-length", body.len()).body(Body::from(body)).unwrap())
        }
    })).route("/sse", get(move || {
        let store = stream_store.clone();
        async move {
            let bytes = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"response-sse\",\"output\":[],\"status\":\"completed\"}}\n\n";
            let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = bytes.as_bytes().chunks(7)
                .map(|chunk| Ok(bytes::Bytes::copy_from_slice(chunk))).collect();
            capture_response(Some(session(store, "sse", "responses")),
                Response::builder().header("content-type", "text/event-stream")
                    .body(Body::from_stream(futures_util::stream::iter(chunks))).unwrap())
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    for path in ["json", "sse"] {
        let text = reqwest::get(format!("http://{address}/{path}"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(text.contains(&format!("response-{path}")));
    }
    store.close();
    let connection = Connection::open(store.database_path()).unwrap();
    let values: Vec<String> = connection
        .prepare("SELECT payload FROM claas_experiences ORDER BY sequence")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(values.len(), 2);
    for value in &values {
        let value: Value = serde_json::from_str(value).unwrap();
        assert!(value["exact_tokens"].is_null());
        assert_eq!(value["scope"]["application_id"], "app");
    }
    server.abort();
    drop(connection);
    std::fs::remove_file(store.database_path()).unwrap();
}

#[tokio::test]
async fn oversized_capture_preserves_every_caller_byte() {
    let store = CaptureStore::open(configuration()).unwrap().unwrap();
    let mut capture = session(store.clone(), "oversized", "responses");
    capture.policy.maximum_experience_bytes = 1;
    let response = capture_response(
        Some(capture),
        Response::new(Body::from("unchanged response")),
    );
    let bytes = axum::body::to_bytes(response.into_body(), 100)
        .await
        .unwrap();
    assert_eq!(&bytes[..], b"unchanged response");
    assert_eq!(store.skipped_count(), 1);
    store.close();
    std::fs::remove_file(store.database_path()).unwrap();
}
