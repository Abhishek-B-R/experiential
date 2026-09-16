//! Scope, endpoint, generation, fallback, and response-body lease checks.

use super::*;

fn fixture() -> (std::path::PathBuf, Configuration, Admission) {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "claas-serving-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&root).unwrap();
    let binding: Binding = serde_json::from_value(json!({
        "scope":{"user_id":"user","application_id":"app"}, "alias":"model",
        "registry_path": root.join("registry.json"), "state_path":root.join("state.json"),
        "admission_lock_path":root.join("admission.lock"), "private_base_url":"http://127.0.0.1:8888"
    })).unwrap();
    let revision: Revision = serde_json::from_value(json!({
        "scope":binding.scope, "policy_revision":"policy", "model_id":"Qwen/Qwen3.5-4B", "model_revision":"commit",
        "tokenizer_id":"tokenizer", "tokenizer_revision":"tokenizer-commit", "adapter_directory":null, "manifest_sha256":null
    })).unwrap();
    std::fs::write(
        &binding.registry_path,
        json!({"scope":binding.scope,"generation":0,"active":revision,"previous":null}).to_string(),
    )
    .unwrap();
    std::fs::write(&binding.state_path, json!({"schema_version":1,"scope":binding.scope,"generation":0,"policy_revision":"policy","model_name":model_name(&revision),"paused":false,"binding_sha256":digest(&serde_json::to_value(&binding).unwrap())}).to_string()).unwrap();
    let admission = serde_json::from_value(json!({
        "request_id":"request", "alias":"model", "alias_revision_id":"alias-v1", "stream":false,
        "include_usage":false, "exact_model_id":"qwen3.5-4b", "route_reason":"direct",
        "maximum_total_attempts":1, "maximum_same_deployment_attempts":1, "caller_scope":"org:user",
        "route":[{"provider":"openai-compatible","deployment_id":"private", "dialect":"openai_compatible",
        "url":"http://127.0.0.1:8888/v1/chat/completions", "headers":{}, "timeout_seconds":10,
        "model_id":"Qwen/Qwen3.5-4B", "model_revision":"commit",
        "upstream_payload":{"model":"Qwen/Qwen3.5-4B","messages":[]},"idempotency_key":"request"}]
    })).unwrap();
    (
        root,
        Configuration {
            bindings: vec![binding],
        },
        admission,
    )
}

#[test]
fn rejects_fallback_wrong_origin_and_stale_generation() {
    let (root, config, admission) = fixture();
    let mut fallback = admission.clone();
    fallback.route.push(fallback.route[0].clone());
    assert!(bind(&config, &mut fallback).is_err());
    let mut foreign = admission.clone();
    foreign.route[0].url = "https://provider.example/v1/chat/completions".into();
    assert!(bind(&config, &mut foreign).is_err());
    let mut unchanged = admission.clone();
    unchanged.caller_scope = Some("org:other-user".into());
    assert!(bind(&config, &mut unchanged).is_err());
    unchanged.route[0].url = "https://provider.example/v1/chat/completions".into();
    assert!(bind(&config, &mut unchanged).unwrap().is_none());
    assert_eq!(
        unchanged.route[0].upstream_payload["model"],
        "Qwen/Qwen3.5-4B"
    );
    let path = &config.bindings[0].state_path;
    let mut state: Value = read_json(path).unwrap();
    state["generation"] = json!(1);
    std::fs::write(path, state.to_string()).unwrap();
    assert!(bind(&config, &mut admission.clone()).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn private_origin_cannot_escape_admission_through_other_alias_or_fallback() {
    let (root, config, admission) = fixture();
    let mut other_alias = admission.clone();
    other_alias.alias = "unbound-alias".into();
    assert!(bind(&config, &mut other_alias).is_err());
    other_alias.route[0].url = "http://localhost:8888/v1/chat/completions".into();
    assert!(bind(&config, &mut other_alias).is_err());
    let mut public = other_alias.route[0].clone();
    public.url = "https://provider.example/v1/chat/completions".into();
    other_alias.route.insert(0, public);
    assert!(bind(&config, &mut other_alias).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn shared_lock_lives_until_response_body_completes_or_is_dropped() {
    let (root, config, original) = fixture();
    let mut admission = original.clone();
    let lease = bind(&config, &mut admission).unwrap();
    assert!(admission.route[0].model_id.starts_with("claas-"));
    let response = hold(lease, Response::new(Body::from("hello")));
    let writer = open_lock(&config.bindings[0].admission_lock_path).unwrap();
    assert!(writer.try_lock().is_err());
    assert_eq!(
        axum::body::to_bytes(response.into_body(), 10)
            .await
            .unwrap(),
        "hello"
    );
    writer.try_lock().unwrap();
    writer.unlock().unwrap();
    let response = hold(
        bind(&config, &mut original.clone()).unwrap(),
        Response::new(Body::from("aborted")),
    );
    assert!(writer.try_lock().is_err());
    drop(response);
    writer.try_lock().unwrap();
    drop(writer);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_configuration_rejects_shared_private_origin() {
    let (root, config, _) = fixture();
    let first = serde_json::to_value(&config.bindings[0]).unwrap();
    let mut second = first.clone();
    second["scope"]["application_id"] = json!("another-app");
    second["alias"] = json!("another-alias");
    second["private_base_url"] = json!("http://localhost:8888");
    let error =
        serde_json::from_value::<Configuration>(json!({"bindings":[first,second]})).unwrap_err();
    assert!(error.to_string().contains("one application per private"));
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_configuration_rejects_non_loopback_or_non_origin_urls() {
    let (root, config, _) = fixture();
    let mut binding = serde_json::to_value(&config.bindings[0]).unwrap();
    for raw in [
        "https://provider.example",
        "http://provider.example",
        "https://127.0.0.1:8001",
        "http://user:secret@localhost:8001",
        "http://@localhost:8001",
        "http://localhost:8001?secret=value",
        "http://localhost:8001?",
        "http://localhost:8001#fragment",
        "http://localhost:8001#",
        "http://localhost:8001/v1",
        "http://localhost:8001//",
        "http://localhost:8001/other/..",
        "http://127.1:8001",
        "http://localhost.example:8001",
        "http://localhost:0",
        "http://localhost:65536",
        " http://localhost:8001",
    ] {
        binding["private_base_url"] = json!(raw);
        let error =
            serde_json::from_value::<Configuration>(json!({"bindings":[binding]})).unwrap_err();
        assert!(error.to_string().contains("loopback HTTP"));
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn native_configuration_accepts_local_origins_without_restricting_cloud_routes() {
    let (root, config, mut admission) = fixture();
    let mut binding = serde_json::to_value(&config.bindings[0]).unwrap();
    for raw in [
        "http://127.0.0.1",
        "http://localhost",
        "http://[::1]",
        "http://127.0.0.1:8001/",
        "http://localhost:8001/",
        "http://[::1]:8001/",
    ] {
        binding["private_base_url"] = json!(raw);
        let parsed =
            serde_json::from_value::<Configuration>(json!({"bindings":[binding]})).unwrap();
        admission.alias = "ordinary-cloud-alias".into();
        admission.route[0].url = "https://provider.example/v1/chat/completions".into();
        assert!(bind(&parsed, &mut admission).unwrap().is_none());
        assert_eq!(
            admission.route[0].upstream_payload["model"],
            "Qwen/Qwen3.5-4B"
        );
        assert_eq!(
            origin(&admission.route[0].url),
            Some(("https".into(), "provider.example".into(), 443))
        );
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn provider_identity_and_declared_revision_match_before_adapter_rewrite() {
    let (root, config, original) = fixture();
    assert_ne!(original.exact_model_id, original.route[0].model_id);
    let mut correct = original.clone();
    assert!(bind(&config, &mut correct).unwrap().is_some());
    assert_eq!(correct.exact_model_id, "qwen3.5-4b");
    assert!(correct.route[0].model_id.starts_with("claas-"));
    assert_eq!(
        correct.route[0].upstream_payload["model"],
        correct.route[0].model_id
    );
    let mut unversioned = original.clone();
    unversioned.route[0].model_revision = None;
    assert!(bind(&config, &mut unversioned).unwrap().is_some());
    let mut wrong_model = original.clone();
    wrong_model.route[0].model_id = "Other/Model".into();
    wrong_model.route[0].upstream_payload["model"] = json!("Other/Model");
    assert!(bind(&config, &mut wrong_model).is_err());
    let mut wrong_revision = original.clone();
    wrong_revision.route[0].model_revision = Some("other-commit".into());
    assert!(bind(&config, &mut wrong_revision).is_err());
    let mut wrong_payload = original.clone();
    wrong_payload.route[0].upstream_payload["model"] = json!("Other/Model");
    assert!(bind(&config, &mut wrong_payload).is_err());
    std::fs::remove_dir_all(root).unwrap();
}
