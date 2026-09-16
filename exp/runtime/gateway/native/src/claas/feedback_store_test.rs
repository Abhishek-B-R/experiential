//! Real SQLite feedback durability, isolation, retention, and immutable membership.

use super::super::feedback_contracts::EpisodeStatus;
use super::*;

static FIXTURE_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct Fixture {
    path: std::path::PathBuf,
    policy: Policy,
}
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "claas-feedback-{}-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            FIXTURE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE claas_experiences (
            user_id TEXT,application_id TEXT,response_id TEXT,expires_at INTEGER,payload TEXT,payload_bytes INTEGER);",
            )
            .unwrap();
        initialize(&connection).unwrap();
        Self {
            path,
            policy: Policy {
                scope: Scope {
                    user_id: "user".into(),
                    application_id: "app".into(),
                },
                enabled: true,
                maximum_experiences: 100,
                maximum_storage_bytes: 100_000,
                maximum_experience_bytes: 10_000,
                retention_seconds: 60,
            },
        }
    }
    fn capture(&self, id: &str, parent: Option<&str>) {
        let payload = json!({"scope":{"user_id":"user","application_id":"app"},
            "response_id":id,"parent_response_id":parent});
        Connection::open(&self.path)
            .unwrap()
            .execute(
                "INSERT INTO claas_experiences VALUES ('user','app',?1,100,?2,?3)",
                params![id, payload.to_string(), payload.to_string().len() as i64],
            )
            .unwrap();
    }
    fn feedback(&self, request: FeedbackRequest) -> Result<Value, FeedbackError> {
        put_feedback(&self.path, &self.policy.scope, &self.policy, request, 1)
    }
    fn finalize(&self, request: FinalizeEpisodeRequest) -> Result<Value, FeedbackError> {
        finalize_episode(&self.path, &self.policy.scope, &self.policy, request, 1)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
fn feedback() -> FeedbackRequest {
    FeedbackRequest {
        application_id: "app".into(),
        feedback_id: "feedback".into(),
        response_id: Some("one".into()),
        episode_id: None,
        text: Some("The address is incorrect.".into()),
        reward: None,
        success: None,
    }
}
fn episode() -> FinalizeEpisodeRequest {
    FinalizeEpisodeRequest {
        application_id: "app".into(),
        episode_id: "episode".into(),
        response_ids: vec!["one".into(), "two".into()],
        status: EpisodeStatus::Completed,
    }
}

#[test]
fn durable_ack_replays_exact_record_and_rejects_changed_feedback() {
    let fixture = Fixture::new();
    fixture.capture("one", None);
    let first = fixture.feedback(feedback()).unwrap();
    assert_eq!(first["replayed"], false);
    assert!(first["record"]["feedback"]["reward"].is_null());
    let replay = fixture.feedback(feedback()).unwrap();
    assert_eq!(replay["record"], first["record"]);
    assert_eq!(replay["replayed"], true);
    let stored: String = Connection::open(&fixture.path)
        .unwrap()
        .query_row("SELECT payload FROM claas_feedback", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&stored).unwrap(),
        first["record"]
    );
    let mut changed = feedback();
    changed.success = Some(false);
    assert_eq!(fixture.feedback(changed), Err(FeedbackError::Conflict));
}

#[test]
fn unknown_response_can_be_retried_after_capture_and_scopes_do_not_cross() {
    let mut fixture = Fixture::new();
    assert_eq!(
        fixture.feedback(feedback()),
        Err(FeedbackError::MissingEvidence)
    );
    fixture.capture("one", None);
    fixture.policy.scope.user_id = "other".into();
    assert_eq!(
        fixture.feedback(feedback()),
        Err(FeedbackError::MissingEvidence)
    );
    fixture.policy.scope.user_id = "user".into();
    assert!(fixture.feedback(feedback()).is_ok());
    fixture.policy.enabled = false;
    assert_eq!(
        fixture.feedback(feedback()),
        Err(FeedbackError::MissingEvidence)
    );
}

#[test]
fn explicit_episode_is_immutable_ordered_and_not_a_success_label() {
    let fixture = Fixture::new();
    fixture.capture("one", None);
    fixture.capture("two", Some("one"));
    let mut reversed = episode();
    reversed.response_ids.reverse();
    assert!(matches!(
        fixture.finalize(reversed),
        Err(FeedbackError::Invalid(_))
    ));
    let first = fixture.finalize(episode()).unwrap();
    assert_eq!(first["record"]["episode"]["status"], "completed");
    assert!(first["record"]["episode"].get("success").is_none());
    assert_eq!(fixture.finalize(episode()).unwrap()["replayed"], true);
    fixture.capture("three", None);
    let mut append = episode();
    append.response_ids.push("three".into());
    assert_eq!(fixture.finalize(append), Err(FeedbackError::Conflict));
    let mut other = episode();
    other.episode_id = "other".into();
    assert_eq!(fixture.finalize(other), Err(FeedbackError::Conflict));
    let mut request = feedback();
    request.response_id = None;
    request.episode_id = Some("episode".into());
    assert!(fixture.feedback(request).is_ok());
}

#[test]
fn duplicate_members_unknown_parents_and_cross_scope_parents_fail_closed() {
    let fixture = Fixture::new();
    fixture.capture("one", Some("other-parent"));
    let mut request = episode();
    request.response_ids = vec!["one".into(), "one".into()];
    assert!(matches!(
        fixture.finalize(request),
        Err(FeedbackError::Invalid(_))
    ));
    let mut request = episode();
    request.response_ids = vec!["one".into()];
    assert_eq!(
        fixture.finalize(request),
        Err(FeedbackError::MissingEvidence)
    );
    let count: i64 = Connection::open(&fixture.path)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM claas_episodes", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn eviction_and_expiration_remove_all_dependent_feedback_and_members() {
    let fixture = Fixture::new();
    fixture.capture("one", None);
    fixture.capture("two", Some("one"));
    fixture.finalize(episode()).unwrap();
    let mut request = feedback();
    request.response_id = None;
    request.episode_id = Some("episode".into());
    fixture.feedback(request).unwrap();
    let connection = Connection::open(&fixture.path).unwrap();
    connection
        .execute("DELETE FROM claas_experiences WHERE response_id='two'", [])
        .unwrap();
    prune(&connection, 2).unwrap();
    for table in ["claas_feedback", "claas_episodes", "claas_episode_members"] {
        let count: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
    fixture.feedback(feedback()).unwrap();
    prune(&connection, 100).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM claas_feedback", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn capacity_and_storage_failure_never_acknowledge_or_erase_prior_feedback() {
    let mut fixture = Fixture::new();
    fixture.capture("one", None);
    fixture.policy.maximum_experiences = 2;
    fixture.feedback(feedback()).unwrap();
    let mut second = feedback();
    second.feedback_id = "second".into();
    assert_eq!(fixture.feedback(second), Err(FeedbackError::Capacity));
    assert_eq!(fixture.feedback(feedback()).unwrap()["replayed"], true);
    Connection::open(&fixture.path).unwrap().execute_batch("CREATE TRIGGER fail_feedback BEFORE INSERT ON claas_feedback BEGIN SELECT RAISE(ABORT,'fail'); END;").unwrap();
    fixture.policy.maximum_experiences = 100;
    let mut third = feedback();
    third.feedback_id = "third".into();
    assert_eq!(fixture.feedback(third), Err(FeedbackError::Storage));
}

#[test]
fn concurrent_duplicates_acknowledge_one_committed_row() {
    let fixture = Fixture::new();
    fixture.capture("one", None);
    let results = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| fixture.feedback(feedback())))
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        results
            .iter()
            .filter(|value| value["replayed"] == false)
            .count(),
        1
    );
}

#[test]
fn scalar_and_binary_are_strict_and_missing_does_not_become_zero() {
    for body in [
        json!({"reward":2}),
        json!({"reward":true}),
        json!({"success":0}),
        json!({"reward":-1,"success":false}),
        json!({"user_id":"forged"}),
    ] {
        let mut request = serde_json::to_value(feedback()).unwrap();
        request
            .as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        let parsed = serde_json::from_value::<FeedbackRequest>(request);
        assert!(parsed.is_err() || parsed.unwrap().validate().is_err());
    }
    let mut request = feedback();
    request.success = Some(false);
    assert!(request.validate().is_ok());
    assert!(request.reward.is_none());
}

#[test]
fn feedback_byte_limit_counts_capture_payloads() {
    let original = Fixture::new();
    original.capture("one", None);
    let record = original.feedback(feedback()).unwrap()["record"].clone();
    let record_bytes = serde_json::to_string(&record).unwrap().len();
    let mut fixture = Fixture::new();
    fixture.capture("one", None);
    let captured: usize = Connection::open(&fixture.path)
        .unwrap()
        .query_row("SELECT payload_bytes FROM claas_experiences", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap() as usize;
    fixture.policy.maximum_storage_bytes = record_bytes + captured - 1;
    fixture.policy.maximum_experience_bytes = fixture.policy.maximum_storage_bytes;
    assert_eq!(fixture.feedback(feedback()), Err(FeedbackError::Capacity));
    fixture.policy.maximum_storage_bytes += 1;
    assert!(fixture.feedback(feedback()).is_ok());
}
