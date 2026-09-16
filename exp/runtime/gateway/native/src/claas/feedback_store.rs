//! Transactional feedback and immutable episode membership beside captured traffic.

use std::path::Path;
use std::time::Duration;

use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::feedback_contracts::{FeedbackError, FeedbackRequest, FinalizeEpisodeRequest};
use super::{Policy, Scope};

/// Persist explicit feedback before returning its acknowledgement payload.
pub(crate) fn put_feedback(
    path: &Path,
    scope: &Scope,
    policy: &Policy,
    request: FeedbackRequest,
    now: u64,
) -> Result<Value, FeedbackError> {
    request.validate()?;
    require_scope(scope, policy, &request.application_id)?;
    let mut connection = open(path)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(storage)?;
    prune(&transaction, now)?;
    let digest = digest(&request)?;
    if let Some(payload) = retained(
        &transaction,
        "claas_feedback",
        "feedback_id",
        scope,
        &request.feedback_id,
        &digest,
    )? {
        transaction.commit().map_err(storage)?;
        return Ok(json!({"record": payload, "replayed": true}));
    }
    let expires_at = if let Some(response_id) = &request.response_id {
        response(&transaction, scope, response_id, now)?.0
    } else {
        transaction
            .query_row(
                "SELECT expires_at FROM claas_episodes
             WHERE user_id=?1 AND application_id=?2 AND episode_id=?3 AND expires_at>?4",
                params![
                    scope.user_id,
                    scope.application_id,
                    request.episode_id,
                    now as i64
                ],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(storage)?
            .ok_or(FeedbackError::MissingEvidence)?
    };
    let record = json!({
        "schema_version": 1,
        "scope": {"user_id": scope.user_id, "application_id": scope.application_id},
        "feedback": request,
        "created_at": now,
    });
    let payload = serde_json::to_string(&record).map_err(|_| FeedbackError::Storage)?;
    reserve_capacity(&transaction, scope, policy, payload.len())?;
    transaction
        .execute(
            "INSERT INTO claas_feedback
         (user_id,application_id,feedback_id,response_id,episode_id,expires_at,
          request_digest,payload_bytes,payload) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                scope.user_id,
                scope.application_id,
                request.feedback_id,
                request.response_id,
                request.episode_id,
                expires_at,
                digest,
                payload.len() as i64,
                payload
            ],
        )
        .map_err(storage)?;
    transaction.commit().map_err(storage)?;
    Ok(json!({"record": record, "replayed": false}))
}

/// Freeze an episode's explicit membership and caller-reported terminal state.
pub(crate) fn finalize_episode(
    path: &Path,
    scope: &Scope,
    policy: &Policy,
    request: FinalizeEpisodeRequest,
    now: u64,
) -> Result<Value, FeedbackError> {
    request.validate()?;
    require_scope(scope, policy, &request.application_id)?;
    let mut connection = open(path)?;
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(storage)?;
    prune(&transaction, now)?;
    let digest = digest(&request)?;
    if let Some(payload) = retained(
        &transaction,
        "claas_episodes",
        "episode_id",
        scope,
        &request.episode_id,
        &digest,
    )? {
        transaction.commit().map_err(storage)?;
        return Ok(json!({"record": payload, "replayed": true}));
    }
    let mut expires_at = i64::MAX;
    let mut preceding = std::collections::HashSet::new();
    for response_id in &request.response_ids {
        let (expires, payload) = response(&transaction, scope, response_id, now)?;
        expires_at = expires_at.min(expires);
        if let Some(parent) = payload.get("parent_response_id").and_then(Value::as_str) {
            // Resolve in the same scope, even when a forged payload points to another caller.
            response(&transaction, scope, parent, now)?;
            if !preceding.contains(parent) {
                return Err(FeedbackError::Invalid(
                    "Include every Responses parent before its child when finalizing an episode.",
                ));
            }
        }
        let owner = transaction
            .query_row(
                "SELECT episode_id FROM claas_episode_members
             WHERE user_id=?1 AND application_id=?2 AND response_id=?3",
                params![scope.user_id, scope.application_id, response_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(storage)?;
        if owner.is_some() {
            return Err(FeedbackError::Conflict);
        }
        preceding.insert(response_id.as_str());
    }
    let record = json!({
        "schema_version": 1,
        "scope": {"user_id": scope.user_id, "application_id": scope.application_id},
        "episode": request,
        "finalized_at": now,
    });
    let payload = serde_json::to_string(&record).map_err(|_| FeedbackError::Storage)?;
    reserve_capacity(&transaction, scope, policy, payload.len())?;
    transaction
        .execute(
            "INSERT INTO claas_episodes
         (user_id,application_id,episode_id,expires_at,request_digest,payload_bytes,payload)
         VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![
                scope.user_id,
                scope.application_id,
                request.episode_id,
                expires_at,
                digest,
                payload.len() as i64,
                payload
            ],
        )
        .map_err(storage)?;
    for response_id in &request.response_ids {
        transaction
            .execute(
                "INSERT INTO claas_episode_members(user_id,application_id,episode_id,response_id)
             VALUES (?1,?2,?3,?4)",
                params![
                    scope.user_id,
                    scope.application_id,
                    request.episode_id,
                    response_id
                ],
            )
            .map_err(storage)?;
    }
    transaction.commit().map_err(storage)?;
    Ok(json!({"record": record, "replayed": false}))
}

fn open(path: &Path) -> Result<Connection, FeedbackError> {
    // Feedback never creates an unconfigured capture database.
    let connection =
        Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE).map_err(storage)?;
    connection
        .busy_timeout(Duration::from_millis(250))
        .map_err(storage)?;
    initialize(&connection)?;
    Ok(connection)
}

pub(crate) fn initialize(connection: &Connection) -> Result<(), FeedbackError> {
    connection
        .execute_batch(
            "PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON; PRAGMA secure_delete=ON;
         CREATE TABLE IF NOT EXISTS claas_feedback (
           sequence INTEGER PRIMARY KEY AUTOINCREMENT,
           user_id TEXT NOT NULL, application_id TEXT NOT NULL, feedback_id TEXT NOT NULL,
           response_id TEXT, episode_id TEXT, expires_at INTEGER NOT NULL,
           request_digest TEXT NOT NULL, payload_bytes INTEGER NOT NULL, payload TEXT NOT NULL,
           UNIQUE(user_id,application_id,feedback_id),
           CHECK((response_id IS NULL) != (episode_id IS NULL)));
         CREATE TABLE IF NOT EXISTS claas_episodes (
           sequence INTEGER PRIMARY KEY AUTOINCREMENT,
           user_id TEXT NOT NULL, application_id TEXT NOT NULL, episode_id TEXT NOT NULL,
           expires_at INTEGER NOT NULL, request_digest TEXT NOT NULL,
           payload_bytes INTEGER NOT NULL, payload TEXT NOT NULL,
           UNIQUE(user_id,application_id,episode_id));
         CREATE TABLE IF NOT EXISTS claas_episode_members (
           user_id TEXT NOT NULL, application_id TEXT NOT NULL,
           episode_id TEXT NOT NULL, response_id TEXT NOT NULL,
           PRIMARY KEY(user_id,application_id,response_id),
           FOREIGN KEY(user_id,application_id,episode_id)
             REFERENCES claas_episodes(user_id,application_id,episode_id) ON DELETE CASCADE);
         CREATE INDEX IF NOT EXISTS claas_feedback_scope_sequence
           ON claas_feedback(user_id,application_id,sequence);
         CREATE INDEX IF NOT EXISTS claas_episode_scope_sequence
           ON claas_episodes(user_id,application_id,sequence);
         CREATE INDEX IF NOT EXISTS claas_episode_member_owner
           ON claas_episode_members(user_id,application_id,episode_id);",
        )
        .map_err(storage)?;
    Ok(())
}

fn response(
    connection: &Connection,
    scope: &Scope,
    response_id: &str,
    now: u64,
) -> Result<(i64, Value), FeedbackError> {
    let row = connection
        .query_row(
            "SELECT expires_at,payload FROM claas_experiences
         WHERE user_id=?1 AND application_id=?2 AND response_id=?3 AND expires_at>?4",
            params![scope.user_id, scope.application_id, response_id, now as i64],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(storage)?
        .ok_or(FeedbackError::MissingEvidence)?;
    let payload: Value = serde_json::from_str(&row.1).map_err(|_| FeedbackError::Storage)?;
    if payload["scope"]["user_id"] != scope.user_id
        || payload["scope"]["application_id"] != scope.application_id
        || payload["response_id"] != response_id
    {
        return Err(FeedbackError::Storage);
    }
    Ok((row.0, payload))
}

fn retained(
    connection: &Connection,
    table: &str,
    id_column: &str,
    scope: &Scope,
    id: &str,
    digest: &str,
) -> Result<Option<Value>, FeedbackError> {
    // Table and column are module-owned literals, never caller data.
    let sql = format!(
        "SELECT request_digest,payload FROM {table}
        WHERE user_id=?1 AND application_id=?2 AND {id_column}=?3"
    );
    let row = connection
        .query_row(
            &sql,
            params![scope.user_id, scope.application_id, id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(storage)?;
    match row {
        Some((prior_digest, _)) if prior_digest != digest => Err(FeedbackError::Conflict),
        Some((_, payload)) => serde_json::from_str(&payload)
            .map(Some)
            .map_err(|_| FeedbackError::Storage),
        None => Ok(None),
    }
}

fn reserve_capacity(
    connection: &Connection,
    scope: &Scope,
    policy: &Policy,
    added_bytes: usize,
) -> Result<(), FeedbackError> {
    let (records, bytes): (i64, i64) = connection.query_row(
        "SELECT COUNT(*),COALESCE(SUM(payload_bytes),0) FROM (
           SELECT payload_bytes FROM claas_experiences WHERE user_id=?1 AND application_id=?2
           UNION ALL SELECT payload_bytes FROM claas_feedback WHERE user_id=?1 AND application_id=?2
           UNION ALL SELECT payload_bytes FROM claas_episodes WHERE user_id=?1 AND application_id=?2)",
        params![scope.user_id, scope.application_id], |row| Ok((row.get(0)?,row.get(1)?)),
    ).map_err(storage)?;
    if records >= policy.maximum_experiences as i64
        || added_bytes > policy.maximum_experience_bytes
        || bytes.saturating_add(added_bytes as i64) > policy.maximum_storage_bytes as i64
    {
        return Err(FeedbackError::Capacity);
    }
    Ok(())
}

pub(crate) fn prune(connection: &Connection, now: u64) -> Result<(), FeedbackError> {
    connection
        .execute(
            "DELETE FROM claas_episodes WHERE expires_at<=?1 OR EXISTS (
          SELECT 1 FROM claas_episode_members m WHERE m.user_id=claas_episodes.user_id
            AND m.application_id=claas_episodes.application_id
            AND m.episode_id=claas_episodes.episode_id AND NOT EXISTS (
              SELECT 1 FROM claas_experiences e WHERE e.user_id=m.user_id
                AND e.application_id=m.application_id AND e.response_id=m.response_id))",
            [now as i64],
        )
        .map_err(storage)?;
    connection
        .execute(
            "DELETE FROM claas_episode_members WHERE NOT EXISTS (
          SELECT 1 FROM claas_episodes e WHERE e.user_id=claas_episode_members.user_id
            AND e.application_id=claas_episode_members.application_id
            AND e.episode_id=claas_episode_members.episode_id)",
            [],
        )
        .map_err(storage)?;
    connection
        .execute(
            "DELETE FROM claas_feedback WHERE expires_at<=?1
          OR (response_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM claas_experiences e WHERE e.user_id=claas_feedback.user_id
              AND e.application_id=claas_feedback.application_id
              AND e.response_id=claas_feedback.response_id))
          OR (episode_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM claas_episodes e WHERE e.user_id=claas_feedback.user_id
              AND e.application_id=claas_feedback.application_id
              AND e.episode_id=claas_feedback.episode_id))",
            [now as i64],
        )
        .map_err(storage)?;
    Ok(())
}

fn require_scope(scope: &Scope, policy: &Policy, application: &str) -> Result<(), FeedbackError> {
    if !policy.enabled
        || scope.user_id != policy.scope.user_id
        || scope.application_id != policy.scope.application_id
        || scope.application_id != application
    {
        return Err(FeedbackError::MissingEvidence);
    }
    Ok(())
}

fn digest(value: &impl serde::Serialize) -> Result<String, FeedbackError> {
    let bytes = serde_json::to_vec(value).map_err(|_| FeedbackError::Storage)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn storage(_: rusqlite::Error) -> FeedbackError {
    FeedbackError::Storage
}

#[cfg(test)]
#[path = "feedback_store_test.rs"]
mod tests;
