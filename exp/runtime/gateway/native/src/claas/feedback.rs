//! Key-authenticated, durable-before-ack feedback and episode-finalization routes.

use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Value};

use super::feedback_contracts::{FeedbackError, FeedbackRequest, FinalizeEpisodeRequest};
use super::{feedback_store, Scope};
use crate::errors::PublicError;
use crate::respond::{bearer_key, error_response, json_response};
use crate::server::AppState;

const MAXIMUM_BODY_BYTES: usize = 1024 * 1024;

pub(crate) async fn feedback(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    respond(write(state, headers, body, false).await)
}

pub(crate) async fn finalize(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    respond(write(state, headers, body, true).await)
}

fn respond(result: Result<Value, PublicError>) -> Response {
    match result {
        Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
        Err(error) => error_response(&error),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Authority {
    user_id: String,
}

enum Request {
    Feedback(FeedbackRequest),
    Finalize(FinalizeEpisodeRequest),
}

async fn write(
    state: AppState,
    headers: HeaderMap,
    body: Body,
    finalize: bool,
) -> Result<Value, PublicError> {
    // Admission and body handling share the existing finite active-request bound.
    let permit = state.permits.clone().try_acquire_owned().map_err(|_| {
        let mut error = PublicError::new(
            503,
            "claas_busy",
            "Feedback capacity is busy. Retry this request.",
            "api_error",
        );
        error.retry_after_seconds = Some(1);
        error
    })?;
    let raw_key = bearer_key(&headers)?;
    let authority = tokio::time::timeout(
        state.request_timeout,
        state
            .bridge
            .call("claas_authority", json!({"raw_key": raw_key}).to_string()),
    )
    .await
    .map_err(|_| public_error(FeedbackError::Storage))??;
    let authority: Authority =
        serde_json::from_str(&authority).map_err(|_| PublicError::internal())?;
    let capture = state.capture.as_ref().ok_or_else(not_enabled)?;
    let bytes = tokio::time::timeout(
        state.request_timeout,
        axum::body::to_bytes(body, MAXIMUM_BODY_BYTES),
    )
    .await
    .map_err(|_| PublicError::invalid_json())?
    .map_err(|_| PublicError::request_too_large())?;
    let request = if finalize {
        let request: FinalizeEpisodeRequest =
            serde_json::from_slice(&bytes).map_err(|_| invalid_schema())?;
        request.validate().map_err(public_error)?;
        Request::Finalize(request)
    } else {
        let request: FeedbackRequest =
            serde_json::from_slice(&bytes).map_err(|_| invalid_schema())?;
        request.validate().map_err(public_error)?;
        Request::Feedback(request)
    };
    let application_id = match &request {
        Request::Feedback(request) => &request.application_id,
        Request::Finalize(request) => &request.application_id,
    };
    let policy = capture
        .policy_for_scope(&authority.user_id, application_id)
        .ok_or_else(not_enabled)?
        .clone();
    let scope = Scope {
        user_id: authority.user_id,
        application_id: application_id.clone(),
    };
    let path = std::path::PathBuf::from(capture.database_path());
    // The blocking task holds the permit through SQLite commit, even if the client disconnects.
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| FeedbackError::Storage)?
            .as_secs();
        match request {
            Request::Feedback(request) => {
                feedback_store::put_feedback(&path, &scope, &policy, request, now)
            }
            Request::Finalize(request) => {
                feedback_store::finalize_episode(&path, &scope, &policy, request, now)
            }
        }
    })
    .await
    .map_err(|_| PublicError::internal())?
    .map_err(public_error)
}

fn invalid_schema() -> PublicError {
    PublicError::new(
        400,
        "claas_invalid_request",
        "Request does not match the CLaaS feedback or episode schema.",
        "invalid_request_error",
    )
}

fn not_enabled() -> PublicError {
    PublicError::new(
        404,
        "claas_not_enabled",
        "CLaaS capture is not enabled for this key and application.",
        "invalid_request_error",
    )
}

fn public_error(error: FeedbackError) -> PublicError {
    let (status, code, message) = match error {
        FeedbackError::Invalid(message) => (400, "claas_invalid_request", message),
        FeedbackError::MissingEvidence => (409, "claas_evidence_unavailable", "Retained evidence is not available in this application. Capture may still be committing; retry the same request."),
        FeedbackError::Conflict => (409, "claas_conflict", "This identifier already has different immutable content or membership."),
        FeedbackError::Capacity => (507, "claas_capacity", "The configured feedback retention capacity is full."),
        FeedbackError::Storage => (503, "claas_storage_unavailable", "Feedback was not acknowledged. Retry with the same identifier."),
    };
    let mut public = PublicError::new(
        status,
        code,
        message,
        if status >= 500 {
            "api_error"
        } else {
            "invalid_request_error"
        },
    );
    if matches!(
        error,
        FeedbackError::MissingEvidence | FeedbackError::Storage
    ) {
        public.retry_after_seconds = Some(1);
    }
    public
}
