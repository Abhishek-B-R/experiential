//! Reseller and account-verdict tests for the pre-stream 4xx read: Novita's
//! flat envelope (sentence relay, lane limitation, MODEL_NOT_FOUND policy,
//! NOT_ENOUGH_BALANCE under 403) and the 429 body read (OpenAI
//! insufficient_quota, a throttle's token as ledger detail). Split from
//! `upstream.rs` for the module line budget; `open_against_body` is the
//! parent test module's loopback harness.

use super::tests::open_against_body;
use super::*;

#[tokio::test]
async fn a_resellers_flat_400_relays_its_sentence() {
    // Novita's envelope: no `error` object (live shape, 2026-09-15). The
    // caller must see the provider's sentence, and the ledger must record
    // the attempt as detailed, exactly as it does for an OpenAI body.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"INVALID_REQUEST_BODY\",\
         \"message\":\"max_tokens must be less than or equal to 131072\",\"metadata\":{}}",
        "deepseek/deepseek-v4.1-flash",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(!failure.failover_eligible);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("max_tokens must be less than or equal to 131072")
    );
    assert_eq!(
        failure.public_error().message,
        "provider rejected the request: max_tokens must be less than or equal to 131072"
    );
}

#[tokio::test]
async fn a_resellers_flat_lane_limitation_400_fails_over() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"INVALID_REQUEST_BODY\",\
         \"message\":\"System message must be at the beginning.\",\"metadata\":{}}",
        "pa/gpt-5.6-luna",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(
        failure.failover_eligible,
        "another rung can carry the request"
    );
    assert!(failure
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("System message must be at the beginning")));
}

#[tokio::test]
async fn a_resellers_flat_403_balance_verdict_is_provider_quota() {
    // Novita answers an unfunded account with 403 NOT_ENOUGH_BALANCE; a
    // status-only read filed it as a credential failure, which the house
    // exhaustion sweep (reading provider_quota) never sees.
    let failure = open_against_body(
        "403 Forbidden",
        "{\"code\":403,\"reason\":\"NOT_ENOUGH_BALANCE\",\
         \"message\":\"Insufficient balance\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
    assert!(failure.failover_eligible);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("http 403: NOT_ENOUGH_BALANCE"),
        "an account-state failure keeps only the status and token"
    );
    // Its credential siblings keep the credential class.
    let denied = open_against_body(
        "403 Forbidden",
        "{\"code\":403,\"reason\":\"ACCESS_DENY\",\"message\":\"Access denied\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(denied.failure_class, FailureClass::ProviderAuthentication);
}

#[tokio::test]
async fn a_429_naming_an_exhausted_account_is_provider_quota_and_a_throttle_keeps_its_code() {
    // OpenAI's live shape for an unfunded account: HTTP 429, code
    // insufficient_quota (docs, 2026-09). The status says throttle; the
    // body says the account is dead.
    let quota = open_against_body(
        "429 Too Many Requests",
        "{\"error\":{\"message\":\"You exceeded your current quota, please check your \
         plan and billing details.\",\"type\":\"insufficient_quota\",\"param\":null,\
         \"code\":\"insufficient_quota\"}}",
        "m",
    )
    .await;
    assert_eq!(quota.failure_class, FailureClass::ProviderQuota);
    assert!(quota.failover_eligible);
    assert_eq!(
        quota.provider_detail.as_deref(),
        Some("http 429: insufficient_quota")
    );
    // A genuine throttle keeps its class and the code token rides into
    // the ledger only; the public error stays the generic throttle text.
    let throttled = open_against_body(
        "429 Too Many Requests",
        "{\"code\":429,\"reason\":\"TOKEN_LIMIT_EXCEEDED\",\
         \"message\":\"Token limit exceeded, please try again later\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(throttled.failure_class, FailureClass::Throttled);
    assert!(throttled.failover_eligible);
    assert_eq!(
        throttled.provider_detail.as_deref(),
        Some("http 429: TOKEN_LIMIT_EXCEEDED")
    );
    assert_eq!(
        throttled.public_error().message,
        "provider throttled the request; retry after the delay in the Retry-After header"
    );
    // No body leaves the throttle with the status-only detail.
    let bare = open_against_body("429 Too Many Requests", "", "m").await;
    assert_eq!(bare.failure_class, FailureClass::Throttled);
    assert_eq!(bare.provider_detail.as_deref(), Some("http 429"));
}

#[tokio::test]
async fn a_resellers_flat_model_not_found_400_takes_the_lane_policy() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"MODEL_NOT_FOUND\",\"message\":\"Model not found\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
    assert!(failure.failover_eligible);
}
