//! Upstream provider HTTP transport over one shared pooled client.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::dialects::Dialect;
use crate::errors::{Failure, FailureClass};
use crate::param_attribution::{
    bounded_masked_line, content_filtered_completion, generic_error_code,
    rejected_by_lane_limitation, rejected_by_routing_gate, rejected_caller_reference_not_found,
    rejected_code, rejected_detail, rejected_model_not_found, rejected_parameter, sanitized_detail,
};
use crate::rate_limit_headers::{harvest_rate_limit_headers, retry_after_seconds};

/// Build the shared pooled upstream client, mirroring the pooling constants in
/// `providers.async_transport` (64 keep-alive) and its no-redirect policy so a
/// provider 3xx can never re-send credentials to an attacker-chosen location.
///
/// `connect_timeout` bounds only the TCP+TLS connect phase; a dead lane whose
/// host never accepts the connection fails over after this window instead of
/// hanging on the per-deployment request timeout.
pub fn build_client(connect_timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .connect_timeout(connect_timeout)
        .redirect(reqwest::redirect::Policy::none())
        .use_rustls_tls()
        .build()
        .map_err(|error| format!("upstream client construction failed: {error}"))
}

/// Classify one sanitized HTTP or connection failure by status only,
/// mirroring `providers.errors._transport_failure`: classes, wording, the
/// same-deployment retry policy, and failover eligibility across the
/// certified deployment ladder.
pub fn transport_failure(status: Option<u16>) -> Failure {
    let (class, message, retryable, failover) = match status {
        Some(401) | Some(403) => (
            FailureClass::ProviderAuthentication,
            "provider authentication failed; ask the gateway operator to verify \
             the provider connection credential",
            false,
            true,
        ),
        Some(404) => (
            FailureClass::ProviderNotFound,
            "provider deployment was not found; ask the gateway operator to verify \
             the deployment model ID in the catalog",
            false,
            true,
        ),
        Some(429) => (
            FailureClass::Throttled,
            "provider throttled the request; retry after the delay in the Retry-After header",
            false,
            true,
        ),
        // 402 is the provider ACCOUNT's billing state (trial quota exhausted,
        // postpaid billing disabled), never the caller's request fields: it is
        // operator-actionable deadness, so it fails over in every failover mode
        // instead of surfacing a corrective 400 to the caller.
        Some(402) => (
            FailureClass::ProviderQuota,
            "provider account quota or billing is exhausted; ask the gateway operator \
             to fund or enable the provider account",
            false,
            true,
        ),
        Some(408) => (
            FailureClass::Timeout,
            "provider request timed out; retry the request",
            true,
            true,
        ),
        Some(code) if code >= 500 => (
            FailureClass::ProviderInternal,
            "provider service failed; retry after a short delay",
            true,
            true,
        ),
        Some(409) | Some(425) => (
            FailureClass::ProviderInternal,
            "provider reported a transient conflict; retry the request",
            true,
            true,
        ),
        Some(code) if (400..500).contains(&code) => (
            FailureClass::InvalidRequest,
            "provider rejected the request; verify the request fields against \
             the model alias capabilities",
            false,
            false,
        ),
        // Redirects are disabled, so a 3xx (or any other status) is an
        // unexpected provider response, never followed.
        Some(_) => (
            FailureClass::ProviderInternal,
            "provider returned an unexpected status; retry the request",
            false,
            true,
        ),
        None => (
            FailureClass::Transport,
            "provider transport failed; retry the request",
            true,
            true,
        ),
    };
    Failure::new(class, message).with_retry(retryable, failover)
}

/// Classify a lead that connected but never completed the request/response-header
/// phase within `phase_timeout`. A deployment that accepted the connection but
/// stalled awaiting response headers is the same dead-lane signal as a stalled
/// first byte, so it mirrors `relay::first_byte_timeout_failure`: failover-eligible
/// (advance to the next certified rung) but deliberately *not* same-deployment
/// retryable. Redialing the same stalled deployment would only burn another full
/// header-timeout window before failing over; skipping straight to the next rung
/// keeps a stalled lead's cost near one fail-fast window. It stays a
/// `FailureClass::Timeout`, so it feeds the health circuit like other timeouts.
fn open_timeout_failure() -> Failure {
    Failure::new(
        FailureClass::Timeout,
        "provider did not send response headers in time",
    )
    .with_retry(false, true)
}

/// Open one streaming POST and return the response on HTTP success. The
/// timeout bounds only the request/response-header phase; body-read pacing is
/// bounded per chunk by the caller, mirroring the python transport split.
///
/// `raw_body` carries the exact pre-serialized body for body-signing dialects
/// (Bedrock SigV4): its signature covers those exact bytes, so it is sent
/// verbatim with the signed headers instead of re-serializing `payload`.
#[allow(clippy::too_many_arguments)]
pub async fn open_stream(
    client: &reqwest::Client,
    url: &str,
    headers: &HashMap<String, String>,
    idempotency_key: &str,
    payload: &Value,
    raw_body: Option<&str>,
    phase_timeout: Duration,
    dialect: Dialect,
) -> Result<reqwest::Response, Failure> {
    let mut request = client.post(url);
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("idempotency-key") {
            continue;
        }
        request = request.header(name, value);
    }
    request = request.header("Idempotency-Key", idempotency_key);
    let send = match raw_body {
        Some(body) => request.body(body.to_string()).send(),
        None => request.json(payload).send(),
    };
    let phase_started = Instant::now();
    let response = match tokio::time::timeout(phase_timeout, send).await {
        Ok(Ok(response)) => response,
        Ok(Err(error)) => {
            if error.is_timeout() {
                return Err(open_timeout_failure());
            }
            // The failure stays a content-free transport class on the wire;
            // the engine's own account of WHICH transport fault (never
            // provider text) rides to the ledger so the row is diagnosable.
            return Err(transport_failure(None)
                .with_provider_detail(Some(transport_error_detail("open", &error))));
        }
        Err(_) => return Err(open_timeout_failure()),
    };
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        // Rate-limit facts are read off the headers before anything consumes
        // the response: a 429's `retry-after` and remaining-quota counts ride
        // the failure into settlement (never to the caller), where the
        // control plane sizes throttle windows and persists them per attempt.
        let rate_limit = harvest_rate_limit_headers(response.headers());
        let retry_after = retry_after_seconds(response.headers());
        let failure =
            transport_failure(Some(status)).with_rate_limit_facts(rate_limit.clone(), retry_after);
        // Only the generic client-error class may carry attribution: the body
        // is read bounded, and the relayable facts are a validated parameter
        // path plus the provider's own bounded explanation of what the caller
        // got wrong; every other class stays content-free. A 403 is read too,
        // only to tell an aggregator routing gate from a credential verdict,
        // a 404 to tell a caller's dangling reference from a missing model,
        // and a 429 to tell an exhausted ACCOUNT from a throttle and to file
        // the provider's code token (never its sentence) in the ledger.
        // Every status-only classification carries the status itself as its
        // ledger detail (`http 503`): the class alone could not tell a 502
        // relay from a 500 model fault, and none of these classes relays
        // detail to the caller.
        let failure = if failure.failure_class == FailureClass::InvalidRequest {
            failure
        } else {
            failure.with_provider_detail(Some(status_detail(status)))
        };
        if failure.failure_class != FailureClass::InvalidRequest
            && status != 403
            && status != 404
            && status != 429
        {
            return Err(failure);
        }
        // The attribution read never outlives the rung's own header-phase
        // budget: a provider that answers its status and then stalls the body
        // costs at most what was left of that window, never a further two
        // seconds past the caller's deadline. A throttle is the hot path
        // under load and its failover must stay near-immediate, so its read
        // gets only the short budget: the small envelope normally arrives
        // with the headers, and a provider that stalls after a 429 simply
        // fails over content-free as before.
        let read_timeout = if status == 429 {
            THROTTLE_BODY_READ_TIMEOUT
        } else {
            ERROR_BODY_READ_TIMEOUT
        };
        let body_budget = read_timeout.min(phase_timeout.saturating_sub(phase_started.elapsed()));
        let body = match tokio::time::timeout(body_budget, bounded_error_body(response)).await {
            Ok(Some(body)) => Some(body),
            _ => None,
        };
        if status == 429 {
            // OpenAI answers an out-of-quota account with 429
            // `insufficient_quota`, the same status as a throttle. A status-
            // only read filed both as `throttled`, so the house exhaustion
            // sweep (which reads `provider_quota`) never saw the account die.
            // Any other code token rides the failure into the ledger only
            // (a throttle's public error never relays detail): Novita's
            // `RATE_LIMIT_EXCEEDED` versus `TOKEN_LIMIT_EXCEEDED` names which
            // window closed, which its headers do not (2026-09-14: 205
            // Novita 429s with 992-999 of 1000 requests remaining and no
            // Retry-After).
            let code = body
                .as_deref()
                .and_then(|body| rejected_code(dialect, body));
            let detail = code
                .as_deref()
                .filter(|token| !generic_error_code(token))
                .map(|token| format!("{}: {token}", status_detail(status)));
            if crate::stream_errors::is_quota_code(code.as_deref()) {
                return Err(transport_failure(Some(402))
                    .with_provider_detail(detail)
                    .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            return Err(match detail {
                Some(detail) => failure.with_provider_detail(Some(detail)),
                None => failure,
            });
        }
        if status == 404 {
            // OpenAI answers 404 for an `item_reference`, `conversation`, or
            // similar handle the caller sent but the provider does not hold
            // (store=false items are never persisted). The catalog is fine and
            // every rung would answer the same, so it is the caller's 400 with
            // the provider's sentence, never a lane 404 that walks the ladder.
            if body
                .as_deref()
                .is_some_and(|body| rejected_caller_reference_not_found(dialect, body))
            {
                let request_words: Vec<&str> = payload
                    .get("model")
                    .and_then(Value::as_str)
                    .into_iter()
                    .collect();
                let detail = body
                    .as_deref()
                    .and_then(|body| rejected_detail(dialect, body, &request_words));
                let parameter = body
                    .as_deref()
                    .and_then(|body| rejected_parameter(dialect, body));
                return Err(Failure::new(
                    FailureClass::InvalidRequest,
                    "the request references a provider-side item, response, or conversation \
                     the provider does not hold; resend that content inline",
                )
                .with_retry(false, false)
                .with_rejected_parameter(parameter)
                .with_provider_detail(detail)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            return Err(failure);
        }
        if status == 403 {
            if body
                .as_deref()
                .is_some_and(|body| rejected_by_routing_gate(dialect, body))
            {
                return Err(Failure::new(
                    FailureClass::ProviderNotFound,
                    "provider does not route this model for the gateway's account; ask \
                     the gateway operator to change or disable the lane",
                )
                .with_retry(false, true)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            // A reseller's balance verdict under a 403 (Novita answers an
            // unfunded account `403 NOT_ENOUGH_BALANCE`) is the ACCOUNT's
            // funding state, not a credential one: it takes the quota class
            // the house exhaustion sweep and pool rotation read, with the
            // status and token as its ledger detail like every other
            // operator-facing class.
            let code = body
                .as_deref()
                .and_then(|body| rejected_code(dialect, body));
            if crate::stream_errors::is_quota_code(code.as_deref()) {
                let token = code.unwrap_or_default();
                return Err(transport_failure(Some(402))
                    .with_provider_detail(Some(format!("{}: {token}", status_detail(status))))
                    .with_rate_limit_facts(rate_limit.clone(), retry_after));
            }
            return Err(failure);
        }
        // A client-error status whose body names a missing model is the
        // catalog's fault, not the caller's: it takes the 404 policy so the
        // ladder advances instead of surfacing one dead rung as a 400.
        if body
            .as_deref()
            .is_some_and(|body| rejected_model_not_found(dialect, body))
        {
            return Err(transport_failure(Some(404))
                .with_provider_detail(Some(format!("{}: model_not_found", status_detail(status))))
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
        }
        let parameter = body
            .as_deref()
            .and_then(|body| rejected_parameter(dialect, body));
        // The payload's own model id is a caller-known word: a provider
        // sentence naming it unquoted (Anthropic's client-version gate does)
        // must not be redacted as infrastructure.
        let request_words: Vec<&str> = payload
            .get("model")
            .and_then(Value::as_str)
            .into_iter()
            .collect();
        // A 4xx whose body is a COMPLETION finished `content_filter` (Azure
        // Foundry's DeepSeek lanes answer their output filter this way, with
        // no error envelope at all) is the model's verdict on the content:
        // file and answer it as a refusal, the blocked label kept ledger-only,
        // never as a request-shape error and never a failover (the next rung
        // refuses the same content or, worse, serves it).
        if let Some(filtered) = body
            .as_deref()
            .and_then(|body| content_filtered_completion(dialect, body))
        {
            let reason = crate::stream_errors::refusal_reason(
                Some(&filtered.code),
                filtered.message.as_deref(),
            );
            let detail = filtered
                .message
                .as_deref()
                .and_then(|message| sanitized_detail(message, &request_words))
                .or(Some(filtered.code));
            return Err(Failure::refusal(reason)
                .with_provider_detail(detail)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
        }
        let code = body
            .as_deref()
            .and_then(|body| rejected_code(dialect, body));
        let detail = body
            .as_deref()
            .and_then(|body| rejected_detail(dialect, body, &request_words))
            // A sentence the identifier screen dropped still leaves the
            // provider's own code token: "invalid_value" beats "verify the
            // request fields" for the caller and the ledger alike. A generic
            // family type or bare status adds nothing and is not relayed.
            .or_else(|| code.clone().filter(|token| !generic_error_code(token)));
        // A content-filter CODE under a 4xx is the model's verdict on the
        // content (Azure and Gemini answer 400 for it), not a request-shape
        // error: file and answer it as a refusal naming its bounded category,
        // detail kept ledger-only. Only the authoritative code decides here; a
        // sentence saying "blocked by" could be about a firewall or a limit.
        if crate::stream_errors::is_refusal_code(code.as_deref()) {
            let reason = crate::stream_errors::refusal_reason(code.as_deref(), None);
            return Err(Failure::refusal(reason)
                .with_provider_detail(detail)
                .with_rate_limit_facts(rate_limit.clone(), retry_after));
        }
        // A sentence naming a limitation of THIS lane's serving stack (a chat
        // template that rejects a mid-conversation system turn the OpenAI
        // contract allows) keeps the caller's class and detail but fails over:
        // another rung serves the same request, and only a route with no other
        // rung surfaces the 400.
        let lane_limitation = body
            .as_deref()
            .is_some_and(|body| rejected_by_lane_limitation(dialect, body));
        return Err(failure
            .with_retry(false, lane_limitation)
            .with_rejected_parameter(parameter)
            .with_provider_detail(detail));
    }
    Ok(response)
}

/// The ledger detail of one status-only classification.
fn status_detail(status: u16) -> String {
    format!("http {status}")
}

/// Longest transport cause text kept in a ledger detail.
const TRANSPORT_CAUSE_LIMIT: usize = 120;

/// The engine's own account of one connection-level failure, for the ledger.
///
/// A bare `transport` class hid what actually broke (2026-09-15: ~150
/// transport settlements a day with no cause recorded), so the failure names
/// the phase (`open` before headers, `stream` mid-body), reqwest's fault kind,
/// and the innermost cause's own sentence (an OS error such as "Connection
/// reset by peer (os error 54)", a TLS alert, hyper's "connection closed
/// before message completed"). The reqwest layer's own text is skipped
/// because it names the request URL; the cause text is then held to the same
/// identifier mask as a provider sentence and bounded, so nothing shaped like
/// a host or handle crosses. This is engine vocabulary about the engine's own
/// socket, never provider body content, and no class it rides on relays
/// detail to callers.
pub(crate) fn transport_error_detail(phase: &str, error: &reqwest::Error) -> String {
    let kind = if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "connect failed"
    } else if error.is_body() {
        "body read failed"
    } else if error.is_decode() {
        "decode failed"
    } else if error.is_request() {
        "request failed"
    } else if error.is_redirect() {
        "redirect refused"
    } else {
        "failed"
    };
    let mut cause: Option<String> = None;
    let mut source = std::error::Error::source(error);
    while let Some(inner) = source {
        cause = Some(inner.to_string());
        source = inner.source();
    }
    let mut detail = format!("{phase} {kind}");
    if let Some(cause) = cause {
        let collapsed = cause.split_whitespace().collect::<Vec<_>>().join(" ");
        let bounded: String = bounded_masked_line(&collapsed, &[])
            .chars()
            .take(TRANSPORT_CAUSE_LIMIT)
            .collect();
        if !bounded.is_empty() {
            detail.push_str(": ");
            detail.push_str(&bounded);
        }
    }
    detail
}

/// Longest provider error body read for parameter attribution.
const ERROR_BODY_READ_LIMIT: usize = 16 * 1024;

/// Bound on the whole attribution body read; a stalling error stream is
/// abandoned and the failure stays content-free.
const ERROR_BODY_READ_TIMEOUT: Duration = Duration::from_secs(2);

/// Bound on a 429 body read: only the code token is wanted, the envelope
/// normally rides in the same segment as the headers, and a throttle's
/// failover must not wait on a provider that dribbles its error body.
const THROTTLE_BODY_READ_TIMEOUT: Duration = Duration::from_millis(250);

/// Read at most `ERROR_BODY_READ_LIMIT` bytes of one error response body.
async fn bounded_error_body(mut response: reqwest::Response) -> Option<String> {
    let mut collected: Vec<u8> = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        if collected.len() + chunk.len() > ERROR_BODY_READ_LIMIT {
            return None;
        }
        collected.extend_from_slice(&chunk);
    }
    String::from_utf8(collected).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_failure_flags_mirror_the_python_taxonomy() {
        // (status, retryable_same_deployment, failover_eligible)
        let table = [
            (Some(401), false, true),
            (Some(403), false, true),
            (Some(404), false, true),
            (Some(429), false, true),
            (Some(402), false, true),
            (Some(408), true, true),
            (Some(500), true, true),
            (Some(503), true, true),
            (Some(409), true, true),
            (Some(425), true, true),
            (Some(400), false, false),
            (Some(422), false, false),
            (Some(301), false, true),
            (None, true, true),
        ];
        for (status, retryable, failover) in table {
            let failure = transport_failure(status);
            assert_eq!(
                failure.retryable_same_deployment, retryable,
                "retryable for {status:?}"
            );
            assert_eq!(
                failure.failover_eligible, failover,
                "failover for {status:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_402_with_the_literal_tokenhub_body_classes_provider_quota() {
        // TokenHub (the Tencent relay) answers HTTP 402 with provider code
        // 401008 on EVERY request shape once the account's free trial is
        // exhausted and postpaid billing is off. Classing that invalid_request
        // blamed 652 callers' request fields for the provider's billing state
        // (2026-09 incident): the class must be the provider-side quota family,
        // the rung must fail over, and the body must never be relayed.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer).await;
            let body = "{\"error\":{\"code\":401008,\"message\":\"free trial quota exhausted \
                        and postpaid billing is not enabled - enable in Console > Online \
                        Inference Service\",\"type\":\"payment_required\"}}";
            let response = format!(
                "HTTP/1.1 402 Payment Required\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let client = build_client(Duration::from_secs(2)).expect("client");
        let failure = open_stream(
            &client,
            &format!("http://{addr}/v1/chat/completions"),
            &HashMap::new(),
            "idem-402",
            &serde_json::json!({"model": "m", "messages": []}),
            None,
            Duration::from_secs(5),
            Dialect::OpenAiCompatible,
        )
        .await
        .expect_err("a 402 must classify as a failure");
        assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
        assert!(failure.failover_eligible, "an unfunded rung must fail over");
        assert!(!failure.retryable_same_deployment);
        assert!(
            failure.rejected_parameter.is_none(),
            "a billing failure must stay content-free"
        );
        // The only detail is the engine's own status token, never body text.
        assert_eq!(failure.provider_detail.as_deref(), Some("http 402"));
        assert!(!failure.public_error().message.contains("402"));
        assert!(
            !failure.safe_message.contains("request fields"),
            "the caller must never be told to fix their fields for a provider billing state"
        );
    }

    pub(super) async fn open_against_body(
        status_line: &str,
        body: &'static str,
        model: &str,
    ) -> Failure {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let status_line = status_line.to_string();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer).await;
            let response = format!(
                "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let client = build_client(Duration::from_secs(2)).expect("client");
        open_stream(
            &client,
            &format!("http://{addr}/v1/chat/completions"),
            &HashMap::new(),
            "idem-4xx",
            &serde_json::json!({"model": model, "messages": []}),
            None,
            Duration::from_secs(5),
            Dialect::OpenAiCompatible,
        )
        .await
        .expect_err("a 4xx must classify as a failure")
    }

    #[tokio::test]
    async fn a_dropped_provider_sentence_still_relays_the_provider_code() {
        // The sentence names an account handle: the handle is masked and the
        // sentence around it still reaches the caller.
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"code\":\"invalid_value\",\"type\":\"invalid_request_error\",\
             \"message\":\"Invalid value for organization org_a1b2c3d4e5f6: not allowed\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("Invalid value for organization [redacted]: not allowed")
        );
        assert_eq!(
            failure.public_error().message,
            "provider rejected the request: Invalid value for organization [redacted]: not allowed"
        );
    }

    #[tokio::test]
    async fn a_404_for_a_callers_dangling_item_reference_is_the_callers_400() {
        let failure = open_against_body(
            "404 Not Found",
            "{\"error\":{\"message\":\"Item with id 'rs_0000' not found. Items are not \
             persisted when `store` is set to false.\",\"type\":\"invalid_request_error\",\
             \"param\":\"input\",\"code\":null}}",
            "gpt-6-astra",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(
            !failure.failover_eligible,
            "every rung would answer the same"
        );
        assert_eq!(failure.public_error().status_code, 400);
        assert!(failure
            .provider_detail
            .as_deref()
            .is_some_and(|detail| detail.starts_with("Item with id 'rs_0000' not found")));
    }

    #[tokio::test]
    async fn a_vllm_flat_400_with_trailing_help_text_relays_its_sentence() {
        // Exact body captured live from an Azure Foundry DeepSeek deployment
        // (2026-09-15): 438 such 400s in 48h had settled with no detail
        // because the body is a flat vLLM object followed by a help line.
        let failure = open_against_body(
            "400 Bad Request",
            "{\"object\":\"error\",\"message\":\"Tool 'g' not found in tools list.\",\
             \"type\":\"BadRequestError\",\"param\":null,\"code\":400}\n\
             Please check this guide to understand why this error code might have been returned \n\
             https://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints#http-status-codes\n",
            "DeepSeek-V4-Flash",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("Tool 'g' not found in tools list.")
        );
        assert_eq!(
            failure.public_error().message,
            "provider rejected the request: Tool 'g' not found in tools list."
        );
    }

    #[tokio::test]
    async fn xai_and_novita_envelopes_relay_their_sentences() {
        // xAI spells the sentence as a string `error` beside a `code`
        // (captured live 2026-09-15).
        let xai = open_against_body(
            "400 Bad Request",
            "{\"code\":\"invalid-argument\",\"error\":\"Argument not supported on this \
             model: presencePenalty\"}",
            "grok-4.20-multi-agent",
        )
        .await;
        assert_eq!(xai.failure_class, FailureClass::InvalidRequest);
        assert_eq!(
            xai.provider_detail.as_deref(),
            Some("Argument not supported on this model: presencePenalty")
        );
        // Novita answers a flat gRPC-style object whose `reason` is the token
        // (captured live 2026-09-15).
        let novita = open_against_body(
            "400 Bad Request",
            "{\"code\":400,\"reason\":\"INVALID_PARAMETER\",\"message\":\"tools is not \
             supported by this model\",\"metadata\":{}}",
            "deepseek/deepseek-v4.1-flash",
        )
        .await;
        assert_eq!(novita.failure_class, FailureClass::InvalidRequest);
        assert_eq!(
            novita.provider_detail.as_deref(),
            Some("tools is not supported by this model")
        );
    }

    #[tokio::test]
    async fn status_only_classifications_carry_the_status_as_ledger_detail() {
        // A 503 relay fault and a 500 model fault were indistinguishable on
        // the ledger; the status token now rides as the detail, ledger-only.
        let failure = open_against_body(
            "503 Service Unavailable",
            "{\"error\":{\"message\":\"upstream connect error to 10.0.0.7\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
        assert_eq!(failure.provider_detail.as_deref(), Some("http 503"));
        assert_eq!(
            failure.public_error().message,
            "provider service failed; retry after a short delay",
            "a server-side class never relays detail to the caller"
        );
        let throttled = open_against_body("429 Too Many Requests", "", "m").await;
        assert_eq!(throttled.failure_class, FailureClass::Throttled);
        assert_eq!(throttled.provider_detail.as_deref(), Some("http 429"));
    }

    #[tokio::test]
    async fn a_refused_connection_names_the_transport_fault_for_the_ledger() {
        // Bind then drop the listener so the port refuses the connect.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        drop(listener);
        let client = build_client(Duration::from_secs(2)).expect("client");
        let failure = open_stream(
            &client,
            &format!("http://{addr}/v1/chat/completions"),
            &HashMap::new(),
            "idem-refused",
            &serde_json::json!({"model": "m", "messages": []}),
            None,
            Duration::from_secs(5),
            Dialect::OpenAiCompatible,
        )
        .await
        .expect_err("a refused connect is a failure");
        assert_eq!(failure.failure_class, FailureClass::Transport);
        let detail = failure
            .provider_detail
            .as_deref()
            .expect("transport detail");
        assert!(detail.starts_with("open connect failed"), "{detail}");
        assert!(
            !detail.contains("127.0.0.1") && !detail.contains(&addr.port().to_string()),
            "the socket address never rides in the detail: {detail}"
        );
        assert_eq!(
            failure.public_error().message,
            "provider transport failed; retry the request"
        );
    }

    #[tokio::test]
    async fn a_404_naming_a_missing_model_keeps_the_lane_policy() {
        let failure = open_against_body(
            "404 Not Found",
            "{\"error\":{\"message\":\"The model `x` does not exist or you do not have \
             access to it.\",\"type\":\"invalid_request_error\",\"param\":\"model\",\
             \"code\":\"model_not_found\"}}",
            "x",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
        assert!(failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_bodiless_404_keeps_the_lane_policy() {
        let failure = open_against_body("404 Not Found", "", "m").await;
        assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
        assert!(failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_blocked_by_sentence_without_a_refusal_code_stays_a_request_error() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"code\":\"invalid_value\",\"message\":\"Request blocked by the \
             organization policy for this parameter.\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    }

    #[tokio::test]
    async fn a_content_filter_4xx_is_a_refusal_not_a_request_shape_error() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"code\":\"content_filter\",\"message\":\"The response was \
             filtered due to the prompt triggering the content management policy.\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert_eq!(failure.public_error().status_code, 400);
        assert_eq!(failure.public_error().code, "refusal");
        // The content_filter code names the content-policy category.
        assert_eq!(
            failure.refusal_reason,
            Some(crate::errors::RefusalReason::ContentPolicy)
        );
        // The sanitized sentence (or the code token when it must drop) rides
        // to the ledger; a refusal never relays it to the caller.
        assert!(failure.provider_detail.is_some());
        assert_eq!(
            failure.public_error().message,
            "provider refused the request: content policy"
        );
    }

    #[tokio::test]
    async fn an_azure_completion_finished_content_filter_under_a_400_is_a_refusal() {
        // Captured live from Azure AI Foundry (DeepSeek-V4-Flash, 2026-09-15):
        // a 400 carrying a chat.completion body and no error envelope.
        let failure = open_against_body(
            "400 Bad Request",
            "{\"id\":\"chatcmpl-802d5a802bf84292896e446052595\",\"model\":\"\",\"choices\":\
             [{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\"},\
             \"finish_reason\":\"content_filter\",\"content_filter_results\":{\"error\":\
             {\"code\":\"content_filter\",\"message\":\"Response content blocked by label \
             'MultiSeverity_ViolenceScore'.\"}}}],\"usage\":{\"prompt_tokens\":55,\
             \"total_tokens\":55},\"created\":1789466192,\"object\":\"chat.completion\",\
             \"prompt_filter_results\":null}",
            "DeepSeek-V4-Flash",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert_eq!(
            failure.refusal_reason,
            Some(crate::errors::RefusalReason::ContentPolicy)
        );
        assert!(!failure.failover_eligible);
        assert!(!failure.retryable_same_deployment);
        // The quoted label is caller-visible vocabulary, so the sentence
        // survives the identifier screen into the ledger; the caller still
        // gets only the bounded refusal.
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("Response content blocked by label 'MultiSeverity_ViolenceScore'.")
        );
        assert_eq!(failure.public_error().status_code, 400);
        assert_eq!(failure.public_error().code, "refusal");
    }

    #[tokio::test]
    async fn an_aggregator_routing_gate_403_is_not_a_credential_failure() {
        let failure = open_against_body(
            "403 Forbidden",
            "{\"error\":{\"message\":\"thinkingmachines/inkling:free is only available \
             on agentic harnesses.\",\"code\":403,\"metadata\":{\"routing_funnel\":\
             [{\"step\":\"Initial Endpoints\",\"endpoint_count\":1}],\
             \"failed_routing_step\":\"Gate Free Endpoints by Agentic Harness\"}}}",
            "thinkingmachines/inkling:free",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
        assert!(failure.failover_eligible);
        assert!(!failure.retryable_same_deployment);
        assert!(failure.provider_detail.is_none());
    }

    #[tokio::test]
    async fn a_plain_403_stays_a_credential_failure() {
        let failure = open_against_body(
            "403 Forbidden",
            "{\"error\":{\"message\":\"Forbidden\",\"code\":403}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
        assert!(failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_lane_limitation_400_keeps_the_class_but_fails_over() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"message\":\"System message must be at the beginning.\",\
             \"type\":\"invalid_request_error\"}}",
            "qwen3.8-27b",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(
            failure.failover_eligible,
            "another rung can carry the request"
        );
        assert!(!failure.retryable_same_deployment);
        assert!(failure
            .provider_detail
            .as_deref()
            .is_some_and(|detail| detail.contains("System message must be at the beginning")));
    }

    #[tokio::test]
    async fn a_lane_limitation_phrase_echoed_outside_the_message_does_not_fail_over() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"message\":\"Invalid value for temperature.\",\
             \"type\":\"invalid_request_error\",\"param\":\"temperature\",\
             \"echo\":\"System message must be at the beginning\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(!failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_403_naming_a_step_without_a_walked_funnel_stays_a_credential_failure() {
        let failure = open_against_body(
            "403 Forbidden",
            "{\"error\":{\"message\":\"Forbidden\",\"code\":403,\
             \"metadata\":{\"failed_routing_step\":\"Authenticate\"}}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
    }

    #[tokio::test]
    async fn an_ordinary_400_does_not_fail_over() {
        let failure = open_against_body(
            "400 Bad Request",
            "{\"error\":{\"message\":\"Invalid value for temperature.\",\
             \"type\":\"invalid_request_error\"}}",
            "m",
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
        assert!(!failure.failover_eligible);
    }

    #[tokio::test]
    async fn a_zai_429_is_quota_for_code_1113_and_a_throttle_otherwise() {
        // Z.ai: 429 + code 1113 is an empty balance (quota, fails over); 1302 a rate limit.
        let quota = open_against_body(
            "429 Too Many Requests",
            "{\"error\":{\"code\":\"1113\",\"message\":\"Insufficient balance or no \
             resource package. Please recharge.\"}}",
            "glm-4.6",
        )
        .await;
        assert_eq!(quota.failure_class, FailureClass::ProviderQuota);
        assert!(quota.failover_eligible && !quota.retryable_same_deployment);
        assert!(quota.provider_detail.is_none() && quota.rejected_parameter.is_none());
        let limit = open_against_body(
            "429 Too Many Requests",
            "{\"error\":{\"code\":\"1302\",\"message\":\"Rate limit reached for requests\"}}",
            "glm-4.6",
        )
        .await;
        assert_eq!(limit.failure_class, FailureClass::Throttled);
    }

    #[test]
    fn header_phase_timeout_fails_over_without_a_same_deployment_redial() {
        // A lead that connects but never completes the response-header phase must
        // skip straight to the next rung (failover-eligible) instead of redialing
        // the same stalled deployment for another full header-timeout window.
        let failure = open_timeout_failure();
        assert_eq!(failure.failure_class, FailureClass::Timeout);
        assert!(
            !failure.retryable_same_deployment,
            "a header-phase stall must not redial the same deployment"
        );
        assert!(
            failure.failover_eligible,
            "a header-phase stall must fail over to the next certified rung"
        );
    }
}

#[cfg(test)]
#[path = "upstream_reseller_tests.rs"]
mod reseller_tests;
