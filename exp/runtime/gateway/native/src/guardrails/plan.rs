//! Native execution of a deterministic-only output guardrail chain.
//!
//! The control plane resolves the per-identity policy once at admission and
//! hands the ordered chain over on the admission wire whenever every check
//! binds a deterministic detector. Rust then inspects and redacts the
//! buffered completion in place, so a guarded request pays no python
//! callback, no GIL acquisition, and no JSON round trip of the completion.
//!
//! A chain with any non-deterministic adapter is not planned here: it keeps
//! crossing the python boundary unchanged.
//!
//! This module never logs completions, matches, or replacements.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass};
use crate::events::Event;
use crate::guardrails::detector::Detector;

/// Compiled deterministic detectors, keyed by policy `adapter_id`.
pub type DetectorMap = HashMap<String, Arc<Detector>>;

/// One resolved deterministic check in an output chain.
#[derive(Debug, Clone, Deserialize)]
pub struct PlanCheck {
    pub action: String,
    pub adapter_id: String,
}

/// The resolved deterministic output chain for one admitted request.
#[derive(Debug, Clone, Deserialize)]
pub struct OutputPlan {
    #[serde(default)]
    pub protected: bool,
    pub max_response_bytes: usize,
    pub checks: Vec<PlanCheck>,
}

/// The buffered completion projection an output check inspects.
struct Completion {
    text: String,
    refusal: bool,
    tool_calls: Vec<ToolCall>,
}

/// One completed tool invocation presented to an output check.
struct ToolCall {
    call_id: String,
    name: String,
    arguments: String,
}

/// Build one fail-closed guardrail failure without completion content.
fn error_failure() -> Failure {
    Failure::new(
        FailureClass::Guardrail,
        "A gateway guardrail could not complete this request.",
    )
}

/// Build the blocked-by-policy failure without completion content.
fn block_failure() -> Failure {
    Failure::new(
        FailureClass::Guardrail,
        "The request was blocked by a gateway guardrail.",
    )
}

/// Project collected events into the inspected completion.
fn projection(events: &[Event]) -> Completion {
    let mut completion = Completion {
        text: String::new(),
        refusal: false,
        tool_calls: Vec::new(),
    };
    for event in events {
        match event {
            Event::TextDelta(delta) | Event::ProviderTextDelta { delta, .. } => {
                completion.text.push_str(delta);
            }
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. } => {
                completion.refusal = true;
            }
            Event::ToolCallCompleted { call, .. } => completion.tool_calls.push(ToolCall {
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                arguments: call.raw_arguments.clone(),
            }),
            _ => {}
        }
    }
    completion
}

/// The UTF-8 size of the canonical classifier subject.
///
/// This mirrors `GuardrailCompletion.content_bytes`: deterministic JSON with
/// sorted keys, no insignificant whitespace, and no ASCII escaping, so the
/// native bound admits and rejects exactly what the python bound does.
fn content_bytes(completion: &Completion) -> usize {
    let calls: Vec<Value> = completion
        .tool_calls
        .iter()
        .map(|call| {
            json!({
                "arguments": call.arguments,
                "call_id": call.call_id,
                "name": call.name,
            })
        })
        .collect();
    let subject = json!({
        "refusal": completion.refusal,
        "text": completion.text,
        "tool_calls": calls,
    });
    crate::encode::compact_json(&subject).len()
}

/// Apply the fail-closed rule for one uncertain check.
///
/// Returns `Ok(())` when a non-protected identity skips the check and
/// continues the remaining chain.
fn uncertain(plan: &OutputPlan) -> Result<(), Failure> {
    if plan.protected {
        return Err(error_failure());
    }
    Ok(())
}

/// Run one deterministic output chain over the buffered events.
///
/// `deadline` is the request-wide deadline already in force on this route.
/// An expired deadline is an uncertain check, exactly as in the python
/// chain.
pub fn enforce(
    plan: &OutputPlan,
    detectors: &DetectorMap,
    events: Vec<Event>,
    deadline: Instant,
) -> Result<Vec<Event>, Failure> {
    let completion = projection(&events);
    if content_bytes(&completion) > plan.max_response_bytes {
        return Err(error_failure());
    }
    let mut text = completion.text;
    let mut rewritten = false;
    for check in &plan.checks {
        if deadline.saturating_duration_since(Instant::now()) == Duration::ZERO {
            uncertain(plan)?;
            continue;
        }
        let Some(detector) = detectors.get(&check.adapter_id) else {
            uncertain(plan)?;
            continue;
        };
        let redacted = match detector.redact(&text) {
            Ok(value) => value,
            Err(_) => {
                uncertain(plan)?;
                continue;
            }
        };
        let mut flagged = redacted.is_some();
        let mut limited = false;
        for call in &completion.tool_calls {
            match detector.matches(&call.arguments) {
                Ok(found) => flagged |= found,
                Err(_) => limited = true,
            }
        }
        if limited {
            uncertain(plan)?;
            continue;
        }
        if !flagged {
            continue;
        }
        match check.action.as_str() {
            "allow" => {}
            "modify" => {
                // Tool-call arguments are inspected but never rewritten, so a
                // completion that carries one cannot be modified.
                if !completion.tool_calls.is_empty() {
                    return Err(block_failure());
                }
                if let Some(value) = redacted {
                    text = value;
                    rewritten = true;
                }
            }
            "block" => return Err(block_failure()),
            _ => return Err(error_failure()),
        }
    }
    if !rewritten {
        return Ok(events);
    }
    Ok(crate::guardrails::apply_text_replacement(&events, &text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;
    use crate::guardrails::detector::DetectorSpec;

    /// Build one detector map with a single email rule under `adapter_id`.
    fn detectors(adapter_id: &str, replacement: &str) -> DetectorMap {
        let detector = Detector::compile(&DetectorSpec {
            patterns: vec![],
            builtin_patterns: vec!["email".to_string()],
            replacement: replacement.to_string(),
        })
        .expect("the rule compiles");
        HashMap::from([(adapter_id.to_string(), Arc::new(detector))])
    }

    /// Build one single-check plan with the given action.
    fn plan(action: &str, protected: bool) -> OutputPlan {
        OutputPlan {
            protected,
            max_response_bytes: 1_048_576,
            checks: vec![PlanCheck {
                action: action.to_string(),
                adapter_id: "pii".to_string(),
            }],
        }
    }

    /// One future deadline that never expires during a unit test.
    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(30)
    }

    /// Build one completed tool call event.
    fn tool_event(arguments: &str) -> Event {
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: arguments.to_string(),
                custom: false,
            },
        }
    }

    #[test]
    fn a_clean_completion_is_returned_unchanged() {
        let events = vec![Event::TextDelta("all clear".to_string()), Event::Completed];
        let result = enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect("the chain allows");
        assert!(matches!(result[0], Event::TextDelta(ref text) if text == "all clear"));
    }

    #[test]
    fn a_modify_check_redacts_the_buffered_text() {
        let events = vec![
            Event::TextDelta("mail ada@example.com".to_string()),
            Event::Completed,
        ];
        let result = enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect("the chain modifies");
        assert!(matches!(result[0], Event::TextDelta(ref text) if text == "mail [REDACTED]"));
    }

    #[test]
    fn a_block_check_fails_the_request() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let failure = enforce(
            &plan("block", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect_err("the chain blocks");
        assert_eq!(failure.failure_class, FailureClass::Guardrail);
    }

    #[test]
    fn a_tool_call_match_is_blocked_rather_than_rewritten() {
        let events = vec![
            Event::TextDelta("see attachment".to_string()),
            tool_event("{\"to\":\"ada@example.com\"}"),
        ];
        let failure = enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            deadline(),
        )
        .expect_err("a tool completion cannot be rewritten");
        assert_eq!(
            failure.safe_message,
            "The request was blocked by a gateway guardrail."
        );
    }

    #[test]
    fn an_oversized_completion_fails_closed() {
        let events = vec![Event::TextDelta("hello there".to_string())];
        let mut small = plan("modify", false);
        small.max_response_bytes = 8;
        let failure = enforce(&small, &detectors("pii", "[REDACTED]"), events, deadline())
            .expect_err("the subject exceeds the policy bound");
        assert_eq!(
            failure.safe_message,
            "A gateway guardrail could not complete this request."
        );
    }

    #[test]
    fn a_missing_adapter_fails_closed_only_when_protected() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let empty: DetectorMap = HashMap::new();
        assert!(enforce(&plan("modify", true), &empty, events.clone(), deadline()).is_err());
        let skipped = enforce(&plan("modify", false), &empty, events, deadline())
            .expect("a non protected identity skips the check");
        assert!(matches!(skipped[0], Event::TextDelta(ref text) if text == "ada@example.com"));
    }

    #[test]
    fn an_expired_deadline_is_an_uncertain_check() {
        let events = vec![Event::TextDelta("ada@example.com".to_string())];
        let expired = Instant::now() - Duration::from_secs(1);
        assert!(enforce(
            &plan("modify", true),
            &detectors("pii", "[REDACTED]"),
            events,
            expired,
        )
        .is_err());
    }
}
