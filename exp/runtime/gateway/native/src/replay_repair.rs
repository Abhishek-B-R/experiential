//! The one repair the waterfall performs on a caller's replayed input.
//!
//! OpenAI binds a Responses reasoning item's `encrypted_content` to the
//! organization (Azure: the tenant) that sealed it and refuses every other
//! replay with `invalid_encrypted_content`. The waterfall re-dials the same
//! rung once with those items stripped and remembers the stripped payload
//! for the rest of the request's ladder on that rung; this module owns the
//! strip, the caller-facing disclosure header, and the operator line.
//!
//! Telemetry: the refused dial and its re-dial run under ONE reservation, so
//! the ledger records one attempt and never the refusal. What does record it:
//! the `x-gateway-replay-repair` header on HTTP responses (the WebSocket
//! Responses transport carries no response headers), the data-plane counter
//! `encrypted_reasoning_stripped`, and one content-free operator line, each
//! written only once the re-dial has actually opened. A stateless caller
//! keeps the foreign items in its history, so the repair recurs on every
//! later turn of that conversation until the caller drops them.

use serde_json::{json, Value};

use crate::waterfall::DeploymentWire;

/// Response header disclosing a data-plane repair of the caller's replayed
/// input on the attempt that served.
pub const REPLAY_REPAIR_HEADER: &str = "x-gateway-replay-repair";

/// The one repair the waterfall performs: the replayed reasoning items whose
/// `encrypted_content` the rung refused were stripped and the rung re-dialed.
pub const ENCRYPTED_REASONING_STRIPPED: &str = "encrypted_reasoning_stripped";

/// The disclosure header pairs of one served attempt; empty when the input
/// reached the provider exactly as replayed.
pub fn replay_repair_headers(encrypted_reasoning_stripped: bool) -> Vec<(String, String)> {
    if encrypted_reasoning_stripped {
        vec![(
            REPLAY_REPAIR_HEADER.to_string(),
            ENCRYPTED_REASONING_STRIPPED.to_string(),
        )]
    } else {
        Vec::new()
    }
}

/// The Responses payload with every replayed reasoning item that carries
/// `encrypted_content` removed, or `None` when there is nothing to strip.
///
/// OpenAI binds an encrypted reasoning payload to the organization (Azure:
/// the tenant) that sealed it and refuses every other with
/// `invalid_encrypted_content`, so a conversation whose earlier turn another
/// lane served, or whose history the caller assembled from another account,
/// cannot replay those items here. The item is optional on replay: the model
/// resumes from the visible message, tool-call, and tool-result items, so
/// dropping it trades hidden reasoning continuity for a served turn.
///
/// The assistant items a stripped reasoning item governed (the function and
/// custom tool calls and the assistant message of that same provider turn,
/// up to the next user or tool-result item) lose their provider `id` as
/// well: OpenAI ties a replayed item id to the reasoning item of its turn
/// and refuses the id without it ("was provided without its required
/// 'reasoning' item"), while an id-less item is accepted as caller-authored.
/// Every kept item keeps its position; nothing outside `input` changes.
pub fn without_encrypted_reasoning(payload: &Value) -> Option<Value> {
    let mut repaired = payload.clone();
    let items = repaired.get_mut("input")?.as_array_mut()?;
    let before = items.len();
    let mut keep: Vec<bool> = Vec::with_capacity(before);
    let mut governed = false;
    for item in items.iter_mut() {
        let kind = item.get("type").and_then(Value::as_str);
        let role = item.get("role").and_then(Value::as_str);
        if kind == Some("reasoning")
            && item
                .get("encrypted_content")
                .is_some_and(|content| !content.is_null())
        {
            keep.push(false);
            governed = true;
            continue;
        }
        let same_turn_output = matches!(kind, Some("function_call") | Some("custom_tool_call"))
            || (matches!(kind, Some("message") | None) && role == Some("assistant"))
            || kind == Some("reasoning");
        if governed && same_turn_output {
            if let Some(object) = item.as_object_mut() {
                object.remove("id");
            }
        } else {
            governed = false;
        }
        keep.push(true);
    }
    if keep.iter().all(|kept| *kept) {
        return None;
    }
    let mut index = 0;
    items.retain(|_| {
        let kept = keep[index];
        index += 1;
        kept
    });
    Some(repaired)
}

/// Emit the content-free operator line for one stripped re-dial.
pub fn log_encrypted_reasoning_stripped(request_id: &str, wire: &DeploymentWire) {
    let line = json!({
        "event": "encrypted_reasoning_stripped",
        "request_id": request_id,
        "provider": wire.provider,
        "deployment_id": wire.deployment_id,
    });
    eprintln!("exp-gateway-native: {line}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripping_encrypted_reasoning_removes_only_those_items_and_keeps_order() {
        let payload = json!({
            "model": "gpt-test",
            "store": false,
            "input": [
                {"role": "user", "content": "plan"},
                // A call from a turn whose reasoning was never replayed keeps its id.
                {"type": "function_call", "id": "fc_0", "call_id": "call_0", "name": "ls", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_0", "output": "."},
                {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "rsn_foreign=="},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
                {"type": "reasoning", "id": "rs_b", "summary": [], "encrypted_content": "rsn_foreign_2=="},
                {"type": "message", "id": "msg_9", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "done"}]},
                // A reasoning item replayed by id alone is not an encrypted payload.
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "reasoning", "summary": [], "encrypted_content": null},
                {"role": "user", "content": "continue"},
                {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "exec", "arguments": "{}"},
            ],
            "include": ["reasoning.encrypted_content"],
        });
        let repaired = without_encrypted_reasoning(&payload).expect("two items to strip");
        let kept: Vec<&Value> = repaired["input"]
            .as_array()
            .expect("input")
            .iter()
            .collect();
        assert_eq!(kept.len(), 10);
        assert!(kept
            .iter()
            .all(|item| item.get("encrypted_content").is_none_or(Value::is_null)));
        assert_eq!(kept[0]["content"], "plan");
        // Governed by no stripped reasoning item: the id survives.
        assert_eq!(kept[1]["id"], "fc_0");
        // The call and the message of the stripped turns lose their ids and
        // keep everything else.
        assert_eq!(kept[3]["type"], "function_call");
        assert!(kept[3].get("id").is_none());
        assert_eq!(kept[3]["call_id"], "call_1");
        assert_eq!(kept[5]["type"], "message");
        assert!(kept[5].get("id").is_none());
        assert_eq!(kept[5]["status"], "completed");
        // The by-id reasoning item of the same turn also loses its id; the
        // null-content one is kept as sent.
        assert!(kept[6].get("id").is_none());
        assert_eq!(kept[7]["encrypted_content"], Value::Null);
        // A later turn with no stripped reasoning keeps its ids.
        assert_eq!(kept[8]["content"], "continue");
        assert_eq!(kept[9]["id"], "fc_2");
        // Everything beside the input is untouched.
        assert_eq!(repaired["include"], payload["include"]);
        assert_eq!(repaired["store"], false);

        // Nothing to strip: no repair, so the provider's verdict surfaces.
        assert!(without_encrypted_reasoning(&repaired).is_none());
        assert!(without_encrypted_reasoning(&json!({"model": "m", "input": "text"})).is_none());
        assert!(without_encrypted_reasoning(&json!({"model": "m", "messages": []})).is_none());
    }

    #[test]
    fn the_repair_header_appears_only_on_a_repaired_attempt() {
        assert!(replay_repair_headers(false).is_empty());
        assert_eq!(
            replay_repair_headers(true),
            vec![(
                "x-gateway-replay-repair".to_string(),
                "encrypted_reasoning_stripped".to_string(),
            )]
        );
    }
}
