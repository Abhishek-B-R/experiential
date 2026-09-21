//! Anthropic Messages frame mapping: usage legs accumulate across the
//! message lifecycle and fold at `message_stop`, refusal blocks and refusal
//! stop reasons mark the stream, and extended-thinking blocks normalize to
//! dedicated thinking events so callers receive the reasoning they pay for.

use serde_json::Value;

mod cache_write;

use super::{
    finish_open_tools, finish_open_tools_truncated, malformed, optional_text, parse_object,
    refusal_failure, Normalizer,
};
use crate::encode::compact_json;
use crate::errors::Failure;
use crate::events::{
    bounded_ledger_sum, count_if_present, count_or_zero, require_string, require_u64, Event,
    ToolAccumulator, Usage,
};

impl Normalizer {
    /// Fold the latest decoded Anthropic counters into the shared usage shape.
    fn anthropic_usage(&self) -> Result<Usage, Failure> {
        let input_tokens = self
            .input_tokens
            .map(|fresh| {
                bounded_ledger_sum(
                    &[fresh, self.cache_read, self.cache_write],
                    "Anthropic input",
                )
            })
            .transpose()
            .map_err(|message| malformed(&message))?;
        Ok(Usage {
            input_tokens,
            output_tokens: self.output_tokens,
            cached_input_tokens: Some(self.cache_read),
            // Keep the cache-less wire shape; thinking is billed inside output
            // and has no provider-reported subset to expose here.
            cache_creation_input_tokens: (self.cache_write > 0).then_some(self.cache_write),
            cache_creation_1h_input_tokens: self.cache_write_1h.filter(|_| self.cache_write > 0),
            reasoning_tokens: None,
        })
    }

    pub(super) fn feed_anthropic(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| frame.event.clone())
            .unwrap_or_default();
        let mut events = Vec::new();
        match event_type.as_str() {
            "message_start" => {
                let message = payload
                    .get("message")
                    .and_then(Value::as_object)
                    .ok_or_else(|| {
                        malformed("Anthropic message_start.message must be an object")
                    })?;
                let usage = message
                    .get("usage")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message_start.usage must be an object"))?;
                // Primary omissions are unknown. Optional cache legs use the
                // provider's zero-when-omitted convention.
                self.input_tokens = count_if_present(usage, "input_tokens", "Anthropic usage")
                    .map_err(|message| malformed(&message))?;
                self.cache_read = count_or_zero(
                    usage,
                    "cache_read_input_tokens",
                    "Anthropic cache_read_input_tokens",
                )
                .map_err(|message| malformed(&message))?;
                self.cache_write = count_or_zero(
                    usage,
                    "cache_creation_input_tokens",
                    "Anthropic cache_creation_input_tokens",
                )
                .map_err(|message| malformed(&message))?;
                self.cache_write_1h = cache_write::hour_subset(usage, self.cache_write)?;
                self.output_tokens = self.output_tokens.max(
                    count_if_present(usage, "output_tokens", "Anthropic usage")
                        .map_err(|message| malformed(&message))?,
                );
                // Surface the start-frame meters at once: the Messages encoder
                // mirrors them on its own `message_start` (Claude Code reads
                // the input legs there), and the settlement tracker holds them
                // as the best known count until the terminal report, which
                // supersedes them at `message_stop` (server-tool turns re-read
                // fetched results as input, so the start count undercounts).
                let usage = self.anthropic_usage()?;
                self.usage = Some(usage.clone());
                events.push(Event::Usage(usage));
            }
            "content_block_start" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                let block = payload
                    .get("content_block")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic content block must be an object"))?;
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let call_id = require_string(block, "id", "Anthropic tool ID")
                            .map_err(|message| malformed(&message))?;
                        let name = require_string(block, "name", "Anthropic tool name")
                            .map_err(|message| malformed(&message))?;
                        if self.tools.contains_key(&index) {
                            return Err(malformed("Anthropic stream repeated a tool-call start"));
                        }
                        self.reserve_tool_entry(index)?;
                        self.tools
                            .insert(index, ToolAccumulator::new(call_id.clone(), name.clone()));
                        events.push(Event::ToolCallStarted {
                            custom: false,
                            index,
                            call_id,
                            name,
                            namespace: None,
                            caller: None,
                        });
                    }
                    Some("server_tool_use") => {
                        // Provider-executed server tool (web search): same
                        // start/argument lifecycle as a client tool, but on
                        // dedicated events so it never becomes client tool
                        // history or a tool_use stop reason.
                        let call_id = require_string(block, "id", "Anthropic server tool ID")
                            .map_err(|message| malformed(&message))?;
                        let name = require_string(block, "name", "Anthropic server tool name")
                            .map_err(|message| malformed(&message))?;
                        if self.tools.contains_key(&index) {
                            return Err(malformed("Anthropic stream repeated a tool-call start"));
                        }
                        self.reserve_tool_entry(index)?;
                        let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
                        tool.server = true;
                        self.tools.insert(index, tool);
                        events.push(Event::ServerToolUseStarted {
                            index,
                            call_id,
                            name,
                        });
                    }
                    Some(block_type) if block_type.ends_with("_tool_result") => {
                        // A server tool's result (`web_search_tool_result`,
                        // `tool_search_tool_result`, ...) arrives whole in the
                        // start frame and is carried verbatim so the caller
                        // (and its next-turn echo) sees exactly what the
                        // provider produced.
                        let serialized = compact_json(&Value::Object(block.clone()));
                        self.reserve_tool_bytes(serialized.len())?;
                        events.push(Event::ServerToolResult {
                            index,
                            block: serialized,
                        });
                    }
                    Some("text") => {
                        // The boundary event lets the Messages encoder mirror
                        // the provider's text-block structure, which is what
                        // citations attach to.
                        events.push(Event::TextBlockStarted { index });
                        let text = optional_text(block, "text", "Anthropic initial text")?;
                        if !text.is_empty() {
                            events.push(Event::TextDelta(text));
                        }
                    }
                    Some("refusal") => {
                        self.refusal_seen = true;
                        events.push(Event::RefusalDelta(optional_text(
                            block,
                            "refusal",
                            "Anthropic refusal",
                        )?));
                    }
                    Some("thinking") => {
                        let text = optional_text(block, "thinking", "Anthropic initial thinking")?;
                        if !text.is_empty() {
                            events.push(Event::ThinkingDelta { index, delta: text });
                        }
                    }
                    Some("redacted_thinking") => {
                        // Redacted thinking arrives whole in the start frame.
                        let data = optional_text(block, "data", "Anthropic redacted thinking")?;
                        events.push(Event::RedactedThinking { index, data });
                    }
                    // Unknown block kinds with no gateway-visible output are
                    // skipped rather than rejected.
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                let delta = payload
                    .get("delta")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic content delta must be an object"))?;
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = optional_text(delta, "text", "Anthropic text delta")?;
                        if !text.is_empty() {
                            events.push(Event::TextDelta(text));
                        }
                    }
                    Some("input_json_delta") => {
                        let fragment =
                            optional_text(delta, "partial_json", "Anthropic argument delta")?;
                        self.reserve_tool_bytes(fragment.len())?;
                        let tool = self.tools.get_mut(&index).ok_or_else(|| {
                            malformed("provider emitted arguments before a tool start")
                        })?;
                        tool.raw_arguments.push_str(&fragment);
                        events.push(if tool.server {
                            Event::ServerToolArgumentsDelta {
                                index,
                                delta: fragment,
                            }
                        } else {
                            Event::ToolArgumentsDelta {
                                index,
                                delta: fragment,
                            }
                        });
                    }
                    Some("citations_delta") => {
                        // One whole citation object attached to the open text
                        // block, carried verbatim (server-tool answers cite
                        // their web sources through these).
                        let citation = delta
                            .get("citation")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                malformed("Anthropic citations_delta.citation must be an object")
                            })?;
                        let serialized = compact_json(&Value::Object(citation.clone()));
                        self.reserve_tool_bytes(serialized.len())?;
                        events.push(Event::CitationDelta {
                            index,
                            citation: serialized,
                        });
                    }
                    Some("refusal_delta") => {
                        self.refusal_seen = true;
                        events.push(Event::RefusalDelta(optional_text(
                            delta,
                            "refusal",
                            "Anthropic refusal delta",
                        )?));
                    }
                    Some("thinking_delta") => {
                        let text = optional_text(delta, "thinking", "Anthropic thinking delta")?;
                        if !text.is_empty() {
                            events.push(Event::ThinkingDelta { index, delta: text });
                        }
                    }
                    Some("signature_delta") => {
                        let signature =
                            optional_text(delta, "signature", "Anthropic signature delta")?;
                        if !signature.is_empty() {
                            events.push(Event::ThinkingSignature { index, signature });
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                if let Some(mut tool) = self.tools.remove(&index) {
                    if !tool.completed {
                        // The stop reason arrives in the following
                        // message_delta, so a fragment left open by the
                        // output budget cannot be told from garbage yet.
                        self.complete_tool_deferring_failure(index, &mut tool, &mut events);
                    }
                    self.tools.insert(index, tool);
                }
            }
            "message_delta" => {
                let delta = payload
                    .get("delta")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message delta must be an object"))?;
                if let Some(Value::String(reason)) = delta.get("stop_reason") {
                    self.stop_reason = Some(reason.clone());
                }
                let usage = payload
                    .get("usage")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message_delta.usage must be an object"))?;
                self.output_tokens = self.output_tokens.max(
                    count_if_present(usage, "output_tokens", "Anthropic usage")
                        .map_err(|message| malformed(&message))?,
                );
                // The terminal usage report supersedes message_start when its
                // input legs are present: server-tool turns re-read fetched
                // results as input, so the start-frame count undercounts the
                // billed total severely (verified live 2026-08-31).
                self.input_tokens = self.input_tokens.max(
                    count_if_present(usage, "input_tokens", "Anthropic message_delta")
                        .map_err(|message| malformed(&message))?,
                );
                for (key, slot) in [
                    ("cache_read_input_tokens", &mut self.cache_read),
                    ("cache_creation_input_tokens", &mut self.cache_write),
                ] {
                    if let Some(value) = count_if_present(usage, key, "Anthropic message_delta")
                        .map_err(|message| malformed(&message))?
                    {
                        *slot = value;
                    }
                }
                if usage.contains_key("cache_creation_input_tokens")
                    || usage.contains_key("cache_creation")
                {
                    self.cache_write_1h = cache_write::hour_subset(usage, self.cache_write)?;
                }
                // The relay may be cancelled before message_stop arrives.
                // Retain these decoded meters without changing event timing.
                self.usage = Some(self.anthropic_usage()?);
                if self.stop_reason.as_deref() == Some("refusal") && !self.refusal_seen {
                    self.refusal_seen = true;
                    events.push(Event::RefusalDelta(String::new()));
                }
            }
            "message_stop" => {
                let truncated = self.stop_reason.as_deref() == Some("max_tokens");
                self.resolve_deferred_tool_failure(truncated)?;
                if self.refusal_seen
                    || !matches!(
                        self.stop_reason.as_deref(),
                        Some("end_turn" | "stop_sequence" | "tool_use" | "pause_turn")
                    )
                {
                    // No successful final reason authorized a zero-argument
                    // call. Preserve the start, without inventing its input.
                    for tool in self.tools.values_mut() {
                        if !tool.custom && tool.raw_arguments.is_empty() {
                            tool.completed = true;
                        }
                    }
                }
                events.extend(if truncated {
                    finish_open_tools_truncated(&mut self.tools)?
                } else {
                    finish_open_tools(&mut self.tools)?
                });
                events.push(Event::Usage(self.anthropic_usage()?));
                if self.refusal_seen || self.stop_reason.as_deref() == Some("refusal") {
                    events.push(Event::Failed(refusal_failure()));
                } else if self.stop_reason.as_deref() == Some("max_tokens") {
                    events.push(Event::Incomplete);
                } else if self.stop_reason.as_deref() == Some("pause_turn") {
                    // A paused server-tool turn must keep its stop reason:
                    // the caller resumes it by resending the conversation,
                    // and an end_turn rewrite would end the task instead.
                    events.push(Event::PausedTurn);
                } else if self.dropped_cut_call {
                    // A call cut mid-fragment under a non-truncating stop
                    // reason was dropped at its block stop.
                    events.push(Event::Incomplete);
                } else {
                    events.push(Event::Completed);
                }
            }
            "error" => {
                // The provider names its failure mechanism only inside this
                // frame; the bounded detail rides the failure into the ledger.
                let (code, message) = match payload.get("error").and_then(Value::as_object) {
                    Some(error) => (
                        error.get("type").and_then(Value::as_str),
                        error.get("message").and_then(Value::as_str),
                    ),
                    None => (None, None),
                };
                events.push(Event::Failed(self.provider_stream_failure(
                    "anthropic_messages",
                    code,
                    message,
                )));
            }
            "ping" => {}
            _ => {
                return Err(malformed(&format!(
                    "Anthropic stream emitted an unsupported event (type {})",
                    super::bounded_wire_token(&event_type),
                )));
            }
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use crate::dialects::{Dialect, Normalizer};
    use crate::events::Event;
    use crate::sse::SseEvent;

    fn frame(payload: serde_json::Value) -> SseEvent {
        SseEvent {
            event: None,
            data: payload.to_string(),
        }
    }

    #[test]
    fn anthropic_partial_usage_never_resets_or_invents_primary_counts() {
        for (start, delta, expected) in [
            (serde_json::json!({}), serde_json::json!({}), (None, None)),
            (
                serde_json::json!({"input_tokens":13}),
                serde_json::json!({}),
                (Some(13), None),
            ),
            (
                serde_json::json!({"input_tokens":13,"output_tokens":7}),
                serde_json::json!({}),
                (Some(13), Some(7)),
            ),
            (
                serde_json::json!({"input_tokens":13}),
                serde_json::json!({"output_tokens":7}),
                (Some(13), Some(7)),
            ),
            (
                serde_json::json!({"cache_creation_input_tokens":5}),
                serde_json::json!({"output_tokens":0}),
                (None, Some(0)),
            ),
        ] {
            let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
            normalizer
                .feed(&frame(
                    serde_json::json!({"type":"message_start","message":{"usage":start}}),
                ))
                .unwrap();
            normalizer.feed(&frame(serde_json::json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":delta}))).unwrap();
            let observed = normalizer.observed_usage().unwrap();
            assert_eq!((observed.input_tokens, observed.output_tokens), expected);
            let events = normalizer
                .feed(&frame(serde_json::json!({"type":"message_stop"})))
                .unwrap();
            let final_usage = events
                .iter()
                .find_map(|event| match event {
                    Event::Usage(usage) => Some(usage),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                (final_usage.input_tokens, final_usage.output_tokens),
                expected
            );
        }
    }

    #[test]
    fn thinking_blocks_normalize_to_dedicated_events() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }));
        assert!(normalizer.feed(&start).expect("start").is_empty());

        let delta = frame(serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step one"},
        }));
        let events = normalizer.feed(&delta).expect("thinking delta");
        assert!(matches!(
            events.as_slice(),
            [Event::ThinkingDelta { index: 0, delta }] if delta == "step one"
        ));

        let signature = frame(serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig=="},
        }));
        let events = normalizer.feed(&signature).expect("signature delta");
        assert!(matches!(
            events.as_slice(),
            [Event::ThinkingSignature { index: 0, signature }] if signature == "sig=="
        ));

        let redacted = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "opaque=="},
        }));
        let events = normalizer.feed(&redacted).expect("redacted block");
        assert!(matches!(
            events.as_slice(),
            [Event::RedactedThinking { index: 1, data }] if data == "opaque=="
        ));
    }

    /// Frame shapes captured live from one web_search stream (2026-08-31):
    /// server tool use with a leading empty input fragment, the whole result
    /// in its start frame, a cited answer, terminal usage that supersedes the
    /// start-frame input count, and the pause_turn stop reason.
    #[test]
    fn server_tool_frames_normalize_to_dedicated_events() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let start_message = frame(serde_json::json!({
            "type": "message_start",
            "message": {"usage": {"input_tokens": 2230, "output_tokens": 25}},
        }));
        // The start-frame meters surface early so the Messages encoder can put
        // them on its own `message_start` (Claude Code reads input there); the
        // terminal report still supersedes them at `message_stop`.
        assert!(matches!(
            normalizer.feed(&start_message).expect("start").as_slice(),
            [Event::Usage(usage)]
                if usage.input_tokens == Some(2230)
                    && usage.output_tokens == Some(25)
                    && usage.cached_input_tokens == Some(0)
        ));

        let start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
        }));
        let events = normalizer.feed(&start).expect("server tool start");
        assert!(matches!(
            events.as_slice(),
            [Event::ServerToolUseStarted { index: 0, call_id, name }]
                if call_id == "srvtoolu_1" && name == "web_search"
        ));

        // The provider legally leads with an empty input fragment.
        for partial in ["", "{\"query\": \"python\"}"] {
            let delta = frame(serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": partial},
            }));
            let events = normalizer.feed(&delta).expect("server input delta");
            assert!(matches!(
                events.as_slice(),
                [Event::ServerToolArgumentsDelta { index: 0, delta }] if delta == partial
            ));
        }
        let stop = frame(serde_json::json!({"type": "content_block_stop", "index": 0}));
        let events = normalizer.feed(&stop).expect("server tool stop");
        assert!(matches!(
            events.as_slice(),
            [Event::ServerToolUseCompleted { index: 0, call }]
                if call.raw_arguments == "{\"query\": \"python\"}" && !call.custom
        ));

        let result = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "url": "https://example.com"}],
                "caller": {"type": "direct"},
            },
        }));
        let events = normalizer.feed(&result).expect("server tool result");
        match events.as_slice() {
            [Event::ServerToolResult { index: 1, block }] => {
                assert!(block.contains("\"caller\":{\"type\":\"direct\"}"));
            }
            other => panic!("unexpected events: {other:?}"),
        }

        let text_start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 2,
            "content_block": {"citations": [], "type": "text", "text": ""},
        }));
        let events = normalizer.feed(&text_start).expect("text start");
        assert!(matches!(
            events.as_slice(),
            [Event::TextBlockStarted { index: 2 }]
        ));
        let citation = frame(serde_json::json!({
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": {"type": "web_search_result_location", "cited_text": "3.14.7"},
            },
        }));
        let events = normalizer.feed(&citation).expect("citation delta");
        match events.as_slice() {
            [Event::CitationDelta { index: 2, citation }] => {
                assert!(citation.contains("\"cited_text\":\"3.14.7\""));
            }
            other => panic!("unexpected events: {other:?}"),
        }

        // The terminal usage report supersedes the start-frame input legs.
        let message_delta = frame(serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "pause_turn", "stop_sequence": null},
            "usage": {
                "input_tokens": 12284,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 6,
                "output_tokens": 103,
                "server_tool_use": {"web_search_requests": 1},
            },
        }));
        assert!(normalizer.feed(&message_delta).expect("delta").is_empty());
        // A disconnect here still settles the just-decoded terminal meters.
        let observed = normalizer.observed_usage().expect("delta usage retained");
        assert_eq!(observed.input_tokens, Some(12294));
        assert_eq!(observed.output_tokens, Some(103));
        assert_eq!(observed.cache_creation_input_tokens, Some(6));
        let stop_message = frame(serde_json::json!({"type": "message_stop"}));
        let events = normalizer.feed(&stop_message).expect("message stop");
        match events.as_slice() {
            [Event::Usage(usage), Event::PausedTurn] => {
                assert_eq!(usage.input_tokens, Some(12294));
                assert_eq!(usage.output_tokens, Some(103));
                assert_eq!(usage.cached_input_tokens, Some(4));
                // Anthropic bills thinking inside output_tokens and publishes
                // no separate count, so the reasoning subset stays unknown.
                assert_eq!(usage.reasoning_tokens, None);
            }
            other => panic!("unexpected events: {other:?}"),
        }
    }

    fn tool_fragment_stream(stop_reason: &str) -> Vec<SseEvent> {
        tool_argument_stream(stop_reason, "{\"city\": \"Par")
    }

    fn tool_argument_stream(stop_reason: &str, partial_json: &str) -> Vec<SseEvent> {
        vec![
            frame(serde_json::json!({
                "type": "message_start",
                "message": {"id": "msg_1", "usage": {"input_tokens": 5, "output_tokens": 0}},
            })),
            frame(serde_json::json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
            })),
            frame(serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": partial_json},
            })),
            frame(serde_json::json!({"type": "content_block_stop", "index": 0})),
            frame(serde_json::json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {"output_tokens": 7},
            })),
            frame(serde_json::json!({"type": "message_stop"})),
        ]
    }

    fn chat_projection(events: &[Event]) -> (Vec<serde_json::Value>, serde_json::Value) {
        let mut encoder =
            crate::encode::ChatSseEncoder::new_with_ignored("request", "alias", 0, true, vec![]);
        let mut wire = encoder.start().expect("chat starts");
        for event in events {
            wire.extend(encoder.feed(event).expect("chat event encodes"));
        }
        let chunks = wire
            .iter()
            .filter_map(|frame| frame.strip_prefix("data: "))
            .filter(|data| data.trim() != "[DONE]")
            .map(|data| serde_json::from_str(data).expect("chat chunk JSON"))
            .collect();
        let aggregate = crate::encode::completed_chat_body_with_ignored(
            "request",
            "alias",
            0,
            events,
            &[],
            false,
        )
        .expect("nonstream chat aggregates");
        assert!(aggregate.failure.is_none());
        (chunks, aggregate.body)
    }

    #[test]
    fn empty_tool_max_tokens_never_invents_arguments_or_a_completed_call() {
        // The empty input in a start frame is a placeholder, not evidence that
        // a zero-argument call finished. Block stop precedes the stop reason.
        for block_stop in [true, false] {
            for empty_delta in [true, false] {
                let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
                let mut events = Vec::new();
                for frame in tool_argument_stream("max_tokens", "") {
                    let payload: serde_json::Value = serde_json::from_str(&frame.data).unwrap();
                    if (!block_stop && payload["type"] == "content_block_stop")
                        || (!empty_delta && payload["type"] == "content_block_delta")
                    {
                        continue;
                    }
                    events.extend(normalizer.feed(&frame).expect("empty truncated tool"));
                    assert!(!events.iter().any(|event| match event {
                        Event::ToolCallCompleted { .. } => true,
                        Event::ToolArgumentsDelta { delta, .. } => !delta.is_empty(),
                        _ => false,
                    }));
                }
                assert!(matches!(events.last(), Some(Event::Incomplete)));
                let (chunks, body) = chat_projection(&events);
                assert!(chunks
                    .iter()
                    .any(|chunk| chunk["choices"][0]["finish_reason"] == "length"));
                assert!(chunks.iter().all(|chunk| {
                    chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                        .as_str()
                        .is_none_or(str::is_empty)
                }));
                assert_eq!(body["choices"][0]["finish_reason"], "length");
                assert!(body["choices"][0]["message"].get("tool_calls").is_none());
            }
        }
    }

    #[test]
    fn zero_argument_tool_completes_only_after_a_normal_final_reason() {
        for reason in ["tool_use", "end_turn", "stop_sequence"] {
            let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
            let mut events = Vec::new();
            for frame in tool_argument_stream(reason, "") {
                let payload: serde_json::Value = serde_json::from_str(&frame.data).unwrap();
                let next = normalizer.feed(&frame).expect("zero-argument tool");
                if payload["type"] == "content_block_stop" {
                    assert!(
                        next.is_empty(),
                        "empty completion must wait for stop reason"
                    );
                }
                events.extend(next);
            }
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event, Event::ToolCallCompleted { call, .. } if call.raw_arguments == "{}"
                    ))
                    .count(),
                1
            );
            assert!(matches!(events.last(), Some(Event::Completed)));
            let (chunks, body) = chat_projection(&events);
            assert!(chunks
                .iter()
                .any(|chunk| chunk["choices"][0]["finish_reason"] == "tool_calls"));
            assert_eq!(
                body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                "{}"
            );
        }
    }

    #[test]
    fn complete_argument_objects_still_finish_at_block_stop_without_buffering() {
        // An explicit {} is real provider output. Required-field validation
        // belongs to the caller's schema, not the JSON-shape normalizer.
        for arguments in ["{}", "{\"city\": \"Paris\"}"] {
            for reason in ["tool_use", "max_tokens"] {
                let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
                let mut events = Vec::new();
                for frame in tool_argument_stream(reason, arguments) {
                    let payload: serde_json::Value = serde_json::from_str(&frame.data).unwrap();
                    let next = normalizer.feed(&frame).expect("complete argument object");
                    if payload["type"] == "content_block_stop" {
                        assert!(
                            matches!(next.as_slice(), [Event::ToolCallCompleted { call, .. }]
                            if call.raw_arguments == arguments)
                        );
                    }
                    events.extend(next);
                }
                let (_, body) = chat_projection(&events);
                assert_eq!(
                    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                    arguments
                );
                assert_eq!(
                    body["choices"][0]["finish_reason"],
                    if reason == "max_tokens" {
                        "length"
                    } else {
                        "tool_calls"
                    }
                );
            }
        }
    }

    #[test]
    fn partial_argument_bytes_reach_chat_immediately_and_remain_verbatim_on_length() {
        let partial = "{\"city\": \"Par";
        for block_stop in [true, false] {
            let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
            let mut events = Vec::new();
            let mut encoder = crate::encode::ChatSseEncoder::new_with_ignored(
                "request",
                "alias",
                0,
                false,
                vec![],
            );
            encoder.start().unwrap();
            for frame in tool_argument_stream("max_tokens", partial) {
                let payload: serde_json::Value = serde_json::from_str(&frame.data).unwrap();
                if !block_stop && payload["type"] == "content_block_stop" {
                    continue;
                }
                let next = normalizer.feed(&frame).expect("partial tool");
                for event in &next {
                    let wire = encoder.feed(event).expect("encodes immediately");
                    if payload["type"] == "content_block_delta" {
                        assert!(
                            matches!(event, Event::ToolArgumentsDelta { delta, .. } if delta == partial)
                        );
                        let chunk: serde_json::Value =
                            serde_json::from_str(wire[0].strip_prefix("data: ").unwrap()).unwrap();
                        assert_eq!(
                            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
                            partial
                        );
                    }
                }
                events.extend(next);
            }
            assert!(!events
                .iter()
                .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
            let (_, body) = chat_projection(&events);
            assert_eq!(body["choices"][0]["finish_reason"], "length");
            assert!(body["choices"][0]["message"].get("tool_calls").is_none());
        }
    }

    #[test]
    fn a_tool_call_cut_off_by_the_output_budget_is_incomplete_not_malformed() {
        // Anthropic reveals max_tokens only in message_delta, AFTER the tool
        // block stopped with its arguments still an open fragment. That is the
        // provider's own truncation: the unfinished call is dropped and the
        // stream ends Incomplete (raise max_tokens), never a 502.
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let mut events = Vec::new();
        for frame in tool_fragment_stream("max_tokens") {
            events.extend(
                normalizer
                    .feed(&frame)
                    .expect("truncated tool stream normalizes"),
            );
        }
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
        assert!(matches!(events.last(), Some(Event::Incomplete)));
    }

    #[test]
    fn a_tool_call_cut_mid_fragment_under_tool_use_is_the_providers_cut() {
        // The block closed on an open fragment and the stop reason then said
        // `tool_use`: a model never ends a well-formed call mid-string, so
        // the stop reason misreports the cut. The call is dropped and the
        // turn settles Incomplete instead of failing the served stream.
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let mut events = Vec::new();
        for frame in tool_fragment_stream("tool_use") {
            events.extend(normalizer.feed(&frame).expect("cut tool stream normalizes"));
        }
        assert!(!events
            .iter()
            .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
        assert!(
            matches!(events.last(), Some(Event::Incomplete)),
            "{events:?}"
        );
    }

    #[test]
    fn a_tool_call_with_garbage_arguments_still_fails_when_the_provider_finished() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        // A syntax error INSIDE the arguments is corruption, not a cut.
        let frames = tool_argument_stream("tool_use", "{\"city\": }");
        let mut outcome = Ok(Vec::new());
        for frame in &frames {
            outcome = normalizer.feed(frame);
            if outcome.is_err() {
                break;
            }
        }
        let failure = outcome.expect_err("unparsable arguments on a finished turn are malformed");
        assert_eq!(
            failure.failure_class,
            crate::errors::FailureClass::MalformedResponse
        );
        assert!(failure.safe_message.contains("not valid JSON"));
    }

    /// A native tool-search result block (Anthropic tool search on an
    /// all-Anthropic route) takes the same verbatim carry as web search and
    /// round-trips byte-for-byte through the Messages encoder.
    #[test]
    fn tool_search_result_blocks_carry_verbatim_and_round_trip() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let block = serde_json::json!({
            "type": "tool_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
            },
        });
        let start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": block,
        }));
        let events = normalizer.feed(&start).expect("tool search result");
        let carried = match events.as_slice() {
            [Event::ServerToolResult {
                index: 0,
                block: carried,
            }] => carried.clone(),
            other => panic!("unexpected events: {other:?}"),
        };
        assert_eq!(carried, crate::encode::compact_json(&block));

        let mut encoder = crate::encode_messages::MessagesSseEncoder::new("req", "alias");
        encoder.start().expect("starts");
        let frames = encoder.feed(&events[0]).expect("encodes");
        let start_frame = frames
            .iter()
            .find(|frame| frame.starts_with("event: content_block_start"))
            .expect("the result block starts");
        let data = start_frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("data line");
        let payload: serde_json::Value = serde_json::from_str(data).expect("json");
        assert_eq!(payload["content_block"], block);
        assert_eq!(
            crate::encode::compact_json(&payload["content_block"]),
            carried
        );
    }
}
