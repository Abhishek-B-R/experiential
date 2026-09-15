//! A provider stream that closes cleanly WITHOUT a terminal frame after it
//! already produced output. Before any output the relay reports
//! `ended_without_terminal` (failover-eligible, nothing to preserve); after
//! output the served tokens are real and the close is the provider's cut:
//! Gemini legitimately ends turns this way and completes, an OpenAI-compatible
//! relay that declared its finish and only dropped `[DONE]` settles by that
//! finish, and every other wire settles `Incomplete` with any mid-fragment
//! call dropped (gpt-5.6-luna on OpenAI's Responses wire, 33 of 34
//! post-commit streams in 14 days, 2026-09-14). Child of `dialects` so the
//! parent stays under the hand-authored line budget.

use super::{finish_open_tools_relay, Dialect, Normalizer};
use crate::errors::Failure;
use crate::events::{Event, ProviderOutputItemStatus};

impl Normalizer {
    /// Synthesize the terminal events for a stream that closed cleanly
    /// without an explicit terminal frame, or nothing when a terminal already
    /// ended the stream or nothing was served (the caller then fails it
    /// closed as terminal-less). Errors only when finishing the open tool
    /// calls fails: a syntactically invalid streamed argument object stays
    /// malformed.
    pub fn on_stream_end(&mut self) -> Result<Vec<Event>, Failure> {
        if self.terminal {
            return Ok(Vec::new());
        }
        // An OpenAI-compatible finish reason already seen is a complete
        // ending whether or not output followed it (a content_filter finish
        // with nothing served is the declared refusal, Azure Foundry DeepSeek
        // 2026-09-15); it settles exactly as `[DONE]` would.
        if self.dialect == Dialect::OpenAiCompatible && self.finish_reason.is_some() {
            self.log_stream_end("declared_finish");
            return self.end_with(self.openai_compatible_stream_end()?);
        }
        if !self.emitted_output {
            return Ok(Vec::new());
        }
        let events = match self.dialect {
            // Gemini ends some streams right after its last content frame
            // without a `finishReason` frame: a complete answer.
            Dialect::GeminiGenerateContent => {
                let mut events = Vec::new();
                if let Some(usage) = self.usage.take() {
                    events.push(Event::Usage(usage));
                }
                events.push(Event::Completed);
                events
            }
            _ => {
                self.log_stream_end("incomplete");
                let mut events = Vec::new();
                if self.dialect == Dialect::OpenAiResponses {
                    events.extend(
                        self.openai_close_unfinished_items(ProviderOutputItemStatus::Incomplete),
                    );
                }
                let (tool_events, _dropped) = finish_open_tools_relay(&mut self.tools, "stream_end")?;
                events.extend(tool_events);
                if let Some(usage) = self.usage.take() {
                    events.push(Event::Usage(usage));
                }
                events.push(Event::Incomplete);
                events
            }
        };
        self.end_with(events)
    }

    fn end_with(&mut self, events: Vec<Event>) -> Result<Vec<Event>, Failure> {
        if events.iter().any(Event::is_terminal) {
            self.terminal = true;
        }
        Ok(events)
    }

    fn log_stream_end(&self, verdict: &str) {
        let line = serde_json::json!({
            "event": "stream_ended_without_terminal_after_output",
            "dialect": dialect_name(self.dialect),
            "open_tools": self.tools.values().filter(|tool| !tool.completed).count(),
            "verdict": verdict,
        });
        eprintln!("exp-gateway-native: {line}");
    }
}

fn dialect_name(dialect: Dialect) -> &'static str {
    match dialect {
        Dialect::OpenAiResponses => "openai_responses",
        Dialect::AnthropicMessages => "anthropic_messages",
        Dialect::OpenAiCompatible => "openai_compatible",
        Dialect::GeminiGenerateContent => "gemini_generate_content",
        Dialect::BedrockConverseStream => "bedrock_converse_stream",
    }
}
