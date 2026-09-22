//! Capture authorized provider evidence before public protocol projection.

use std::sync::Arc;

use super::collector::Collector;
use crate::admission::Admission;
use crate::events::Event;
use crate::waterfall::Won;

pub(crate) struct GeminiObserver {
    collector: Arc<Collector>,
    request_id: String,
    generation: u64,
}

impl GeminiObserver {
    pub(super) fn new(collector: Arc<Collector>, request_id: String, generation: u64) -> Self {
        Self {
            collector,
            request_id,
            generation,
        }
    }

    pub(crate) fn observe(&self, part: &serde_json::Value) {
        self.collector
            .gemini_part(&self.request_id, self.generation, part);
    }
}

pub(crate) struct Observer {
    collector: Arc<Collector>,
    request_id: String,
    reasoning_exposed: bool,
}

impl Observer {
    pub(crate) fn observe(&self, event: &Event) {
        match event {
            Event::ReasoningContentDelta { delta, .. } if self.reasoning_exposed => {
                self.collector.reasoning(&self.request_id, delta);
            }
            Event::ToolCallCompleted { call, .. } => {
                self.collector.tool_call(&self.request_id, call)
            }
            _ => {}
        }
    }
}

/// Only the selected attempt contributes. Capture policy still gates persistence;
/// private provider reasoning never becomes plaintext merely because capture is on.
pub(crate) fn observe_winner(
    collector: Option<Arc<Collector>>,
    admission: &Admission,
    guard: &crate::settlement::AttemptGuard,
    won: &mut Won,
) {
    let Some(collector) = collector else { return };
    let depth = match won {
        Won::Committed(attempt) => attempt.depth,
        Won::Settled(attempt) => attempt.depth,
        Won::Failed(_) => return,
    };
    if let Some(wire) = admission.route.get(depth) {
        collector.observe_winner_model(&admission.request_id, &wire.exact_model_id);
    }
    let observer = Observer {
        collector,
        request_id: admission.request_id.clone(),
        reasoning_exposed: admission.reasoning_exposed_at(depth),
    };
    observer
        .collector
        .observe_attempt(&admission.request_id, guard.capture_observation());
    match won {
        Won::Committed(attempt) => {
            for event in &attempt.prefix {
                observer.observe(event);
            }
            attempt.relay.set_capture_reasoning(observer);
        }
        Won::Settled(attempt) => {
            for event in &attempt.events {
                observer.observe(event);
            }
        }
        Won::Failed(_) => {}
    }
}
