//! Capture authorized provider evidence before public protocol projection.

use std::sync::Arc;

use super::collector::Collector;
use crate::admission::Admission;
use crate::events::Event;
use crate::waterfall::Won;

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
    won: &mut Won,
) {
    let Some(collector) = collector else { return };
    let depth = match won {
        Won::Committed(attempt) => attempt.depth,
        Won::Settled(attempt) => attempt.depth,
        Won::Failed(_) => return,
    };
    let observer = Observer {
        collector,
        request_id: admission.request_id.clone(),
        reasoning_exposed: admission.reasoning_exposed_at(depth),
    };
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
