//! Bounded provider facts retained across cancellation of an owning future.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::events::{Event, Usage};
use crate::relay::track_event;

/// Shared only by one physical attempt's guard and relay, reset at rebind.
#[derive(Clone, Default)]
pub(crate) struct Observation(Arc<Mutex<Observed>>);

#[derive(Clone)]
pub(crate) struct Observed {
    pub started_at: SystemTime,
    started: Instant,
    pub terminal_at: Option<SystemTime>,
    pub duration: Option<Duration>,
    pub usage: Option<Usage>,
    pub tool_names: Vec<String>,
    pub terminal: Option<Event>,
    pub first_token_at: Option<SystemTime>,
}

impl Default for Observed {
    fn default() -> Self {
        Self {
            started_at: SystemTime::now(),
            started: Instant::now(),
            terminal_at: None,
            duration: None,
            usage: None,
            tool_names: Vec::new(),
            terminal: None,
            first_token_at: None,
        }
    }
}

impl Observation {
    /// A repaired dial resets its meter but retains the physical attempt's start.
    pub(crate) fn next_dial(&self) -> Self {
        let previous = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self(Arc::new(Mutex::new(Observed {
            started_at: previous.started_at,
            started: previous.started,
            ..Observed::default()
        })))
    }

    /// Remember normalized facts before public delivery can suspend or fail.
    pub(crate) fn record(&self, event: &Event) {
        let mut observed = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A queued provider terminal fixes the meter/outcome. Earlier tool
        // events may still emerge from normalization before public delivery.
        if observed.terminal.is_some() && (event.is_terminal() || matches!(event, Event::Usage(_)))
        {
            return;
        }
        if let Event::Usage(usage) = event {
            // Partial meters are observations too, never an invented zero.
            if usage.input_tokens.is_some() || usage.output_tokens.is_some() {
                match &mut observed.usage {
                    Some(previous) => previous.merge_observed(usage),
                    None => observed.usage = Some(usage.clone()),
                }
            }
        } else {
            let Observed {
                usage, tool_names, ..
            } = &mut *observed;
            track_event(event, usage, tool_names);
        }
        if event.is_terminal() {
            observed.terminal = Some(event.clone());
            observed.terminal_at = Some(SystemTime::now());
            observed.duration = Some(observed.started.elapsed());
        }
    }

    /// Replace an across-dial aggregate, where a newly unknown leg must not
    /// inherit the earlier dial's known subtotal through cumulative merging.
    pub(crate) fn record_dial_total(&self, usage: Usage) {
        let mut observed = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if observed.terminal.is_none() {
            observed.usage = Some(usage);
        }
    }

    /// Stop-sequence filtering can refine a buffered provider terminal.
    pub(crate) fn record_effective_terminal(&self, event: &Event) {
        if event.is_terminal() {
            let mut observed = self
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if observed.terminal_at.is_none() {
                observed.terminal_at = Some(SystemTime::now());
                observed.duration = Some(observed.started.elapsed());
            }
            observed.terminal = Some(event.clone());
        }
    }

    pub(crate) fn record_first_token(&self, at: Option<SystemTime>) {
        let mut observed = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if observed.first_token_at.is_none() {
            observed.first_token_at = at;
        }
    }

    pub(crate) fn snapshot(&self) -> Observed {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_preserves_partial_usage_and_terminal_precedence() {
        let observation = Observation::default();
        observation.record(&Event::Usage(Usage {
            input_tokens: Some(12),
            ..Usage::default()
        }));
        observation.record(&Event::Incomplete);
        observation.record(&Event::Completed);
        let snapshot = observation.snapshot();
        let usage = snapshot.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, None);
        assert!(matches!(snapshot.terminal, Some(Event::Incomplete)));
    }
}
