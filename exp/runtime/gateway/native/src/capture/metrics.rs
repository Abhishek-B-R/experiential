//! Winning-attempt timing and usage, borrowed from native accounting observations.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::events::Usage;
use crate::settlement::Observation;

/// Provider attempt facts, not request-wide totals across failed fallback attempts.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Metrics {
    pub started_at: f64,
    pub first_token_at: Option<f64>,
    pub terminal_at: Option<f64>,
    pub duration_ms: Option<f64>,
    pub usage: Option<Usage>,
    /// False for an interrupted or partial meter, even when some counts are known.
    pub usage_complete: bool,
}

fn timestamp(at: SystemTime) -> f64 {
    at.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

impl Metrics {
    pub(crate) fn observed(observation: &Observation) -> Self {
        let observed = observation.snapshot();
        Self {
            started_at: timestamp(observed.started_at),
            first_token_at: observed.first_token_at.map(timestamp),
            terminal_at: observed.terminal_at.map(timestamp),
            duration_ms: observed
                .duration
                .map(|elapsed| elapsed.as_secs_f64() * 1000.0),
            usage_complete: observed.terminal.is_some()
                && observed.usage.as_ref().is_some_and(Usage::has_token_counts),
            usage: observed.usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Event;

    #[test]
    fn metrics_preserve_unknown_usage_and_provider_timestamps() {
        let observation = Observation::default();
        observation.record_first_token(Some(SystemTime::now()));
        observation.record(&Event::Usage(Usage {
            input_tokens: Some(17),
            ..Usage::default()
        }));
        let partial = Metrics::observed(&observation);
        assert!(!partial.usage_complete);
        assert!(partial.terminal_at.is_none());
        assert_eq!(partial.usage.unwrap().output_tokens, None);
        observation.record(&Event::Usage(Usage {
            output_tokens: Some(5),
            reasoning_tokens: Some(2),
            ..Usage::default()
        }));
        observation.record(&Event::Completed);
        let complete = Metrics::observed(&observation);
        assert!(complete.usage_complete);
        assert!(complete.first_token_at.unwrap() >= complete.started_at);
        assert!(complete.terminal_at.unwrap() >= complete.first_token_at.unwrap());
        assert!(complete.duration_ms.unwrap() >= 0.0);
        assert_eq!(complete.usage.as_ref().unwrap().input_tokens, Some(17));
        assert_eq!(complete.usage.as_ref().unwrap().reasoning_tokens, Some(2));
        assert_eq!(complete.usage.as_ref().unwrap().cached_input_tokens, None);
    }
}
