//! Restore native tool identities after provider-side function translation.
//!
//! Namespaced function arguments remain incremental. Translated custom inputs
//! cannot be safely exposed until their JSON wrapper is complete: the relay
//! suppresses wrapper deltas and emits one decoded input delta immediately
//! before completion. The normalizer already bounds, accumulates and validates
//! these arguments before this seam; no second argument buffer is retained.

use std::collections::{HashMap, HashSet, VecDeque};

use serde_json::Value;

use crate::errors::{Failure, FailureClass};
use crate::events::Event;

/// Provider-facing name -> (original name, namespace, custom input format).
pub type NativeToolTranslation = HashMap<String, (String, Option<String>, bool)>;

/// One attempt's inverse mapping and still-open translated custom calls.
#[derive(Default)]
pub struct NativeToolInversion {
    translation: NativeToolTranslation,
    custom_indices: HashSet<u32>,
}

impl NativeToolInversion {
    pub fn new(translation: NativeToolTranslation) -> Self {
        Self {
            translation,
            custom_indices: HashSet::new(),
        }
    }

    /// Transform normalized events without retaining another copy of arguments.
    ///
    /// Starts remain immediate commitment signals. Invalid wrappers fail after
    /// that commitment rather than becoming executable raw JSON custom input.
    /// Only normalizer-validated completed calls flush deferred custom input.
    pub fn push(&mut self, mut event: Event, ready: &mut VecDeque<Event>) -> Result<(), Failure> {
        match &mut event {
            Event::ToolCallStarted {
                index,
                name,
                namespace,
                custom,
                ..
            } => {
                if let Some((origin_name, origin_namespace, is_custom)) = self.translation.get(name)
                {
                    *name = origin_name.clone();
                    *namespace = origin_namespace.clone();
                    if *is_custom {
                        *custom = true;
                        self.custom_indices.insert(*index);
                    }
                }
            }
            Event::ToolArgumentsDelta { index, .. } if self.custom_indices.contains(index) => {
                return Ok(());
            }
            Event::ToolCallCompleted { index, call } => {
                if let Some((origin_name, origin_namespace, is_custom)) =
                    self.translation.get(&call.name)
                {
                    call.name = origin_name.clone();
                    call.namespace = origin_namespace.clone();
                    if *is_custom {
                        let input = unwrap_custom_input(&call.raw_arguments)?;
                        if !self.custom_indices.remove(index) {
                            return Err(malformed_custom_input());
                        }
                        call.custom = true;
                        call.raw_arguments = input;
                        ready.push_back(Event::ToolArgumentsDelta {
                            index: *index,
                            delta: call.raw_arguments.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
        ready.push_back(event);
        Ok(())
    }
}

/// Accept only the single required string the translated declaration describes.
fn unwrap_custom_input(raw_arguments: &str) -> Result<String, Failure> {
    let Ok(Value::Object(mut object)) = serde_json::from_str::<Value>(raw_arguments) else {
        return Err(malformed_custom_input());
    };
    match object.remove("input") {
        Some(Value::String(input)) if object.is_empty() => Ok(input),
        _ => Err(malformed_custom_input()),
    }
}

fn malformed_custom_input() -> Failure {
    Failure::new(
        FailureClass::MalformedResponse,
        "Translated custom tool arguments must contain exactly one input string",
    )
    .with_retry(false, true)
}

#[cfg(test)]
#[path = "codex_native_inversion_tests.rs"]
mod tests;
