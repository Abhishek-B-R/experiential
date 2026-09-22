//! Provider output-item metadata encoding.

use super::{ProviderAssistantMessagePhase, ProviderOutputItemStatus};
use serde_json::Value;

pub(super) fn add_provider_item_metadata(
    payload: &mut Value,
    item_id: &Option<String>,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
) {
    if let Some(item_id) = item_id {
        payload["item_id"] = Value::String(item_id.clone());
    }
    if let Some(status) = status {
        payload["status"] = Value::String(status.as_str().to_string());
    }
    if let Some(phase) = phase {
        payload["phase"] = Value::String(phase.as_str().to_string());
    }
}
