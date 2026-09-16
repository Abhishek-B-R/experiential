//! Reassemble only completed public SSE exchanges for asynchronous capture.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

/// Return a protocol response only after a valid public terminal event.
pub(super) fn completed_response(bytes: &[u8], protocol: &str) -> Option<Value> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut chunks = Vec::new();
    let mut done = false;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            done = true;
            continue;
        }
        let value: Value = serde_json::from_str(data).ok()?;
        if value.get("error").is_some() {
            return None;
        }
        if protocol == "responses" {
            match value.get("type").and_then(Value::as_str) {
                Some("response.completed" | "response.incomplete") => {
                    return value
                        .get("response")
                        .filter(|response| response.is_object())
                        .cloned()
                }
                Some("response.failed" | "error") => return None,
                _ => {}
            }
        } else {
            chunks.push(value);
        }
    }
    if protocol != "chat_completions" || !done {
        return None;
    }
    assemble_chat(chunks)
}

fn assemble_chat(chunks: Vec<Value>) -> Option<Value> {
    let first = chunks.first()?;
    let mut result = json!({"id": first.get("id")?, "object": "chat.completion",
        "model": first.get("model")?, "created": first.get("created").unwrap_or(&Value::Null)});
    let mut choices: BTreeMap<u64, Value> = BTreeMap::new();
    for chunk in &chunks {
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            result["usage"] = usage.clone();
        }
        for choice in chunk.get("choices")?.as_array()? {
            let index = choice.get("index")?.as_u64()?;
            let target = choices.entry(index).or_insert_with(
                || json!({"index": index, "message": {"role": "assistant"}, "finish_reason": null}),
            );
            let message = target.get_mut("message")?.as_object_mut()?;
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                for (key, value) in delta {
                    if key == "tool_calls" {
                        merge_tools(message, value)?;
                    } else if key == "role" {
                        message.insert(key.clone(), value.clone());
                    } else {
                        append(message, key, value)?;
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
                target["finish_reason"] = reason.clone();
            }
        }
    }
    if choices.is_empty()
        || choices
            .values()
            .any(|choice| choice["finish_reason"].is_null())
    {
        return None;
    }
    result["choices"] = Value::Array(choices.into_values().collect());
    Some(result)
}

fn append(object: &mut Map<String, Value>, key: &str, addition: &Value) -> Option<()> {
    if addition.is_null() {
        return Some(());
    }
    match object.get_mut(key) {
        None => {
            object.insert(key.to_owned(), addition.clone());
        }
        Some(Value::String(text)) => text.push_str(addition.as_str()?),
        Some(Value::Array(values)) => values.extend(addition.as_array()?.iter().cloned()),
        Some(value) if value == addition => {}
        _ => return None,
    }
    Some(())
}

fn merge_tools(message: &mut Map<String, Value>, addition: &Value) -> Option<()> {
    let tools = message
        .entry("tool_calls")
        .or_insert_with(|| json!([]))
        .as_array_mut()?;
    for delta in addition.as_array()? {
        let index = delta.get("index")?.as_u64()? as usize;
        // Provider indexes cannot allocate an unbounded sparse vector.
        if index > tools.len() {
            return None;
        }
        if index == tools.len() {
            tools.push(json!({"type": "function", "function": {}}));
        }
        let target = tools[index].as_object_mut()?;
        for (key, value) in delta.as_object()? {
            match key.as_str() {
                "index" => {}
                "function" => {
                    let function = target.get_mut("function")?.as_object_mut()?;
                    for (key, value) in value.as_object()? {
                        append(function, key, value)?;
                    }
                }
                "type" | "id" => {
                    target.insert(key.clone(), value.clone());
                }
                _ => {
                    append(target, key, value)?;
                }
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incomplete_chat_stream_never_becomes_training_evidence() {
        let chunk = br#"data: {"id":"chatcmpl-1","model":"model","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}

data: [DONE]
"#;
        assert!(completed_response(chunk, "chat_completions").is_none());
    }

    #[test]
    fn complete_chat_preserves_tool_arguments() {
        let first = json!({"id":"chatcmpl-1","model":"model","choices":[{"index":0,
            "delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function",
                "function":{"name":"search","arguments":"{\"query\":"}}]},"finish_reason":null}]});
        let second = json!({"id":"chatcmpl-1","model":"model","choices":[{"index":0,
            "delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"test\"}"}}]},
            "finish_reason":"tool_calls"}]});
        let bytes = format!("data: {first}\n\ndata: {second}\n\ndata: [DONE]\n\n");
        let response = completed_response(bytes.as_bytes(), "chat_completions").unwrap();
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"query\":\"test\"}"
        );
    }
}
