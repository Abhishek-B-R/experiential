//! Lossless completed Messages projection from ordered public SSE data frames.

use serde_json::{Map, Value};
use std::collections::BTreeMap;

struct Block {
    value: Value,
    arguments: Option<String>,
    closed: bool,
}

pub(super) fn assemble(frames: &[Value]) -> Option<Value> {
    let mut message: Option<Value> = None;
    let mut blocks: BTreeMap<u64, Block> = BTreeMap::new();
    let mut stopped = false;
    for frame in frames {
        if stopped {
            return None;
        }
        match frame.get("type")?.as_str()? {
            "ping" => {}
            "message_start" if message.is_none() => {
                let start = frame.get("message")?;
                if start.get("type")?.as_str()? != "message"
                    || !start.get("content")?.as_array()?.is_empty()
                {
                    return None;
                }
                message = Some(start.clone());
            }
            "content_block_start" if message.is_some() => {
                let index = frame.get("index")?.as_u64()?;
                if blocks.contains_key(&index) || index != blocks.len() as u64 {
                    return None;
                }
                blocks.insert(
                    index,
                    Block {
                        value: frame.get("content_block")?.clone(),
                        arguments: None,
                        closed: false,
                    },
                );
            }
            "content_block_delta" => {
                let block = blocks.get_mut(&frame.get("index")?.as_u64()?)?;
                if block.closed {
                    return None;
                }
                let delta = frame.get("delta")?;
                match delta.get("type")?.as_str()? {
                    "text_delta" => append(
                        block.value.as_object_mut()?,
                        "text",
                        delta.get("text")?.as_str()?,
                    )?,
                    "thinking_delta" => append(
                        block.value.as_object_mut()?,
                        "thinking",
                        delta.get("thinking")?.as_str()?,
                    )?,
                    "signature_delta" => append(
                        block.value.as_object_mut()?,
                        "signature",
                        delta.get("signature")?.as_str()?,
                    )?,
                    "input_json_delta" => block
                        .arguments
                        .get_or_insert_with(String::new)
                        .push_str(delta.get("partial_json")?.as_str()?),
                    "citations_delta" => block
                        .value
                        .as_object_mut()?
                        .entry("citations")
                        .or_insert_with(|| Value::Array(vec![]))
                        .as_array_mut()?
                        .push(delta.get("citation")?.clone()),
                    _ => return None,
                }
            }
            "content_block_stop" => {
                let block = blocks.get_mut(&frame.get("index")?.as_u64()?)?;
                if block.closed {
                    return None;
                }
                if let Some(arguments) = &block.arguments {
                    block.value["input"] = serde_json::from_str(arguments).ok()?;
                }
                block.closed = true;
            }
            "message_delta" => {
                let target = message.as_mut()?.as_object_mut()?;
                for (key, value) in frame.get("delta")?.as_object()? {
                    target.insert(key.clone(), value.clone());
                }
                if let Some(usage) = frame.get("usage") {
                    let target = target.get_mut("usage")?.as_object_mut()?;
                    for (key, value) in usage.as_object()? {
                        target.insert(key.clone(), value.clone());
                    }
                }
            }
            "message_stop" => stopped = true,
            _ => return None,
        }
    }
    let mut message = message?;
    if !stopped
        || !message.get("stop_reason").is_some_and(Value::is_string)
        || blocks.values().any(|block| !block.closed)
    {
        return None;
    }
    message["content"] = Value::Array(blocks.into_values().map(|block| block.value).collect());
    Some(message)
}

fn append(object: &mut Map<String, Value>, key: &str, delta: &str) -> Option<()> {
    match object
        .entry(key)
        .or_insert_with(|| Value::String(String::new()))
    {
        Value::String(text) => {
            text.push_str(delta);
            Some(())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preserves_thinking_signatures_tool_inputs_and_requires_complete_lifecycle() {
        let frames = vec![
            json!({"type":"message_start","message":{"type":"message","id":"msg","role":"assistant","content":[],"usage":{"input_tokens":2},"stop_reason":null}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" exactly\0雪\n"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}),
            json!({"type":"message_stop"}),
        ];
        let result = assemble(&frames).unwrap();
        assert_eq!(result["content"][0]["thinking"], " exactly\0雪\n");
        assert_eq!(result["content"][0]["signature"], "signed");
        assert_eq!(result["content"][1]["input"], json!({"x":1}));
        assert!(assemble(&frames[..frames.len() - 1]).is_none());
    }
}
