//! Allocation-free JSON sizing and bounded final encoding for capture destinations.

use std::io::{self, Write};

use serde::Serialize;
use serde_json::Value;

/// UTF-8 compact JSON length, without constructing escaped strings.
pub(super) fn string_bytes(text: &str) -> usize {
    text.as_bytes().iter().fold(2usize, |size, byte| {
        size.saturating_add(match byte {
            b'"' | b'\\' | b'\n' | b'\r' | b'\t' | 8 | 12 => 2,
            0..=31 => 6,
            _ => 1,
        })
    })
}

pub(super) fn optional_string_bytes(text: Option<&str>) -> usize {
    text.map_or(4, string_bytes)
}

/// Count an already parsed JSON tree; only scalar numbers require formatting.
pub(super) fn json_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(value) => {
            if *value {
                4
            } else {
                5
            }
        }
        Value::Number(value) => value.to_string().len(),
        Value::String(value) => string_bytes(value),
        Value::Array(values) => values.iter().fold(
            2usize.saturating_add(values.len().saturating_sub(1)),
            |size, value| size.saturating_add(json_bytes(value)),
        ),
        Value::Object(values) => values.iter().fold(
            2usize.saturating_add(values.len().saturating_sub(1)),
            |size, (key, value)| {
                size.saturating_add(string_bytes(key))
                    .saturating_add(1)
                    .saturating_add(json_bytes(value))
            },
        ),
    }
}

/// Charge retained tree nodes as well as string and vector capacity.
/// Objects enter through serde, without caller-controlled spare map capacity.
pub(super) fn heap_bytes(value: &Value) -> usize {
    match value {
        Value::String(text) => text.capacity(),
        Value::Array(values) => values.iter().fold(
            values
                .capacity()
                .saturating_mul(std::mem::size_of::<Value>()),
            |size, value| size.saturating_add(heap_bytes(value)),
        ),
        Value::Object(values) => {
            values
                .iter()
                .fold(values.len().saturating_mul(256), |size, (key, value)| {
                    size.saturating_add(key.capacity())
                        .saturating_add(heap_bytes(value))
                })
        }
        _ => 0,
    }
}

/// Stop the final serializer before an oversized payload can allocate without bound.
pub(super) fn encode(value: &impl Serialize, maximum: usize) -> Option<String> {
    struct Limited {
        bytes: Vec<u8>,
        maximum: usize,
    }
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > self.maximum.saturating_sub(self.bytes.len()) {
                return Err(io::Error::other("capture record exceeds storage budget"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Limited {
        bytes: Vec::new(),
        maximum,
    };
    serde_json::to_writer(&mut output, value).ok()?;
    String::from_utf8(output.bytes).ok()
}

#[cfg(test)]
#[path = "budget_test.rs"]
mod tests;
