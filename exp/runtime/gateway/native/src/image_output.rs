//! Validation and wire shapes for complete generated images inside chat turns.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass};

/// Reject invalid image payloads without exposing provider content in diagnostics.
fn invalid_image() -> Failure {
    Failure::new(
        FailureClass::MalformedResponse,
        "provider returned an invalid generated image",
    )
    .with_retry(false, false)
}

/// Validate an inline raster image and preserve its encoded bytes exactly.
pub fn inline_image(media_type: &str, data: &str) -> Result<String, Failure> {
    let format = match media_type {
        "image/png" => image::ImageFormat::Png,
        "image/jpeg" => image::ImageFormat::Jpeg,
        "image/webp" => image::ImageFormat::WebP,
        "image/gif" => image::ImageFormat::Gif,
        _ => return Err(invalid_image()),
    };
    let bytes = STANDARD.decode(data).map_err(|_| invalid_image())?;
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(&bytes), format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16_384);
    limits.max_image_height = Some(16_384);
    limits.max_alloc = Some(128 * 1024 * 1024);
    reader.limits(limits);
    reader.decode().map_err(|_| invalid_image())?;
    Ok(format!("data:{media_type};base64,{data}"))
}

/// Decode the OpenRouter-style image URL envelope without fetching external URLs.
pub fn chat_image(value: &Value) -> Result<String, Failure> {
    let url = value
        .pointer("/image_url/url")
        .and_then(Value::as_str)
        .ok_or_else(invalid_image)?;
    let (media_type, data) = url
        .strip_prefix("data:")
        .and_then(|value| value.split_once(";base64,"))
        .ok_or_else(invalid_image)?;
    inline_image(media_type, data)
}

/// Encode the chat images extension for one completed image.
pub fn chat_image_value(url: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": url}})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::{Dialect, Normalizer};
    use crate::events::Event;
    use crate::sse::SseEvent;

    const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

    #[test]
    fn inline_raster_is_preserved_and_remote_or_active_content_is_refused() {
        let url = inline_image("image/png", PNG).unwrap();
        assert_eq!(chat_image(&chat_image_value(&url)).unwrap(), url);
        for image in [
            json!({}),
            json!({"image_url":{"url":"https://example.test/a.png"}}),
            json!({"image_url":{"url":"data:image/svg+xml;base64,PHN2Zz4="}}),
        ] {
            let failure = chat_image(&image).unwrap_err();
            assert!(!failure.retryable_same_deployment);
        }
        assert!(inline_image("image/png", "iVBORw0KGgo=").is_err());
        assert!(inline_image("image/png", "garbage").is_err());
        assert!(inline_image("image/png", "aGVsbG8=").is_err());
    }

    #[test]
    fn both_chat_dialects_preserve_visible_images_without_thought_images() {
        let url = inline_image("image/png", PNG).unwrap();
        let cases = [
            (
                Dialect::OpenAiCompatible,
                json!({"choices":[{"index":0,"delta":{"content":"A cat", "images":[chat_image_value(&url)]},"finish_reason":"stop"}]}),
            ),
            (
                Dialect::GeminiGenerateContent,
                json!({"candidates":[{"content":{"parts":[{"thought":true,"inlineData":{"mimeType":"image/png","data":PNG}},{"text":"A cat"},{"inlineData":{"mimeType":"image/png","data":PNG}}]},"finishReason":"STOP"}]}),
            ),
        ];
        for (dialect, payload) in cases {
            let mut normalizer = Normalizer::new(dialect);
            let events = normalizer
                .feed(&SseEvent {
                    event: None,
                    data: payload.to_string(),
                })
                .unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, Event::Image(_)))
                    .count(),
                1
            );
            assert!(events
                .iter()
                .any(|e| matches!(e, Event::Image(value) if value == &url)));
            assert!(events
                .iter()
                .any(|e| matches!(e, Event::TextDelta(value) if value == "A cat")));
        }
    }
}
