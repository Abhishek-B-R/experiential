# Generated images in chat

Image-emitting Chat models on Gemini and OpenAI-compatible wires return inline raster images
in `choices[].delta.images[].image_url.url` while streaming and
`choices[].message.images[].image_url.url` when buffered. Each URL is a complete base64 PNG,
JPEG, WebP, or GIF. Text and token usage remain in their ordinary fields. Replay generated
images as assistant `content` parts with `type: "image_url"`; Gemini reconstructs caller-owned
history with its documented signature-validation bypass, as it does for replayed function calls.
This does not preserve provider-private reasoning signatures. Assistant image history is refused
on fallback wires that cannot preserve it.
Image lanes request text and images explicitly on Gemini and OpenRouter, allow bounded SSE
frames up to 64 MiB, and share the 64 MiB aggregate output bound. Output guardrails that inspect
text cannot inspect image pixels, so guarded image lanes buffer and fail closed before delivery.
Raster bytes must decode within 16,384 pixels per dimension and a 128 MiB decode allocation bound.
Known image-emitting models on unsupported completion surfaces are refused before dispatch; use
Chat Completions or a separately supported Images API route. The Images API capability remains
independent.
