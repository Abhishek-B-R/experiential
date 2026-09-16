# Gateway capture hosting interface

`exp.runtime.gateway.capture` owns the stored message/response shape and the
bounded lifecycle handoff. A hosting application supplies consent and a typed
`CaptureWriter`. No database, provider connection, or hosted account is required
to use the capture primitives.

The public lifecycle is `serialize_capture_messages` -> `PromptCaptureBuffer`
-> `CaptureWriter.enqueue` after durable request acceptance. Response capture
uses `ResponseCaptureRegistry` and `ResponseCaptureHandoff` to reconcile final
settlement with the relayed response. The host's destination receives typed
`PromptCapturePayload`, `ResponseCapturePayload`, and `CaptureDiscard` records.

The destination must preserve enqueue order. A response or discard must not
overtake its accepted prompt. The host remains responsible for consent checks
at the durable write, tenant authorization, retention, and deletion. Failure of
observability persistence must not become failure of serving or accounting.

The host registers response permission only after its final settlement decision.
The relay claims that permission before parsing a completed response. If the
caller disconnects first, `handoff.park` retains a bounded prefix until settlement
either claims it or it expires. Both arrival orders preserve the same explicit
`client_disconnected` and `truncated` markers. A replay has no new settlement
permission and therefore cannot create a second content capture.

Serialization changes only the saved copy. Message arrays preserve the canonical
gateway message shape. Responses retain their complete JSON body or ordered SSE
data frames, including unknown fields. Invalid durable-text characters are
normalized without merging object keys. Oversized prompts and non-stream
responses are rejected; oversized streams may retain an explicitly truncated
prefix. The destination must also enforce its own physical storage limits.

These primitives are a hosting interface, not an automatic local collection
workflow. Local gateway launch does not enable content capture through this
module alone. They are separate from anonymous product telemetry and the
content-free request/attempt accounting ledger.
