# Gateway capture hosting interface

Experiential owns one Rust collector for authenticated request context, HTTP
response capture, bounded lifecycle state and worker-owned delivery. Python
prepares the effective request during admission and configures a destination.
There is no Python callback per response chunk. Database work runs on the
destination worker; eligible response completion waits for its acknowledgement.

```python
from exp_gateway_native import CaptureCollector
from exp.runtime.gateway.native_capture import CaptureConfiguration, CaptureController

configuration = CaptureConfiguration()  # requires a hosted settlement decision
collector = CaptureCollector(configuration.model_dump_json(), write_record)
capture = CaptureController(collector, application_for=allowed_application)
# Pass capture to NativeControlPlane and collector to serve_native_gateway.
# Only after the final hosted lane decision:
collector.settle(request_id, keep_prompt=True, keep_response=served)
# A BYOK lane or denied policy retains nothing:
collector.settle(request_id, keep_prompt=False, keep_response=False)
```

`allowed_application(authorization)` returns an explicitly configured application
id or `None`. Organization and identity always come from authenticated authority,
never request metadata. The projection includes expanded messages, tool definitions,
generation settings and semantic provider context after input guardrails. Transport
replay keys and resolved provider credentials are excluded. Prompts may themselves
contain sensitive information; this is not content redaction or encryption.

The synchronous `write_record(str)` destination runs on a dedicated Rust-owned
worker. Validate its input with `CaptureRecord.model_validate_json`. Schema version
1 includes the authenticated scope, effective request, optional response, model and
deployment provenance, and capture timestamp. The selected model is null for an
accepted request that failed before routing. A successful exchange emits one complete
record after both output completion and permission. A prompt-only permission emits
only the prompt. This avoids a delayed prompt update overwriting a full response.
A hosted collector
does not enqueue content until terminal eligibility permits it, so queue overload
cannot lose a BYOK deletion behind an already queued prompt.

Chat Completions, Responses and Messages HTTP surfaces share the same native tap.
JSON bodies and ordered SSE data payloads retain unknown fields. The observation
boundary is the native HTTP listener, which may feed a hosted relay; it does not
prove that an end user consumed every byte. `truncated` and
`client_disconnected` explicitly distinguish a prefix from complete evidence.
Output and settlement may arrive in either order. Keyed replays do not attach a
second response tap. WebSocket and batch response capture are not added here.

The winning rung's provider-returned plaintext reasoning is retained separately
in `provider_reasoning` when that rung explicitly permits reasoning exposure.
This preserves reasoning even when the public Responses representation carries
only an opaque continuation. Private provider reasoning is not decrypted for
capture. Capture permission never grants permission to expose hidden reasoning.
`provider_tool_calls_json` retains completed calls as escaped JSON, including exact
argument text even when Messages presents the arguments as a parsed input object.
Chat tool turns on exposure-enabled routes return plaintext without appending
an opaque token to the same delta field; private routes retain authenticated tokens.

Postgres cannot represent NUL or lone UTF-16 surrogates. Affected request contexts
and responses include `source_json`, an escaped JSON string containing the exact
source value alongside the normalized query projection. Consumers recover the
request with `restore_capture_context`; response consumers decode `source_json`
when present. Reasoning uses `provider_reasoning_source_json` for the same case.
The escaped sidecars count toward all record limits. Oversize evidence is excluded,
not silently advertised as lossless. Historical reasoning stays in captured input
even when provider execution must omit it at a new user boundary.

The native queue carries structured request records and original response wire
buffers, not expanded response JSON trees. Records own immutable request context.
Only the destination worker parses response JSON, one record at a time.
JSON sizing counts fields and escapes without encoding response buffers; the
destination performs bounded final encoding. A
hosted Python destination receives that encoded record. Native destinations consume
the structure directly. Request admission still crosses the Python/Rust boundary
as JSON; exceptional lossless sidecars and raw tool-call strings also use JSON.

Delivery limits bound record count, each final encoded payload and retained record
memory, including a record currently held by a slow destination. String/vector
capacity and conservative object-node charges are included; shared trees are charged
in every owning queue. The writer additionally owns one bounded response's decoded
tree while encoding and persisting it. Separate bounds cover in-flight entry count and memory, total response-buffer capacity
and request lifetime. Expiration runs on collector operations and once per second
on an idle destination worker; a blocked destination delays idle maintenance but
does not remove the memory caps. A response reserves its complete buffer allowance
before reading any provider bytes. Saturated delivery waits for capacity rather
than discarding records. Sustained storage pressure therefore increases latency and
limits throughput to what the destination can persist.

When capture is required but admission cannot register it, the gateway returns a
sanitized `capture_unavailable` 503 before provider dispatch. Policy-disabled capture
still serves normally. An eligible response does not finish successfully until its
destination write succeeds; a failed write terminates its HTTP body with an error.
Bytes already streamed cannot be withdrawn. Hosted eligibility can arrive after
the response ends, so hosts must also monitor destination failure counters for
these late writes. Destination exceptions never print potentially sensitive details.
`counts()` returns pending records, retained delivery bytes, successful destination calls,
destination failures, delivery drops and collector skips. A bounded `close()` drains
while releasing the GIL; a blocked destination cannot extend that caller's deadline.
A drain timeout returns false and leaves accepted delivery records queued, including
producers already waiting for space. It does not purge them. The host must keep the
process alive to finish draining; this memory queue is not a crash-recovery journal.
Unsettled hosted records still expire or are discarded on close because no durable
content permission has been granted. Per-record size limits and explicit retention
policies still apply; this overload guarantee does not mean unlimited retention.

Destinations must enforce their own current consent, identity ownership, consent
generation, retention and physical storage constraints at the durable write. Native
admission policy is a performance gate, not a replacement for those checks. Capture
does not alter user-visible content, provider attribution, billing or the content-free
accounting ledger. The local CLI integration supplies the same collector with
identity/application bindings and a SQLite sink, without a hosted settlement gate.
