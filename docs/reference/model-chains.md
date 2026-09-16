# Ordered model chains

A direct Chat Completions, Responses, or Messages alias can reference another model's provider chain. Image generation, embeddings, and project selection keep their direct exact-model semantics. Provider pools still certify only one exact model; a model reference is not equivalence certification.

## Configuration

`ModelCatalog.gateway_model_chains` holds `GatewayModelChain` records keyed by canonical model ID. Each record names its exact-model pool, revision, optional frozen `ModelStagePolicy`, and ordered rungs. A rung is either `GatewayDeploymentRung(deployment_id=...)` or `GatewayModelReferenceRung(model_id=...)`.

A chain permits at most one reference and requires a direct first rung. The child can reference another model. Reciprocal links are legal. Execution marks canonical models visited, skips a reference to an entered model, and resumes the parent's remaining providers when the child is exhausted:

```text
A: a1, reference B, a2
B: b1, reference A, b2

Order after eligible failures: a1, b1, skip repeated A, b2, a2
```

Expansion allows at most sixteen model stages and 256 examined rungs. The request keeps one deadline and an eight-attempt total cap. Skipped references are not provider attempts. No later settlement rewinds an in-flight request.

## Admission and policy

`ExecutionSnapshot.model_stages` freezes model, selected pool, providers, policy, revision, ancestry, and rung positions. `stage_for_depth()` identifies the destination for a physical attempt. Narrowing removes unsupported providers while preserving surviving stage boundaries. Affinity and cache-marker ordering operate inside a segment, never across a reference.

Each stage owns its failover mode, cache threshold, and throttle-redial schedule. A child cannot inherit the root's schedule accidentally. Sticky and reasoning-pinned providers receive their own stage's full redial allowance; the final admitted provider receives its own full allowance too. Every retry still consumes the shared request attempt cap. Exhausting scheduled redials advances rather than reapplying the no-schedule cache-threshold rule.

A live reasoning issuer retains first position after authorization and capability narrowing. Cache or affinity preferences cannot demote it before a real eligible failure. A filtered issuer is not restored. Following an eligible failure, authorized compatible successors can run with provider-bound reasoning removed as required by the existing continuation contract. Meaningful outward output still ends fallback eligibility.

## Accounting and recovery

Each provider call retains the requested alias while its attempt records the actual stage model, pool, prices, and usage. `x-gateway-canonical-model` identifies the model that produced the committed response. Root and destination budget scopes both apply, with equal scopes deduplicated. Hosted stores must implement the same stage checks at their atomic reservation boundary; authoring a reference never grants access or money.

`RecoveryHost` supplies immutable, already-loaded observations and preallocated leases. `SessionRecoveryRegistry` keeps bounded worker-local cache history, separately from aggregate cache fractions and official status feeds. Return decisions require matching failure recovery and this caller's own prefix/account cache evidence. Missing or newer negative evidence blocks an optional return. A sticky child start requires `AuthorizationSnapshot.descendant_start_authorized`; its default is false, so hosted root-funding checks must explicitly authorize that case.

## Verification boundary

Pure tests cover graph expansion, immutable stage projections, policy budgets, expiry, and evidence scope. Native loopback tests exercise all three conversational APIs in streaming and non-streaming forms, checking requested alias, actual-model header, exact attempt order, and closed accounting. These controlled tests do not establish real-provider cache hits, billing invoices, or deployed platform compatibility. Those require tests of the actual released engine, platform, schema, and configuration together.
