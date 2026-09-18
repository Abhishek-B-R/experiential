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

`RecoveryHost` supplies immutable, already-loaded transport observations and preallocated leases. `CredentialEnvironment.resolve_credential` returns authentication and an opaque worker-local binding receipt together; an authoritative missing result never falls through to another credential source. Static wire profiles freeze that receipt before recovery selection and retain it through direct and swept settlement. Profiles without a receipt, including unbound refreshable signers, serve normally but provide no elective recovery or cache warmth. Shared `OperationalScope` contains only exact resolved endpoint/model/known region topology; the local credential receipt and tenant identity are excluded. Shared account-specific recovery is unavailable without a stable generation returned atomically with the credential. Requests carrying `inference_geo` supply no recovery binding until the requested geography can be proved against the resolved wire; their normal provider payload and serving behavior remain unchanged. `SessionRecoveryRegistry` keeps bounded worker-local cache history, separately from aggregate cache fractions and official status feeds. Return decisions require matching failure recovery and this caller's own prefix/account cache evidence. Missing or newer negative evidence blocks an optional return. A sticky child start requires `AuthorizationSnapshot.descendant_start_authorized`; its default is false, so hosted root-funding checks must explicitly authorize that case.

Pool and deployment budgets may target reachable children of the active direct alias's pinned graph. Authoring validates the snapshot outside the database write lock, then rechecks the exact alias revision inside the write transaction. Snapshot files must be regular, non-symlinked, digest-valid local files. Root pool controls keep the existing no-file path. POSIX reads use directory-relative no-follow handles; Windows reads reject reparse points and hold checked directories against replacement until the file closes. Snapshot parsing has an adjustable authoring-only resource budget, defaulting to 64 MiB. This is not a catalog validity limit. Larger reviewed files can be admitted by setting `[gateway].budget_snapshot_max_bytes` in the project's existing `settings.toml`, or by passing `snapshot_max_bytes` to `SQLiteBudgetStore`. Values must be positive integers no greater than 2^63-1; the error states the required bytes and configured limit. No file I/O runs under the budget write lock. A refused child pool can be skipped only when the atomic ledger returns a `BudgetRefusalBinding` proving it is destination-only for this exact request after every applicable request-wide budget check passed; root, shared, unknown, and mismatched refusals stop the request. A hosted store with later request-wide checks must not issue that binding before those checks pass. Rejected pools are remembered only within that request, never as provider health.

## Snapshot authority and rollout

Budget authoring requires the current supported normalized snapshot schema. It rejects unknown fields at every validated level, then checks the published default-excluding catalog identity against the alias's pinned digest. It does not use the serving reader's tolerant cross-version path. A future schema, unknown policy, or digest mismatch requires rebuilding the snapshot with a supported engine and reactivating the alias before changing child budgets. Root pool controls still need no snapshot read.

An absent or empty `model_chains` field preserves the current pre-chain schema-5 identity. Current-main readers refuse populated chains through their same-schema digest check. Older schema-4 readers treat schema 5 as foreign and can silently discard populated chain policy, including an explicitly unavailable model. The schema number therefore cannot make activation safe: those readers must be excluded by the host capability and rollback floor before any populated chain is published. Direct-only snapshots retain their existing cross-version serving compatibility.

Hosts must prevent publication or activation of populated chains until every eligible reader and writer supports the chain contract. That requires an enforced fleet/build capability check, not a schema-number comparison or an operator's assumption. Keep chain semantics inactive while old, unknown, or unstamped workers can receive the snapshot, and establish a rollback floor before activation. Publishing the engine package alone does not satisfy this deployment condition.

The compiled extension exposes `exp_gateway_native.MODEL_STAGE_CONTRACT_VERSION` as integer `1`. Contract 1 means the native data plane consumes each deployment's actual canonical model identity and stage-local throttle-redial schedule. It does not certify host authorization, funding, or recovery behavior. A host must also verify Python's model-chain contract. Missing, non-integer, or unknown native markers cannot be replaced with a package-version comparison: a higher-numbered build can still lack the required wire handling. Python refuses staged admission without this exact native marker; routes without model stages do not require it.

## Verification boundary

Pure tests cover graph expansion, immutable stage projections, policy budgets, expiry, and evidence scope. Native loopback tests exercise all three conversational APIs in streaming and non-streaming forms, checking requested alias, actual-model header, exact attempt order, and closed accounting. These controlled tests do not establish real-provider cache hits, billing invoices, or deployed platform compatibility. Those require tests of the actual released engine, platform, schema, and configuration together.
