"""Prepare and execute simulation fidelity through existing durable runtime owners."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from filelock import FileLock
from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import DualReport, EvaluationReferences, JudgeConfig
from exp.common.models import load_model_catalog
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import ResolvedModel
from exp.runtime.models.structured_completion import (
    CompletionPlan,
    ModelCompletion,
    StructuredCompletion,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.runtime.sessions.store import SessionStore
from exp.simulation.evaluation import FixedWorkflow, evaluate_workflow
from exp.simulation.retrieval.jev import JEVCueSelector, JEVPolicy, JEVSelector
from exp.simulation.retrieval.selection import EvidencePolicy
from exp.simulation.world_model.durable import SessionPolicy
from exp.simulation.world_model.experience_memory import (
    MemoryVersion,
    memory_head,
    memory_records,
    publish_trajectory,
    read_memory_version,
    require_memory_scope,
)
from exp.simulation.world_model.grounded_session import open_grounded_session
from exp.simulation.world_model.runtime import (
    load_grounded_world_model,
    load_grounded_world_model_artifact,
)
from exp.simulation.world_model.session_research import OnlineResearchPolicy


class FidelityRetrieval(ContractModel):
    """Native JEV relevance decisions through a configured decision-model alias.

    Attributes:
        jev_alias: Catalog alias served by the native decision wire.
        jev: Frozen rubric; it must equal any reranker bound in the simulator artifact.
        memory_top_k: Maximum research records delivered to one prediction.
    """

    jev_alias: str = Field(min_length=1)
    jev: JEVPolicy
    memory_top_k: int = Field(default=4, gt=0, strict=True)


class FidelityMemory(ContractModel):
    """Cross-trajectory memory: learn from this trajectory, or evaluate a frozen checkpoint.

    Attributes:
        lineage: Project-local versioned memory sequence.
        mode: ``learn`` pins the current head and publishes the next version after a
            completed trajectory; ``frozen`` reads an exact checkpoint and never publishes.
        checkpoint: Exact frozen version; absent in learning mode.
    """

    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    mode: Literal["learn", "frozen"]
    checkpoint: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def checkpoint_mode(self) -> "FidelityMemory":
        """Reserved evaluation names its checkpoint; learning always follows the head."""
        if (self.mode == "frozen") != (self.checkpoint is not None):
            raise ValueError("frozen memory requires a checkpoint; learning uses the head")
        return self


class FidelitySpec(ContractModel):
    """Fixed actions and independent references for an existing simulation.

    Attributes:
        run_id: New logical run identity; repeated completed runs replay saved results.
        world_model: Exact existing simulation manifest, including rules and retrieval policy.
        task: Public instruction and current observations, without a reference answer.
        workload: Fixed supplied actions, not a live actor-equivalence experiment.
        references: Evaluator-only outcomes; never passed into prediction or retrieval.
        judge: Frozen five-axis rubric and prompt.
        judge_alias: Configured model used only for independent judging.
        deadline: Absolute dispatch and commit cutoff.
        maximum_physical_attempts: Global ceiling including provider retries and judging.
        maximum_output_tokens: Explicit per-generation output allowance.
        json_object_output: Declared JSON-mode treatment for supported prediction routes.
        output_protocol: Plain text framing or typed native-value proposals.
        request_reservation_usd: Accounting estimate, not an asserted tariff upper bound.
        online_research: Optional declared per-step reply feedback; absent freezes knowledge.
        retrieval: Native JEV selection for originals and memory, resolved from the catalog.
        memory: Cross-trajectory memory lineage and its learning or frozen mode.
    """

    run_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,79}$")
    world_model: ArtifactInput
    task: TaskCase
    workload: FixedWorkflow
    references: EvaluationReferences
    judge: JudgeConfig
    judge_alias: str = Field(min_length=1)
    deadline: AwareDatetime
    maximum_physical_attempts: int = Field(gt=0, strict=True)
    maximum_output_tokens: int = Field(gt=0, strict=True)
    json_object_output: bool = Field(default=False, strict=True)
    output_protocol: Literal["text", "json-tool"] = "text"
    request_reservation_usd: float = Field(gt=0, allow_inf_nan=False)
    online_research: OnlineResearchPolicy | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    retrieval: FidelityRetrieval | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    memory: FidelityMemory | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def separate_references(self) -> "FidelitySpec":
        """Bind action inventories before any model or credential is consulted."""
        if self.output_protocol == "json-tool" and self.json_object_output:
            raise ValueError("Typed proposal and JSON text modes are distinct protocols")
        if self.task.reference_answer is not None or self.task.instruction != self.workload.task:
            raise ValueError("public task must match workload and exclude reference answers")
        if self.references.steps and (
            len(self.references.steps) != len(self.workload.actions)
            or any(
                reference.action != {"name": action.name, "arguments": action.arguments}
                for reference, action in zip(
                    self.references.steps, self.workload.actions, strict=True
                )
            )
        ):
            raise ValueError("references must bind exactly the supplied public actions")
        if self.online_research is not None and (
            len(self.references.steps) != len(self.workload.actions)
            or any(not row.reply_available for row in self.references.steps)
        ):
            raise ValueError("online research requires explicitly disclosed post-step replies")
        if self.memory is not None and self.retrieval is None:
            raise ValueError("memory delivery requires a registered JEV retrieval policy")
        if self.memory is not None and (self.memory.mode == "learn") != (
            self.online_research is not None
        ):
            raise ValueError("memory learning needs feedback; frozen checkpoints exclude it")
        if (
            self.retrieval is not None
            and self.online_research is not None
            and (self.online_research.selection_policy_sha256 != self.memory_selection_identity)
        ):
            raise ValueError("online research must register the spec's JEV memory selector")
        return self

    @property
    def memory_selection_identity(self) -> str | None:
        """Identity of the JEV memory selector this spec constructs, without a provider."""
        if self.retrieval is None:
            return None
        return JEVCueSelector(
            policy=self.retrieval.jev,
            top_k=self.retrieval.memory_top_k,
            decide=_never,
            record=_discard,
        ).identity


def _never(request: DecisionRequest) -> JsonObject:
    """Identity-only selector seam; never dispatches."""
    raise AssertionError("identity selector cannot decide")


def _discard(receipt: JsonObject) -> None:
    """Identity-only selector seam; records nothing."""


class PreparedFidelity(ContractModel):
    """Immutable evaluation registration, without credentials or another execution loop.

    Attributes:
        id: Content-derived plan identity.
        spec: Complete frozen evaluation inputs and limits.
        catalog_sha256: Model configuration identity; credential references stay local.
        code_revision: Installed producer identity.
    """

    id: str
    spec: FidelitySpec
    catalog_sha256: str
    code_revision: str


def prepare_fidelity_evaluation(
    project: ProjectStore, spec: FidelitySpec, *, code_revision: str
) -> PreparedFidelity:
    """Verify and freeze a workload without provider calls or credential resolution.

    Args:
        project: Existing project owning the simulation and model catalog.
        spec: Public workload plus separately typed evaluator references.
        code_revision: Exact producing implementation identity.

    Returns:
        Saved registration; reusing a run ID with different inputs is rejected.
    """
    spec = FidelitySpec.model_validate_json(spec.model_dump_json())
    plan_id = stable_id("fidelity", {"run_id": spec.run_id})
    if project.artifacts.exists(plan_id):
        saved = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(plan_id, "plan.json")
        )
        if saved.spec != spec:
            raise ValueError("fidelity run inputs changed; use the saved spec or a new run ID")
        return saved
    manifest = project.artifacts.read(spec.world_model.artifact_id).manifest
    if artifact_input(manifest) != spec.world_model:
        raise ValueError("simulation manifest differs from the supplied frozen reference")
    artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    catalog = load_model_catalog(project.model_catalog_path)
    registry = RuntimeModelCatalog(catalog)
    predictor, _ = registry.snapshot(artifact.model_alias)
    judge_model, judge_caps = registry.snapshot(spec.judge_alias)
    if predictor != artifact.model:
        raise ValueError("configured predictor differs from the simulation; bind a new artifact")
    if spec.judge.model_identity != judge_model.model_id or judge_caps.supports_tools is not True:
        raise ValueError("judge must match the rubric model and support named structured output")
    for alias in (artifact.model_alias, spec.judge_alias):
        connection = catalog.connections[catalog.models[alias].connection]
        if connection.provider != "openai-compatible":
            raise ValueError("this fidelity execution profile requires openai-compatible HTTP")
    bound = artifact.selection_policy_sha256
    if bound is not None and (spec.retrieval is None or spec.retrieval.jev.identity != bound):
        raise ValueError("simulator binds a JEV reranker; register the same policy in retrieval")
    if spec.retrieval is not None:
        record = catalog.models.get(spec.retrieval.jev_alias)
        if record is None or catalog.connections[record.connection].provider != "typesafe":
            raise ValueError("JEV retrieval requires a configured native decision-model alias")
    if spec.memory is not None and spec.memory.mode == "frozen":
        assert spec.memory.checkpoint is not None
        frozen = read_memory_version(project.artifacts, spec.memory.lineage, spec.memory.checkpoint)
        require_memory_scope(
            frozen,
            world_model=spec.world_model,
            selection_policy_sha256=str(spec.memory_selection_identity),
        )
    prepared = PreparedFidelity(
        id=plan_id,
        spec=spec,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        code_revision=code_revision,
    )
    project.artifacts.write_json(
        artifact_id=plan_id,
        artifact_type="fidelity-plan",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            inputs=(spec.world_model,),
            code_revision=code_revision,
        ),
        files={"plan.json": prepared.model_dump(mode="json")},
    )
    return prepared


def saved_fidelity_report(project: ProjectStore, prepared: PreparedFidelity) -> DualReport | None:
    """Read completed evidence without credentials, provider calls, or a live deadline."""
    identity = prepared.id + "-report"
    if not project.artifacts.exists(identity):
        return None
    return DualReport.model_validate_json(project.artifacts.read_bytes(identity, "report.json"))


def execute_fidelity_evaluation(
    project: ProjectStore,
    prepared: PreparedFidelity,
    *,
    progress: ProgressHook | None = None,
    research_selection: EvidencePolicy | None = None,
    clock: Callable[[], datetime] | None = None,
) -> DualReport:
    """Run after explicit spend consent, or replay a completed evaluation exactly.

    Args:
        project: Owner of the verified registration and durable execution artifacts.
        prepared: Exact saved plan. The caller authorizes unpriced spend explicitly;
            physical-attempt, per-output, and deadline bounds remain enforced.
        progress: Existing observational progress hook shared with the CLI.
        research_selection: Injected session-only selector for specs without retrieval.
        clock: Optional aware clock for qualification.

    Returns:
        Separate mechanical and mandatory judge results. Incomplete earlier dispatches
        stop visibly rather than authorizing an automatic repurchase.
    """
    now = clock or (lambda: datetime.now(UTC))
    saved = PreparedFidelity.model_validate_json(
        project.artifacts.read_bytes(prepared.id, "plan.json")
    )
    if saved != prepared:
        raise ValueError("execution differs from the saved fidelity registration")
    directory = project.artifacts.project_directory
    with FileLock(directory / (prepared.id + ".lock"), timeout=0):
        previous = saved_fidelity_report(project, prepared)
        intent = prepared.id + "-execution"
        if previous is not None:
            report(progress, "Saved fidelity result")
            _consume_trajectory(project, prepared, now())
            return previous
        if project.artifacts.exists(intent):
            raise ValueError(
                "incomplete fidelity run; inspect saved operations before new dispatch"
            )
        spec = prepared.spec
        if spec.retrieval is not None and research_selection is not None:
            raise ValueError("registered JEV retrieval cannot be replaced by an injected selector")
        if (
            spec.online_research is not None
            and spec.retrieval is None
            and research_selection is None
        ):
            raise ValueError("online research needs its registered memory selector")
        if (
            spec.online_research is not None
            and research_selection is not None
            and (research_selection.identity != spec.online_research.selection_policy_sha256)
        ):
            raise ValueError("memory selector differs from the registered fidelity spec")
        if now() >= spec.deadline:
            raise ValueError("fidelity deadline expired before dispatch")
        identity = sha256_json(prepared.model_dump(mode="json"))
        ledger = DispatchLedger(
            directory / (prepared.id + "-physical.sqlite"),
            spec.maximum_physical_attempts,
            identity,
        )
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(),
            ledger=ledger,
            artifacts=project.artifacts,
            code_revision=prepared.code_revision,
            deadline=spec.deadline,
            clock=now,
        )
        catalog = load_model_catalog(project.model_catalog_path)
        if sha256_json(catalog.model_dump(mode="json")) != prepared.catalog_sha256:
            raise ValueError("model configuration changed; prepare a new fidelity run")
        registry = RuntimeModelCatalog(catalog, transport_factory=lambda: transport)
        artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
        predictor = registry.resolve(artifact.model_alias)
        judge_model = registry.resolve(spec.judge_alias)
        budget = RequestBudget(project, identity=identity, maximum_cost_usd=None)
        originals: JEVSelector | None = None
        prior: MemoryVersion | None = None
        prior_identity: JsonObject | None = None
        if spec.retrieval is not None:
            decide = _decision_caller(
                registry.resolve(spec.retrieval.jev_alias), transport, spec.deadline, now
            )
            record = _receipt_recorder(project, prepared, now)
            if artifact.selection_policy_sha256 is not None:
                originals = JEVSelector(policy=spec.retrieval.jev, decide=decide, record=record)
            research_selection = JEVCueSelector(
                policy=spec.retrieval.jev,
                top_k=spec.retrieval.memory_top_k,
                decide=decide,
                record=record,
            )
        if spec.memory is not None:
            prior = (
                memory_head(project.artifacts, spec.memory.lineage)
                if spec.memory.checkpoint is None
                else read_memory_version(
                    project.artifacts, spec.memory.lineage, spec.memory.checkpoint
                )
            )
            require_memory_scope(
                prior,
                world_model=spec.world_model,
                selection_policy_sha256=str(spec.memory_selection_identity),
            )
            prior_identity = {
                "lineage": spec.memory.lineage,
                "mode": spec.memory.mode,
                "version": 0 if prior is None else prior.version,
                "consumed_trajectories": 0 if prior is None else len(prior.consumed),
                "library": None if prior is None else prior.library.model_dump(mode="json"),
            }

        def completion(resolved: ResolvedModel, *, structured: bool) -> StructuredCompletion:
            """Share provider resolution, physical admission and logical accounting."""
            return StructuredCompletion(
                client=resolved.client,
                plan=CompletionPlan(
                    model=resolved.snapshot,
                    capabilities=resolved.capabilities,
                    deadline=spec.deadline,
                    output_tokens=spec.maximum_output_tokens,
                    maximum_cost_per_call=spec.request_reservation_usd,
                    code_revision=prepared.code_revision,
                    output_protocol="tool" if structured else "json",
                ),
                budget=budget,
                artifacts=project.artifacts,
                clock=now,
            )

        prediction = completion(predictor, structured=False)
        judging = completion(judge_model, structured=True)
        runtime = load_grounded_world_model(
            project.artifacts,
            spec.world_model.artifact_id,
            client=predictor.client,
            capabilities=predictor.capabilities,
            evidence_selector=originals,
        )
        project.artifacts.write_json(
            artifact_id=intent,
            artifact_type="fidelity-execution",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=now(),
                code_revision=prepared.code_revision,
                inputs=(artifact_input(project.artifacts.read(prepared.id).manifest),),
            ),
            files={
                "execution.json": {
                    "plan_sha256": identity,
                    **({"memory": prior_identity} if prior_identity is not None else {}),
                }
            },
        )
        report(progress, "Predicting supplied actions")
        session = open_grounded_session(
            runtime=runtime,
            task=spec.task,
            store=SessionStore(directory / (prepared.id + "-sessions.sqlite"), clock=now),
            artifacts=project.artifacts,
            session_id=prepared.id,
            policy=SessionPolicy(
                deadline=spec.deadline,
                maximum_operations=len(spec.workload.actions),
                inference_identity=identity,
                code_revision=prepared.code_revision,
            ),
            complete=ModelCompletion(prediction),
            lock_path=directory / (prepared.id + "-session.lock"),
            maximum_output_tokens=spec.maximum_output_tokens,
            json_object_output=spec.json_object_output,
            output_protocol=spec.output_protocol,
            research_policy=spec.online_research,
            research_selection=research_selection,
            research_complete=completion(predictor, structured=True)
            if spec.online_research is not None
            else None,
            research_prior=memory_records(project.artifacts, prior)
            if prior_identity is not None
            else None,
            research_prior_identity=prior_identity,
            clock=now,
        )

        def judge_call(body: JsonObject, logical_id: str) -> JsonObject:
            """Expose the real judging stage without adding a retry or scoring loop."""
            report(progress, "Judging simulation fidelity")
            return judging(body, logical_id)

        result = evaluate_workflow(
            session,
            spec.workload,
            references=spec.references,
            judge=spec.judge,
            judge_complete=judge_call,
            artifacts=project.artifacts,
            clock=now,
            code_revision=prepared.code_revision,
            evaluation_id=prepared.id + "-report",
            reply_feedback_source=spec.online_research.feedback_source
            if spec.online_research is not None
            else None,
        )
        _consume_trajectory(project, prepared, now())
        report(progress, "Fidelity evaluation completed")
        return result


def saved_memory_receipt(project: ProjectStore, prepared: PreparedFidelity) -> JsonObject | None:
    """Read the memory version a completed learning trajectory produced, if any."""
    identity = prepared.id + "-memory"
    if not project.artifacts.exists(identity):
        return None
    return json.loads(project.artifacts.read_bytes(identity, "memory.json"))


def _consume_trajectory(project: ProjectStore, prepared: PreparedFidelity, now: datetime) -> None:
    """Publish a completed learning trajectory exactly once; replay resumes publication.

    Deterministic code decides admission: only a learning spec whose every planned step
    was predicted and disclosed is consumed. Model-authored investigations enter as
    unresolved research, never as accepted rules.
    """
    spec = prepared.spec
    receipt_id = prepared.id + "-memory"
    if (
        spec.memory is None
        or spec.memory.mode != "learn"
        or spec.online_research is None
        or project.artifacts.exists(receipt_id)
    ):
        return
    pinned = json.loads(project.artifacts.read_bytes(prepared.id + "-execution", "execution.json"))
    memory = pinned.get("memory")
    if not isinstance(memory, dict) or memory.get("lineage") != spec.memory.lineage:
        raise ValueError("learning execution did not pin its memory version")
    evidence = json.loads(
        project.artifacts.read_bytes(prepared.id + "-report-intent", "evidence.json")
    )
    feedback = evidence.get("sources", {}).get("online_feedback", {})
    receipts = [row for row in feedback.get("receipts", []) if isinstance(row, dict)]
    research = [
        str(row["id"])
        for row in receipts
        if isinstance(row.get("investigation"), dict) or row.get("reply_matched") is False
    ]
    outcome: JsonObject = {
        "lineage": spec.memory.lineage,
        "parent_version": memory["version"],
        "research_ids": list(research),
        "feedback_receipts": len(receipts),
    }
    if evidence.get("stop_reason") != "completed" or len(receipts) != len(spec.workload.actions):
        outcome |= {"published": False, "reason": "incomplete trajectory is not consumed"}
    else:
        parent_number = memory["version"]
        parent = (
            read_memory_version(project.artifacts, spec.memory.lineage, int(parent_number))
            if parent_number
            else None
        )
        trajectory: JsonObject = {
            "task": spec.task.instruction,
            "task_lineage": spec.task.lineage_group_id,
            "initial_context": spec.task.initial_context,
            "actions": [{"name": a.name, "arguments": a.arguments} for a in spec.workload.actions],
            "observed_replies": [row.reply for row in spec.references.steps],
            "source": spec.online_research.feedback_source,
        }
        version, consumed = publish_trajectory(
            project.artifacts,
            lineage=spec.memory.lineage,
            parent=parent,
            world_model=spec.world_model,
            selection_policy_sha256=str(spec.memory_selection_identity),
            trajectory=trajectory,
            research=research,
            created_at=now,
            code_revision=prepared.code_revision,
        )
        outcome |= {
            "published": consumed,
            "reason": "consumed" if consumed else "duplicate trajectory already consumed",
            "version": version.version,
            "consumed_trajectories": len(version.consumed),
            "library": version.library.model_dump(mode="json"),
        }
    project.artifacts.write_json(
        artifact_id=receipt_id,
        artifact_type="fidelity-memory",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=now,
            code_revision=prepared.code_revision,
            inputs=(artifact_input(project.artifacts.read(prepared.id + "-report").manifest),),
        ),
        files={"memory.json": outcome},
    )


def _decision_caller(
    resolved: ResolvedModel,
    transport: AdmittedJsonTransport,
    deadline: datetime,
    clock: Callable[[], datetime],
) -> Callable[[DecisionRequest], JsonObject]:
    """Send JEV decisions on the catalog's native wire through the admitted transport."""
    profile_of = getattr(resolved.client, "gateway_wire_profile", None)
    if profile_of is None:
        raise ValueError("JEV alias does not resolve to a native decision client")
    profile = profile_of()
    if profile.dialect != "typesafe_systemone" or not profile.decisions_url:
        raise ValueError("JEV alias does not expose the native decision wire")

    def decide(request: DecisionRequest) -> JsonObject:
        """Dispatch one decision; physical attempts share the run's ledger and deadline."""
        remaining = (deadline - clock()).total_seconds()
        response = transport.post(
            profile.decisions_url,
            headers=profile.headers,
            payload=request.provider_body(profile.model_id),
            timeout_seconds=max(min(remaining, 120.0), 0.001),
        )
        if response.status_code != 200 or not isinstance(response.body, dict):
            raise RuntimeError(f"JEV decision returned HTTP {response.status_code}")
        return response.body

    return decide


def _receipt_recorder(
    project: ProjectStore, prepared: PreparedFidelity, clock: Callable[[], datetime]
) -> Callable[[JsonObject], None]:
    """Retain exact JEV requests, raw answers and selected IDs as immutable receipts."""

    def record(receipt: JsonObject) -> None:
        """Persist one content-addressed receipt beside the run's other artifacts."""
        identity = stable_id("jev-receipt", {"run": prepared.id, "receipt": receipt})
        if not project.artifacts.exists(identity):
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type="jev-receipt",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=clock(), code_revision=prepared.code_revision
                ),
                files={"receipt.json": {"run": prepared.id, **receipt}},
            )

    return record


def fidelity_memory_plan(
    project: ProjectStore, spec: FidelitySpec, prepared: PreparedFidelity | None = None
) -> str | None:
    """Describe the memory a run read (once executed) or would read, without provider calls."""
    if spec.memory is None:
        return None
    intent = None if prepared is None else prepared.id + "-execution"
    if intent is not None and project.artifacts.exists(intent):
        pinned = json.loads(project.artifacts.read_bytes(intent, "execution.json")).get("memory")
        if isinstance(pinned, dict):
            return (
                f"{spec.memory.lineage} {pinned.get('mode')} run read v{pinned.get('version')} "
                f"({pinned.get('consumed_trajectories')} consumed trajectories)"
            )
    if spec.memory.checkpoint is not None:
        frozen = read_memory_version(project.artifacts, spec.memory.lineage, spec.memory.checkpoint)
        return (
            f"{spec.memory.lineage} frozen checkpoint v{frozen.version} "
            f"({len(frozen.consumed)} consumed trajectories); outcomes never update memory"
        )
    head = memory_head(project.artifacts, spec.memory.lineage)
    count = 0 if head is None else len(head.consumed)
    return (
        f"{spec.memory.lineage} learning from v{0 if head is None else head.version} "
        f"({count} consumed trajectories); a completed trajectory publishes the next version"
    )


def fidelity_decision_trace(project: ProjectStore, prepared: PreparedFidelity) -> list[JsonObject]:
    """Return per-step model decisions and deterministic commits from saved artifacts only."""
    path = project.artifacts.project_directory / (prepared.id + "-sessions.sqlite")
    if not path.exists():
        return []
    state = SessionStore(path).read_session(prepared.id).state
    transitions = state.get("transitions")
    receipts: dict[str, JsonObject] = {}
    evidence_id = prepared.id + "-report-intent"
    if project.artifacts.exists(evidence_id):
        evidence = json.loads(project.artifacts.read_bytes(evidence_id, "evidence.json"))
        feedback = evidence.get("sources", {}).get("online_feedback", {})
        receipts = {
            str(row["operation_id"]): row
            for row in feedback.get("receipts", [])
            if isinstance(row, dict)
        }
    rows: list[JsonObject] = []
    for index, transition in enumerate(transitions if isinstance(transitions, list) else []):
        if not isinstance(transition, dict):
            continue
        evidence = transition.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        selection = evidence.get("session_research_selection")
        feedback_row = receipts.get(str(transition.get("operation_id")), {})
        investigation = feedback_row.get("investigation")
        rows.append(
            {
                "step": index,
                "committed_version": index + 1,
                "request_sha256": evidence.get("request_sha256"),
                "originals": evidence.get("grounded_transition_ids"),
                "memory_selected": selection.get("selected_ids")
                if isinstance(selection, dict)
                else None,
                "proposal": transition.get("proposal"),
                "reply_matched": feedback_row.get("reply_matched"),
                "research_id": feedback_row.get("id") if isinstance(investigation, dict) else None,
                "hypothesis": investigation.get("hypothesis")
                if isinstance(investigation, dict)
                else None,
            }
        )
    return rows
