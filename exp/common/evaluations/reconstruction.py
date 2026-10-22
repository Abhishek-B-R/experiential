"""One blinded LM rubric alongside independent available mechanical checks."""

import copy
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from filelock import FileLock
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
    validate_artifact_id,
)
from exp.common.judging.definition import JudgeDefinition
from exp.common.models.client import RequestCapacityExceeded
from exp.common.project import ArtifactStore, artifact_input


class WorkflowStep(ContractModel):
    """One planned slot, including failed and blocked transitions.

    Attributes:
        step: Zero-based position in the frozen workload.
        action: Public action without evaluator-private fields.
        status: Valid consequence or explicit unavailable outcome.
        reply: Actual candidate reply, absent when unavailable.
        before: Candidate-owned prestate, never reference rescue.
        after: Actual poststate, unchanged on failed commit.
        error_type: Sanitized operational failure classification."""

    step: int = Field(ge=0, strict=True)
    action: JsonObject
    status: Literal["valid", "unknown", "unsupported", "failure", "blocked", "unattempted"]
    reply: JsonValue = None
    before: JsonObject | None = None
    after: JsonObject | None = None
    error_type: str | None = None


class WorkflowEvidence(ContractModel):
    """One complete candidate-owned chronology supplied to independent evaluation.

    Attributes:
        workflow_id: Immutable workflow identity.
        knowledge_sha256: Behavioral artifact identity when available.
        task: Public instruction, without scorer guidance.
        task_context: Allowed initial context.
        initial_state: Candidate initializer state.
        steps: Every planned slot in order.
        sources: Admitted reference evidence available to the judge.
        contract: Declared external tool/state interface.
        final_answer: Actor answer, if the workload includes one.
        stop_reason: Actual workflow termination reason."""

    workflow_id: str = Field(min_length=1)
    knowledge_sha256: Sha256 | None = None
    task: str = Field(min_length=1)
    task_context: JsonObject = Field(default_factory=dict)
    initial_state: JsonObject
    steps: tuple[WorkflowStep, ...]
    sources: JsonObject = Field(default_factory=dict)
    contract: JsonObject = Field(default_factory=dict)
    final_answer: JsonValue = None
    stop_reason: str = "unknown"

    @model_validator(mode="after")
    def inventory(self) -> Self:
        """Validate the declared evaluation inventory and unavailable outcomes."""
        if tuple(s.step for s in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("Complete ordered planned steps required")
        stopped = False
        for step in self.steps:
            if stopped and step.status not in ("blocked", "unattempted"):
                raise ValueError("Workflow continues after hard stop")
            if step.status != "valid":
                stopped = True
        return self


class ReferenceStep(ContractModel):
    """Explicitly available independent reference for one action.

    Attributes:
        before: Optional observed prestate, required for full-state or local exact checks.
        step: Planned slot index.
        action: Exact action tied to the reference.
        reply_available: Whether reply truth is actually available.
        reply: Reference consequence, including literal null when declared.
        state_available: Whether the complete poststate is available.
        after: Complete declared poststate when available."""

    before: JsonObject | None = None
    step: int = Field(ge=0, strict=True)
    action: JsonObject
    reply_available: bool = Field(default=False, strict=True)
    reply: JsonValue = None
    state_available: bool = Field(default=False, strict=True)
    after: JsonObject | None = None

    @model_validator(mode="after")
    def available(self) -> Self:
        """Validate the declared evaluation inventory and unavailable outcomes."""
        if self.state_available and self.before is None:
            raise ValueError("Full-state exact checks require observed prestate")
        if self.state_available and self.after is None:
            raise ValueError("Full reference state required")
        if not self.state_available and self.after is not None:
            raise ValueError("Undeclared reference state")
        if not self.reply_available and self.reply is not None:
            raise ValueError("Undeclared reference reply")
        return self


class IndependentCheck(ContractModel):
    """An externally performed task or policy check, separate from transition exactness.

    Attributes:
        name: Check purpose, such as goal or task policy.
        passed: Actual independently determined outcome.
        reference_id: Locator of the check evidence."""

    name: str = Field(min_length=1)
    passed: bool = Field(strict=True)
    reference_id: str = Field(min_length=1)


class EvaluationReferences(ContractModel):
    """Available truth capabilities; descriptions never imply exact native knowledge.

    Attributes:
        capability: Reference source type, not proof of native execution here.
        description: Observation limits and context supplied to the judge.
        steps: Independent transition references when available.
        independent_checks: Goal/policy checks produced by another owner."""

    capability: Literal["native_executor", "recorded_reference_only", "description_only"]
    description: str = ""
    steps: tuple[ReferenceStep, ...] = ()
    independent_checks: tuple[IndependentCheck, ...] = ()

    @model_validator(mode="after")
    def authority(self) -> Self:
        """Validate the declared evaluation inventory and unavailable outcomes."""
        if self.capability == "description_only" and any(
            s.reply_available or s.state_available for s in self.steps
        ):
            raise ValueError("Descriptions do not establish exact native references")
        return self


class JudgeConfig(ContractModel):
    """A versioned five-axis rubric and explicitly provisional judge identity.

    Attributes:
        definition: Criterion inventory and all 1-5 anchors.
        prompt: Exact protocol and evidence instructions.
        model_identity: Registered judge model identity.
        calibration: Provisional until independently validated.
        protocol: Versioned ordered-list or code-owned keyed criterion envelope.
        mechanical_feedback: Explicitly expose blinded exact-check summaries to the judge."""

    definition: JudgeDefinition
    prompt: str = Field(min_length=1)
    model_identity: str = Field(min_length=1)
    calibration: Literal["provisional"] = "provisional"
    protocol: Literal["ordered-v1", "keyed-v2", "keyed-v3"] = "ordered-v1"
    mechanical_feedback: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def rubric(self) -> Self:
        """Validate the declared evaluation inventory and unavailable outcomes."""
        if len(self.definition.dimensions) != 5 or any(
            a.min_score != 1 or a.max_score != 5 for a in self.definition.dimensions
        ):
            raise ValueError("Exactly five fully anchored 1–5 axes required")
        return self


class AxisOutcome(ContractModel):
    """One assessment or explicit reason that assessment is unavailable.

    Attributes:
        dimension_id: Registered rubric criterion.
        status: Assessment, missing evidence, request failure or no dispatch.
        score: Integer 1-5 only for an assessed criterion.
        rationale: Nonblank explanation of score or absence.
        evidence_refs: Actual visible references supporting a score.
        critical_flags: Consequential errors that averaging must not hide."""

    dimension_id: str
    status: Literal["assessed", "unassessable", "judge_failure", "judge_unknown", "not_run"]
    score: int | None = Field(default=None, ge=1, le=5, strict=True)
    rationale: str = Field(min_length=1)
    evidence_refs: tuple[str, ...] = ()
    critical_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def assessment(self) -> Self:
        """Validate the declared evaluation inventory and unavailable outcomes."""
        if not self.rationale.strip():
            raise ValueError("Nonblank explanation required")
        if self.status == "assessed" and (self.score is None or not self.evidence_refs):
            raise ValueError("Assessed axes require score and evidence references")
        if self.status != "assessed" and self.score is not None:
            raise ValueError("Unavailable assessments cannot invent scores")
        return self


class JudgeAxis(AxisOutcome):
    """The model may abstain; only the evaluator records request failures."""

    status: Literal["assessed", "unassessable"]


class JudgeOutput(ContractModel):
    """Model-owned assessments, excluding evaluator-owned failure statuses.

    Attributes:
        axes: Exactly five outcomes in registered rubric order."""

    axes: tuple[JudgeAxis, ...] = Field(min_length=5, max_length=5)


class DualReport(ContractModel):
    """Independent mechanical truth and a mandatory, explicit judge outcome.

    Attributes:
        workflow_id: Evaluated workload identity.
        knowledge_sha256: Frozen behavioral artifact identity.
        evidence_sha256: Actual chronology content identity.
        reference_sha256: Independent reference content identity.
        judge_config_sha256: Exact model, prompt and rubric identity.
        planned_steps: Complete planned denominator, including blocked suffixes.
        judge_status: Completed, failed, or not dispatched.
        judge_failure: Explicit unavailable/failure explanation.
        axes: All five scores or all explicit unavailable outcomes.
        mechanical: Independent exactness, continuity and availability measures."""

    workflow_id: str
    knowledge_sha256: Sha256 | None = None
    evidence_sha256: str
    reference_sha256: str
    judge_config_sha256: str
    planned_steps: int
    judge_status: Literal["completed", "failed", "unknown", "not_run"]
    judge_failure: str | None = None
    axes: tuple[AxisOutcome, ...]
    mechanical: JsonObject

    @model_validator(mode="after")
    def judge_inventory(self) -> Self:
        """Validate the declared evaluation inventory and unavailable outcomes."""
        if len(self.axes) != 5 or len({a.dimension_id for a in self.axes}) != 5:
            raise ValueError("Every dual report requires five distinct judge outcomes")
        if self.judge_status == "completed" and any(
            a.status not in ("assessed", "unassessable") for a in self.axes
        ):
            raise ValueError(
                "Completed judging requires assessed or explicitly unassessable outcomes"
            )
        if self.judge_status == "failed" and any(a.status != "judge_failure" for a in self.axes):
            raise ValueError("Failed judging must retain all five explicit failure outcomes")
        if self.judge_status == "unknown" and any(a.status != "judge_unknown" for a in self.axes):
            raise ValueError("Ambiguous judging requires five explicit unknown outcomes")
        if self.judge_status == "not_run" and any(a.status != "not_run" for a in self.axes):
            raise ValueError("Undispatched judging requires five explicit not-run outcomes")
        return self


def default_judge(
    model_identity: str,
    *,
    rubric: Literal["behavior", "comparison"] = "behavior",
    protocol: Literal["ordered-v1", "keyed-v2", "keyed-v3"] = "ordered-v1",
    mechanical_feedback: bool = False,
) -> JudgeConfig:
    """Return separately versioned behavioral or comparison-style provisional criteria."""
    assets = Path(__file__).with_name("reconstruction_prompts")
    definition = JudgeDefinition.model_validate_json((assets / (rubric + ".json")).read_text())
    prompt = (
        assets / ("judge-keyed.txt" if protocol in ("keyed-v2", "keyed-v3") else "judge.txt")
    ).read_text()
    if protocol == "keyed-v3":
        prompt += (
            "\nProtocol keyed-v3: Every criterion must explicitly include score. "
            "Use an integer 1-5 for assessed, or null for unassessable. Never omit score.\n"
        )
    return JudgeConfig(
        definition=definition,
        prompt=prompt,
        protocol=protocol,
        mechanical_feedback=mechanical_feedback,
        model_identity=model_identity,
    )


def judge_wire_schema(config: JudgeConfig) -> JsonObject:
    """Derive the ordered wire inventory from the registered rubric, not a second list."""
    schema = JudgeOutput.model_json_schema()
    axis_schema = schema["$defs"]["JudgeAxis"]
    if config.protocol in ("keyed-v2", "keyed-v3"):
        if config.protocol == "keyed-v3":
            axis_schema["required"] = list(
                dict.fromkeys(
                    [*axis_schema["required"], "score", "evidence_refs", "critical_flags"]
                )
            )
        item = copy.deepcopy(axis_schema)
        item["properties"].pop("dimension_id")
        item["required"] = [key for key in item["required"] if key != "dimension_id"]
        schema["properties"]["axes"] = {
            "type": "object",
            "properties": {
                d.dimension_id: copy.deepcopy(item) for d in config.definition.dimensions
            },
            "required": [d.dimension_id for d in config.definition.dimensions],
            "additionalProperties": False,
        }
        return schema
    ordered = []
    for dimension in config.definition.dimensions:
        item = copy.deepcopy(axis_schema)
        item["properties"]["dimension_id"] = {"type": "string", "const": dimension.dimension_id}
        ordered.append(item)
    schema["properties"]["axes"] = {
        "type": "array",
        "prefixItems": ordered,
        "items": False,
        "minItems": len(ordered),
        "maxItems": len(ordered),
    }
    return schema


def projection(evidence: WorkflowEvidence, references: EvaluationReferences) -> JsonObject:
    """Remove provenance labels, not meaningful entity IDs or domain fields."""
    return {
        "workflow": "workflow-0",
        "task": evidence.task,
        "task_context": evidence.task_context,
        "initial_state": evidence.initial_state,
        "steps": [s.model_dump(mode="json") for s in evidence.steps],
        "sources": {f"source-{i}": value for i, value in enumerate(evidence.sources.values())},
        "contract": evidence.contract,
        "final_answer": evidence.final_answer,
        "stop_reason": evidence.stop_reason,
        "reference_capability": references.capability,
        "description": references.description,
        "reference_steps": [s.model_dump(mode="json") for s in references.steps],
        "independent_checks": [
            {"check_id": f"check-{i}", "name": check.name, "passed": check.passed}
            for i, check in enumerate(references.independent_checks)
        ],
    }


def mechanical_checks(evidence: WorkflowEvidence, references: EvaluationReferences) -> JsonObject:
    """Compare saved replies/states and carried continuity without executing a native target."""
    if references.steps and len(references.steps) != len(evidence.steps):
        raise ValueError("Reference inventory differs from planned workflow")
    rows: list[JsonValue] = []
    current = evidence.initial_state
    continuity = True
    continuity_available = True
    continuity_checked_steps = 0
    full_reference = bool(references.steps) and all(
        r.reply_available and r.state_available for r in references.steps
    )
    initial_matches = (
        strict_equal(evidence.initial_state, references.steps[0].before)
        if references.steps and references.steps[0].before is not None
        else None
    )
    workflow_exact = initial_matches is True
    horizon_results: JsonObject = {}
    for step in evidence.steps:
        reference = references.steps[step.step] if references.steps else None
        if reference and (
            reference.step != step.step or not strict_equal(reference.action, step.action)
        ):
            raise ValueError("Reference action/step binding differs")
        valid = step.status == "valid"
        context_matches = (
            reference is not None
            and step.before is not None
            and strict_equal(step.before, reference.before)
        )
        reply = (
            None
            if reference is None or not reference.reply_available or not context_matches
            else valid and strict_equal(step.reply, reference.reply)
        )
        state = (
            None
            if reference is None or not reference.state_available or not context_matches
            else valid and strict_equal(step.after, reference.after)
        )
        trajectory_reply = (
            None
            if reference is None or not reference.reply_available
            else valid and strict_equal(step.reply, reference.reply)
        )
        trajectory_state = (
            None
            if reference is None or not reference.state_available
            else valid and strict_equal(step.after, reference.after)
        )
        if step.status not in ("blocked", "unattempted"):
            if step.before is None or (valid and step.after is None):
                continuity_available = False
            else:
                continuity_checked_steps += 1
                if not strict_equal(step.before, current):
                    continuity = False
        if valid and step.after is not None:
            current = step.after
        workflow_exact = workflow_exact and trajectory_reply is True and trajectory_state is True
        if step.step + 1 in (6, 12, 24):
            prefix_reference = bool(references.steps) and all(
                r.reply_available and r.state_available for r in references.steps[: step.step + 1]
            )
            horizon_results[str(step.step + 1)] = (
                (workflow_exact and continuity and continuity_available)
                if prefix_reference
                else None
            )
        rows.append(
            {
                "step": step.step,
                "status": step.status,
                "reference_context_matches": context_matches,
                "reply_exact": reply,
                "trajectory_state_exact": trajectory_state,
                "trajectory_reply_exact": trajectory_reply,
                "state_exact": state,
            }
        )
    return {
        "capability": references.capability,
        "initial_state_exact": initial_matches,
        "native_execution_performed_by_evaluator": False,
        "native_execution_status": "not_performed_saved_comparison_only",
        "exact_reference_available": full_reference,
        "planned_steps": len(evidence.steps),
        "valid_steps": sum(s.status == "valid" for s in evidence.steps),
        "blocked_steps": sum(s.status == "blocked" for s in evidence.steps),
        "failed_steps": sum(
            s.status in ("failure", "unknown", "unsupported") for s in evidence.steps
        ),
        "unattempted_steps": sum(s.status == "unattempted" for s in evidence.steps),
        "workflow_joint_exact": (workflow_exact and continuity and continuity_available)
        if full_reference
        else None,
        "steps": rows,
        "carried_continuity": continuity
        if continuity_available and continuity_checked_steps
        else None,
        "continuity_checked_steps": continuity_checked_steps,
        "horizon_joint_exact": horizon_results,
        "first_semantic_divergence": next(
            (
                row["step"]
                for row in rows
                if isinstance(row, dict)
                and row["status"] == "valid"
                and (
                    row["trajectory_reply_exact"] is False or row["trajectory_state_exact"] is False
                )
            ),
            None,
        ),
        "first_hard_stop": next((s.step for s in evidence.steps if s.status != "valid"), None),
        "independent_checks": [c.model_dump(mode="json") for c in references.independent_checks],
        "goal_check_available": any(c.name == "goal" for c in references.independent_checks),
    }


class JudgeNotRun(RuntimeError):
    """The registered judge was not dispatched, for example because budget was exhausted."""


class JudgeDispatchUnknown(RuntimeError):
    """A consumed judge intent may have reached the provider but has no saved result."""


def _evaluate_locked(
    evidence: WorkflowEvidence,
    references: EvaluationReferences,
    *,
    judge: JudgeConfig,
    complete: Callable[[JsonObject, str], JsonObject],
    artifacts: ArtifactStore,
    artifact_id: str,
    clock: Callable[[], datetime],
    code_revision: str,
) -> DualReport:
    """Retain a mechanical result and mandatory judge outcome with immutable evidence.

    A missing reference is not success. Unassessable, failed and undispatched judgments
    are different statuses. No score can override the independent mechanical outcome.
    """
    mechanical = mechanical_checks(evidence, references)
    public = projection(evidence, references)
    if judge.mechanical_feedback:
        findings = copy.deepcopy(mechanical)
        findings["independent_checks"] = public["independent_checks"]
        public["mechanical_findings"] = findings
    allowed = (
        set(public)
        | {"workflow-0"}
        | {f"step-{s.step}" for s in evidence.steps}
        | {f"reference-step-{s.step}" for s in references.steps}
        | {f"source-{i}" for i in range(len(evidence.sources))}
        | {f"check-{i}" for i in range(len(references.independent_checks))}
    )
    body: JsonObject = {
        "response_schema": judge_wire_schema(judge),
        "messages": [
            {"role": "system", "content": judge.prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "rubric": judge.definition.model_dump(mode="json"),
                        "required_dimension_order": [
                            d.dimension_id for d in judge.definition.dimensions
                        ],
                        "output_schema": judge_wire_schema(judge),
                        "allowed_evidence_refs": sorted(allowed),
                        "workflow": public,
                    }
                ),
            },
        ],
    }
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=clock(), inputs=(), code_revision=code_revision
    )
    intent_files: dict[str, JsonValue] = {
        "evidence.json": evidence.model_dump(mode="json"),
        "references.json": references.model_dump(mode="json"),
        "judge.json": judge.model_dump(mode="json"),
        "request.json": body,
    }
    intent_exists = artifacts.exists(artifact_id + "-intent")
    if intent_exists:
        intent = artifacts.read(artifact_id + "-intent")
        if intent.manifest.code_revision != code_revision or any(
            sha256_json(json.loads(artifacts.read_bytes(artifact_id + "-intent", name)))
            != sha256_json(value)
            for name, value in intent_files.items()
        ):
            raise ValueError("saved evaluation identity differs from the current inputs")
        if artifacts.exists(artifact_id):
            return DualReport.model_validate_json(artifacts.read_bytes(artifact_id, "report.json"))
    else:
        artifacts.write_json(
            artifact_id=artifact_id + "-intent",
            artifact_type="workflow-evaluation-intent",
            envelope=envelope,
            files=intent_files,
        )
    status: Literal["completed", "failed", "unknown", "not_run"] = "completed"
    failure = None
    try:
        if artifacts.exists(artifact_id + "-response"):
            raw = json.loads(artifacts.read_bytes(artifact_id + "-response", "response.json"))
        else:
            if intent_exists:
                raise JudgeDispatchUnknown(
                    "prior judge dispatch has no saved response; billing unknown"
                )
            raw = complete(body, artifact_id)
            artifacts.write_json(
                artifact_id=artifact_id + "-response",
                artifact_type="workflow-judge-response",
                envelope=envelope,
                files={"response.json": raw},
            )
        if judge.protocol in ("keyed-v2", "keyed-v3"):
            keyed = raw.get("axes")
            ids = tuple(d.dimension_id for d in judge.definition.dimensions)
            if set(raw) != {"axes"} or not isinstance(keyed, dict) or set(keyed) != set(ids):
                raise ValueError("judge keyed axes differ from the registered inventory")
            converted: list[JsonValue] = []
            for dimension_id in ids:
                value = keyed[dimension_id]
                if not isinstance(value, dict) or "dimension_id" in value:
                    raise ValueError("keyed judge assessments cannot redefine criterion identity")
                converted.append({"dimension_id": dimension_id, **value})
            output = JudgeOutput.model_validate({"axes": converted})
        else:
            output = JudgeOutput.model_validate(raw)
        if tuple(a.dimension_id for a in output.axes) != tuple(
            a.dimension_id for a in judge.definition.dimensions
        ):
            raise ValueError("judge axes differ from the registered rubric inventory/order")

        if any(set(axis.evidence_refs) - allowed for axis in output.axes):
            raise ValueError("judge cites unavailable evidence")
        axes = tuple(
            AxisOutcome.model_validate(axis.model_dump(mode="json")) for axis in output.axes
        )
    except JudgeDispatchUnknown as error:
        status, failure = "unknown", str(error)
        axes = tuple(
            AxisOutcome(dimension_id=a.dimension_id, status="judge_unknown", rationale=failure)
            for a in judge.definition.dimensions
        )
    except (JudgeNotRun, RequestCapacityExceeded) as error:
        status, failure = "not_run", str(error)
        axes = tuple(
            AxisOutcome(dimension_id=a.dimension_id, status="not_run", rationale=failure)
            for a in judge.definition.dimensions
        )
    except Exception as error:  # noqa: BLE001 - judge failures remain in the complete denominator.
        status, failure = "failed", type(error).__name__
        axes = tuple(
            AxisOutcome(dimension_id=a.dimension_id, status="judge_failure", rationale=failure)
            for a in judge.definition.dimensions
        )
    report = DualReport(
        workflow_id=evidence.workflow_id,
        knowledge_sha256=evidence.knowledge_sha256,
        evidence_sha256=sha256_json(evidence.model_dump(mode="json")),
        reference_sha256=sha256_json(references.model_dump(mode="json")),
        judge_config_sha256=sha256_json(judge.model_dump(mode="json")),
        planned_steps=len(evidence.steps),
        judge_status=status,
        judge_failure=failure,
        axes=axes,
        mechanical=mechanical,
    )
    parent = artifact_input(artifacts.read(artifact_id + "-intent").manifest)
    artifacts.write_json(
        artifact_id=artifact_id,
        artifact_type="workflow-dual-evaluation",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=clock(), inputs=(parent,), code_revision=code_revision
        ),
        files={"report.json": report.model_dump(mode="json")},
    )
    return report


def evaluate(
    evidence: WorkflowEvidence,
    references: EvaluationReferences,
    *,
    judge: JudgeConfig,
    complete: Callable[[JsonObject, str], JsonObject],
    artifacts: ArtifactStore,
    artifact_id: str,
    clock: Callable[[], datetime],
    code_revision: str,
) -> DualReport:
    """Evaluate or replay identical saved inputs without purchasing a judge twice.

    A live caller owns a nonblocking project lease. Completed reports and saved responses
    replay exactly. An interrupted intent without a response remains explicitly undispatched
    or ambiguous, never permission to repeat an externally billed request.
    """
    validate_artifact_id(artifact_id)
    lock = artifacts.project_directory.resolve() / ("evaluation-" + artifact_id + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock, timeout=0):
        return _evaluate_locked(
            evidence,
            references,
            judge=judge,
            complete=complete,
            artifacts=artifacts,
            artifact_id=artifact_id,
            clock=clock,
            code_revision=code_revision,
        )


def strict_equal(actual: JsonValue, expected: JsonValue) -> bool:
    """Compare exact JSON types, preserving booleans and numeric representations."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            strict_equal(actual[k], expected[k]) for k in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            strict_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected
