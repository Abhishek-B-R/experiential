"""Dual evaluation replays saved judgments and preserves unavailable measurements."""

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, JsonValue, sha256_json
from exp.common.evaluations.reconstruction import (
    EvaluationReferences,
    ReferenceStep,
    WorkflowEvidence,
    WorkflowStep,
    default_judge,
    evaluate,
    judge_wire_schema,
    mechanical_checks,
)
from exp.common.project import ArtifactManifest, ProjectConfig, ProjectStore

NOW = datetime(2030, 1, 1, tzinfo=UTC)


def evidence() -> WorkflowEvidence:
    """Provide one complete synthetic read, independent of a provider response."""
    return WorkflowEvidence(
        workflow_id="read",
        task="Read items",
        initial_state={"items": []},
        steps=(
            WorkflowStep(
                step=0,
                action={"name": "read"},
                status="valid",
                reply=[],
                before={"items": []},
                after={"items": []},
            ),
        ),
    )


def references() -> EvaluationReferences:
    """Provide a separate saved reference without claiming benchmark qualification."""
    return EvaluationReferences(
        capability="recorded_reference_only",
        steps=(
            ReferenceStep(
                step=0,
                action={"name": "read"},
                before={"items": []},
                reply_available=True,
                reply=[],
                state_available=True,
                after={"items": []},
            ),
        ),
    )


def test_saved_judge_response_recovers_after_publication_crash_without_repurchase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash after paid output retention, then publish and replay the same result once."""
    project = ProjectStore(tmp_path, "evaluation")
    project.initialize(ProjectConfig(project_id="evaluation"))
    judge = default_judge("scripted")
    calls: list[str] = []

    def complete(_: JsonObject, identity: str) -> JsonObject:
        """Return only protocol-valid ratings; this test establishes persistence, not fidelity."""
        calls.append(identity)
        return {
            "axes": [
                {
                    "dimension_id": axis.dimension_id,
                    "status": "unassessable",
                    "score": None,
                    "rationale": "Synthetic persistence fixture, no human calibration.",
                    "evidence_refs": [],
                    "critical_flags": [],
                }
                for axis in judge.definition.dimensions
            ]
        }

    original = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, JsonValue],
    ) -> ArtifactManifest:
        """Fail only final publication, after the response artifact has been saved."""
        if artifact_type == "workflow-dual-evaluation":
            raise RuntimeError("publication interrupted")
        return original(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            envelope=envelope,
            files=files,
        )

    store = project.artifacts
    monkeypatch.setattr(store, "write_json", crash)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        evaluate(
            evidence(),
            references(),
            judge=judge,
            complete=complete,
            artifacts=store,
            artifact_id="evaluation",
            clock=lambda: NOW,
            code_revision="fixture",
        )
    monkeypatch.setattr(store, "write_json", original)
    result = evaluate(
        evidence(),
        references(),
        judge=judge,
        complete=complete,
        artifacts=store,
        artifact_id="evaluation",
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert (
        evaluate(
            evidence(),
            references(),
            judge=judge,
            complete=complete,
            artifacts=store,
            artifact_id="evaluation",
            clock=lambda: NOW,
            code_revision="fixture",
        )
        == result
    )
    assert calls == ["evaluation"]
    assert result.mechanical["workflow_joint_exact"] is True
    assert result.judge_status == "completed"
    changed = evidence().model_copy(update={"task": "Changed task"})
    with pytest.raises(ValueError, match="identity differs"):
        evaluate(
            changed,
            references(),
            judge=judge,
            complete=complete,
            artifacts=store,
            artifact_id="evaluation",
            clock=lambda: NOW,
            code_revision="fixture",
        )
    assert calls == ["evaluation"]


def test_unexecuted_continuity_is_unavailable() -> None:
    """Zero observations cannot establish state continuity."""
    blocked = evidence().model_copy(
        update={
            "steps": (WorkflowStep(step=0, action={"name": "read"}, status="blocked"),),
        }
    )
    measured = mechanical_checks(blocked, references())
    assert measured["carried_continuity"] is None
    assert measured["continuity_checked_steps"] == 0


def test_interrupted_judge_dispatch_stays_unknown_without_a_second_purchase(tmp_path: Path) -> None:
    """A crash after intent cannot be called undispatched or resumed with a fresh request."""
    project = ProjectStore(tmp_path, "interrupted")
    project.initialize(ProjectConfig(project_id="interrupted"))
    calls: list[str] = []

    def interrupted(_: JsonObject, identity: str) -> JsonObject:
        """Simulate process interruption after dispatch may have reached a provider."""
        calls.append(identity)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate(
            evidence(),
            references(),
            judge=default_judge("scripted"),
            complete=interrupted,
            artifacts=project.artifacts,
            artifact_id="interrupted",
            clock=lambda: NOW,
            code_revision="fixture",
        )
    report = evaluate(
        evidence(),
        references(),
        judge=default_judge("scripted"),
        complete=interrupted,
        artifacts=project.artifacts,
        artifact_id="interrupted",
        clock=lambda: NOW,
        code_revision="fixture",
    )
    assert calls == ["interrupted"]
    assert report.judge_status == "unknown"
    assert all(axis.status == "judge_unknown" for axis in report.axes)


def test_keyed_judge_protocol_carries_mechanical_findings_and_fixed_inventory(
    tmp_path: Path,
) -> None:
    """A new wire profile owns criterion IDs and exposes actual saved discrepancies."""
    import json

    project = ProjectStore(tmp_path, "keyed")
    project.initialize(ProjectConfig(project_id="keyed"))
    judge = default_judge("scripted", protocol="keyed-v2", mechanical_feedback=True)
    bad = evidence().model_copy(
        update={"steps": (evidence().steps[0].model_copy(update={"reply": ["invented"]}),)}
    )

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Use a fixed keyed response, independent of ordering and repeated ID copying."""
        messages = body["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        payload = json.loads(str(messages[1]["content"]))
        assert payload["workflow"]["mechanical_findings"]["first_semantic_divergence"] == 0
        return {
            "axes": {
                axis.dimension_id: {
                    "status": "assessed",
                    "score": 2,
                    "rationale": "The actual reply differs from the recorded reference.",
                    "evidence_refs": ["step-0", "reference-step-0"],
                    "critical_flags": ["wrong_reply"],
                }
                for axis in judge.definition.dimensions
            }
        }

    report = evaluate(
        bad,
        references(),
        judge=judge,
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="keyed",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "completed"
    assert report.mechanical["workflow_joint_exact"] is False
    assert tuple(axis.dimension_id for axis in report.axes) == tuple(
        axis.dimension_id for axis in judge.definition.dimensions
    )


def test_keyed_judge_cannot_drop_an_axis(tmp_path: Path) -> None:
    """A keyed envelope does not excuse missing criteria or synthesize successful ratings."""
    project = ProjectStore(tmp_path, "missing-axis")
    project.initialize(ProjectConfig(project_id="missing-axis"))
    report = evaluate(
        evidence(),
        references(),
        judge=default_judge("scripted", protocol="keyed-v2"),
        complete=lambda *_: {"axes": {}},
        artifacts=project.artifacts,
        artifact_id="missing-axis",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "failed"
    assert len(report.axes) == 5
    assert all(axis.status == "judge_failure" for axis in report.axes)


def test_capacity_refusal_is_not_a_dispatched_judge_failure(tmp_path: Path) -> None:
    """Direct and actor evaluation retain the same explicit pre-dispatch status."""
    from exp.common.models.client import RequestCapacityExceeded

    project = ProjectStore(tmp_path, "capacity")
    project.initialize(ProjectConfig(project_id="capacity"))

    def refused(body: JsonObject, identity: str) -> JsonObject:
        """Refuse before calling a provider; no hidden capacity retry."""
        raise RequestCapacityExceeded("declared context capacity exceeded before dispatch")

    report = evaluate(
        evidence(),
        references(),
        judge=default_judge("scripted"),
        complete=refused,
        artifacts=project.artifacts,
        artifact_id="capacity",
        clock=lambda: NOW,
        code_revision="test",
    )
    assert report.judge_status == "not_run"
    assert len(report.axes) == 5 and all(axis.status == "not_run" for axis in report.axes)


@pytest.mark.parametrize("protocol", ["ordered-v1", "keyed-v2"])
def test_judge_wire_format_does_not_enable_mechanical_feedback(
    tmp_path: Path,
    protocol: str,
) -> None:
    """An envelope change preserves input authority and blinded provenance."""
    import json

    from exp.common.evaluations.reconstruction import IndependentCheck, JudgeConfig

    project = ProjectStore(tmp_path, "blinding")
    project.initialize(ProjectConfig(project_id="blinding"))
    config = default_judge("scripted").model_dump(mode="json")
    config["protocol"] = protocol
    judge = JudgeConfig.model_validate(config)
    refs = references().model_copy(
        update={
            "independent_checks": (
                IndependentCheck(name="goal", passed=False, reference_id="secret-provider-route"),
            )
        }
    )
    requests: list[JsonObject] = []

    def complete(body: JsonObject, identity: str) -> JsonObject:
        """Retain input even though this synthetic reviewer declines assessment."""
        requests.append(body)
        return {"axes": {}}

    evaluate(
        evidence(),
        refs,
        judge=judge,
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="no-feedback",
        clock=lambda: NOW,
        code_revision="test",
    )
    first = json.dumps(requests[0])
    assert "mechanical_findings" not in first
    assert "secret-provider-route" not in first
    evaluate(
        evidence(),
        refs,
        judge=judge.model_copy(update={"mechanical_feedback": True}),
        complete=complete,
        artifacts=project.artifacts,
        artifact_id="feedback",
        clock=lambda: NOW,
        code_revision="test",
    )
    second = json.dumps(requests[1])
    assert "mechanical_findings" in second and "secret-provider-route" not in second


def test_reply_only_reference_does_not_invent_hidden_state() -> None:
    """Recorded observations support carried reply checks without a full native prestate."""
    evidence = WorkflowEvidence(
        workflow_id="reply-only",
        task="Read a record",
        initial_state={},
        steps=(WorkflowStep(step=0, action={"name": "read"}, status="valid", reply={"id": "a"}),),
    )
    references = EvaluationReferences(
        capability="recorded_reference_only",
        description="Archived reply only",
        steps=(
            ReferenceStep(step=0, action={"name": "read"}, reply_available=True, reply={"id": "a"}),
        ),
    )
    report = mechanical_checks(evidence, references)
    rows = report["steps"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    row = rows[0]
    assert row["trajectory_reply_exact"] is True
    assert row["reply_exact"] is None and row["state_exact"] is None
    assert report["workflow_joint_exact"] is None


def test_strict_keyed_judge_requires_the_score_its_parser_needs() -> None:
    """Schema and parser agree that assessed criteria cannot omit numeric ratings."""
    config = default_judge("glm", protocol="keyed-v3")
    schema = judge_wire_schema(config)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    axis = properties["axes"]
    assert isinstance(axis, dict)
    axes = axis["properties"]
    assert isinstance(axes, dict)
    for value in axes.values():
        assert isinstance(value, dict) and isinstance(value["required"], list)
        assert "score" in value["required"]
    assert sha256_json(config) != sha256_json(default_judge("glm", protocol="keyed-v2"))
