"""Knowledge pinned in a simulator never exposes the evaluated task's own evidence."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_bytes
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.fidelity import FidelitySpec, prepare_fidelity_evaluation
from exp.optimize.evaluation.tests.fidelity_memory_test import memory_project
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibraryEntry
from exp.simulation.reconstruction.revision import ResearchNote
from exp.simulation.world_model.artifact import persist_grounded_world_model
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

SECRET = "HELDOUT_REPLY_SECRET: reply-0"
OWN = "held-out-trace"


def bound(root: Path, knowledge: SimulationKnowledge | None = None, case: str = "") -> FidelitySpec:
    """A held-out task (sourced from ``OWN``) on the fixture simulator with this knowledge."""
    project, base = memory_project(root)
    artifact = load_grounded_world_model_artifact(project.artifacts, base.world_model)
    if case == "library":
        knowledge = SimulationKnowledge(
            source_trace_ids=("observed",), transfer_context=_selected(project)
        )
    model = persist_grounded_world_model(
        project.artifacts,
        artifact.serving_rag,
        model_alias=artifact.model_alias,
        model=artifact.model,
        created_at=artifact.created_at,
        code_revision="test",
        top_k=artifact.top_k,
        knowledge=knowledge,
    )
    data = base.model_dump(mode="json")
    data["world_model"] = artifact_input(model.manifest).model_dump(mode="json")
    data["task"]["source_trace_ids"] = [OWN]
    data["task"]["origin"] = "mined"
    data["output_protocol"] = "json-tool"
    spec = FidelitySpec.model_validate(data)
    assert spec.task.partition == "held_out"
    return spec


def _selected(project: ProjectStore) -> JsonObject:
    """A prior-library selection whose entry derives from the held-out trace."""
    payload = {
        "trajectory": {
            "task": "Inspect current entries",
            "task_lineage": "other",
            "source_trace_ids": [OWN],
            "actions": [{"name": "inspect", "arguments": {"key": "current"}}],
            "observed_replies": [SECRET],
        }
    }
    manifest = project.artifacts.write_json(
        artifact_id="held-out-source",
        artifact_type="consumed-trajectory",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"trajectory.json": payload},
    )
    ref = artifact_input(manifest)
    stored = project.artifacts.read_bytes(ref.artifact_id, "trajectory.json")
    entry = LibraryEntry(
        entry_id="held-out-source",
        kind="trajectory",
        artifact=ref,
        path="trajectory.json",
        payload_sha256=sha256_bytes(stored),
        disposition="source",
        derived_from=ref,
    )
    return {"selected": [{"entry": entry.model_dump(mode="json"), "payload": payload}]}


NOTE = ResearchNote(
    id="leak", content=SECRET, investigation_sha256="a" * 64, source_trace_ids=(OWN,)
)


@pytest.mark.parametrize(
    ("knowledge", "case", "message"),
    [
        (SimulationKnowledge(source_trace_ids=(OWN,), rules=SECRET), "", "own recorded"),
        (SimulationKnowledge(rules=SECRET), "", "no recorded provenance"),
        (
            SimulationKnowledge(source_trace_ids=("observed",), research_history=(NOTE,)),
            "",
            "own recorded",
        ),
        (None, "library", "own recorded"),
        (
            SimulationKnowledge(
                source_trace_ids=("observed",),
                research_history=(NOTE.model_copy(update={"source_trace_ids": ()}),),
            ),
            "",
            "research note leak",
        ),
    ],
    ids=["constructor", "unknown", "research-note", "library", "unprovenanced-note"],
)
def test_knowledge_exposing_the_task_is_refused_before_any_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    knowledge: SimulationKnowledge | None,
    case: str,
    message: str,
) -> None:
    """Every predictor-visible knowledge channel is checked; refusal precedes credentials."""
    monkeypatch.delenv("FIDELITY_TEST_KEY", raising=False)
    spec = bound(tmp_path, knowledge, case)
    project = ProjectStore(tmp_path, "fidelity-test")
    with pytest.raises(ValueError, match=message):
        prepare_fidelity_evaluation(project, spec, code_revision="test")


@pytest.mark.parametrize(
    "knowledge",
    [
        SimulationKnowledge(rules="Orders ship in two days.", authored=True),
        SimulationKnowledge(source_trace_ids=("observed",), rules="Orders ship in two days."),
    ],
    ids=["authored", "other-traces"],
)
def test_authored_or_independent_knowledge_is_accepted(
    tmp_path: Path, knowledge: SimulationKnowledge
) -> None:
    """Hand-written rules and knowledge from other traces remain valid held-out simulators."""
    spec = bound(tmp_path, knowledge)
    project = ProjectStore(tmp_path, "fidelity-test")
    assert prepare_fidelity_evaluation(project, spec, code_revision="test").spec == spec
