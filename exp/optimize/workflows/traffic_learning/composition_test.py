"""The optional traffic pipeline composes into the real generic learning transaction."""

import asyncio
import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest
from exp.optimize.claas.lifecycle.cycle import base_revision
from exp.optimize.claas.lifecycle.cycle_test import Admission, ReceiptBackend, Serving, config
from exp.optimize.workflows.traffic_learning.composition import TrafficProviders, run_traffic_cycle
from exp.optimize.workflows.traffic_learning.configuration import TrafficWorkflowConfig
from exp.optimize.workflows.traffic_learning.evaluation_test import world_reply
from exp.optimize.workflows.traffic_learning.sources.holdouts import begin_evaluation_holdout
from exp.optimize.workflows.traffic_learning.sources.preparation_test import traffic
from exp.optimize.workflows.traffic_learning.sources.retention import prune_evidence
from exp.simulation.claas.harness import ClaasWorldModel, SourceDisclosure
from exp.simulation.claas.harness_test import RecordingClient, limits, model_snapshot
from exp.simulation.claas.provider import ClaasBoundedProvider


def providers() -> TrafficProviders:
    """Build provider fixtures with real bounds and deterministic synthesis/judgment receipts."""
    disclosure = SourceDisclosure(scope=traffic()[0].scope, model=model_snapshot())
    bounds = limits(maximum_steps=2, maximum_model_calls=256, maximum_total_cost_usd=1)

    def synthesis(request: ModelRequest) -> JsonObject:
        """Supply one alternate task using the declared tool contract."""
        return {"user_message": "Look up and report another claim."}

    def judge(request: ModelRequest) -> JsonObject:
        """Create a fixture score difference, without claiming learned model quality."""
        payload = json.loads(request.messages[-1].content or "{}")
        answer = payload["visible_trajectory"][-1]["assistant_action"]["content"]
        return {"score": 1.0 if answer == "4" else 0.0, "feedback": "fixture score"}

    return TrafficProviders(
        synthesis=ClaasBoundedProvider(
            client=RecordingClient(synthesis),
            model=model_snapshot(),
            limits=bounds,
            source_disclosure=disclosure,
        ),
        practice=ClaasWorldModel(
            client=RecordingClient(world_reply),
            model=model_snapshot(),
            limits=bounds,
            source_disclosure=disclosure,
        ),
        evaluation=ClaasWorldModel(
            client=RecordingClient(world_reply),
            model=model_snapshot(),
            limits=bounds,
            source_disclosure=disclosure,
            purpose="evaluation",
        ),
        judge=ClaasBoundedProvider(
            client=RecordingClient(judge),
            model=model_snapshot(),
            limits=bounds,
            source_disclosure=disclosure,
        ),
    )


@pytest.mark.parametrize("fail_backend", [False, True])
def test_traffic_workflow_keeps_source_retention_and_generic_cycle_evidence(
    tmp_path: Path, fail_backend: bool
) -> None:
    """Success and failure retain the same pre-dispatch source manifest for expiration."""
    settings = config().model_copy(update={"scope": traffic()[0].scope})
    admission = Admission()
    serving = Serving(admission, base_revision(settings))

    def backend(lineage: str) -> ReceiptBackend:
        """Inject an explicit receipt-only adapter into the production core."""
        result = ReceiptBackend(tmp_path / "checkpoints", lineage, serving)
        result.fail = fail_backend
        return result

    operation = run_traffic_cycle(
        directory=tmp_path,
        config=settings,
        workflow=TrafficWorkflowConfig(
            scope=settings.scope, world_model_alias="world", judge_alias="judge"
        ),
        experiences=traffic(),
        providers=providers(),
        serving=serving,
        admission=admission,
        backend_factory=backend,
    )
    if fail_backend:
        with pytest.raises(RuntimeError, match="deliberate backend failure"):
            asyncio.run(operation)
    else:
        state = asyncio.run(operation)
        assert state.decision and state.decision.approved
        assert state.decision.score_kind == "synthetic_judge"
    run = next((tmp_path / "cycles").iterdir())
    context = json.loads((run / "context.json").read_bytes())
    assert context["workflow"] == "traffic-learning-v1" and context["synthesis"]
    assert context["split"]["fit"] and context["split"]["held_out"]
    assert prune_evidence(tmp_path, retained_source_ids=set()).deleted_cycle_ids == (run.name,)
    assert not run.exists()


def test_pending_heldout_request_blocks_synthesis_and_training(tmp_path: Path) -> None:
    """Ambiguous evaluation captures fail before any world-model request or compute acquisition."""
    settings = config().model_copy(update={"scope": traffic()[0].scope})
    admission = Admission()
    serving = Serving(admission, base_revision(settings))
    source = providers()
    begin_evaluation_holdout(tmp_path, run_id="evaluation", request_id="unacknowledged")
    with pytest.raises(ValueError, match="pending"):
        asyncio.run(
            run_traffic_cycle(
                directory=tmp_path,
                config=settings,
                workflow=TrafficWorkflowConfig(
                    scope=settings.scope, world_model_alias="world", judge_alias="judge"
                ),
                experiences=traffic(),
                providers=source,
                serving=serving,
                admission=admission,
                backend_factory=lambda lineage: ReceiptBackend(
                    tmp_path / "checkpoints", lineage, serving
                ),
            )
        )
    assert source.synthesis.reserved_calls == 0
    assert not serving.sampled
