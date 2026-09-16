"""All traffic simulation and optional compute calls share one bounded reservation."""

import asyncio
import sqlite3
import sys
from pathlib import Path

import pytest

from exp.common.claas import ClaasScope, Experience
from exp.common.core.locks import FileLockTimeout, file_write_lock
from exp.common.models import ModelCapabilities, ModelSnapshot
from exp.optimize.claas.backends.modal.configuration import ModalExecutionConfig
from exp.optimize.claas.configuration import CycleLimits, LocalClaasConfig
from exp.optimize.claas.execution import ExecutionSettings
from exp.optimize.claas.lifecycle.cycle import base_revision, run_cycle
from exp.optimize.claas.lifecycle.cycle_test import Admission, ReceiptBackend, Serving, config
from exp.optimize.workflows.traffic_learning.composition import TrafficProviders
from exp.optimize.workflows.traffic_learning.composition_test import (
    providers as fixture_providers,
)
from exp.optimize.workflows.traffic_learning.configuration import TrafficWorkflowConfig
from exp.optimize.workflows.traffic_learning.execution import (
    ConfiguredTrafficSource,
    ProviderPlan,
    TrafficProviderSettings,
    load_provider_settings,
    provider_plan,
    save_provider_settings,
)
from exp.optimize.workflows.traffic_learning.sources.preparation_test import traffic
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.claas.harness import ClaasWorldModel
from exp.simulation.claas.harness_test import model_snapshot
from exp.simulation.claas.provider import ClaasBoundedProvider


def configuration() -> LocalClaasConfig:
    """Select a finite learner recipe independent of traffic provider choices."""
    return LocalClaasConfig(
        scope=ClaasScope(user_id="local", application_id="tools"),
        base_model="student",
        base_model_revision="a" * 40,
        tokenizer_id="student",
        tokenizer_revision="a" * 40,
        limits=CycleLimits(
            maximum_scenarios=2, maximum_rollouts_per_scenario=1, maximum_episode_steps=2
        ),
    )


def runtime() -> ExecutionSettings:
    """Name a runtime without selecting or constructing any providers."""
    return ExecutionSettings(
        private_base_url="http://127.0.0.1:8000", worker_python=Path(sys.executable)
    )


def providers() -> TrafficProviderSettings:
    """Reserve conservative call bounds before authentication."""
    return TrafficProviderSettings(
        maximum_world_call_cost_usd=0.01, maximum_judge_call_cost_usd=0.01
    )


def workflow() -> TrafficWorkflowConfig:
    """Explicitly opt the fixture application into traffic-based learning."""
    return TrafficWorkflowConfig(
        scope=configuration().scope, world_model_alias="world", judge_alias="judge"
    )


def test_complete_scheduled_cost_includes_every_world_and_judge_call() -> None:
    """Reserve both policies, every tool turn, synthesis, and all practice rollouts."""
    plan = provider_plan(configuration(), providers(), runtime(), workflow(), evaluation_tasks=3)
    assert plan.synthesis.maximum_model_calls == 2
    assert plan.practice.maximum_model_calls == 4
    assert plan.evaluation.maximum_model_calls == 12
    assert plan.judge.maximum_model_calls == 6
    assert plan.maximum_cost_usd == pytest.approx(0.24)


def test_modal_and_provider_cost_share_one_cycle_ceiling() -> None:
    """Independent compute authorization cannot evade the combined spending bound."""
    modal = ModalExecutionConfig(
        app_name="app",
        volume_name="volume",
        gpu="A10G",
        timeout_seconds=10,
        startup_timeout_seconds=10,
        maximum_container_rate_usd_per_second=1.0,
        authorized_maximum_cost_usd=20.0,
    )
    selected = runtime().model_copy(update={"modal": modal})
    with pytest.raises(ValueError, match="provider and compute"):
        provider_plan(
            configuration().model_copy(update={"compute": "modal"}),
            providers(),
            selected,
            workflow(),
            evaluation_tasks=3,
        )


def test_provider_settings_are_separate_and_required(tmp_path: Path) -> None:
    """A generic runtime never supplies an implicit world-model cost authorization."""
    with pytest.raises(ValueError, match="traffic providers are not configured"):
        load_provider_settings(tmp_path)
    save_provider_settings(tmp_path, providers())
    assert load_provider_settings(tmp_path) == providers()
    assert not (tmp_path / "execution.json").exists()


def test_each_cycle_replans_rotated_evaluation_before_provider_resolution(tmp_path: Path) -> None:
    """Source replacement changes the task count without retaining stale provider reservoirs."""
    _exercise_rotating_cohort(tmp_path, maximum_cost=5.0)


def test_rotated_cohort_over_budget_fails_before_any_new_provider_construction(
    tmp_path: Path,
) -> None:
    """A larger cohort cannot spend against stale consent or fail only after training."""
    _exercise_rotating_cohort(tmp_path, maximum_cost=0.25)


def _exercise_rotating_cohort(directory: Path, *, maximum_cost: float) -> None:
    """Run two actual generic cycles with persisted replacement captures and fixture providers."""

    database = directory / "capture.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
          CREATE TABLE claas_experiences(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            experience_id UNIQUE,user_id,application_id,response_id,expires_at,payload);
          CREATE TABLE claas_feedback(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id,application_id,feedback_id,response_id,episode_id,expires_at,payload);
          CREATE TABLE claas_episodes(sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id,application_id,episode_id,expires_at,payload);
          CREATE TABLE claas_episode_members(user_id,application_id,episode_id,response_id);
        """)

    def replace_captures(sources: tuple[Experience, ...]) -> None:
        """Expire original captures and append exact replacements with monotonic sequence IDs."""
        with sqlite3.connect(database) as connection:
            connection.execute("DELETE FROM claas_experiences")
            connection.executemany(
                "INSERT INTO claas_experiences(experience_id,user_id,application_id,"
                "response_id,expires_at,payload) VALUES(?,?,?,?,unixepoch()+3600,?)",
                (
                    (
                        item.experience_id,
                        item.scope.user_id,
                        item.scope.application_id,
                        item.response_id,
                        item.model_dump_json(),
                    )
                    for item in sources
                ),
            )

    class Catalog(RuntimeModelCatalog):
        """Supply static model snapshots without any credential-backed provider resolution."""

        def __init__(self) -> None:
            """Leave provider construction to the explicitly injected fixture factory."""

        def snapshot(self, alias: str) -> tuple[ModelSnapshot, ModelCapabilities]:
            """Provide the same immutable world/judge identities as fixture receipts."""
            return model_snapshot(), ModelCapabilities()

    settings = config().model_copy(
        update={
            "scope": traffic()[0].scope,
            "limits": config().limits.model_copy(update={"maximum_cost_usd": maximum_cost}),
        }
    )
    flow = TrafficWorkflowConfig(
        scope=settings.scope, world_model_alias="world", judge_alias="judge"
    )
    state_dir = directory / "application"
    admission = Admission()
    serving = Serving(admission, base_revision(settings))
    plans: list[ProviderPlan] = []

    def create_providers(
        config: LocalClaasConfig,
        workflow: TrafficWorkflowConfig,
        plan: ProviderPlan,
        catalog: RuntimeModelCatalog,
    ) -> TrafficProviders:
        """Resolve only after the new cohort's plan is checked, while mutation stays excluded."""
        with (
            pytest.raises(FileLockTimeout),
            file_write_lock(state_dir / "cycle", what="evaluation", timeout_s=0),
        ):
            pytest.fail("cohort changed between budgeting and credential construction")
        plans.append(plan)
        fixture = fixture_providers()
        return TrafficProviders(
            synthesis=ClaasBoundedProvider(
                client=fixture.synthesis.client,
                model=fixture.synthesis.model,
                limits=plan.synthesis,
                source_disclosure=fixture.synthesis.source_disclosure,
            ),
            practice=ClaasWorldModel(
                client=fixture.practice.client,
                model=fixture.practice.model,
                limits=plan.practice,
                source_disclosure=fixture.practice.source_disclosure,
            ),
            evaluation=ClaasWorldModel(
                client=fixture.evaluation.client,
                model=fixture.evaluation.model,
                limits=plan.evaluation,
                source_disclosure=fixture.evaluation.source_disclosure,
                purpose="evaluation",
            ),
            judge=ClaasBoundedProvider(
                client=fixture.judge.client,
                model=fixture.judge.model,
                limits=plan.judge,
                source_disclosure=fixture.judge.source_disclosure,
            ),
        )

    source = ConfiguredTrafficSource(
        config=settings,
        workflow=flow,
        runtime=runtime(),
        settings=providers(),
        catalog=Catalog(),
        database_path=database,
        provider_factory=create_providers,
    )

    async def cycle() -> None:
        """Use the production generic cycle and its actual application locking."""
        state = await run_cycle(
            directory=state_dir,
            config=settings,
            plan=source,
            serving=serving,
            admission=admission,
            backend_factory=lambda lineage: ReceiptBackend(
                state_dir / "checkpoints", lineage, serving
            ),
        )
        assert state.stage == "complete" and state.decision
        assert state.decision.paired_mean_delta is not None

    replace_captures(traffic()[:16])
    asyncio.run(cycle())
    assert plans[0].judge.maximum_model_calls == 4
    replace_captures(
        tuple(
            item.model_copy(update={"experience_id": "replacement-" + item.experience_id})
            for item in traffic()
        )
    )
    if maximum_cost < 1:
        with pytest.raises(ValueError, match="provider and compute reservations"):
            asyncio.run(cycle())
        assert len(plans) == 1
    else:
        asyncio.run(cycle())
        assert plans[1].judge.maximum_model_calls == 14
        assert plans[1].evaluation.maximum_model_calls == 28
