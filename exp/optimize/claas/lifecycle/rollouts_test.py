"""Exact authored-environment rollouts preserve provenance and teacher/student budgets."""

import asyncio
from pathlib import Path

import pytest

from exp.optimize.claas.lifecycle.cycle import base_revision, training_spec
from exp.optimize.claas.lifecycle.cycle_test import (
    Admission,
    LookupEnvironment,
    Serving,
    config,
    scenario,
)
from exp.optimize.claas.lifecycle.rollouts import collect_practice
from exp.optimize.claas.training_contracts import validate_training_batch


@pytest.mark.parametrize("objective", ["sdpo", "hybrid", "reinforce"])
def test_arbitrary_environment_rollouts_preserve_original_tokens_and_budget(
    tmp_path: Path, objective: str
) -> None:
    """No traffic, world model, or source references are needed to produce a valid batch."""
    settings = config()
    admission = Admission()
    admission.paused = True
    serving = Serving(admission, base_revision(settings))
    environment = LookupEnvironment()
    per_example = 4 if objective == "reinforce" else 10
    spec = training_spec(settings).model_copy(
        update={"objective": objective, "max_batch_tokens": 2 * per_example + 1}
    )
    batch = asyncio.run(
        collect_practice(
            scenarios=(scenario("authored-1"), scenario("authored-2")),
            environment=environment,
            serving=serving,
            revision=base_revision(settings),
            spec=spec,
            limits=settings.limits,
            directory=tmp_path,
            cycle_id="authored-practice",
        )
    )
    validate_training_batch(spec, batch, None)
    assert len(batch.examples) == 2
    for example in batch.examples:
        assert example.experience.exact_tokens is not None
        assert example.experience.exact_tokens.prompt_token_ids == (1, 2)
        assert example.experience.exact_tokens.response_token_ids == (3, 4)
        assert example.experience.provenance.source_kind == "environment"
    assert environment.closed == environment.opened
