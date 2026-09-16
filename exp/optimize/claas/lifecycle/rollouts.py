"""Collect exact student rollouts from arbitrary explicitly supplied environments."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from exp.common.claas import Experience, ExperienceProvenance
from exp.common.claas.scenarios import EnvironmentEpisode, EnvironmentStep, Scenario
from exp.common.core.artifacts import ContractModel
from exp.common.core.files import write_text_atomic
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema
from exp.optimize.claas.configuration import CycleLimits
from exp.optimize.claas.training_contracts import (
    ClaasTrainingSpec,
    TrainingBatch,
    TrainingExample,
    teacher_feedback_text,
)
from exp.runtime.claas.registry import ServingRevision
from exp.runtime.claas.serving.contracts import PolicySample
from exp.runtime.environments.learning import Environment


class ServingController(Protocol):
    """A paused, revision-bound policy sampler with explicit GPU lifecycle ownership."""

    async def pause_and_drain(self) -> None:
        """Stop new requests and await admitted operations."""
        ...

    async def wake(self) -> None:
        """Restore inference memory without admitting requests."""
        ...

    async def sleep(self) -> None:
        """Release inference GPU memory while keeping admission paused."""
        ...

    async def load_revision(self, revision: ServingRevision) -> None:
        """Load and verify the named revision while paused."""
        ...

    async def resume(self) -> None:
        """Resume public requests only after registry publication."""
        ...

    async def sample_for_evaluation(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> PolicySample:
        """Sample exact token evidence while public traffic remains paused."""
        ...

    async def tokenize_training_text(self, text: str) -> tuple[int, ...]:
        """Tokenize only new teacher context with the loaded policy's pinned tokenizer."""
        ...


class RevisionPolicy:
    """Load the exact evaluation revision before each action on one shared GPU."""

    def __init__(self, serving: ServingController, revision: ServingRevision) -> None:
        """Bind immutable policy identity without loading or sampling."""
        self.serving = serving
        self.revision = revision

    @property
    def policy_revision(self) -> str:
        """Return the immutable evaluation identity."""
        return self.revision.policy_revision

    async def act(
        self, *, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> AssistantAction:
        """Load the bound revision, then sample one action on the private endpoint."""
        await self.serving.load_revision(self.revision)
        sample = await self.serving.sample_for_evaluation(messages, tools, request_id)
        _verify_sample(sample, self.revision)
        return sample.action


class PracticeReceipt(ContractModel):
    """Exact student receipts and separate environment evidence for one environment episode."""

    episode: EnvironmentEpisode
    samples: tuple[PolicySample, ...]
    experiences: tuple[Experience, ...]


async def run_owned[T](operation: Callable[[], T]) -> T:
    """Own a blocking provider call through cancellation and its configured deadline."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def collect_practice(
    *,
    scenarios: Sequence[Scenario],
    environment: Environment,
    serving: ServingController,
    revision: ServingRevision,
    spec: ClaasTrainingSpec,
    limits: CycleLimits,
    directory: Path,
    cycle_id: str,
    operation_timeout_seconds: float = 120,
) -> TrainingBatch:
    """Persist environment episodes and build one bounded optimizer batch from new actions.

    Feedback is attached only to the action the environment just evaluated.
    Missing scalar feedback stays absent. Exact samples are never re-tokenized,
    truncated, or sourced from historical provider traffic.
    """
    if not 0 < operation_timeout_seconds <= 3600:
        raise ValueError("practice operation deadline must be between zero and 3600 seconds")
    examples: list[TrainingExample] = []
    tokens = 0
    for index, scenario in enumerate(scenarios[: limits.maximum_scenarios]):
        for rollout in range(limits.maximum_rollouts_per_scenario):
            identity = f"{cycle_id}-{index}-{rollout}"
            if (
                scenario.scope != revision.scope
                or scenario.environment_id != environment.environment_id
            ):
                raise ValueError("practice scenario scope or environment identity differs")
            session = await asyncio.wait_for(
                environment.open(scenario), timeout=operation_timeout_seconds
            )
            messages = scenario.messages
            steps: list[EnvironmentStep] = []
            reason: Literal["terminal", "step_limit", "failed"] = "failed"
            samples: list[PolicySample] = []
            experiences: list[Experience] = []
            try:
                for step in range(limits.maximum_episode_steps):
                    request_id = f"{identity}-{step}"
                    sample = await asyncio.wait_for(
                        serving.sample_for_evaluation(messages, scenario.tools, request_id),
                        timeout=operation_timeout_seconds,
                    )
                    _verify_sample(sample, revision)
                    samples.append(sample)
                    experience = Experience(
                        experience_id=request_id,
                        response_id=request_id,
                        episode_id=identity,
                        scope=revision.scope,
                        protocol="chat_completions",
                        captured_at=datetime.now(UTC),
                        request=sample.request,
                        response=sample.response,
                        provenance=ExperienceProvenance(
                            source_kind=scenario.source_kind,
                            source_id=scenario.scenario_id,
                            model_id=revision.model_id,
                            model_revision=revision.model_revision,
                            policy_revision=revision.policy_revision,
                            source_experience_ids=scenario.source_experience_ids,
                        ),
                        exact_tokens=sample.exact_tokens,
                    )
                    experiences.append(experience)
                    transition = await asyncio.wait_for(
                        session.step(sample.action), timeout=operation_timeout_seconds
                    )
                    steps.append(EnvironmentStep(action=sample.action, transition=transition))
                    messages = transition.messages
                    length = len(sample.exact_tokens.prompt_token_ids) + len(
                        sample.exact_tokens.response_token_ids
                    )
                    eligible = (
                        spec.objective == "reinforce" or transition.feedback is not None
                    ) and (spec.objective == "sdpo" or transition.reward is not None)
                    teacher_length = 0
                    if eligible and spec.objective != "reinforce":
                        feedback_tokens = await asyncio.wait_for(
                            serving.tokenize_training_text(
                                teacher_feedback_text(transition.feedback or "")
                            ),
                            timeout=operation_timeout_seconds,
                        )
                        teacher_length = length + len(feedback_tokens)
                    batch_length = length + teacher_length
                    if (
                        eligible
                        and len(examples) < spec.max_batch_examples
                        and length <= spec.max_sequence_tokens
                        and teacher_length <= spec.max_sequence_tokens
                        and tokens + batch_length <= spec.max_batch_tokens
                    ):
                        examples.append(
                            TrainingExample(
                                experience=experience,
                                scalar_reward=transition.reward,
                                text_feedback=transition.feedback,
                            )
                        )
                        tokens += batch_length
                    if transition.terminal:
                        reason = "terminal"
                        break
                else:
                    reason = "step_limit"
            finally:
                evidence = await asyncio.wait_for(
                    session.close(reason), timeout=operation_timeout_seconds
                )
                receipt = PracticeReceipt(
                    episode=EnvironmentEpisode(
                        scenario=scenario, steps=tuple(steps), end_reason=reason, evidence=evidence
                    ),
                    samples=tuple(samples),
                    experiences=tuple(experiences),
                )
                write_text_atomic(directory / f"{identity}.json", receipt.model_dump_json() + "\n")
            if len(examples) >= spec.max_batch_examples or tokens >= spec.max_batch_tokens:
                break
        if len(examples) >= spec.max_batch_examples or tokens >= spec.max_batch_tokens:
            break
    if not examples:
        raise ValueError(
            "practice produced no token-exact examples with the selected objective's feedback; "
            "inspect practice receipts or choose the matching objective"
        )
    return TrainingBatch(
        batch_id=cycle_id,
        expected_policy_revision=revision.policy_revision,
        examples=tuple(examples),
    )


def _verify_sample(sample: PolicySample, revision: ServingRevision) -> None:
    """Reject a sampler that returned another model, tokenizer, or policy revision."""
    token = sample.exact_tokens
    if (
        token.model_id != revision.model_id
        or token.model_revision != revision.model_revision
        or token.tokenizer_id != revision.tokenizer_id
        or token.tokenizer_revision != revision.tokenizer_revision
        or token.policy_revision != revision.policy_revision
    ):
        raise ValueError("sample identity differs from the loaded policy revision")
