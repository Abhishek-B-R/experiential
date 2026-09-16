"""Policy sampling contracts that preserve original tokenizer and rollout evidence."""

from __future__ import annotations

from typing import Protocol

from exp.common.claas import ExactTokenEvidence
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema
from exp.runtime.claas.registry import ServingRevision


class PolicySample(ContractModel):
    """One exact action with logical Chat envelopes and an untouched server receipt."""

    action: AssistantAction
    exact_tokens: ExactTokenEvidence
    request: JsonObject
    response: JsonObject
    raw_completion: JsonObject


class ExactPolicySampler(Protocol):
    """A sampler bound to one trusted, immutable serving revision."""

    @property
    def revision(self) -> ServingRevision:
        """Return the exact scope, base weights, tokenizer, and policy being sampled."""
        ...

    @property
    def policy_revision(self) -> str:
        """Return the currently bound policy revision."""
        ...

    async def sample(
        self,
        messages: tuple[ModelMessage, ...],
        tools: tuple[ToolSchema, ...],
        request_id: str,
    ) -> PolicySample:
        """Generate one action without reconstructing any completion token evidence."""
        ...


class CompletionDecoder(Protocol):
    """Model-family-specific interpretation of raw text, never of its token evidence."""

    def decode(
        self, text: str, request_id: str, tools: tuple[ToolSchema, ...] = ()
    ) -> AssistantAction:
        """Interpret the sampled text as one assistant action."""
        ...
