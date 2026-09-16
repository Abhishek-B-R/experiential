"""Finite, explicitly disclosed hosted calls for CLaaS synthesis and evaluation."""

from __future__ import annotations

from threading import Lock

from exp.common.claas import ClaasScope
from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import (
    BoundModelClient,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
)
from exp.common.models.model import ModelFinishReason
from exp.simulation.claas.harness import SourceDisclosure, WorldModelLimitError, WorldModelLimits


class ClaasBoundedProvider:
    """Share finite reservations across synthesis or judgment calls and retries."""

    def __init__(
        self,
        *,
        client: ModelClient,
        model: ModelSnapshot,
        limits: WorldModelLimits,
        source_disclosure: SourceDisclosure | None = None,
    ) -> None:
        """Bind a provider snapshot without making a call or loading credentials."""
        if not isinstance(client, BoundModelClient) or client.model_snapshot != model:
            raise ValueError("provider client must be bound to the configured recipient")
        self.client, self.model, self.limits = client, model, limits
        self.source_disclosure = source_disclosure
        self._calls = 0
        self._poisoned = False
        self._lock = Lock()

    @property
    def reserved_calls(self) -> int:
        """Return all dispatched attempts, including failed and unreported calls."""
        with self._lock:
            return self._calls

    def complete(self, scope: ClaasScope, request: ModelRequest) -> ModelResponse:
        """Reserve a worst-case call before disclosing source-derived content."""
        self.authorize_source(scope)
        if len(canonical_json_bytes(request)) > self.limits.maximum_request_bytes:
            raise WorldModelLimitError("provider request byte limit exceeded")
        if (
            request.maximum_output_tokens is None
            or request.maximum_output_tokens > self.limits.maximum_output_tokens
        ):
            raise WorldModelLimitError("provider output token limit exceeded")
        with self._lock:
            next_calls = self._calls + 1
            if (
                self._poisoned
                or next_calls > self.limits.maximum_model_calls
                or next_calls * self.limits.maximum_call_cost_usd
                > self.limits.maximum_total_cost_usd
            ):
                raise WorldModelLimitError("provider call or cost budget exhausted")
            self._calls = next_calls
        response = self.client.complete(request)
        if response.model != self.model:
            raise ValueError("provider response model differs from the frozen snapshot")
        cost = response.economics.cost_usd
        if cost is not None and cost.value > self.limits.maximum_call_cost_usd:
            with self._lock:
                self._poisoned = True
            raise WorldModelLimitError("provider cost exceeded its conservative reservation")
        if len(canonical_json_bytes(response)) > self.limits.maximum_materialized_response_bytes:
            raise WorldModelLimitError("materialized provider response byte limit exceeded")
        if response.finish_reason == ModelFinishReason.LENGTH:
            raise ValueError("provider returned a truncated structured response")
        if response.output.tool_calls or response.output.content is None:
            raise ValueError("provider must return structured text without tool execution")
        return response

    def authorize_source(self, scope: ClaasScope) -> None:
        """Verify disclosure against the actual client recipient before any dispatch."""
        if (
            self.source_disclosure != SourceDisclosure(scope=scope, model=self.model)
            or self.client.model_snapshot != self.model
        ):
            raise ValueError(
                "source disclosure must authorize this scope and exact provider recipient"
            )
