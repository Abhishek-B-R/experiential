"""Admission, drain, sleep, and adapter reload around a private vLLM server.

Registry activation and evaluation decisions belong to orchestration. The caller
must durably commit its selected registry pointer before calling ``resume``.
"""

from __future__ import annotations

import asyncio
import math

import httpx
from pydantic import BaseModel, ConfigDict, Field

from exp.common.models import ModelMessage
from exp.common.tasks import ToolSchema
from exp.runtime.claas.registry import ServingRevision
from exp.runtime.claas.serving.contracts import CompletionDecoder, PolicySample
from exp.runtime.claas.serving.vllm import VllmPolicySampler, serving_model_name


class ServingPausedError(RuntimeError):
    """This application is draining, sleeping, or waiting for a verified reload."""


class _SleepState(BaseModel):
    """The server's explicit memory-state receipt."""

    is_sleeping: bool = Field(strict=True)


class _Model(BaseModel):
    """A listed base or loaded adapter name."""

    model_config = ConfigDict(extra="ignore")
    id: str


class _Models(BaseModel):
    """The current publicly addressable routes on this private server."""

    model_config = ConfigDict(extra="ignore")
    data: tuple[_Model, ...]


class VllmServingLifecycle:
    """Serialize control changes while draining all admitted and evaluation requests.

    This controller owns admission only for requests routed through this instance.
    Its private vLLM endpoint must not be exposed as a second public traffic path.
    Sleep frees vLLM memory, not the Modal container or its billing allocation.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        base: ServingRevision,
        decoder: CompletionDecoder,
        max_tokens: int = 2048,
        drain_timeout_seconds: float = 120,
    ) -> None:
        """Start paused until explicit wake and revision loading establish readiness."""
        if base.adapter_directory is not None:
            raise ValueError("base must identify the frozen model without an adapter")
        if not math.isfinite(drain_timeout_seconds) or not 0 < drain_timeout_seconds <= 3600:
            raise ValueError("drain timeout must be finite, positive, and at most one hour")
        self._client = client
        self._base = base
        self._decoder = decoder
        self._max_tokens = max_tokens
        self._timeout = drain_timeout_seconds
        self._revision: ServingRevision | None = None
        self._loaded_adapter: str | None = None
        self._awake = False
        self._admitting = False
        self._active = 0
        self._condition = asyncio.Condition()
        self._control = asyncio.Lock()

    @property
    def revision(self) -> ServingRevision:
        """Return only a successfully loaded revision."""
        if self._revision is None:
            raise ServingPausedError(
                "no revision is loaded; wake and load the active registry revision"
            )
        return self._revision

    @property
    def policy_revision(self) -> str:
        """Return the revision label sampled by admitted requests."""
        return self.revision.policy_revision

    async def pause_and_drain(self) -> None:
        """Stop admission immediately and wait a bounded time for in-flight work."""
        async with self._control:
            async with self._condition:
                self._admitting = False
                async with asyncio.timeout(self._timeout):
                    await self._condition.wait_for(lambda: self._active == 0)

    def _require_drained(self) -> None:
        """Reject control changes until all admitted work has completed."""
        if self._admitting or self._active:
            raise ServingPausedError("pause and drain all requests before changing compute state")

    async def sleep(self) -> None:
        """Offload the unchanged base with sleep level one and verify the receipt."""
        async with self._control:
            self._require_drained()
            self._revision = None
            if self._loaded_adapter is not None:
                response = await self._client.post(
                    "/v1/unload_lora_adapter", json={"lora_name": self._loaded_adapter}
                )
                response.raise_for_status()
                self._loaded_adapter = None
            self._awake = False
            response = await self._client.post("/sleep", params={"level": 1})
            response.raise_for_status()
            if not await self._is_sleeping():
                raise ServingPausedError("vLLM did not confirm sleep; keep admission paused")

    async def wake(self) -> None:
        """Restore memory without reopening admission or changing the active pointer."""
        async with self._control:
            self._require_drained()
            self._revision = None
            self._awake = False
            response = await self._client.post("/wake_up")
            response.raise_for_status()
            if await self._is_sleeping():
                raise ServingPausedError("vLLM remains asleep; keep admission paused")
            self._awake = True

    async def _is_sleeping(self) -> bool:
        """Read the server's actual sleep flag instead of trusting a POST status alone."""
        response = await self._client.get("/is_sleeping")
        response.raise_for_status()
        return _SleepState.model_validate_json(response.content).is_sleeping

    async def load_revision(self, revision: ServingRevision) -> None:
        """Load one versioned LoRA while paused, preserving immutable base identity."""
        fields = ("scope", "model_id", "model_revision", "tokenizer_id", "tokenizer_revision")
        if any(getattr(revision, field) != getattr(self._base, field) for field in fields):
            raise ValueError("revision belongs to another scope, base weights, or tokenizer")
        async with self._control:
            self._require_drained()
            if not self._awake:
                raise ServingPausedError("wake the server before loading a serving revision")
            model = serving_model_name(revision)
            self._revision = None
            listed = await self._model_names()
            if serving_model_name(self._base) not in listed:
                raise ServingPausedError(
                    "vLLM has a different base route; keep admission paused "
                    "and restart the bound server"
                )
            if self._loaded_adapter is not None and self._loaded_adapter in listed:
                response = await self._client.post(
                    "/v1/unload_lora_adapter", json={"lora_name": self._loaded_adapter}
                )
                response.raise_for_status()
                listed.remove(self._loaded_adapter)
            self._loaded_adapter = None
            if revision.adapter_directory is not None:
                # A restarted controller cannot trust that a preexisting name
                # still points to the selected on-disk artifact. Reload it.
                if model in listed:
                    response = await self._client.post(
                        "/v1/unload_lora_adapter", json={"lora_name": model}
                    )
                    response.raise_for_status()
                response = await self._client.post(
                    "/v1/load_lora_adapter",
                    json={"lora_name": model, "lora_path": revision.adapter_directory},
                )
                response.raise_for_status()
                self._loaded_adapter = model
            if model not in await self._model_names():
                raise ServingPausedError(
                    "loaded revision is absent from vLLM models; keep admission paused"
                )
            self._revision = revision

    async def _model_names(self) -> set[str]:
        """Read actual route names from the configured private server."""
        response = await self._client.get("/v1/models")
        response.raise_for_status()
        return {item.id for item in _Models.model_validate_json(response.content).data}

    async def resume(self) -> None:
        """Reopen admission after the caller durably commits its selected revision."""
        async with self._control:
            self._require_drained()
            if not self._awake or self._revision is None:
                raise ServingPausedError("wake and load a verified revision before resuming")
            self._admitting = True

    async def sample(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> PolicySample:
        """Serve an admitted request or explicitly reject it during a sleep cycle."""
        return await self._sample(messages, tools, request_id, evaluation=False)

    async def sample_for_evaluation(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> PolicySample:
        """Sample a loaded candidate while public admission remains closed."""
        return await self._sample(messages, tools, request_id, evaluation=True)

    async def _sample(
        self,
        messages: tuple[ModelMessage, ...],
        tools: tuple[ToolSchema, ...],
        request_id: str,
        *,
        evaluation: bool,
    ) -> PolicySample:
        """Pin one revision until its complete sampling operation leaves the drain set."""
        async with self._control:
            async with self._condition:
                if not self._awake or self._revision is None or (self._admitting == evaluation):
                    raise ServingPausedError(
                        "application is paused or not in the requested evaluation mode"
                    )
                revision = self._revision
                self._active += 1
        try:
            sampler = VllmPolicySampler(
                client=self._client,
                revision=revision,
                decoder=self._decoder,
                max_tokens=self._max_tokens,
            )
            return await sampler.sample(messages, tools, request_id)
        finally:
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()
