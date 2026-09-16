"""Admission and drain stay closed across sleep, reload, failure, and evaluation."""

from __future__ import annotations

import asyncio
from typing import cast

import httpx
import pytest
from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage
from exp.runtime.claas.decoding import TextCompletionDecoder
from exp.runtime.claas.lifecycle import ServingPausedError, VllmServingLifecycle
from exp.runtime.claas.vllm import serving_model_name
from exp.runtime.claas.vllm_test import revision

_JSON = TypeAdapter(JsonObject)


class ControlServer:
    """Observable server state with a deliberately held in-flight completion."""

    def __init__(self) -> None:
        """Start awake with only the exact pinned base route."""
        self.sleeping = False
        self.models = {serving_model_name(revision())}
        self.paths: list[str] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_load = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        """Implement control and sampling receipts while making ordering visible."""
        path = request.url.path
        self.paths.append(path)
        payload = _JSON.validate_json(request.content) if request.content else {}
        if path == "/sleep":
            self.sleeping = True
        if path == "/wake_up":
            self.sleeping = False
        if path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": self.sleeping})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": name} for name in self.models]})
        if path == "/v1/load_lora_adapter":
            if self.fail_load:
                return httpx.Response(500)
            self.models.add(cast(str, payload["lora_name"]))
        if path == "/v1/unload_lora_adapter":
            self.models.remove(cast(str, payload["lora_name"]))
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1], "count": 1, "max_model_len": 4096})
        if path == "/v1/completions":
            self.entered.set()
            await self.release.wait()
            return httpx.Response(
                200,
                json={
                    "id": "r",
                    "model": payload["model"],
                    "created": 1,
                    "choices": [
                        {
                            "index": 0,
                            "text": "done",
                            "finish_reason": "stop",
                            "prompt_token_ids": [1],
                            "token_ids": [2],
                            "logprobs": {"tokens": ["token_id:2"], "token_logprobs": [-0.3]},
                        }
                    ],
                },
            )
        return httpx.Response(200, json={})


def test_drain_precedes_sleep_and_evaluation_stays_private() -> None:
    """No sleep races with a completion and no candidate opens public admission."""

    async def run() -> None:
        """Drive one complete awake, drain, train-window, evaluate, resume cycle."""
        server = ControlServer()
        async with httpx.AsyncClient(
            base_url="http://owned", transport=httpx.MockTransport(server.handle)
        ) as client:
            lifecycle = VllmServingLifecycle(
                client=client, base=revision(), decoder=TextCompletionDecoder()
            )
            messages = (ModelMessage(role="user", content="task"),)
            with pytest.raises(ServingPausedError):
                await lifecycle.sample(messages, (), "before")
            await lifecycle.wake()
            await lifecycle.load_revision(revision())
            await lifecycle.resume()
            sampling = asyncio.create_task(lifecycle.sample(messages, (), "request"))
            await server.entered.wait()
            draining = asyncio.create_task(lifecycle.pause_and_drain())
            await asyncio.sleep(0)
            assert not draining.done()
            assert "/sleep" not in server.paths
            server.release.set()
            assert (await sampling).exact_tokens.policy_revision == "base"
            await draining
            await lifecycle.sleep()
            with pytest.raises(ServingPausedError):
                await lifecycle.sample(messages, (), "asleep")
            await lifecycle.wake()
            candidate = revision().model_copy(
                update={
                    "policy_revision": "candidate",
                    "adapter_directory": "/adapters/candidate/student",
                    "manifest_sha256": "a" * 64,
                }
            )
            await lifecycle.load_revision(candidate)
            result = await lifecycle.sample_for_evaluation(messages, (), "eval")
            assert result.exact_tokens.policy_revision == "candidate"
            with pytest.raises(ServingPausedError):
                await lifecycle.sample(messages, (), "public")
            await lifecycle.resume()
            assert (
                await lifecycle.sample(messages, (), "live")
            ).exact_tokens.policy_revision == "candidate"
            with pytest.raises(ServingPausedError):
                await lifecycle.sleep()

    asyncio.run(run())


def test_failed_reload_requires_restoring_known_revision() -> None:
    """A failed load cannot preserve a false serving label or reopen admission."""

    async def run() -> None:
        """Fail a candidate load, then explicitly restore the base and resume."""
        server = ControlServer()
        async with httpx.AsyncClient(
            base_url="http://owned", transport=httpx.MockTransport(server.handle)
        ) as client:
            lifecycle = VllmServingLifecycle(
                client=client, base=revision(), decoder=TextCompletionDecoder()
            )
            await lifecycle.wake()
            await lifecycle.load_revision(revision())
            server.fail_load = True
            candidate = revision().model_copy(
                update={
                    "policy_revision": "candidate",
                    "adapter_directory": "/adapters/candidate/student",
                    "manifest_sha256": "a" * 64,
                }
            )
            with pytest.raises(httpx.HTTPStatusError):
                await lifecycle.load_revision(candidate)
            with pytest.raises(ServingPausedError):
                await lifecycle.resume()
            with pytest.raises(ValueError, match="scope, base"):
                await lifecycle.load_revision(
                    candidate.model_copy(update={"model_revision": "other"})
                )
            await lifecycle.load_revision(revision())
            await lifecycle.resume()

    asyncio.run(run())


def test_sleep_and_controller_restart_reload_adapter_bytes() -> None:
    """Sleep unloads LoRA first, and a fresh controller reloads any existing target name."""

    async def run() -> None:
        """Drive repeated wake/load and a controller restart against one persistent server."""
        server = ControlServer()
        candidate = revision().model_copy(
            update={
                "policy_revision": "candidate",
                "adapter_directory": "/adapters/candidate/student",
                "manifest_sha256": "a" * 64,
            }
        )
        async with httpx.AsyncClient(
            base_url="http://owned", transport=httpx.MockTransport(server.handle)
        ) as client:
            lifecycle = VllmServingLifecycle(
                client=client, base=revision(), decoder=TextCompletionDecoder()
            )
            await lifecycle.wake()
            await lifecycle.load_revision(candidate)
            await lifecycle.sleep()
            assert serving_model_name(candidate) not in server.models
            assert server.paths.index("/v1/unload_lora_adapter") < server.paths.index("/sleep")
            await lifecycle.wake()
            with pytest.raises(ServingPausedError):
                await lifecycle.resume()
            await lifecycle.load_revision(candidate)
            fresh = VllmServingLifecycle(
                client=client, base=revision(), decoder=TextCompletionDecoder()
            )
            await fresh.wake()
            await fresh.load_revision(candidate)
            assert server.paths.count("/v1/load_lora_adapter") == 3
            assert server.paths.count("/v1/unload_lora_adapter") == 2
            server.models.remove(serving_model_name(revision()))
            with pytest.raises(ServingPausedError, match="different base"):
                await fresh.load_revision(candidate)
            with pytest.raises(ServingPausedError):
                await fresh.resume()

    asyncio.run(run())
