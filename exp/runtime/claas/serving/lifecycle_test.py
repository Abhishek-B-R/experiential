"""Admission and drain stay closed across sleep, reload, failure, and evaluation."""

from __future__ import annotations

import asyncio
from typing import cast

import httpx
import pytest
from pydantic import TypeAdapter

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage
from exp.runtime.claas.serving.decoding import TextCompletionDecoder
from exp.runtime.claas.serving.lifecycle import ServingPausedError, VllmServingLifecycle
from exp.runtime.claas.serving.vllm import serving_model_name
from exp.runtime.claas.serving.vllm_test import revision

_JSON = TypeAdapter(JsonObject)


class ControlServer:
    """Observable server state with a deliberately held in-flight completion."""

    def __init__(self) -> None:
        """Start awake with only the exact pinned base route."""
        self.sleeping = False
        self.models = {serving_model_name(revision())}
        self.paths: list[str] = []
        self.completions: list[JsonObject] = []
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
            self.completions.append(payload)
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


@pytest.mark.parametrize(("requested", "expected"), [(None, 256), (7, 7), (4000, 256)])
def test_evaluation_response_cap_reaches_http_payload(requested: int | None, expected: int) -> None:
    """The HTTP generation cap honors a smaller caller budget and the server ceiling."""

    async def run() -> None:
        """Load a private revision and inspect its actual completion request body."""
        server = ControlServer()
        server.release.set()
        async with httpx.AsyncClient(
            base_url="http://owned", transport=httpx.MockTransport(server.handle)
        ) as client:
            lifecycle = VllmServingLifecycle(
                client=client, base=revision(), decoder=TextCompletionDecoder(), max_tokens=256
            )
            await lifecycle.wake()
            await lifecycle.load_revision(revision())
            await lifecycle.sample_for_evaluation(
                (ModelMessage(role="user", content="task"),), (), "eval", max_tokens=requested
            )
            assert server.completions[0]["max_tokens"] == expected
            with pytest.raises(ValueError, match="max_tokens"):
                await lifecycle.sample_for_evaluation((), (), "invalid", max_tokens=0)
            assert len(server.completions) == 1

    asyncio.run(run())


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


def test_feedback_tokenization_requires_paused_revision_and_holds_drain_lease() -> None:
    """Feedback counting cannot race sleep or reload, and cancellation releases its lease."""

    async def run() -> None:
        """Hold a tokenizer response while control attempts to drain the loaded revision."""
        server = ControlServer()
        entered, release = asyncio.Event(), asyncio.Event()

        async def handle(request: httpx.Request) -> httpx.Response:
            """Hold only raw teacher tokenization and delegate ordinary control operations."""
            if request.url.path == "/tokenize":
                assert _JSON.validate_json(request.content) == {
                    "model": serving_model_name(revision()),
                    "prompt": "teacher feedback",
                    "add_special_tokens": False,
                }
                entered.set()
                await release.wait()
                return httpx.Response(200, json={"tokens": [7], "count": 1, "max_model_len": 4096})
            return await server.handle(request)

        async with httpx.AsyncClient(
            base_url="http://owned", transport=httpx.MockTransport(handle)
        ) as client:
            lifecycle = VllmServingLifecycle(
                client=client, base=revision(), decoder=TextCompletionDecoder()
            )
            with pytest.raises(ServingPausedError):
                await lifecycle.tokenize_training_text("teacher feedback")
            await lifecycle.wake()
            await lifecycle.load_revision(revision())
            await lifecycle.resume()
            with pytest.raises(ServingPausedError):
                await lifecycle.tokenize_training_text("teacher feedback")
            await lifecycle.pause_and_drain()
            tokenizing = asyncio.create_task(lifecycle.tokenize_training_text("teacher feedback"))
            await entered.wait()
            with pytest.raises(ServingPausedError, match="pause and drain"):
                await lifecycle.sleep()
            draining = asyncio.create_task(lifecycle.pause_and_drain())
            await asyncio.sleep(0.01)
            assert not draining.done()
            tokenizing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await tokenizing
            await draining
            release.set()
            assert await lifecycle.tokenize_training_text("teacher feedback") == (7,)
            await lifecycle.sleep()

    asyncio.run(run())
