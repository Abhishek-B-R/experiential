"""Real loopback sampling and fail-closed exact-token evidence validation."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx
import pytest
from pydantic import TypeAdapter

from exp.common.claas import ClaasScope
from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelMessage
from exp.common.tasks import ToolSchema
from exp.runtime.claas.registry import ServingRevision
from exp.runtime.claas.serving.decoding import HermesCompletionDecoder
from exp.runtime.claas.serving.vllm import VllmPolicySampler, serving_model_name

_JSON = TypeAdapter(JsonObject)


def revision() -> ServingRevision:
    """Return one exact application model fixture without downloading weights."""
    return ServingRevision(
        scope=ClaasScope(user_id="user", application_id="app"),
        policy_revision="base",
        model_id="Qwen/model",
        model_revision="a" * 40,
        tokenizer_id="Qwen/model",
        tokenizer_revision="b" * 40,
    )


class SamplingServer:
    """An actual HTTP server emitting original IDs that deliberately differ from text encoding."""

    def __init__(self) -> None:
        """Initialize deterministic responses and observable request evidence."""
        self.requests: list[tuple[str, JsonObject]] = []
        self.mode = "valid"
        self.server: ThreadingHTTPServer | None = None

    def dispatch(self, path: str, payload: JsonObject) -> JsonObject:
        """Return native vLLM wire shapes, with configurable invalid evidence."""
        self.requests.append((path, payload))
        if path == "/tokenize":
            return {"tokens": [11, 22], "count": 2, "max_model_len": 4096}
        result: JsonObject = {
            "id": "cmpl-exact",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "text": '<tool_call>{"name":"search","arguments":{"q":"claim"}}</tool_call>',
                    "finish_reason": "stop",
                    "prompt_token_ids": [11, 22],
                    "token_ids": [101, 102],
                    "logprobs": {
                        "tokens": ["token_id:101", "token_id:102"],
                        "token_logprobs": [-0.2, -0.3],
                    },
                }
            ],
        }
        choices = cast(list[JsonObject], result["choices"])
        if self.mode == "changed_prompt":
            choices[0]["prompt_token_ids"] = [11, 23]
        if self.mode == "missing_ids":
            choices[0].pop("token_ids")
        if self.mode == "mismatched_logprobs":
            choices[0]["logprobs"] = {
                "tokens": ["token_id:999", "token_id:102"],
                "token_logprobs": [-0.2, -0.3],
            }
        if self.mode == "wrong_model":
            result["model"] = "other-model"
        return result


@pytest.fixture
def server() -> Iterator[tuple[SamplingServer, str]]:
    """Serve the actual HTTP surface and prove requests traverse a socket."""
    state = SamplingServer()

    class Handler(BaseHTTPRequestHandler):
        """Minimal bounded fixture transport with no request logging."""

        def do_POST(self) -> None:
            """Decode one request and emit the fixture's deterministic server receipt."""
            payload = _JSON.validate_json(self.rfile.read(int(self.headers["Content-Length"])))
            body = json.dumps(state.dispatch(self.path, payload)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: str) -> None:
            """Suppress fixture HTTP access logs."""

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_real_http_preserves_original_tokens_and_tool_action(
    server: tuple[SamplingServer, str],
) -> None:
    """Exact IDs come from the server, and logical envelopes retain ordinary Chat shape."""
    state, url = server

    async def run() -> None:
        """Drive the public async sampler across a real loopback socket."""
        async with httpx.AsyncClient(base_url=url) as client:
            sampler = VllmPolicySampler(
                client=client, revision=revision(), decoder=HermesCompletionDecoder()
            )
            result = await sampler.sample(
                (ModelMessage(role="user", content="Check this claim"),),
                (ToolSchema(name="search", description="Search", input_schema={"type": "object"}),),
                "req",
            )
        assert result.exact_tokens.prompt_token_ids == (11, 22)
        assert result.exact_tokens.response_token_ids == (101, 102)
        assert result.exact_tokens.response_logprobs == (-0.2, -0.3)
        assert result.action.tool_calls[0].name == "search"
        assert result.request["messages"] == [{"role": "user", "content": "Check this claim"}]
        assert result.response["object"] == "chat.completion"
        assert state.requests[1][1]["prompt"] == [11, 22]
        assert state.requests[1][1]["temperature"] == 1.0
        assert state.requests[1][1]["top_k"] == -1

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", ["changed_prompt", "missing_ids", "mismatched_logprobs", "wrong_model"]
)
def test_server_evidence_cannot_be_reconstructed(
    server: tuple[SamplingServer, str], mode: str
) -> None:
    """Wrong or absent provenance produces a failure rather than fabricated evidence."""
    state, url = server
    state.mode = mode

    async def run() -> None:
        """Drive one invalid completion from the actual endpoint."""
        async with httpx.AsyncClient(base_url=url) as client:
            sampler = VllmPolicySampler(
                client=client, revision=revision(), decoder=HermesCompletionDecoder()
            )
            with pytest.raises(ValueError):
                await sampler.sample((ModelMessage(role="user", content="task"),), (), "req")

    asyncio.run(run())


def test_training_text_tokenization_uses_raw_prompt_and_no_special_tokens(
    server: tuple[SamplingServer, str],
) -> None:
    """Only the new teacher text crosses tokenization, without any generation request."""
    state, url = server

    async def run() -> None:
        """Request exact feedback IDs from the real local tokenizer HTTP endpoint."""
        async with httpx.AsyncClient(base_url=url) as client:
            sampler = VllmPolicySampler(
                client=client, revision=revision(), decoder=HermesCompletionDecoder()
            )
            assert await sampler.tokenize_training_text("\nFeedback: check the policy") == (11, 22)
            assert await sampler.tokenize_training_text("") == ()
            with pytest.raises(ValueError, match="one MiB"):
                await sampler.tokenize_training_text("a" * 1_048_577)
        assert state.requests == [
            (
                "/tokenize",
                {
                    "model": serving_model_name(revision()),
                    "prompt": "\nFeedback: check the policy",
                    "add_special_tokens": False,
                },
            )
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "receipt",
    [
        {"tokens": [True], "count": 1, "max_model_len": 4096},
        {"tokens": [-1], "count": 1, "max_model_len": 4096},
        {"tokens": [7], "count": 2, "max_model_len": 4096},
        {"tokens": [7, 8], "count": 2, "max_model_len": 1},
    ],
)
def test_training_text_rejects_invalid_token_receipts(receipt: JsonObject) -> None:
    """Strict feedback counts cannot accept boolean, negative, inconsistent, or oversized IDs."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Return one deliberately invalid tokenization response."""
        return httpx.Response(200, json=receipt)

    async def run() -> None:
        """Validate the server receipt without manufacturing replacement IDs."""
        async with httpx.AsyncClient(
            base_url="http://owned", transport=httpx.MockTransport(respond)
        ) as client:
            sampler = VllmPolicySampler(
                client=client, revision=revision(), decoder=HermesCompletionDecoder()
            )
            with pytest.raises(ValueError):
                await sampler.tokenize_training_text("feedback")

    asyncio.run(run())
