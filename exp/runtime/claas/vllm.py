"""Token-in vLLM HTTP sampling for an explicitly pinned, privately controlled server.

The caller owns the HTTP client and must launch the server with the matching
model and tokenizer revisions. Returned token IDs and logprobs are required and
validated; text is never re-tokenized into claimed rollout evidence.
"""

from __future__ import annotations

from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from exp.common.claas import ExactTokenEvidence
from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import AssistantAction, ModelMessage
from exp.common.tasks import ToolSchema
from exp.runtime.claas.registry import ServingRevision
from exp.runtime.claas.sampling import CompletionDecoder, PolicySample

TokenId = Annotated[int, Field(strict=True, ge=0)]
Logprob = Annotated[float, Field(strict=True, allow_inf_nan=False, le=0)]
_JSON = TypeAdapter(JsonObject)


class _WireModel(BaseModel):
    """Validate required server evidence while allowing unrelated provider fields."""

    model_config = ConfigDict(extra="ignore")


class _Tokenized(_WireModel):
    """The tokenizer's original prompt IDs and context bound."""

    tokens: tuple[TokenId, ...] = Field(min_length=1)
    count: int = Field(strict=True, ge=1)
    max_model_len: int = Field(strict=True, ge=1)


class _Logprobs(_WireModel):
    """Per-generated-token behavior probabilities without null placeholders."""

    token_logprobs: tuple[Logprob, ...] = Field(min_length=1)
    tokens: tuple[str, ...] = Field(min_length=1)


class _Choice(_WireModel):
    """One complete token-in sampling result."""

    index: Literal[0]
    text: str
    finish_reason: Literal["stop", "length"]
    token_ids: tuple[TokenId, ...] = Field(min_length=1)
    prompt_token_ids: tuple[TokenId, ...] = Field(min_length=1)
    logprobs: _Logprobs


class _Completion(_WireModel):
    """The server identity and exactly one unmodified completion choice."""

    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    created: int = Field(strict=True, ge=0)
    choices: tuple[_Choice, ...] = Field(min_length=1, max_length=1)


def serving_model_name(revision: ServingRevision) -> str:
    """Name base and adapter routes by their complete immutable identity."""
    identity = revision.model_dump(mode="json")
    if revision.adapter_directory is None:
        identity.pop("policy_revision")
    return "claas-" + sha256_json(identity)


def _message(message: ModelMessage) -> JsonObject:
    """Render text and prior tool actions without losing tool-call linkage."""
    if message.content_parts:
        raise ValueError("CLaaS exact sampling supports text messages only")
    payload: JsonObject = {"role": message.role, "content": message.content}
    if message.role == "tool":
        payload["tool_call_id"] = message.tool_call_id
    if message.assistant_action:
        payload.update(_action_message(message.assistant_action))
    return payload


def _action_message(action: AssistantAction) -> JsonObject:
    """Render an executable action as an ordinary OpenAI Chat message."""
    result: JsonObject = {"role": "assistant", "content": action.content}
    if action.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in action.tool_calls
        ]
    return result


class VllmPolicySampler:
    """Sample exact original IDs using /tokenize then /v1/completions.

    Example:
        ``sampler = VllmPolicySampler(client=client, revision=revision,
        decoder=HermesCompletionDecoder(), max_tokens=2048)``
        ``sample = await sampler.sample(messages, tools, request_id="practice-1")``

    The private server must expose token IDs and use unmodified full-distribution
    sampling. Generation controls override model generation-config defaults.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        revision: ServingRevision,
        decoder: CompletionDecoder,
        max_tokens: int = 2048,
    ) -> None:
        """Bind an owned transport, immutable identity, decoder, and finite output cap."""
        if isinstance(max_tokens, bool) or not 1 <= max_tokens <= 131072:
            raise ValueError("max_tokens must be between one and 131072")
        self._client = client
        self._revision = revision
        self._decoder = decoder
        self._max_tokens = max_tokens

    @property
    def revision(self) -> ServingRevision:
        """Return the frozen revision used for this sampler's every request."""
        return self._revision

    @property
    def policy_revision(self) -> str:
        """Return the frozen policy label without consulting a mutable registry."""
        return self._revision.policy_revision

    async def sample(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSchema, ...], request_id: str
    ) -> PolicySample:
        """Generate one original token sequence and normalize only its logical envelope."""
        if not messages or not request_id.strip():
            raise ValueError("sampling needs messages and a nonblank request_id")
        model = serving_model_name(self.revision)
        logical: JsonObject = {
            "model": model,
            "messages": [_message(message) for message in messages],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if tools:
            logical["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in tools
            ]
        token_request = {
            key: value for key, value in logical.items() if key in ("model", "messages", "tools")
        }
        token_request.update({"add_generation_prompt": True, "add_special_tokens": False})
        token_response = await self._client.post("/tokenize", json=token_request)
        token_response.raise_for_status()
        prompt = _Tokenized.model_validate_json(token_response.content)
        if (
            prompt.count != len(prompt.tokens)
            or prompt.count + self._max_tokens > prompt.max_model_len
        ):
            raise ValueError(
                "exact prompt exceeds server context; shorten the scenario or output cap"
            )
        raw_request: JsonObject = {
            "model": model,
            "prompt": list(prompt.tokens),
            "request_id": request_id,
            "max_tokens": self._max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "n": 1,
            "stream": False,
            "echo": False,
            "logprobs": 0,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "add_special_tokens": False,
        }
        result = await self._client.post("/v1/completions", json=raw_request)
        result.raise_for_status()
        raw = _JSON.validate_json(result.content)
        completion = _Completion.model_validate(raw)
        choice = completion.choices[0]
        if completion.model != model or choice.prompt_token_ids != prompt.tokens:
            raise ValueError("vLLM returned another model or changed original prompt token IDs")
        if choice.logprobs.tokens != tuple(f"token_id:{token}" for token in choice.token_ids):
            raise ValueError("vLLM logprobs are not aligned with original returned token IDs")
        if len(choice.token_ids) > self._max_tokens:
            raise ValueError("server completion exceeds the requested maximum token count")
        exact = ExactTokenEvidence(
            model_id=self.revision.model_id,
            model_revision=self.revision.model_revision,
            policy_revision=self.policy_revision,
            tokenizer_id=self.revision.tokenizer_id,
            tokenizer_revision=self.revision.tokenizer_revision,
            sampling_temperature=1.0,
            sampling_top_p=1.0,
            sampling_top_k=None,
            prompt_token_ids=prompt.tokens,
            response_token_ids=choice.token_ids,
            response_logprobs=choice.logprobs.token_logprobs,
        )
        action = self._decoder.decode(choice.text, request_id, tools)
        unknown = {call.name for call in action.tool_calls} - {tool.name for tool in tools}
        if unknown:
            raise ValueError(
                "policy emitted an undeclared tool; retain the completion as a failure"
            )
        response: JsonObject = {
            "id": completion.id,
            "object": "chat.completion",
            "model": model,
            "created": completion.created,
            "choices": [
                {
                    "index": 0,
                    "message": _action_message(action),
                    "finish_reason": "length"
                    if choice.finish_reason == "length"
                    else "tool_calls"
                    if action.tool_calls
                    else "stop",
                }
            ],
        }
        return PolicySample(
            action=action,
            exact_tokens=exact,
            request=logical,
            response=response,
            raw_completion=raw,
        )
