"""Numeric thinking caps survive decoding, admission, dispatch and replay identity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat

_PROFILE = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions",
    model_id="qwen3.8-max",
    supports_reasoning=True,
    reasoning_wire_format="reasoning_effort",
    supported_reasoning_efforts=("low", "medium", "xhigh"),
)


def test_budget_reaches_qwen_without_an_advisory_effort() -> None:
    """An enable switch cannot cause the provider to receive a second depth dial."""
    request = decode_chat(
        {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 4096,
            "enable_thinking": True,
            "max_output_tokens": 8192,
        }
    ).request
    public, provider = route_generation_parameter_requests((_PROFILE,), request)
    payload = dialect_stream_payload(_PROFILE, provider)
    assert payload["thinking_budget"] == 4096
    assert payload["enable_thinking"] is True
    assert payload["max_completion_tokens"] == 8192
    assert "max_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert "reasoning" not in payload
    assert public.thinking_budget == 4096
    assert provider.thinking_default_enable is False
    changed = request.model_copy(update={"thinking_budget": 2048})
    assert canonical_request_sha256(request) != canonical_request_sha256(changed)


@pytest.mark.parametrize(
    "profile",
    (
        replace(_PROFILE, url="https://api.openai.com/v1/chat/completions"),
        replace(_PROFILE, url="https://openrouter.ai/api/v1/chat/completions"),
        replace(
            _PROFILE, url="https://dashscope-intl.aliyuncs.com.attacker.test/v1/chat/completions"
        ),
        replace(_PROFILE, dialect="openai_responses"),
        replace(_PROFILE, supports_reasoning=False, supported_reasoning_efforts=()),
        replace(_PROFILE, reasoning_effort_required=True, reasoning_effort="medium"),
        replace(_PROFILE, model_id="non-reasoning-model"),
    ),
)
def test_unsupported_routes_never_drop_the_budget(profile: GatewayWireProfile) -> None:
    """Both admission and direct dispatch reject incapable or unknown wires."""
    request = decode_chat(
        {
            "model": "qwen3.8-max",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 4096,
        }
    ).request
    with pytest.raises(ProviderParameterError) as admission:
        route_generation_parameter_requests((profile,), request)
    assert admission.value.param == "thinking_budget"
    with pytest.raises(ProviderParameterError) as dispatch:
        dialect_stream_payload(profile, request)
    assert dispatch.value.param == "thinking_budget"


@pytest.mark.parametrize("budget", (0, -1, True, "4096", 1.5, {}))
def test_invalid_budgets_fail_at_decode(budget: object) -> None:
    """Numeric bounds cannot be coerced from bools, strings or fractional values."""
    from typing import cast

    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            cast(
                JsonObject,
                {
                    "model": "qwen3.8-max",
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking_budget": budget,
                },
            )
        )
    assert error.value.detail.param == "thinking_budget"


@pytest.mark.parametrize(
    "control",
    (
        {"enable_thinking": False},
        {"reasoning_effort": "high"},
        {"reasoning": {"effort": "low"}},
        {"reasoning": {"enabled": False}},
        {"thinking": {"type": "disabled"}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ),
)
def test_budget_rejects_conflicting_controls(control: JsonObject) -> None:
    """An explicit off switch or effort must not erase the numeric budget."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat(
            {
                "model": "qwen3.8-max",
                "messages": [{"role": "user", "content": "hi"}],
                "thinking_budget": 4096,
                **control,
            }
        )
    assert error.value.detail.param == "thinking_budget"
