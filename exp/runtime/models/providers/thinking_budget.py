"""Preserve numeric Chat thinking budgets on native Qwen Cloud wires."""

from __future__ import annotations

from urllib.parse import urlsplit

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError

_QWEN_BUDGET_HOSTS = frozenset(
    {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
        "maas.qwencloudapi.com",
        "token-plan.ap-southeast-1.maas.aliyuncs.com",
    }
)


def require_thinking_budget_support(profile: GatewayWireProfile, request: GatewayRequest) -> None:
    """Refuse a numerical budget unless the selected wire can carry it unchanged.

    Qwen Cloud's Chat API documents ``thinking_budget`` with ``enable_thinking``.
    A generic reasoning-effort capability is insufficient: other compatible
    servers may ignore unknown fields or replace a numerical cap with a tier.
    """
    if request.thinking_budget is None:
        return
    origin = urlsplit(profile.url)
    if (
        profile.dialect != "openai_compatible"
        or origin.scheme != "https"
        or origin.hostname not in _QWEN_BUDGET_HOSTS
        or not (profile.model_id == "qwen3.8-max" or profile.model_id.startswith("qwen3.8-max-"))
        or not profile.supports_reasoning
        or profile.reasoning_effort_required
    ):
        raise ProviderParameterError(
            message=(
                "This route cannot enforce thinking_budget. Choose a native Qwen3.8-Max Cloud "
                "reasoning route, or remove thinking_budget and explicitly choose reasoning_effort."
            ),
            param="thinking_budget",
            code="unsupported_parameter",
        )
    if request.reasoning_effort is not None:
        raise ProviderParameterError(
            message="thinking_budget and reasoning_effort are mutually exclusive. Choose one.",
            param="thinking_budget",
            code="invalid_parameter",
        )
