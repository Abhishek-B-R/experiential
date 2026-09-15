"""Translate alternate enable-thinking Chat request shapes to canonical reasoning.

Clients express "turn thinking on" several non-canonical ways on
/v1/chat/completions: the Responses-style nested ``reasoning:{effort}``,
OpenRouter's unified ``reasoning:{enabled, max_tokens, exclude}``, the
Anthropic-style ``thinking:{type}`` (``enabled`` or ``adaptive``), the vLLM-native
``chat_template_kwargs:{enable_thinking}``, and DashScope's top-level
``enable_thinking``. Each is admitted and translated here to the canonical flat
``reasoning_effort`` (never dropped — dropping would leave thinking silently
off), so one caller payload works in any shape. The model-aware default effort
for a level-less enable is resolved later, at the route adaptation seam, via
``GatewayRequest.thinking_default_enable``.
"""

from __future__ import annotations

from exp.common.models.model import ReasoningEffort
from exp.runtime.models.providers.reasoning_compat import thinking_config_reasoning_effort
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.wire_models import _ChatRequest

# Disclosure tokens (unified path->action(reason) vocabulary).
_TRANSLATED = "{path}->translated(reasoning_effort)"
_IGNORED = "{path}->ignored(explicit_reasoning_effort)"
_BUDGET_DROPPED = "budget_tokens->dropped(not_carried)"
_MAX_TOKENS_TRANSLATED = "reasoning.max_tokens->translated(reasoning_effort)"
_EXCLUDE_DROPPED = "reasoning.exclude->dropped(not_carried)"


class _EnableThinkingResult:
    """The resolved canonical reasoning controls plus caller disclosures."""

    __slots__ = ("reasoning_effort", "thinking_default_enable", "disclosures")

    def __init__(
        self,
        reasoning_effort: ReasoningEffort | None,
        thinking_default_enable: bool,
        disclosures: tuple[str, ...],
    ) -> None:
        self.reasoning_effort = reasoning_effort
        self.thinking_default_enable = thinking_default_enable
        self.disclosures = disclosures


def _reasoning_object_intent(request: _ChatRequest) -> bool | None:
    """Fold OpenRouter's ``reasoning`` object into one enable/disable vote.

    ``effort`` votes by tier (``none`` disables), ``enabled`` votes literally,
    and a ``max_tokens`` budget is an enable (OpenRouter infers ``enabled``
    from either depth control). The three must agree; an explicit
    ``enabled: false`` beside a tier or budget is a caller error named by
    field. An empty object carries no intent.
    """
    reasoning = request.reasoning
    if reasoning is None:
        return None
    votes: list[bool] = []
    if reasoning.effort is not None:
        votes.append(reasoning.effort != "none")
    if reasoning.max_tokens is not None:
        votes.append(True)
    if reasoning.enabled is not None:
        votes.append(reasoning.enabled)
    if votes and any(vote != votes[0] for vote in votes):
        raise invalid_field(
            "reasoning.enabled",
            "reasoning.enabled must agree with reasoning.effort / reasoning.max_tokens: "
            "an effort tier or a token budget turns thinking on.",
        )
    return votes[0] if votes else None


def translate_enable_thinking(request: _ChatRequest) -> _EnableThinkingResult:
    """Resolve the effective reasoning control from the flat and alternate fields.

    The explicit flat ``reasoning_effort`` always wins; a level-less enable defers
    to the model default (``thinking_default_enable``). Alternate fields that
    disagree on enable-vs-disable are a caller error and rejected by name.
    OpenRouter's ``max_tokens`` budget maps to the nearest effort tier through
    the same table the Messages surface uses for a thinking budget (disclosed as
    translated); ``exclude`` has no canonical equivalent and is disclosed as not
    carried.
    """
    reasoning = request.reasoning
    reasoning_intent = _reasoning_object_intent(request)
    reasoning_present = reasoning_intent is not None

    thinking_enable: bool | None = None
    thinking_present = request.thinking is not None
    if request.thinking is not None:
        # ``adaptive`` is Anthropic's 4.6+ on-mode; on this surface it carries
        # the same intent as ``enabled`` (think at the route's default depth).
        thinking_enable = request.thinking.type in {"enabled", "adaptive"}

    cck_enable = (
        request.chat_template_kwargs.enable_thinking
        if request.chat_template_kwargs is not None
        else None
    )
    cck_present = cck_enable is not None
    flat_enable_present = request.enable_thinking is not None

    # Fields present-but-inert: told about, never carried.
    dropped: list[str] = []
    if request.thinking is not None and request.thinking.budget_tokens is not None:
        dropped.append(_BUDGET_DROPPED)
    if reasoning is not None and reasoning.max_tokens is not None:
        dropped.append(_MAX_TOKENS_TRANSLATED)
    if reasoning is not None and reasoning.exclude:
        dropped.append(_EXCLUDE_DROPPED)

    alternates = (
        ("reasoning", reasoning_present),
        ("thinking", thinking_present),
        ("chat_template_kwargs", cck_present),
        ("enable_thinking", flat_enable_present),
    )

    # Explicit flat reasoning_effort wins: every present alternate field is a no-op
    # the caller is told about, and the flat value is passed through unchanged.
    if request.reasoning_effort is not None:
        disclosures = [
            *dropped,
            *(_IGNORED.format(path=path) for path, present in alternates if present),
        ]
        return _EnableThinkingResult(request.reasoning_effort, False, tuple(disclosures))

    # No explicit flat value: fold the alternate fields into one intent. Each
    # present field votes enable ("on", possibly at a level) or disable ("none").
    votes = [
        vote
        for vote in (reasoning_intent, thinking_enable, cck_enable, request.enable_thinking)
        if vote is not None
    ]
    if votes and any(vote != votes[0] for vote in votes):
        raise invalid_field(
            "thinking",
            "conflicting enable-thinking fields: reasoning/thinking/chat_template_kwargs/"
            "enable_thinking must all enable or all disable.",
        )

    disclosures = [
        *dropped,
        *(_TRANSLATED.format(path=path) for path, present in alternates if present),
    ]

    if not votes:
        # No alternate field carried an intent (absent, or an empty object).
        return _EnableThinkingResult(None, False, tuple(disclosures))
    if votes[0] is False:
        # All present fields disable → canonical none.
        return _EnableThinkingResult("none", False, tuple(disclosures))
    # Enabled: a nested reasoning effort pins the level, a budget snaps to its
    # nearest tier; otherwise defer the model-aware default to the adaptation
    # seam.
    if reasoning is not None and reasoning.effort is not None:
        return _EnableThinkingResult(reasoning.effort, False, tuple(disclosures))
    if reasoning is not None and reasoning.max_tokens is not None:
        budget_effort = thinking_config_reasoning_effort(
            {"type": "enabled", "budget_tokens": reasoning.max_tokens}
        )
        return _EnableThinkingResult(budget_effort, False, tuple(disclosures))
    return _EnableThinkingResult(None, True, tuple(disclosures))
