"""Tests for the disconnect usage estimate."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.gateway.disconnect_estimate import (
    FALLBACK_CHARACTERS_PER_TOKEN,
    estimate_disconnect_usage,
)
from exp.runtime.gateway.native_settlement import StreamedOutput
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayEventKind


def _request() -> GatewayRequest:
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="Explain why the sky is blue in two sentences."),
        ),
    )


def _disconnect(usage: GatewayUsage | None = None) -> GatewayEvent:
    return GatewayEvent(
        kind=GatewayEventKind.FAILED,
        sequence_number=0,
        failure=GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED, safe_message="caller disconnected"
        ),
        usage=usage,
        usage_incomplete_due_to_disconnect=True,
    )


def _tokens(text: str) -> int:
    return len(reservation_encoder().encode_ordinary(text))


def test_opened_disconnect_without_any_report_is_priced_from_prompt_and_streamed_text() -> None:
    """The counted prompt and the tokenized deltas replace the meter the provider never sent."""
    request = _request()
    streamed = StreamedOutput(text="Sunlight scatters off air molecules.", reasoning="short waves")
    estimated = estimate_disconnect_usage(
        _disconnect(),
        request=request,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=streamed,
    )
    assert estimated.usage_estimated is True
    assert estimated.usage_incomplete_due_to_disconnect is True
    usage = estimated.usage
    assert usage is not None
    assert usage.input_tokens == counted_input_tokens(request) > 0
    assert usage.reasoning_tokens == _tokens("short waves")
    assert usage.output_tokens == _tokens(streamed.text) + _tokens("short waves")
    assert usage.cached_input_tokens is None
    assert "usage_estimated" not in estimated.model_dump()


def test_observed_legs_win_unless_the_streamed_text_already_exceeds_them() -> None:
    """An Anthropic message-start report (input, cache, one output token) keeps its input legs."""
    request = _request()
    observed = GatewayUsage(
        input_tokens=1_200, output_tokens=1, cached_input_tokens=1_000, reasoning_tokens=None
    )
    streamed = StreamedOutput(text="word " * 40)
    estimated = estimate_disconnect_usage(
        _disconnect(observed),
        request=request,
        surface=GatewayApiSurface.MESSAGES,
        opened=True,
        streamed=streamed,
    )
    usage = estimated.usage
    assert usage is not None
    assert usage.input_tokens == 1_200
    assert usage.cached_input_tokens == 1_000
    assert usage.output_tokens == _tokens(streamed.text) > 1
    assert usage.reasoning_tokens == 0
    # A running cumulative report at or above the estimate is the meter.
    ahead = GatewayUsage(input_tokens=1_200, output_tokens=500, reasoning_tokens=200)
    kept = estimate_disconnect_usage(
        _disconnect(ahead),
        request=request,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=streamed,
    ).usage
    assert kept is not None
    assert (kept.output_tokens, kept.reasoning_tokens) == (500, 200)


def test_cache_legs_are_dropped_without_a_reported_input_total() -> None:
    """A cache subset cannot be squared with an estimated prompt count."""
    observed = GatewayUsage(output_tokens=3, cached_input_tokens=50)
    usage = estimate_disconnect_usage(
        _disconnect(observed),
        request=_request(),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=None,
    ).usage
    assert usage is not None
    assert usage.cached_input_tokens is None
    assert usage.output_tokens == 3


def test_overflow_extrapolates_from_the_retained_ratio_or_the_fallback_density() -> None:
    """Text past the data plane's bound is counted, never forgotten."""
    retained = "alpha beta gamma delta " * 8
    usage = estimate_disconnect_usage(
        _disconnect(),
        request=_request(),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=StreamedOutput(text=retained, text_overflow_chars=len(retained)),
    ).usage
    assert usage is not None
    assert usage.output_tokens == 2 * _tokens(retained)
    bare = estimate_disconnect_usage(
        _disconnect(),
        request=_request(),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        opened=True,
        streamed=StreamedOutput(reasoning_overflow_chars=41),
    ).usage
    assert bare is not None
    assert bare.reasoning_tokens == -(-41 // FALLBACK_CHARACTERS_PER_TOKEN)
    assert bare.output_tokens == bare.reasoning_tokens


@pytest.mark.parametrize(
    ("opened", "surface", "marker"),
    [
        (False, GatewayApiSurface.CHAT_COMPLETIONS, True),
        (True, GatewayApiSurface.DECISIONS, True),
        (True, GatewayApiSurface.CHAT_COMPLETIONS, False),
    ],
)
def test_unopened_decision_and_ordinary_settlements_are_left_alone(
    opened: bool, surface: GatewayApiSurface, marker: bool
) -> None:
    """Only an opened, dispatched disconnect on a generation surface is estimated."""
    terminal = (
        _disconnect()
        if marker
        else GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=0,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED, safe_message="cancelled"
            ),
        )
    )
    unchanged = estimate_disconnect_usage(
        terminal,
        request=_request(),
        surface=surface,
        opened=opened,
        streamed=StreamedOutput(text="partial answer"),
    )
    assert unchanged is terminal
    assert unchanged.usage is None
    assert unchanged.usage_estimated is False


def test_estimated_marker_requires_a_disconnect_with_both_totals() -> None:
    """The marker never rides a clean terminal or a half-empty meter."""
    with pytest.raises(ValueError, match="estimated usage requires"):
        GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=1, output_tokens=1),
            usage_estimated=True,
        )
    with pytest.raises(ValueError, match="estimated usage requires"):
        GatewayEvent(
            kind=GatewayEventKind.FAILED,
            sequence_number=0,
            failure=GatewayFailure(
                failure_class=GatewayFailureClass.CANCELLED, safe_message="cancelled"
            ),
            usage=GatewayUsage(input_tokens=1),
            usage_incomplete_due_to_disconnect=True,
            usage_estimated=True,
        )
