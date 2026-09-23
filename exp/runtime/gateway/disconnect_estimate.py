"""Estimate the meter of a dispatched attempt whose caller left before the provider's final usage.

Most OpenAI-shaped wires report token usage only in the stream's final frame.
When the caller disconnects first, the data plane closes the upstream and the
provider still bills the prompt it processed and the tokens it generated up to
the cut, so settling that attempt as unknown leaves real provider spend
uncharged. The gateway already tokenizes every prompt for its reservation and
already saw every generated delta, so it can price the work with its own
tokenizer: observed legs win, estimated legs fill the holes, and the result is
labelled ``estimated`` (never ``observed``) wherever it lands.
"""

from __future__ import annotations

from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.native_settlement import StreamedOutput
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayUsage

FALLBACK_CHARACTERS_PER_TOKEN = 4
"""Characters per token assumed for overflow text when nothing was retained to calibrate on."""


def estimate_disconnect_usage(
    terminal: GatewayEvent,
    *,
    request: ServingRequest,
    surface: GatewayApiSurface | None,
    opened: bool,
    streamed: StreamedOutput | None,
) -> GatewayEvent:
    """Fill a cancelled disconnect's unreported meter legs from gateway evidence.

    Applies only to the trusted ``usage_incomplete_due_to_disconnect`` marker on
    an attempt whose provider had accepted the request (``opened``): a dispatch
    the provider never answered has no billed work to estimate. Decisions keep
    their own hold contract untouched.

    Args:
        terminal: The normalized cancelled terminal from the settlement.
        request: The admitted request, whose prompt the gateway tokenizes.
        surface: The frozen request surface.
        opened: Whether the provider's response headers arrived.
        streamed: Generated text the data plane observed before the cut.

    Returns:
        The terminal with an ``estimated`` usage, or the terminal unchanged.
    """
    if (
        not terminal.usage_incomplete_due_to_disconnect
        or not opened
        or surface is GatewayApiSurface.DECISIONS
    ):
        return terminal
    observed = terminal.usage
    streamed = streamed or StreamedOutput()
    reasoning_estimate = _text_tokens(streamed.reasoning, streamed.reasoning_overflow_chars)
    visible_estimate = _text_tokens(streamed.text, streamed.text_overflow_chars)
    input_tokens = (
        counted_input_tokens(request)
        if observed is None or observed.input_tokens is None
        else observed.input_tokens
    )
    # Reasoning is an output subset (the ledger's pricing contract), so the
    # estimate folds it into output. A provider's running report can trail
    # the text already streamed (Anthropic reports one output token at
    # message start), so the larger of report and estimate is the meter.
    reasoning_tokens = _largest(
        None if observed is None else observed.reasoning_tokens, reasoning_estimate
    )
    output_tokens = _largest(
        None if observed is None else observed.output_tokens,
        visible_estimate + reasoning_estimate,
    )
    if reasoning_tokens is not None and reasoning_tokens > output_tokens:
        output_tokens = reasoning_tokens
    # Cache legs are subsets of a REPORTED input total; without one they
    # cannot be squared with the estimated prompt count and stay unknown.
    reported_input = observed is not None and observed.input_tokens is not None
    usage = GatewayUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=observed.cached_input_tokens if reported_input else None,
        cache_creation_input_tokens=(
            observed.cache_creation_input_tokens if reported_input else None
        ),
        cache_creation_1h_input_tokens=(
            observed.cache_creation_1h_input_tokens if reported_input else None
        ),
        reasoning_tokens=reasoning_tokens,
        tool_names=() if observed is None else observed.tool_names,
        web_search_requests=0 if observed is None else observed.web_search_requests,
        tool_search_requests=0 if observed is None else observed.tool_search_requests,
    )
    return terminal.model_copy(update={"usage": usage, "usage_estimated": True})


def _text_tokens(text: str, overflow_chars: int) -> int:
    """Count one output leg with the reservation tokenizer, extrapolating overflow."""
    counted = len(reservation_encoder().encode_ordinary(text)) if text else 0
    if overflow_chars == 0:
        return counted
    retained_chars = len(text)
    if retained_chars == 0 or counted == 0:
        return counted + -(-overflow_chars // FALLBACK_CHARACTERS_PER_TOKEN)
    return counted + -(-overflow_chars * counted // retained_chars)


def _largest(observed: int | None, estimate: int) -> int:
    """The observed count when it is at least the estimate, else the estimate."""
    return estimate if observed is None else max(observed, estimate)
