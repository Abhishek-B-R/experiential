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

from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayFailure, GatewayRequest
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.native_settlement import (
    StreamedOutput,
    streamed_output_from_settlement,
    terminal_from_settlement,
    tool_search_requests_from_settlement,
    web_search_requests_from_settlement,
)
from exp.runtime.gateway.reservation_tokenizer import reservation_encoder
from exp.runtime.gateway.stream_contracts import GatewayEvent, GatewayUsage

if TYPE_CHECKING:
    from exp.runtime.gateway.native_execution import InflightRequest

FALLBACK_CHARACTERS_PER_TOKEN = 4
"""Characters per token assumed for overflow text when nothing was retained to calibrate on."""


def settled_terminal(
    data: JsonObject,
    entry: InflightRequest,
    *,
    parsed: tuple[GatewayEvent, GatewayFailure | None] | None = None,
) -> tuple[GatewayEvent, GatewayFailure | None]:
    """Build one in-flight request's terminal from its settlement, disconnect estimate applied.

    Deterministic over the retained payload, so the direct settle and the
    sweep's replay of the same settlement produce the same meter.

    Args:
        data: Parsed native settlement payload.
        entry: The owning in-flight request (its prompt and frozen surface).
        parsed: Already validated terminal used to stamp receipt time before tokenization.

    Returns:
        The normalized terminal event and optional failure.
    """
    terminal, failure = parsed or terminal_from_settlement(
        data, surface=entry.authorization.surface
    )
    terminal = estimate_disconnect_usage(
        terminal,
        request=entry.request,
        surface=entry.authorization.surface,
        opened=data.get("opened") is True,
        streamed=streamed_output_from_settlement(data),
    )
    if terminal.usage_estimated and terminal.usage is not None:
        terminal = terminal.model_copy(
            update={
                "usage": terminal.usage.model_copy(
                    update={
                        "web_search_requests": web_search_requests_from_settlement(data),
                        "tool_search_requests": tool_search_requests_from_settlement(data),
                    }
                )
            }
        )
    return terminal, failure


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
    a completion attempt whose provider had accepted the request (``opened``)
    and whose data plane sent its generated-output evidence: a dispatch the
    provider never answered has no billed work to estimate, a data plane that
    predates the evidence (or sent it malformed) leaves the meter unknown, and
    generated images are billed per image, so any image keeps the meter
    unknown too. Explicit single-dial proof is required: a repaired dial's
    retained prefix cannot price earlier work or release its unresolved hold.
    Decisions, embeddings, and image requests keep their own contracts untouched.

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
        or not isinstance(request, GatewayRequest)
        or streamed is None
        or not streamed.single_dial
        or streamed.images > 0
    ):
        return terminal
    observed = terminal.usage
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
