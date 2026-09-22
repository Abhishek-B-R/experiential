"""Conservative full-request token counting before provider dispatch."""

from typing import Protocol, runtime_checkable

from exp.common.models import ModelRequest


@runtime_checkable
class TokenCounter(Protocol):
    """Counts the full serialized request before a model client can send it."""

    def count(self, request: ModelRequest) -> int:
        """Return a conservative number of context tokens required by one request.

        Args:
            request: Complete provider-neutral request before provider conversion.

        Returns:
            A nonnegative count that includes all visible request content.
        """
        ...


class Utf8UpperBoundTokenCounter:
    """Provider-neutral byte upper bound used when no exact tokenizer is supplied."""

    def count(self, request: ModelRequest) -> int:
        """Count UTF-8 request bytes plus per-message framing as a conservative token bound.

        Args:
            request: Complete provider-neutral request to preflight.

        Returns:
            A conservative nonnegative bound that never silently shortens request content.
        """
        rendered = request.model_dump_json(exclude_none=False)
        return len(rendered.encode("utf-8")) + 4 * len(request.messages)
