"""Drop underscore-prefixed vendor-private top-level fields with disclosure.

Routers and proxies in front of the gateway attach their own bookkeeping to the
Chat body under an underscore-prefixed name (``_omnirouteSkipContextRelay`` is
the one seen most). The name is private to that hop and carries no meaning for
any downstream provider, so the field is removed here, before the compatibility
manifest sees the body, and named in the request's disclosed
``ignored_parameters``.

This is a whole-class rule rather than one entry per spelling: an underscore
prefix is the convention such hops already use, and each new spelling would
otherwise be a fresh pre-admission rejection. Dropping is never silent, so a
caller that believed the field did something still learns that it did not.
Every other unknown top-level field stays rejected by name.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject

VENDOR_PRIVATE_PREFIX = "_"
"""Prefix marking a top-level field as private to an intermediate hop."""


def vendor_private_disclosure(field: str) -> str:
    """Build the disclosure naming one dropped vendor-private field.

    Args:
        field: Top-level field name as the caller spelled it.

    Returns:
        The ``path->dropped(reason)`` disclosure recorded for the request.
    """
    return f"{field}->dropped(vendor_private)"


def drop_vendor_private_fields(payload: JsonObject) -> tuple[JsonObject, tuple[str, ...]]:
    """Remove underscore-prefixed top-level fields from one request body.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload when no such field is present; otherwise a shallow
        copy without them, paired with one disclosure per dropped field in the
        order the caller sent them.
    """
    dropped = tuple(field for field in payload if field.startswith(VENDOR_PRIVATE_PREFIX))
    if not dropped:
        return payload, ()
    remaining = {key: value for key, value in payload.items() if key not in dropped}
    return remaining, tuple(vendor_private_disclosure(field) for field in dropped)
