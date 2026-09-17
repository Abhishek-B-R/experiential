"""OpenRouter provider-routing preferences the gateway sets per request.

OpenRouter load-balances one model id across upstream providers and accepts a
request-level ``provider`` object that narrows the candidates. The gateway
uses exactly one such preference: the zero-data-retention constraint, which
restricts the request to OpenRouter's published ZDR endpoint list (the public
``GET /api/v1/endpoints/zdr``) and denies data collection, so the aggregator
serves from a retention-free upstream or refuses. The constraint TIGHTENS only:
any preference already on the payload is kept, and ``zdr`` / ``data_collection``
are forced to the strict values regardless of what was there.

The metadata opt-in header makes OpenRouter name the upstream it selected on
the response, which the data plane records per attempt as the settlement's
``upstream_provider``.
"""

from __future__ import annotations

from typing import Final

from exp.common.core.artifacts import JsonObject, JsonValue

OPENROUTER_PROVIDER_ID: Final = "openrouter"
"""The catalog provider id of OpenRouter rungs (the only wire with this knob)."""

OPENROUTER_METADATA_HEADER: Final = "X-OpenRouter-Metadata"
OPENROUTER_METADATA_ENABLED: Final = "enabled"
"""Opt-in response metadata: the selected endpoint's provider rides the body."""

ZDR_PROVIDER_PREFERENCES: Final[JsonObject] = {"zdr": True, "data_collection": "deny"}
"""The strict values the constraint forces onto ``payload["provider"]``."""


def constrain_openrouter_zero_data_retention(payload: JsonObject) -> JsonObject:
    """Return ``payload`` with OpenRouter's ZDR routing constraint applied.

    Args:
        payload: A built Chat Completions payload for an OpenRouter rung.

    Returns:
        A new payload whose ``provider`` object carries ``zdr: true`` and
        ``data_collection: "deny"``. Other keys of an existing ``provider``
        object survive; a non-object ``provider`` value is replaced, and a
        looser ``zdr: false`` or ``data_collection: "allow"`` is overridden.
        The input is never mutated.
    """
    existing = payload.get("provider")
    preferences: JsonObject = dict(existing) if isinstance(existing, dict) else {}
    preferences.update(ZDR_PROVIDER_PREFERENCES)
    tightened: JsonValue = preferences
    return {**payload, "provider": tightened}


def openrouter_metadata_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return ``headers`` plus the OpenRouter metadata opt-in.

    Args:
        headers: The rung's static wire headers.

    Returns:
        A new mapping with ``X-OpenRouter-Metadata: enabled`` added.
    """
    return {**headers, OPENROUTER_METADATA_HEADER: OPENROUTER_METADATA_ENABLED}
