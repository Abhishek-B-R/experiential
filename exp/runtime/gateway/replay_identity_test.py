"""Deterministic request identity for native non-conversational decisions."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, canonical_json_bytes, sha256_json
from exp.runtime.gateway.decisions_contracts import ChoiceQuestion, DecisionRequest, NoulQuestion
from exp.runtime.gateway.replay_identity import canonical_request_sha256


def test_decision_identity_is_plain_canonical_json_independent_of_mapping_order() -> None:
    """State and question ordering do not create a different content fingerprint."""
    noul = NoulQuestion(instructions={"test": "clear", "language": "日本語"})
    choice = ChoiceQuestion(instructions="Select", criteria={"yes": "allowed", "no": None})
    request = DecisionRequest(state={"b": [2, 1], "a": "你好"}, questions={"n": noul, "c": choice})
    reordered = DecisionRequest(
        state={"a": "你好", "b": [2, 1]},
        questions={
            "c": ChoiceQuestion(instructions="Select", criteria={"no": None, "yes": "allowed"}),
            "n": NoulQuestion(instructions={"language": "日本語", "test": "clear"}),
        },
    )
    assert canonical_request_sha256(request) == sha256_json(request)
    assert canonical_request_sha256(request) == canonical_request_sha256(reordered)
    assert canonical_request_sha256(request) == canonical_request_sha256(
        DecisionRequest.model_validate_json(request.model_dump_json())
    )
    serialized = canonical_json_bytes(request)
    assert b'"surface":"decisions"' in serialized
    assert b'"messages"' not in serialized
    assert b'"idempotency_key"' not in serialized
    assert b'"input_token_reservation"' not in serialized


@pytest.mark.parametrize(
    "change",
    [
        {"state": {"value": 2}},
        {"questions": {"renamed": {"type": "noul", "instructions": "Accept?"}}},
        {"questions": {"check": {"type": "noul", "instructions": "Reject?"}}},
        {
            "questions": {
                "check": {"type": "noul", "instructions": "Accept?", "criteria": {"true": "valid"}}
            }
        },
        {
            "questions": {
                "check": {
                    "type": "choice",
                    "instructions": "Accept?",
                    "criteria": {"a": None, "b": None},
                }
            }
        },
        {
            "questions": {
                "check": {"type": "score", "instructions": "Accept?", "criteria": ["low", "high"]}
            }
        },
    ],
)
def test_decision_identity_changes_with_provider_significant_content(change: JsonObject) -> None:
    """State, identifiers, instructions, question types, and criteria all join the hash."""
    request = DecisionRequest(
        state={"value": 1}, questions={"check": NoulQuestion(instructions="Accept?")}
    )
    changed = DecisionRequest.model_validate({**request.model_dump(mode="json"), **change})
    assert canonical_request_sha256(request) != canonical_request_sha256(changed)


def test_score_criteria_order_is_significant() -> None:
    """Unlike object key order, ordered score criteria change the provider request."""
    request = DecisionRequest.model_validate(
        {
            "state": "record",
            "questions": {
                "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]}
            },
        }
    )
    reordered = DecisionRequest.model_validate(
        {
            "state": "record",
            "questions": {
                "score": {"type": "score", "instructions": "Rate", "criteria": ["high", "low"]}
            },
        }
    )
    assert canonical_request_sha256(request) != canonical_request_sha256(reordered)
