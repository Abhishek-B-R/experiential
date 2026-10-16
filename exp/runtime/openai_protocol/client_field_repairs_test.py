"""Tests for the disclosed client-field repairs on the OpenAI surfaces."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.client_field_repairs import (
    CHAT_COT_ID_DROPPED,
    OUTPUT_PHASE_DROPPED,
    OUTPUT_STATUS_DROPPED,
    STREAM_OBFUSCATION_IGNORED,
    STREAM_OPTIONS_NOT_STREAMING,
    accept_responses_stream_options,
    drop_chat_message_cot_ids,
    drop_unanchored_output_markers,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def test_untagged_chat_payload_is_returned_unchanged() -> None:
    """No ``cot_id`` means no copy and no disclosure."""
    payload: JsonObject = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    repaired, disclosures = drop_chat_message_cot_ids(payload)
    assert repaired is payload
    assert disclosures == ()


def test_cot_id_is_removed_from_every_message_without_mutating_the_caller() -> None:
    """Every tagged message loses the tag; one disclosure names the removal."""
    payload: JsonObject = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi", "cot_id": "a"},
            {"role": "assistant", "content": "yo", "cot_id": "b"},
        ],
    }
    repaired, disclosures = drop_chat_message_cot_ids(payload)
    assert repaired["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    assert disclosures == (CHAT_COT_ID_DROPPED,)
    assert cast("list[JsonObject]", payload["messages"])[0]["cot_id"] == "a"


def test_markers_with_an_item_id_are_kept() -> None:
    """A provider-issued output item replays exactly; nothing is repaired."""
    payload: JsonObject = {
        "input": [
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed"},
            {"type": "function_call", "status": "completed", "call_id": "c", "name": "f"},
        ]
    }
    repaired, disclosures = drop_unanchored_output_markers(payload)
    assert repaired is payload
    assert disclosures == ()


def test_input_status_on_non_assistant_messages_is_left_for_the_wire_model() -> None:
    """Official input-message status needs no repair (the wire model accepts it)."""
    payload: JsonObject = {"input": [{"role": "user", "content": "x", "status": "completed"}]}
    repaired, disclosures = drop_unanchored_output_markers(payload)
    assert repaired is payload
    assert disclosures == ()


def test_unanchored_status_and_phase_are_dropped_and_each_kind_disclosed_once() -> None:
    """Status (assistant only) and phase (any role) go when the item has no id."""
    payload: JsonObject = {
        "input": [
            {"role": "assistant", "content": "a", "status": "completed"},
            {"role": "assistant", "content": "b", "status": "completed", "phase": "commentary"},
            {"role": "user", "content": "c", "phase": "final_answer"},
        ]
    }
    repaired, disclosures = drop_unanchored_output_markers(payload)
    assert repaired["input"] == [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert disclosures == (OUTPUT_STATUS_DROPPED, OUTPUT_PHASE_DROPPED)


@pytest.mark.parametrize(
    ("payload", "disclosures"),
    (
        ({"stream": True, "stream_options": {"include_obfuscation": False}}, ()),
        ({"stream": True, "stream_options": None}, ()),
        (
            {"stream": True, "stream_options": {"include_obfuscation": True}},
            (STREAM_OBFUSCATION_IGNORED,),
        ),
        ({"stream_options": {}}, (STREAM_OPTIONS_NOT_STREAMING,)),
    ),
)
def test_stream_options_are_consumed(payload: JsonObject, disclosures: tuple[str, ...]) -> None:
    """The object never reaches the wire models; only ignored settings are disclosed."""
    repaired, named = accept_responses_stream_options(payload)
    assert "stream_options" not in repaired
    assert named == disclosures


@pytest.mark.parametrize(
    ("options", "param"),
    (
        ("yes", "stream_options"),
        ({"include_usage": True}, "stream_options.include_usage"),
        ({"include_obfuscation": "no"}, "stream_options.include_obfuscation"),
    ),
)
def test_malformed_stream_options_keep_a_named_rejection(options: JsonValue, param: str) -> None:
    """A malformed object or unknown key is the caller's named 400."""
    with pytest.raises(OpenAIProtocolError) as error:
        accept_responses_stream_options({"stream": True, "stream_options": options})
    assert error.value.detail.param == param
