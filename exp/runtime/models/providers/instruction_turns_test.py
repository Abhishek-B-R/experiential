# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Folding instruction turns a lane's chat template cannot carry in place."""

from __future__ import annotations

from exp.common.models.model import ModelMessage
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.models.providers.instruction_turns import (
    fold_instruction_turns_after_the_first,
    fold_trailing_instruction_turns,
)


def test_trailing_instruction_after_a_user_turn_folds_into_that_turn() -> None:
    """Claude Code's Environment prompt follows the user turn; it becomes user text."""
    folded = fold_trailing_instruction_turns(
        (
            GatewayMessage(role="system", content="You are precise."),
            GatewayMessage(
                role="user",
                content="Fix the tests.",
                provider_text_blocks=(
                    {
                        "type": "text",
                        "text": "Fix the tests.",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(role="system", content="# Environment\nPlatform: linux"),
            GatewayMessage(role="developer", content="<total_tokens>1</total_tokens>"),
        )
    )
    assert [message.role for message in folded] == ["system", "user"]
    assert folded[0].content == "You are precise."
    assert folded[1].content == (
        "Fix the tests.\n\n# Environment\nPlatform: linux\n\n<total_tokens>1</total_tokens>"
    )
    # The cache-marker carrier no longer describes the flattened text.
    assert folded[1].provider_text_blocks == ()


def test_trailing_instruction_after_a_tool_result_becomes_a_user_turn() -> None:
    """After a tool_result the reminder cannot merge into a tool message; it is re-roled."""
    folded = fold_trailing_instruction_turns(
        (
            GatewayMessage(role="user", content="Run it."),
            GatewayMessage(role="assistant", content="", tool_calls=()),
            GatewayMessage(role="tool", content="ok", tool_call_id="call-1"),
            GatewayMessage(role="system", content="<total_tokens>1</total_tokens>"),
        )
    )
    assert [message.role for message in folded] == ["user", "assistant", "tool", "user"]
    assert folded[3].content == "<total_tokens>1</total_tokens>"


def test_instructions_followed_by_conversation_are_untouched() -> None:
    """Only a run that ENDS the conversation is folded; earlier ones keep their role."""
    messages = (
        GatewayMessage(role="system", content="You are precise."),
        GatewayMessage(role="user", content="hi"),
        GatewayMessage(role="system", content="Now be terse."),
        GatewayMessage(role="user", content="go"),
    )
    assert fold_trailing_instruction_turns(messages) == messages


def test_an_instruction_only_conversation_is_left_alone() -> None:
    messages = (GatewayMessage(role="system", content="You are precise."),)
    assert fold_trailing_instruction_turns(messages) == messages


def test_model_messages_fold_the_same_way() -> None:
    """The buffered builder's typed messages follow the identical rule."""
    folded = fold_trailing_instruction_turns(
        (
            ModelMessage(role="user", content="Fix it."),
            ModelMessage(role="system", content="Reminder."),
        )
    )
    assert len(folded) == 1
    assert folded[0].role == "user"
    assert folded[0].content == "Fix it.\n\nReminder."


def _claude_code_tool_loop() -> tuple[GatewayMessage, ...]:
    """Claude Code's live mid-conversation-system shape across one tool loop."""
    return (
        GatewayMessage(role="system", content="You are Claude Code."),
        GatewayMessage(role="user", content="Fix the tests."),
        GatewayMessage(role="system", content="# Environment\nPlatform: linux"),
        GatewayMessage(role="assistant", content="", tool_calls=()),
        GatewayMessage(role="tool", content="ok", tool_call_id="call-1"),
        GatewayMessage(role="system", content="<total_tokens>1</total_tokens>"),
        GatewayMessage(role="user", content="Now run them."),
    )


def test_leading_only_fold_keeps_the_first_system_turn_and_rewrites_every_other() -> None:
    """The Environment prompt joins the user turn; the tool-loop reminder is re-roled."""
    folded = fold_instruction_turns_after_the_first(_claude_code_tool_loop())
    assert [message.role for message in folded] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "user",
    ]
    assert folded[0].content == "You are Claude Code."
    assert folded[1].content == "Fix the tests.\n\n# Environment\nPlatform: linux"
    assert folded[4].content == "<total_tokens>1</total_tokens>"
    assert folded[5].content == "Now run them."


def test_leading_only_fold_merges_a_run_of_leading_instructions_into_one() -> None:
    """The Qwen template raises on a SECOND leading system turn too, so the run is one."""
    folded = fold_instruction_turns_after_the_first(
        (
            GatewayMessage(
                role="system",
                content="You are precise.",
                provider_text_blocks=(
                    {"type": "text", "text": "You are precise.", "cache_control": {"type": "e"}},
                ),
            ),
            GatewayMessage(role="developer", content="Answer in French."),
            GatewayMessage(role="user", content="hi"),
        )
    )
    assert [message.role for message in folded] == ["system", "user"]
    assert folded[0].content == "You are precise.\n\nAnswer in French."
    assert folded[0].provider_text_blocks == ()
    assert folded[1].content == "hi"


def test_leading_only_fold_keeps_a_leading_developer_turn_as_developer() -> None:
    """A single leading developer turn is emitted as system by the wire; no rewrite here."""
    messages = (
        GatewayMessage(role="developer", content="Be terse."),
        GatewayMessage(role="user", content="hi"),
    )
    assert fold_instruction_turns_after_the_first(messages) == messages


def test_leading_only_fold_returns_the_input_when_nothing_moves() -> None:
    messages = (
        GatewayMessage(role="system", content="You are precise."),
        GatewayMessage(role="user", content="hi"),
        GatewayMessage(role="assistant", content="hello"),
        GatewayMessage(role="user", content="go"),
    )
    assert fold_instruction_turns_after_the_first(messages) is messages
    no_system = (GatewayMessage(role="user", content="hi"),)
    assert fold_instruction_turns_after_the_first(no_system) is no_system


def test_leading_only_fold_leaves_non_text_instructions_in_place() -> None:
    """A replayed native item is not plain text and is never rewritten."""
    native = GatewayMessage(
        role="developer",
        provider_native_item={"type": "message", "role": "developer", "content": "verbatim"},
    )
    messages = (
        GatewayMessage(role="system", content="You are precise."),
        GatewayMessage(role="user", content="hi"),
        native,
        GatewayMessage(role="system", content="Reminder."),
    )
    folded = fold_instruction_turns_after_the_first(messages)
    assert folded[2] == native
    assert folded[3].role == "user"
    assert folded[3].content == "Reminder."


def test_leading_only_fold_leaves_an_instruction_with_ordered_anthropic_blocks_in_place() -> None:
    """The plural block carrier is structure the text does not describe; it is never rewritten."""
    carried = GatewayMessage(
        role="system",
        content="Reminder.",
        provider_anthropic_blocks=({"type": "text", "text": "Reminder."},),
    )
    messages = (
        GatewayMessage(role="system", content="You are precise."),
        GatewayMessage(role="user", content="hi"),
        carried,
    )
    folded = fold_instruction_turns_after_the_first(messages)
    assert folded == messages
    assert fold_trailing_instruction_turns(messages) == messages


def test_leading_only_fold_applies_to_model_messages() -> None:
    folded = fold_instruction_turns_after_the_first(
        (
            ModelMessage(role="system", content="You are precise."),
            ModelMessage(role="user", content="hi"),
            ModelMessage(role="system", content="Reminder."),
            ModelMessage(role="user", content="go"),
        )
    )
    assert [message.role for message in folded] == ["system", "user", "user"]
    assert folded[1].content == "hi\n\nReminder."
