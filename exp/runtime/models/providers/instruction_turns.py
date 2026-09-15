# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Folding instruction turns a lane's chat template cannot carry in place.

Coding agents put ``system`` turns INSIDE the conversation: Claude Code's
mid-conversation-system beta appends its Environment prompt after the first
user turn and a ``<total_tokens>`` reminder after every ``tool_result``. The
OpenAI Chat contract allows a system message anywhere, but two serving stacks
do not:

* DeepSeek V4 ends a tools+reasoning turn empty when the conversation ENDS on
  an instruction (:func:`fold_trailing_instruction_turns`, applied by origin
  and model-family recognition).
* The official Qwen3.6+ chat template (``chat_template.jinja``,
  ``{%- if message.role == "system" %}{%- if not loop.first %}
  raise_exception('System message must be at the beginning.')``) refuses ANY
  system message that is not the first message -- a second leading system
  turn included -- so a vLLM origin serving it answers HTTP 400 for the
  whole request (1,441 such rejections on the Experiential Cloud qwen3.8-27b
  rung in the seven days to 2026-09-15; the rejection is a lane limitation
  the ladder fails over, and a route with no other rung surfaces it).
  :func:`fold_instruction_turns_after_the_first` is applied on a rung that
  declares ``system_messages_leading_only``.

Both folds preserve position and text: an instruction run whose preceding
message is a text-only ``user`` turn is appended to that turn (blank-line
separated, the same order); otherwise each instruction is re-roled as one
``user`` message in place, so the provider still sees the text exactly where
the caller placed it. Every message that is not plain text is untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.models.model import ModelMessage
from exp.runtime.gateway.contracts import GatewayMessage

SYSTEM_FOLD_DISCLOSURE = "messages.system->folded(system_messages_leading_only)"
"""Disclosure recorded when a route may fold non-leading instruction turns.

Reported in ``ignored_parameters`` like the other message-shape coercions so
a caller can see that a rung on this route rewrote its system turns into the
user turn rather than rejecting or silently dropping them.
"""


def fold_trailing_instruction_turns[M: (GatewayMessage, ModelMessage)](
    messages: Sequence[M],
) -> tuple[M, ...]:
    """Move instruction turns that END a conversation into the user turn.

    DeepSeek V4 (verified live 2026-09-12 on the OpenRouter and Azure rungs,
    identical on the Chat and Messages surfaces): a request that carries
    ``tools`` and a reasoning control and whose LAST message is a ``system``
    (or ``developer``) turn ends the model's answer before any visible token
    about a quarter of the time -- an empty completion or a reasoning-only
    turn with ``finish_reason: stop``, billed but content-free (6 of 24 on the
    Messages surface, 5 of 24 on Chat, 0 of 22 with the instruction moved into
    the user turn). Claude Code produces exactly that shape: its Environment
    prompt and ``<total_tokens>`` reminder ride the mid-conversation-system
    beta as trailing ``system`` messages after the user turn or the last
    ``tool_result``.

    Instructions that are followed by a later user, tool, or assistant message
    are untouched (the leading system prompt included), as is every message
    that is not plain text.

    Args:
        messages: The ordered conversation for one provider request.

    Returns:
        The same messages with any trailing instruction run folded; the input
        tuple itself when there is nothing to fold.
    """
    ordered = tuple(messages)
    end = len(ordered)
    start = end
    while start > 0 and _is_plain_instruction(ordered[start - 1]):
        start -= 1
    if start == end or start == 0:
        # Nothing trails, or the whole conversation is instructions (a leading
        # system prompt with no user turn is the provider's own problem).
        return ordered
    return tuple(_fold_run(list(ordered[:start]), ordered[start:]))


def fold_instruction_turns_after_the_first[M: (GatewayMessage, ModelMessage)](
    messages: Sequence[M],
) -> tuple[M, ...]:
    """Leave exactly one leading instruction turn; fold every other into user text.

    For a rung whose chat template accepts a system message only as the very
    first message (``system_messages_leading_only``): a run of plain leading
    instructions is merged into ONE ``system`` turn (blank-line separated, the
    first turn's role and order kept), and every later plain instruction is
    folded the way :func:`fold_trailing_instruction_turns` folds a trailing
    one -- appended to the preceding text-only ``user`` turn, else re-roled as
    a ``user`` message in place (the Qwen template accepts consecutive user
    turns). A leading ``developer`` turn stays ``developer``: the Chat wire
    already emits it as ``system``.

    Args:
        messages: The ordered conversation for one provider request.

    Returns:
        The folded conversation; the input tuple itself when nothing changes.
    """
    ordered = tuple(messages)
    leading = 0
    while leading < len(ordered) and _is_plain_instruction(ordered[leading]):
        leading += 1
    head: list[M] = []
    if leading > 0:
        first = ordered[0]
        if leading == 1:
            head.append(first)
        else:
            merged = "\n\n".join(message.content or "" for message in ordered[:leading])
            head.append(
                first.model_copy(update={"content": merged, **_cleared_text_carriers(first)})
            )
    changed = leading > 1
    index = leading
    while index < len(ordered):
        message = ordered[index]
        if not _is_plain_instruction(message):
            head.append(message)
            index += 1
            continue
        run_end = index
        while run_end < len(ordered) and _is_plain_instruction(ordered[run_end]):
            run_end += 1
        head = _fold_run(head, ordered[index:run_end])
        changed = True
        index = run_end
    return tuple(head) if changed else ordered


def _fold_run[M: (GatewayMessage, ModelMessage)](
    head: list[M],
    instructions: Sequence[M],
) -> list[M]:
    """Fold one run of plain instructions onto ``head``, which must be non-empty."""
    previous = head[-1]
    texts = [message.content or "" for message in instructions]
    if _is_plain_user_text(previous):
        merged = "\n\n".join([previous.content or "", *texts])
        head[-1] = previous.model_copy(
            update={"content": merged, **_cleared_text_carriers(previous)}
        )
        return head
    head.extend(
        message.model_copy(update={"role": "user", **_cleared_text_carriers(message)})
        for message in instructions
    )
    return head


def _is_plain_instruction(message: GatewayMessage | ModelMessage) -> bool:
    """Whether a message is a text-only system/developer instruction."""
    if message.role not in {"system", "developer"} or message.content is None:
        return False
    if message.content_parts:
        return False
    if isinstance(message, GatewayMessage):
        return message.provider_native_item is None and message.provider_anthropic_block is None
    return True


def _is_plain_user_text(message: GatewayMessage | ModelMessage) -> bool:
    """Whether a message is a text-only user turn that can absorb an instruction."""
    if message.role != "user" or message.content is None:
        return False
    if isinstance(message, ModelMessage):
        return not message.content_parts
    return (
        message.provider_native_item is None
        and message.provider_anthropic_block is None
        and message.provider_anthropic_blocks is None
        and not message.content_parts
    )


def _cleared_text_carriers(message: GatewayMessage | ModelMessage) -> dict[str, object]:
    """Field overrides that drop block-structured carriers made stale by a text fold."""
    if isinstance(message, GatewayMessage):
        return {"provider_text_blocks": ()}
    return {}
