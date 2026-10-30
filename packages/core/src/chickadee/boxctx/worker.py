"""The worker's head: its fixed instructions, "done so
far" and the brief, the first messages of each task's conversation.

"Done so far" is the records of every task already committed in the request, oldest first. It is
history, so it stays true however far the tree has moved, and it is append-only, so each task's
head repeats the last one's bytes and the provider's cache matches them.

The packet worker's context (its packed context, its attempts' conversation and its compaction)
is parked as it stood: the task worker's conversation is
`conversation.BoxConversation`, which fits by eviction alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from chickadee.boxctx.records import TaskRecord, render_task_full, render_task_line
from chickadee.client.request import Message
from chickadee.context.request import est_text

__all__ = [
    "DONE_SO_FAR",
    "TASKS_OMITTED",
    "render_design_layer",
    "task_head",
]

DONE_SO_FAR: Final = "## DONE SO FAR"
"""The section's header. Its home since the renderer that named
section headers retired."""

TASKS_OMITTED: Final = (
    "[{k} earlier tasks in this request omitted — what they changed is in the files; "
    "read() and symbol() show it as it is now]"
)
"""What stands for the records "done so far" cannot hold: the worker has no
`recall()`, so the pointer names what it can call."""


def render_design_layer[R](
    records: Sequence[R],
    budget_tokens: int,
    *,
    full_of: Callable[[R], str],
    line_of: Callable[[R], str],
    omitted: str = TASKS_OMITTED,
) -> str:
    """The "done so far" layer: the records, oldest first, within `budget_tokens`. `full_of` and
    `line_of` render one record, whole and as one line.

    Over budget, the oldest records collapse to one line first — they are the least likely
    to matter to the task at hand — and if even one line each does not fit, the newest
    lines that do are kept under one line saying how many were left out and where their
    effects can be seen. No records render nothing (omit-empty)."""
    if not records:
        return ""
    lines = [full_of(r) for r in records]
    i = 0
    while _tokens(lines) > budget_tokens and i < len(records):
        lines[i] = line_of(records[i])
        i += 1
    if _tokens(lines) <= budget_tokens:
        return _section(lines)
    kept: list[str] = []
    for line in reversed(lines):
        left = len(records) - len(kept) - 1
        trial = [omitted.format(k=left), line, *kept] if left else [line, *kept]
        if _tokens(trial) > budget_tokens:
            break
        kept.insert(0, line)
    left = len(records) - len(kept)
    return _section([omitted.format(k=left), *kept] if left else kept)


def task_head(
    fixed: str, done: Sequence[TaskRecord], brief: str, budget_tokens: int
) -> tuple[Message, Message]:
    """The head of one task's conversation: the fixed instructions, the
    same bytes every task so the provider's cache holds them; then "done so far", each
    committed task's record oldest first, collapsing to one line each over `budget_tokens`
    (`render_design_layer`'s rule); then the brief."""
    layer = render_design_layer(
        done,
        budget_tokens,
        full_of=render_task_full,
        line_of=render_task_line,
        omitted=TASKS_OMITTED,
    )
    user = f"{layer}\n\n{brief}" if layer else brief
    return Message(role="system", content=fixed), Message(role="user", content=user)


def _section(lines: Sequence[str]) -> str:
    return "\n".join([DONE_SO_FAR, *lines])


def _tokens(lines: Sequence[str]) -> int:
    return est_text(_section(lines))
