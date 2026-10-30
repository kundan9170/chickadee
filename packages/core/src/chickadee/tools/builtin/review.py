"""`review`, a built-in: Athena's answer to
the open task's report.

A thin run over the `desk` port (`box/desk.TaskDesk.review`). The answer says what happens
next: `accept` commits the task; `feedback` is the worker's next message in
the same conversation; `continue` opens the next round of one held for a message from Zeus,
with no message; `take_over` makes the task hers; `re_cut` replaces it with the
ones she delegates next; `drop` gives it up. `fresh` says what stays (decision 4): on
`feedback`, `take_over` or `re_cut`, `true` puts every file back to the task's start and a list
only those; omitted, the changes are kept. `abandon`, `drop`'s name before, is answered as
renamed.
"""

from __future__ import annotations

from typing import cast

from chickadee.schema.authorization import Tier
from chickadee.schema.ids import TASK_KEY_RE
from chickadee.schema.journal import Fresh, ReviewAnswer
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

ANSWERS: tuple[ReviewAnswer, ...] = (
    "accept",
    "feedback",
    "continue",
    "take_over",
    "re_cut",
    "drop",
)
"""What she may answer: `abandon` is not offered, and a call that sends it is told its new name
(`ToolSpec.retyped`), as a renamed argument is."""

SPEC = ToolSpec(
    name="review",
    family="task",
    parallel="serial",
    # A commit and a restore are the desk's own effects (`task_commit`, `worker_step`), and a
    # feedback round's writes the worker's steps: this call records none of its own.
    mutates="nothing",
    tier=Tier.T0,
    model_calls="box",
    schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "the open task's id: T3, or a re-cut's piece T3.1",
                "pattern": TASK_KEY_RE.pattern,
            },
            "answer": {
                "type": "string",
                "enum": list(ANSWERS),
                "description": "accept, feedback, continue, take_over, re_cut or drop",
            },
            "text": {
                "type": "string",
                "description": (
                    "for feedback, the worker's next message; for any other answer, why"
                ),
            },
            "fresh": {
                "oneOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}, "minItems": 1},
                ],
                "description": (
                    "with feedback, take_over or re_cut: true puts every file back to the "
                    "task's start, a list of paths only those; leave it out to keep the changes"
                ),
            },
        },
        "required": ["task", "answer"],
        "additionalProperties": False,
    },
    description="athena/review.md",
    retyped={
        "answer": (
            "one of accept, feedback, continue, take_over, re_cut or drop; abandon is "
            "renamed: use drop"
        )
    },
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=False,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    desk = required(ctx.desk, "desk")
    answer = cast("ReviewAnswer", args.text("answer"))
    raw = args.sent.get("fresh")
    fresh: Fresh = (
        tuple(str(p) for p in cast("list[object]", raw))
        if isinstance(raw, list)
        else raw is True
    )
    got = desk.review(
        args.text("task"), answer, args.text_or_none("text"), call_id=call.id, fresh=fresh
    )
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"desk"}), run)
