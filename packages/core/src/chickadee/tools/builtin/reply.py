"""`reply`, a built-in: Athena's answer to a message from Zeus delivered to her
(`tell`), one per message, owed before her next `delegate` or `plan`.

A thin run over the `desk` port (`box/desk.TaskDesk.reply`), which journals `tell_reply` and
tells Zeus: `applied` (the task in hand covers it), `planned` (a task added for it),
`needs_design` (a change the approved design does not hold) or `declined` (not done, and why).
The last two wake him; the first two are read at his next turn.
"""

from __future__ import annotations

from typing import cast, get_args

from chickadee.schema.authorization import Tier
from chickadee.schema.journal import TELL_ID, TellOutcome
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="reply",
    family="task",
    parallel="serial",
    # The desk's `tell_reply` line is a journal record, not a change to any file.
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "tell": {
                "type": "string",
                "description": "the message's id: tell-1",
                "pattern": TELL_ID,
            },
            "outcome": {
                "type": "string",
                "enum": list(get_args(TellOutcome)),
                "description": "applied, planned, needs_design or declined",
            },
            "text": {
                "type": "string",
                "minLength": 1,
                "description": "what you did about it, or why not, in a sentence or two",
            },
        },
        "required": ["tell", "outcome", "text"],
        "additionalProperties": False,
    },
    description="athena/reply.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=True,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    got = required(ctx.desk, "desk").reply(
        args.text("tell"),
        cast("TellOutcome", args.text("outcome")),
        args.text("text"),
        call_id=call.id,
    )
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"desk"}), run)
