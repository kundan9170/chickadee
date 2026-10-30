"""`outgrown`, a built-in: Athena ends a one-task plan whose
finishing needs a decision nobody made, a different problem than the brief describes, or work
the brief does not cover.

A thin run over the `desk` port (`box/desk.TaskDesk.end_outgrown`): refused in a design's
episode and while a task is open; answered, the episode ends `OUTGROWN` when she replies, and
her reply is the report Zeus reads (what is missing, what is done, what she proposes, and which
of the three it outgrew). It takes no argument. The desk journals a `plan_outgrown` line,
once, which a resume reads.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="outgrown",
    family="task",
    parallel="serial",
    # The desk's `plan_outgrown` line is a journal record, not a change to any file.
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    description="athena/outgrown.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=True,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del args
    got = required(ctx.desk, "desk").end_outgrown(call_id=call.id)
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"desk"}), run)
