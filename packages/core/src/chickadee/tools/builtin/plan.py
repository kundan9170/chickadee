"""`plan`, a built-in:
Athena's plan.

A thin run over the `desk` port (`box/desk.TaskDesk.plan`). With a work folder she writes
`plan/` with `doc` and calls `plan()`, which reads and checks it; where the work has no folder
(a one-task plan's re-cut) the plan is its `text`. Her first plan is asked at stop 2; a later
one is a revision, shown to the person and never asked.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="plan",
    family="task",
    parallel="serial",
    # The desk writes its own lines; the work happens in later tasks, each recorded there.
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "only where the work has no folder: the whole plan, in markdown; with a "
                    "folder, leave it out: plan/ is read"
                ),
                "minLength": 1,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    description="athena/plan.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=False,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    got = required(ctx.desk, "desk").plan(args.text_or_none("text"), call_id=call.id)
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"desk"}), run)
