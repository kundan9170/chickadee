"""`finish`, a built-in (the owner's word): the worker ends its task with its report.

A model that ends its turns only by calling a tool looped on `shell("echo 'task completed'")`
instead of replying. This is the call that ends the task: its `report` is the worker's reply,
read as one would be (`tools/loop.Reply.final`). A reply with no tool call still ends it too.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="finish",
    family="talk",
    parallel="serial",
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "report": {
                "type": "string",
                "minLength": 1,
                "description": "what you changed and why, the commands you ran and what they "
                "printed, and anything you could not do or are unsure of",
            },
        },
        "required": ["report"],
        "additionalProperties": False,
    },
    description="worker/finish.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=True,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return as_text(SPEC, required(ctx.finish, "finish").finish(args.text("report")))


TOOL = Tool(SPEC, frozenset({"finish"}), run)
