"""`delegate`, a built-in: one task, handed
to the worker.

A thin run over the `desk` port (`box/desk.TaskDesk.delegate`): with a work folder, a task
of the approved plan by its id, its brief the pinned task file; where the
work has no folder, a one-line title and the brief's markdown. The harness gives the task its
id, journals it, and runs the worker's first round; its answer is the round's report and the
facts measured. One task at a time: refused while one is open.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="delegate",
    family="task",
    parallel="serial",
    # The worker's writing calls are its own steps (`box/steps.StepRecord`), journaled by the
    # worker's dispatcher: this call records none of its own.
    mutates="nothing",
    tier=Tier.T0,
    model_calls="box",
    schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "a task of the approved plan, by its id: T1, T2, a piece T2.1",
                "minLength": 1,
            },
            "title": {
                "type": "string",
                "description": "only where the work has no folder: the task in one line",
                "minLength": 1,
            },
            "brief": {
                "type": "string",
                "description": "only where the work has no folder: the task's brief",
                "minLength": 1,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    description="athena/delegate.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=False,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    desk = required(ctx.desk, "desk")
    got = desk.delegate(
        args.text_or_none("title"),
        args.text_or_none("brief"),
        task=args.text_or_none("task"),
        call_id=call.id,
    )
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"desk"}), run)
