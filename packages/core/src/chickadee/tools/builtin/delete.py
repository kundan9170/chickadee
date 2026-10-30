"""`delete`, a built-in: a file removed, without asking.

Through the `edits` port, which proceeds only when undo can bring the file back (tracked, or
untracked, not ignored and within the snapshot's cap) and otherwise refuses, naming why and the
shell, which asks the person. A directory is the shell's.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="delete",
    paths=("path",),
    family="edit",
    parallel="serial",
    mutates="source",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "the file, from the repository root"},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    description="worker/delete.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS, "DENIED", "FILE_ABSENT"}),
    effects="dispatcher",
    idempotent=False,
    anchor="whole",
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    with required(ctx.edits, "edits").edit() as draft:
        refused = draft.delete(args.text("path"))
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
