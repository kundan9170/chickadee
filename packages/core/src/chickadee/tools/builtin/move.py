"""`move`, a built-in: a file to a new path, without asking.

Through the `edits` port: `path` is checked as `delete` checks it, `destination` must not
exist and is checked as any write is, and missing directories are made. The file keeps its
text and its mode.
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
    name="move",
    paths=("path", "destination"),
    family="edit",
    parallel="serial",
    mutates="source",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "the file, from the repository root"},
            "destination": {
                "type": "string",
                "description": "its new path, which must not exist",
            },
        },
        "required": ["path", "destination"],
        "additionalProperties": False,
    },
    description="worker/move.md",
    bound=Bound("none"),
    failures=frozenset({"SYNTAX_FAIL", INVALID_ARGUMENTS, "DENIED", "FILE_ABSENT"}),
    effects="dispatcher",
    idempotent=False,
    anchor="whole",
    # `path` names a file on every edit tool; `target` was also `read`'s handle.
    renamed={"source": "path", "target": "destination"},
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    with required(ctx.edits, "edits").edit() as draft:
        refused = draft.move(args.text("path"), args.text("destination"))
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
