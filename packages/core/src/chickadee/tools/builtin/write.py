"""`write`, a built-in: the worker's `write`.

A target's whole content, written to the draft through the `edits` port: an
existing target replaced, a declared create target created. The draft's one write checks the
path, and refuses a placeholder standing in for code the file holds.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import BLOCK, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="write",
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
            "content": {
                "type": "string",
                "description": "the file's whole new content",
                BLOCK: True,
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    description="worker/write.md",
    # Cut by the worker's pipeline (`Clip`: `worker.result_max`), not by the draft.
    bound=Bound("none"),
    failures=frozenset({"SYNTAX_FAIL", INVALID_ARGUMENTS, "DENIED", "ELIDED"}),
    effects="dispatcher",
    idempotent=True,
    anchor="whole",
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    with required(ctx.edits, "edits").edit() as draft:
        refused = draft.apply(args.text("path"), None, args.text("content"))
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
