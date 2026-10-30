"""`create`, a built-in: a new file's whole content.

Through the `edits` port, the one write path: the path is checked, and an existing file is
refused, so a create never replaces what is there. A new file may be empty.
"""

from __future__ import annotations

from chickadee.edits.draft import DraftRefused
from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import BLOCK, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="create",
    paths=("path",),
    family="edit",
    parallel="serial",
    mutates="source",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "the new file, from the repository root"},
            "content": {"type": "string", "description": "its whole content", BLOCK: True},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    description="worker/create.md",
    bound=Bound("none"),
    failures=frozenset({"SYNTAX_FAIL", INVALID_ARGUMENTS, "DENIED", "ELIDED"}),
    effects="dispatcher",
    idempotent=False,
    anchor="whole",
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    path = args.text("path")
    with required(ctx.edits, "edits").edit() as draft:
        if not isinstance(draft.text(path), DraftRefused):
            return DraftRefused(
                "DENIED",
                f"{path} already exists",
                do="edit it, giving old and new; to replace it all, delete it first",
            )
        refused = draft.apply(path, None, args.text("content"))
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
