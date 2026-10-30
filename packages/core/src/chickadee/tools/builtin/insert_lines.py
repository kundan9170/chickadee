"""`insert_lines`, a built-in: the worker's `insert_lines`.

`content`'s lines inserted into a target of the candidate's draft after the line `after` names,
`0` being the file's start: an empty region, so nothing the file holds is replaced. The anchor
is written in the run's anchor style and resolved against the draft as it now reads
(`tools/anchors.replace_lines`).
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered, replace_lines
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import ANCHOR_ARG, BLOCK, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="insert_lines",
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
            "after": {
                ANCHOR_ARG: "after",
                "description": "the anchor of the line they follow; 0 is the file's start",
            },
            "content": {"type": "string", "description": "the lines to insert", BLOCK: True},
        },
        "required": ["path", "after", "content"],
        "additionalProperties": False,
    },
    description="worker/insert_lines.md",
    bound=Bound("none"),
    failures=frozenset(
        {"SYNTAX_FAIL", INVALID_ARGUMENTS, "DENIED", "STALE_ANCHOR", "FILE_ABSENT", "ELIDED"}
    ),
    effects="dispatcher",
    idempotent=False,
    anchor="line",
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    path = args.text("path")
    with required(ctx.edits, "edits").edit() as draft:
        refused = replace_lines(
            draft, path, (args.anchor("after"), "after"), None, args.text("content")
        )
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
