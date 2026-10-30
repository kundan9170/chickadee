"""`replace_lines`, a built-in: the worker's `replace_lines`.

Lines `start` to `end` of a target in the candidate's draft replaced by `content`'s lines; an
empty `content` deletes them. Each anchor is written in the run's anchor style (`12`, or
`12:a3f`), whose schema `RolePolicy.resolved` gives its arguments, and resolved against the
draft as it now reads: a line that moved is `STALE_ANCHOR`, never written
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
    name="replace_lines",
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
            "start": {
                ANCHOR_ARG: "line",
                "description": "the first line replaced, its anchor as the file is shown",
            },
            "end": {
                ANCHOR_ARG: "line",
                "description": "the last line replaced, its anchor; start again for one line",
            },
            "content": {
                "type": "string",
                "description": "the lines that replace them; empty deletes them",
                BLOCK: True,
            },
        },
        "required": ["path", "start", "end", "content"],
        "additionalProperties": False,
    },
    description="worker/replace_lines.md",
    bound=Bound("none"),
    failures=frozenset(
        {
            "SYNTAX_FAIL",
            INVALID_ARGUMENTS,
            "DENIED",
            "STALE_ANCHOR",
            "FILE_ABSENT",
            "ELIDED",
            "BAD_FORMAT",
        }
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
            draft,
            path,
            (args.anchor("start"), "line"),
            args.anchor("end"),
            args.text("content"),
        )
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
