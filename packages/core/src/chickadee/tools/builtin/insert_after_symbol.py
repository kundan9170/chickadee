"""`insert_after_symbol`, a built-in: the worker's `insert_after_symbol`.

`content`'s lines inserted into a target of the candidate's draft after a named declaration's
last line: an empty region, so nothing the file holds is replaced. The name is resolved on the
draft as it now reads (`edits/symbols`, through `tools/anchors.replace_symbol`); a file no
parser reads is `UNSUPPORTED`.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered, replace_symbol
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import BLOCK, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="insert_after_symbol",
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
            "symbol": {
                "type": "string",
                "minLength": 1,
                "description": "the declaration to insert after, qualified if not unique",
            },
            "content": {
                "type": "string",
                "description": "the lines to insert after the declaration",
                BLOCK: True,
            },
        },
        "required": ["path", "symbol", "content"],
        "additionalProperties": False,
    },
    description="worker/insert_after_symbol.md",
    bound=Bound("none"),
    failures=frozenset(
        {
            "SYNTAX_FAIL",
            INVALID_ARGUMENTS,
            "DENIED",
            "NO_MATCH",
            "AMBIGUOUS",
            "FILE_ABSENT",
            "ELIDED",
            "UNSUPPORTED",
        }
    ),
    effects="dispatcher",
    idempotent=False,
    anchor="symbol",
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    with required(ctx.edits, "edits").edit() as draft:
        refused = replace_symbol(
            draft, args.text("path"), args.text("symbol"), args.text("content"), after=True
        )
        return refused if refused is not None else answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
