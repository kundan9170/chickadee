"""`edit`, a built-in: the worker's `edit`.

Exact text replaced in a target of the candidate's draft: `old` must occur exactly once, or
every occurrence is replaced under `replace_all`; several edits of one file go as an `edits`
list, applied in order as one call, every one or none, a `replace_all` beside it each one's
default. The one matcher finds each, and the draft's one write writes it (`tools/anchors.py`).
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered, replace_text
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import BLOCK, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

_EDIT: dict[str, object] = {
    "old": {
        "type": "string",
        "description": "the exact text to replace, as the file holds it",
        BLOCK: True,
    },
    "new": {"type": "string", "description": "the text that replaces it", BLOCK: True},
    "replace_all": {
        "type": "boolean",
        "description": "replace every occurrence; otherwise old must occur exactly once",
    },
}

SPEC = ToolSpec(
    name="edit",
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
            **_EDIT,
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": _EDIT,
                    "required": ["old", "new"],
                    "additionalProperties": False,
                },
                "minItems": 1,
                "description": (
                    "several edits of the file, applied in order as one call; replace_all "
                    "beside it is each edit's default"
                ),
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    description="worker/edit.md",
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
        }
    ),
    effects="dispatcher",
    idempotent=False,
    anchor="text",
    exactly_one=(("old", "new"), ("edits",)),
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    path = args.text("path")
    edits = args.objects("edits") or (args,)
    # Sent beside `edits`, it is each edit's default, never dropped.
    every = args.boolean("replace_all", False)
    with required(ctx.edits, "edits").edit() as draft:
        for i, edit in enumerate(edits):
            refused = replace_text(
                draft,
                path,
                edit.text("old"),
                edit.text("new"),
                lines=False,
                replace_all=edit.boolean("replace_all", every),
                label=f"edit {i}: " if len(edits) > 1 else "",
            )
            if refused is not None:
                return refused
        return answered(draft)


TOOL = Tool(SPEC, frozenset({"edits"}), run)
