"""`note`, a built-in: Athena's note on the design she is building.

A thin run over the `design_notes` port (`boxctx.athena.note`): one `note` line on the session
log, durable before it is shown, kept for the design's life and through her compaction.
Over `athena.note_max` tokens is the port's refusal.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="note",
    family="record",
    parallel="any",
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "what to keep for the rest of this design, in your own words",
                "minLength": 1,
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    description="athena/note.md",
    bound=Bound("one_line"),
    failures=frozenset({INVALID_ARGUMENTS}),
    effects="none",
    idempotent=False,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    got = required(ctx.design_notes, "design_notes").note(args.text("text"))
    return Said(f"{got} kept") if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"design_notes"}), run)
