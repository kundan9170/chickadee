"""`note_concern`, a built-in.

An advisory anchored to the paths it is about, one typed line.

A thin run over the `notes` port. What the call may touch is the port's to decide; whether it
may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "note_concern")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    notes = required(ctx.notes, "notes")
    got = notes.note_concern(args.text("text"), args.texts("anchor"))
    return Said(f"{got} recorded") if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"notes"}), run)
