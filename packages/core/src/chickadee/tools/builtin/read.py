"""`read`, a built-in.

Open a path or reuse a handle, through the role's `FileView`.

A thin run over the `files` port. What the call may touch is the port's to decide; whether it
may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "read")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    files = required(ctx.files, "files")
    # `validate` admits exactly one of the two (`SPEC.exactly_one`); the Reader takes either.
    path = args.text_or_none("path")
    target = path if path is not None else args.text("handle")
    return files.read(
        target,
        lines=args.text_or_none("lines"),
        at=args.text_or_none("at"),
        depth=args.text_or_none("depth"),
        max_bytes=args.integer("max_bytes", None),
        # The call it answers, so a box role's home can say which call showed which lines.
        call_id=call.id,
    )


TOOL = Tool(SPEC, frozenset({"files"}), run)
