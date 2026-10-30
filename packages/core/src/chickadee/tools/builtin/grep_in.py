"""`grep_in`, a built-in.

Search inside a file `read` already opened.

A thin run over the `files` port. What the call may touch is the port's to decide; whether it
may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "grep_in")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    files = required(ctx.files, "files")
    return files.grep_in(
        args.text("handle"), args.text("pattern"), context=args.integer("context", 2)
    )


TOOL = Tool(SPEC, frozenset({"files"}), run)
