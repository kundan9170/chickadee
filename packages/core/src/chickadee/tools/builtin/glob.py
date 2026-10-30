"""`glob`, a built-in: the file paths matching a glob, through the `search` port, which
reads names and never contents, and leaves out what the deny list names.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "glob")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    search = required(ctx.search, "search")
    return as_text(SPEC, search(args.text("pattern"), kind="path", cancel=ctx.cancel))


TOOL = Tool(SPEC, frozenset({"search"}), run)
