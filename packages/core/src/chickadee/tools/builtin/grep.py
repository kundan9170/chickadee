"""`grep`, a built-in: a regular expression over every file the repository does not
ignore, configuration and documentation included, through the `search` port.

The port's owner (`scout/search.Search`) applies the deny list to every hit and bounds
what it shows; this run only names what the model sent.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "grep")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    search = required(ctx.search, "search")
    found = search(
        args.text("pattern"), kind="text", paths=args.texts("paths"), cancel=ctx.cancel
    )
    return as_text(SPEC, found)


TOOL = Tool(SPEC, frozenset({"search"}), run)
