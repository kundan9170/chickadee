"""`neighbors`, a built-in.

A symbol's or path's neighbourhood, `depth` hops out.

A thin run over the `navigator` port. What the call may touch is the port's to decide; whether
it may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "neighbors")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    navigator = required(ctx.navigator, "navigator")
    return as_text(SPEC, navigator.neighbors(args.text("target"), args.integer("depth", 1)))


TOOL = Tool(SPEC, frozenset({"navigator"}), run)
