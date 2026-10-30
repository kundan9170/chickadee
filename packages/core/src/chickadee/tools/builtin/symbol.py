"""`symbol`, a built-in.

Everything about a symbol: definition, signature, doc, callers, callees.

A thin run over the `navigator` port. What the call may touch is the port's to decide; whether
it may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "symbol")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return as_text(SPEC, required(ctx.navigator, "navigator").symbol(args.text("name")))


TOOL = Tool(SPEC, frozenset({"navigator"}), run)
