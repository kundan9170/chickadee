"""`impact`, a built-in.

What depends on the symbols that would change.

A thin run over the `navigator` port. What the call may touch is the port's to decide; whether
it may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "impact")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return as_text(SPEC, required(ctx.navigator, "navigator").impact(args.texts("symbols")))


TOOL = Tool(SPEC, frozenset({"navigator"}), run)
