"""`locate`, a built-in.

A bounded search agent on the cheap rung, for the thing that cannot be named.

A thin run over the `locator` port. What the call may touch is the port's to decide; whether it
may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "locate")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    locator = required(ctx.locator, "locator")
    answer = locator.locate(args.text("question"), cancel=ctx.cancel)
    return answer if not isinstance(answer, str) else as_text(SPEC, answer)


TOOL = Tool(SPEC, frozenset({"locator"}), run)
