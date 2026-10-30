"""`recall`, a built-in.

Pointers into prior designs, runs and outcomes, never their bodies.

A thin run over the `journal` port. What the call may touch is the port's to decide; whether it
may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "recall")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return required(ctx.journal, "journal").recall(args.text("query"))


TOOL = Tool(SPEC, frozenset({"journal"}), run)
