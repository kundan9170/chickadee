"""`pause`, a built-in: Zeus suspends the session's run, which goes on in the
background once handed over. D215's suspend: the step in flight put back, no `run_end`, so
`coder` with its work continues it. A thin run over the `team` port."""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "pause")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call, args
    return required(ctx.team, "team").pause()


TOOL = Tool(SPEC, frozenset({"team"}), run)
