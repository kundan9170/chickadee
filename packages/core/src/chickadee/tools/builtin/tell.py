"""`tell`, a built-in: Zeus's message to the team about the session's live or
suspended run, in his words, the person's turns it came from its authority. The
run delivers it at its next boundary. A thin run over the `team` port."""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "tell")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return required(ctx.team, "team").tell(args.text("text"), args.integers("from"))


TOOL = Tool(SPEC, frozenset({"team"}), run)
