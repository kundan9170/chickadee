"""`find_skill`, a built-in.

The skills that best match what Zeus is looking for, ranked by BM25 over their names,
descriptions and keywords (`skills/search.py`), best first: the way to the right skill once the
resident index no longer lists them all. A thin run over the `skills` port.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "find_skill")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return as_text(SPEC, required(ctx.skills, "skills").find(args.text("query")))


TOOL = Tool(SPEC, frozenset({"skills"}), run)
