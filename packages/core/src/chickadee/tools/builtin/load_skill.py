"""`load_skill`, a built-in.

A skill's body, whole and pinned for the session, or a typed refusal that loaded nothing. A
thin run over the `skills` port, which records the load against this call: the tool line that
carries the answer is the one a cut lifts while the skill is pinned. The answer is never
bounded again (`Said`): half a skill reads as the whole of it.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "load_skill")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    got = required(ctx.skills, "skills").load(args.text("name"), call.id)
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"skills"}), run)
