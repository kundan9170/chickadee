"""`follow_up`, a built-in.

One more round of an agent `spawn_agent` started, its conversation kept: Zeus's message is its
next turn, and its new report, citations checked, is the answer. A thin run over the `agents`
port (`agents/spawn.py`), which refuses, runs and records.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "follow_up")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    got = required(ctx.agents, "agents").follow_up(
        args.text("agent"),
        args.text("message"),
        call_id=call.id,
        cancel=ctx.cancel,
    )
    # Said whole, as `spawn_agent`'s report is: bounded by the agent's own limits.
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"agents"}), run)
