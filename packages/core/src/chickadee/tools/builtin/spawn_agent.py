"""`spawn_agent`, a built-in.

One round of an agent, on the profile a skill names or the one Zeus names, given the task and
the handles he passes and nothing else; its report, its citations checked, is the answer. A thin
run over the `agents` port (`agents/spawn.py`), which resolves, refuses, runs and records.
"""

from __future__ import annotations

from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "spawn_agent")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    got = required(ctx.agents, "agents").spawn(
        args.text_or_none("skill"),
        args.text_or_none("profile"),
        args.text("task"),
        args.texts("handles"),
        call_id=call.id,
        cancel=ctx.cancel,
    )
    # The report is bounded by the agent's own limits, and cutting it would drop the
    # checked-citation header from what Zeus reads, so it is said whole.
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"agents"}), run)
