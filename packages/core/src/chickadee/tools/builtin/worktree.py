"""`worktree`, a built-in: the person asks, Zeus moves the work.

`enter` puts the session in a worktree of its own, the one it already holds or a new one;
`leave` brings it back to the person's checkout and keeps the worktree; `land` brings the
worktree's work into the checkout (`apply`, its commits as they are, or `squash`, one commit);
`discard` drops it. A thin run over the `worktree` port (`zeus/worktree.WorktreeMoves`), which
does every git step and asks the person before a landing, a discard, or a worktree their words
did not name.
"""

from __future__ import annotations

from chickadee.context.results import InvalidArgument
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "worktree")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    port = required(ctx.worktree, "worktree")
    action = args.text("action")
    how = args.text_or_none("how")
    if how is not None and action != "land":
        return InvalidArgument(
            "how", f"how is land's alone, and this is {action}", do="leave how out"
        )
    if action == "enter":
        return port.enter(operation_id=call.operation_id, cancel=ctx.cancel)
    if action == "leave":
        return port.leave()
    if action == "land":
        return port.land(how or "bring", operation_id=call.operation_id, cancel=ctx.cancel)
    return port.discard(operation_id=call.operation_id, cancel=ctx.cancel)


TOOL = Tool(SPEC, frozenset({"worktree"}), run)
