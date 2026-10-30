"""`shell`, a built-in.

A command, under the role's authority and sandbox (`shell.py`, D93; the worker's with
D74's profile).

A thin run over the `exec` port. What the call may touch is the port's to decide; whether it may
run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

from chickadee.context.results import Interrupted
from chickadee.errors import InfraError
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "shell")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    shell = required(ctx.exec, "exec")
    try:
        return shell.run(args, operation_id=call.operation_id, cancel=ctx.cancel)
    except InfraError as exc:
        # A wrap or spawn that failed before anything ran, so the call did not run, and that
        # is known now. Raised, it would leave the call waiting and the next turn's repair
        # would call it unknown. `require_sandbox` with no backend is not this: the shell
        # answers it `SANDBOX_REQUIRED`.
        return Interrupted(
            SPEC.name,
            f"not run: {exc}",
            do="nothing ran, and the same command will fail the same way: do not send it "
            "again; tell the person, or go on without running it",
        )


TOOL = Tool(SPEC, frozenset({"exec"}), run)
