"""`request_exec`, a built-in: the scout's escape hatch, requested and never
chosen.

A thin run over the `exec_request` port, whose owner grants or refuses the command by a
deterministic policy, runs a granted one under DISCOVERY, and journals its own lifecycle
(`effects="self"`). The answer is what the model reads back, a refusal included.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INLINE_BYTES_DEFAULT, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, TextAnswer, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="request_exec",
    family="act",
    parallel="serial",
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "minLength": 1,
                "description": "one read-only command, run exactly as written or refused",
            },
            "reason": {
                "type": "string",
                "minLength": 1,
                "description": "why the structured tools could not answer this",
            },
        },
        "required": ["command", "reason"],
        "additionalProperties": False,
    },
    description="scout/request_exec.md",
    # The scout surface's byte cap, not the shell's: `limits.py` sizes the scout's largest
    # result by `zeus.inline_bytes`, and an escape's output is a result like any other.
    bound=Bound("inline_bytes", INLINE_BYTES_DEFAULT, None),
    failures=frozenset({INVALID_ARGUMENTS}),
    # A granted run journals its own `scout_escape` lifecycle (`Escapes`).
    effects="self",
    idempotent=False,
)
"""The typed request. `tier` is T0 because a granted run is sealed: DISCOVERY is
read-only and networkless, which is what T0 is — and an unsealed run is refused unless
the person allowed it, so the reason holds."""


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    escapes = required(ctx.exec_request, "exec_request")
    return TextAnswer(escapes.request(args.text("command"), args.text("reason")), "bytes")


TOOL = Tool(SPEC, frozenset({"exec_request"}), run)
