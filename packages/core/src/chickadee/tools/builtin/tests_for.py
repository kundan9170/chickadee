"""`tests_for`, a built-in: the test files that depend on the given files.

A thin run over the `navigator` port (`GraphNavigator.tests_for`): within three import hops,
backwards, from the graph, with the command that runs them, and what the graph cannot see said
every time. "Run these first", never proof that nothing else broke.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INLINE_BYTES_DEFAULT, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool, as_text
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="tests_for",
    paths=("paths",),
    family="navigate",
    parallel="any",
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "the changed files, from the repository root",
            },
        },
        "required": ["paths"],
        "additionalProperties": False,
    },
    description="worker/tests_for.md",
    bound=Bound("inline_bytes", INLINE_BYTES_DEFAULT, None),
    failures=frozenset({INVALID_ARGUMENTS, "INDEX_UNAVAILABLE"}),
    effects="none",
    idempotent=True,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    return as_text(SPEC, required(ctx.navigator, "navigator").tests_for(args.texts("paths")))


TOOL = Tool(SPEC, frozenset({"navigator"}), run)
