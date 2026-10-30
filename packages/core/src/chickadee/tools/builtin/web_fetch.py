"""`web_fetch`, a built-in: one web page as text.

A thin run over the `web` port (`external/web.Web`): the URL through `web_fetch`'s guards, the
GET through the external-call policy, the page's text in an untrusted region, paged by
`web.fetch_max_chars`. The port sizes what it returns, so the answer is never cut again: a
second cut would lose the region's close and the offset that reads on.
"""

from __future__ import annotations

from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="web_fetch",
    family="read",
    # One at a time: a round's `Web` remembers what appeared and what it fetched, unlocked.
    parallel="serial",
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "minLength": 1,
                "description": "a URL you were given, found in a search result or read on a "
                "page you fetched, exactly as it appeared",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "description": "where in the page's text to start, from a previous answer's "
                "`offset`; 0 when omitted",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    description="researcher/web_fetch.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS, "URL_REFUSED", "FETCH_FAILED"}),
    effects="none",
    idempotent=True,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    web = required(ctx.web, "web")
    got = web.fetch(args.text("url"), args.integer("offset", 0), call.id, ctx.cancel)
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"web"}), run)
