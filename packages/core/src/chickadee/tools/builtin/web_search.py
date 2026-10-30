"""`web_search`, a built-in: results, never pages.

A thin run over the `web` port (`external/web.Web.search`): one source's API, `web` when none
is named (the person's engine, or the keyless trial), through the external-call policy; at
most `web.search_results` results,
each a title, a URL and a snippet, in an untrusted region. Every URL it shows may be fetched
next. The port sizes what it returns, so the answer is never cut again.
"""

from __future__ import annotations

from chickadee.external.search.engines import WEB
from chickadee.external.search.sources import official
from chickadee.schema.authorization import Tier
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Said, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="web_search",
    family="read",
    # One at a time: a round's `Web` admits each result's URL, unlocked.
    parallel="serial",
    mutates="nothing",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "what to look for, in the words a page about it would use",
            },
            "source": {
                "type": "string",
                "enum": [WEB, *(s.name for s in official())],
                "description": "where to search: `web` (the default) is the whole web; the "
                "others are one site's own search",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    description="researcher/web_search.md",
    bound=Bound("none"),
    failures=frozenset({INVALID_ARGUMENTS, "SEARCH_FAILED"}),
    effects="none",
    idempotent=True,
)


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    web = required(ctx.web, "web")
    got = web.search(args.text("query"), args.text_or_none("source"), call.id)
    return Said(got) if isinstance(got, str) else got


TOOL = Tool(SPEC, frozenset({"web"}), run)
