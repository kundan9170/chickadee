"""`tools`, a built-in.

The tools offered, by query (each one's brief) or by name (one full schema, as shown): 33 §3
row 8, §4 tier 3.

A thin run over the `catalog` port. What the call may touch is the port's to decide; whether it
may run, and what it costs, is the role's pipeline's.
"""

from __future__ import annotations

import json

from chickadee.tools.catalogue import InvalidCall
from chickadee.tools.context import ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, TextAnswer, Tool
from chickadee.tools.validate import Arguments, InvalidArguments, ToolBriefs, tools

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "tools")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    catalog = required(ctx.catalog, "catalog")
    query, name = args.text_or_none("query"), args.text_or_none("name")
    answer = tools(
        query=query,
        name=name,
        surface=catalog.offered(),
        descriptions=catalog.descriptions(),
        entry_cap=catalog.entry_cap,
    )
    if isinstance(answer, InvalidArguments):
        return InvalidCall(answer)
    if isinstance(answer, ToolBriefs):
        rows = [f"{b.name} — {b.line}" for b in answer.briefs]
        if answer.truncated:
            rows.append(
                f"[{answer.matched - len(answer.briefs)} more matched and are not shown; "
                "narrow the query]"
            )
        if not rows:
            rows = [f"no tool matches {query!r}"]
        return TextAnswer("\n".join(rows), "lines")
    if not answer:
        return TextAnswer(f"no tool is named {name!r}", "lines")
    # The schema a call must satisfy is the one the model is shown, the role's meta
    # arguments included.
    spec = next(s for s in catalog.offered() if s.name == answer["name"])
    shown = {**answer, "parameters": dict(catalog.shown_schema(spec))}
    return TextAnswer(json.dumps(shown, indent=1, sort_keys=True), "bytes")


TOOL = Tool(SPEC, frozenset({"catalog"}), run)
