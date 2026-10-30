"""`doc`, a built-in: a document in the caller's part of the work folder.

Zeus writes `design/`, Athena `plan/`; `file` is named from that folder. Three operations, each
the edit tool's own over the `docs` port, so a document is written the one way a file is: the
call all or none, recorded as a step, snapshot and undo. `write` creates or replaces the whole
file (`write`'s), `edit` is an exact-text replace (`edit`'s), `remove` deletes it (`delete`'s).
Which part, and whether a work is open, is the port's (`work.py`). Only Markdown is written: a
design says what to build, never the code.
"""

from __future__ import annotations

from chickadee.context.results import InvalidArgument
from chickadee.edits.draft import DraftRefused
from chickadee.errors import RepoLeaseBusy
from chickadee.tools.anchors import answered, replace_text
from chickadee.tools.catalogue import args_sha
from chickadee.tools.context import DocsPort, ToolContext, required
from chickadee.tools.spec import SURFACE, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC: ToolSpec = next(s for s in SURFACE if s.name == "doc")


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    port = required(ctx.docs, "docs")
    op = args.text("op")
    missing = _missing(op, args)
    if missing is not None:
        return missing
    try:
        return _written(call, args, port, op)
    except RepoLeaseBusy as busy:
        # A run holds the repository (or opening the session's worktree found it held);
        # a write now would land between its packets, so the call is refused, never raised.
        return DraftRefused("DENIED", str(busy), do="call doc again once the run ends")


def _written(call: Invocation, args: Arguments, port: DocsPort, op: str) -> Answer:
    if op != "remove":
        # `work` reads the config, which reads this registry: imported here, not at the top.
        from chickadee.work import MARKDOWN_ONLY, markdown_error

        # Before the port resolves it: Zeus's first call opens the work, and a refused name
        # must open nothing. Removing is left open, so a file written in older versions can go.
        why = markdown_error(args.text("file"))
        if why is not None:
            return DraftRefused("DENIED", why, do=f"name a .md file: {MARKDOWN_ONLY}")
    path = port.resolve(args.text("file"), args.text_or_none("title"))
    if isinstance(path, DraftRefused):
        return path
    with port.edit(call, args_sha(args.sent)) as draft:
        if op == "write":
            refused = draft.apply(path, None, args.text("content"))
        elif op == "edit":
            refused = replace_text(
                draft,
                path,
                args.text("old"),
                args.text("new"),
                lines=False,
                replace_all=args.boolean("replace_all", False),
            )
        else:
            refused = draft.delete(path)
        answer = refused if refused is not None else answered(draft)
    return port.noted(answer)


def _missing(op: str, args: Arguments) -> InvalidArgument | None:
    """Each operation's own arguments, named when absent: the schema cannot say which goes with
    which `op` in the subset it validates."""
    needs = {"write": ("content",), "edit": ("old", "new"), "remove": ()}[op]
    absent = [n for n in needs if args.text_or_none(n) is None]
    if absent:
        return InvalidArgument(
            absent[0],
            f"{op} takes {' and '.join(absent)}, and they are missing",
            do=f"send doc again with {' and '.join(absent)}",
        )
    return None


TOOL = Tool(SPEC, frozenset({"docs"}), run)
