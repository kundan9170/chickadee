"""`apply_patch`, a built-in: the worker's `apply_patch`.

A patch, Codex's V4A envelope or a unified diff, read by its format's own parser
(`edits/formats.parse_patch`) and applied to the candidate's draft as one call, every hunk and
file or none. Each hunk is matched as whole lines by the one matcher, within the scope its V4A
`@@` lines name; an `*** Add File` section creates a declared create target. Deletes and moves
are refused, as the parser refuses them.

Its spec carries the V4A grammar (`edits/formats/apply_patch.lark`). Decoded under it
(`ToolContext.constrained`), it takes V4A alone, the grammar's language; every patch is still
parsed and validated, since constrained output is not guaranteed to conform.
"""

from __future__ import annotations

from chickadee.edits.draft import DraftRefused
from chickadee.edits.formats import parse_patch
from chickadee.edits.formats.base import Create, Delete, Hunks, Move
from chickadee.schema.adapter import ParseError
from chickadee.schema.authorization import Tier
from chickadee.tools.anchors import answered, replace_text
from chickadee.tools.context import DraftCall, ToolContext, required
from chickadee.tools.spec import BLOCK, INVALID_ARGUMENTS, Bound, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

__all__ = ["SPEC", "TOOL"]

SPEC = ToolSpec(
    name="apply_patch",
    family="edit",
    parallel="serial",
    mutates="source",
    tier=Tier.T0,
    model_calls="none",
    schema={
        "type": "object",
        "properties": {
            "patch": {
                "type": "string",
                "description": "a V4A patch, or a unified diff",
                BLOCK: True,
            },
        },
        "required": ["patch"],
        "additionalProperties": False,
    },
    description="worker/apply_patch.md",
    bound=Bound("none"),
    failures=frozenset(
        {
            "SYNTAX_FAIL",
            INVALID_ARGUMENTS,
            "DENIED",
            "NO_MATCH",
            "AMBIGUOUS",
            "FILE_ABSENT",
            "ELIDED",
            "BAD_FORMAT",
        }
    ),
    effects="dispatcher",
    idempotent=False,
    anchor="text",
    grammar="apply_patch",
)


_FIX_FORMAT = "fix what the reason names in the patch and send it again"
"""A patch the format's parser refused: the hint names where, and the fix is in the patch."""


def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    del call
    with required(ctx.edits, "edits").edit() as draft:
        refused = _patched(draft, args.text("patch"), v4a_only=SPEC.name in ctx.constrained)
        return refused if refused is not None else answered(draft)


def _patched(draft: DraftCall, patch: str, *, v4a_only: bool) -> DraftRefused | None:
    parsed = parse_patch(patch, draft.targets, v4a_only=v4a_only)
    if isinstance(parsed, ParseError):
        return DraftRefused("BAD_FORMAT", parsed.hint, do=_FIX_FORMAT)
    if not parsed.edits:
        held = "no V4A section" if v4a_only else "no V4A section and no unified diff"
        return DraftRefused(
            "BAD_FORMAT",
            f"the patch holds {held}",
            do="send the patch with one section per file, in the form this tool's description "
            "shows",
        )
    for path, edit in parsed.edits:
        if isinstance(edit, Delete):
            refused = draft.delete(path)
            if refused is not None:
                return refused
            continue
        if isinstance(edit, Move):
            refused = draft.move(path, edit.target)
            if refused is None and edit.hunks is not None:
                refused = _hunks(draft, edit.target, edit.hunks)
            if refused is not None:
                return refused
            continue
        if isinstance(edit, Hunks):
            refused = _hunks(draft, path, edit)
            if refused is not None:
                return refused
            continue
        if isinstance(edit, Create) and isinstance(draft.text(path), str):
            # A create is a declared create target's, and only while it is absent.
            return DraftRefused(
                "DENIED",
                f"{path} exists, and a create is for an absent file",
                do="send its section as an update of the file instead",
            )
        refused = draft.apply(path, None, edit.content)
        if refused is not None:
            return refused
    return None


def _hunks(draft: DraftCall, path: str, edit: Hunks) -> DraftRefused | None:
    """An update section's hunks on `path`, each narrowed to its `@@` scopes, in order."""
    for i, hunk in enumerate(edit.hunks):
        refused = replace_text(
            draft,
            path,
            hunk.old,
            hunk.new,
            lines=hunk.lines,
            scopes=edit.scopes[i] if edit.scopes else (),
            label=f"hunk {i}: ",
        )
        if refused is not None:
            return refused
    return None


TOOL = Tool(SPEC, frozenset({"edits"}), run)
