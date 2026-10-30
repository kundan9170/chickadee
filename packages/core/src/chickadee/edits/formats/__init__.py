"""Reply formats: `FORMATS`, the one registry of how a one-shot model writes its
edits. A format is a module here, its entry below and its template under the adapter family's
`templates/formats/`; the packet rules, the parse order and the load rule need no change.

A format whose edits a tool may be decoded under keeps its grammar beside its parser
(`apply_patch.lark`), read by `grammar`; guard row E6 holds every grammar here.
"""

from __future__ import annotations

from importlib import resources

from chickadee.edits.formats import apply_patch, search_replace, unified_diff, whole_file
from chickadee.edits.formats.base import FileEdits, Format, Targets
from chickadee.schema.adapter import ParseError

__all__ = ["DEFAULT", "FORMATS", "grammar", "parse_patch"]

FORMATS: dict[str, Format] = {
    f.name: f
    for f in (whole_file.FORMAT, search_replace.FORMAT, unified_diff.FORMAT, apply_patch.FORMAT)
}

DEFAULT = whole_file.FORMAT
"""A worker model's format when it names none and is offered no edit tool, and
Athena's repair's, by name."""


def parse_patch(
    text: str, targets: Targets, *, v4a_only: bool = False
) -> FileEdits | ParseError:
    """A patch the worker's `apply_patch` tool was given: a V4A envelope, else a
    unified diff, each read by its own format's parser. Empty when it holds neither. With
    `v4a_only`, the tool was decoded under the V4A grammar: a unified diff is not
    its language, and is not read."""
    v4a = apply_patch.parse(text, targets)
    if isinstance(v4a, ParseError) or v4a.edits or v4a_only:
        return v4a
    return unified_diff.parse(text, targets)


def grammar(name: str) -> str:
    """The Lark grammar format `name` keeps beside its parser (`<name>.lark`): what a
    tool whose spec names it is decoded under."""
    if name not in FORMATS:
        raise KeyError(f"no format {name!r} keeps a grammar")
    return resources.files(__package__).joinpath(f"{name}.lark").read_text(encoding="utf-8")
