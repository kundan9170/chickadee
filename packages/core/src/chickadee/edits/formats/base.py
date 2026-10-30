"""What a reply format is, and what it turns a reply into.

A format reads a one-shot model's final reply and says, per path the reply names, what the
model wrote: a whole file, a create, or ordered text hunks. It does not decide whether the
packet allows that: the packet rules are `adapters/base`'s, applied once for every format, so a
new format is a parser and a template and nothing else. It reads no file and calls no model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from chickadee.edits.hunk import Hunk
from chickadee.schema.adapter import ParseError

__all__ = [
    "Create",
    "FileEdit",
    "FileEdits",
    "Format",
    "Hunks",
    "Parse",
    "Targets",
    "Whole",
    "collected",
    "diff_hunk",
    "hunk",
    "whole_lines",
]


@dataclass(frozen=True)
class Targets:
    """The packet's paths, as a parser needs them to tell which path a block names."""

    writable: tuple[str, ...]
    """In the packet's target order; a delete target is never one."""
    deletes: tuple[str, ...] = ()
    creates: frozenset[str] = frozenset()
    """The writable paths that are new files, absent until written."""

    @property
    def named(self) -> frozenset[str]:
        """Every path the packet declares, writable or not."""
        return frozenset(self.writable) | frozenset(self.deletes)


@dataclass(frozen=True)
class Whole:
    """A file's new content from a format that writes whole files: an edit or a create."""

    content: str


@dataclass(frozen=True)
class Create:
    """A format's create form (an empty SEARCH, `--- /dev/null`, `*** Add File`): the new
    file's content, for a declared create target only."""

    content: str


@dataclass(frozen=True)
class Hunks:
    """Ordered exact-text hunks on an existing file."""

    hunks: tuple[Hunk, ...]
    scopes: tuple[tuple[str, ...], ...] = ()
    """The `@@` lines V4A names before each hunk, one tuple per hunk, outermost first; empty
    when the format names none. One-shot ignores them (a `Hunk` has no field for them); the
    worker's draft narrows each hunk's match to them."""


@dataclass(frozen=True)
class Delete:
    """A format's removal of a file (`*** Delete File`): the box's edit tool
    performs it when undo could bring the file back; the packet rules refuse it."""


@dataclass(frozen=True)
class Move:
    """A format's move of a file to `target` (`*** Move to`), with the hunks that
    edit it on the way, if any: a delete and a create in one edit call."""

    target: str
    hunks: Hunks | None = None


type FileEdit = Whole | Create | Hunks | Delete | Move


@dataclass(frozen=True)
class FileEdits:
    """What a reply wrote, per path it named, in the reply's order, each path once. Empty
    when the reply holds nothing in the format: it may be a tool request, or prose."""

    edits: tuple[tuple[str, FileEdit], ...] = ()


type Parse = Callable[[str, Targets], FileEdits | ParseError]


@dataclass(frozen=True)
class Format:
    """A reply format: its parser, its prompt and the words the packet rules use for it."""

    name: str
    parse: Parse
    template: str
    """Its prompt, under the adapter family's `templates/`."""
    markers: tuple[str, ...]
    """The lines that open its edits, which a call never holds."""
    create: str
    """How it writes a create, named when a create is refused or missing."""
    unit: str
    """What one of its edits is called: `no {unit} for` a delete target."""
    each: str
    """How a reply covers every target: the first half of the missing-target hint."""
    no_block: str
    """The hint for a reply with no edit and no tool request."""


def whole_lines(lines: Iterable[str]) -> str:
    """Lines as a file holds them, each ending its newline: a hunk's text, readable as the
    lines it names. Its matching is `Hunk.lines`'."""
    return "".join(f"{ln}\n" for ln in lines)


def hunk(old: Iterable[str], new: Iterable[str]) -> Hunk:
    """A hunk of whole lines, matched as lines: never part of a line, the file's last
    line with or without its final newline."""
    return Hunk(old=whole_lines(old), new=whole_lines(new), lines=True)


def diff_hunk(lines: list[str] | tuple[str, ...]) -> Hunk | None:
    """A diff-shaped hunk's lines (` ` context, `-` old, `+` new) as one hunk of whole lines:
    old is the context and `-` lines, new the context and `+` lines, each without its prefix.
    `\\ No newline at end of file` is dropped: a lines hunk keeps the file's final-newline
    state. Blank lines after the last are dropped too: inside a hunk a blank line is an empty
    context line, at its end it is spacing. None for a hunk with no lines."""
    kept = [ln for ln in lines if not ln.startswith("\\")]
    while kept and kept[-1] == "":
        kept.pop()
    if not kept:
        return None
    return hunk(
        (ln[1:] for ln in kept if not ln.startswith("+")),
        (ln[1:] for ln in kept if not ln.startswith("-")),
    )


def collected(found: Mapping[str, list[Hunk] | Hunks | Create | Delete | Move]) -> FileEdits:
    """What a hunk-writing parser gathered per path, in the reply's order."""
    kept = (
        e if isinstance(e, Create | Hunks | Delete | Move) else Hunks(tuple(e))
        for e in found.values()
    )
    return FileEdits(tuple(zip(found, kept, strict=True)))
