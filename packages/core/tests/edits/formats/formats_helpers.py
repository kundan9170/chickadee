"""Reply-format test helpers. Unique basename.

Each format is driven through its own parser (`FORMATS[name].parse`): the packet rules every
format met after it, and the one-shot adapter that applied them, retired with the packet loop.
"""

from __future__ import annotations

from typing import Literal

from chickadee.edits.formats import FORMATS
from chickadee.edits.formats.base import Create, FileEdit, FileEdits, Hunks, Targets, Whole
from chickadee.edits.hunk import Hunk
from chickadee.edits.match import resolve
from chickadee.schema.adapter import ParseError

Mode = Literal["edit", "create", "delete"]
E: Mode = "edit"
C: Mode = "create"
D: Mode = "delete"

FILE = "def go() -> int:\n    return 1\n\n\ndef stop() -> None:\n    pass\n"


def targets(*named: tuple[str, Mode]) -> Targets:
    """The paths a parser is given, `app.py` alone by default: the writable ones in order, the
    deletes, and which are new files."""
    named = named or (("app.py", E),)
    return Targets(
        writable=tuple(p for p, m in named if m != D),
        deletes=tuple(p for p, m in named if m == D),
        creates=frozenset(p for p, m in named if m == C),
    )


def parse(fmt: str, text: str, given: Targets | None = None) -> FileEdits | ParseError:
    return FORMATS[fmt].parse(text, given or targets())


def hunks(path: str, *pairs: tuple[str, str]) -> tuple[str, FileEdit]:
    """A reply format's hunks on `path`, matched as whole lines."""
    return path, Hunks(tuple(Hunk(old=o, new=n, lines=True) for o, n in pairs))


def whole(path: str, content: str) -> tuple[str, FileEdit]:
    return path, Whole(content)


def create(path: str, content: str) -> tuple[str, FileEdit]:
    return path, Create(content)


def edits(*per_path: tuple[str, FileEdit]) -> FileEdits:
    return FileEdits(per_path)


def applied(result: FileEdits | ParseError, text: str = FILE) -> str:
    """The one path's hunks resolved against `text` by the one matcher."""
    assert isinstance(result, FileEdits), result
    ((_, found),) = result.edits
    assert isinstance(found, Hunks), found
    out = resolve(text, found.hunks)
    assert isinstance(out, str), out
    return out
