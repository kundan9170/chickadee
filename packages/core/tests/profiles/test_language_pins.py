"""Pins: which language each kind of path the tests write gets.

Before the move three tables answered it: the graph's own `EXTENSIONS`, the shipped profiles,
and the reader's `outline.LANGUAGES`, which labelled a preview and told a citation of a missing
file from a word with a dot in it. Now the profiles answer all of it, through
`ProfileSet.language_of`. Each answer that moved says what it was; T-3.164's Log says why.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pytest

from chickadee.context.citations import _names_a_file  # pyright: ignore[reportPrivateUsage]
from chickadee.context.outline import sniff
from chickadee.profiles import ProfileSet
from chickadee.secrets.denylist import Denylist


class LanguagePin(NamedTuple):
    language: str | None
    """The profile that reads the file, and what the code graph indexes it under; None is a
    gap."""
    grammar: str | None
    """The tree-sitter grammar that parses it."""
    kind: str
    """The kind a preview of it is labelled with."""
    cited: bool
    """Whether `name:12` with that bare name is a citation when no such file exists."""


PINS: dict[str, LanguagePin] = {
    "app.py": LanguagePin("python", "python", "python", True),
    "pkg/sub/mod.py": LanguagePin("python", "python", "python", True),
    "stubs/types.pyi": LanguagePin("python", "python", "python", True),
    "tests/test_app.py": LanguagePin("python", "python", "python", True),
    "src/a.js": LanguagePin("javascript", "javascript", "javascript", True),
    "src/esm.mjs": LanguagePin("javascript", "javascript", "javascript", True),
    "src/common.cjs": LanguagePin("javascript", "javascript", "javascript", True),
    "src/view.jsx": LanguagePin("javascript", "javascript", "javascript", True),
    # Moved twice. T-3.164: the graph read these as `typescript` and `tsx` though no profile
    # owned them, so they became a gap. T-3.165: the TypeScript profile owns them, each
    # extension with its own grammar.
    "src/a.ts": LanguagePin("typescript", "typescript", "typescript", True),
    "src/types.d.ts": LanguagePin("typescript", "typescript", "typescript", True),
    "src/view.tsx": LanguagePin("typescript", "tsx", "typescript", True),
    # Moved twice. T-3.164: a preview said `go`, `rust`, `java` and `c` for languages the
    # harness could not read, so they became text. T-3.165: each language's profile reads them,
    # and a preview names the profile (`include/a.h` is C's).
    "main.go": LanguagePin("go", "go", "go", True),
    "lib.rs": LanguagePin("rust", "rust", "rust", True),
    "Main.java": LanguagePin("java", "java", "java", True),
    "include/a.h": LanguagePin("c", "c", "c", True),
    # A language with no profile is still a gap and previewed as text; a missing `main.zig:12`
    # is not taken for a citation now that no hand-written suffix list names `.zig`.
    "main.zig": LanguagePin(None, None, "plain text", False),
    "README.md": LanguagePin(None, None, "markdown", True),
    "pyproject.toml": LanguagePin(None, None, "toml", True),
    "package.json": LanguagePin(None, None, "json", True),
    "notes.txt": LanguagePin(None, None, "plain text", True),
    "Makefile": LanguagePin(None, None, "plain text", False),
    ".gitignore": LanguagePin(None, None, "plain text", False),
}


@pytest.fixture(scope="module")
def shipped() -> ProfileSet:
    return ProfileSet.load()


@pytest.mark.parametrize(("path", "pin"), sorted(PINS.items()))
def test_language_of_a_path_is_pinned(shipped: ProfileSet, path: str, pin: LanguagePin) -> None:
    found = shipped.language_of(path)
    name = Path(path).name
    assert (
        LanguagePin(
            found.name if found is not None else None,
            found.grammar if found is not None else None,
            sniff(path, b"x = 1\n").kind,
            _names_a_file(name, lambda _p: None, Denylist()),
        )
        == pin
    )
