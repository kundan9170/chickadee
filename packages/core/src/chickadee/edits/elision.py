"""Elision: new text that stands a placeholder in for code it omits.

A model asked for a whole file, or a whole region, sometimes writes the part it changed and a
comment for the rest (`#... existing code...`). Applied, that comment replaces the code it
names, which is then gone. A placeholder is only an elision when it is new: a comment the old
text already holds is the file's own, so the check needs both texts and is not a parser's.

The draft calls `find_elision` for every replaced region (`edits/draft.py`); the one-shot
check over a parsed patch set retired with the packet loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from chickadee.edits.lines import split_lines

__all__ = ["PLACEHOLDER", "Elision", "find_elision", "is_placeholder"]

_MARKER: Final = r"(?:#+|//+|/\*+|<!--|--|;+|\*+)"
_ELLIPSIS: Final = r"(?:\.{3,}|…)"
_SUBJECT: Final = (
    r"(?:(?:the )?rest of (?:the )?(?:file|code|module|class|function|method|implementation)"
    r"|(?:existing|remaining|unchanged|previous|other|original)"
    r" (?:code|methods|functions|imports|logic|implementation)"
    r"|code)"
)
_STATE: Final = (
    r"(?:(?:is |are |remains? |stays? )?(?:unchanged|the same|as before|omitted|elided|here)"
    r"|(?:omitted|elided) for brevity|goes here)"
)
PLACEHOLDER: Final = re.compile(
    rf"^\s*{_MARKER}\s*\(?\s*(?P<lead>{_ELLIPSIS})?\s*{_SUBJECT}"
    rf"(?:\s*,?\s*(?P<state>{_STATE}))?\s*\)?\s*(?P<trail>{_ELLIPSIS})?\s*(?:\*/|-->)?\s*$",
    re.IGNORECASE,
)
"""A comment line that is only a placeholder for omitted code: what it names (`rest of the
file`, `existing code`) and nothing after it but an ellipsis or a state (`unchanged`,
`omitted`). One list: a new spelling is added here."""


def is_placeholder(line: str) -> bool:
    """Whether `line` stands for omitted code. The whole line must be the placeholder, and it
    must say so with an ellipsis or a state: `# rest of the function handles retries` and
    `# existing code paths call this` are comments about code, and refusing them would refuse
    a correct edit."""
    m = PLACEHOLDER.match(line)
    return m is not None and any(m.group(g) for g in ("lead", "state", "trail"))


@dataclass(frozen=True)
class Elision:
    line: int
    """1-based, in the new text."""
    text: str


def find_elision(new: str, old: str) -> Elision | None:
    """The first placeholder line in `new` that `old` does not already hold, or None."""
    held = {line.strip() for line in split_lines(old) if is_placeholder(line)}
    for n, line in enumerate(split_lines(new), start=1):
        if is_placeholder(line) and line.strip() not in held:
            return Elision(n, line.strip())
    return None
