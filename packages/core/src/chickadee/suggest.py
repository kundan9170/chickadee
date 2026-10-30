"""The nearest known name to one that was not found: the one helper behind a
refusal or a report that offers "did you mean". A hint, never a correction: a caller
says what was asked for and offers this, and never acts on it.

`graph/names.py` is a different concept (qualified symbol names)."""

from __future__ import annotations

import difflib
from collections.abc import Iterable

__all__ = ["nearest"]


def nearest(name: str, known: Iterable[str], cutoff: float = 0.6) -> str | None:
    """The known name most like `name`, scored by `difflib`, or None when none reaches
    `cutoff` (0.6 is `difflib`'s own default). `known` is read in sorted order, so a tie
    resolves the same way every time."""
    close = difflib.get_close_matches(name, sorted(known), n=1, cutoff=cutoff)
    return close[0] if close else None
