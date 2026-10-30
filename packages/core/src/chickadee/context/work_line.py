"""The session's `work` line, the work folder's record on the session log.

Its own module, imported by `context/lines.PARSED` so the fold and the wire type it, and by
`work.py`, which writes and reads it: the work folder is `work.py`'s, and the line its record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import with_config

from chickadee.context.vocabulary import LINE, SessionKind

__all__ = ["WorkLine"]


@with_config(LINE)
@dataclass(frozen=True)
class WorkLine:
    """The session's work folder: opened by Zeus's first `doc` call, closed
    by the handoff. At most one is open: the latest line, while its state is `open`."""

    KIND: ClassVar[SessionKind] = "work"
    name: str
    """The folder's `<date>-<slug>`, under `dir`."""
    dir: str
    """`work.dir` as it was when the work opened, from the repository root."""
    state: Literal["open", "landed", "kept", "dropped"]
