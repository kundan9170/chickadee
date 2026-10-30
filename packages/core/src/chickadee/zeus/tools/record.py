"""Family E — record: the `notes` port Zeus's `note_concern` and `note_rejected` are given.


Each appends one typed line to the session log. No model call, no effect, nothing read. The
tools are built-ins (`tools/builtin/`); the receipt they answer names what was recorded, and
the line itself is the record, which outlives the exchange that compaction collapses like any
other.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from chickadee.context.concerns import note_concern
from chickadee.context.ledger import note_rejected
from chickadee.context.results import ToolFailure

if TYPE_CHECKING:
    from chickadee.context.session import Session

__all__ = ["SessionNotes"]


class SessionNotes:
    """`NotesPort` over one session, every note by `zeus`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def note_concern(self, text: str, anchor: Sequence[str]) -> str | ToolFailure:
        return note_concern(self._session, text, anchor, "zeus")

    def note_rejected(self, text: str, reason: str, *, at: int) -> str | ToolFailure:
        """`at` is the assistant turn that made the call: where the rejection came from."""
        return note_rejected(self._session, text=text, reason=reason, by="zeus", at=at)
