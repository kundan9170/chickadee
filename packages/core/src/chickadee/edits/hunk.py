"""One search/replace hunk and why one did not match: the edit tools' units of a change.

Moved from `schema/patch.py` when the Applier and the patch contract retired:
the edit tools and their reply formats are the hunk's one remaining home. Of the Applier's apply
reasons only the match failures survive, since the draft is now the only thing that matches.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = ["Hunk", "MatchReason"]


class MatchReason(StrEnum):
    """Why `old` did not resolve to one place in the text (`edits/match.py`)."""

    NO_MATCH = "no_match"
    AMBIGUOUS = "ambiguous"


class Hunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    old: str
    new: str
    lines: bool = False
    """Match `old` as whole lines, never part of one, the file's last line with or without
    its final newline. A reply format's hunks set it; an edit tool's literal text, which may be
    a fragment of a line, leaves it unset."""
