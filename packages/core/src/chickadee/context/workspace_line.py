"""The session's `workspace` line: which workspace a turn ran in, the checkout or the
session's worktree, and how far its handoff went.

Its own module, as `context/work_line.py` is: `context/lines.PARSED` types it for the fold and
the wire, and everything that reads where the session's work is imports it from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import Field, with_config

from chickadee.context.vocabulary import LINE, SessionKind
from chickadee.schema.isolation import WorkspaceMode

__all__ = ["WorkspaceRef"]


@with_config(LINE)
@dataclass(frozen=True)
class WorkspaceRef:
    """The durable pointer a turn keeps to the workspace it ran in.

    A reference, deliberately: the worktree may be gone by the time anyone reads the turn
    back, and recording ids rather than a live object is what lets a resumed session say so
    instead of failing to open something it assumed was there."""

    KIND: ClassVar[SessionKind] = "workspace"
    mode: WorkspaceMode
    lease_id: str
    source_ref: str
    source_head: str
    branch: Annotated[str | None, Field(default=None)]
    """`in_place` has no branch of its own."""
    worktree_path: Annotated[str | None, Field(default=None)]
    """None once discarded — absence is a fact, not an error."""
    handoff: Literal["pending", "apply", "squash", "keep", "discard", "bring"]
