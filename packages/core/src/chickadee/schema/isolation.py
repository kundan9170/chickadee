"""Contract 11 — IsolationSpec and GitHandoffChoice.

How a run is isolated and how the result is handed back. Every mode requires a lease,
because the repository-wide WRITE lease is what makes isolation real rather than advisory.
No mode permits a push.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

WIRE = ConfigDict(extra="forbid", frozen=True)

WorkspaceMode = Literal["worktree", "in_place"]
"""Where a run writes: a worktree of its own, or the person's checkout in place (contract 11).
The one name for the two modes every layer above passes around."""


class GitHandoffChoice(StrEnum):
    """What the user does with the run's worktree branch.

    `APPLY` and `SQUASH` are separate because they are different git operations, not two
    spellings of one. Each packet's commit is a rollback unit, and that property
    only survives `APPLY` — after `SQUASH` the packets are welded into a single commit and
    undoing one means undoing all of them. Collapsing them into a single `apply_squash`
    member would put that choice in the engine, where the user cannot reach it.
    """

    APPLY = "apply"
    """Replay the packet commits onto the source branch, history intact."""
    SQUASH = "squash"
    """Collapse the run into one commit on the source branch."""
    KEEP = "keep"
    """Leave the worktree branch in place for the user to handle."""
    DISCARD = "discard"
    """Delete the worktree branch. Never touches the source."""
    BRING = "bring"
    """Write the run's changes into the source checkout as uncommitted edits: no branch moves,
    nothing is staged or committed. Appended: every value above keeps its
    meaning."""


class IsolationSpec(BaseModel):
    model_config = WIRE

    mode: WorkspaceMode
    lease_id: str
    branch: str | None = None
    worktree_path: str | None = None
    pre_image_sha: str | None = None
    dirty_snapshot_sha: str | None = None

    @model_validator(mode="after")
    def _mode_requirements(self) -> IsolationSpec:
        if not self.lease_id:
            raise ValueError("every isolation mode requires a lease_id (D39)")
        if self.mode == "worktree":
            if not self.branch or not self.worktree_path:
                raise ValueError("mode=worktree requires branch and worktree_path")
        else:
            if not self.pre_image_sha or not self.dirty_snapshot_sha:
                raise ValueError(
                    "mode=in_place requires both snapshot shas — in-place must durably "
                    "snapshot pre-existing changes before it touches anything (D39)"
                )
        return self
