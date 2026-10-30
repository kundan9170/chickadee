"""What a session leaves behind: `HandoffSummary`, the answer of
`PacketEngine.handoff(session_id)`.

Types only, so the wire can carry them before the facade that fills them is built:
`handoff()` (`session_handoff.summarize`) selects every field here from what the
session's records and its worktree already hold. Nothing is computed in these types (invariant
19). The verification half, the gates' reports and checkpoints, retired with the gates: no task
run is verified, so `unverified` says what was not; `T-4.10` reads the
task loop's facts into it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from chickadee.budget.spend import Spend
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.handoff import HandoffPreview
from chickadee.schema.journal import Outcome

__all__ = ["HandoffSummary"]


@dataclass(frozen=True)
class HandoffSummary:
    """What a session leaves behind, and which completion choices are live right now. The
    git half is `HandoffPreview` carried whole, whose `refusal(choice)` `finalize()` itself
    asks, so a summary never offers a choice `finalize` then refuses."""

    session_id: str
    git: HandoffPreview | None
    """The worktree the session still writes, described; None when it has none: no `coder`
    call wrote one, it ran in place, or its last handoff applied, squashed or discarded it."""
    spend: Spend
    """The session's spend (08, session scope)."""
    episodes: tuple[str, ...]
    """Run ids, oldest first."""
    last_outcome: Outcome | None
    """The last episode's `run_end` outcome; None while it is live, or with no episode."""
    unverified: tuple[str, ...]
    """One line per claim the session could not make (invariant 7), every run-header
    line among them, so nothing said at the start is missing at the end. No task run is
    verified by a gate, so a run's behaviour is never claimed verified here."""
    workspace: WorkspaceRef | None = None
    """The `workspace` line `git` describes: its branch, base and handoff so far (`pending`,
    or `keep`)."""
    refused: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)
    """Each choice `git.refusal` refuses now, with why: what a host does not offer. The rest
    `finalize` carries out."""
    offers: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)
    """Each choice `finalize` carries out now, with what it does (`HandoffPreview.says`): the
    line a host shows beside it, so every frontend says the same."""
    diff: str = ""
    """`base..tip` as `git diff` prints it, cut at `handoff.DIFF_CAP` with a line saying so."""
    message: str = ""
    """In place: the commit message offered for the person's own commit
    of the session's work, each accepted task by its subject, goal and what ran. Nothing is
    committed for them. Empty with a worktree, whose commits carry their own."""
    notes: tuple[str, ...] = ()
    """What bears on the choice itself: an interrupted handoff not yet settled, work in the
    worktree no choice lands, a worktree that cannot be described. What a review shows the
    person; `unverified` is the record's (the run header's lines, behaviour not verified)."""
