"""What a rewind is.

A rewind is a line, never a deletion. An `undo` with `conversation` hides, from the fold, the
lines from its `to_turn` up to itself whose kind `REWIND` says is the conversation; a `redo`
reverses it. `SessionState` re-folds its own records when either is applied, reading
`hidden_seqs`, so every reader of the fold — assembly, compaction, recall, citations — sees
only what is visible, and none can bring a hidden line back.

Also here, each the one home of its rule: the safe point a fork or a rewind is written at
(`safe_point_refusal`), the checks an `undo` and a `redo` must pass, the `code_snapshot` line
the fork and the undo read (written by `zeus/snapshots.py`), and `record_rewind`, which
appends an `undo` and the kept-code notice that goes with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Literal

from pydantic import Field, TypeAdapter, with_config

from chickadee.context import standing
from chickadee.context.lines import RedoLine, UndoLine
from chickadee.context.vocabulary import LINE, SESSION_KINDS, PayloadError, SessionKind
from chickadee.journal.errors import JournalError
from chickadee.schema.isolation import WorkspaceMode

if TYPE_CHECKING:
    from chickadee.context.project import RunRecords
    from chickadee.context.session import Session, SessionState
    from chickadee.context.store import SessionEvent

__all__ = [
    "CODE_SNAPSHOT",
    "REWIND",
    "CodeSnapshotLine",
    "ForkRefused",
    "Rewound",
    "arrive",
    "check_redo",
    "check_undo",
    "hidden_seqs",
    "moved_on",
    "record_rewind",
    "safe_point_refusal",
    "taken",
]

Rewound = Literal["hidden", "kept"]

REWIND: Final[Mapping[SessionKind, Rewound]] = {
    # The conversation: what the model read, said, and was told.
    "turn": "hidden",
    "tool": "hidden",
    "constraint": "hidden",
    "rejected": "hidden",
    "concern": "hidden",
    "note": "hidden",
    "summary": "hidden",
    "compaction": "hidden",
    "scout": "hidden",
    # An agent is its spawning turn's, as a `locate` is: rewind past the turn and the
    # agent is not in context either. What it spent is a `spend` line, and stays.
    "agent": "hidden",
    "bound": "hidden",
    "frame": "hidden",
    "packet_record": "hidden",
    "design_record": "hidden",
    # A load goes with the tool line it pins: rewind past it and the skill is not in
    # context, so it is not pinned either.
    "skill_loaded": "hidden",
    # What happened whatever is rewound: money spent stays spent, a cap cannot be reset by
    # rewinding, an episode's run happened, and the rewinds themselves are the record.
    "spend": "kept",
    "cap": "kept",
    # A switch and a pin are the person's commands, like a cap: taking turns back leaves the
    # rung where they moved it.
    "model_switched": "kept",
    "title": "kept",  # D239: the session's name, the person's or the title role's, not a turn
    "pinned": "kept",
    "limits": "kept",
    "episode": "kept",
    "shell_effect": "kept",  # a declared write happened, whatever is rewound
    "box_call": "kept",  # retired: an older log's record of a call the box made
    "plan": "kept",  # D186: a fact about the design, not about the turn that asked for it
    "box_compaction": "kept",  # retired: an older log's trace
    "workspace": "kept",
    "work": "kept",  # D204: a folder was opened or closed, whatever is rewound
    "prefix": "kept",
    "fork": "kept",
    "code_snapshot": "kept",
    "undo": "kept",
    "redo": "kept",
    # A person was asked and answered whatever is taken back later (who decided is
    # provenance), and hiding an asked line would leave its kept answer naming nothing.
    "question_asked": "kept",
    "question_answered": "kept",
    "plan_decision": "kept",  # retired: an older log's decision at a design's gate
    # The person's unload is their act, like a cap: undoing turns never re-pins what they let
    # go.
    "skill_unloaded": "kept",
    # Untrusted text crossed into context: it was read, whatever is rewound after.
    "untrusted_read": "kept",
    # A handoff changed the person's repository: undoing turns does not unsay it.
    "handoff_intended": "kept",
    "handoff_completed": "kept",
    # A grant is the person's decision about the session, like a cap: taking turns back does
    # not take back what they allowed for it. A fork still starts with none.
    "command_grant": "kept",
    "action_granted": "kept",  # an action ran under a grant, whatever is rewound
    "hook": "kept",  # a hook ran and decided, whatever is rewound
    "external_call": "kept",  # a request left the machine, whatever is rewound
    # A tell reached the team, or waits for its next boundary: the run acted on it
    # whatever is rewound, and its reply names it.
    "tell": "kept",
}
"""Every `SessionKind`, placed. A kind added later without a place fails a test."""

assert set(REWIND) == set(SESSION_KINDS), "every session kind is hidden or kept by a rewind"


def hidden_seqs(lines: Sequence[tuple[SessionEvent, object]]) -> frozenset[int]:
    """The seqs a rewind hides: for each `undo` with `conversation` that no `redo` reverses,
    every line from its `to_turn` up to it whose kind is hidden. `lines` is every folded line,
    hidden or not, indexed by seq from 1."""
    redone = {r.undo_seq for _, r in lines if isinstance(r, RedoLine) and r.conversation}
    hidden: set[int] = set()
    for _, undo in lines:
        if isinstance(undo, UndoLine) and undo.conversation and undo.seq not in redone:
            hidden.update(
                e.seq
                for e, _ in lines[undo.to_turn - 1 : undo.seq - 1]
                if REWIND[e.kind] == "hidden"
            )
    return frozenset(hidden)


# -- the safe point ---------------------------------------------------------------------------


class ForkRefused(ValueError):
    """The fork point is not a safe point (§5.1.1 PRECOND). Nothing was created."""

    def __init__(self, parent_id: str, at_seq: int, why: str) -> None:
        self.parent_id = parent_id
        self.at_seq = at_seq
        self.why = why
        super().__init__(f"cannot fork {parent_id} at seq {at_seq}: {why}")


def safe_point_refusal(state: SessionState, records: RunRecords | None = None) -> str | None:
    """Why `state`'s last seq is not a SAFE POINT, or None (§5.1.1 PRECOND): a call still
    waiting for its result, or a run still executing. A fork there would hand the child a
    run executing in the parent; an undo or a redo would take back a turn mid-exchange.

    A run's end is read from its journal when `records` is given: a readable journal with no
    `run_end` is a run in flight. A journal that does not exist is not a live run — a run
    dispatched but not yet journaled still has its `coder` call waiting, which the first
    check catches. One that exists and will not read says nothing either way, so it is
    refused: a run that cannot be shown to have ended is not a safe point (invariant 20),
    and the refusal is typed where the journal's own error would escape as itself."""
    pending = state.pending_calls()
    if pending:
        return f"call {pending[0]} is still waiting for its result"
    jobs = state.jobs.ids()
    if jobs:
        # A fork, an undo or a redo would rewrite the tree under it.
        return f"job {jobs[0]} is still running; wait for it or kill it"
    if records is None:
        return None
    for episode in state.episodes():
        try:
            run = records.projection(episode.run_id)
        except FileNotFoundError:
            continue
        except (JournalError, OSError) as exc:
            return f"run {episode.run_id}'s journal did not read ({type(exc).__name__})"
        if run.end is None:
            # Its way out, now that the team may work while the person talks. A pause
            # is no way out: a paused run has no `run_end` either, and is refused the same.
            return (
                f"run {episode.run_id} is still in flight or paused; let it end, or the person "
                "ends it with /drop"
            )
    return None


# -- the fold's checks ------------------------------------------------------------------------


def check_undo(state: SessionState, seq: int, line: UndoLine) -> UndoLine:
    """An undo takes back something, from a user turn the person can still see, at a safe
    point."""
    undo = replace(line, seq=seq)
    if not (undo.code or undo.conversation):
        raise PayloadError("an undo takes back the code, the conversation, or both")
    turn = state.turn(undo.to_turn)
    if turn is None or turn.role != "user":
        raise PayloadError(f"seq {undo.to_turn} is not a user turn the person can see")
    why = safe_point_refusal(state)
    if why is not None:
        raise PayloadError(f"an undo is written at a safe point: {why}")
    return undo


def check_redo(state: SessionState, seq: int, line: RedoLine) -> RedoLine:
    """A redo reverses one undo's code, its conversation, or both, each half once and only a
    half the undo took. A conversation redo is offered only until the conversation moves on;
    the code half stays offered while its snapshot exists."""
    redo = replace(line, seq=seq)
    held = state.line(redo.undo_seq)
    undo = held[1] if held is not None else None
    if not isinstance(undo, UndoLine):
        raise PayloadError(f"seq {redo.undo_seq} is not an undo")
    if not (redo.code or redo.conversation):
        raise PayloadError("a redo takes back the code, the conversation, or both")
    earlier = [
        r for _, r in state.lines() if isinstance(r, RedoLine) and r.undo_seq == undo.seq
    ]
    for half, asked, taken in (
        ("code", redo.code, undo.code),
        ("conversation", redo.conversation, undo.conversation),
    ):
        if asked and not taken:
            raise PayloadError(f"the undo at seq {undo.seq} did not take back the {half}")
        if asked and any(getattr(r, half) for r in earlier):
            raise PayloadError(f"the undo at seq {undo.seq} already has its {half} redone")
    if redo.conversation:
        later = moved_on(state, undo.seq)
        if later is not None:
            raise PayloadError(
                f"seq {later} was written after the undo at seq {undo.seq}; its conversation "
                "can no longer be redone"
            )
    why = safe_point_refusal(state)
    if why is not None:
        raise PayloadError(f"a redo is written at a safe point: {why}")
    return redo


def moved_on(state: SessionState, seq: int) -> int | None:
    """The first conversation line after `seq`, hidden by a later rewind or not, other than a
    harness notice: once one is written the conversation has moved on, and
    the lines a redo would bring back could clash with it (a call id, a constraint id)."""
    from chickadee.context.session import Turn  # session imports this module

    for event, record in state.lines(seq + 1, with_hidden=True):
        if REWIND[event.kind] == "hidden" and not (
            isinstance(record, Turn) and record.role == "harness"
        ):
            return event.seq
    return None


# -- the turn snapshots -----------------------------------------------------------------------

CODE_SNAPSHOT: Final[SessionKind] = "code_snapshot"


@with_config(LINE)
@dataclass(frozen=True)
class CodeSnapshotLine:
    """Payload of a `code_snapshot` line: one
    phase of one workspace's snapshot in one turn, parked under `ref`. Written by
    `zeus/snapshots.py`, and kept unread by the context fold; read here, by the fork and the
    undo, through `taken`."""

    KIND: ClassVar[SessionKind] = CODE_SNAPSHOT
    turn: int
    """The seq of the user turn that wrote."""
    phase: Literal["before", "after"]
    key: str
    """The workspace within the turn: the run's id for a worktree, `in-place` for the
    checkout."""
    mode: WorkspaceMode
    root: str
    """The directory the tree was written from: the `after` is taken there too."""
    tree: str
    ref: str
    head: Annotated[str | None, Field(default=None)]
    """The workspace's commit; absent on an unborn head."""
    skipped: Annotated[tuple[str, ...], Field(default=())]
    """Untracked paths over `undo.untracked_max_bytes`, left out. A skipped
    path is unknown to this snapshot, not absent from the workspace: a reader comparing two
    snapshots leaves it out. An in-place `before` reuses the pre-run tree, which skips
    nothing, so a large untracked file of the person's is in it and not in the `after`; read
    without this rule, it would look deleted by the harness."""
    part: int = 0
    """0 for the person's own turn; 1, 2, … for each wake turn after it, whose work is the
    turn's next part, one undo unit with it."""


_SNAPSHOT: Final = TypeAdapter(CodeSnapshotLine)


def taken(session: Session) -> tuple[CodeSnapshotLine, ...]:
    """Every turn snapshot on the session's log, oldest first, read through its type."""
    return tuple(
        _SNAPSHOT.validate_json(session.blob(event.payload_sha))
        for event, _ in session.state.lines()
        if event.kind == CODE_SNAPSHOT
    )


# -- arriving at a fork --------------------------------------------


def arrive(
    session: Session,
    parent_id: str,
    at_seq: int,
    reason: str,
    records: RunRecords | None,
    kept_paths: Sequence[str],
    runs: Sequence[str],
) -> None:
    """A fork's child is told, once, on arrival: every effect at or before `at_seq` that no
    fork can undo, and, when it kept the parent's code, what changed in turns it cannot see.
    A turn with a snapshot has its code restored or named, so its worktree kinds are not
    listed again."""
    snapped = frozenset(s.turn for s in taken(session) if s.turn <= at_seq)
    effects = standing.journaled_effects(session.state, records, hi=at_seq, accounted=snapped)
    if effects:
        ws = session.state.workspace_at(at_seq)
        head = ws.source_head if ws is not None else None
        standing.notice(
            session, standing.fork_effects(parent_id, at_seq, reason, effects, head)
        )
    if kept_paths:
        standing.notice(session, standing.code_kept(kept_paths, runs))


# -- the writer -------------------------------------------------------------------------------


def record_rewind(
    session: Session, undo: UndoLine, kept_paths: Sequence[str], runs: Sequence[str]
) -> int:
    """Append `undo`, and when it took the conversation and kept the code, the one notice
    naming what changed in turns the model no longer sees (§5.11.2 trigger
    6). `kept_paths` are `workspace.harness_paths`' and `runs` the runs that wrote them, which
    the caller reads: this package reads no git. Returns the undo's seq."""
    with session.lock:
        seq = session.append(undo).seq
        if undo.conversation and not undo.code and kept_paths:
            standing.notice(session, standing.code_kept(kept_paths, runs))
        return seq
