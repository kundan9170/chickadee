"""Every effect's lifecycle, recorded in one place, and what each effect kind is.

Every effect has an `operation_id` and a write-ahead `effect_intended → effect_started →
effect_completed` lifecycle. This module is the only writer of those three lines: `perform`
for an effect being performed inside one block, `begin` for one that is completed elsewhere (a
background job, intended on the conversation's worker and completed on its waiter), and
`reopen` for resume finishing one a crash left open. The order
is built in rather than checked afterwards; the journal file's legality, for a journal written
by anything, stays the projection's, which records a damaged lifecycle as a defect.

The log is the caller's (`EffectLog`): the run journal through `on_journal`, or the session log
through the scout's sink. The payloads are the same frozen models either way, so no reader has
to know where a line landed.

`KINDS` holds what is known about each kind, once: which reconciliation resume runs for it,
which restore mechanism it uses, and whether it stays inside a run's worktree.

Depends only on the schema, so the journal package imports neither the executor, the
workspace nor the orchestrator.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from chickadee.errors import HarnessBug
from chickadee.schema.journal import (
    EffectCompletedP,
    EffectIntendedP,
    EffectStartedP,
    EventKind,
    P,
)

if TYPE_CHECKING:
    from chickadee.journal.projection import Operation

__all__ = [
    "EFFECT_EVENTS",
    "KINDS",
    "Appends",
    "EffectKind",
    "EffectLog",
    "Intent",
    "KindRule",
    "OnRaise",
    "OpenEffect",
    "Outcome",
    "Reconcile",
    "Restore",
    "ambiguous",
    "begin",
    "failed_if",
    "on_journal",
    "perform",
    "reopen",
    "sealed",
]


class EffectKind(StrEnum):
    """`EffectIntendedP.effect_kind`. The wire field stays a `str` (frozen); this is the
    one place its values are spelled."""

    BOOTSTRAP = "bootstrap"
    SCOUT_ESCAPE = "scout_escape"
    SHELL_WRITE = "shell_write"
    WORKER_STEP = "worker_step"
    """A box role's writing call: an edit or a command, recorded with the
    trees before and after it."""
    TASK_COMMIT = "task_commit"
    """An accepted task's commit in the run's worktree, found by its
    trailer."""
    TASK_KNOWN_GOOD = "task_known_good"
    """An accepted task in place, where nothing is committed."""
    APPROVAL_COMMIT = "approval_commit"
    """A work folder's part approved at a stop, or a plan revised, as its own commit in the
    run's worktree, found by its trailer. Never in place."""
    JOB = "shell_job"
    """A Zeus shell command that ran on past its call: intended at the yield,
    completed by its waiter at the exit, on the session log."""


EFFECT_EVENTS: Final = frozenset(
    {EventKind.EFFECT_INTENDED, EventKind.EFFECT_STARTED, EventKind.EFFECT_COMPLETED}
)
"""The three lifecycle lines: each is durable before `EffectLog.record` returns.
The journal's fsync policy and the session sink's sync both read this set."""

type Outcome = Literal["succeeded", "failed", "ambiguous"]


class EffectLog(Protocol):
    """Where an effect's lines go: the run journal (`on_journal`), or the session log through
    the scout's sink. `record` returns only once a line in `EFFECT_EVENTS` is durable."""

    def record(self, kind: EventKind, payload: P) -> int: ...


class Appends(Protocol):
    """What `journal.writer.Journal` and `journal.port.JournalPort` both are."""

    def append(
        self,
        kind: EventKind,
        payload: Any,
        *,
        task_id: str | None = None,
        round: int | None = None,
    ) -> int: ...


@dataclass(frozen=True)
class _OnJournal:
    journal: Appends
    task_id: str | None = None
    round: int | None = None

    def record(self, kind: EventKind, payload: P) -> int:
        return self.journal.append(kind, payload, task_id=self.task_id, round=self.round)


def on_journal(
    journal: Appends, *, task_id: str | None = None, round: int | None = None
) -> EffectLog:
    """The run journal as an effect's log, every line bound to `task_id` and its `round`,
    or to the run when none is given."""
    return _OnJournal(journal, task_id, round)


@dataclass(frozen=True)
class Intent:
    """The caller's half of `EffectIntendedP`. Whatever it names (a preimage manifest, a parked
    tree, a head) must already be durable: `perform` writes the intent first, and cannot check
    that what it points at exists."""

    operation_id: str
    kind: EffectKind
    approved_sha: str
    preconditions: tuple[str, ...]
    idempotent: bool
    surface: tuple[str, ...]
    preimages: str | None = None


type OnRaise = Callable[[Exception], Literal["failed", "ambiguous"]]


def ambiguous(exc: Exception) -> Literal["ambiguous"]:
    """The default: an exception proves nothing about what reached the world."""
    return "ambiguous"


def failed_if(*proven: type[Exception]) -> OnRaise:
    """`failed` for an exception of a type the caller knows is raised before anything
    changed; `ambiguous` for any other."""

    def rule(exc: Exception) -> Literal["failed", "ambiguous"]:
        return "failed" if isinstance(exc, proven) else "ambiguous"

    return rule


class OpenEffect:
    """An operation whose completion is not yet written. Completed exactly once."""

    def __init__(self, log: EffectLog, operation_id: str) -> None:
        self._log = log
        self.operation_id = operation_id
        self.completed = False

    def record(self, kind: EventKind, payload: P) -> int:
        """A line that belongs under this operation and is not part of its lifecycle: an
        action's `exec` record, written before the completion so a command that ran is never
        a result the journal lost between the two writes."""
        if kind in EFFECT_EVENTS:
            raise HarnessBug(f"{kind.value} for {self.operation_id} is the recorder's to write")
        return self._log.record(kind, payload)

    def complete(
        self,
        outcome: Outcome,
        *,
        observed: tuple[str, ...] = (),
        post_state: str | None = None,
        result: str | None = None,
    ) -> None:
        if self.completed:
            raise HarnessBug(f"{self.operation_id} completed twice; one id, one lifecycle")
        self._log.record(
            EventKind.EFFECT_COMPLETED,
            EffectCompletedP(
                operation_id=self.operation_id,
                outcome=outcome,
                observed_surface=observed,
                post_state_sha=post_state,
                result_sha=result,
            ),
        )
        self.completed = True


def begin(log: EffectLog, intent: Intent) -> OpenEffect:
    """Write the intent and the start, and hand back the open effect for whoever completes it.
    `perform` is this inside a block; a caller that completes the effect on another thread, or
    later, holds the handle itself and must complete it exactly once."""
    log.record(
        EventKind.EFFECT_INTENDED,
        EffectIntendedP(
            operation_id=intent.operation_id,
            effect_kind=intent.kind.value,
            approved_sha=intent.approved_sha,
            preconditions=intent.preconditions,
            idempotent=intent.idempotent,
            declared_surface=intent.surface,
            preimages_sha=intent.preimages,
        ),
    )
    log.record(EventKind.EFFECT_STARTED, EffectStartedP(operation_id=intent.operation_id))
    return OpenEffect(log, intent.operation_id)


@contextmanager
def perform(
    log: EffectLog, intent: Intent, *, on_raise: OnRaise = ambiguous
) -> Generator[OpenEffect]:
    """`begin`, yield the open effect, and see it closed.

    - An `Exception` that leaves the block with the effect open closes it with
      `on_raise(exc)`, nothing observed, and is re-raised unchanged.
    - A normal exit with the effect open is a `HarnessBug`: an effect whose outcome nobody
      stated.
    - Anything that is not an `Exception` (a kill, `KeyboardInterrupt`) writes nothing,
      because a kill cannot: recovery reads the operation as started.
    """
    effect = begin(log, intent)
    try:
        yield effect
    except Exception as exc:
        if not effect.completed:
            effect.complete(on_raise(exc))
        raise
    if not effect.completed:
        raise HarnessBug(f"{intent.operation_id} left open: an effect must state its outcome")


def reopen(log: EffectLog, operation: Operation) -> OpenEffect:
    """Resume's way into a lifecycle a crash left open: the start is written when the
    operation stopped at its intent, and the handle completes it. A completed operation is a
    `HarnessBug`: one operation id, one lifecycle."""
    if operation.completed is not None:
        raise HarnessBug(f"{operation.operation_id} is already completed; it is not reopened")
    if not operation.started:
        log.record(
            EventKind.EFFECT_STARTED, EffectStartedP(operation_id=operation.operation_id)
        )
    return OpenEffect(log, operation.operation_id)


# -- what each kind is --------------------------------------------------------------------


class Reconcile(StrEnum):
    """Which reconciliation resume runs for an operation of a kind.
    Each member is one method of resume's rebuild; this names it and copies none of it."""

    TRAILER = "trailer"
    """A commit is found by its trailer since the run's head: landed, not, or cannot tell."""
    HEAD = "head"
    """In place, the head is the known-good state."""
    SEALED = "sealed"
    """Re-runnable only if the seal was enforced (`scout.escapes.reconcile`). No resume acts
    on it."""
    STEP = "step"
    """A box step cut off: the paths changed since its pre-image tree are put back
    to it and the role is told the call was interrupted; one whose command declared the
    network is never re-run, its outcome unknown (`box/resume.py`)."""
    JOB = "job"
    """A job a crash left running: its process group reaped only when the
    pid's start time matches, then completed `ambiguous`, ended `lost` (`zeus/jobs.py`)."""
    NONE = "none"
    """No recovery path reads this kind."""


class Restore(StrEnum):
    """How an effect of a kind is taken back."""

    TREE = "tree"
    """`workspace.py`'s git-tree refs: a footprint known only afterwards."""
    NONE = "none"


@dataclass(frozen=True)
class KindRule:
    recovery: Reconcile
    restore: Restore
    worktree_only: bool
    """Confined to a run's worktree, so a fork undoes it by never inheriting the worktree.
    """

    @property
    def sealed(self) -> bool:
        """Changes nothing but its own output: a reader listing what a run did to the tree or
        the world (a fork's notice, a handoff's open operations) has nothing to list for it,
        finished or cut short."""
        return self.recovery is Reconcile.SEALED


def sealed(kind: str | None) -> bool:
    """`KindRule.sealed` for a journaled kind; a kind the table does not know is not sealed,
    since what it did is unknown."""
    try:
        return KINDS[EffectKind(kind)].sealed if kind is not None else False
    except ValueError:
        return False


KINDS: Final[Mapping[EffectKind, KindRule]] = MappingProxyType(
    {
        # Run-scoped, while resume works per task.
        EffectKind.BOOTSTRAP: KindRule(Reconcile.NONE, Restore.NONE, True),
        # Ran in the checkout, read-only.
        EffectKind.SCOUT_ESCAPE: KindRule(Reconcile.SEALED, Restore.NONE, False),
        # A declared Zeus write: on the session log, never resumed,
        # since it belongs to no run. A call a crash cut short is `INTERRUPTED`, what it did
        # unknown, and its snapshot ref is what takes it back.
        EffectKind.SHELL_WRITE: KindRule(Reconcile.NONE, Restore.TREE, False),
        # A box step: the worktree's, or the person's checkout in place, so no fork
        # is assumed to undo it; its trees are what take it back.
        EffectKind.WORKER_STEP: KindRule(Reconcile.STEP, Restore.TREE, False),
        # An accepted task: its commit found by trailer; in place, the
        # person's checkout, which no fork undoes.
        EffectKind.TASK_COMMIT: KindRule(Reconcile.TRAILER, Restore.NONE, True),
        EffectKind.TASK_KNOWN_GOOD: KindRule(Reconcile.HEAD, Restore.NONE, False),
        # The work folder's own commits: found by trailer as a task's
        # are; never made in place.
        EffectKind.APPROVAL_COMMIT: KindRule(Reconcile.TRAILER, Restore.NONE, True),
        # A background job: on the session log; what it wrote after its yield is not
        # recorded and never taken back; the person's checkout is not a run's.
        EffectKind.JOB: KindRule(Reconcile.JOB, Restore.NONE, False),
    }
)
