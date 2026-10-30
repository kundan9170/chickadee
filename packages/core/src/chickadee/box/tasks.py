"""The task book: the fold of a run's task lines, what resume, the
records and the report read.

Pure: a projection in, entries out, no file and no clock. One writer at a time, so at most one
task is open. A task is named by its plan: `T3`, a re-cut's pieces `T3.1`; without a
plan the harness names it by lineage (`next_folderless`). Each attempt is its own entry, keyed
`T3`, then `T3@2` (`ids.task_key`): the journal's `task_id` is the key; models read the name.

Its lines: `task_delegated` opens a task (`delegated`); each `task_reported` ends a round
(`reported`); each `task_reviewed` is Athena's answer: `feedback` opens the next round
(`delegated` again), `take_over` makes it hers (`taken_over`), `accept` is `accepted` until its
commit effect completes (`committed`), `re_cut` replaces it (`replaced`), and `drop` gives it
up (`abandoned`), whether or not its restore has finished, which is resume's to see.
`abandon`, `drop`'s name in older journals, folds as `drop` does, so an earlier run
reads. A `fresh`
put-back is a step under the review, never a state of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal

from chickadee.ids import parent_of, split_task_key, task_key
from chickadee.journal.effects import EffectKind
from chickadee.journal.projection import Placed, RunProjection
from chickadee.schema.journal import (
    EffectCompletedP,
    EffectIntendedP,
    ReviewAnswer,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
)

__all__ = [
    "Lineage",
    "TaskBook",
    "TaskEntry",
    "TaskState",
    "growth",
    "lineage",
    "next_folderless",
    "one_task",
]

TaskState = Literal[
    "delegated",
    "reported",
    "held",
    "taken_over",
    "accepted",
    "committed",
    "abandoned",
    "replaced",
]
"""`held`: its last round was held for a message from Zeus, with no report;
Athena's review, `continue` among its answers, is next."""

_OPEN: frozenset[TaskState] = frozenset(
    {"delegated", "reported", "held", "taken_over", "accepted"}
)
"""A task whose work is not yet committed or put back: the one writer's."""

_REVIEWED: dict[ReviewAnswer, TaskState] = {
    "accept": "accepted",
    "feedback": "delegated",
    "continue": "delegated",  # A held round's next, with no message
    "take_over": "taken_over",
    "abandon": "abandoned",
    "re_cut": "replaced",
    "drop": "abandoned",
}

_COMMITS: frozenset[str] = frozenset({EffectKind.TASK_COMMIT, EffectKind.TASK_KNOWN_GOOD})


@dataclass(frozen=True)
class TaskEntry:
    id: str
    """The attempt's key, as the journal groups its lines: `T3`, `T3@2`."""
    title: str
    by: Literal["zeus", "athena"]
    depth: int
    state: TaskState
    rounds: int
    start_tree: str
    commit: str | None = None
    call_id: str | None = None
    """The `delegate` call that made it, so a resume continues it rather than delegating it
    twice."""
    answer: ReviewAnswer | None = None
    """Athena's latest review of it, or None before her first."""
    reviews: tuple[str, ...] = ()
    """The `review` calls that answered it, in order: a resume continues a cut-off call from
    its line rather than reviewing twice."""
    plan_task: str | None = None
    """The approved plan's task it runs, or None."""

    @property
    def name(self) -> str:
        """The task's id as every model and person reads it: the plan's. The one rule
        for it: the plan task the attempt runs, else the id its key names. They differ only in
        a journal written by an older version, whose `T2` could run the plan's `T1`."""
        return self.plan_task or split_task_key(self.id)[0]

    @property
    def attempt(self) -> int:
        return split_task_key(self.id)[1]


class TaskBook:
    """Every task of a run, in the order each was delegated."""

    def __init__(self, run: RunProjection) -> None:
        entries: dict[str, TaskEntry] = {}
        commits: set[str] = set()
        for task, lines in run.tasks.items():
            for line in lines:
                p = line.payload
                if isinstance(p, TaskDelegatedP):
                    entries[task] = TaskEntry(
                        id=task,
                        title=p.title,
                        by=p.by,
                        depth=p.depth,
                        state="delegated",
                        rounds=0,
                        start_tree=p.start_tree,
                        call_id=p.call_id,
                        plan_task=p.plan_task,
                    )
                elif task not in entries:
                    continue
                elif isinstance(p, TaskReportedP):
                    state: TaskState = "held" if p.stopped == "held" else "reported"
                    entries[task] = replace(entries[task], state=state, rounds=p.round)
                elif isinstance(p, TaskReviewedP):
                    entry = entries[task]
                    entries[task] = replace(
                        entry,
                        state=_REVIEWED[p.answer],
                        answer=p.answer,
                        reviews=(*entry.reviews, p.call_id),
                    )
                elif isinstance(p, EffectIntendedP) and p.effect_kind in _COMMITS:
                    commits.add(p.operation_id)
                elif (
                    isinstance(p, EffectCompletedP)
                    and p.operation_id in commits
                    and p.outcome == "succeeded"
                ):
                    entries[task] = replace(
                        entries[task], state="committed", commit=p.post_state_sha
                    )
        self._entries = tuple(entries.values())

    def entries(self) -> tuple[TaskEntry, ...]:
        return self._entries

    def open(self) -> TaskEntry | None:
        """The task not yet committed or put back, if any: at most one."""
        found = [e for e in self._entries if e.state in _OPEN]
        if len(found) > 1:
            raise ValueError(f"two tasks open at once: {[e.id for e in found]}; one writer")
        return found[0] if found else None

    def get(self, task: str) -> TaskEntry | None:
        """The attempt keyed `task`."""
        return next((e for e in self._entries if e.id == task), None)

    def latest(self, task: str) -> TaskEntry | None:
        """The newest attempt of the task named `task`."""
        found = [e for e in self._entries if e.name == task]
        return found[-1] if found else None

    def next_key(self, task: str) -> str:
        """The key of `task`'s next attempt: its first, or the one after the newest."""
        latest = self.latest(task)
        return task_key(task, latest.attempt + 1 if latest is not None else 1)

    def of_plan(self, task: str) -> TaskEntry | None:
        """The latest delegation of the plan's `task` not dropped: what runs it, or ran it.
        A dropped attempt may be delegated again; a committed or re-cut one never is."""
        found = [e for e in self._entries if e.plan_task == task and e.state != "abandoned"]
        return found[-1] if found else None


def one_task(book: TaskBook) -> bool:
    """Whether the run is a one-task plan's: its first task is Zeus's. A design's
    first task is Athena's, delegated from her approved plan."""
    entries = book.entries()
    return bool(entries) and entries[0].by == "zeus"


@dataclass(frozen=True)
class Lineage:
    """Which failed attempt each task replaces, the one fold the "last attempt" note, a plan's
    growth and the folderless names read.

    Read from the ids, never from the order of delegation: an attempt after
    the first replaces the attempt before it, when that one was dropped; a piece (`T3.1`)
    replaces its parent's attempt that was re-cut."""

    replaces: Mapping[str, TaskReviewedP]
    """Each attempt's replaced attempt, by the review that ended it, keyed by attempt."""
    cut: TaskReviewedP | None
    """The newest `re_cut`: where a task with no plan names its next pieces."""
    last: TaskReviewedP | None
    """The newest review of all."""


def lineage(run: RunProjection) -> Lineage:
    lines: list[Placed[TaskDelegatedP] | Placed[TaskReviewedP]] = [
        *run.lines(TaskDelegatedP),
        *run.lines(TaskReviewedP),
    ]
    reviewed: dict[str, TaskReviewedP] = {}
    """Each attempt's newest review, by key."""
    replaces: dict[str, TaskReviewedP] = {}
    cut: TaskReviewedP | None = None
    last: TaskReviewedP | None = None
    for line in sorted(lines, key=lambda placed: placed.seq):
        p = line.payload
        if isinstance(p, TaskDelegatedP):
            name, attempt = split_task_key(p.task_id)
            if attempt > 1:
                before = reviewed.get(task_key(name, attempt - 1))
                if before is not None and before.answer in ("drop", "abandon"):
                    replaces[p.task_id] = before
                continue
            parent = parent_of(name)
            if parent is None:
                continue
            found = [
                r
                for key, r in reviewed.items()
                if split_task_key(key)[0] == parent and r.answer == "re_cut"
            ]
            if found:
                replaces[p.task_id] = found[-1]
        else:
            reviewed[p.task_id] = p
            last = p
            if p.answer == "re_cut":
                cut = p
    return Lineage(replaces, cut, last)


def next_folderless(run: RunProjection, book: TaskBook) -> str:
    """The key of a task delegated with no plan to name it: after a drop, the
    dropped task's next attempt; after a re-cut of `Tk`, its next piece `Tk.<n>`, until another
    re-cut; otherwise the next top-level task."""
    seen = lineage(run)
    last = seen.last
    if last is not None and last.answer in ("drop", "abandon"):
        name = split_task_key(last.task_id)[0]
        newest = book.latest(name)
        if newest is not None and newest.id == last.task_id:
            # Not delegated again since: the next delegation is its next attempt.
            return book.next_key(name)
    if seen.cut is not None:
        parent = split_task_key(seen.cut.task_id)[0]
        pieces = {e.name for e in book.entries() if parent_of(e.name) == parent}
        return f"{parent}.{len(pieces) + 1}"
    tops = [int(e.name[1:]) for e in book.entries() if parent_of(e.name) is None]
    return f"T{max(tops, default=0) + 1}"


def growth(run: RunProjection, book: TaskBook) -> str | None:
    """How a one-task plan grew: "1 task approved, 3 delegated (T1.1, T1.2
    from a re-cut of T1)". A piece names its parent. None for a design's run, and for a
    one-task plan that delegated only its own task: it did not grow."""
    entries = book.entries()
    if not one_task(book) or len(entries) < 2:
        return None
    pieces: dict[str, list[str]] = {}
    for task, review in lineage(run).replaces.items():
        if review.answer == "re_cut":
            pieces.setdefault(split_task_key(review.task_id)[0], []).append(task)
    said = "; ".join(
        f"{', '.join(ts)} from a re-cut of {parent}" for parent, ts in pieces.items()
    )
    return f"1 task approved, {len(entries)} delegated" + (f" ({said})" if said else "")
