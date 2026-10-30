"""What a run that continues a work carries from the work's earlier runs: their task books, the
plan they approved, and where an
unfinished task stood. The desk reads it to seed the inherited plan, refuse a committed task,
number a task's next attempt after its earlier ones, and let an unfinished task go on from
what its earlier attempt left; Athena's head reads it to say where the work stands.

**The one fold of a work's tasks across runs** (`work_tasks`): each task by its name, as its
latest attempt left it, its attempts counted, and whether a commit it made has since left the
branch. `WorkProgress` reads the same fold, so the desk and Zeus's status never disagree on
where a task stands.

Read from the earlier runs' journals, which ended or were superseded before this run began, so
nothing here changes under the run; a resume reads the same (`Continuation` is in the run's
request). A run that starts a work carries nothing (`Carried.none()`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chickadee.box.request import Continuation
from chickadee.box.tasks import TaskBook, TaskEntry
from chickadee.journal.projection import RunProjection
from chickadee.process import ProcessSupervisor, run_git
from chickadee.work import approved_plans, plan_ids, plan_pin

__all__ = ["Carried", "PriorTask", "reachable", "work_tasks"]


@dataclass(frozen=True)
class PriorTask:
    """One task of a work, as its latest attempt across the runs read left it."""

    run_id: str
    entry: TaskEntry
    attempts: int
    """Every attempt of it across those runs."""
    in_place: bool = False
    """Its run worked in place, where a task is never a commit, so nothing can be lost."""
    lost: bool = False
    """It committed, and that commit has left the branch since (a reset, a rebase): it is to
    do again."""

    @property
    def done(self) -> bool:
        """Committed, and the commit still holds."""
        return self.entry.state == "committed" and not self.lost

    @property
    def unfinished(self) -> bool:
        """Started and stopped before its review settled it: what it changed may still be in
        the tree, uncommitted."""
        return self.entry.state not in ("committed", "abandoned", "replaced")


def work_tasks(
    runs: Sequence[tuple[str, RunProjection]],
    root: Path,
    supervisor: ProcessSupervisor | None = None,
) -> dict[str, PriorTask]:
    """Each task of a work by its name, in the order first delegated: its latest attempt
    across `runs` (oldest first), which is where it stands. A committed task is never
    delegated again unless its commit was lost, so the latest attempt is always the one that
    holds. Whether a commit was lost is asked of git once, here."""
    tasks: dict[str, PriorTask] = {}
    for run_id, projection in runs:
        start = projection.start
        in_place = start is not None and start.isolation == "in_place"
        for entry in TaskBook(projection).entries():
            before = tasks.get(entry.name)
            attempts = (before.attempts if before is not None else 0) + 1
            tasks[entry.name] = PriorTask(run_id, entry, attempts, in_place)
    sup = supervisor or ProcessSupervisor()
    return {
        name: PriorTask(t.run_id, t.entry, t.attempts, t.in_place, lost=_lost(t, root, sup))
        for name, t in tasks.items()
    }


def _lost(task: PriorTask, root: Path, supervisor: ProcessSupervisor) -> bool:
    commit = task.entry.commit
    if task.entry.state != "committed" or task.in_place or not commit:
        return False
    return not reachable(root, commit, supervisor)


@dataclass(frozen=True)
class Carried:
    continues: Continuation | None
    prior: tuple[tuple[str, RunProjection], ...] = ()
    """The earlier runs, oldest first."""
    _tasks: dict[str, PriorTask] = field(default_factory=lambda: {})

    @staticmethod
    def none() -> Carried:
        return Carried(None)

    @staticmethod
    def of(
        continues: Continuation | None,
        read: Callable[[str], RunProjection],
        root: Path,
    ) -> Carried:
        """What `continues` carries, each earlier run read by `read`; `root` is where the
        work's branch is checked out."""
        if continues is None:
            return Carried.none()
        prior = tuple((run_id, read(run_id)) for run_id in continues.prior_runs)
        return Carried(continues, prior, work_tasks(prior, root))

    @property
    def continuing(self) -> bool:
        return self.continues is not None

    def task(self, name: str) -> PriorTask | None:
        return self._tasks.get(name)

    def tasks(self) -> tuple[PriorTask, ...]:
        return tuple(self._tasks.values())

    def committed(self) -> tuple[PriorTask, ...]:
        """The tasks done in the earlier runs, their commits still on the branch."""
        return tuple(t for t in self._tasks.values() if t.done)

    def plan_tree(self, root: Path) -> str | None:
        """The approved plan this run inherits, as its run pinned it."""
        c = self.continues
        if c is None or c.plan_run is None or c.plan_version is None:
            return None
        return plan_pin(root, c.plan_run, c.plan_version)

    def approved_ids(self, root: Path) -> dict[str, str]:
        """Every id an approved version of the plan held in the earlier runs, by its file,
        oldest first."""
        return plan_ids(
            root,
            [
                plan_pin(root, run_id, line.version)
                for run_id, projection in self.prior
                for line in approved_plans(projection)
            ],
        )


def reachable(root: Path, commit: str, supervisor: ProcessSupervisor | None = None) -> bool:
    """Whether `commit` is on the branch checked out at `root`: a task whose commit left HEAD
    (a reset, a rebase) is `lost`, and may run again."""
    done = run_git(
        root,
        "merge-base",
        "--is-ancestor",
        commit,
        "HEAD",
        supervisor=supervisor or ProcessSupervisor(),
    )
    return done.exit_code == 0
