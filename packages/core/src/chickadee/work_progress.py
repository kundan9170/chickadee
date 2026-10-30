"""Where a work stands: its approved design and
plan, each against the folder as it is now, each plan task's state across every run of the
work, its latest run, and Zeus's tells to it. What `coder`'s dispatch decides by, and what Zeus
reads at a turn.

**A projection, never a second record**, as `open_works.py` is: read from the runs'
saved requests and journals and from git. The engine keeps the last one only as a memo
(`Progresses`), read again whenever anything it was read from moved. The one fact no record
holds, whether a run's thread is running in this engine, is asked of the caller (`live`).

**One design tree per run.** A run's approved design is the tree its request pinned
(`WorkPin.tree`); an amendment opens a new run. So a plan's
approval is bound to the design tree of the run that approved it, or, inherited, to the one it
was approved under (`Continuation.plan_bound_to`).

Nothing here calls a model, and nothing is written: the trees are hashed through a throwaway
index (`workspace.folder_tree`).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol

from chickadee.box.carry import PriorTask, work_tasks
from chickadee.box.episode import opened_with
from chickadee.box.request import CoderRequest
from chickadee.box.tasks import TaskBook
from chickadee.context.lines import TellLine
from chickadee.journal import paths
from chickadee.journal.errors import JournalError
from chickadee.journal.projection import RunProjection
from chickadee.journal.runs import RunProjections
from chickadee.process import ProcessSupervisor, run_git
from chickadee.schema.journal import TellDeliveredP, TellOutcome, TellReplyP
from chickadee.state.roots import Roots
from chickadee.work import approved_plans, part_dir, plan_pin, plan_tasks
from chickadee.workspace import folder_tree, pinned_text

__all__ = [
    "DONE",
    "JournalRuns",
    "PartState",
    "Progresses",
    "RunAt",
    "RunReader",
    "StampedRuns",
    "TaskStatus",
    "TellStatus",
    "WorkKey",
    "WorkProgress",
    "complete",
    "progress",
    "status_text",
]

TaskState = Literal[
    "pending", "open", "held", "committed", "unchecked", "dropped", "replaced", "lost"
]
RunState = Literal["live", "suspended", "ended"]
TellState = Literal["sent", "delivered", "replied", "unanswered"]

DONE: Final = frozenset({"committed", "unchecked", "dropped", "replaced"})
"""A plan task that needs nothing more: committed (or, in place, `unchecked`), or settled by
Athena's review (dropped, or re-cut into pieces that the plan lists)."""


@dataclass(frozen=True)
class WorkKey:
    """Which work: a design's folder (`dir`, `name`), or a one-task plan, named by the run that
    delegated it first."""

    dir: str | None = None
    name: str | None = None
    first_run: str | None = None

    @property
    def text(self) -> str:
        if self.name is not None:
            return f"work:{self.dir}/{self.name}"
        return f"task:{self.first_run}"

    def owns(self, run_id: str, request: CoderRequest) -> bool:
        """Whether run `run_id`, opened with `request`, is this work's."""
        if self.name is not None:
            pin = request.design.work if request.design is not None else None
            return pin is not None and (pin.dir, pin.name) == (self.dir, self.name)
        if request.task is None:
            return False
        first = request.continues.prior_runs[0] if request.continues is not None else run_id
        return first == self.first_run


@dataclass(frozen=True)
class PartState:
    """The design's or the plan's approval against what is on disk now."""

    version: int
    """0: never approved; then one more for each approved tree that differs from the last."""
    approved: str | None
    """The tree (a one-task plan: the brief's sha) the latest approval pinned."""
    current: str | None
    """The same folder's tree now, on disk; None when absent, or for a one-task plan's brief,
    which only the next `coder` call carries."""
    bound_to: str | None = None
    """The plan only: the design tree its approval was given under."""

    @property
    def changed(self) -> bool:
        return self.current is not None and self.current != self.approved


@dataclass(frozen=True)
class TaskStatus:
    id: str
    """The plan's id."""
    title: str
    state: TaskState
    key: str | None = None
    """The latest attempt's key, once delegated."""
    commit: str | None = None
    run: str | None = None
    """The run that holds or settled it."""


@dataclass(frozen=True)
class RunAt:
    run_id: str
    state: RunState
    outcome: str | None
    """`run_end.outcome` once ended."""
    at_task: str | None
    """The task open when it stopped, or now: its id."""


@dataclass(frozen=True)
class TellStatus:
    """A tell to the work's team: `sent` until a run delivers it, `delivered` until
    Athena replies, `replied` with her outcome; `unanswered` when the run it was delivered in
    ended with no reply."""

    id: str
    state: TellState
    outcome: TellOutcome | None = None


@dataclass(frozen=True)
class WorkProgress:
    key: WorkKey
    runs: tuple[str, ...]
    """The work's runs, oldest first."""
    design: PartState
    plan: PartState
    plan_run: str | None
    """The run whose approved plan is the latest, and `plan_version` its version there."""
    plan_version: int | None
    plan_sha: str | None
    tasks: tuple[TaskStatus, ...]
    """The latest approved plan's tasks in its order, then any task it no longer lists."""
    run: RunAt | None
    """The work's latest run."""
    tells: tuple[TellStatus, ...] = ()
    """The session's tells to the work's runs, oldest first."""


class RunReader(Protocol):
    """What `progress` reads of a run: its request (`episode.opened_with`) and its journal
    (`RunProjections.readable`). None when the run left no journal, or one from before
    contracts v2."""

    def request(self, run_id: str) -> CoderRequest | None: ...

    def projection(self, run_id: str) -> RunProjection | None: ...


class StampedRuns(RunReader, Protocol):
    """A `RunReader` that also says, cheaply, which runs there are and whether a run's journal
    moved, so `Progresses` reads a work again only when one did."""

    def candidates(self) -> Sequence[str]: ...

    def stamp(self, run_id: str) -> object: ...


class JournalRuns:
    """`RunReader` over the clone's run journals. A run's request is read
    once and kept: its config snapshot never changes. Its journal is the engine's one fold of it
    (`RunProjections`), advanced by what was appended since."""

    def __init__(self, roots: Roots, runs: RunProjections) -> None:
        self._roots = roots
        self._runs = runs
        self._lock = threading.Lock()
        """Guards `_requests` only: the engine shares one reader between Zeus's turn and the
        dispatch. A leaf: the snapshot is read outside it, so two first reads may both read."""
        self._requests: dict[str, CoderRequest | None] = {}

    def request(self, run_id: str) -> CoderRequest | None:
        with self._lock:
            if run_id in self._requests:
                return self._requests[run_id]
        try:
            found: CoderRequest | None = opened_with(self._roots, run_id)
        except (FileNotFoundError, JournalError, OSError, ValueError):
            # No snapshot, or one before contracts v2: not a work's run that can continue.
            found = None
        with self._lock:
            self._requests[run_id] = found
        return found

    def projection(self, run_id: str) -> RunProjection | None:
        try:
            return self._runs.readable(run_id)
        except (FileNotFoundError, JournalError, OSError):
            return None

    def stamp(self, run_id: str) -> object:
        """What moves when the run's journal does: its size and time, as git's own index
        tells a changed file (an append always grows it). None when there is no journal."""
        try:
            st = paths.journal_path(self._roots, run_id).stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def candidates(self) -> tuple[str, ...]:
        """Every run of the clone, oldest first by when its journal was made: a work's runs
        may be another session's, one this session adopted the work from."""
        root = paths.runs_root(self._roots)
        try:
            found = [
                (config.stat().st_mtime_ns, d.name)
                for d in root.iterdir()
                if (config := d / paths.CONFIG_NAME).is_file()
            ]
        except OSError:
            return ()
        return tuple(name for _, name in sorted(found))


def progress(
    key: WorkKey,
    candidates: Sequence[str],
    read: RunReader,
    root: Path,
    *,
    live: Callable[[str], bool],
    supervisor: ProcessSupervisor | None = None,
    tells: Sequence[TellLine] = (),
) -> WorkProgress:
    """Where the work `key` stands. `candidates` are run ids, oldest first, that may be its
    (the session's episodes, and the repository's other runs for a work a session adopted);
    `root` is where its folder is and its branch is checked out. Whether a run's commits can
    be checked is its own request's (`workspace`), since a work's runs may differ. `tells` are
    the session's `tell` lines: those naming one of the work's runs are its."""
    sup = supervisor or ProcessSupervisor()
    runs: list[tuple[str, CoderRequest, RunProjection]] = []
    for run_id in dict.fromkeys(candidates):
        request = read.request(run_id)
        if request is None or not key.owns(run_id, request):
            continue
        projection = read.projection(run_id)
        if projection is not None and projection.start is not None:
            runs.append((run_id, request, projection))
    design = _design(key, runs, root)
    plan, plan_run, plan_version, plan_sha = _plan(key, runs, root)
    tasks = _tasks(runs, root, plan_run, plan_version, supervisor=sup)
    latest = runs[-1] if runs else None
    run = _run_at(latest[0], latest[2], live) if latest is not None else None
    return WorkProgress(
        key=key,
        runs=tuple(r for r, _, _ in runs),
        design=design,
        plan=plan,
        plan_run=plan_run,
        plan_version=plan_version,
        plan_sha=plan_sha,
        tasks=tasks,
        run=run,
        tells=_tells(tells, runs),
    )


class Progresses:
    """`progress` read again only when something it reads moved. The frame
    asks at every user turn, and a work moves only when a run appends to its journal, starts or
    stops running, a part's folder is edited, or the branch moves. Each is cheap to see (a stat
    per journal, a walk of the two folders, one `rev-parse`), so the git reads `progress`
    makes (each folder hashed, each pin read, each commit checked against the branch) are made
    only after one did. The engine holds one, shared by Zeus's frame and the dispatch."""

    def __init__(self, runs: StampedRuns, supervisor: ProcessSupervisor | None = None) -> None:
        self.runs = runs
        self._supervisor = supervisor or ProcessSupervisor()
        self._lock = threading.Lock()
        """Guards `_held` only. A leaf: `progress` runs outside it, so two first reads may both
        read, and the later one is kept."""
        self._held: dict[tuple[WorkKey, Path], tuple[object, WorkProgress]] = {}

    def of(
        self,
        key: WorkKey,
        root: Path,
        *,
        live: Callable[[str], bool],
        tells: Sequence[TellLine] = (),
    ) -> WorkProgress:
        """Where `key` stands, as `progress` reads it."""
        candidates = tuple(self.runs.candidates())
        # Taken before the read, so a change made while reading moves the next stamp and is
        # read then; taken after, it could be kept as read and never read again. A tell's
        # delivery and reply are its run's journal, which the stamp already reads.
        stamp = (self._stamp(key, root, candidates, live), tuple(t.id for t in tells))
        with self._lock:
            held = self._held.get((key, root))
        if held is not None and held[0] == stamp:
            return held[1]
        found = progress(
            key,
            candidates,
            self.runs,
            root,
            live=live,
            supervisor=self._supervisor,
            tells=tells,
        )
        with self._lock:
            self._held[(key, root)] = (stamp, found)
        return found

    def _stamp(
        self,
        key: WorkKey,
        root: Path,
        candidates: Sequence[str],
        live: Callable[[str], bool],
    ) -> object:
        owned = [
            run_id
            for run_id in dict.fromkeys(candidates)
            if (request := self.runs.request(run_id)) is not None and key.owns(run_id, request)
        ]
        head = run_git(
            root, "rev-parse", "--verify", "--quiet", "HEAD", supervisor=self._supervisor
        )
        folders = (
            tuple(
                _files(root / part_dir(key.dir, key.name, part)) for part in ("design", "plan")
            )
            if key.name is not None and key.dir is not None
            else ()
        )
        return (
            tuple((run_id, self.runs.stamp(run_id), live(run_id)) for run_id in owned),
            head.stdout,
            folders,
        )


def _files(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """Every file under `folder`, with its size and time: what `folder_tree` hashes, and more
    (ignored files too), so an edit it would see always moves this."""
    found: list[tuple[str, int, int]] = []
    for dirpath, _dirs, names in os.walk(folder):
        for name in names:
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                continue
            found.append((str(path.relative_to(folder)), st.st_size, st.st_mtime_ns))
    return tuple(sorted(found))


def complete(p: WorkProgress) -> bool:
    """Every plan task needs nothing more, and the latest run ended `COMPLETED`: a task
    Athena dropped on purpose keeps no work open, and a run that ended `PARTIAL` is not complete
    whatever its tasks say."""
    return (
        p.run is not None
        and p.run.outcome == "COMPLETED"
        and bool(p.tasks)
        and all(t.state in DONE for t in p.tasks)
    )


# -- the design ---------------------------------------------------------------------------


def _design(
    key: WorkKey, runs: Sequence[tuple[str, CoderRequest, RunProjection]], root: Path
) -> PartState:
    if key.name is None or key.dir is None:
        return PartState(0, None, None)
    trees: list[str] = []
    for _run_id, request, _ in runs:
        pin = request.design.work if request.design is not None else None
        if pin is not None and (not trees or trees[-1] != pin.tree):
            trees.append(pin.tree)
    current = folder_tree(root, part_dir(key.dir, key.name, "design"))
    return PartState(len(trees), trees[-1] if trees else None, current)


# -- the plan -----------------------------------------------------------------------------


def _plan(
    key: WorkKey, runs: Sequence[tuple[str, CoderRequest, RunProjection]], root: Path
) -> tuple[PartState, str | None, int | None, str | None]:
    """The latest approved plan: its tree (a one-task plan: its brief's sha), the design tree
    it was approved under, and the run and version that hold its pin."""
    versions: list[str] = []
    found: tuple[str, int, str, str | None] | None = None
    for run_id, request, projection in runs:
        pin = request.design.work if request.design is not None else None
        own_design = pin.tree if pin is not None else None
        for line in approved_plans(projection):
            if key.name is None:
                approved = request.task.sha if request.task is not None else line.sha
            else:
                approved = plan_pin(root, run_id, line.version) or line.sha
            inherited = line.inherited_from is not None
            bound = (
                request.continues.plan_bound_to
                if inherited and request.continues is not None
                else own_design
            )
            if not versions or versions[-1] != approved:
                versions.append(approved)
            found = (run_id, line.version, line.sha, bound)
    current = (
        folder_tree(root, part_dir(key.dir, key.name, "plan"))
        if key.name is not None and key.dir is not None
        else None
    )
    if found is None:
        return PartState(0, None, current), None, None, None
    run_id, version, sha, bound = found
    return PartState(len(versions), versions[-1], current, bound), run_id, version, sha


# -- the tasks ----------------------------------------------------------------------------


def _tasks(
    runs: Sequence[tuple[str, CoderRequest, RunProjection]],
    root: Path,
    plan_run: str | None,
    plan_version: int | None,
    *,
    supervisor: ProcessSupervisor,
) -> tuple[TaskStatus, ...]:
    listed: list[tuple[str, str]] = []
    if plan_run is not None and plan_version is not None:
        tree = plan_pin(root, plan_run, plan_version)
        readme = pinned_text(root, tree, "README.md") if tree is not None else None
        listed = [(task, title) for task, title, _file in plan_tasks(readme or "")]
    found = work_tasks(
        [(run_id, projection) for run_id, _, projection in runs], root, supervisor
    )
    last, last_ended = (runs[-1][0], runs[-1][2].end is not None) if runs else (None, True)
    # A task is open only while the run that holds it may go on: the latest, not yet ended.
    latest = {
        name: _status(t, ended=t.run_id != last or last_ended) for name, t in found.items()
    }
    out = [latest.get(task, TaskStatus(task, title, "pending")) for task, title in listed]
    seen = {task for task, _ in listed}
    out += [status for name, status in latest.items() if name not in seen]
    return tuple(out)


def _status(task: PriorTask, *, ended: bool) -> TaskStatus:
    """Where one task stands, as `work_tasks` left it."""
    entry = task.entry

    def said(state: TaskState, commit: str | None = None) -> TaskStatus:
        return TaskStatus(entry.name, entry.title, state, entry.id, commit, task.run_id)

    match entry.state:
        case "committed":
            if task.in_place:
                return said("unchecked", entry.commit)
            return said("lost" if task.lost else "committed", entry.commit)
        case "abandoned":
            return said("dropped")
        case "replaced":
            return said("replaced")
        case "held" if not ended:
            return said("held")
        case _:
            # Not settled: open while its run may go on, and pending again once it ended,
            # which put its files back or left them for the next run's review.
            return said("pending" if ended else "open")


# -- the tells ----------------------------------------------------------------------------


def _tells(
    tells: Sequence[TellLine], runs: Sequence[tuple[str, CoderRequest, RunProjection]]
) -> tuple[TellStatus, ...]:
    """Each tell to one of the work's runs, matched to the run that delivered it and the reply:
    a tell written while a run was paused is delivered by the run that continues it."""
    mine = {run_id for run_id, _, _ in runs}
    delivered: dict[str, bool] = {}
    """Each delivered tell, and whether the run it was delivered in ended."""
    replied: dict[str, TellOutcome] = {}
    for _, _, projection in runs:
        for p in projection.lines(TellDeliveredP):
            delivered[p.payload.tell] = projection.end is not None
        for p in projection.lines(TellReplyP):
            replied[p.payload.tell] = p.payload.outcome
    out: list[TellStatus] = []
    for tell in tells:
        if tell.run_id not in mine:
            continue
        if tell.id in replied:
            out.append(TellStatus(tell.id, "replied", replied[tell.id]))
        elif tell.id in delivered:
            out.append(TellStatus(tell.id, "unanswered" if delivered[tell.id] else "delivered"))
        else:
            out.append(TellStatus(tell.id, "sent"))
    return tuple(out)


# -- the run ------------------------------------------------------------------------------


def _run_at(run_id: str, projection: RunProjection, live: Callable[[str], bool]) -> RunAt:
    book = TaskBook(projection)
    held = book.open()
    at = held.name if held is not None else None
    if projection.end is not None:
        return RunAt(run_id, "ended", projection.end.outcome, at)
    return RunAt(run_id, "live" if live(run_id) else "suspended", None, at)


# -- what Zeus reads ----------------------------------------------------------------------


def status_text(p: WorkProgress) -> str:
    """The work's status as Zeus reads it at a turn: deterministic, so a frame is due
    only when it changes."""
    lines = [f'<open_work key="{p.key.text}">']
    if p.key.name is not None:
        lines.append(f"design  {_part(p.design, 'design')}")
    lines.append(f"plan    {_part(p.plan, 'plan' if p.key.name is not None else 'brief')}")
    if p.tasks:
        lines.append("tasks   " + " · ".join(_task(t) for t in p.tasks))
    if p.run is not None:
        lines.append(f"run     {_run(p.run)}")
    if p.tells:
        lines.append("tells   " + " · ".join(_tell(t) for t in p.tells))
    lines.append(f"next    {_next(p)}")
    lines.append("</open_work>")
    return "\n".join(lines)


def _part(part: PartState, what: str) -> str:
    if part.version == 0:
        return "not approved yet"
    said = f"v{part.version} approved"
    if part.current is None:
        return said
    return said + (" · changed on disk since" if part.changed else " · unchanged")


def _task(t: TaskStatus) -> str:
    match t.state:
        case "committed":
            return f"{t.id} committed {t.commit[:7] if t.commit else ''}".rstrip()
        case "unchecked":
            return f"{t.id} done (in place, not checkable)"
        case "lost":
            return f"{t.id} lost (its commit left the branch)"
        case "held":
            return f"{t.id} held for a message"
        case _:
            return f"{t.id} {t.state}"


def _tell(t: TellStatus) -> str:
    return f"{t.id} replied: {t.outcome}" if t.state == "replied" else f"{t.id} {t.state}"


def _run(r: RunAt) -> str:
    at = f" at {r.at_task}" if r.at_task else ""
    if r.state == "ended":
        return f"{r.run_id} ended {r.outcome}{at}"
    return f"{r.run_id} {r.state}{at}"


def _next(p: WorkProgress) -> str:
    name = p.key.name
    call = f'coder(work="{name}")' if name is not None else "coder(title, brief) with its brief"
    if complete(p) and not p.design.changed:
        return "complete: change the design to add to it"
    if p.run is not None and p.run.state == "live":
        return "the team is working on it"
    if p.design.changed:
        return f"{call} asks the person to approve the design's changes, then continues"
    return f"{call} continues from where it stopped; nothing done is redone"
