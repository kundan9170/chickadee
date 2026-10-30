"""Three reads for a frontend: two of a finished or running episode, a blob a run's record names
and what a task or one step changed; and one workspace file,
the CODE pane's view of a file no task changed.

**Nothing here is a new fact.** A blob is read through `JournalRecords.blob`, and only when a
record of the run names it (`journal.gc.blob_shas`, the one reading of which fields are
blobs), so a sha cannot be probed for. A diff is between two trees the journal already holds:
a task's `start_tree` and its last step's `post_state` (or its last report's `end_tree`); a
step's `preimages_sha` and `post_state_sha`. A task's tip is where its facts end
(`box.steps.reached` over `task_steps`), so the screen and the task's report never name
different trees. Its counts are `workspace.diff_stats` and its text `handoff.diff_of`, the
handoff review's; neither follows renames, so the counts and the text name the same paths.

**A file is read only when `files()` lists it** (`graph.ignore.listed`, the rule `@` completes
by), so the route never says more than `@` does: an ignored, denied or absent path is
`NotListed`.

Read-only: no read writes the journal, the store or the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chickadee.box.steps import reached, task_steps
from chickadee.context.project import JournalRecords
from chickadee.errors import ChickadeeError, RunNotStarted
from chickadee.handoff import diff_of
from chickadee.inspect import BLOB_LIMIT
from chickadee.journal.effects import EffectKind
from chickadee.journal.errors import BlobMissing
from chickadee.journal.gc import blob_shas
from chickadee.journal.projection import RunProjection
from chickadee.schema.journal import TaskDelegatedP, TaskReportedP
from chickadee.workspace import PathDiff, diff_stats

__all__ = [
    "FileText",
    "NotListed",
    "NotNamed",
    "NotReadable",
    "RunBlob",
    "RunDiff",
    "file_text",
    "run_blob",
    "run_diff",
]


class NotNamed(ChickadeeError):
    """A sha no record of the run names: there is nothing of the run's to serve by it."""


class NotReadable(ChickadeeError):
    """A diff asked of what the run does not hold: no such task or step, both or neither named,
    or an operation that is not a step."""


class NotListed(ChickadeeError):
    """A file `files()` does not list: ignored, refused by the read rule, or not there."""


@dataclass(frozen=True)
class FileText:
    """A workspace file as a frontend reads it: its text, or why there is none."""

    path: str
    size: int
    """Bytes."""
    text: str | None
    """UTF-8. None when `withheld` says why."""
    withheld: str | None = None
    """Why there is no text: larger than `BLOB_LIMIT`, or not UTF-8 (a binary)."""


def file_text(root: Path, path: str, listed: tuple[str, ...]) -> FileText:
    """`path` under `root`, when `listed` (what `files()` answers) holds it."""
    if path not in listed:
        raise NotListed(f"{path} is not a file this session lists")
    data = (root / path).read_bytes()
    if len(data) > BLOB_LIMIT:
        why = f"larger than {BLOB_LIMIT} bytes; named, not inlined"
        return FileText(path, len(data), None, withheld=why)
    try:
        return FileText(path, len(data), data.decode("utf-8"))
    except UnicodeDecodeError:
        return FileText(path, len(data), None, withheld="not UTF-8 text (a binary file)")


@dataclass(frozen=True)
class RunBlob:
    """A run blob as a frontend reads it: its text, or why there is none."""

    run_id: str
    sha: str
    field: str
    """The field of the first record naming it: `brief_sha`, `report_sha`, `output_sha` …"""
    size: int | None
    """Bytes; None when the blob is missing."""
    text: str | None
    """UTF-8, any byte that is not replaced. None when `withheld` says why."""
    withheld: str | None = None
    """Why there is no text: larger than `BLOB_LIMIT`, named and not inlined, as the
    inspector does; or missing, with the journal's reason."""


@dataclass(frozen=True)
class RunDiff:
    """A task's or one step's change, between two trees the journal names."""

    run_id: str
    task_id: str | None
    operation_id: str | None
    base: str
    tip: str | None
    """None while nothing has changed: a task with no step, or a step not completed."""
    files: tuple[PathDiff, ...]
    text: str
    """`git diff base tip`, cut at `handoff.DIFF_CAP` with a line saying so; empty when
    nothing changed."""


def run_blob(records: JournalRecords, run_id: str, sha: str) -> RunBlob:
    """Blob `sha` of run `run_id`, when a record of that run names it. `RunNotStarted` when the
    run has no journal; `NotNamed` when no record names `sha`."""
    field = _named(_projection(records, run_id), sha)
    try:
        data = records.blob(run_id, sha)
    except BlobMissing as exc:
        return RunBlob(run_id, sha, field, None, None, withheld=f"missing: {exc}")
    if len(data) > BLOB_LIMIT:
        why = f"larger than {BLOB_LIMIT} bytes; named, not inlined"
        return RunBlob(run_id, sha, field, len(data), None, withheld=why)
    return RunBlob(run_id, sha, field, len(data), data.decode("utf-8", "replace"))


def _projection(records: JournalRecords, run_id: str) -> RunProjection:
    """The run's fold; a run with no journal is `RunNotStarted`, as `PacketEngine.journal`
    says it, never a missing file."""
    try:
        return records.projection(run_id)
    except FileNotFoundError:
        raise RunNotStarted(
            f"run {run_id} has no journal yet: it has not started, or was stopped before it "
            "started"
        ) from None


def _named(run: RunProjection, sha: str) -> str:
    for placed in (p for lines in run.by_type.values() for p in lines):
        for field, named in blob_shas(placed.payload):
            if named == sha:
                return field
    raise NotNamed(f"no record of run {run.run_id} names blob {sha}")


def run_diff(
    records: JournalRecords,
    root: Path,
    run_id: str,
    *,
    task_id: str | None = None,
    operation_id: str | None = None,
) -> RunDiff:
    """What task `task_id`, or step `operation_id`, of run `run_id` changed: exactly one is
    named. `root` is the repository whose object store holds the trees (a worktree shares
    its repository's). `RunNotStarted` when the run has no journal; `NotReadable` when the
    run holds no such task or step."""
    if (task_id is None) == (operation_id is None):
        raise NotReadable("name one of a task or a step (operation), not both or neither")
    run = _projection(records, run_id)
    if task_id is not None:
        base, tip = _task_trees(run, task_id)
    else:
        assert operation_id is not None
        base, tip = _step_trees(run, operation_id)
    if tip is None or tip == base:
        return RunDiff(run_id, task_id, operation_id, base, None, (), "")
    files = diff_stats(root, base, tip)
    return RunDiff(run_id, task_id, operation_id, base, tip, files, diff_of(root, base, tip))


def _task_trees(run: RunProjection, task_id: str) -> tuple[str, str | None]:
    """The task's start tree, and the tree its steps reached (`reached`, where its facts end);
    with no step completed, its last report's end tree."""
    delegated = [p.payload for p in run.lines(TaskDelegatedP) if p.payload.task_id == task_id]
    if not delegated:
        raise NotReadable(f"run {run.run_id} has no task {task_id}")
    base = delegated[0].start_tree
    tip: str | None = reached(task_steps(run, task_id), base)
    if tip == base:
        reports = [p.payload for p in run.lines(TaskReportedP) if p.payload.task_id == task_id]
        tip = reports[-1].facts.end_tree if reports else None
    return base, tip


def _step_trees(run: RunProjection, operation_id: str) -> tuple[str, str | None]:
    """One step's tree before (its intent's pre-image) and after (its completion's)."""
    op = run.operations.get(operation_id)
    if op is None or op.kind != EffectKind.WORKER_STEP.value or op.intended is None:
        raise NotReadable(f"run {run.run_id} has no step {operation_id}")
    base = op.intended.preimages_sha
    if base is None:
        raise NotReadable(f"step {operation_id} names no tree before it")
    return base, op.completed.post_state_sha if op.completed is not None else None
