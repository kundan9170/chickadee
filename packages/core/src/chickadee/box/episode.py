"""One episode of the box: a `CoderRequest` in, a `CoderReport` out.

The request carries work already approved at its stop (`box/request.py`). The order is Zeus's
port contract (`zeus.delegate.BoxPort`): `Box.open` names the run and takes the repository
WRITE lease, and does nothing else, so the session's `episode` line is durable before any work.
`BoxRun.report` then does the work, in this order:

1. create the run's journal and write `run_start`: the goal, the approved work's sha, the
   templates in force, the caps;
2. start the run's budget scope, and compose the task loop over the journal (`compose.py`);
3. run it (`box/loop.TaskLoop`): Athena's request, its plan, tasks and reviews;
4. assemble the report from the journal, store it, and write `run_end`: the outcome, why,
   and the counts of tasks and commits. Never `SOLVED`: no gate ran;

and releases the lease on every exit, a raise included. A paused run (the first
interrupt, or the process leaving) is not a report: it returns the run id `resume`
continues, and writes no `run_end`. A run the harness failed in is reported `BLOCKED`
with the error and what its journal holds, and writes no `run_end` either: it is left as a
crash leaves it, for resume.

A run opened by an older version is not resumed: its journal, or the request its
config snapshot holds, is refused.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError

from chickadee import __version__
from chickadee.box import report as rep
from chickadee.box.compose import EpisodeSetting, RunParts, compose_episode
from chickadee.box.loop import LoopResult, TaskLoop
from chickadee.box.request import CoderRequest
from chickadee.box.resume import Asking, never_asking
from chickadee.box.shell import RunSuspended
from chickadee.box.tasks import TaskBook, growth
from chickadee.box.tells import unanswered
from chickadee.errors import HarnessBug
from chickadee.hooks.bind import bind_once, write_bound
from chickadee.hooks.fire import Hooks, HookSet, RoleHooks
from chickadee.hooks.payloads import EpisodeEnded
from chickadee.hooks.points import HookPoint
from chickadee.hooks.sinks import RunHookSink
from chickadee.journal import paths
from chickadee.journal.errors import JournalCorrupt, JournalError
from chickadee.journal.projection import RunProjection
from chickadee.journal.writer import Journal
from chickadee.process import DIFF_FLAGS, git
from chickadee.schema import SCHEMA_VERSION, sha256_hex
from chickadee.schema.ids import new_run_id
from chickadee.schema.journal import EventKind, RunEndP, RunStartP
from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import EMPTY_TREE, SnapshotSite, Workspace, head_commit

__all__ = [
    "Box",
    "BoxEnv",
    "BoxRun",
    "Complete",
    "OpenedWith",
    "Paused",
    "RunSnapshot",
    "end_superseded",
    "opened_with",
]


class OpenedWith(BaseModel):
    """What a box run was opened with, kept in its config snapshot so a resume knows what
    it was asked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: CoderRequest


class RunSnapshot(BaseModel):
    """A box run's config snapshot (`Journal.create`), read back through its type."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    box: OpenedWith


_log = logging.getLogger(__name__)


def _harness_error(exc: Exception) -> str:
    return f"the run stopped on a harness error: {type(exc).__name__}: {exc}"


def _snapshot(request: CoderRequest) -> dict[str, object]:
    """The run's config snapshot: a serializer, writing the request out."""
    return RunSnapshot(box=OpenedWith(request=request)).model_dump(mode="json")


def opened_with(roots: Roots, run_id: str) -> CoderRequest:
    """The request run `run_id` was opened with, read back from its config snapshot. A run
    opened by an older version holds a request of the packet loop: refused."""
    raw = Journal.config_snapshot(roots, run_id)
    try:
        return RunSnapshot.model_validate(raw).box.request
    except ValidationError as exc:
        why = (
            "the run was opened before contracts v2: a packet loop's run is not resumed "
            "(D196); the tag pre-d196 resumes it"
            if _packet_request(raw)
            # A v2 snapshot that does not validate is damage, said as such: pointing at
            # pre-d196 would send the person to a tag that cannot resume it.
            else f"the run's config snapshot does not validate: {exc}"
        )
        raise JournalCorrupt(None, str(paths.run_dir(roots, run_id)), why) from exc


_V2_REQUEST_KEYS = frozenset(CoderRequest.model_fields)


def _packet_request(raw: object) -> bool:
    """Whether a snapshot holds the packet loop's request: one naming a field a v2 request
    does not have (its tier, approval, hunks or verification policy)."""
    box: object = cast("dict[str, object]", raw).get("box") if isinstance(raw, dict) else None
    request: object = (
        cast("dict[str, object]", box).get("request") if isinstance(box, dict) else None
    )
    return isinstance(request, dict) and bool(
        set(cast("dict[str, object]", request)) - _V2_REQUEST_KEYS
    )


@dataclass(frozen=True)
class BoxEnv:
    """What the engine hands the box, composed once per episode."""

    workspace_for: Callable[[str, CoderRequest], Workspace]
    """Takes the repository WRITE lease for run `run_id` and returns where it writes:
    the session's worktree, or the person's checkout for an in-place request."""
    reattach: Callable[[str], Workspace]
    """Takes the lease again for run `run_id`, over the workspace it already has
    (`workspace.reattach`): how a paused or interrupted run is continued."""
    parts_for: Callable[[Workspace], RunParts]
    """The long-lived parts a run in that workspace is composed over."""
    setting: Callable[[], EpisodeSetting]
    """The episode's settings, made on its thread when it starts: its model clients are opened
    there and closed with it."""
    asking: Asking = never_asking
    """Whether a step's permission question still waits on the session log."""
    cancel: threading.Event | None = None
    """The person's soft stop: the episode pauses at the next task boundary."""
    abort: threading.Event | None = None
    """The person's hard stop."""
    suspend: threading.Event | None = None
    """Set with `abort` when the process is leaving: the run stops now, as a hard stop
    stops it, but is left for resume, as a crash leaves it: no `run_end`."""
    run_id: Callable[[], str] = field(default=new_run_id)


@dataclass(frozen=True)
class Complete:
    """A `coder` call on a work already complete: every task of its plan is done and its
    last run ended `COMPLETED`, so nothing runs."""

    work: str


@dataclass(frozen=True)
class Paused:
    """A run interrupted at a task boundary, or stopped where it stood because the
    process is leaving: nothing is reported until it resumes."""

    run_id: str


def end_superseded(roots: Roots, run_id: str, why: str) -> None:
    """End a suspended run another run of its work replaces: an amended design opens a new run,
    so the suspended one is never resumed beside it.
    Written as the run's own end, `ABORTED` with `why`, which its readers already read as an
    end. Nothing of its workspace is touched: what its open task left is the
    new run's to go on from (`box/carry.py`). A run already ended is left as it is."""
    journal = Journal.open_for_resume(roots, run_id)
    try:
        run = journal.projection()
        if run.end is not None:
            return
        book = TaskBook(run)
        spend = run.spend
        journal.append(
            EventKind.RUN_END,
            RunEndP(
                outcome="ABORTED",
                why=why,
                tasks=len(book.entries()),
                committed=sum(e.state == "committed" for e in book.entries()),
                tokens=spend.tokens_in + spend.tokens_out,
                cost_usd=spend.cost_usd,
                wall_ms=0,
            ),
        )
    finally:
        journal.close()


class Box:
    def __init__(self, env: BoxEnv) -> None:
        self.env = env

    def open(self, request: CoderRequest) -> BoxRun:
        """Name the run and take the lease. Nothing else."""
        run_id = self.env.run_id()
        workspace = self.env.workspace_for(run_id, request)
        return BoxRun(self.env, request, run_id, workspace, resuming=False)

    def resume(self, run_id: str) -> BoxRun:
        """The run `run_id` named, to be continued from its journal: a run paused at a
        task boundary, or one the process died in. Takes the lease again over the
        run's own workspace, reads back the request it was opened with, and does nothing
        else; `report()` continues it."""
        workspace = self.env.reattach(run_id)
        try:
            request = opened_with(resolve_roots(workspace.root), run_id)
        except BaseException:
            workspace.release()
            raise
        return BoxRun(self.env, request, run_id, workspace, resuming=True)


class BoxRun:
    """One named episode, not yet running. `report()` runs it, or continues it, once."""

    def __init__(
        self,
        env: BoxEnv,
        request: CoderRequest,
        run_id: str,
        workspace: Workspace,
        *,
        resuming: bool,
    ) -> None:
        self._env = env
        self._request = request
        self.run_id = run_id
        self._workspace = workspace
        self._resuming = resuming
        self._done = False
        self._released = False
        self._setting: EpisodeSetting | None = None
        self._hooks: HookSet | None = None

    @property
    def request(self) -> CoderRequest:
        return self._request

    @property
    def workspace(self) -> SnapshotSite:
        """Where a turn snapshot of this run's workspace is taken. A resumed run's
        in-place pre-run snapshot is older than the checkout, so it is not offered as a turn's
        `before`."""
        return self._workspace.snapshot_site(fresh=not self._resuming)

    def abandon(self) -> None:
        """The report will never be asked for: release the lease. A fresh run did nothing; a
        resumed one is left as its journal stands, to be continued again."""
        self._done = True
        self._finish()

    def report(self) -> rep.CoderReport | Paused:
        """The run's report. An error the run did not answer is reported too, never raised
        into the call that asked: Zeus reads it, and the session holds the result."""
        if self._done:
            raise HarnessBug(f"run {self.run_id} was already reported or abandoned")
        try:
            return self._report()
        except Exception as exc:
            # Before the run had a journal, or past naming what it holds: the error alone.
            _log.error("episode %s stopped on a harness error", self.run_id, exc_info=exc)
            return rep.CoderReport.before_run("BLOCKED", _harness_error(exc))
        finally:
            self._finish()

    def _finish(self) -> None:
        # Once: `abandon` after a report, or a report after `abandon`, releases nothing twice.
        if self._released:
            return
        self._done = self._released = True
        self._workspace.release()

    def _report(self) -> rep.CoderReport | Paused:
        parts = self._env.parts_for(self._workspace)
        self._setting = self._env.setting()
        roots = resolve_roots(self._workspace.root)
        if self._resuming:
            journal = Journal.open_for_resume(roots, self.run_id)
        else:
            journal = Journal.create(roots, self.run_id, _snapshot(self._request))
        try:
            base = self._started(journal, parts)
            started = perf_counter()
            try:
                loop = self._loop(journal, parts, base)
                result = loop.resume() if self._resuming else loop.run()
            except RunSuspended:
                return Paused(self.run_id)
            except Exception as exc:
                if self._suspended():
                    # Raised as the pause tore the run down: the person left, the harness did
                    # not fail. Left as a crash leaves it, for resume.
                    _log.info("run %s paused through %r", self.run_id, exc)
                    return Paused(self.run_id)
                return self._failed(journal, base, exc)
            # A pause stops the loop as a hard stop does, and whatever it ended as, the run
            # is left for resume: no `run_end`.
            if result.paused or self._suspended():
                return Paused(self.run_id)
            return self._assemble(journal, result, base, started, loop)
        finally:
            journal.close()
            # Before `_finish` gives the lease back: no server, and no re-check writing the
            # index, outlives the episode's hold on its workspace.
            if parts.servers is not None:
                parts.servers.close()

    def _started(self, journal: Journal, parts: RunParts) -> str:
        """The commit what this run changed is measured from, after `run_start` is written,
        or read back on a resume. HEAD in a worktree, and in the person's checkout the snapshot
        of their own uncommitted work, so their changes are never counted as the box's.
        The run's hooks are bound here, where its script runner is."""
        root = self._workspace.root
        mode = self._workspace.spec.mode
        # Resets the run scope only; a resumed run's spend is restored from its journal.
        parts.budget.start_run(self.run_id)
        self._hooks, unwritten = bind_once(
            RunHookSink(journal),
            self._settings().hooks,
            parts.scripts,
            where=f"run {self.run_id}",
        )
        if self._resuming:
            start = journal.projection().start
            if start is None:
                raise HarnessBug(
                    f"run {self.run_id} has no run_start; there is nothing to resume"
                )
            # A run that crashed before its set was written, or started by an older version.
            if unwritten is not None:
                write_bound(RunHookSink(journal), self._hooks, unwritten)
            if start.isolation != mode:
                raise HarnessBug(
                    f"run {self.run_id} started {start.isolation}, and its workspace is now "
                    f"{mode}"
                )
            return self._workspace.spec.dirty_snapshot_sha or start.repo_head or EMPTY_TREE
        # A repository with no commit yet (`git init`, a new project) has no HEAD. The run
        # starts from nothing: `repo_head` is recorded empty, which every reader of it takes as
        # "no lower bound" (`Projector.landed`, `undo`), and its changes are measured from the
        # empty tree, as `workspace.snapshot` names the pre-image there.
        head = head_commit(root) or ""
        configured, caps = parts.budget.run_caps()
        setting = self._settings()
        request = self._request
        journal.append(
            EventKind.RUN_START,
            RunStartP(
                goal=request.goal,
                work_sha=request.work_sha,
                template_shas=_template_shas(request, setting),
                config_sha=_config_sha(setting, self._hooks.digest),
                repo_head=head,
                engine_version=__version__,
                schema_version=SCHEMA_VERSION,
                isolation=mode,
                hard_cost_cap_usd=configured,
                caps=caps,
            ),
        )
        # `run_start` stays a run's first line; its `config_sha` already holds the set's digest.
        if unwritten is not None:
            write_bound(RunHookSink(journal), self._hooks, unwritten)
        return self._workspace.spec.dirty_snapshot_sha or head or EMPTY_TREE

    def _loop(self, journal: Journal, parts: RunParts, base: str) -> TaskLoop:
        env = self._env
        return compose_episode(
            parts,
            self._settings(),
            journal,
            self._request,
            cancel=env.abort,
            pause=env.cancel,
            asking=env.asking,
            base=base,
            hooks=self._hooks,
        )

    def _suspended(self) -> bool:
        suspend = self._env.suspend
        return suspend is not None and suspend.is_set()

    def _settings(self) -> EpisodeSetting:
        if self._setting is None:
            raise HarnessBug(f"run {self.run_id}'s settings were read before it started")
        return self._setting

    def _assemble(
        self, journal: Journal, result: LoopResult, base: str, started: float, loop: TaskLoop
    ) -> rep.CoderReport:
        report = self._reported(journal, result, base)
        # The work folder's records come last; what the run changed is read
        # again after them, so the report's diff holds their commit too.
        if loop.record_report(report):
            report = self._reported(journal, result, base)
        report = self._ended(journal, report)
        run = journal.projection()
        tasks = report.tasks
        stored = journal.put_blob(report.model_dump_json().encode("utf-8"))
        spend = run.spend
        journal.append(
            EventKind.RUN_END,
            RunEndP(
                outcome=report.outcome,
                why=report.why,
                tasks=len(tasks),
                committed=sum(t.commit is not None for t in tasks),
                tokens=spend.tokens_in + spend.tokens_out,
                cost_usd=spend.cost_usd,
                wall_ms=int((perf_counter() - started) * 1000),
                handoff_path=f"{paths.BLOBS_DIR}/{stored}",
            ),
        )
        return report

    def _ended(self, journal: Journal, report: rep.CoderReport) -> rep.CoderReport:
        """`episode_ended`: what the run's hooks say of how it ended, carried in the
        report Zeus reads. Read back on a resume that ends it again, never asked twice."""
        hooks = RoleHooks(
            Hooks(self._bound_hooks(), RunHookSink(journal)), "athena", run_id=self.run_id
        )
        payload = EpisodeEnded(
            outcome=report.outcome,
            why=report.why or "",
            tasks=len(report.tasks),
            committed=sum(t.commit is not None for t in report.tasks),
        )
        note = hooks.noted_once(
            HookPoint.EPISODE_ENDED, payload, call_id=f"episode:{self.run_id}"
        )
        return report if note is None else report.model_copy(update={"harness": (note,)})

    def _bound_hooks(self) -> HookSet:
        if self._hooks is None:
            raise HarnessBug(f"run {self.run_id}'s hooks were read before it started")
        return self._hooks

    def _reported(self, journal: Journal, result: LoopResult, base: str) -> rep.CoderReport:
        run = journal.projection()
        tasks = rep.reported_tasks(run)
        handles, numstat = self._handles(journal, run, base)
        return rep.CoderReport(
            outcome=result.outcome,
            why=result.why,
            growth=growth(run, TaskBook(run)),
            run_id=run.run_id,
            tasks=tasks,
            athena=result.athena,
            change_summary=rep.summarize(tasks, numstat),
            handles=handles,
            cost=run.spend,
            unanswered=unanswered(run),
        )

    def _failed(self, journal: Journal, base: str, exc: Exception) -> rep.CoderReport:
        """A run the harness failed: `BLOCKED`, naming the error, with what the journal
        holds — its handles, the diff of whatever landed, the spend. The journal is left
        unended, as a crash leaves it, for resume."""
        _log.error("run %s stopped on a harness error", self.run_id, exc_info=exc)
        try:
            run = journal.projection()
        except JournalError:
            # A journal that cannot be read names nothing: the error is the report.
            return rep.CoderReport.before_run("BLOCKED", _harness_error(exc))
        tasks = rep.reported_tasks(run)
        handles, numstat = self._handles(journal, run, base)
        return rep.CoderReport(
            outcome="BLOCKED",
            why=_harness_error(exc),
            run_id=run.run_id,
            tasks=tasks,
            change_summary=rep.summarize(tasks, numstat),
            handles=handles,
            cost=run.spend,
            unanswered=unanswered(run),
        )

    def _handles(
        self, journal: Journal, run: RunProjection, base: str
    ) -> tuple[rep.ReportHandles, tuple[tuple[int, int, str], ...]]:
        """The run's journal, the diff of what it changed since `base` when anything did, and
        each failing output, as handles; with the diff's numstat."""
        root = self._workspace.root
        numstat = _numstat(git(root, "diff", "--numstat", *DIFF_FLAGS, base))
        journal_handle = rep.stored_handle("journal", run.run_id, _journal_bytes(journal))
        diff_handle = None
        if numstat:
            diff = git(root, "diff", *DIFF_FLAGS, base).encode("utf-8")
            journal.put_blob(diff)
            diff_handle = rep.stored_handle(
                "diff", run.run_id, diff, preview=rep.diff_preview(numstat)
            )
        handles = rep.ReportHandles(
            journal=journal_handle,
            diff=diff_handle,
            failing_output=rep.failing_output(run, journal.get_blob),
        )
        return handles, numstat


def _template_shas(request: CoderRequest, setting: EpisodeSetting) -> dict[str, str]:
    """The handoff templates in force, by kind: the one the work was written to, and the ones
    Athena plans and briefs to."""
    shas = {"brief": setting.brief_template_sha}
    if request.design is not None:
        shas["design"] = request.design.template_sha
        shas["plan"] = sha256_hex(setting.head.plan_template)
    elif request.task is not None:
        shas["brief"] = request.task.template_sha
    return shas


def _config_sha(setting: EpisodeSetting, hooks: str) -> str:
    """`run_start.config_sha`: the episode's knobs, named, so two runs under different limits
    or hooks (`hooks`, the bound set's digest) are told apart."""
    knobs = {
        "worker": dict(setting.worker.limits),
        "athena": dict(setting.athena.limits),
        "untracked_max_bytes": setting.untracked_max_bytes,
        "task_seconds": setting.task_seconds,
        "recut_cap": setting.recut_cap,
        "one_task_max": setting.one_task_max,
        "design_layer": setting.design_layer,
        "note_cap": setting.note_cap,
        "hooks": hooks,
    }
    return sha256_hex(repr(sorted(knobs.items())))


def _numstat(text: str) -> tuple[tuple[int, int, str], ...]:
    rows: list[tuple[int, int, str]] = []
    for line in text.splitlines():
        added, removed, path = line.split("\t", 2)
        rows.append((_count(added), _count(removed), path))
    return tuple(rows)


def _count(field: str) -> int:
    # A binary file's counts are "-": it changed, and has no lines to count.
    return 0 if field == "-" else int(field)


def _journal_bytes(journal: Journal) -> bytes:
    return paths.journal_path(journal.roots, journal.run_id).read_bytes()
