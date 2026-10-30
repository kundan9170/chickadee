"""Zeus's jobs: a shell command that runs on past its call, and its exit.

Every command Zeus's shell runs comes through `JobTable.spawn`. It starts through
`Executor.start`, waits on the job inside its yield window, and answers as `Executor.run` would
when the command ends there. Past the window it yields: the step around it ends, a `JOB`
effect is begun on the session log, and the call answers `Yielded`. The job's waiter thread,
at the exit, completes that effect and, when the notification rule says so, writes a
`JOB_EXITED` notice and tells the conversation a wake may be due. No other record of a job
exists: its state is the session's fold of those effects (`SessionState.jobs`), so a finished
job never counts as running, here or in another process.

**Guarded state.** The table's `_lock` guards `_entries`, `_recording`, `_next` and every
entry's `alongside`. Each entry's `lock` guards its `yielded`, `effect`, `yield_tree`,
`settings` and `stepped`; the waiter, `spawn`'s yield and `stepped` all take it, so the yield
and the exit race under one lock and each exit is answered exactly once, to the call or to the
waiter. An entry is in `_entries` from its yield until its waiter takes it out, first thing at
the exit, and in `_recording` from then until everything its exit writes is written, so `close`
waits for it too; a lock is never taken while another entry's is held.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, Protocol

from pydantic import TypeAdapter

from chickadee.context.handle import Handle
from chickadee.context.session import Session, SessionState
from chickadee.context.standing import Notice, NoticeTrigger, hook_noted, job_exited
from chickadee.context.store import session_dir
from chickadee.context.vocabulary import canonical_json
from chickadee.durable.files import unlink_durably
from chickadee.edits.lines import split_lines
from chickadee.errors import HarnessBug
from chickadee.executor.executor import Executor, Started
from chickadee.executor.types import (
    ExecRequest,
    ExecResult,
    PolicyClass,
    SandboxLevel,
    joined_output,
)
from chickadee.git_guard import Held, held, load, put_back
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.payloads import JobExited
from chickadee.hooks.points import HookPoint
from chickadee.ids import job_of, job_operation, job_precondition, operation_component
from chickadee.journal.effects import EffectKind, Intent, OpenEffect, begin, reopen
from chickadee.journal.projection import Operation
from chickadee.limits import ResolvedLimit, limit, resolved_value
from chickadee.process.supervisor import (
    GRACE_S,
    JOB_CHECK_S,
    Cancelled,
    Job,
    JobExit,
    reap_group,
)
from chickadee.process.supervisor import process_started as started_at
from chickadee.schema import sha256_hex
from chickadee.shell import JobKind, Yielded
from chickadee.tools.context import JobOutput
from chickadee.workspace import changed_paths, current_tree, snapshot_operation
from chickadee.zeus.writes import SessionEffects, ShellEffect, guarded_before, shell_effects

__all__ = [
    "JobIntent",
    "JobLimits",
    "JobReport",
    "JobTable",
    "JobView",
    "exit_unread",
    "job_limits",
    "wake_ready",
]

TAIL_LINES: Final = 40
"""The end of a job's output its exit notice carries; the whole of it is behind the handle."""

_STEP_WAIT_S: Final = 30.0
"""The longest a job's exit waits for its own step to report (`stepped`). The step closes on
the conversation's worker right after the yield; one that raised never reports, and the exit
then measures from the yield's own snapshot."""

_REAP_S: Final = 2 * GRACE_S + JOB_CHECK_S + 1
"""How long a killed job's waiter is joined: the group's reap (TERM, then KILL after the
grace), the waiter's next look, and a second to record the exit."""

TAIL_READ_BYTES: Final = 1024 * 1024
"""The most of each stream `output` reads of a running job: a tail is the end of a log, and a
server's log can run to `zeus.job_max_log_bytes`, each read of it a new session blob."""

KEPT_END_BYTES: Final = 512 * 1024
"""What a job keeps of each stream once it has ended: its first and its last 512 KiB, so a
failure at startup and the final error both survive, and a job's record is at most 2 MiB. The
middle of a longer log is dropped and the record says how much; the output files are deleted
once the record is written, so a job's output is in one place, the session's store."""

Ended = Literal["exited", "timed_out", "killed", "ceiling", "lost"]


def _default(name: str) -> int:
    return int(limit(f"zeus.{name}").value)


@dataclass(frozen=True)
class JobLimits:
    """`zeus.shell_yield_s` and the `zeus.job_*` rows, resolved (`limits.py`). The defaults are
    the register's, read from it."""

    yield_s: int = _default("shell_yield_s")
    max_jobs: int = _default("max_jobs")
    max_runtime_s: int = _default("job_max_runtime_s")
    max_wait_s: int = _default("job_max_wait_s")
    max_log_bytes: int = _default("job_max_log_bytes")
    inline_bytes: int = _default("shell_inline_bytes")


def job_limits(resolved: Sequence[ResolvedLimit]) -> JobLimits:
    """The table's limits from the config's resolved rows (`limits.py`)."""

    def row(name: str) -> int:
        return resolved_value(resolved, "zeus", f"zeus.{name}")

    return JobLimits(
        yield_s=row("shell_yield_s"),
        max_jobs=row("max_jobs"),
        max_runtime_s=row("job_max_runtime_s"),
        max_wait_s=row("job_max_wait_s"),
        max_log_bytes=row("job_max_log_bytes"),
        inline_bytes=row("shell_inline_bytes"),
    )


@dataclass(frozen=True)
class JobIntent:
    """What a `JOB` effect's intent says of its job, as its preconditions:
    written at the yield, read back by `views` and by resume. The one place they are spelled,
    but for the id, whose key is `ids.job_precondition`'s (the session's fold reads it)."""

    job_id: str
    kind: JobKind
    step: str
    """The call's operation, which names the job's two files and its command's call."""
    pid: int
    pgid: int | None
    started: str | None
    """`process_started` at the spawn: resume signals the pid only when it still matches."""
    effects_measured: bool
    unsandboxed: bool
    walled: bool
    """A sandbox held it, with git's settings read-only. One that none held
    has them put back at its exit, or by resume (decision 2, Amendment 1)."""

    def preconditions(self) -> tuple[str, ...]:
        return (
            job_precondition(self.job_id),
            f"kind={self.kind}",
            f"pid={self.pid}",
            f"pgid={self.pgid if self.pgid is not None else ''}",
            f"started={self.started or ''}",
            f"step={self.step}",
            f"effects_measured={str(self.effects_measured).lower()}",
            f"walled={str(self.walled).lower()}",
            *(("unsandboxed",) if self.unsandboxed else ()),
        )

    @classmethod
    def read(cls, preconditions: Sequence[str]) -> JobIntent:
        pre = dict(p.split("=", 1) for p in preconditions if "=" in p)
        pgid = pre.get("pgid")
        return cls(
            job_id=job_of(tuple(preconditions)) or "?",
            kind="server" if pre.get("kind") == "server" else "task",
            step=pre.get("step", ""),
            pid=int(pre.get("pid") or 0),
            pgid=int(pgid) if pgid else None,
            started=pre.get("started") or None,
            effects_measured=pre.get("effects_measured") == "true",
            unsandboxed="unsandboxed" in preconditions,
            walled=pre.get("walled") == "true",
        )


@dataclass(frozen=True)
class JobReport:
    """A job's own facts, the blob its `JOB` effect's completion names."""

    job_id: str
    kind: JobKind
    line: str
    exit_code: int | None
    """None for `lost`: the harness did not see it end."""
    ended: Ended
    by: str | None
    notify: bool
    wall_ms: int | None
    wrote: tuple[str, ...]
    overwrote: tuple[str, ...]
    alongside: tuple[str, ...]
    """The jobs that ran in its workspace while it did: `wrote` may be theirs as well."""
    moved: bool
    """The tree at the exit is not the tree at the yield."""
    git_restored: tuple[str, ...]
    """git's settings it changed, put back at its exit."""
    output: str | None
    """The handle id of the kept output, when the harness saw the exit; `None` for `lost`."""
    kept: str | None
    """The session blob of the kept output (`KEPT_END_BYTES` of each stream's two ends, joined),
    which `job tail` reads once the job has ended; `None` when it printed nothing to keep."""
    dropped: int
    """The bytes from the middle of its streams that were not kept."""
    out_bytes: int
    err_bytes: int


_REPORT: Final = TypeAdapter(JobReport)


@dataclass(frozen=True)
class JobView:
    """One job as `job list` and `/jobs` show it: from its effect, and its report once ended."""

    job_id: str
    operation_id: str
    step: str
    """The call's operation, which names the job's two files."""
    kind: JobKind
    running: bool
    report: JobReport | None
    state: str
    """How it stands, worded once for every reader (`job list`, `/jobs`): `running`, or how it
    ended (`_how`)."""
    row: str
    """Its one line, as `job list` and `/jobs` both show it."""


@dataclass
class _Running:
    job_id: str
    operation_id: str
    """The job's own: `job:<the call's operation>`."""
    step: str
    kind: JobKind
    line: str
    out: Path
    err: Path
    executor: Executor
    root: Path
    """The workspace it runs in: the checkout, or the session's worktree once the view moved."""
    store: Callable[[bytes], Handle]
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: Started | None = None
    yielded: bool = False
    effect: OpenEffect | None = None
    yield_tree: str | None = None
    stepped: list[tuple[tuple[str, ...], str]] = field(
        default_factory=list[tuple[tuple[str, ...], str]]
    )
    alongside: set[str] = field(default_factory=set[str])
    settings: Held | None = None
    """What `git_guard` holds, as its step left it: what its exit puts back when no wall held
    it (`walled`)."""
    walled: bool = True
    has_step: bool = False
    """Its command ran as a recorded step (`ShellWrites`), which reports the tree it left."""
    step_reported: threading.Event = field(default_factory=threading.Event)
    """Set once its own step has said where it left the tree (`stepped`): its exit waits for
    that, or it would measure from before the step's put-backs (`_STEP_WAIT_S`)."""
    recorded: threading.Event = field(default_factory=threading.Event)
    """Set once its waiter has written all it writes (the effect's completion, and the notice
    when it notifies), however it ended."""
    held: tuple[str, ...] = ()
    """The jobs running at its yield that kept it from becoming one (`zeus.max_jobs`), so it
    ran to its end in its call. Set by `_run` on the calling thread only."""
    ceiling: bool = False
    """It ended inside its window past `zeus.job_max_log_bytes`. Set by `_run`, as `held` is."""


_Run = Callable[[_Running, ExecRequest, "threading.Event | None", bool], "ExecResult | Yielded"]


class _JobCall:
    """One shell call's spawn (`ShellJobs.spawn`): runs the command through the table, and once
    it has, says which running jobs kept it from becoming a job, as the table decided at its
    yield."""

    def __init__(self, run: _Run, entry: Callable[[str], _Running], background: bool) -> None:
        self._run = run
        self._entry = entry
        self._background = background
        self.held: tuple[str, ...] = ()
        self.ceiling = False

    def __call__(
        self, request: ExecRequest, operation_id: str, cancel: threading.Event | None
    ) -> ExecResult | Yielded:
        entry = self._entry(operation_id)
        try:
            return self._run(entry, request, cancel, self._background)
        finally:
            self.held = entry.held
            self.ceiling = entry.ceiling


class JobTable:
    """One session's jobs: `ShellJobs` for its shell, `Beside` for its steps, and the `job`
    tool's port. `wake` is told when an exit that notifies has been written."""

    def __init__(
        self,
        session: Session,
        *,
        limits: JobLimits | None = None,
        wake: Callable[[], None] | None = None,
        hooks: RoleHooks | None = None,
        team: Callable[[], str | None] | None = None,
        board: Callable[[str], str] | None = None,
    ) -> None:
        self._session = session
        self._team = team
        self._board = board
        """A run's task board by its id (`PacketEngine._run_board`), what `tail` shows of the
        run; None, and its tail says only that it works."""
        """The run working in the background now, named as `job list` shows it
        beside the jobs; None when none is."""
        self._limits = limits if limits is not None else JobLimits()
        self._wake = wake
        self._hooks = hooks
        """Zeus's hooks: told each exit this table tells him of (`job_exited`)."""
        self._lock = threading.Lock()
        self._entries: dict[str, _Running] = {}
        self._recording: dict[str, _Running] = {}
        self._started: dict[Path, int] = {}
        """Jobs ever started, by workspace: what `Beside.started_in` answers."""
        """Jobs that have ended and whose waiter is still writing their exit."""
        self._next = 1 + sum(1 for e in self._effects() if e.is_job)
        self._dir = session_dir(session.root, session.session_id) / "jobs"

    # -- ShellJobs ----------------------------------------------------------------------

    def full(self) -> tuple[str, ...]:
        running = self.running()
        return running if len(running) >= self._limits.max_jobs else ()

    def spawn(
        self,
        *,
        executor: Executor,
        root: Path,
        store: Callable[[bytes], Handle],
        background: bool,
        kind: JobKind,
        line: str,
    ) -> _JobCall:
        def entry(operation_id: str) -> _Running:
            out, err = self._files(operation_id)
            return _Running(
                job_id="",
                operation_id=job_operation(operation_id),
                step=operation_id,
                kind=kind,
                line=line,
                out=out,
                err=err,
                executor=executor,
                root=root,
                store=store,
            )

        return _JobCall(self._run, entry, background)

    # -- Beside -------------------------------------------------------------------------

    def running(self) -> tuple[str, ...]:
        """The open `JOB` effects, from the session's fold: what every count reads."""
        with self._session.lock:
            return self._session.state.jobs.ids()

    def started_in(self, root: Path) -> int:
        with self._lock:
            return self._started.get(root.resolve(), 0)

    def running_in(self, root: Path) -> tuple[str, ...]:
        where = root.resolve()
        return tuple(e.job_id for e in self._live() if e.root.resolve() == where)

    def stepped(self, root: Path, step: str, paths: tuple[str, ...], tree: str) -> None:
        where = root.resolve()
        with self._lock:
            # A job that has just ended is still waiting for its own step's report.
            entries = [*self._entries.values(), *self._recording.values()]
        for entry in entries:
            if entry.root.resolve() != where:
                continue
            with entry.lock:
                if entry.effect is None or entry.effect.completed:
                    continue
                if entry.step == step:
                    # Its own step: the job began where the step left the tree, after what the
                    # step put back, so a put-back is never the job's write.
                    entry.yield_tree = tree
                    entry.step_reported.set()
                else:
                    entry.stepped.append((paths, tree))

    # -- the tool's and the person's handles (`tools.context.JobsPort`) -----------------

    @property
    def inline_bytes(self) -> int:
        return self._limits.inline_bytes

    @property
    def max_wait_s(self) -> int:
        return self._limits.max_wait_s

    def known(self, job_id: str) -> bool:
        return self._view(job_id) is not None

    def is_run(self, job_id: str) -> bool:
        return self._run_id(job_id) is not None

    def run_board(self, job_id: str) -> str:
        run_id = self._run_id(job_id)
        if run_id is None:
            return f"{job_id} is not the team's run"
        if self._board is None:
            return f"run {run_id} · working in the background"
        return self._board(run_id)

    def _run_id(self, job_id: str) -> str | None:
        """The id of the session's run working now, when `job_id` names it: its row names it
        `run <id>`, so either spelling is the run."""
        working = self._team() if self._team is not None else None
        if working is None or not working.startswith("run "):
            return None
        run_id = working.removeprefix("run ")
        return run_id if job_id in (run_id, working) else None

    def listing(self) -> str:
        rows = [v.row for v in self.views()]
        working = self._team() if self._team is not None else None
        if working is not None:
            # The team's run beside the jobs; it is no job, so `kill` never ends it.
            rows.append(
                f"{working} · run · working in the background; pause stops it, the user's "
                "/drop ends it"
            )
        return "\n".join(rows) if rows else "no jobs in this session"

    def output(self, job_id: str) -> JobOutput:
        """A running job's from its files, each stream's end; an ended job's from its record,
        the files being gone."""
        view = self._view(job_id)
        if view is None:
            return JobOutput("", "", 0)
        report = view.report
        if report is not None:
            text = self._session.blob(report.kept) if report.kept is not None else b""
            return JobOutput(text.decode("utf-8", "replace"), "", report.dropped, kept=True)
        out_path, err_path = self._files(view.step)
        out, out_skipped = _read_ends(out_path, 0, TAIL_READ_BYTES)
        err, err_skipped = _read_ends(err_path, 0, TAIL_READ_BYTES)
        return JobOutput(out, err, out_skipped + err_skipped)

    def await_exit(
        self, job_id: str, timeout_s: float, cancel: threading.Event | None
    ) -> str | None:
        """The exit, worded, of a running job, or None at `timeout_s`; one that already ended
        answers from its report."""
        entry = self.entry(job_id)
        if entry is None:
            with self._lock:
                # Ended, its exit still being written: the answer is its report, once there.
                entry = self._recording.get(job_id)
        elif self.wait(job_id, timeout_s, cancel) is None:
            return None
        if entry is not None:
            # Seen, so never told: the answer is this call's, once the waiter has recorded it,
            # however long its tree takes. A stop is still a stop.
            while not entry.recorded.wait(JOB_CHECK_S):
                if cancel is not None and cancel.is_set():
                    raise Cancelled
        view = self._view(job_id)
        if view is None or view.report is None:
            return None
        return f"{job_id} {_how(view.report)}"

    def views(self) -> tuple[JobView, ...]:
        running = set(self.running())
        out: list[JobView] = []
        for effect in self._effects():
            if not effect.is_job:
                continue
            intent = JobIntent.read(effect.preconditions)
            report = None
            if effect.result is not None:
                report = _REPORT.validate_json(self._session.blob(effect.result))
            live = intent.job_id in running
            out.append(
                JobView(
                    job_id=intent.job_id,
                    operation_id=effect.operation_id,
                    step=intent.step or effect.operation_id,
                    kind=intent.kind,
                    running=live,
                    report=report,
                    state=(state := _state(live, report)),
                    row=_row(intent.job_id, intent.kind, state, report),
                )
            )
        return tuple(out)

    def _view(self, job_id: str) -> JobView | None:
        return next((v for v in self.views() if v.job_id == job_id), None)

    def entry(self, job_id: str) -> _Running | None:
        with self._lock:
            return self._entries.get(job_id)

    def kill(self, job_id: str, by: str) -> bool:
        """Ends a running job, naming who asked before the signal. Writes
        nothing: the waiter records the exit. False when no such job is running."""
        entry = self.entry(job_id)
        if entry is None or entry.started is None:
            return False
        return entry.started.job.kill(by)

    def wait(
        self, job_id: str, timeout_s: float, cancel: threading.Event | None
    ) -> JobExit | None:
        """The job's exit, or None at `timeout_s` (bounded by `zeus.job_max_wait_s`). An exit
        this wait returns is `seen`: it is never told again. Raises
        `Cancelled` on the token, the job left running. A job of no entry (ended before this
        process) answers its recorded exit through `views`, not here."""
        entry = self.entry(job_id)
        if entry is None or entry.started is None:
            return None
        bound = min(timeout_s, self._limits.max_wait_s)
        return entry.started.job.wait(bound, cancel, observe=True)

    # -- the session's open and close ---------------------------------------------------

    def reconcile(self) -> None:
        """Before the first turn: every `JOB` effect a crash left open is
        reopened; its group is reaped only when its pid's start time is the one recorded, and
        it is completed `ambiguous`, ended `lost`, with its notice when the rule says so. It
        never wakes anyone: its lines are read with the person's next turn."""
        if self._live():
            raise HarnessBug("reconcile runs before the session's first command")
        log = SessionEffects(self._session)
        for effect in self._effects():
            if not effect.is_job or effect.outcome is not None:
                continue
            intent = JobIntent.read(effect.preconditions)
            if intent.pid > 0 and intent.started and started_at(intent.pid) == intent.started:
                reap_group(intent.pgid)
            out_path, err_path = self._files(intent.step or effect.operation_id)
            whole, dropped = _kept(out_path, err_path)
            restored = () if intent.walled else self._restored(intent.step)
            report = JobReport(
                job_id=intent.job_id,
                kind=intent.kind,
                line=self._line_of(intent.step),
                exit_code=None,
                ended="lost",
                by=None,
                notify=True,
                wall_ms=None,
                wrote=(),
                overwrote=(),
                alongside=(),
                moved=False,
                git_restored=restored,
                output=None,
                kept=self._session.put_blob(whole.encode("utf-8")) if whole else None,
                dropped=dropped,
                out_bytes=_size(out_path),
                err_bytes=_size(err_path),
            )
            # Its intent is its first line; any later one is its start.
            opened = reopen(
                log, Operation(effect.operation_id, None, started=len(effect.events) > 1)
            )
            opened.complete("ambiguous", result=self._put(report))
            self._tell(report, _notice(report, tail=_tail(whole)))
        # Every job is recorded now, and none of this table's has started: what is left in the
        # folder is a crash's (a job recorded before its files went, a command cut off inside
        # its window), and its record, where it has one, holds what is kept of it.
        if self._dir.is_dir():
            for path in sorted(self._dir.iterdir()):
                unlink_durably(path)

    def close(self) -> None:
        """After the conversation's worker is joined: every running job is
        killed, `by="close"`, and its waiter completes its effect before this returns."""
        live = [e for e in self._live() if e.started is not None]
        # Each kill waits out its group's grace, so all are sent at once: the close takes one
        # grace, not one per job.
        kills = [
            threading.Thread(target=e.started.job.kill, args=("close",), name="job-close")
            for e in live
            if e.started is not None
        ]
        for kill in kills:
            kill.start()
        for kill in kills:
            kill.join()
        # And every job that ended just before, whose waiter is still writing its exit: the
        # session's writer closes after this, and an exit written then would be lost.
        with self._lock:
            ending = [
                e for i, e in self._recording.items() if i not in {x.job_id for x in live}
            ]
        for entry in (*live, *ending):
            if entry.started is not None:
                entry.started.job.join(_REAP_S)

    # -- the call -----------------------------------------------------------------------

    def _run(
        self,
        entry: _Running,
        request: ExecRequest,
        cancel: threading.Event | None,
        background: bool,
    ) -> ExecResult | Yielded:
        self._dir.mkdir(parents=True, exist_ok=True)
        with entry.lock:
            # Held across the spawn, so an exit at once waits for the entry to be whole.
            started = entry.executor.start(
                request,
                out=entry.out,
                err=entry.err,
                max_runtime_s=self._limits.max_runtime_s if entry.kind == "task" else None,
                max_log_bytes=self._limits.max_log_bytes,
                on_exit=lambda _job, exit_: self._exited(entry, exit_),
            )
            if isinstance(started, ExecResult):
                return started
            entry.started = started
        job = started.job
        window = 0 if background else self._limits.yield_s
        deadline = request.timeout_s
        try:
            if window > 0:
                exit_ = job.wait(window, cancel)
                if exit_ is not None:
                    entry.ceiling = exit_.ended == "ceiling"
                    return self._finished(entry, exit_)
            full = () if window == 0 and not background else self.full()
            if (window == 0 and not background) or full:
                # `zeus.shell_yield_s = 0`, or no room for one more: today's blocking call.
                entry.held = full
                exit_ = job.wait(max(deadline - window, 0), cancel)
                if exit_ is None:
                    ended, killed = self._reaped(job, "timeout")
                    return self._finished(entry, ended, timed_out=killed)
                entry.ceiling = exit_.ended == "ceiling"
                return self._finished(entry, exit_)
        except Cancelled:
            ended, killed = self._reaped(job, "cancel")
            return self._finished(entry, ended, cancelled=killed)
        return self._yield(entry, started, request)

    @staticmethod
    def _reaped(job: Job, by: str) -> tuple[JobExit, bool]:
        """A command its call ends inside the window: killed, and its waiter's outcome, and
        whether the kill ended it (False: it had exited on its own as the call gave up)."""
        killed = job.kill(by)
        job.join(_REAP_S)
        ended = job.outcome
        if ended is None:
            raise HarnessBug(f"job {job.pid} outlived its reap")
        return ended, killed

    def _yield(
        self, entry: _Running, started: Started, request: ExecRequest
    ) -> ExecResult | Yielded:
        job = started.job
        with entry.lock:
            done = job.outcome
            if done is not None:
                # It ended as it yielded: it is the call's, as one inside its window is.
                return self._finished(entry, done)
            with self._lock:
                entry.job_id = f"j{self._next}"
                self._next += 1
                where = entry.root.resolve()
                for other in self._entries.values():
                    if other.root.resolve() == where:
                        other.alongside.add(entry.job_id)
                        entry.alongside.add(other.job_id)
                self._entries[entry.job_id] = entry
                self._started[where] = self._started.get(where, 0) + 1
            try:
                return self._begun(entry, started, request)
            except BaseException:
                # No record of it could be written: it must not run on unrecorded, nor stay
                # listed as running beside every later step.
                with self._lock:
                    self._entries.pop(entry.job_id, None)
                job.kill("harness")
                raise

    def _begun(self, entry: _Running, started: Started, request: ExecRequest) -> Yielded:
        """The yield's record, under `entry.lock`: its tree, its hold of git's settings, and the
        `JOB` effect's intent."""
        job = started.job
        # Re-taken by its step's `stepped` once the step has put back what it does.
        tree = snapshot_operation(
            entry.root, entry.operation_id, run_id=self._session.session_id
        )
        entry.yield_tree = tree
        entry.walled = started.sandbox is not SandboxLevel.NONE and not request.unsandboxed
        entry.has_step = request.policy is PolicyClass.WORKTREE
        if not entry.walled:
            # What its exit puts back: the hold its step took before the command ran, the
            # one resume reads too; taken now only for a command with no step.
            hold = self._step_hold(entry.step)
            entry.settings = hold if hold is not None else held(entry.root)
        measured = (
            request.policy is PolicyClass.ACTION
            and started.sandbox is not SandboxLevel.NONE
            and not request.unsandboxed
        )
        intent = JobIntent(
            job_id=entry.job_id,
            kind=entry.kind,
            step=entry.step,
            pid=job.pid,
            pgid=job.pgid,
            started=job.started,
            effects_measured=measured,
            unsandboxed=request.unsandboxed,
            walled=entry.walled,
        )
        entry.effect = begin(
            SessionEffects(self._session),
            Intent(
                entry.operation_id,
                EffectKind.JOB,
                approved_sha=sha256_hex(entry.line.encode("utf-8")),
                preconditions=intent.preconditions(),
                idempotent=False,
                surface=(),
                preimages=tree,
            ),
        )
        entry.yielded = True
        return Yielded(
            entry.job_id,
            entry.kind,
            _read_ends(entry.out, KEPT_END_BYTES, KEPT_END_BYTES)[0],
            _read_ends(entry.err, KEPT_END_BYTES, KEPT_END_BYTES)[0],
            job.elapsed_ms(),
            started.sandbox,
            started.network_enforced,
        )

    def _finished(
        self,
        entry: _Running,
        exit_: JobExit,
        *,
        timed_out: bool = False,
        cancelled: bool = False,
    ) -> ExecResult:
        """A command that ended inside its window: `run`'s answer, its files then removed."""
        started = entry.started
        if started is None:
            raise HarnessBug(f"{entry.step} finished before it started")
        result = entry.executor.finish(started, exit_, timed_out=timed_out, cancelled=cancelled)
        unlink_durably(started.job.out)
        unlink_durably(started.job.err)
        return result

    # -- the exit -----------------------------------------------------------------------

    def _exited(self, entry: _Running, exit_: JobExit) -> None:
        """On the job's waiter thread, once. `recorded` is set when all it
        writes is written, however that went: a recording that raises closes the effect
        `ambiguous`, as `perform` closes one, so the job never reads as running for ever."""
        with entry.lock:
            if not entry.yielded or entry.effect is None:
                return  # the call answers it
            effect = entry.effect
            with self._lock:
                self._entries.pop(entry.job_id, None)
                self._recording[entry.job_id] = entry
        if entry.has_step:
            entry.step_reported.wait(_STEP_WAIT_S)
        try:
            told = self._record(entry, effect, exit_)
            # The record holds what is kept of the output: the files go (`KEPT_END_BYTES`).
            unlink_durably(entry.out)
            unlink_durably(entry.err)
            if told is not None:
                report, notice = told
                self._session.record_notice(notice)
                if self._wake is not None:
                    self._wake()
                # After the wake, so a hook's script never holds back the turn the exit opens.
                self._hooked(report)
        except Exception:
            if not effect.completed:
                effect.complete("ambiguous")
            raise
        finally:
            entry.recorded.set()
            with self._lock:
                self._recording.pop(entry.job_id, None)

    def _tell(self, report: JobReport, notice: Notice) -> None:
        """The job's exit notice, then what Zeus's hooks say of it."""
        self._session.record_notice(notice)
        self._hooked(report)

    def _hooked(self, report: JobReport) -> None:
        """`job_exited`: Zeus's hooks told the exit, their note a notice after it. On
        this waiter's thread; `Hooks` fires one firing at a time across threads."""
        hooks = self._hooks
        if hooks is None or not hooks.has(HookPoint.JOB_EXITED):
            return
        payload = JobExited(job=report.job_id, command=report.line, exit_code=report.exit_code)
        note = hooks.fire(HookPoint.JOB_EXITED, payload).note
        if note:
            self._session.record_notice(hook_noted(note))

    def _record(
        self, entry: _Running, effect: OpenEffect, exit_: JobExit
    ) -> tuple[JobReport, Notice] | None:
        """Completes the job's effect with its report; the report and the notice to tell,
        when it notifies."""
        with entry.lock:
            with self._lock:
                alongside = tuple(sorted(entry.alongside))
            # No wall held it (approved, or run with no sandbox), so a change
            # to what `git_guard` holds is put back now, as its step would have. A walled job
            # could not have made one, and putting back would undo the person's own.
            restored = (
                put_back(entry.settings, held(entry.root))
                if not entry.walled and entry.settings is not None
                else ()
            )
            tree, _ = current_tree(entry.root, untracked_max_bytes=None)
            wrote, overwrote = self._attributed(entry, tree)
            notify = exit_.by is None and not exit_.seen
            whole, dropped = _kept(entry.out, entry.err)
            handle = entry.store(whole.encode("utf-8"))
            report = JobReport(
                job_id=entry.job_id,
                kind=entry.kind,
                line=entry.line,
                exit_code=exit_.exit_code,
                ended=exit_.ended,
                by=exit_.by,
                notify=notify,
                wall_ms=exit_.wall_ms,
                wrote=wrote,
                overwrote=overwrote,
                alongside=alongside,
                moved=tree != entry.yield_tree,
                git_restored=restored,
                output=handle.id,
                kept=handle.content_sha,
                dropped=dropped,
                out_bytes=_size(entry.out),
                err_bytes=_size(entry.err),
            )
            outcome = (
                "succeeded" if exit_.ended == "exited" and exit_.exit_code == 0 else "failed"
            )
            effect.complete(outcome, observed=wrote, post_state=tree, result=self._put(report))
        return (report, _notice(report, tail=_tail(whole))) if notify else None

    def _attributed(
        self, entry: _Running, tree: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """What changed in its workspace that no step recorded, and what it
        wrote after a step had. Steps in another root are not counted (`stepped`); a job that
        ran beside another cannot tell their writes apart, which its `alongside` says."""
        start = entry.yield_tree
        if start is None or start == tree:
            return (), ()
        changed = set(changed_paths(entry.root, start, tree))
        claimed: dict[str, str] = {}
        for paths, after in entry.stepped:
            for p in paths:
                claimed[p] = after
        wrote = tuple(sorted(changed - claimed.keys()))
        by_tree: dict[str, set[str]] = {}
        for p, after in claimed.items():
            by_tree.setdefault(after, set()).add(p)
        overwrote: set[str] = set()
        for after, paths in by_tree.items():
            if after != tree:
                overwrote |= paths & set(changed_paths(entry.root, after, tree))
        return wrote, tuple(sorted(overwrote))

    # -- reading ------------------------------------------------------------------------

    def _effects(self) -> tuple[ShellEffect, ...]:
        with self._session.lock:
            return shell_effects(
                self._session.state, lambda e: self._session.blob(e.payload_sha)
            )

    def _line_of(self, step: str) -> str:
        """The command of the call `step` names, from the call's own line on the log: a job
        settled after a crash has no other record of it."""
        call_id = step.removeprefix(f"{self._session.session_id}:")
        with self._session.lock:
            for turn in self._session.state.turns():
                for c in turn.calls:
                    if c.call_id == call_id and isinstance(c.args.get("command"), str):
                        return str(c.args["command"])
        return ""

    def _restored(self, step: str) -> tuple[str, ...]:
        """For a job a crash cut off that no wall held: what `git_guard` holds, put back as its
        step held it before the command ran. Nothing when the step named
        no hold."""
        effect = self._step_effect(step)
        kept = self._step_hold(step)
        if effect is None or effect.root is None or kept is None:
            return ()
        return put_back(kept, held(effect.root))

    def _step_effect(self, step: str) -> ShellEffect | None:
        return next((e for e in self._effects() if e.operation_id == step), None)

    def _step_hold(self, step: str) -> Held | None:
        """What `git_guard` held before the command of the call `step` ran, as its step's intent
        names it (`zeus.writes.guarded_before`): the baseline a job's exit and resume both put
        back to. None for a command that ran as no step."""
        effect = self._step_effect(step)
        blob = None if effect is None else guarded_before(effect.preconditions)
        return None if blob is None else load(self._session.blob(blob))

    def _files(self, step: str) -> tuple[Path, Path]:
        """The stdout and stderr files of the command the call `step` ran."""
        stem = operation_component(step)
        return self._dir / f"{stem}.out", self._dir / f"{stem}.err"

    def _live(self) -> list[_Running]:
        with self._lock:
            return list(self._entries.values())

    def _put(self, report: JobReport) -> str:
        return self._session.put_blob(canonical_json(_REPORT.dump_python(report, mode="json")))


def exit_unread(state: SessionState, read_before: int) -> bool:
    """Whether a job's exit notice is on the log at or after `read_before`, the first seq the
    last request Zeus's model answered did not carry: one he has not read,
    which a wake turn would open with. A later answer is not proof: a notice written while
    its request was in flight follows the request and precedes the answer."""
    return any(
        t.seq >= read_before and t.notice is not None and _wakes(t.notice)
        for t in state.turns()
    )


def _wakes(notice: Notice) -> bool:
    """The notices that open a wake turn: a job's exit; a run's end, whose
    `handle` names the run; and the team's reply that needs Zeus, whose `handle` names the tell.
    A pause is said on trigger 4 with no run named, and a reply that does not need him
    on trigger 10 with no tell named: neither wakes anybody."""
    if notice.trigger is NoticeTrigger.JOB_EXITED:
        return True
    named = notice.trigger in (NoticeTrigger.RUN_ENDED, NoticeTrigger.TEAM_REPLIED)
    return named and notice.handle is not None


def wake_ready(session: Session, ledger: MayStart, read_before: Callable[[], int]) -> bool:
    """The gate's half that reads the log: an exit notice Zeus has not
    read, and a ledger that may start a turn. `read_before` is the loop's (`ZeusLoop`)."""
    with session.lock:
        unread = exit_unread(session.state, read_before())
    return unread and ledger.may_start()


class MayStart(Protocol):
    def may_start(self) -> bool: ...


def _notice(report: JobReport, *, tail: str) -> Notice:
    """Trigger 8 for a job's report: its exit as `_how` words it, and what it touched."""
    return job_exited(
        job_id=report.job_id,
        kind=report.kind,
        line=report.line,
        how=_how(report),
        moved=report.moved,
        wrote=report.wrote,
        overwrote=report.overwrote,
        alongside=report.alongside,
        git_restored=report.git_restored,
        tail=tail,
        handle=report.output,
        dropped=report.dropped,
    )


def _state(running: bool, report: JobReport | None) -> str:
    return "running" if running else _how(report) if report is not None else "ended"


def _row(job_id: str, kind: JobKind, state: str, report: JobReport | None) -> str:
    line = f" · {report.line}" if report is not None and report.line else ""
    return f"{job_id} · {kind} · {state}{line}"


def _kept(out: Path, err: Path) -> tuple[str, int]:
    """What a job keeps of its output once it has ended: each stream's two ends
    (`KEPT_END_BYTES`), joined, and the bytes dropped from their middles."""
    out_text, out_dropped = _read_ends(out, KEPT_END_BYTES, KEPT_END_BYTES)
    err_text, err_dropped = _read_ends(err, KEPT_END_BYTES, KEPT_END_BYTES)
    return joined_output(out_text, err_text), out_dropped + err_dropped


def _read_ends(path: Path, head: int, tail: int) -> tuple[str, int]:
    """`path`'s first `head` and last `tail` bytes, each cut to whole lines, and how many bytes
    between them are not read: the whole file when it fits. A file read from both ends marks
    the gap; one read from its end only (`head` 0) does not, its reader says so."""
    try:
        with path.open("rb") as f:
            size = f.seek(0, 2)
            f.seek(0)
            if size <= head + tail:
                return f.read().decode("utf-8", "replace"), 0
            first = f.read(head)
            f.seek(size - tail)
            last = f.read()
    except OSError:
        return "", 0
    if b"\n" in first:
        first = first[: first.rfind(b"\n") + 1]
    last = last[last.find(b"\n") + 1 :]
    dropped = size - len(first) - len(last)
    gap = f"[... {dropped} bytes not kept ...]\n" if head else ""
    return first.decode("utf-8", "replace") + gap + last.decode("utf-8", "replace"), dropped


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _tail(text: str) -> str:
    return "".join(split_lines(text, keepends=True)[-TAIL_LINES:])


def _how(report: JobReport) -> str:
    took = f" after {_duration(report.wall_ms)}" if report.wall_ms is not None else ""
    if report.ended == "lost":
        return "lost: the harness stopped while it ran, so its end was not seen"
    if report.ended == "timed_out":
        return f"reaped at zeus.job_max_runtime_s{took}"
    if report.ended == "ceiling":
        return f"killed past zeus.job_max_log_bytes{took}"
    if report.ended == "killed":
        return f"killed by {report.by}{took}"
    code = report.exit_code
    if code is not None and code < 0:
        return f"ended by signal {-code}{took}"
    return f"exited {code}{took}"


def _duration(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"
