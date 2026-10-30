"""The box over real parts: a committed python repository, its indexed graph, the box's edit
port and shell (on a scripted supervisor, so a command runs nothing and answers what the test
says), the session's ledger, the real workspace lease, and scripted Athena and worker rungs:
what a model would supply is the test's script.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from repos import init_repo
from sandboxes import Enforcing, Scripted

from chickadee.approval.stops import WorkspacePreview
from chickadee.box import athena as box_athena
from chickadee.box.compose import EpisodeSetting, RoleSetting, RunParts
from chickadee.box.episode import Box, BoxEnv
from chickadee.box.loop import HeadParts
from chickadee.box.request import ApprovedDesign, CoderRequest, OneTask
from chickadee.box.resume import never_asking
from chickadee.box.rung import BoxRung
from chickadee.box.shell import shell_for
from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable
from chickadee.config import RungEndpoint, WorkspacePolicy
from chickadee.context.terms import Vision
from chickadee.engine import NonInteractive
from chickadee.executor.executor import Executor
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.hooks.bind import HookConfig
from chickadee.hooks.builtins import default_set
from chickadee.hooks.fire import HookLimits, Hooks, HookSet, RoleHooks
from chickadee.hooks.sinks import RunHookSink
from chickadee.hooks.spec import HookSpec
from chickadee.init.detect import resolve
from chickadee.init.lease import LeaseBusy, RepoLease
from chickadee.journal import paths
from chickadee.journal.port import JournalPort
from chickadee.process.supervisor import Completed
from chickadee.profiles.loader import load_profile
from chickadee.roles import ATHENA_TASK, DEFAULT_RUNGS, TASK_WORKER
from chickadee.schema import RetryParams, sha256_hex
from chickadee.schema.interaction import InteractionPort
from chickadee.schema.journal import EventKind
from chickadee.scout.search import Search
from chickadee.secrets.denylist import Denylist
from chickadee.state.roots import roots as resolve_roots
from chickadee.tools.policy import RolePolicy
from chickadee.workspace import Workspace, acquire, reattach

APP = "def go() -> int:\n    return 1\n"

DESIGN = "# Make go return 2\n\n## What changes\n- `go()` in `app.py` returns 2.\n"
TEMPLATE_SHA = sha256_hex("the template")


def make_repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {"app.py": APP, "tests/test_app.py": "def test_go():\n    assert True\n"},
    )


def design(text: str = DESIGN, workspace: str = "worktree") -> CoderRequest:
    """A design as approved at stop 1."""
    approved = ApprovedDesign(text=text, sha=sha256_hex(text), template_sha=TEMPLATE_SHA)
    return CoderRequest.model_validate({"design": approved, "workspace": workspace})


def one_task(title: str, brief: str, workspace: str = "worktree") -> CoderRequest:
    """A one-task plan as approved at stop 2."""
    task = OneTask(title=title, brief=brief, sha=sha256_hex(brief), template_sha=TEMPLATE_SHA)
    return CoderRequest.model_validate({"task": task, "workspace": workspace})


def lease_is_free(source: Path) -> bool:
    identity = resolve(source)
    try:
        lease = RepoLease.acquire(identity.git_common_dir, identity.repository_id)
    except LeaseBusy:
        return False
    lease.release()
    return True


def read_rows(role: str) -> dict[str, int]:
    """`role`'s read rows of the limits register at their defaults: what the
    episode resolves for it with no config."""
    from chickadee.limits import LIMITS

    suffixes = (
        "whole_file_bytes",
        "read_bytes",
        "read_lines",
        "result_max",
        "raise_share",
        "raise_strikes",
        "held_turns",
        "held_tokens",
        "insist_showings",
    )
    return {
        lim.name: int(lim.value)
        for lim in LIMITS
        if lim.applies_to == role and lim.name.split(".", 1)[1] in suffixes
    }


WORKER_LIMITS = {
    **read_rows("worker"),
    "worker.repeat": 2,
    "worker.print_only": 1,
    "worker.max_tool_calls": 40,
    "worker.result_max": 8000,
}


def _endpoint(model: str, window: int, output: int) -> RungEndpoint:
    return RungEndpoint(
        base_url="http://127.0.0.1:1",
        model_id=model,
        context_window=window,
        max_output_tokens=output,
        native_tools=True,
    )


@dataclass
class BoxRig:
    """One session's box over `source`, its models scripted. Records every workspace it leased
    and every index it opened, and closes the indexes when the test does. `stdout` and
    `exit_code` are what every command answers."""

    source: Path
    athena_script: Sequence[object] = ()
    worker_script: Sequence[object] = ()
    interaction: InteractionPort = field(default_factory=NonInteractive)
    ledger: CostLedger = field(
        default_factory=lambda: CostLedger(
            PriceTable({}), BudgetCaps(hard_cost_cap_usd=10.0), rungs=DEFAULT_RUNGS
        )
    )
    stdout: bytes = b"1 passed\n"
    exit_code: int = 0
    parts_hook: Callable[[Workspace], None] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    """CL-10's first interrupt: set, and the run pauses at its next task boundary."""
    abort: threading.Event = field(default_factory=threading.Event)
    suspend: threading.Event = field(default_factory=threading.Event)
    """The pause as the process leaves, set with `abort`."""
    authority: Any = None
    """What a command above T0 is put to; None refuses it."""
    asking: Callable[[str], bool] | None = None
    """Whether a step's permission question was never answered (the session log's in an
    engine); None, never."""
    workspaces: list[Workspace] = field(default_factory=list[Workspace])
    indexers: list[Indexer] = field(default_factory=list[Indexer])
    rungs: dict[str, Any] = field(default_factory=dict[str, Any])
    """The scripted rungs each episode made, by role: what each was asked."""
    worker_policy: RolePolicy = TASK_WORKER
    """The worker's tool policy, as a config would resolve it."""

    def box(self) -> Box:
        return Box(
            BoxEnv(
                workspace_for=self._workspace,
                reattach=self._reattach,
                parts_for=self.parts_for,
                setting=self.setting,
                cancel=self.cancel,
                abort=self.abort,
                suspend=self.suspend,
                asking=self.asking if self.asking is not None else never_asking,
            )
        )

    def close(self) -> None:
        for indexer in self.indexers:
            indexer.close()

    def setting(self) -> EpisodeSetting:
        from engines import ScriptedRung

        worker = ScriptedRung(self.worker_script)  # type: ignore[arg-type]
        athena = ScriptedRung(self.athena_script)  # type: ignore[arg-type]
        self.rungs = {"worker": worker, "athena": athena}
        return EpisodeSetting(
            worker=RoleSetting(
                rung=BoxRung(worker, _endpoint("low-x", 100_000, 4_000), "low-x", "low-param"),  # type: ignore[arg-type]
                policy=self.worker_policy,
                limits=WORKER_LIMITS,
                vision=Vision("low-param"),
                params=RetryParams(temperature=0.2, max_tokens=4000),
                budget=90_000,
                keep_recent=4,
            ),
            athena=RoleSetting(
                rung=BoxRung(
                    athena, _endpoint("high-x", 200_000, 8_000), "high-x", "high-param"
                ),  # type: ignore[arg-type]
                policy=ATHENA_TASK,
                limits=ATHENA_LIMITS,
                vision=Vision("high-param"),
                params=RetryParams(temperature=0.2, max_tokens=8000),
                budget=180_000,
                keep_recent=4,
            ),
            head=HeadParts(box_athena.instructions(), "PLAN TEMPLATE", "BRIEF TEMPLATE"),
            brief_template_sha=TEMPLATE_SHA,
            workspace=WorkspacePreview("worktree", "main"),
            untracked_max_bytes=1 << 20,
            task_seconds=1800,
            recut_cap=2,
            one_task_max=3,
            design_layer=4000,
            note_cap=500,
        )

    def _workspace(self, run_id: str, request: CoderRequest) -> Workspace:
        ws = acquire(
            self.source,
            run_id=run_id,
            goal=request.goal,
            policy=WorkspacePolicy(mode=request.workspace),
        )
        self.workspaces.append(ws)
        return ws

    def _reattach(self, run_id: str) -> Workspace:
        ws = reattach(self.source, run_id=run_id)
        self.workspaces.append(ws)
        return ws

    def parts_for(self, ws: Workspace) -> RunParts:
        if self.parts_hook is not None:
            self.parts_hook(ws)
        indexer = Indexer(ws.root)
        indexer.full_index()
        self.indexers.append(indexer)

        def executor(journal: JournalPort) -> Executor:
            supervisor = Scripted(self.stdout, self.exit_code)
            return Executor(Enforcing(), put_blob=journal.put_blob, supervisor=supervisor)

        return RunParts(
            worktree=ws.root,
            profiles=(load_profile("python"),),
            denylist=Denylist(),
            graph=Graph(ws.root, indexer.conn),
            indexer=indexer,
            budget=self.ledger,
            interaction=self.interaction,
            shell_for=shell_for(
                ws.root, Enforcing(), executor, self.authority, suspend=self.suspend
            ),
            search=Search(root=ws.root),
        )


def with_hooks(rig: BoxRig, *specs: HookSpec, limits: HookLimits | None = None) -> None:
    """`rig`'s episodes configured with `specs` as their `[[hooks]]`, bound when each
    run starts, as the config's are."""
    setting = rig.setting

    def configured() -> EpisodeSetting:
        return replace(setting(), hooks=HookConfig(specs, limits or HookLimits()))

    rig.setting = configured  # type: ignore[method-assign]


def journal_text(ws: Workspace, run_id: str) -> str:
    """The run's journal as written, for asserting what it recorded."""
    return paths.journal_path(resolve_roots(ws.root), run_id).read_text(encoding="utf-8")


def journal_exists(ws: Workspace, run_id: str) -> bool:
    return paths.run_dir(resolve_roots(ws.root), run_id).exists()


class FakeRunJournal:
    """The run journal as the shell reads it: lines appended in order, blobs by sha."""

    run_id = "run-1"

    def __init__(self) -> None:
        self.lines: list[tuple[EventKind, Any, str | None, int | None]] = []
        self.blobs: dict[str, bytes] = {}

    def append(
        self,
        kind: EventKind,
        payload: Any,
        *,
        task_id: str | None = None,
        round: int | None = None,
    ) -> int:
        self.lines.append((kind, payload, task_id, round))
        return len(self.lines)

    def put_blob(self, content: bytes) -> str:
        sha = sha256_hex(content)
        self.blobs[sha] = content
        return sha

    def kinds(self) -> list[EventKind]:
        return [kind for kind, _, _, _ in self.lines]


# -- the task worker's rig ---------------------------------------------------------


class KilledMidCall(BaseException):
    """The process dying: not an `Exception`, so nothing completes what was in flight."""


class CrashingCommand(Scripted):
    """A supervisor whose command writes the worktree (unless `writes` is False), then the
    process dies."""

    def __init__(self, root: Path, *, writes: bool = True) -> None:
        super().__init__()
        self._root = root
        self._writes = writes

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Completed:
        if self._writes:
            (self._root / "app.py").write_text("half written\n", encoding="utf-8")
            (self._root / "made.txt").write_text("made\n", encoding="utf-8")
        raise KilledMidCall


WORKER_RUN = "2026-10-15-0a1b2c3d"
WORK_DIR = "docs/chickadee"
"""`[work] dir`'s default: the folder the rigs' edit ports refuse, as composed."""


def worker_call(tool: str, cid: str, /, **args: object) -> Any:
    """A call as the worker sends it: its arguments, no purpose line."""
    from chickadee.schema import ToolCall

    return ToolCall(id=cid, name=tool, arguments=dict(args))


def worker_rig(
    root: Path,
    script: Sequence[object],
    *,
    stdout: bytes = b"1 passed\n",
    exit_code: int = 0,
    limits: dict[str, int] | None = None,
    task_seconds: int = 1800,
    cancel: threading.Event | None = None,
    fixed: str | None = "You are the worker.",
    graph: Any = None,
    indexer: Any = None,
    supervisor: Any = None,
    resume: bool = False,
    journal: Any = None,
    authority: Any = None,
    policy: Any = None,
    python_calls: bool = False,
    hooks: HookSet | None = None,
    tells: Any = None,
) -> tuple[Any, Any, Any]:
    """A task worker over `root`: a real run journal, the box's edit port and shell (on a
    scripted supervisor, so a command runs nothing and answers `stdout`), and a scripted model.
    `supervisor` replaces the scripted one; `resume` reopens the run's journal rather than
    creating it, and `journal` is one already open (an episode's, `episode_rig`); `policy`
    replaces the worker's `TASK_WORKER`; `python_calls` sets its `calling = "python"`,
    its blocks run in a plain child, since no sandbox runs on every test machine; `hooks`, the
    set its role holds, as `compose` gives it; `tells`, a factory of the run's `Tells`
    from its journal. Returns the worker, its journal and the scripted rung."""
    from engines import ScriptedRung
    from sandboxes import Enforcing, Scripted

    from chickadee.box.compose import _reader  # pyright: ignore[reportPrivateUsage]
    from chickadee.box.edits import WORKER_FENCE, WorkFence, WorktreeEdits
    from chickadee.box.rung import BoxRung
    from chickadee.box.shell import BoxShell
    from chickadee.box.syntax import SyntaxChecker
    from chickadee.box.worker import WorkerParts, WorkerRun
    from chickadee.config import RungEndpoint
    from chickadee.context.terms import ReadRules, Vision
    from chickadee.journal.writer import Journal
    from chickadee.roles import TASK_WORKER
    from chickadee.schema import RetryParams
    from chickadee.tools.context import ToolContext
    from chickadee.zeus.tools.graphnav import GraphNavigator

    roots = resolve_roots(root)
    if journal is None:
        journal = (
            Journal.open_for_resume(roots, WORKER_RUN)
            if resume
            else Journal.create(roots, WORKER_RUN, {})
        )
    python = load_profile("python")
    executor = Executor(
        Enforcing(),
        put_blob=journal.put_blob,
        supervisor=supervisor if supervisor is not None else Scripted(stdout, exit_code),
    )
    rung = ScriptedRung(script)  # type: ignore[arg-type]
    # The shell names the worker's current task on every claim; the worker is made after it.
    made: list[WorkerRun] = []
    shell = BoxShell(
        executor,
        root,
        journal,
        authority,
        asker="worker",
        task=lambda: made[0].task if made else None,
    )
    edits = WorktreeEdits(
        root,
        Denylist(),
        SyntaxChecker(root, (python,)),
        untracked_max_bytes=1 << 20,
        work=WorkFence(WORK_DIR, WORKER_FENCE),
    )
    # The reader the episode composes (`compose._reader`), so a read here is the worker's.
    worker_limits = limits if limits is not None else WORKER_LIMITS
    reader = _reader(
        root,
        Denylist(),
        Vision("low-param"),
        journal,
        None,
        rules=ReadRules.of("worker", {**read_rows("worker"), **worker_limits}),
        now=lambda: made[0].now() if made else None,
    )
    endpoint = RungEndpoint(
        base_url="http://127.0.0.1:1",
        model_id="low-x",
        context_window=100_000,
        max_output_tokens=4_000,
        native_tools=True,
    )
    chosen = TASK_WORKER if policy is None else policy
    if python_calls:
        from dataclasses import replace

        chosen = replace(chosen, calling="python")
    worker = WorkerRun(
        BoxRung(rung, endpoint, "low-x", "low-param"),  # type: ignore[arg-type]
        chosen,
        ToolContext(
            role="worker",
            files=reader,
            handles=reader,
            navigator=GraphNavigator(None, (python,)),
            search=Search(root=root),
            edits=edits,
            exec=shell,
            hooks=None
            if hooks is None
            else RoleHooks(
                Hooks(hooks, RunHookSink(journal)),
                "worker",
                "low-x",
                run_id=journal.run_id,
                place=lambda: (made[0].task, made[0].round) if made else (None, None),
            ),
        ),
        journal,
        worker_limits,
        WorkerParts(
            root=root,
            profiles=(python,),
            graph=graph,
            indexer=indexer,
            params=RetryParams(temperature=0.2, max_tokens=4000),
            budget=90_000,
            keep_recent=4,
            untracked_max_bytes=1 << 20,
            task_seconds=task_seconds,
            syntax=SyntaxChecker(root, (python,)),
            work_dir=WORK_DIR,
            python_unsandboxed=python_calls,
        ),
        cancel=cancel,
        fixed=fixed,
        tells=None if tells is None else tells(journal),
    )
    made.append(worker)
    return worker, journal, rung


def delegate(journal: Any, root: Path, task: str = "T1", brief: str = "Do it.") -> str:
    """`task` delegated by Athena from `root`'s tree now, as `T-3.156`'s `delegate` writes it.
    Returns its start tree."""
    from chickadee.schema.journal import TaskDelegatedP
    from chickadee.workspace import current_tree

    start, _ = current_tree(root, untracked_max_bytes=None)
    journal.append(
        EventKind.TASK_DELEGATED,
        TaskDelegatedP(
            task_id=task,
            call_id=f"d-{task}",
            title=f"task {task}",
            brief_sha=journal.put_blob(brief.encode()),
            by="athena",
            start_tree=start,
        ),
        task_id=task,
    )
    return start


# -- an episode's rig: the desk and Athena over the worker -------------------------

ATHENA_LIMITS = {
    **read_rows("athena"),
    "athena.max_tool_calls": 100,
    "athena.max_cumulative_result_tokens": 100_000,
    "athena.repeat": 2,
}


def athena_call(tool: str, cid: str, /, **args: object) -> Any:
    """A call as Athena sends it: its arguments, no purpose line."""
    from chickadee.schema import ToolCall

    return ToolCall(id=cid, name=tool, arguments=dict(args))


def _sha(text: str) -> str:
    from chickadee.schema.ids import sha256_hex

    return sha256_hex(text)


def approve_all(stop: Any) -> Any:
    """A stop the person approves as shown."""
    from chickadee.approval.stops import StopOutcome
    from chickadee.schema.interaction import StopAnswer

    return StopOutcome(StopAnswer.APPROVE, stop.text, stop.sha)


@dataclass
class EpisodeRig:
    """The desk, Athena and the worker over one repository and one run journal, each model
    scripted. `stops` records each stop shown."""

    root: Path
    journal: Any
    worker: Any
    worker_rung: Any
    desk: Any
    athena: Any
    athena_rung: Any
    indexer: Any
    stops: list[Any] = field(default_factory=list[Any])


def episode_rig(
    root: Path,
    athena_script: Sequence[object],
    worker_script: Sequence[object],
    *,
    stop: Callable[[Any], Any] = approve_all,
    in_place: bool = False,
    recut_cap: int = 2,
    one_task_max: int = 3,
    resume: bool = False,
    athena_limits: dict[str, int] | None = None,
    stdout: bytes = b"1 passed\n",
    exit_code: int = 0,
    supervisor: Any = None,
    authority: Any = None,
    pause: threading.Event | None = None,
    asking: Callable[[str], bool] | None = None,
    symbols: Callable[[], Any] | None = None,
    policy: Any = None,
    work: Any = None,
    tells: Callable[[Any], Any] | None = None,
) -> EpisodeRig:
    """An episode over `root`: a real run journal, the worker of `worker_rig` over it, the
    commit on `root`'s git, the desk asking stop 2 through `stop`, and Athena on a scripted
    rung with the task policy, her edit port and shell the box's, both shells asking
    `authority` above T0. `resume` reopens the run's journal, as a restart does. `work` is the
    session's open work (`WorkLine`), whose `plan/` Athena's `doc` writes. `tells` makes the
    run's `Tells` from its journal, shared by the worker, the desk and Athena as `compose`
    shares it."""
    from engines import ScriptedRung
    from indexers import RecordingIndexer
    from sandboxes import Enforcing, Scripted

    from chickadee.approval.stops import WorkspacePreview
    from chickadee.box.athena import Athena, AthenaParts
    from chickadee.box.commit import Projector
    from chickadee.box.compose import _reader  # pyright: ignore[reportPrivateUsage]
    from chickadee.box.desk import DeskParts, TaskDesk
    from chickadee.box.edits import ATHENA_FENCE, PlanDocs, WorkFence, WorkRule, WorktreeEdits
    from chickadee.box.rung import BoxRung
    from chickadee.box.shell import BoxShell
    from chickadee.box.syntax import SyntaxChecker
    from chickadee.config import RungEndpoint
    from chickadee.context.terms import ReadRules, Vision
    from chickadee.journal.writer import Journal
    from chickadee.roles import ATHENA_TASK
    from chickadee.schema import RetryParams
    from chickadee.tools.context import ToolContext
    from chickadee.zeus.tools.graphnav import GraphNavigator

    roots = resolve_roots(root)
    journal = (
        Journal.open_for_resume(roots, WORKER_RUN)
        if resume
        else Journal.create(roots, WORKER_RUN, {})
    )
    indexer = RecordingIndexer()
    told = None if tells is None else tells(journal)
    worker, _, worker_rung = worker_rig(
        root,
        worker_script,
        journal=journal,
        indexer=indexer,
        stdout=stdout,
        exit_code=exit_code,
        supervisor=supervisor,
        authority=authority,
        tells=None if told is None else lambda _journal: told,
    )
    shown: list[Any] = []

    scripted = stop

    def ask(stop: Any, *, reread: Any = None) -> Any:
        # A folder stop: approved as the folder stands after the scripted answer ran, as
        # `ask_stop` approves it, its tree the version.
        shown.append(stop)
        out = scripted(stop)
        if reread is None or out.answer.value != "approve":
            return out
        from dataclasses import replace as _replace

        now = reread()
        return _replace(out, text=now.text, sha=_sha(now.text), version=now.version)

    python = load_profile("python")
    projector = Projector(root, indexer, in_place=in_place, run_id=WORKER_RUN)
    # The episode's hooks as `compose_episode` binds them where nothing is configured: the
    # built-in defaults over the run's journal, `owed_review` among them.
    hooks = Hooks(default_set(), RunHookSink(journal))
    desk = TaskDesk(
        journal,
        worker,
        projector,
        ask,
        DeskParts(
            root=root,
            workspace=WorkspacePreview("in_place" if in_place else "worktree", "main"),
            profiles=(python,),
            untracked_max_bytes=1 << 20,
            recut_cap=recut_cap,
            one_task_max=one_task_max,
            work_dir=WORK_DIR,
            records=WorktreeEdits(
                root,
                Denylist(),
                SyntaxChecker(root, (python,)),
                edit_check="off",
                untracked_max_bytes=1 << 20,
            ),
        ),
        indexer=indexer,
        pause=pause,
        asking=asking if asking is not None else never_asking,
        symbols=symbols,
        hooks=RoleHooks(hooks, "athena", "high-x", run_id=WORKER_RUN),
        tells=told,
    )
    executor = Executor(
        Enforcing(),
        put_blob=journal.put_blob,
        supervisor=supervisor if supervisor is not None else Scripted(stdout, exit_code),
    )
    shell = BoxShell(executor, root, journal, authority, asker="athena", task=lambda: desk.held)

    def port(fence: WorkRule | None) -> Any:
        return WorktreeEdits(
            root,
            Denylist(),
            SyntaxChecker(root, (python,)),
            untracked_max_bytes=1 << 20,
            work=None if fence is None else WorkFence(WORK_DIR, fence),
        )

    edits = port(ATHENA_FENCE)
    her_limits = athena_limits if athena_limits is not None else ATHENA_LIMITS
    hers: list[Any] = []
    reader = _reader(
        root,
        Denylist(),
        Vision("high-param"),
        journal,
        None,
        rules=ReadRules.of("athena", {**read_rows("athena"), **her_limits}),
        now=lambda: hers[0].now() if hers else None,
    )
    endpoint = RungEndpoint(
        base_url="http://127.0.0.1:1",
        model_id="high-x",
        context_window=200_000,
        max_output_tokens=8_000,
        native_tools=True,
    )
    athena_rung = ScriptedRung(athena_script)  # type: ignore[arg-type]
    athena = Athena(
        BoxRung(athena_rung, endpoint, "high-x", "high-param"),  # type: ignore[arg-type]
        policy if policy is not None else ATHENA_TASK,
        ToolContext(
            role="athena",
            files=reader,
            handles=reader,
            navigator=GraphNavigator(None, (python,)),
            edits=edits,
            exec=shell,
            docs=PlanDocs(root, port(None), work),
            hooks=RoleHooks(
                hooks, "athena", "high-x", run_id=WORKER_RUN, place=lambda: (desk.held, None)
            ),
        ),
        journal,
        her_limits,
        desk,
        AthenaParts(
            params=RetryParams(temperature=0.2, max_tokens=8000), budget=180_000, keep_recent=4
        ),
        cancel=None,
        tells=told,
    )
    hers.append(athena)
    return EpisodeRig(
        root, journal, worker, worker_rung, desk, athena, athena_rung, indexer, shown
    )
