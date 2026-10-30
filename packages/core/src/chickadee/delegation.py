"""The engine's delegation: the work's stop at dispatch, one box episode at a time, and the
episode's composition.

The engine is composition only, so what composing an episode takes lives here:

- **the stop at dispatch** (`stop_at_dispatch`): a design is shown at stop 1, a one-task plan's
  brief at stop 2, both before any lease. What the person approved is the work the
  episode runs (invariant 15); a note returns to Zeus as the call's result and a cancel
  ends the call, each the person's constraint in their own words. The router's tier,
  the consent question and the D187 gate retired with the work tier;
- **which languages judge the work**: every profile the checkout holds, each file judged by its
  own;
- **one episode at a time** (`OneEpisode`): a second `coder()` while one is in flight is
  `EngineBusy`, naming the run. The cross-process lease stays the workspace's;
- **the episode** (`BoxEpisode`): the box's `BoxEnv` from the engine's values, and the run on a
  thread of its own, started lazily, so `coder()` names the run and does nothing else (item
  10). Its parts (the graph's connection, the model clients) are made, used and closed on that
  thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from chickadee.adapters.base import first_attempt_params
from chickadee.approval.commands import CommandTable
from chickadee.approval.stops import (
    Reread,
    StopOutcome,
    StopWithdrawn,
    WorkspacePreview,
    ask_stop,
    stop_of,
)
from chickadee.approval.tier import Preferences
from chickadee.box import athena as box_athena
from chickadee.box.compose import EpisodeSetting, RoleSetting, RunParts
from chickadee.box.episode import Box, BoxEnv, BoxRun, Paused, opened_with
from chickadee.box.loop import HeadParts
from chickadee.box.report import CoderReport
from chickadee.box.request import ApprovedDesign, CoderRequest, OneTask, WorkPin
from chickadee.box.rung import BoxRung
from chickadee.box.shell import shell_for
from chickadee.box.spawn import HarnessSpawn, index_resolver, run_argv
from chickadee.box.steps import Beside
from chickadee.box.tells import Told
from chickadee.boxctx.athena import NumberedLog
from chickadee.budget.manager import CostLedger
from chickadee.client import client_for
from chickadee.client.live import LiveSink
from chickadee.config import EngineConfig, WorkspacePolicy
from chickadee.context.attachment import AttachCaps
from chickadee.context.questions import for_run
from chickadee.context.view import (
    held_worktree,
    latest_workspace,
    run_worktree,
    still_written,
)
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.errors import EngineBusy, HarnessBug
from chickadee.executor.executor import Executor
from chickadee.executor.types import ExecResult
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.graph.servers import EpisodeServers
from chickadee.graph.store import index_exists
from chickadee.handoffs import resolve as resolve_template
from chickadee.hooks.bind import HookConfig
from chickadee.hooks.script import sealed_runner
from chickadee.init.detect import detect_profiles, repository_key
from chickadee.isolation.base import SandboxBackend, select
from chickadee.isolation.types import tool_cache
from chickadee.journal.port import JournalPort
from chickadee.journal.projection import RunProjection
from chickadee.journal.reader import follow
from chickadee.limits import resolved_value, role_limits, window
from chickadee.process.supervisor import ProcessSupervisor, bind_session
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile
from chickadee.schema import JournalEvent
from chickadee.schema.interaction import InteractionPort, StopAnswer, StopKind
from chickadee.schema.isolation import WorkspaceMode
from chickadee.schema.journal import TellOutcome
from chickadee.scout.search import Search
from chickadee.secrets.denylist import Denylist
from chickadee.secrets.denylist import resolve as resolve_denylist
from chickadee.secrets.walk_memo import WalkMemo
from chickadee.state.roots import roots as resolve_roots
from chickadee.work import design_errors, part_dir
from chickadee.workspace import (
    SnapshotSite,
    Workspace,
    WorktreeGone,
    abandon_worktree,
    acquire,
    commit_of,
    folder_tree,
    head_and_branch,
    pin_tree,
    reattach,
    reattach_worktree,
)
from chickadee.zeus.delegate import Delegation
from chickadee.zeus.tools.talk import PersonAuthority

if TYPE_CHECKING:
    from chickadee.context.session import SessionState

__all__ = [
    "BoxEpisode",
    "EpisodeInputs",
    "EpisodeLog",
    "OneEpisode",
    "Stopped",
    "chosen_workspace",
    "open_episode",
    "profiles_for",
    "resume_episode",
    "session_worktree",
    "stop_at_dispatch",
    "stop_at_work",
    "workspace_preview",
]

TAIL_POLL_S: Final = 0.05
"""How often `BoxEpisode.events()` looks for new journal lines while the run is going."""


# -- the languages ----------------------------------------------------------------------------


def profiles_for(root: Path) -> tuple[LanguageProfile, ...]:
    """Every installed profile the checkout holds files of: each file
    an episode writes is judged by its own language's. None detected judges nothing: a
    written file is then `syntax unchecked`, never refused."""
    profiles = ProfileSet.load(repo=root)
    evidence, _gaps = detect_profiles(root, profiles)
    return tuple(p for p in (profiles.by_name(e.name) for e in evidence) if p is not None)


# -- the stop at dispatch ---------------------------------------------------------------------


def workspace_preview(root: Path, mode: WorkspaceMode) -> WorkspacePreview:
    """Where the run would start: the checkout's branch, or its commit when detached."""
    head, branch = head_and_branch(root)
    return WorkspacePreview(mode=mode, source_ref=branch or head)


@dataclass(frozen=True)
class Stopped:
    """The work's stop ended without an approval: nothing runs, and `report` says why."""

    report: CoderReport
    outcome: StopOutcome | None
    """The person's answer at the stop; None when the stop was never asked, because the work
    could not start whatever the answer (a work folder refused for its shape)."""


def chosen_workspace(
    asked: WorkspaceMode | None,
    *,
    folder_in_worktree: bool | None,
    default: WorkspaceMode,
) -> WorkspaceMode:
    """Where a `coder` call runs (Amendment 4), the one home of the rule: Zeus's `workspace`
    when it sent one; else where the
    design's work folder is (`folder_in_worktree`, None for a one-task plan), since its plan
    and records name the folder there; else `default`, where the session
    was last put (`context/view.session_mode`)."""
    if asked is not None:
        return asked
    if folder_in_worktree is not None:
        return "worktree" if folder_in_worktree else "in_place"
    return default


def stop_at_dispatch(
    delegation: Delegation,
    port: InteractionPort,
    *,
    root: Path,
    templates: dict[str, Path],
    workspace: WorkspaceMode,
    cancel: threading.Event | None = None,
    notice: str | None = None,
) -> CoderRequest | Stopped:
    """The work's stop, asked before anything is leased: a design at stop 1, a
    one-task plan's brief at stop 2, by Zeus. `workspace` is where it runs, chosen at dispatch
    (`chosen_workspace`). Approved, the request the episode runs, with the
    text the person approved, after any edit, and its sha (invariant 15). A note ends the call
    `ASK`, the note verbatim, for Zeus to revise by; a cancel ends it `ABORTED`, as does
    `cancel`, Zeus's turn stopped hard while it waits."""
    kind = StopKind.DESIGN if delegation.kind == "design" else StopKind.PLAN
    try:
        said = ask_stop(
            stop_of(
                kind,
                "zeus",
                delegation.text,
                workspace_preview(root, workspace),
                notice=notice,
            ),
            port,
            cancel=cancel,
        )
    except StopWithdrawn:
        return _withdrawn(kind)
    stopped = _not_approved(said, kind)
    if stopped is not None:
        return stopped
    if delegation.kind == "design":
        template = resolve_template(root, templates, "design")
        design = ApprovedDesign(text=said.text, sha=said.sha, template_sha=template.sha)
        return CoderRequest(design=design, workspace=workspace)
    if delegation.title is None:
        raise HarnessBug("a one-task plan with no title reached its stop")
    template = resolve_template(root, templates, "brief")
    task = OneTask(
        title=delegation.title, brief=said.text, sha=said.sha, template_sha=template.sha
    )
    return CoderRequest(task=task, workspace=workspace)


def _withdrawn(kind: StopKind) -> Stopped:
    """A stop Zeus's turn was stopped at before the person answered it: nothing approved, and
    nothing leased."""
    stop = "the design" if kind is StopKind.DESIGN else "the one-task plan"
    why = f"the turn was stopped while {stop} waited for the user; nothing was approved"
    return Stopped(CoderReport.before_run("ABORTED", why), None)


def _not_approved(said: StopOutcome, kind: StopKind) -> Stopped | None:
    """A stop that ended without an approval, as the call's report; None when approved."""
    # In the report's words: Zeus reads `why`, and the stops are our names.
    stop = "the design" if kind is StopKind.DESIGN else "the one-task plan"
    if said.answer is StopAnswer.NOTE:
        why = f"the user sent a note on {stop}: {said.note}"
        return Stopped(CoderReport.before_run("ASK", why), said)
    if said.answer is StopAnswer.CANCEL:
        if said.bound is not None:
            why = f"{stop} was not approved: {said.bound}"
        else:
            why = f"the user cancelled {stop}" + (f": {said.note}" if said.note else "")
        return Stopped(CoderReport.before_run("ABORTED", why), said)
    return None


def stop_at_work(
    delegation: Delegation,
    port: InteractionPort,
    *,
    repo: Path,
    root: Path,
    templates: dict[str, Path],
    workspace: WorkspaceMode,
    readme: str,
    write: Callable[[str], None],
    scope: str,
    notice: Callable[[], str | None] = lambda: None,
    cancel: threading.Event | None = None,
) -> CoderRequest | Stopped:
    """Stop 1 for a design in its work folder (A5), asked before
    anything is leased. `readme` is `design/README.md` with the harness's two sections in
    place, which `write` puts in the folder; the folder's shape is checked, and refused naming
    what is wrong without asking; the stop shows the README with the folder, re-read at each
    answer, since the person edits the files themselves; approved, `design/` is pinned under
    `scope` and the request carries its tree. `root` is where the folder is, the session's
    worktree or, in place, the checkout; `repo` the person's checkout."""
    work = delegation.work
    if work is None:
        raise HarnessBug("a design with no work folder reached the work stop")
    here = "in_place" if root.resolve() == repo.resolve() else "worktree"
    if workspace != here:
        # The run's root must be where the folder is (4b): its plan, briefs and records all
        # name the folder there.
        where = "the checkout" if here == "in_place" else "the session's worktree"
        why = (
            f"not asked, not run: the work {work.name} is in {where}, so it runs "
            f'with workspace="{here}", not "{workspace}"'
        )
        return Stopped(CoderReport.before_run("BLOCKED", why), None)
    folder = part_dir(work.dir, work.name, "design")
    write(readme)
    template = resolve_template(repo, templates, "design")

    def reread() -> Reread:
        text = (root / folder / "README.md").read_text(encoding="utf-8")
        errors = tuple(design_errors(root / folder, template.text))
        return Reread(text, folder_tree(root, folder) or "", errors)

    first = reread()
    if first.errors:
        why = "not asked, not run: the design is not in shape: " + "; ".join(first.errors)
        return Stopped(CoderReport.before_run("BLOCKED", why), None)
    shown = stop_of(
        StopKind.DESIGN,
        "zeus",
        first.text,
        workspace_preview(repo, workspace),
        folder=str(root / folder),
        # Asked only now: a stop refused before it is shown has said nothing once.
        notice=notice(),
    )
    try:
        said = ask_stop(shown, port, reread=reread, cancel=cancel)
    except StopWithdrawn:
        return _withdrawn(StopKind.DESIGN)
    stopped = _not_approved(said, StopKind.DESIGN)
    if stopped is not None:
        return stopped
    # Named by its tree, so a later design of the session never moves an earlier one's pin.
    tree = said.version
    if not tree:
        raise HarnessBug(f"the stop approved {folder} with no tree")
    pin_tree(root, tree, folder, scope, f"design-{tree}")
    design = ApprovedDesign(
        text=said.text,
        sha=said.sha,
        template_sha=template.sha,
        work=WorkPin(name=work.name, dir=work.dir, tree=tree),
    )
    return CoderRequest(design=design, workspace=workspace)


# -- one episode at a time --------------------------------------------------------------------


class OneEpisode:
    """One episode per engine: a second `coder()` while one is in flight
    is refused, never queued, because a race on the lease corrupts a worktree rather than
    raising. `_lock` guards `_current` and nothing else; nothing is called under it.

    `_current` is the claim, the session it is for and, once opened, its episode. Each claim
    has its own token, so an episode that ends while it is still being opened releases its own
    claim and never the next one. One engine serves several sessions, so what one session may
    do to the run in flight (pause it, drop it, list it, be refused for it) asks whose it is.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: tuple[object, str, BoxEpisode | None] | None = None
        self._closing = False
        """Set by `stop`, under `_lock`: no episode opens after it. Zeus's turn outlives the
        pause for a moment, and a `coder` call he made in it would be ended, not paused."""

    def open(
        self, make: Callable[[Callable[[], None]], BoxEpisode], *, session_id: str
    ) -> BoxEpisode:
        """`make(release)` opens session `session_id`'s episode; `release` frees the engine
        when it ends."""
        token = object()
        with self._lock:
            if self._closing:
                raise EngineBusy("the engine is closing; the run was not started")
            if self._current is not None:
                held = self._current[2]
                what = f"run {held.run_id}" if held is not None else "a run being opened"
                raise EngineBusy(
                    f"{what} is in flight on this engine; wait for its result, or delegate "
                    "from another engine"
                )
            self._current = (token, session_id, None)
        try:
            episode = make(lambda: self._release(token))
        except BaseException:
            self._release(token)
            raise
        with self._lock:
            # The episode may have ended, and released its claim, while it was opened.
            if self._holds(token):
                self._current = (token, session_id, episode)
        return episode

    def running(self, run_id: str) -> bool:
        """Whether run `run_id` is the episode in flight on this engine: `WorkProgress`'s one
        fact that no record holds (a run with no `run_end` is live here, or suspended)."""
        held = self.current()
        return held is not None and held.run_id == run_id

    def current(self, session_id: str | None = None) -> BoxEpisode | None:
        """The episode in flight on this engine, once opened; with `session_id`, only when it
        is that session's. None otherwise."""
        with self._lock:
            claim = self._current
        if claim is None or (session_id is not None and claim[1] != session_id):
            return None
        return claim[2]

    def busy(self, session_id: str | None = None) -> str | None:
        """What is in flight on this engine, as `open` would name it; with `session_id`, only
        when it is that session's. None otherwise."""
        with self._lock:
            claim = self._current
        if claim is None or (session_id is not None and claim[1] != session_id):
            return None
        held = claim[2]
        return f"run {held.run_id}" if held is not None else "a run being opened"

    def stop(self) -> None:
        """The engine is closing: the episode in flight is paused (`BoxEpisode.suspend`),
        so its lease is back before `close()` returns, started or not, and resume continues
        it."""
        with self._lock:
            self._closing = True
            held = self._current[2] if self._current is not None else None
        if held is not None:
            held.suspend()

    def _release(self, token: object) -> None:
        with self._lock:
            if self._holds(token):
                self._current = None

    def _holds(self, token: object) -> bool:
        """Whether the claim is still `token`'s. Called under `_lock`."""
        current = self._current
        return current is not None and current[0] is token


# -- the episode ------------------------------------------------------------------------------


def session_worktree(
    repo: Path, log: EpisodeLog, session_id: str, goal: str, policy: WorkspacePolicy
) -> Workspace:
    """The session's one worktree, its WRITE lease taken: reattached when the
    session's latest `workspace` line names one still its to write (added back from its branch
    if the directory is gone), created and recorded when not, and removed again if it cannot be
    recorded. The one constructor a `coder` call and Zeus's first `doc` call share; a dirty
    checkout seeds it, and its line's
    `source_head` is the person's `HEAD`, never the seed. The caller releases the lease."""
    with log.lock:
        written = held_worktree(log.state)
        here = still_written(latest_workspace(log.state))
    if written is not None and written.worktree_path is not None:
        with suppress(WorktreeGone):
            again = reattach_worktree(repo, Path(written.worktree_path), branch=written.branch)
            if here != written:
                # The session had left it for the checkout: back into it, said
                # by a line naming it again, so the view and the next call follow the work.
                log.record_workspace(written)
            return again
    worktree = policy.model_copy(update={"mode": "worktree"})
    workspace = acquire(repo, run_id=session_id, goal=goal, policy=worktree, session=session_id)
    try:
        person_head, source_branch = head_and_branch(repo)
        log.record_workspace(
            WorkspaceRef(
                mode="worktree",
                lease_id=workspace.spec.lease_id,
                source_ref=source_branch or "HEAD",
                source_head=person_head or commit_of(workspace.root, "HEAD"),
                branch=workspace.spec.branch,
                worktree_path=workspace.spec.worktree_path,
                handoff="pending",
            )
        )
    except BaseException:
        # Nothing names the worktree, so nothing would ever reattach or hand it off.
        abandon_worktree(repo, workspace)
        workspace.release()
        raise
    return workspace


class EpisodeLog(NumberedLog, Protocol):
    """The session as an episode needs it (`context.session.Session`): the log Athena's notes
    are written to, and the `workspace` lines that say which worktree the session owns.
    """

    @property
    def state(self) -> SessionState: ...

    def record_workspace(self, ref: WorkspaceRef) -> int: ...

    def blob(self, sha: str) -> bytes:
        """A session blob: a tell's text, or the bytes of a file the person sent."""
        ...


@dataclass(frozen=True)
class EpisodeInputs:
    """The engine's values an episode is composed from."""

    repo: Path
    config: EngineConfig
    denylist: Denylist
    ledger: CostLedger
    """The session's ledger: every episode charges the same one, and each role's rung
    is cut its view of it."""
    backend: SandboxBackend
    interaction: InteractionPort
    session_id: str
    session: EpisodeLog | None = None
    """The session's writer: Athena's notes go on it, and it records the session's
    worktree. None: she has no `note`, and each run has a worktree of its own."""
    asking: Callable[[str], bool] | None = None
    """Whether a step's permission question still waits on the session log."""
    prefs: Callable[[], Preferences] | None = None
    """The person's approval.toml, read per call by every shell, edit port and step.
    None: no preferences, for a caller with no config."""
    commands: CommandTable | None = None
    """The engine's `[[commands]]` table, given to the episode's executor beside the
    deny list; None is the shipped table."""
    beside: Beside | None = None
    """The session's jobs: a step beside one says so."""
    live: LiveSink | None = None
    """Where both roles' replies are shown as they stream: the session's running turn."""
    accepted: bool = False
    """Whether this clone's owner accepted the repository's executable config: a
    repository's hook script is bound only then."""
    runs: Callable[[str], RunProjection] | None = None
    """The engine's one fold of the run journals (C2): a run that continues a work reads its
    earlier runs through it. None: no run continues one."""
    replied: Callable[[str, TellOutcome, str], None] | None = None
    """The team's reply to a tell, on the session log as Zeus reads it: a
    trigger-10 notice, which wakes him when the reply needs him. None: nobody is told."""


def open_episode(
    request: CoderRequest, inputs: EpisodeInputs, release: Callable[[], None]
) -> BoxEpisode:
    """Name the run and take the lease (`Box.open`). Nothing else: no journal is written until
    the episode is started (item 10)."""
    return _Composed(inputs, release).open(lambda box: box.open(request))


def resume_episode(
    run_id: str, inputs: EpisodeInputs, release: Callable[[], None]
) -> BoxEpisode:
    """Take the lease again over run `run_id`'s workspace, to continue it (`Box.resume`). A run
    opened before contracts v2 is refused here, naming D196 (`episode.opened_with`)."""
    opened_with(resolve_roots(inputs.repo), run_id)
    return _Composed(inputs, release).open(lambda box: box.resume(run_id))


class _Composed:
    """One episode's `BoxEnv`, from the engine's values."""

    def __init__(self, inputs: EpisodeInputs, release: Callable[[], None]) -> None:
        self.inputs = inputs
        self.release = release
        self.stack = ExitStack()
        """What the episode's parts opened, closed on the episode's thread when it ends."""
        self.cancel = threading.Event()
        """The soft stop: the episode pauses at the next task boundary."""
        self.abort = threading.Event()
        """The hard stop: a request in flight is abandoned."""
        self.suspend = threading.Event()
        """The pause as the process leaves: set before `abort`, so what the hard stop
        cuts off is left for resume rather than ended."""
        self.told = threading.Event()
        """Set when Zeus tells the run something: its next boundary looks
        (`box/tells.Tells`, whose token this is)."""
        self.run_id: str | None = None
        """The run `Box.open` named, which the stops and the shells' questions name."""

    def open(self, how: Callable[[Box], BoxRun]) -> BoxEpisode:
        inputs = self.inputs
        env = BoxEnv(
            workspace_for=self._workspace,
            reattach=lambda run_id: reattach(
                inputs.repo, run_id=run_id, worktree=self._wrote_in(run_id)
            ),
            parts_for=self._parts,
            setting=self._setting,
            asking=inputs.asking or (lambda _: False),
            cancel=self.cancel,
            abort=self.abort,
            suspend=self.suspend,
        )
        run = how(Box(env))
        self.run_id = run.run_id
        return BoxEpisode(run, self)

    def _workspace(self, run_id: str, request: CoderRequest) -> Workspace:
        """The session's one worktree: reattached when the session's
        latest `workspace` line names one still its to write (added back from its branch if the
        directory is gone), created and recorded when not, and removed again if it cannot be
        recorded. Only the lease is this call's. In place, and a run with no session, are a
        run's own workspace, as before."""
        inputs = self.inputs
        policy = inputs.config.workspace.model_copy(update={"mode": request.workspace})
        log = inputs.session
        if request.workspace == "in_place" or log is None:
            return acquire(
                inputs.repo,
                run_id=run_id,
                goal=request.goal,
                policy=policy,
            )
        return session_worktree(inputs.repo, log, inputs.session_id, request.goal, policy)

    def _told(self) -> tuple[Told, ...]:
        session = self.inputs.session
        return () if session is None else _told_from(session)

    def _wrote_in(self, run_id: str) -> Path | None:
        log = self.inputs.session
        if log is None:
            return None
        with log.lock:
            return run_worktree(log.state, run_id)

    def _setting(self) -> EpisodeSetting:
        """The episode's settings from the config: each role's rung, tools and limits, and the
        templates in force in the repository."""
        inputs, config = self.inputs, self.inputs.config
        limits = config.resolved_limits
        templates = config.templates.configured()
        brief = resolve_template(inputs.repo, templates, "brief")
        plan = resolve_template(inputs.repo, templates, "plan")
        return EpisodeSetting(
            worker=self._role("worker"),
            athena=self._role("athena"),
            head=HeadParts(
                fixed=box_athena.instructions(),
                plan_template=plan.text,
                brief_template=brief.text,
            ),
            brief_template_sha=brief.sha,
            workspace=workspace_preview(inputs.repo, "worktree"),
            untracked_max_bytes=config.undo.untracked_max_bytes,
            task_seconds=resolved_value(limits, "worker", "worker.task_seconds"),
            recut_cap=resolved_value(limits, "athena", "athena.max_recut_depth"),
            one_task_max=resolved_value(limits, "athena", "athena.one_task_max_tasks"),
            design_layer=resolved_value(limits, "worker", "worker.design_layer"),
            note_cap=resolved_value(limits, "athena", "athena.note_max"),
            hooks=HookConfig.of(config, accepted=inputs.accepted),
        )

    def _role(self, name: Literal["worker", "athena"]) -> RoleSetting:
        """A box role's rung, tools and limits as the config resolves them. Its
        client closes with the episode; a hard stop abandons a request in flight."""
        config = self.inputs.config
        rung_name = config.rung_name_for(name)
        endpoint = config.rung_for(name)
        client = client_for(endpoint)
        self.stack.callback(client.close)
        limits = config.resolved_limits
        return RoleSetting(
            rung=BoxRung(
                client,
                endpoint,
                config.rungs[rung_name],
                rung_name,
                self.inputs.ledger.for_rung(rung_name, role=name),
            ),
            policy=config.tool_policy(name),
            limits=role_limits(limits, name),
            vision=config.vision_for(rung_name),
            params=first_attempt_params(endpoint),
            budget=window(endpoint) - endpoint.max_output_tokens,
            keep_recent=resolved_value(limits, name, f"{name}.keep_recent"),
        )

    def _parts(self, workspace: Workspace) -> RunParts:
        """Made on the episode's thread, where they are used and closed: the graph's SQLite
        connection belongs to one thread. A fresh worktree has no index of its own, so
        it is built here; a checkout that has one is brought up to its bytes."""
        inputs, root = self.inputs, workspace.root
        deny_read = resolve_denylist(
            inputs.denylist, [root, Path.home()], memo=WalkMemo.shared()
        )
        # One tool cache per repository, the same in every worktree of it.
        cache = tool_cache(repository_key(root))
        # Chosen once, so each shell is offered on the backend its commands run on.
        backend = inputs.backend or select()
        spawn = HarnessSpawn(
            backend,
            ProcessSupervisor(),
            require_sandbox=inputs.config.sandbox.require_sandbox,
            deny_read=deny_read,
        )
        fresh = not index_exists(root)
        languages = ProfileSet.load(repo=root)
        # The index's edges resolved by each language's server, under the episode's wrap
        # built first, so the index the episode reads has them. Each server is the
        # episode's, kept between passes and closed with it.
        servers = EpisodeServers(
            root.resolve(),
            spawn.supervisor,
            servers_max=inputs.config.harness_limit("lsp.servers_max"),
            idle_s=inputs.config.harness_limit("lsp.idle_s"),
        )
        self.stack.callback(servers.close)
        indexer = Indexer(
            root,
            index_resolver(root, spawn, languages, servers),
            denylist=inputs.denylist,
            profiles=languages,
        )
        self.stack.callback(indexer.close)
        if fresh:
            indexer.full_index()
        else:
            indexer.reindex()
        graph = Graph(root, indexer.conn, profiles=indexer.profiles)
        if self.run_id is None:
            raise HarnessBug("an episode's parts were made before its run was named")
        port = for_run(inputs.interaction, self.run_id)

        sandbox = inputs.config.sandbox

        def executor(journal: JournalPort) -> Executor:
            # The person's `require_sandbox`, as Zeus's executor has (amending
            # D196 §6.1 point 1): with it set, `shell_for` offers no shell where none seals.
            return Executor(
                backend,
                require_sandbox=sandbox.require_sandbox,
                put_blob=journal.put_blob,
                denylist=inputs.denylist,
                deny_read=deny_read,
                commands=inputs.commands,
                tool_cache=cache,
                withheld_env=inputs.config.endpoint_keys,
            )

        def syntax(argv: tuple[str, ...], cwd: Path, timeout_s: int) -> ExecResult:
            return run_argv(spawn, argv, cwd, timeout_s=timeout_s, label="syntax")

        return RunParts(
            servers=servers,
            worktree=root,
            profiles=profiles_for(root),
            languages=indexer.profiles,
            denylist=inputs.denylist,
            graph=graph,
            indexer=indexer,
            budget=inputs.ledger,
            interaction=inputs.interaction,
            shell_for=shell_for(
                root,
                backend,
                executor,
                PersonAuthority(port, self.abort.set),
                inputs.prefs,
                require_sandbox=sandbox.require_sandbox,
                lifts=sandbox.allow_unsandboxed_shell,
                suspend=self.suspend,
            ),
            spawn=syntax,
            notes=inputs.session,
            prefs=inputs.prefs,
            # The scout's search, over the worktree under the run's backend and masks.
            search=Search(
                root=root, backend=backend, denylist=inputs.denylist, deny_read=deny_read
            ),
            work_dir=inputs.config.work.dir,
            work_commit=inputs.config.work.commit,
            beside=inputs.beside,
            live=inputs.live,
            runs=inputs.runs,
            attachment_bytes=inputs.session.blob if inputs.session is not None else None,
            attach_caps=AttachCaps.of(inputs.config),
            told=self._told if inputs.session is not None else None,
            told_token=self.told,
            replied=inputs.replied,
            # `require_sandbox` wins, as it withholds the shell: the lift applies only where
            # the person also let commands run with no sandbox.
            python_unsandboxed=sandbox.allow_unsandboxed_python and not sandbox.require_sandbox,
            scripts=sealed_runner(backend, lambda: partial(run_argv, spawn), root),
        )


def _told_from(session: EpisodeLog) -> tuple[Told, ...]:
    """The session's tells, as the run reads them: each with its text. Read under
    the session's lock, as every reader of its fold is: Zeus's thread writes a `tell` line
    while the run's reads them. A blob never changes, so its text is read after."""
    with session.lock:
        lines = session.state.tells()
    return tuple(
        Told(t.id, session.blob(t.text_sha).decode("utf-8"), t.sources, t.run_id) for t in lines
    )


class BoxEpisode:
    """One delegated run: named, its lease held, and started on its own thread the first time
    its report or its events are asked for. A stop reaches it as CL-10's interrupt: soft at
    the next task boundary, hard now."""

    def __init__(self, run: BoxRun, composed: _Composed) -> None:
        self.run_id = run.run_id
        self.request = run.request
        self._run = run
        self._composed = composed
        self._repo = composed.inputs.repo
        self._lock = threading.Lock()
        """Guards `_thread` and `_ended`: one start, and one end."""
        self._thread: threading.Thread | None = None
        self._ended = False
        self._done = threading.Event()
        self._outcome: CoderReport | Paused | None = None
        self._error: BaseException | None = None

    # -- running ----------------------------------------------------------------------------

    def start(self) -> None:
        """Run the episode, once, on a thread of its own bound to the session."""
        with self._lock:
            if self._thread is not None or self._ended:
                return
            session_id = self._composed.inputs.session_id
            self._thread = threading.Thread(
                target=self._work, args=(session_id,), name=f"box-{self.run_id}", daemon=True
            )
            self._thread.start()

    def report(self) -> CoderReport | Paused:
        """The run's report, or `Paused` when a soft stop paused it at a task boundary.
        Starts the run if nothing has; blocks until it ends."""
        self.start()
        self._done.wait()
        if self._error is not None:
            raise self._error
        if self._outcome is None:
            raise HarnessBug(f"run {self.run_id} ended with neither a report nor an error")
        return self._outcome

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def workspace(self) -> SnapshotSite:
        """Where a turn snapshot of the run's workspace is taken."""
        return self._run.workspace

    def told(self) -> None:
        """Zeus told the run something: its next boundary delivers it."""
        self._composed.told.set()

    def interrupt(self, hard: bool = False) -> None:
        """Soft pauses at the next task boundary, hard stops now. Safe at any time,
        before the start included: a run stopped before it starts pauses at once."""
        self._composed.cancel.set()
        if hard:
            self._composed.abort.set()

    def abandon(self) -> None:
        """The report will never be asked for: release the lease. A started run is not
        abandoned; it is stopped and its report waited for, so the lease is released by it."""
        self._stop_now(suspend=False)

    def suspend(self) -> None:
        """The process is leaving: a started run stops now, as `abandon` stops it, but
        is left for resume (`Paused`, no `run_end`); one not started did nothing, and is given
        up as `abandon` gives it up."""
        self._stop_now(suspend=True)

    def _stop_now(self, *, suspend: bool) -> None:
        with self._lock:
            started = self._thread is not None
            if not started:
                self._ended = True
        if started:
            if suspend and not self._composed.abort.is_set():
                # A hard stop already given (Ctrl-C twice) ends the run as the person asked;
                # leaving after it does not make it resumable again.
                self._composed.suspend.set()
            self.interrupt(hard=True)
            self._done.wait()
            return
        try:
            self._run.abandon()
        finally:
            self._end()

    def _work(self, session_id: str) -> None:
        bind_session(session_id)
        try:
            self._outcome = self._run.report()
        except BaseException as exc:
            self._error = exc
        finally:
            self._end()

    def _end(self) -> None:
        try:
            self._composed.stack.close()
        finally:
            self._composed.release()
            self._done.set()

    # -- watching ---------------------------------------------------------------------------

    def events(self) -> Iterator[JournalEvent]:
        """The run's journal as it is written, from its first line, then live until the run
        ends (`journal.reader.follow`), never from the run's memory, so a watcher sees what
        replay reconstructs (item 8). Starts the run if nothing has. An episode that ended
        before its journal was written has no events."""
        self.start()
        return follow(resolve_roots(self._repo), self.run_id, self._done, TAIL_POLL_S)
