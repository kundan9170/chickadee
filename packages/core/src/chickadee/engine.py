"""`PacketEngine` — the sole public surface.

The skeleton exists from Phase 0 because everything downstream consumes it: the CLI, the
MCP server and the eval rig are all written against this facade, and they may import
nothing else from core (enforced by `test_no_internal_imports.py`). Methods are
filled in by the task that owns their phase; until then they raise `NotImplementedError`,
which is honest in a way that returning an empty result is not.

What the facade is *for* is worth stating, because it looks like indirection. Components
are private so that the engine can enforce ordering no host can skip: approve before spend,
lease before mutate, gates after every patch. A host holding a `Verifier` directly could
call it in an order the invariants forbid, and there would be nothing to stop it.

**The surface, not only the bodies.** This file carried the pre-adoption shape long
after the specs stopped describing it: `route_goal`, `plan`, `run_goal`/`run_dag`/`run_packet`,
`ask`/`investigate`, `RunHandle`, `AnswerHandle`, `PlanPreview`. All of those are deleted.
A skeleton is a *promise about shape* — it is the thing hosts are written against before
anything works — so a stale one is worse than an empty one: it invites a host to be built
against surfaces that will never exist. What replaces them is: one delegation call, one
preview, a resume that takes a session, and reading as a tool call inside the session.

**Phase-3 types are `Any`, on purpose.** `CoderRequest`, `TierDecision`, `CoderPreview` and
the rest land with the tasks that build them; annotating them here would mean either
inventing a second definition — the exact defect `audit_docs.py` check 12 exists to catch —
or importing from packages that do not exist yet. Each `Any` names its real type and the
spec section that owns it, so the task landing it has one place to substitute.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

from chickadee import undo as undo_module
from chickadee.adapters.base import SHARED_TEMPLATES, first_attempt_params, load_template
from chickadee.adapters.frontier import TEMPLATES
from chickadee.agents.profiles import Profile, agent_costs, config_profiles
from chickadee.agents.report import AgentReport as AgentReport
from chickadee.agents.spawn import AgentPorts, Agents
from chickadee.approval.grants import GrantingPort
from chickadee.approval.prefs import PreferenceError, load_preferences
from chickadee.approval.stops import WorkspacePreview as WorkspacePreview
from chickadee.approval.tier import Preferences
from chickadee.bootstrap import BootstrapReport as BootstrapReport
from chickadee.bootstrap import bootstrap_worktree as bootstrap_worktree
from chickadee.box.edits import WorktreeEdits
from chickadee.box.episode import Complete, end_superseded
from chickadee.box.episode import Paused as Paused
from chickadee.box.report import CoderReport as CoderReport
from chickadee.box.report import reported_tasks
from chickadee.box.request import ApprovedDesign as ApprovedDesign
from chickadee.box.request import AttachedFile, Continuation
from chickadee.box.request import CoderRequest as CoderRequest
from chickadee.box.request import OneTask as OneTask
from chickadee.box.spawn import HarnessSpawn, run_argv
from chickadee.box.syntax import SyntaxChecker
from chickadee.budget.family import RungBudget as RungBudget
from chickadee.budget.family import SessionBudget as SessionBudget
from chickadee.budget.family import family_budget, ledger_inputs, session_spend
from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable

# Re-exported for surfaces that drive a model directly (the eval arms): they may import only
# the facade, and the client protocol and the coder path's rendering are what they need
from chickadee.client import ModelClient as ModelClient
from chickadee.client import client_for
from chickadee.client import to_request as to_request
from chickadee.client.affinity import AffinityTable
from chickadee.client.live import ContextUpdate, LiveEvent
from chickadee.client.once import ModelAnswer as ModelAnswer
from chickadee.client.once import ask_once
from chickadee.client.request import ChatRequest, Message
from chickadee.config import EngineConfig, RungEndpoint, is_cap
from chickadee.context import standing
from chickadee.context.attachment import (
    AttachCaps,
    AttachmentPreview,
    CheckedFile,
    RefusedFile,
    check,
    check_turn,
)
from chickadee.context.changes import worktree_changes
from chickadee.context.citations import Verdict as CitationVerdict
from chickadee.context.citations import WebCitation as WebCitation
from chickadee.context.citations import workspace_lines
from chickadee.context.compaction.residue import ModelResidue
from chickadee.context.compaction.session import SessionCompaction
from chickadee.context.homes import Scratch, SessionHome
from chickadee.context.ledger import binding_constraints, record_refusal_note
from chickadee.context.lines import EpisodeLine
from chickadee.context.project import JournalRecords
from chickadee.context.questions import (
    RecordingPort,
    open_in,
    open_reclaimed,
    scoped,
    still_asking,
)
from chickadee.context.reader import Reader
from chickadee.context.recall import Recall
from chickadee.context.replay import ReplayCache
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import InvalidArgument as InvalidArgument
from chickadee.context.results import ToolResult
from chickadee.context.rewind import ForkRefused
from chickadee.context.session import CODER, Session, SessionState
from chickadee.context.standing import (
    Notice,
    NoticeTrigger,
    hook_noted,
    run_reported,
    view_moved,
)
from chickadee.context.store import (
    LOCK_NAME,
    SessionMissing,
    SessionStoreError,
    session_dir,
    sessions_root,
)
from chickadee.context.terms import ReadRules
from chickadee.context.view import (
    REATTACHED,
    Rooted,
    SessionView,
    last_worktree,
    latest_workspace,
    run_workspace,
    session_mode,
)
from chickadee.delegation import (
    BoxEpisode,
    EpisodeInputs,
    OneEpisode,
    Stopped,
    chosen_workspace,
    session_worktree,
    stop_at_dispatch,
    stop_at_work,
)
from chickadee.delegation import open_episode as _open_episode
from chickadee.delegation import resume_episode as _resume_episode
from chickadee.durable import write_new
from chickadee.edits.draft import DraftRefused
from chickadee.errors import (
    AttachmentRefused,
    ConfigError,
    EngineBusy,
    HarnessBug,
    InfraError,
    RunNotStarted,
    SessionError,
)
from chickadee.errors import ChickadeeError as ChickadeeError
from chickadee.executor.executor import Executor
from chickadee.external.hosts import HostApproval
from chickadee.external.policy import ExternalPolicy
from chickadee.external.search.engines import session_sources
from chickadee.external.web import Web, WebLimits, find_lightpanda, page_engine
from chickadee.graph.ignore import listed
from chickadee.graph.indexer import Gap as Gap
from chickadee.graph.indexer import IndexProgress as IndexProgress
from chickadee.graph.indexer import IndexReport as IndexReport
from chickadee.graph.queries import Graph
from chickadee.handoff import HandoffResult
from chickadee.handoff_summary import HandoffSummary
from chickadee.handoffs import ZEUS_HANDOFFS
from chickadee.handoffs import resolve as resolve_handoff
from chickadee.hooks.bind import HookConfig, bind_session
from chickadee.hooks.fire import Hooks, RoleHooks
from chickadee.hooks.payloads import BeforeCompact, SessionEnd, SessionStart
from chickadee.hooks.points import HookPoint
from chickadee.hooks.script import ScriptSpawn, sealed_runner
from chickadee.hooks.show import shown as hooks_shown
from chickadee.hooks.sinks import SessionHookSink
from chickadee.ids import new_session_id
from chickadee.init.detect import NotAGitRepository, repository_key
from chickadee.init.detect import resolve as _detect_root
from chickadee.init.lease import LeaseBusy
from chickadee.init.lease import building as _building
from chickadee.init.orchestrate import IndexSandbox, InitError
from chickadee.init.orchestrate import InitReport as _InitReport
from chickadee.init.orchestrate import init as _init
from chickadee.init.orchestrate import reindex as _reindex
from chickadee.init.orchestrate import reindex_held as _reindex_held
from chickadee.init.prepare import NotHere as _NotHere
from chickadee.init.prepare import make_repository as _make_repository
from chickadee.init.publish import is_initialised as _is_initialised
from chickadee.init.servers import Installed as Installed
from chickadee.init.servers import ServerNeed as ServerNeed
from chickadee.init.servers import install as install_server
from chickadee.init.servers import server_needs as server_needs
from chickadee.inspect import Inspection as Inspection
from chickadee.inspect import state_roots as _state_roots
from chickadee.isolation.base import Probe as Probe
from chickadee.isolation.base import configured as _select_backend
from chickadee.isolation.base import level_of, reads_masked
from chickadee.isolation.base import probe_native as probe_native
from chickadee.isolation.types import tool_cache
from chickadee.journal.errors import JournalCorrupt, JournalError
from chickadee.journal.gc import BULK, BlobClass, GcPlan, GcResult
from chickadee.journal.gc import apply as _gc_apply
from chickadee.journal.gc import plan as _gc_plan
from chickadee.journal.metrics import RunMetrics, derive_metrics
from chickadee.journal.paths import journal_path, run_dir
from chickadee.journal.paths import runs_root as _runs_root
from chickadee.journal.projection import RunProjection
from chickadee.journal.runs import RunProjections
from chickadee.limits import resolved_value, role_limits, window
from chickadee.liveness import Liveness as Liveness
from chickadee.liveness import liveness as _liveness
from chickadee.naming import Namer
from chickadee.open_works import OpenWork, OpenWorkInfo, open_works
from chickadee.open_works import adopt as adopt_work
from chickadee.process import ProcessSupervisor
from chickadee.profiles.loader import ProfileSet
from chickadee.roles import former_rung_error, role
from chickadee.run_header import IN_PLACE_NO_BOOTSTRAP as IN_PLACE_NO_BOOTSTRAP
from chickadee.run_header import NO_BOOTSTRAP as NO_BOOTSTRAP
from chickadee.run_header import HeaderLine as HeaderLine
from chickadee.run_header import RunHeader as RunHeader
from chickadee.run_header import header as _header
from chickadee.run_header import spend_facts as _spend_facts
from chickadee.run_reads import FileText as FileText
from chickadee.run_reads import NotListed as NotListed
from chickadee.run_reads import NotNamed as NotNamed
from chickadee.run_reads import NotReadable as NotReadable
from chickadee.run_reads import RunBlob as RunBlob
from chickadee.run_reads import RunDiff as RunDiff
from chickadee.run_reads import file_text, run_blob, run_diff
from chickadee.schema import GitHandoffChoice, InteractionPort, JournalEvent
from chickadee.schema.authorization import TIER_ORDER, ActionAuthorization
from chickadee.schema.findings import Findings, Unresolved
from chickadee.schema.ids import RUN_ID_RE, new_run_id, sha256_hex
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee.schema.isolation import WorkspaceMode
from chickadee.schema.journal import RunEndP, TellOutcome
from chickadee.scopes import PAGE_DEFAULT, Cursor, Page, ScopeRecord, run_page, session_page
from chickadee.scout.search import Search
from chickadee.secrets.denylist import resolve as _resolve_denylist
from chickadee.secrets.redact import redact_text
from chickadee.secrets.walk_memo import WalkMemo
from chickadee.session_handoff import (
    finalize_session,
    handoff_choice,
    paused_run,
    settle,
    summarize,
    unfinished_runs,
)
from chickadee.sessions import Sessions, SessionTitle, record_title, title_of
from chickadee.shell import AGENT as AGENT_SHELL
from chickadee.shell import Shell
from chickadee.skills.discover import SkillCatalog, SkillRoots, discover
from chickadee.skills.index import render_index
from chickadee.skills.load import Skills
from chickadee.skills.pin import pinned_seqs
from chickadee.state.cap import engine_cap
from chickadee.state.layers import CONFIG_NAME, Layer, LayerReadError, engine_layers
from chickadee.state.load import HostConfig as HostConfig
from chickadee.state.load import RepositoryTrust as RepositoryTrust
from chickadee.state.load import accept as accept_trust
from chickadee.state.load import load_host_config as load_host_config
from chickadee.state.load import repository_trust as repository_trust
from chickadee.state.local import record_work_dir
from chickadee.state.merge import MergeError as MergeError
from chickadee.state.resolve import TrustDecision as TrustDecision
from chickadee.state.roots import USER_DIR, NotARepository, Roots, xdg_config_home
from chickadee.state.roots import roots as _roots
from chickadee.state.seed import SeedReport as SeedReport
from chickadee.state.seed import render as render_seed
from chickadee.state.user import user_file, write_user_file
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import ExecPort, HandleStore, Locator
from chickadee.tools.tool import Said
from chickadee.unbuilt import not_built
from chickadee.undo import CodeOutcome, ForkChoice, Forked, Restored, Undoable, UndoReport
from chickadee.wiring import (
    CatalogModel,
    ModelSwitch,
    ModelWiring,
    Payload,
    pin_of,
    record_pinned,
    record_switch,
    rungs_sha,
    switches_of,
    wired,
)
from chickadee.work import FOLDER_YES, folder_question, free_work_dir, notice_once, open_work
from chickadee.work_progress import (
    JournalRuns,
    Progresses,
    RunReader,
    WorkKey,
    WorkProgress,
    complete,
    status_text,
)
from chickadee.workspace import SnapshotSite, head_and_branch, text_diff, tree_diff
from chickadee.zeus.calls import ZeusTools
from chickadee.zeus.conversation import (
    Built,
    Conversation,
    Conversations,
    Lasting,
    LiveTurn,
    Phase,
)
from chickadee.zeus.delegate import (
    Delegation,
    amended_request,
    attached_files,
    given_back_on_failure,
    inserted,
    render_outcome,
    request_attachments,
    request_section,
    requested,
    task_lines,
    with_attached,
)
from chickadee.zeus.docs import DesignDocs
from chickadee.zeus.jobs import JobTable, JobView, job_limits, wake_ready
from chickadee.zeus.loop import Event, LoopConfig, ZeusLoop
from chickadee.zeus.prefix import Env, assemble, current_platform
from chickadee.zeus.reconcile import run_outcome, tell_ended, told_ended
from chickadee.zeus.snapshots import TurnSnapshots
from chickadee.zeus.tools.graphnav import (
    GraphNavigator,
    indexed_profiles,
    opened,
    opened_until_closed,
)
from chickadee.zeus.tools.navigate import NoLocator, UnavailableLocator, scout_locator
from chickadee.zeus.tools.talk import PersonAuthority
from chickadee.zeus.worktree import WorktreeMoves
from chickadee.zeus.writes import ShellWrites

__all__ = [
    "BULK",
    "DEFAULT_RUN_OPTIONS",
    "AgentReport",
    "ApprovedDesign",
    "BlobClass",
    "BootstrapReport",
    "ChickadeeError",
    "CitationVerdict",
    "CodeOutcome",
    "CoderHandle",
    "CoderReport",
    "CoderRequest",
    "FileText",
    "ForkChoice",
    "Forked",
    "Gap",
    "GcPlan",
    "GcResult",
    "HeaderLine",
    "HostConfig",
    "IndexReport",
    "InitError",
    "InitReport",
    "Inspection",
    "Installed",
    "InvalidArgument",
    "JobView",
    "JournalError",
    "LayerReadError",
    "LeaseBusy",
    "MergeError",
    "ModelAnswer",
    "ModelClient",
    "NonInteractive",
    "NotAGitRepository",
    "NotARepository",
    "NotListed",
    "NotNamed",
    "NotReadable",
    "OneTask",
    "PacketEngine",
    "Paused",
    "Probe",
    "RepositoryTrust",
    "Restored",
    "RunBlob",
    "RunDiff",
    "RunHeader",
    "RunOptions",
    "RunProjection",
    "RunResult",
    "RungBudget",
    "ScriptedInteraction",
    "SeedReport",
    "ServerNeed",
    "SessionBudget",
    "SessionHandle",
    "SessionInfo",
    "TrustDecision",
    "TurnHandle",
    "UndoReport",
    "Undoable",
    "WebCitation",
    "WorkProgress",
    "WorkspacePreview",
    "accept_trust",
    "bootstrap_worktree",
    "gc_runs",
    "hooks_shown",
    "in_repository",
    "init_repo",
    "inspect_runs",
    "install_server",
    "is_initialised",
    "load_host_config",
    "make_repository",
    "probe_native",
    "replay_run",
    "repository_trust",
    "seed_engine_config",
    "server_needs",
    "to_request",
    "user_file",
    "write_user_file",
]

_PHASE = "filled in by its owning phase; see IMPLEMENTATION_PLAN §6"

# -- values the facade returns ----------------------------------------------
#
# Engine-owned, so they live here rather than in `schema/**`: they are the API's shape,
# not the wire's. The preview (`CoderPreview`, `Estimate`) retired with the work tier and the
# plan it showed: the work is shown at its stop.


# Re-exported so a surface reaches init through the facade alone. `init_repo` is
# also a module-level function because `chickadee init` runs *before* there is a config to
# construct a `PacketEngine` with — the one command that cannot assume a prepared
# repository cannot require one.
InitReport = _InitReport


def init_repo(
    repo: Path, *, force: bool = False, say: Callable[[str], None] = lambda _line: None
) -> _InitReport:
    """Prepare a repository, making the folder one first when it is not. The only
    entry point that predates engine construction."""
    return _init(repo, force=force, say=say)


def in_repository(path: Path) -> bool:
    """Whether `path` is inside a git repository: what a frontend asks before it starts an
    engine, which needs one."""
    try:
        _detect_root(path)
    except NotAGitRepository:
        return False
    return True


def make_repository(
    folder: Path, *, say: Callable[[str], None] = lambda _line: None
) -> tuple[str, ...]:
    """`folder` made a git repository, its files one first commit, secrets left out
    (`init/prepare.py`); returns the paths left out. `InitError` for the home
    directory or above."""
    try:
        return _make_repository(folder, say=say)
    except _NotHere as exc:
        raise InitError(str(exc)) from exc


def is_initialised(path: Path) -> bool:
    """Whether the repository holding `path` is prepared: False for a folder
    that is not a repository, or one never initialised. What a frontend asks before it starts
    an engine, so it can prepare the folder first rather than refuse."""
    try:
        return _is_initialised(_detect_root(path).root)
    except NotAGitRepository:
        return False


def replay_run(repo: Path, run_id: str) -> RunProjection:
    """`chickadee replay`: run `run_id`'s journal folded from disk
    (through `RunProjections`, the one fold), every inconsistency a `Defect` on it.
    Like `init_repo`, it runs before any engine: a journal needs no config.
    `FileNotFoundError` for a run with no journal, `JournalError` for one that cannot be
    read, `NotARepository` outside one."""
    return RunProjections(_roots(repo)).projection(run_id)


def inspect_runs(repo: Path | None = None, *, state: Path | None = None) -> Inspection:
    """`chickadee inspect`: the runs and sessions of `repo`, or of a state directory holding
    `runs/` and `sessions/`, read-only. Like `replay_run`, it needs no engine or config, so it
    reads beside a running session. `NotARepository` for a `repo` outside one."""
    if state is not None:
        return Inspection(_state_roots(state))
    return Inspection(_roots(repo if repo is not None else Path.cwd()))


def gc_runs(
    repo: Path,
    *,
    older_than: dt.timedelta,
    classes: Iterable[BlobClass] = BULK,
    run: str | None = None,
    dry_run: bool = False,
) -> tuple[GcPlan, GcResult | None]:
    """`chickadee gc`: the plan, and what applying it removed,
    unless `dry_run`. Like `init_repo`, it runs before any engine: the runs on disk need no
    config. `JournalError` for a class that may not be pruned, `NotARepository` outside one."""
    planned = _gc_plan(
        _runs_root(_roots(repo)), older_than=older_than, classes=classes, run=run
    )
    return planned, None if dry_run else _gc_apply(planned)


def seed_engine_config(path: Path | None = None) -> SeedReport:
    """Write the person's engine config, `~/.config/chickadee/config.toml` unless told
    otherwise, only when no file is there. Never over an existing one, even one that
    appears while this runs (`durable.write_new`). Like `init_repo`, it runs before any engine
    exists: without this file no engine can be built. Reports the path, whether it wrote, and
    the cap in force there. `OSError` when the file cannot be written."""
    target = path if path is not None else xdg_config_home() / USER_DIR / CONFIG_NAME
    written = write_new(target, render_seed().encode("utf-8"))
    try:
        layers: list[Layer] | LayerReadError = engine_layers(target)
    except LayerReadError as exc:
        # A file that does not parse is the person's to fix; the report says so (as init's).
        layers = exc
    cap, note = engine_cap(target, layers)
    return SeedReport(path=target, written=written, hard_cost_cap_usd=cap, note=note)


# `IndexReport` and `Gap` are imported from `graph/indexer.py`, not defined here. The facade
# had its own — `(files_indexed, languages, duration_ms, skipped)` against the indexer's
# `(files_indexed, symbols, edges, edge_classes, gaps, truncated, wall_ms)`, one field in
# common and the timing field spelled differently — while the graph's docs annotate the
# indexer's as "returned by PacketEngine.init_repo/reindex". Two attempts at one type, and
# the facade's dropped `truncated` and `gaps`: the two fields that tell a host "no callers"
# from "I did not finish indexing" Re-exported so a host
# reaches it through the facade alone.


@dataclass(frozen=True)
class RunOptions:
    execution_mode: WorkspaceMode = "worktree"
    source_ref: str | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    outcome: str
    """The box's outcome (`CoderReport.outcome`): a result, never an exception. `PAUSED`
    when a soft stop paused the run at a task boundary: it has not ended, and `resume()`
    continues it."""
    why: str | None = None
    tasks_committed: int = 0
    tasks_total: int = 0
    cost_usd: float = 0.0


DEFAULT_RUN_OPTIONS = RunOptions()
"""A shared default because `RunOptions` is frozen, so every call site can hold the same
instance. Spelled as a singleton rather than `RunOptions()` in the signature only to keep
the mutable-default linter quiet — the value is identical."""


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    last_seq: int
    last_run_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    title: str | None = None
    """The first line of the session's own first user turn, cut, computed when asked and
    stored nowhere. None for a session with no words yet, which renders as *untitled*."""
    continuing: OpenWorkInfo | None = None
    """The open work a new session continues, said once by the start header;
    set only by `create_session(adopt=…)`, None everywhere else."""


class CoderHandle:
    """One delegated `coder` call in flight — an *episode*, not the session.

    Observation and interruption only. The host cannot steer the work through this; steering
    is a new request, which is what keeps the approval bound to something that was actually
    shown to someone.
    """

    def __init__(
        self,
        run_id: str,
        episode: BoxEpisode | None = None,
        ended: Callable[[str], RunEndP | None] | None = None,
    ) -> None:
        self.run_id = run_id
        self._episode = episode
        self._ended = ended
        """Reads a run's `run_end` from its journal: its counts are the journal's."""

    def events(self) -> Iterator[JournalEvent]:
        """Live stream. Tails the journal rather than sharing in-memory state, so an
        observer sees exactly what replay will reconstruct and invariant 3 stays true for
        watchers too. Starts the run if nothing has."""
        return self._live().events()

    def report(self) -> CoderReport | Paused:
        """The box's report, or `Paused`. Starts the run if nothing has; blocks until it
        ends."""
        return self._live().report()

    @property
    def workspace(self) -> SnapshotSite:
        """Where the run writes, as a turn snapshot names it: its root, mode and
        key. Read-only, and holding nothing of the lease."""
        return self._live().workspace

    def result(self) -> RunResult:
        """Blocks until the episode ends, or pauses."""
        outcome = self.report()
        if isinstance(outcome, Paused):
            return RunResult(self.run_id, "PAUSED")
        end = self._ended(self.run_id) if self._ended is not None and outcome.run_id else None
        if end is None:
            return RunResult(self.run_id, outcome.outcome, why=outcome.why)
        return RunResult(
            self.run_id,
            end.outcome,
            why=end.why,
            tasks_committed=end.committed,
            tasks_total=end.tasks,
            cost_usd=end.cost_usd,
        )

    def interrupt(self, hard: bool = False) -> None:
        """Cooperates at task boundaries unless `hard`. Safe at any time."""
        self._live().interrupt(hard)

    def abandon(self) -> None:
        """The result will never be asked for: give the lease back."""
        self._live().abandon()

    def _live(self) -> BoxEpisode:
        if self._episode is None:
            raise HarnessBug(f"handle {self.run_id} has no episode behind it")
        return self._episode

    def cancel(self) -> None:
        self.interrupt(hard=True)


class SessionHandle:
    """What `resume()` returns — `CoderHandle`'s shape, one level up.

    An episode handle dies with its call; a session handle outlives every episode it
    contains. That is why resume is session-scoped: resuming the *run* restored
    packets and lost the design and the constraint ledger.
    """

    def __init__(
        self,
        session_id: str,
        info: SessionInfo | None = None,
        in_flight: CoderHandle | None = None,
        unfinished: tuple[str, ...] = (),
        pages: Callable[[Cursor], Page] | None = None,
    ) -> None:
        self.session_id = session_id
        self._info = info
        self._in_flight = in_flight
        self._unfinished = unfinished
        self._pages = pages
        """The engine's `journal` bound to this session: where `events` reads."""

    @property
    def reconciled(self) -> tuple[str, ...]:
        """What resume found before continuing: every run the session's episodes opened that
        its journal shows unfinished, oldest first. The last is `in_flight`, continued; the
        settling of each effect (file hashes, HEAD and trailers, command state) is the
        orchestrator's resume, and is journaled in the run it settles."""
        return self._unfinished

    @property
    def in_flight(self) -> CoderHandle | None:
        """The episode still running, if resume found one: continued from its journal, and
        already started, since no `coder` call waits on it."""
        return self._in_flight

    def events(self) -> Iterator[ScopeRecord]:
        """The session's records as they stand, in order, through `PacketEngine.journal`. A
        run's own records are its scope's: an `episode` record names the run."""
        if self._pages is None:
            raise HarnessBug(f"session handle {self.session_id} has no journal behind it")
        cursor = Cursor()
        while True:
            page = self._pages(cursor)
            if not page.records:
                return
            yield from page.records
            cursor = page.cursor

    def info(self) -> SessionInfo:
        if self._info is None:
            raise HarnessBug(f"session handle {self.session_id} has no info behind it")
        return self._info


# -- the conversation -------------------------------------------------


class TurnHandle:
    """What `turn()` returns: the live view of a turn, never its record. The session
    log is the record (commit, then emit), so a host that drops the stream loses nothing."""

    def __init__(self, session_id: str, queued: bool, live: LiveTurn) -> None:
        self.session_id = session_id
        self.queued = queued
        """The text waits behind a running turn, and this is that turn's handle."""
        self._live = live

    def events(self) -> Iterator[Event]:
        """The turn's events from its start, then live until it ends. Droppable."""
        return self._live.follow()

    def wait(self, timeout: float | None = None) -> None:
        """Returns when the turn and everything queued behind it has run, or a stop paused
        it. A loop error is re-raised as a `SessionError`; what was committed before it
        stays on the log, and words not yet delivered stay queued."""
        self._live.wait(timeout)


@dataclass
class _EngineRun:
    """`CoderRun` over a `coder` call's dispatch: a run handed over in the background, whose
    answer starts its watcher and says so, or an answer with no run (a stop that
    ended without an approval, a complete work, a refusal)."""

    run_id: str
    answer: Callable[[], ToolResult | Said]
    release: Callable[[], None]
    design_sha: str | None = None
    """The work the person approved at its stop, after any edit there (invariant 15); None
    for a call nothing was approved for."""
    moved: Callable[[], str | None] = lambda: None
    """Moves Zeus's view to where the session's `workspace` line now puts it, and says so when
    this run's call created the worktree."""

    def report(self) -> ToolResult | Said:
        """The call's answer: at once, whether or not a run goes on behind it."""
        told: str | None = None
        try:
            said = self.answer()
        finally:
            # Before the result is handed back, so the next call of this turn reads the
            # worktree; and even when the answer raised, since its worktree may exist already.
            told = self.moved()
        if told is None:
            return said
        if isinstance(said, ToolResult):
            return replace(said, notes=(*said.notes, told))
        return replace(said, text=f"{said.text}\n{told}")

    def abandon(self) -> None:
        self.release()


class _EngineTeam:
    """`TeamPort` over the engine's one episode: Zeus's `pause` and
    `tell`."""

    def __init__(
        self,
        pause: Callable[[], str],
        tell: Callable[[str, tuple[int, ...]], str],
    ) -> None:
        """`pause` and `tell` are the engine's, bound to the session (`pause_run`,
        `_tell_team`)."""
        self._pause = pause
        self._tell = tell

    def pause(self) -> Said | InvalidArgument:
        try:
            return Said(self._pause())
        except SessionError as exc:
            return InvalidArgument(
                "team",
                str(exc),
                do="nothing to pause: the team works only after a coder call",
            )

    def tell(self, text: str, sources: tuple[int, ...]) -> Said | InvalidArgument:
        try:
            return Said(self._tell(text, sources))
        except _TellRefused as refused:
            return refused.argument


def _paused_of(session: Session, runs: RunProjections) -> str | None:
    """`paused_run` over the session's fold, read under its lock: a run's thread writes the
    same session (her notes, its workspace line) while Zeus's reads it."""
    with session.lock:
        return paused_run(session.state, runs)


def _team_replied(
    session: Session, wake: Callable[[], None], tell: str, outcome: TellOutcome, text: str
) -> None:
    """Athena's reply to a tell, on the session log as Zeus reads it: a trigger-10
    notice, from the run's thread; a reply that needs him wakes him, as a run's end does."""
    notice = standing.team_replied(tell, outcome, text)
    with session.lock:
        standing.notice(session, notice)
    if notice.handle is not None:
        wake()


class _TellRefused(Exception):
    """A `tell` the engine refused, with the argument it names."""

    def __init__(self, argument: InvalidArgument) -> None:
        super().__init__(argument.why)
        self.argument = argument


def _refused_paused(delegation: Delegation, run_id: str, request: CoderRequest) -> str | None:
    """Why a `coder` call for another work is refused while the session's run `run_id`, opened
    with `request`, is paused: started, the paused work would be left beside a new
    one. The same work is a design in the same work folder (`WorkKey.owns`), or a one-task plan
    by the same title, whose changed brief is its amendment. None when the call may go
    on."""
    if request.task is not None:
        if delegation.work is None and delegation.title == request.task.title:
            return None
        again = f"coder(title={request.task.title!r}, brief)"
        what = f"the one-task plan {request.task.title!r}"
    else:
        pin = request.design.work if request.design is not None else None
        if pin is None:
            return None  # a design from before work folders: nothing names its work
        work = delegation.work
        if work is not None and WorkKey(dir=work.dir, name=work.name).owns(run_id, request):
            return None
        again = f"coder(work={pin.name!r})"
        what = f"work {pin.name}"
    return (
        f"run {run_id} of {what} is paused; continue it with {again}, or the person drops it "
        "with /drop"
    )


def _watch_run(
    handle: CoderHandle,
    session: Session,
    wake: Callable[[], None],
    ended: Callable[[], None] = lambda: None,
) -> None:
    """Watch a run handed over in the background, on a thread of its own:
    every such run's, whether `coder` or `/resume` started it. At its end: `ended` (the
    workspace's `after`, for a run whose turn took a `before`), its report as a trigger-4 notice
    carrying the handle it names, and Zeus woken to read it, as a job's exit wakes him."""

    def watch() -> None:
        try:
            outcome: CoderReport | Paused = handle.report()
        except BaseException as exc:
            logging.getLogger(__name__).exception(
                "run %s ended with a harness error", handle.run_id
            )
            outcome = CoderReport.before_run(
                "BLOCKED", f"the run failed in the harness: {exc}"
            ).model_copy(update={"run_id": handle.run_id})
        try:
            ended()
        except Exception:
            # The report is told whatever the snapshot did: a missing `after` is the turn's
            # code undo's to report, never a run's end left untold.
            logging.getLogger(__name__).exception(
                "run %s's after snapshot was not taken", handle.run_id
            )
        try:
            told = _tell_run(session, handle.run_id, outcome)
        except Exception:
            # The session closed under it (the process leaving): its next turn tells it.
            logging.getLogger(__name__).exception("run %s's end was not told", handle.run_id)
            return
        if told:
            wake()

    threading.Thread(target=watch, name=f"run-{handle.run_id}-watcher", daemon=True).start()


def _tell_run(session: Session, run_id: str, outcome: CoderReport | Paused) -> bool:
    """The run's end on the session's log, once: a run `tell_ended` told already, at a turn
    that began between its `run_end` and this, is not told again. A pause is said, and its end
    told when it ends after a resume. Returns whether Zeus is to be woken for it."""
    said = render_outcome(outcome)
    text = (said.content or "") if isinstance(said, ToolResult) else said.text
    carries = said.handle if isinstance(said, ToolResult) else None
    with session.lock:
        if isinstance(outcome, Paused):
            # Said, not woken for: whoever paused it (the person, or Zeus's `pause`) knows,
            # and a process leaving has nobody to wake.
            standing.notice(session, Notice(NoticeTrigger.RUN_ENDED, text))
            return False
        if told_ended(session.state, run_id):
            return False
        standing.notice(session, run_reported(run_id, text, carries))
        return True


class _EngineBox:
    """Zeus's `coder` goes through `PacketEngine.coder`: there is one door for delegation, so
    Zeus's call and a host's are the same call. The work's stop is asked
    first, at dispatch, before anything is leased: a design at stop 1, a one-task
    plan's brief at stop 2. Unapproved, the call does no work and its report says why. `coder()`
    then takes the lease and names the run, for the work as the person approved it."""

    def __init__(
        self,
        engine: PacketEngine,
        conversation: Conversation,
        snapshots: TurnSnapshots,
        port: GrantingPort,
        session: Session,
        moved: Callable[[], str | None] = lambda: None,
        docs: DesignDocs | None = None,
        *,
        work: Callable[[], WorkProgress | None] = lambda: None,
        resume: Callable[[str], CoderHandle] | None = None,
        runs: RunReader | None = None,
        running: Callable[[str | None], str | None] = lambda _session_id: None,
        paused: Callable[[], str | None] = lambda: None,
        attended: Callable[[], bool] = lambda: False,
    ) -> None:
        self._engine = engine
        self._attended = attended
        """Whether `coder` hands the work over in the background (`PacketEngine.attended`)."""
        self._running = running
        """What run is in flight on the engine, or only a session's (`OneEpisode.busy`): a
        `coder` call while one is, is answered before any stop is asked."""
        self._paused = paused
        """The session's latest run that started and has not ended (`paused_run`): while it
        is not running, a `coder` call continues its work or is refused."""
        self._work = work
        """Where the session's open work stands: what the dispatch decides
        by. Returning None, every `coder` call starts a work."""
        self._resume = resume
        """Continue a suspended run of the session's (`PacketEngine._continue_run`)."""
        self._runs = runs
        """The clone's runs, for the request a continued work's last run was approved with."""
        self._docs = docs
        """Zeus's `docs` port: where a design's work folder is, and how its README is written;
        None, and a design by `work` cannot be delegated."""
        self._conversation = conversation
        self._snapshots = snapshots
        self._moved = moved
        """Zeus's view following the session's workspace once a run ends."""
        self._port = port
        """The session's port, so the stop reaches its log."""
        self._session = session
        """Where a note or a cancel at the stop becomes the person's constraint."""

    def open(
        self, delegation: Delegation, *, session_id: str, cancel: threading.Event | None
    ) -> _EngineRun:
        engine = self._engine
        refused = self._refused_busy(session_id) if self._attended() else None
        held = self._paused_work(session_id) if refused is None else None
        if held is not None:
            refused = _refused_paused(delegation, *held)
        if refused is not None:
            return self._answered(CoderReport.before_run("BLOCKED", refused))
        with self._session.lock:
            default = session_mode(self._session.state, engine.config.workspace.mode)
        docs = self._docs
        workspace = chosen_workspace(
            delegation.workspace,
            folder_in_worktree=(
                None
                if delegation.work is None or docs is None
                else docs.folder_of(delegation.work).resolve() != engine.repo.resolve()
            ),
            default=default,
        )
        if delegation.work is not None:
            # A work that ran before continues where it stopped: resumed, continued in
            # a new run, or amended; only a work never approved is asked at stop 1 from nothing.
            found = self._work()
            if found is not None and found.runs:
                return self._continued(delegation, found, session_id, workspace, cancel)
        elif held is not None and held[1].task is not None:
            return self._continued_task(delegation, held, session_id, workspace, cancel)
        if delegation.work is not None:
            return self._dispatched(
                self._work_stop(delegation, session_id, workspace, cancel=cancel), session_id
            )
        with self._session.lock:
            attached = request_attachments(self._session.state)
        # A one-task plan's brief lists the files of the turn it answers, so
        # the person approves them with it at stop 2, as a design's Request section does.
        brief = replace(delegation, text=with_attached(delegation.text, attached))
        asked = stop_at_dispatch(
            brief,
            self._port,
            root=engine.repo,
            templates=engine.config.templates.configured(),
            workspace=workspace,
            cancel=cancel,
        )
        return self._dispatched(_carrying(asked, attached_files(attached)), session_id)

    def _dispatched(
        self,
        asked: CoderRequest | Stopped,
        session_id: str,
        mode: Literal["FRESH", "CONTINUE", "AMEND"] = "FRESH",
        *,
        supersedes: tuple[str, str] | None = None,
    ) -> _EngineRun:
        """The run for `asked`, the request a stop approved, handed over in the background; or
        the call's report when the stop ended without an approval. `supersedes` is a suspended
        run this one replaces, and why: ended only once this run is open, so a run that cannot
        open (another session's took the engine while the person answered the stop, or the
        engine is closing) leaves the suspended one to be continued."""
        engine = self._engine
        if isinstance(asked, Stopped):
            note = asked.outcome.note if asked.outcome is not None else None
            if note:
                # The person's words at the stop bind the work that follows, as typed.
                session = self._session
                record_refusal_note(session, note, session.next_seq - 1, scope="design")
            return self._answered(asked.report)
        # Zeus's dispatch writes this run's `episode` line, with the turn and the constraints
        # only it knows; the door writes one for a host's run alone.
        handle = engine.coder(asked, session_id=session_id, caller="zeus")
        # The turn's `before` of this run's workspace: acquired, and nothing run in it yet
        # A snapshot that cannot be taken gives the lease back.
        attended = self._attended()
        with given_back_on_failure(handle.run_id, handle.abandon):
            if supersedes is not None:
                end_superseded(_roots(engine.repo), *supersedes)
            site = handle.workspace
            self._snapshots.before(site)
            if attended:
                self._snapshots.hold(site)
        if attended:

            def abandon() -> None:
                # The call's answer is never asked for, so no watcher ends the hold: the run
                # did nothing, and the turn's own end takes its workspace's `after`.
                try:
                    handle.abandon()
                finally:
                    self._snapshots.run_ended(site)

            # Never attached to Zeus's conversation: a stop of his turn is his alone, and the
            # team is paused or dropped by the person's own command.
            return _EngineRun(
                handle.run_id,
                partial(self._handed_over, handle, mode),
                abandon,
                asked.work_sha,
                self._moved,
            )
        return self._waited(handle, asked.work_sha)

    def _waited(
        self,
        handle: CoderHandle,
        design_sha: str | None = None,
        *,
        answer_id: str | None = None,
    ) -> _EngineRun:
        """Unattended, the call waits for the run and answers its report (as
        amended). A stop that landed while the run was being named, before `interrupt()` could
        see the episode, reaches it as it is attached: its hardness is read
        from the conversation's own token, under the lock `interrupt()` takes, so the episode
        gets each stop exactly once. `answer_id` is the run id the call's answer is recorded
        under, when it is not the run's own (a resumed run's line is the call that opened
        it)."""
        conversation = self._conversation
        conversation.attach_episode(handle)

        def answer() -> ToolResult | Said:
            try:
                outcome = handle.report()
            finally:
                conversation.detach_episode()
            return render_outcome(outcome)

        def abandon() -> None:
            conversation.detach_episode()
            handle.abandon()

        return _EngineRun(answer_id or handle.run_id, answer, abandon, design_sha, self._moved)

    def _answered(self, outcome: CoderReport | Complete) -> _EngineRun:
        """A `coder` call answered with no run: nothing is leased."""
        said = render_outcome(outcome)
        return _EngineRun(new_run_id(), lambda: said, lambda: None)

    def _handed_over(self, handle: CoderHandle, mode: str) -> Said:
        """The run goes on on its own: its watcher waits for its end, and the
        call is answered now."""
        snapshots = self._snapshots
        _watch_run(
            handle,
            self._session,
            self._conversation.wake_due,
            lambda: snapshots.run_ended(handle.workspace),
        )
        return Said(
            f"RUNNING · run {handle.run_id} · mode {mode}: the team works on it on its own. "
            "You are told how it ended at a later turn; meanwhile, answer the user"
        )

    def _continued(
        self,
        delegation: Delegation,
        found: WorkProgress,
        session_id: str,
        workspace: WorkspaceMode,
        cancel: threading.Event | None,
    ) -> _EngineRun:
        """`coder` on a work that ran before:
        complete, it is said so; its design unchanged and its run suspended, the run resumes;
        its run ended, a new run continues it with the plan it approved; its design changed,
        stop 1 asks for the change, and a new run continues it, amended, superseding a
        suspended one. Nothing done is redone, and no stop is asked for what was approved."""
        latest = found.run
        if latest is None:
            raise HarnessBug("a work with runs reached its dispatch with no latest run")
        if latest.state == "live":
            # Another caller's run of this work, on this engine's one episode (a host's, or
            # one a `/resume` started): nothing is opened beside it.
            return self._answered(
                CoderReport.before_run(
                    "BLOCKED",
                    f"run {latest.run_id} of this work is still running; one run at a time. "
                    "Wait for it to end, then call coder with this work again",
                )
            )
        if complete(found) and not found.design.changed:
            return self._answered(
                Complete(delegation.work.name if delegation.work is not None else "")
            )
        if latest.state == "suspended" and not found.design.changed:
            return self._resumed(latest.run_id)
        before = self._runs.request(found.runs[-1]) if self._runs is not None else None
        if before is None or before.design is None:
            raise HarnessBug(f"run {found.runs[-1]} of a work has no design to continue")
        mode: Literal["CONTINUE", "AMEND"] = "CONTINUE"
        request: CoderRequest | Stopped = CoderRequest(
            design=before.design, workspace=workspace, attachments=before.attachments
        )
        if found.design.changed and found.design.approved and found.design.current:
            mode = "AMEND"
            changes = tree_diff(
                self._root_of(delegation),
                found.design.approved,
                found.design.current,
                f"the design approved in run {found.runs[-1]}",
            )
            request = self._work_stop(
                delegation,
                session_id,
                workspace,
                earlier=request_section(before.design.text),
                changes=changes,
                kept=before.attachments,
                cancel=cancel,
            )
        if isinstance(request, Stopped):
            return self._dispatched(request, session_id)
        supersedes = (
            (
                latest.run_id,
                "superseded: the person approved a change to its design, and a new run "
                "continues the work",
            )
            if latest.state == "suspended"
            else None
        )
        return self._dispatched(
            request.model_copy(
                update={
                    "continues": Continuation(
                        mode=mode,
                        prior_runs=found.runs,
                        plan_run=found.plan_run,
                        plan_version=found.plan_version,
                        plan_sha=found.plan_sha,
                        plan_bound_to=found.plan.bound_to,
                        amended_from=found.design.approved if mode == "AMEND" else None,
                    )
                }
            ),
            session_id,
            mode,
            supersedes=supersedes,
        )

    def _resumed(self, run_id: str) -> _EngineRun:
        """RESUME: the suspended run `run_id` continues through the pause door,
        no stop asked. It goes on as any run this engine hands over does: in the background,
        or waited for. The call is answered under an id of its own: the run's episode line is
        the call that opened it, and a second would be refused."""
        if self._resume is None:
            raise HarnessBug("a suspended run reached a box that cannot resume one")
        try:
            handle = self._resume(run_id)
        except SessionError as exc:
            # Its worktree was handed off while it was paused: said, never raised into the
            # turn, as `/resume` says it.
            return self._answered(CoderReport.before_run("BLOCKED", str(exc)))
        if not self._attended():
            return self._waited(handle, answer_id=new_run_id())
        return _EngineRun(
            new_run_id(), partial(self._handed_over, handle, "RESUME"), lambda: None
        )

    def _refused_busy(self, session_id: str) -> str | None:
        """Why a `coder` call is answered at once while a run is in flight on the engine, or
        None: one run at a time, the session's own or another session's. Only the
        session's own run's end is told to its Zeus."""
        mine = self._running(session_id)
        if mine is not None:
            return (
                f"{mine} is working; one run at a time. You are told how it ends; call coder "
                "again after that"
            )
        other = self._running(None)
        if other is not None:
            return (
                f"{other} of another session is working on this engine; one run at a time, "
                "and you are not told how it ends. Tell the user, and call coder again later"
            )
        return None

    def _paused_work(self, session_id: str) -> tuple[str, CoderRequest] | None:
        """The session's paused run and the request it was approved with, when nothing of the
        session's is running: what a `coder` call for the same work continues. None when there
        is none, or its request cannot be read."""
        if self._runs is None or self._running(session_id) is not None:
            return None
        run_id = self._paused()
        if run_id is None:
            return None
        request = self._runs.request(run_id)
        return (run_id, request) if request is not None else None

    def _continued_task(
        self,
        delegation: Delegation,
        held: tuple[str, CoderRequest],
        session_id: str,
        workspace: WorkspaceMode,
        cancel: threading.Event | None,
    ) -> _EngineRun:
        """A one-task plan's `coder`, by its title, while its run is suspended (its brief in the
        design's place; another title is refused before this, by
        `_refused_paused`): the same brief resumes it, no stop; a changed one is an amendment,
        asked at stop 2 with what changed, and once approved a new run does the work, the
        suspended one ended as superseded so it is never resumed beside it."""
        run_id, before = held
        task = before.task
        if task is None:
            raise HarnessBug(f"run {run_id} is not a one-task plan's")
        if sha256_hex(delegation.text) == task.sha or delegation.text == self._sent_brief(
            run_id
        ):
            # The brief approved, or the one Zeus sent before the person edited it at stop 2:
            # either way nothing changed since, and the run goes on with the approved one.
            return self._resumed(run_id)
        engine = self._engine
        asked = stop_at_dispatch(
            delegation,
            self._port,
            root=engine.repo,
            templates=engine.config.templates.configured(),
            workspace=workspace,
            cancel=cancel,
            notice=text_diff(
                task.brief, delegation.text, f"the brief approved in run {run_id}"
            ),
        )
        if isinstance(asked, Stopped):
            return self._dispatched(asked, session_id)
        return self._dispatched(
            asked,
            session_id,
            "AMEND",
            supersedes=(
                run_id,
                "superseded: the person approved a change to its brief, and a new run does "
                "the work",
            ),
        )

    def _sent_brief(self, run_id: str) -> str | None:
        """The brief Zeus sent in the `coder` call that opened run `run_id`, as he sent it
        (invariant 19), before any edit the person made at its stop; None when no call of this
        session's log opened it."""
        with self._session.lock:
            state = self._session.state
            for call in state.calls_upto(state.last_seq):
                line = state.result(call.call_id) if call.tool == CODER else None
                if line is not None and line.run_id == run_id:
                    brief = call.args.get("brief")
                    return brief if isinstance(brief, str) else None
        return None

    def _root_of(self, delegation: Delegation) -> Path:
        docs, work = self._docs, delegation.work
        if docs is None or work is None:
            raise HarnessBug("a design by work reached a box with no docs port")
        return docs.folder_of(work)

    def _work_stop(
        self,
        delegation: Delegation,
        session_id: str,
        workspace: WorkspaceMode,
        *,
        earlier: str | None = None,
        changes: str | None = None,
        kept: Sequence[AttachedFile] = (),
        cancel: threading.Event | None = None,
    ) -> CoderRequest | Stopped:
        """Stop 1 for a design in its work folder: its README with the
        harness's two sections, written through Zeus's port, then the folder's stop. An amended
        design keeps the Request it was approved with, `earlier`, and adds the person's
        new turn after it; `changes`, what changed since that approval, is shown with it, and
        `kept`, the files that approval carried, still go with the work."""
        engine, docs, work = self._engine, self._docs, delegation.work
        if docs is None or work is None:
            raise HarnessBug("a design by work reached a box with no docs port")
        root = docs.folder_of(work)
        readme = root / work.dir / work.name / "design" / "README.md"
        if not readme.is_file():
            why = "not asked, not run: design/README.md does not exist: write it with doc"
            return Stopped(CoderReport.before_run("BLOCKED", why), None)
        with self._session.lock:
            state = self._session.state
            bound = binding_constraints(state, ())
            request = requested(state)
            attached = attached_files(request_attachments(state))
        if isinstance(bound, InvalidArgument):
            raise HarnessBug(f"the constraints parse accepted no longer bind: {bound.why}")
        if earlier is not None:
            request = amended_request(earlier, request)
        text = inserted(readme.read_text(encoding="utf-8"), request, bound)
        work_dir, commit = engine.config.work.dir, engine.config.work.commit

        def shown() -> str | None:
            once = notice_once(_roots(engine.repo).repo_local, work_dir, commit)
            said = [n for n in (changes, once) if n]
            return "\n\n".join(said) if said else None

        asked = stop_at_work(
            delegation,
            self._port,
            repo=engine.repo,
            root=root,
            templates=engine.config.templates.configured(),
            workspace=workspace,
            readme=text,
            write=lambda t: docs.write_readme(work, t),
            scope=session_id,
            notice=shown,
            cancel=cancel,
        )
        return _carrying(asked, (*kept, *attached))


def _carrying(
    asked: CoderRequest | Stopped, files: Sequence[AttachedFile]
) -> CoderRequest | Stopped:
    """An approved request with the files approved with it, each once; a
    stop that ended without an approval, as it was."""
    if isinstance(asked, Stopped) or not files:
        return asked
    by_id = {f.id: f for f in files}
    return asked.model_copy(update={"attachments": tuple(by_id.values())})


# -- the facade --------------------------------------------------------------


def _driven(
    refresh: Callable[[], None],
    moved: Callable[[], object],
    namer: Callable[[], None],
    turn: Callable[[], Iterator[Event]],
) -> Iterator[Event]:
    """One loop turn as the session's worker drives it (`_compose`): the ledger restored from
    the family's projection; a `workspace` line written outside Zeus's own `coder` call
    (a host's run, a handoff) followed, its frame saying so; then the turn, the oldest queued
    word or one a job's exit opens; then, once its words have streamed, the session's
    name, which `namer` makes only the first time it can."""
    refresh()
    moved()
    yield from turn()
    namer()


class PacketEngine:
    """Composition root and the only public entry point."""

    def __init__(
        self,
        repo: Path,
        config: EngineConfig,
        interaction: InteractionPort | None = None,
        *,
        trust: TrustDecision | None = None,
    ) -> None:
        """`trust` is whether this clone's owner accepted the repository's executable config,
        as the loader answered it for **this** config (`Resolved.trust`). The
        engine receives it and never re-derives it: it holds the validated config, not the
        project layer the digest is taken over. One taken for another config is refused.
        Absent, the safe reading holds — a repository's removals from the deny list stay
        inert — so a host that does not ask gets the whole list.

        **The deny list is resolved here, once**, and `_denylist` is the one value
        the run's consumers take: the run header reads it now, Zeus's shell `Executor` and
        `Reader` take it, and the applier's `secret_path` gate, the approval facts
        and the citation check take it as their builders land. Until then
        those still hold the built-in list, which is stricter, never looser, than this one
        would be."""
        if not isinstance(config, EngineConfig):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise HarnessBug(f"config must be an EngineConfig, got {type(config).__name__}")
        if trust is not None and trust.engine is not config:
            raise HarnessBug(
                "the trust decision was taken for a different config than the engine was "
                "given; ask Resolved.trust() of the load whose .engine this is"
            )
        self.repo = repo
        self.config = config
        self._accepted = trust is not None and trust.accepted
        self._denylist = config.secrets.denylist(accepted=self._accepted)
        # The one `[[commands]]` table, built once beside the deny list and given to
        # every executor this engine builds, so every reader of a command agrees.
        self._commands = config.command_table(accepted=self._accepted)
        # An absent port means unattended, not "prompt somehow". Every ask point
        # resolves to its declared default, which is why `Question.default` is required —
        # a question with no default would have nothing to fall back to here.
        self.interaction = interaction if interaction is not None else NonInteractive()
        self.attended = False
        """Whether a person is attached to this engine's sessions (as amended
        2026-10-27): set by the server that gives it their port. Attended, `coder` hands the
        work over in the background and its end wakes Zeus; unattended (the eval, a script, a
        host), nobody talks meanwhile, so `coder` waits for the run and answers its report."""
        self._components: dict[str, object] = {}
        self._caches_lock = threading.Lock()
        """Guards `_caches` only. A leaf: nothing else is taken under it."""
        self._caches: tuple[ReplayCache, RunProjections] | None = None
        """The engine's one cache of session folds and one of run journals, built on first
        use and passed to every reader it composes, so a turn reads only what changed."""
        self._progresses: Progresses | None = None
        """The open works' progress, read again only when it moved, over the runs' requests,
        each read once (`_work_progresses`). Guarded by `_caches_lock`."""
        self._conversations = Conversations(self._compose)
        """The sessions open here, each with its loop and lifecycle."""
        self._episodes = OneEpisode()
        """The one episode this engine runs at a time."""
        self._indexed = threading.Event()
        self._indexed.set()
        """Set unless `index_in_background` is building the repository's index: a
        session composed while it was clear is rebuilt at its next turn, so its graph is the
        index. What takes the lease meanwhile waits in `init.lease.take_write_lease`."""
        self._index_failed: str | None = None
        """Why the background build failed, once it has; written before `_indexed` is set."""
        self._index_progress = IndexProgress("preparing")
        """How far the background build has got, reported while `_indexed` is clear.
        Written by the build's thread alone and read by `liveness()`: one reference to a frozen
        record, so a read sees one whole record and needs no lock."""
        self._starts = threading.Lock()
        """Held while a switch is checked and written, and while a turn or a run starts:
        nothing starts between a switch's "nothing runs" and its line, and each switch is
        checked against the wiring the one before it left. Never held across a build, a turn
        or a run, only across their start."""
        self._handing_off: set[str] = set()
        """Sessions whose handoff is under way, guarded by `_starts`: no run
        starts and no switch is made on one, and the git work runs outside the lock."""
        # What the load found and did not refuse — a declared rung no role
        # runs on — said once, as the engine is built. The run header's `models` kind reads
        # the same notices.
        for notice in config.notices:
            logging.getLogger(__name__).warning("config: %s", notice)
        self._wire()

    def _wire(self) -> None:
        """Construct the components that exist today.

        Deliberately empty of run machinery: Phase 0 has contracts, journal, client,
        config and profiles, and none of them is a component the engine holds open. The
        dict exists so `test_composition.py` can assert what is wired without reaching
        into private modules, and so each phase's task has one obvious place to add to.
        A builder that constructs a deny-list consumer here passes `self._denylist`.
        """

    # ---- index -------------------------------------------------------------

    def index_in_background(self) -> bool:
        """Build the repository's index on a thread of its own when it has none, so a
        person's screen opens at once rather than after `init`: Zeus's reads, search and shell
        work meanwhile, his index-backed tools answer that there is no index yet, and anything
        that takes the repository's lease (a run, a handoff, an undo) waits for it.
        `liveness().indexing` says so until it is done. True when a build
        started; False when the repository is initialised already."""
        if _is_initialised(self.repo):
            return False
        self._index_progress = IndexProgress("preparing")
        self._indexed.clear()
        repository = _detect_root(self.repo).repository_id

        def noted(progress: IndexProgress) -> None:
            self._index_progress = progress

        def build() -> None:
            try:
                # Registered first: a lease this build holds is waited for by this process,
                # never refused as busy (`init.lease.building`).
                with _building(repository):
                    _init(
                        self.repo,
                        denylist=self._denylist,
                        sandbox=self._index_sandbox(),
                        progress=noted,
                    )
            except Exception as exc:  # the build's failure is said, never raised on a thread
                self._index_failed = str(exc)
            finally:
                self._indexed.set()

        threading.Thread(target=build, name="chickadee-index", daemon=True).start()
        return True

    @property
    def indexing(self) -> str | None:
        """What the background build of the index is doing: building, or why it
        failed; None when there is none, or it is done."""
        if not self._indexed.is_set():
            return "building this repository's index"
        if self._index_failed is not None:
            why = self._index_failed
            return f"the index could not be built ({why}); `chickadee init` says why"
        return None

    @property
    def index_progress(self) -> IndexProgress | None:
        """How far the background build has got: set while it runs and never
        otherwise, so a reader tells a build from a failure without reading `indexing`'s
        words."""
        progress = self._index_progress  # read first: a build that ends meanwhile says None
        return None if self._indexed.is_set() else progress

    def init_repo(self) -> _InitReport:
        """Prepare the repository. Deterministic; no model calls, no network."""
        return _init(self.repo, denylist=self._denylist, sandbox=self._index_sandbox())

    def reindex(self, changed: Sequence[Path] | None = None) -> IndexReport:
        """Refresh the checkout's code graph under the repository WRITE lease, with
        the engine's deny list. `RepoNotIndexed` before `init_repo()`."""
        return _reindex(
            self.repo, changed, denylist=self._denylist, sandbox=self._index_sandbox()
        )

    def _index_sandbox(self) -> IndexSandbox:
        """The index's language servers run on the engine's backend, under its
        `require_sandbox`, as every other harness spawn does."""
        sandbox = self.config.sandbox
        return IndexSandbox(_select_backend(sandbox.backend), sandbox.require_sandbox)

    # ---- discovery ----------------------------------------------------------

    @not_built(
        "a task not yet opened (14-scout §2: locate as a host entry point; Zeus's is built)"
    )
    def locate(self, question: str) -> Findings | Unresolved:
        """`-> Findings | Unresolved` (14). The only discovery that spends a model call, on
        the cheap rung, and it returns pointers rather than prose."""
        raise NotImplementedError(_PHASE)

    # ---- the conversation ---------------------------------------------

    def turn(
        self,
        session_id: str,
        text: str,
        *,
        workspace: WorkspaceMode | None = None,
        attachments: Sequence[tuple[str, bytes]] = (),
    ) -> TurnHandle:
        """The person's words to the session's Zeus loop. One loop per open
        session, on a worker thread of its own, so this returns before the model answers.
        While a turn runs the text is queued and delivered at its next boundary, and
        the running turn's handle comes back with `queued=True`. Never refused for being
        early. `workspace`: where the person said this turn's work runs (`--worktree`,
        `--in-place`), carried on their turn line.

        Words are delivered in the order they were typed, each as the next loop turn, and a
        word typed before the worker picks the first one up does not overtake it. A session not
        yet open here is built on its worker without waiting for it: the word waits
        in its queue, and a build that fails says why on the turn's stream.

        `attachments`: the files the person sent with the words, each `(name, bytes)`.
        Checked here as the client checked them, before anything is queued: one that is not an
        image, a PDF or a text file, or that breaks a cap, refuses the turn whole with
        `AttachmentRefused`, and nothing is recorded."""
        checked: tuple[CheckedFile, ...] = ()
        if attachments:
            result = check_turn(attachments, self.attach_caps())
            if isinstance(result, RefusedFile):
                raise AttachmentRefused(result.render())
            checked = result
        if self._conversations.get(session_id) is None:
            # An id no session has is the person's mistake (item 14), refused now: the build
            # it would start runs off this call, and would say so only on the stream. A listing,
            # not a read: the log is replayed by the build, off this call.
            self._sessions().require(session_id)
        conversation = self._conversations.start(session_id)
        with self._starts:
            live, queued = conversation.turn(text, workspace, checked)
        return TurnHandle(session_id, queued, live)

    def preview_attachment(self, session_id: str, name: str, data: bytes) -> AttachmentPreview:
        """`-> AttachmentPreview`: what `data` would be on this session's
        next turn, by the check a turn runs, delivered as Zeus's rung now takes it, and whether
        its text holds a secret. A client's chip, before anything is sent: the client speaks
        the wire alone, so the one check is asked of the server. Records nothing;
        `AttachmentRefused` as a turn would be refused."""
        result = check(name, data, self.attach_caps())
        if isinstance(result, RefusedFile):
            raise AttachmentRefused(result.render())
        config = self._session_config(self._switches_of(session_id, strict=True))
        # A scratch home: the bytes it keeps go when the preview returns, recorded nowhere.
        scratch = Scratch()
        reader = Reader(
            scratch,
            HandleResolver(self.repo, scratch.blob),
            vision=config.vision_for(config.rung_name_for("zeus")),
            rules=ReadRules.of("zeus", role_limits(config.resolved_limits, "zeus")),
        )
        minted = reader.attach(
            result, inline_bytes=config.harness_limit("turn.text_inline_bytes")
        )
        text = minted.text()
        return minted.preview(secret=text is not None and self.holds_secret(text))

    def attach_caps(self) -> AttachCaps:
        """What a turn may carry: the limits register's `turn.*` rows as this config
        sets them, and the edge an image is downscaled to. What `GET /` tells a client."""
        return AttachCaps.of(self.config)

    def spawn_agent(
        self, session_id: str, profile: str, task: str, *, call_id: str
    ) -> AgentReport | InvalidArgument:
        """One round of `profile` on `task`, as Zeus's `spawn_agent` runs it, its report as a
        value: the session's own agents, built with it, so a host measures the same wiring
        Zeus spawns (the researcher's evaluation). Recorded on the session's
        log as any agent's round is; no Zeus turn, and no handles."""
        if self._conversations.get(session_id) is None:
            self._sessions().require(session_id)
        built = self._conversations.open(session_id).opened()
        if built is None or built.agents is None:
            raise SessionError(f"session {session_id} did not open, so it has no agents")
        return built.agents.run(None, profile, task, (), call_id=call_id)

    def ask_model(self, model: str, prompt: str) -> ModelAnswer:
        """`prompt` on the catalog model `model`, once, outside any session: a host's own use
        of a model, such as the researcher's evaluation's judge. Nothing is
        journaled or charged to a session; the answer says what it cost, by the model's price
        row. `ConfigError` for a model the catalog lacks, or one with no price row, before
        anything is sent."""
        if model not in self.config.models:
            known = ", ".join(sorted(self.config.models))
            raise ConfigError(f"no model {model!r} in the catalog; it has {known}")
        prices = PriceTable.from_config(self.config.prices)
        prices.price_for(model)
        endpoint = self.config.models[model]
        request = ChatRequest(
            messages=(Message(role="user", content=prompt),),
            params=first_attempt_params(endpoint),
        )
        return ask_once(client_for(endpoint), request, lambda u: prices.cost(u, model).total)

    def interrupt(self, session_id: str, hard: bool = False) -> None:
        """The session's stop. Soft: the loop's next tool boundary. Hard: that, and the
        token now — a shell is reaped, the model request abandoned — and an episode in
        flight gets the interrupt through its `CoderHandle`. Queued input is kept. A
        stop while the session is still being built stops the turn it is being built for.
        A session not open here has nothing running to stop."""
        conversation = self._conversations.get(session_id)
        if conversation is not None:
            conversation.interrupt(hard)

    def _design_docs(
        self,
        session: Session,
        view: SessionView,
        workspace: Path,
        config: EngineConfig,
        moved: Callable[[], str | None],
    ) -> DesignDocs:
        """Zeus's `docs` port: his `design/` of the session's open work, written in his
        view's root and recorded as his shell's writes are."""

        def open_worktree(goal: str) -> DraftRefused | str | None:
            """His first `doc` call of a design in a worktree session (decision 4b): the
            session's worktree, opened as a `coder` call opens it and its lease given back, and
            the view moved there; a dirty checkout seeds it, as for `coder`."""
            try:
                opened = session_worktree(
                    self.repo, session, session.session_id, goal, config.workspace
                )
            except ConfigError as exc:
                # A configuration the worktree cannot be made under: the person chooses first.
                return DraftRefused(
                    "DENIED",
                    str(exc),
                    do="tell the person what the reason says and ask them to choose; write "
                    "the design once they have",
                )
            opened.release()
            return moved()

        def in_place() -> bool:
            with session.lock:
                return session_mode(session.state, config.workspace.mode) == "in_place"

        return DesignDocs(
            session,
            lambda: view.root,
            workspace,
            work_dir=config.work.dir,
            in_place=in_place,
            commit=config.work.commit,
            open_worktree=open_worktree,
            edits=lambda root: WorktreeEdits(
                root,
                self._denylist,
                SyntaxChecker(root, ProfileSet.load(repo=root).all()),
                untracked_max_bytes=config.undo.untracked_max_bytes,
                prefs=self._prefs,
            ),
            writes=lambda root: ShellWrites(
                session, root, commits=root != workspace, prefs=self._prefs
            ),
            relocate=partial(self._relocate_work, session),
            # B5: the design is frozen while an episode builds it.
            running=self._episodes.busy,
        )

    def _relocate_work(
        self, session: Session, root: Path, unusable: str, why: str
    ) -> str | None:
        """`[work] dir` cannot be used under `root` (`why`), so the first free
        fallback folder is offered to the person, through the session's port so the question
        and its answer are on the log. A yes keeps it for this clone (`state.local`, read as the
        `local` layer by every later load) and moves this engine's own `work.dir`, so the rest
        of the session and every session after it use it. A no, an unattended run (the
        default is `no`) or no free folder keeps the refusal; None then."""
        found = free_work_dir(root, unusable)
        if found is None:
            return None
        if self._port(session).ask(folder_question(found, why)).choice != FOLDER_YES:
            return None
        record_work_dir(_roots(self.repo).repo_local, found)
        work = self.config.work.model_copy(update={"dir": found})
        self.config = self.config.model_copy(update={"work": work})
        return found

    def _worktree_moves(
        self, session: Session, config: EngineConfig, moved: Callable[[], str | None]
    ) -> WorktreeMoves:
        """Zeus's `worktree` port: the session's worktree made the one way a
        `coder` call makes it, its lease given back at once; the view following the session's
        `workspace` line; and the handoff from inside the turn."""

        def open_worktree() -> None:
            opened = session_worktree(
                self.repo, session, session.session_id, "a worktree", config.workspace
            )
            opened.release()

        return WorktreeMoves(
            session,
            self._port(session),
            open_worktree=open_worktree,
            follow=moved,
            land=partial(self._land_in_turn, session.session_id),
            review=partial(self.handoff, session.session_id),
        )

    def _compose(self, conversation: Conversation, stack: ExitStack) -> Built:
        """The loop, from the engine's own values and nothing else.

        The deny list is `self._denylist` everywhere it is read — the shell's `Executor`
        (with its `resolve()` over the workspace and `$HOME` as `deny_read`) and the
        `Reader` — so `read` and `shell` cannot disagree about what is secret.

        Wired as far as the parts exist, and no further:
        - the shell's `ShellAuthority` is the person, through `confirm_action` on the
          engine's port: unattended (`NonInteractive`), a command above T0 is
          refused rather than confirmed;
        - the graph's navigator and `locate` (`_navigation`), over the checkout's
          index; with none, the navigate family answers `INDEX_UNAVAILABLE` and Zeus's own
          ladder is the fallback.

        Runs on the conversation's worker. Each resource is put on `stack` as it is
        opened (the session, then the client), so a build that raises releases exactly what
        it opened, and closing the conversation releases the client, then the session.

        Composed from the session's wiring: the engine config with the session's
        `/rung` switches applied. `Built.stale` says when a switch has moved it since, and
        the conversation rebuilds before its next word."""
        config = self._session_config(self._switches_of(conversation.session_id))
        where = _roots(self.repo)
        workspace = self.repo.resolve()
        # The rung the `zeus` role runs on, the config's one answer.
        zeus_rung = config.rung_name_for("zeus")
        model = config.rungs[zeus_rung]
        endpoint = config.models[model]
        prices = PriceTable.from_config(config.prices)
        # A model with no price row raises at its first charge, after the call was paid
        # for; asked here, before the session is opened, the person learns it before
        # anything is spent or written.
        prices.price_for(model)
        # The session's skills, discovered before anything is opened:
        # a malformed skill or one named for a built-in refuses the session, naming the file.
        catalog = discover(SkillRoots.of(where, external=config.skills.external))

        root = sessions_root(where)
        session_id = conversation.session_id
        folds, runs = self._caches_for(where)
        # The writer and the session's jobs outlive a rebuild: opened by the
        # first build, handed as they are to every later one, closed with the conversation.
        opener = partial(self._lasting, conversation, root, folds, config)
        lasting = conversation.lasting(opener)
        session, jobs = lasting.session, lasting.jobs

        records = JournalRecords(where, runs)
        # The session's ledger starts from what its family has spent — every
        # session sharing its root, each counting what it wrote — never from zero, so a
        # restart or a fork cannot reset the cap. A cap crossed before is still crossed
        # and the projection is read again before every turn, so a sibling's spend
        # is seen at the next one.
        base = BudgetCaps.from_config(config.spend)
        # One ledger per session: Zeus's own calls are recorded on the session log as
        # they are charged, and every loop holds one rung's view of it, never the
        # ledger itself — Zeus his rung's, the box the ledger as its `BudgetPort`,
        # cutting a view per call. The declared rungs are handed over once.
        ledger = CostLedger(prices, base, rungs=config.rung_names, record=session.record_spend)

        def refresh() -> None:
            # The family's cap moves with its latest `cap` line, read with the
            # spend before every turn, so a raise, a cut or a lift lands at the next turn.
            # And each rung's, beside it.
            ledger.restore(
                *ledger_inputs(
                    session_id,
                    session.state,
                    where,
                    folds,
                    runs,
                    base,
                    config.spend.rungs,
                    config.rung_names,
                )
            )

        refresh()
        # The configured backend, the one the header names: `configured` is the one
        # place the engine chooses, so the shell, the box and the header cannot disagree.
        backend = _select_backend(config.sandbox.backend)
        level = level_of(backend)
        # The session's hooks, bound as it opened: Zeus's, and the scout's under them.
        hooks = self._zeus(lasting.hooks, config, session)
        client = client_for(endpoint)
        stack.callback(client.close)

        def rooted(root: Path) -> Rooted:
            """Zeus's read parts over one root: built at composition over the
            checkout, and again over the session's worktree when the view moves there."""
            resolver = HandleResolver(
                workspace=root,
                session_blob=session.blob,
                run_blob=records.blob,
                journal_text=records.journal_text,
            )
            graph = opened_until_closed(root, stack)
            reader = Reader(
                SessionHome(session),
                resolver,
                vision=config.vision_for(zeus_rung),
                denylist=self._denylist,
                languages=ProfileSet.load(repo=root),
                index=graph,
                diagnostics=graph,
                # `at="@changed"` in the session's worktree, against the head it was made
                # from; the person's checkout has no base, and says none is set.
                changes=worktree_changes(root, latest_workspace(session.state)),
                # His caps stay as they are; his held rows answer a re-read.
                rules=ReadRules.of("zeus", role_limits(config.resolved_limits, "zeus")),
            )
            executor = Executor(
                backend,
                require_sandbox=config.sandbox.require_sandbox,
                denylist=self._denylist,
                # Walked per root, as `resolve()` asks: a secret written in one session is
                # masked in the next. Never per spawn, and never under the engine-wide lock.
                # The checkout's own masks too: a command in the worktree can still name a
                # file in the checkout by its absolute path, and its secrets stay masked.
                deny_read=_resolve_denylist(
                    self._denylist,
                    list(dict.fromkeys([root, workspace, Path.home()])),
                    memo=WalkMemo.shared(),
                ),
                commands=self._commands,
                # The repository's tool cache writable, and the harness's keys out of an
                # approved command's environment.
                tool_cache=tool_cache(repository_key(root)),
                withheld_env=config.endpoint_keys,
            )
            shell = Shell(
                executor,
                root,
                lambda data: reader.store("output", data),
                authority=PersonAuthority(
                    self._port(session), partial(conversation.interrupt, hard=True)
                ),
                allow_unsandboxed_shell=config.sandbox.allow_unsandboxed_shell.zeus,
                # Every command that may write is recorded: committed in the
                # session's worktree, never in the person's checkout, and source it
                # changed or created put back (item 14), judged by every installed profile:
                # an empty repository's first files have none detected.
                writes=ShellWrites(
                    session,
                    root,
                    commits=root != workspace,
                    profiles=ProfileSet.load(repo=root).all(),
                    prefs=self._prefs,
                    # A command's write into the work folder is put back: `doc` writes it
                    # Read live, as the person may choose another folder
                    # for this repository mid-session.
                    work_dir=lambda: self.config.work.dir,
                    beside=jobs,
                ),
                prefs=self._prefs,
                jobs=jobs,
                team=partial(self._team_run, session_id),
            )
            navigator, locator = self._navigation(
                config,
                session,
                root,
                graph,
                reader,
                executor,
                ledger,
                prices,
                stack,
                hooks.hooks,
            )
            found = locator if locator is not None else NoLocator()
            # `grep` and `glob` over this root, under the shell's backend and masks.
            search = Search(
                root=root,
                backend=backend,
                denylist=self._denylist,
                deny_read=executor.deny_read,
            )

            def agent_shell(store: HandleStore) -> ExecPort:
                # A review agent's shell: T0 only, never asked, no sandbox waived, its
                # output in the agent's own reader.
                return Shell(
                    executor,
                    root,
                    lambda data: store.store("output", data),
                    profile=AGENT_SHELL,
                    prefs=self._prefs,
                )

            return Rooted(
                root,
                resolver,
                reader,
                shell,
                navigator,
                found,
                search,
                agent_shell=agent_shell,
                # What `lightpanda` is masked from: the repository, whichever
                # root the view is on, and the person's secrets as the shell masks them.
                masked=tuple(dict.fromkeys((root, workspace, *executor.deny_read))),
            )

        # A resumed session reads the worktree its `workspace` line names.
        view = SessionView(workspace, rooted, latest_workspace(session.state))
        docs = self._design_docs(session, view, workspace, config, lambda: moved())

        def moved() -> str | None:
            with session.lock:
                ref = latest_workspace(session.state)
            try:
                entered = view.follow(ref)
            except InfraError as exc:
                # The run's report still reaches Zeus, and is not lost to a part that would
                # not open; the view stays where it was, and it is told so, never left to
                # read the checkout as if it were the worktree. The next turn tries again.
                branch = ref.branch if ref is not None else None
                return f"the view could not move to the session's worktree {branch}: {exc}"
            if entered is None:
                return None
            return view_moved(entered.branch, entered.source_head).text

        # `[tools.zeus]` over the role's default, for the model the rung runs: the prefix lists
        # what it
        # enables, and the loop dispatches the same.
        policy = config.tool_policy("zeus")
        surface = policy.surface(BUILTINS)
        prefix = assemble(
            role("zeus"),
            surface,
            render_index(
                catalog,
                repo_trust=config.repo_trust,
                max_rows=config.harness_limit("skills.index_max_rows"),
                desc_chars=config.harness_limit("skills.index_desc_chars"),
            ),
            Env(
                workspace=str(workspace),
                platform=current_platform(),
                sandbox=level,
                # `allow_secret_read` resolves the mask to nothing, so nothing is
                # masked whatever the sandbox; the block must not say otherwise.
                reads_masked=reads_masked(
                    level, allow_secret_read=self._denylist.allow_secret_read
                ),
                model=endpoint.model_id,
                repo_trust=config.repo_trust,
            ),
            work_dir=config.work.dir,
            # The templates Zeus writes a design and a brief to, as Athena reads them.
            handoffs=tuple(
                (title, resolve_handoff(self.repo, config.templates.configured(), kind).text)
                for kind, title in ZEUS_HANDOFFS
            ),
        )
        basis = window(endpoint)

        compacting = partial(self._compacting, conversation, hooks, basis, model)
        compaction = self._compaction(
            config, endpoint, model, view.resolver, records, ledger, stack, on_cut=compacting
        )
        snapshots = TurnSnapshots(session, config.undo, workspace)
        loop = ZeusLoop(
            session,
            prefix,
            client,
            first_attempt_params(endpoint),
            ZeusTools(
                session,
                view,
                view,
                Recall(root, folds),
                self._port(session),
                shell=view,
                box=_EngineBox(
                    self,
                    conversation,
                    snapshots,
                    self._port(session),
                    session,
                    moved,
                    docs,
                    work=partial(self._open_progress, session, docs),
                    resume=partial(self._continue_run, session, session.session_id),
                    runs=self._work_progresses().runs,
                    running=self._episodes.busy,
                    paused=lambda: _paused_of(session, runs),
                    attended=lambda: self.attended,
                ),
                navigator=view,
                locator=view,
                search=view.search,
                docs=docs,
                jobs=jobs,
                # Bound here, as the person's flag moves the session through it too.
                worktree=(moves := self._worktree_moves(session, config, moved)),
                # Unattended, `coder` waits for its run, so Zeus never has one to pause.
                team=_EngineTeam(
                    partial(self.pause_run, session.session_id),
                    partial(self._tell_team, session.session_id),
                )
                if self.attended
                else None,
                skills=Skills(
                    catalog,
                    session,
                    repo_trust=config.repo_trust,
                    accepted=self._accepted,
                    find_results=config.harness_limit("skills.find_results"),
                    desc_chars=config.harness_limit("skills.index_desc_chars"),
                    costs=agent_costs(config_profiles(config)),
                ),
                # Bound here, for `Built.agents` too: one set of agents a session.
                agents=(
                    agents := self._agents(
                        config, session, catalog, view, ledger, prices, stack
                    )
                ),
                surface=surface,
            ),
            LoopConfig.from_limits(
                config.resolved_limits,
                agents_parallel=config.harness_limit("agents.max_parallel"),
            ),
            interrupts=conversation.stops,
            ledger=ledger.for_rung(zeus_rung, role="zeus"),
            model=model,
            head=lambda: head_and_branch(view.root),
            status=partial(self._open_status, session, docs),
            reconcile=partial(run_outcome, records),
            tell_ended=partial(tell_ended, session, records),
            policy=policy,
            compaction=compaction,
            turn_ended=snapshots.ended,
            turn_woken=snapshots.woken,
            live=conversation.show,
            window=basis,
            hooks=hooks,
            unattended=isinstance(self.interaction, NonInteractive),
            flagged=moves.follow_flag,
            attach=partial(
                view.attach, inline_bytes=config.harness_limit("turn.text_inline_bytes")
            ),
            vision=config.vision_for(zeus_rung),
        )

        driven = partial(_driven, refresh, moved, self._namer(config, session, ledger, stack))

        return Built(
            step=partial(driven, loop.turn),
            session=session,
            ledger=ledger,
            refresh=refresh,
            stale=partial(
                self._stale,
                session,
                rungs_sha(config.rungs),
                unindexed=not self._indexed.is_set(),
            ),
            wake=partial(driven, loop.wake),
            wake_ready=partial(
                wake_ready,
                session,
                ledger.for_rung(zeus_rung, role="zeus"),
                lambda: loop.read_before,
            ),
            max_wakes=resolved_value(config.resolved_limits, "zeus", "zeus.max_wakes"),
            jobs=jobs,
            root=lambda: view.root,
            agents=agents,
            work=partial(self._open_progress, session, docs),
        )

    def _namer(
        self, config: EngineConfig, session: Session, ledger: CostLedger, stack: ExitStack
    ) -> Namer:
        """The `title` role's call, on its rung's client, charged to that rung's view
        of the session's ledger. An unpriced model leaves it off: the first line stands, as
        a residue on an unpriced model turns layer 3 off rather than refusing the session."""
        rung = config.rung_name_for("title")
        model = config.rungs[rung]
        view = ledger.for_rung(rung, role="title")
        client: ModelClient | None = None
        params = None
        if config.interaction.auto_title and model in config.prices:
            endpoint = config.models[model]
            client = client_for(endpoint)
            stack.callback(client.close)
            params = first_attempt_params(endpoint)
        return Namer(
            session,
            client,
            params,
            load_template(SHARED_TEMPLATES, "title.md"),
            charge=lambda usage: view.charge_usage("", usage, model),
            stopped=lambda: ledger.stopped_by(rung),
        )

    def _agents(
        self,
        config: EngineConfig,
        session: Session,
        catalog: SkillCatalog,
        view: SessionView,
        ledger: CostLedger,
        prices: PriceTable,
        stack: ExitStack,
    ) -> Agents:
        """The agents Zeus spawns and continues: each on a profile's role, over the
        session's current root, its own reader and graph connection and, for a reviewer, its
        own shell; a researcher the web and nothing else; charged to its rung."""

        def endpoint(rung: str) -> tuple[str, RungEndpoint] | str:
            model = config.rungs[rung]
            if model not in prices.rows:
                return f"its model {model!r} has no price row (08-budget.md §6)"
            return model, config.models[model]

        # Guards `stack`: agents open their clients on their own threads.
        opening = threading.Lock()

        def client(endpoint: RungEndpoint) -> ModelClient:
            opened = client_for(endpoint)
            with opening:
                stack.callback(opened.close)
            return opened

        web_limits = WebLimits.resolved(config.harness_limit)
        # Chosen once, as the session is composed: a `lightpanda` the person named and cannot
        # have is said now, never inside a researcher's round.
        engine_for = page_engine(
            config.web.fetch.engine,
            web_limits,
            backend=_select_backend(config.sandbox.backend),
            require_sandbox=config.sandbox.require_sandbox,
            binary=find_lightpanda(),
        )
        if isinstance(engine_for, str):
            raise ConfigError(engine_for)
        sources = session_sources(
            config.web.search,
            notify=lambda said: logging.getLogger(__name__).warning("%s", said),
        )
        # One for the session, so a host the person allowed holds across its rounds.
        hosts = (
            HostApproval(self._port(session).ask)
            if config.web.fetch.approval == "per_domain"
            else None
        )

        def ports(profile: Profile) -> AgentPorts:
            needs = BUILTINS.needs(profile.tools)
            # By its role's ceiling, as `spawn_agent` refuses handles, never by the tools a
            # profile kept: a reader narrowed to `grep` needs no `files` and is no researcher.
            if "files" not in role(profile.role).tool_ceiling.ports:
                # A researcher: one `Web` for the round, recording into the
                # session as every agent line does, and no file, graph or shell opened.
                web = Web(
                    ExternalPolicy(session, rung=profile.rung),
                    limits=web_limits,
                    engine=engine_for(view.current().masked),
                    docs=config.web.fetch.docs_domains,
                    approval=hosts,
                    sources=sources,
                )
                return AgentPorts(web=web if "web" in needs else None)
            # Built on the agent's own thread, with a graph connection of its own for its
            # navigator and its reader's aids, closed when its round ends.
            here = view.current()
            graph = opened(here.root)
            navigator = GraphNavigator(
                graph, indexed_profiles(graph, ProfileSet.load(repo=here.root))
            )
            files = here.files.scratch(vision=config.vision_for(profile.rung), aids=graph)
            shell = (
                here.agent_shell(files)
                if here.agent_shell is not None and "exec" in needs
                else None
            )
            close = graph.close if graph is not None else lambda: None
            return AgentPorts(files, navigator, here.search, shell, close=close)

        return Agents(
            session,
            catalog,
            config_profiles(config),
            ceiling=role("zeus").tool_ceiling,
            ceilings=lambda name: role(name).tool_ceiling,
            policy=config.tool_policy,
            endpoint=endpoint,
            client_for=client,
            ports=ports,
            # Zeus's reader opens what he passes, as his `read` would.
            handle_text=lambda handle: view.current().files.text_of(handle),
            # The root the citation is checked against is the view's when it is checked: the
            # session may have moved to its worktree since composition.
            read=lambda path: workspace_lines(view.root, denylist=self._denylist)(path),
            ledger=ledger.for_rung,
            trusted=self._accepted,
            third_party=config.repo_trust == "third_party",
            max_follow_ups=config.harness_limit("agents.max_follow_ups"),
        )

    def _stale(self, session: Session, composed: str, *, unindexed: bool = False) -> bool:
        """Whether the session's wiring has moved since it was composed as `composed`,
        or, `unindexed`, it was composed while the index was being built and it is built now:
        its graph is read at composition. Read on the worker, from the session's own
        fold: a switch is written through this writer while the session is open here
        (`switch_model`)."""
        if unindexed and self._indexed.is_set():
            return True
        with session.lock:
            moves = switches_of(session.state, lambda e: session.blob(e.payload_sha))
        return rungs_sha(self._wiring(m for _, m in moves)) != composed

    def _lasting(
        self,
        conversation: Conversation,
        root: Path,
        folds: ReplayCache,
        config: EngineConfig,
        stack: ExitStack,
    ) -> Lasting:
        """The session's writer and jobs, once per conversation (`Conversation.lasting`).

        Open-only (item 14): a session starts with `create_session()`, so a turn to an id that
        names none is `SessionError`, naming the path, never a new session. The writer is this
        conversation's alone; `Sessions` only describes a
        failure. Its jobs, one table for every root its view reads, are settled
        from a crash before the first turn, and closed before the writer, after the worker's
        last turn, so no exit is recorded under a turn that could still be writing."""
        session_id = conversation.session_id
        with Sessions(root, folds).described(session_id):
            session = open_reclaimed(root, session_id)
        stack.callback(session.close)
        hooks = self._session_hooks(config, session)
        # After the jobs close, so a job's exit is told before the session ends.
        stack.callback(self._session_ended, hooks, config, session)
        jobs = JobTable(
            session,
            limits=job_limits(config.resolved_limits),
            wake=conversation.wake_due,
            hooks=self._zeus(hooks, config, session),
            team=partial(self._team_run, session_id),
            board=self._run_board,
        )
        stack.callback(jobs.close)
        jobs.reconcile()
        return Lasting(session, jobs, hooks)

    def _navigation(
        self,
        config: EngineConfig,
        session: Session,
        workspace: Path,
        graph: Graph | None,
        reader: Reader,
        executor: Executor,
        ledger: CostLedger,
        prices: PriceTable,
        stack: ExitStack,
        hooks: Hooks | None = None,
    ) -> tuple[GraphNavigator, Locator | None]:
        """The graph's navigator, and `locate` over the same graph,
        navigator and executor.

        `graph` is the root's (`graphnav.opened_until_closed`). A checkout with no index gets a
        navigator that says `INDEX_UNAVAILABLE` and no locator (`NoLocator` says the same). The
        scout reads through Zeus's executor, so its search and escapes take Zeus's deny list,
        sandbox and network rules; each `locate` takes its own scratch reader.
        A scout model with no price row would be refused at its charge, after the call
        was paid for: `locate` answers `UNAVAILABLE` instead, and the session goes on."""
        profiles = indexed_profiles(graph, ProfileSet.load(repo=workspace))
        navigator = GraphNavigator(graph, profiles)
        if graph is None:
            return navigator, None
        rung = config.rung_name_for("locate")
        model = config.rungs[rung]
        if model not in prices.rows:
            detail = f"the locate rung's model {model!r} has no price row (08-budget.md §6)"
            return navigator, UnavailableLocator(model, detail)

        def client(endpoint: RungEndpoint) -> ModelClient:
            opened_client = client_for(endpoint)
            stack.callback(opened_client.close)
            return opened_client

        locator = scout_locator(
            config,
            session,
            workspace=workspace,
            graph=graph,
            navigator=navigator,
            reader=reader,
            executor=executor,
            client_for=client,
            profiles=profiles,
            ledger=ledger.for_rung(rung, role="locate"),
            hooks=hooks,
        )
        return navigator, locator

    @staticmethod
    def _compacting(
        conversation: Conversation, hooks: RoleHooks, basis: int, model: str, held: int
    ) -> None:
        """A cut is due, before it is made: the status line says so, and
        Zeus's `before_compact` hooks are told, which can only warn."""
        conversation.show(
            LiveEvent(
                "zeus", None, "context", ContextUpdate(held, basis, model, compacting=True)
            )
        )
        hooks.fire(HookPoint.BEFORE_COMPACT, BeforeCompact(prefix_tokens=held))

    def _session_hooks(self, config: EngineConfig, session: Session) -> Hooks:
        """The session's hooks, bound from the config as it opens, and
        `session_start` told how it opened. A script runs over the repository, sealed by the
        configured backend, or is refused where nothing seals."""
        backend = _select_backend(config.sandbox.backend)

        def spawn() -> ScriptSpawn:
            harness = HarnessSpawn(
                backend,
                ProcessSupervisor(),
                require_sandbox=config.sandbox.require_sandbox,
                deny_read=_resolve_denylist(
                    self._denylist, [self.repo, Path.home()], memo=WalkMemo.shared()
                ),
            )
            return partial(run_argv, harness)

        sink = SessionHookSink(session)
        bound = bind_session(
            sink,
            HookConfig.of(config, accepted=self._accepted),
            sealed_runner(backend, spawn, self.repo),
        )
        hooks = Hooks(bound, sink)
        zeus = self._zeus(hooks, config, session)
        if zeus.has(HookPoint.SESSION_START):
            with session.lock:
                state = session.state
                opened = SessionStart.of(
                    turns=bool(state.turns()), forked=state.forked_from is not None
                )
            note = zeus.fire(HookPoint.SESSION_START, opened).note
            if note:
                session.record_notice(hook_noted(note))
        return hooks

    @staticmethod
    def _zeus(hooks: Hooks, config: EngineConfig, session: Session) -> RoleHooks:
        """`hooks` as Zeus holds them, on the model his rung names now."""
        model = config.rungs[config.rung_name_for("zeus")]
        return RoleHooks(hooks, "zeus", model, session=session.session_id)

    def _session_ended(self, hooks: Hooks, config: EngineConfig, session: Session) -> None:
        """`session_end`, as the conversation closes, before its writer does."""
        zeus = self._zeus(hooks, config, session)
        if zeus.has(HookPoint.SESSION_END):
            with session.lock:
                turns = len(session.state.turns())
            zeus.fire(HookPoint.SESSION_END, SessionEnd(turns=turns))

    def _compaction(
        self,
        config: EngineConfig,
        endpoint: RungEndpoint,
        model: str,
        resolver: Callable[[], HandleResolver],
        records: JournalRecords,
        ledger: CostLedger,
        stack: ExitStack,
        *,
        on_cut: Callable[[int], None] | None = None,
    ) -> SessionCompaction:
        """Zeus's compaction: over the session's resolver and
        run records, on Zeus's endpoint, with the residue model (layer 3) on the
        `residue` role's rung, its own client closed with the conversation, and its calls
        charged to that rung's view of the session's ledger. An unpriced residue
        model turns layer 3 off, named on every cut, rather than refusing the session."""
        rung = config.rung_name_for("residue")
        residue_model = config.rungs[rung]
        residue: ModelResidue | None = None
        off: str | None = None
        if residue_model in config.prices:
            residue_endpoint = config.models[residue_model]
            residue_client = client_for(residue_endpoint)
            stack.callback(residue_client.close)
            residue = ModelResidue(
                residue_client,
                first_attempt_params(residue_endpoint),
                load_template(TEMPLATES, "residue.md"),
            )
        else:
            # A residue call against an unpriced model would be charged at zero, which the
            # budget refuses,
            # and refusing the whole session for it would make a price for the
            # cheap rung a precondition of talking to Zeus. Layer 3 is off instead, and every
            # cut says why on its own line (a residue failure never fails the cut).
            off = f"the residue rung's model {residue_model!r} has no price row; layer 3 is off"
        view = ledger.for_rung(rung, role="residue")
        trigger = resolved_value(config.resolved_limits, "zeus", "all.compact_trigger")
        return SessionCompaction(
            endpoint,
            config.prices[model],
            AffinityTable().row(endpoint.provider).cache_mode,
            resolver=resolver,
            records=records,
            residue=residue,
            charge=lambda usage: view.charge_usage("", usage, residue_model),
            residue_off=off,
            trigger_fraction=trigger / window(endpoint),
            pinned=pinned_seqs,
            on_cut=on_cut,
        )

    # ---- delegation: the only way the repository changes -------------------

    def coder(
        self, req: CoderRequest, *, session_id: str, caller: Literal["host", "zeus"] = "host"
    ) -> CoderHandle:
        """`(CoderRequest) -> CoderHandle`. The only writer of tracked source. The request
        carries work already approved at its stop: Zeus's call asks it at dispatch
        (`_EngineBox`), and a host asks it before it calls.

        Asynchronous: returns immediately and streams. One episode per engine instance, and
        one repository WRITE lease across processes — a second concurrent call is refused
        with `EngineBusy` naming the run in flight, never queued silently, because a lease
        race corrupts a worktree rather than raising.

        Names the run and takes the lease, and nothing else (item 10): the run starts when
        the handle's report, result or events are first asked for. It charges the session's
        ledger, so the episode lives under the session family's caps.

        `caller` says who writes the run's `episode` line: Zeus's dispatch writes his
        (`"zeus"`), since only it knows the turn and the constraints bound at the call; for a
        host's run (`"host"`, the default) the door writes it here, with no turn.
        """
        handle = self._open_run(req, session_id)
        if caller == "zeus":
            return handle
        # A host's run is the session's as Zeus's are: its `episode` line names it, with
        # no turn, before it does any work, so the family's spend and `resume()` find it. A
        # line that cannot be written gives the lease back, as Zeus's dispatch does.
        opened = self._conversations.open(session_id).opened()
        with given_back_on_failure(handle.run_id, handle.abandon):
            if opened is None:
                raise SessionError(f"session {session_id} is not open on this engine")
            # The constraints that bind are in the approved text itself (its Constraints
            # section), not on the line: a host's run cites none.
            opened.session.record_episode(
                EpisodeLine(
                    run_id=handle.run_id,
                    path="planned",
                    tier=None,
                    design_sha=req.work_sha,
                    constraints=(),
                    opened_at_seq=None,
                )
            )
            opened.session.sync()
        return handle

    def _open_run(self, req: CoderRequest, session_id: str) -> CoderHandle:
        """Name the run and take the lease (item 10), and nothing else: the one opening every
        run shares, Zeus's and a host's. Which `episode` line names it is the caller's."""
        with self._starts:
            self._refuse_handing_off(session_id)
            inputs = self._episode_inputs(session_id)
            self._settle(session_id)
            episode = self._episodes.open(
                lambda release: _open_episode(req, inputs, release), session_id=session_id
            )
        return CoderHandle(episode.run_id, episode, self._run_end)

    def _settle(self, session_id: str) -> None:
        """A handoff a crash interrupted, settled by what the repository shows, before
        a run reads the session's worktree or a resume continues one. Under `_starts`."""
        opened = self._conversations.open(session_id).opened()
        if opened is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        settle(opened.session, self.repo)

    def _refuse_handing_off(self, session_id: str) -> None:
        """Under `_starts`: nothing starts on a session whose handoff is under way."""
        if session_id in self._handing_off:
            raise EngineBusy(f"session {session_id} is handing off; try again when it ends")

    def _team_run(self, session_id: str) -> str | None:
        """Session `session_id`'s run working in the background now: what
        `job list` names beside the jobs, and what Zeus's shell may not write beside. None when
        it has none: another session's run on this engine is never this one's, and its
        repository WRITE lease already refuses this session's writes. Unattended, `coder` waits
        for its run, so no run works beside Zeus."""
        return self._episodes.busy(session_id) if self.attended else None

    def _run_board(self, run_id: str) -> str:
        """Run `run_id`'s task board, as `job tail` shows it while the team works:
        each task delegated so far, in the rows its report will list them in."""
        try:
            tasks = reported_tasks(self._projection(run_id))
        except FileNotFoundError:
            tasks = ()  # opened, and nothing journaled yet
        head = f"run {run_id} · working in the background · {len(tasks)} task(s) delegated"
        return "\n".join([head, *(line for task in tasks for line in task_lines(task))])

    def _tell_team(self, session_id: str, text: str, sources: tuple[int, ...]) -> str:
        """Zeus's message to the session's run: its live run, or the suspended one
        that would continue, delivered at its next boundary. Its text within
        `zeus.tell_text_max`, its citations the person's words (checked, and in the fold). A
        live run is woken to deliver it; a suspended one delivers it once it continues."""
        opened = self._conversations.open(session_id).opened()
        if opened is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        session = opened.session
        cap = int(resolved_value(self.config.resolved_limits, "zeus", "zeus.tell_text_max"))
        if not text.strip() or len(text) > cap:
            raise _TellRefused(
                InvalidArgument(
                    "text",
                    "a tell says something" if not text.strip() else f"over {cap} characters",
                    do="say it in a few sentences; a change that needs more is an amendment: "
                    "pause the team, change the design, then call coder",
                )
            )
        with session.lock:
            stray = [n for n in sources if not session.state.is_user_words(n)]
        if stray:
            raise _TellRefused(
                InvalidArgument(
                    "from",
                    f"turn {stray[0]} is not the person's words",
                    do="cite the person's turns this came from, by their [turn N]",
                )
            )
        live = self._episodes.current(session_id)
        if live is not None and not live.done:
            run_id, when = live.run_id, "at the worker's next tool boundary"
        else:
            run_id = _paused_of(session, self._caches_for(_roots(self.repo))[1])
            when = "when the run continues"
        if run_id is None:
            raise _TellRefused(
                InvalidArgument(
                    "team",
                    "no team is working",
                    do="hand the work over with coder",
                )
            )
        with session.lock:
            # A turn cut off after the line was written is asked again on resume: the same
            # words from the same turns to the same run are that tell, never a second one.
            again = next(
                (
                    t
                    for t in reversed(session.state.tells())
                    if (t.run_id, t.sources) == (run_id, sources)
                    and session.blob(t.text_sha) == text.encode("utf-8")
                ),
                None,
            )
            told = again or session.record_tell(
                run_id, text, sources, key=self._work_key(run_id)
            )
        if live is not None and not live.done:
            live.told()
        return f"told the team ({told.id}); delivered {when}"

    def _work_key(self, run_id: str) -> str | None:
        """The work run `run_id` is for, as `WorkProgress` keys it, read from its request."""
        request = self._work_progresses().runs.request(run_id)
        if request is None:
            return None
        pin = request.design.work if request.design is not None else None
        if pin is not None:
            return WorkKey(dir=pin.dir, name=pin.name).text
        first = request.continues.prior_runs[0] if request.continues is not None else run_id
        return WorkKey(first_run=first).text

    def pause_run(self, session_id: str) -> str:
        """Suspend the session's run in flight now (`pause`, `/pause`): a
        suspend, its step in flight put back and no `run_end`, so `coder` with its work, or
        `/resume`, continues it. `SessionError` when the session has none running: another
        session's run is never this one's to pause."""
        episode = self._episodes.current(session_id)
        if episode is None or episode.done:
            raise SessionError("nothing is running in this session")
        episode.suspend()
        return (
            f"run {episode.run_id} paused: nothing it did is lost; call coder with its work to "
            "continue it, or the person drops it with /drop"
        )

    def drop_run(self, session_id: str) -> str:
        """End the session's run for good (`/drop`): a live one is stopped as a hard
        stop ends it, `ABORTED`; a suspended one, the session's latest unfinished run, gets its
        `run_end` as a superseded run does (`end_superseded`). `SessionError` when there is
        neither. Another session's run is never this one's to drop."""
        # Read before `_starts`: opening a session may wait on its build, and nothing waits
        # while holding the lock every run's opening takes.
        opened = self._conversations.open(session_id).opened()
        with self._starts:
            # Decided under `_starts`, which every run's opening holds: a run being resumed is
            # either open, and stopped below, or not yet, and its `run_end` refuses the resume.
            episode = self._episodes.current(session_id)
            if episode is None or episode.done:
                if opened is None:
                    raise SessionError(f"session {session_id} is not open on this engine")
                unfinished = unfinished_runs(opened.session.state, self._projection)
                if not unfinished:
                    raise SessionError("no run of this session is running or paused")
                run_id = unfinished[-1]
                end_superseded(_roots(self.repo), run_id, "dropped by the user")
                return f"run {run_id} dropped: it ended, and what it committed stays"
        # Outside `_starts`: stopping it waits for its thread, and nothing need wait on that.
        episode.abandon()
        return f"run {episode.run_id} dropped: it ended, and what it committed stays"

    def resume(self, session_id: str) -> SessionHandle:
        """A crashed **session** resumes from its journal, not from memory — with
        any in-flight box work as a part of it.

        The session is opened here: its log replayed, and with it the constraint ledger, the
        log's own fold. Each episode its
        `coder` calls opened whose run's journal has no `run_end` is unfinished. The latest is
        continued through `Box.resume` over its own workspace, under this engine's one
        episode; the orchestrator's resume reconciles each effect before any retry.
        It is started here, and how it ends is told to Zeus before his next turn as a notice:
        the `coder` call that opened it was answered `PAUSED`, so nothing else waits on
        it. `EngineBusy` when an episode is already in flight here."""
        info = self.open_session(session_id)
        opened = self._conversations.open(session_id).opened()
        if opened is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        with self._starts:
            self._refuse_handing_off(session_id)
            self._settle(session_id)
        unfinished = unfinished_runs(opened.session.state, self._projection)
        pages = partial(self.journal, session_id)
        if not unfinished:
            return SessionHandle(session_id, info, pages=pages)
        handle = self._continue_run(opened.session, session_id, unfinished[-1])
        conversation = self._conversations.get(session_id)
        if self.attended and conversation is not None:
            # It goes on in the background as a `coder` run does, and its end wakes Zeus with
            # its report; unattended, his next turn is told it.
            _watch_run(handle, opened.session, conversation.wake_due)
        return SessionHandle(session_id, info, handle, unfinished, pages=pages)

    def _continue_run(self, session: Session, session_id: str, run_id: str) -> CoderHandle:
        """Continue the session's unfinished run `run_id` from its journal, started
        now on its own thread: no `coder` call waits on it, and how it ends is told to Zeus
        before his next turn (`tell_ended`). The one door for `resume` and for a `coder` call on
        a work whose run is suspended. Refused when its worktree was handed off."""
        # The worktree the paused run wrote in, if it wrote in one: an in-place run, the default
        # is never refused for an older worktree the session handed off.
        with session.lock:
            recorded = session.state
            wrote_in = run_workspace(recorded, run_id)
            handed = (
                last_worktree(recorded, branch=wrote_in.branch)
                if wrote_in is not None
                else None
            )
        if (
            handed is not None
            and handed.mode == "worktree"
            and handed.handoff not in REATTACHED
        ):
            # Its worktree was applied, squashed or discarded while it was paused (a session
            # written before `session_handoff.refusal` refused that): continued, it would
            # commit where no handoff reaches.
            kept = (
                "its worktree and branch were removed; start the work again"
                if handed.handoff == "discard"
                else f"what it committed stays on branch {handed.branch}: `git merge "
                f"{handed.branch}` brings it over by hand, or start the work again"
            )
            raise SessionError(
                f"run {run_id} was paused and its worktree then handed off ({handed.handoff}), "
                f"so it cannot be continued: {kept}"
            )
        with self._starts:
            # Under `_starts`, as `drop_run` decides: a run dropped before this is ended, and
            # continuing it would write past its `run_end`.
            if self._projection(run_id).end is not None:
                raise SessionError(f"run {run_id} has ended, so it cannot be continued")
            inputs = self._episode_inputs(session_id)
            episode = self._episodes.open(
                lambda release: _resume_episode(run_id, inputs, release), session_id=session_id
            )
        # Started now: no `coder` call waits on it, and a served client cannot wait on its
        # handle. How it ends is told to Zeus before his next turn (`tell_ended`).
        episode.start()
        return CoderHandle(run_id, episode, self._run_end)

    # ---- sessions own durable state ----------------------------------

    def create_session(
        self,
        session_id: str | None = None,
        *,
        pinned: bool = False,
        adopt: str | None = None,
    ) -> SessionInfo:
        """A new session's log, written and released: the session is opened for writing by
        the first `turn()` or `open_session()`, on its own worker. With no id, one is
        minted. An id that already names a session is `SessionError`.

        `pinned` is the eval harness's: the session's first line pins its models and
        caps, so `switch_model` and `set_cap` refuse in it.

        `adopt` is an open work's key (`open_works`): the new session continues it, and the
        answer's `continuing` says so. A key no longer open is
        `SessionError`, before anything is written."""
        sessions = self._sessions()
        chosen = self._open_work(adopt) if adopt is not None else None
        sid = session_id if session_id is not None else new_session_id()
        with sessions.described(sid):
            session = Session.create(sessions.root, sid)
            try:
                if pinned:
                    record_pinned(session)
                if chosen is not None:
                    adopt_work(session, chosen)
            finally:
                session.close()
        # Nothing commits on the person's branch as a session starts: the
        # index row is written with the work's landing, never on its own.
        info = sessions.info(sid)
        return info if chosen is None else replace(info, continuing=chosen.info())

    def open_works(self) -> tuple[OpenWorkInfo, ...]:
        """The repository's open work, the newest first: what a new
        session may continue. Read from every session's log; nothing is written."""
        return tuple(work.info() for work in self._open_works())

    def _open_works(self) -> tuple[OpenWork, ...]:
        sessions = self._sessions()
        return open_works(self.repo, sessions.states(), live=sessions.live)

    def _open_work(self, key: str) -> OpenWork:
        found = next((w for w in self._open_works() if w.key == key), None)
        if found is None:
            raise SessionError(
                f"{key} is not open work in this repository: it was landed, dropped, or is "
                "held by a session open elsewhere; start without it, or pick from open_works()"
            )
        return found

    def open_session(self, session_id: str) -> SessionInfo:
        """Open the session here: its log under this engine's writer, and its loop built.
        `SessionError` when no such session exists, or it is open in another process."""
        self._conversations.open(session_id)
        return self._sessions().info(session_id)

    def latest_session(self) -> SessionInfo | None:
        """The session written to most recently, or None in a repository with none."""
        return self._sessions().latest()

    def sessions(self) -> tuple[SessionInfo, ...]:
        """Every session, the most recently written first, each titled by its first user turn.
        Read without any session's lock, and opening none."""
        return self._sessions().list()

    def session_info(self, session_id: str) -> SessionInfo:
        """One session's stored facts, read without its lock and without opening it: a watcher
        may describe a session another process is writing."""
        return self._sessions().info(session_id)

    def rename(self, session_id: str, text: str) -> SessionInfo:
        """The person's name for the session: a `title` line `by: person`, its first
        line cut as a title is cut, which the `title` role never overwrites. Returns the
        session's info, titled. `ValueError` for a name with no words."""
        title = title_of(text)
        if title is None:
            raise ValueError("a session's name has words")
        named = SessionTitle(title, "person")
        self._write(session_id, "a name", lambda session: record_title(session, named))
        return self.session_info(session_id)

    def hidden(self, session_id: str) -> tuple[int, ...]:
        """The seqs the session's standing rewinds hide now (`rewind.hidden_seqs`): what
        a frontend rendering the raw records leaves out, so no frontend restates the rule."""
        folds, _runs = self._caches_for(self._where(session_id))
        with self._sessions().described(session_id):
            _log, state = folds.get(session_id)
        return tuple(sorted(state.hidden))

    def liveness(self, session_id: str) -> Liveness:
        """Whether anyone is still working on the session: a
        projection over its records and its log's lock, computed per call, stored nowhere,
        writing nothing."""
        where = self._where(session_id)
        folds, runs = self._caches_for(where)
        with self._sessions().described(session_id):
            log, state = folds.get(session_id)
        lock = session_dir(sessions_root(where), session_id) / LOCK_NAME
        # This engine's own loop knows a turn that failed is over; the log cannot.
        open_here = self._conversations.get(session_id)
        failed = open_here.turn_failed if open_here is not None else None
        opening = open_here is not None and open_here.opening
        records = JournalRecords(where, runs)
        # And the run working in the background, which only this engine's episode knows.
        episode = self._episodes.current(session_id) if self.attended else None
        team = episode.run_id if episode is not None and not episode.done else None
        return _liveness(
            state,
            open_in(log),
            records,
            lock,
            failed=failed,
            opening=opening,
            team=team,
            indexing=self.indexing,
            index_progress=self.index_progress,
        )

    def close_session(self, session_id: str) -> None:
        """Release what the session holds here (item 13): its loop's worker thread, its model
        client and its log's lock, after stopping a turn still running. Its history stays,
        and a later `turn()` or `open_session()` opens it again. A session not open here has
        nothing to release."""
        self._conversations.close(session_id)

    @not_built("a task not yet opened (28-context §5)")
    def session_context(
        self, session_id: str, purpose: Any, budget: int, current_input: str = ""
    ) -> Any:
        """`(purpose: Purpose, budget: int) -> SessionContextBundle`."""
        raise NotImplementedError(_PHASE)

    @not_built("a task not yet opened (28-context §5)")
    def expand_turn(self, session_id: str, seq: int) -> str:
        raise NotImplementedError(_PHASE)

    @not_built("a task not yet opened (28-context §2: a read that names its session)")
    def read(self, handle: Any, span: Any = None) -> Any:
        """`(Handle, Span | None) -> ToolResult`

        Reading is a tool call inside the session, not an entry point of its own — which is
        what `ask`/`investigate` being deleted actually means.
        """
        raise NotImplementedError(_PHASE)

    # ---- introspection for hosts -------------------------------------------

    def journal(
        self, scope_id: str, cursor: Cursor | None = None, limit: int = PAGE_DEFAULT
    ) -> Page:
        """A scope's records past `cursor`, at most `limit`, payloads as stored: the seek a
        served `/events` and `/journal` read with. `scope_id` is a session
        **or** one coder episode: both are journal scopes, and both are askable. A run id
        whose run exists names the run; any other id names a session.

        A session is read through the engine's fold cache, which extends its log once for
        every reader, so a reader that comes back with its cursor is handed only
        what was written since. `SessionError` when the session does not exist.

        A run id with no journal yet is `RunNotStarted`, never a missing session: its
        episode is recorded before its journal is written, and one blocked before it started
        never writes one. Only a session someone named with a run-shaped id falls through."""
        at = cursor if cursor is not None else Cursor()
        where = self._where(scope_id)
        if RUN_ID_RE.match(scope_id):
            if run_dir(where, scope_id).is_dir():
                return run_page(journal_path(where, scope_id), scope_id, at, limit)
            if not session_dir(sessions_root(where), scope_id).is_dir():
                raise RunNotStarted(
                    f"run {scope_id} has no journal yet: it has not started, or was stopped "
                    "before it started"
                )
        folds, _runs = self._caches_for(where)
        with self._sessions().described(scope_id):
            log, _state = folds.get(scope_id)
        return session_page(log, at, limit)

    def handoff(self, session_id: str) -> HandoffSummary:
        """`-> HandoffSummary`, per **session**: the worktree the session
        still writes, its diff from the base and the choices it accepts now, the session's own
        spend and episodes, and every claim it could not make, the run header's lines among
        them. Behaviour is reported as not verified until checkpoint results reach it.
        Reads only: an interrupted handoff is named, and settled by `finalize`,
        `resume` or the next run's start."""
        state, payload = self._on_disk(session_id)
        _folds, runs = self._caches_for(self._where(session_id))
        return summarize(
            session_id,
            state,
            payload,
            self.repo,
            spend=session_spend(session_id, state, runs),
            header=(line.text for line in self.header(session_id).lines),
            runs=runs,
            blob=JournalRecords(self._where(session_id), runs).blob,
        )

    def set_cap(
        self, session_id: str, hard_cost_usd: float | None, rung: str | None = None
    ) -> SessionBudget:
        """The person sets the session family's cap, or lifts it with `None`.
        `rung` names one rung's cap beneath the total instead; a rung the cast lacks
        is a `ConfigError`, before anything is written.

        The person's act, never a model's: no tool reaches this method. Checked as a
        config load is: a cap that is not finite or not above zero is a `ConfigError`,
        raised before anything is written. The value is a fsynced `cap` line, and the
        family's latest line is its cap from its next turn. A raise lets a session stopped at
        `BUDGET_CAP` go on, and a cut below the spend stops the next turn. The spend is
        counted either way.

        This is also refused in an eval run: a session `create_session(pinned=True)` made
        is `ConfigError` here."""
        if hard_cost_usd is not None and (
            isinstance(hard_cost_usd, bool)
            or not isinstance(hard_cost_usd, int | float)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not is_cap(hard_cost_usd)
        ):
            raise ConfigError(
                f"a cap is a finite amount above zero, or None to lift it (D144); got "
                f"{hard_cost_usd!r}"
            )
        if rung is not None:
            self._cappable(rung)
        self._refuse_pinned(session_id, "its caps")
        self._write(
            session_id, "a cap", lambda session: session.record_cap(hard_cost_usd, rung)
        )
        return self.budget(session_id)

    def _cappable(self, rung: str) -> None:
        """A rung cap names a declared rung. A former name is refused naming its
        successor, as a config load refuses it; an undeclared one naming the declared rungs.
        Both before anything is written."""
        former = former_rung_error(rung)
        if former is not None:
            raise ConfigError(f"no rung to cap: {former}")
        if rung not in self.config.rungs:
            declared = list(self.config.rung_names)
            raise ConfigError(f"no rung {rung!r} to cap; the declared rungs are {declared}")

    def models(self, session_id: str) -> ModelWiring:
        """`-> ModelWiring`: which catalog model each rung of the session calls, the
        catalog, and the session's switches. Read from the log on disk, without the session's
        lock, as `budget()` is: a `model_switched` line is fsynced before its writer returns."""
        moves = self._switches_of(session_id, strict=True)
        return ModelWiring(
            rungs=self._wiring(moves),
            catalog=tuple(sorted(self.config.models)),
            switches=moves,
            roles=dict(self.config.assignment),
            models={
                name: CatalogModel(
                    m.context_window, m.max_output_tokens, m.provider, model_id=m.model_id
                )
                for name, m in sorted(self.config.models.items())
            },
        )

    def switch_model(self, session_id: str, rung: str, model: str) -> ModelSwitch:
        """`-> ModelSwitch`: move `rung` to catalog model `model` for the rest of
        the session, from the rung's next call.

        The person's act, never a model's: no tool reaches this method. Checked as a
        load is (`EngineConfig.with_rung`), and the model must have a price row, so a call
        after the switch is priced: `ConfigError` naming why, before
        anything is written. A pinned (eval) session refuses it too. Made only while nothing
        runs: `EngineBusy` while the session's turn is running or a run is in flight here, so
        no call is ever in flight at a switch. Recorded as a fsynced `model_switched` line;
        the session's loop is rebuilt before its next word. A turn or a run cannot start while
        the switch is checked and written (`_starts`)."""
        with self._starts:
            self._refuse_handing_off(session_id)
            self._refuse_pinned(session_id, "its models")
            current = self._session_config(self._switches_of(session_id, strict=True))
            if current.rungs.get(rung) == model:
                raise ConfigError(f"rung {rung!r} already runs on model {model!r}")
            switched = current.with_rung(rung, model)
            if model not in switched.prices:
                raise ConfigError(
                    f"model {model!r} has no price row, so a call after the switch could not "
                    f"be charged (08-budget.md §6); add [prices.{model}] to the engine config"
                )
            self._refuse_running(session_id, "switch a model")
            moved = ModelSwitch(
                rung=rung,
                from_model=current.rungs[rung],
                to_model=model,
                model_id=switched.models[model].model_id,
                rungs_sha=rungs_sha(switched.rungs),
            )
            self._write(
                session_id,
                "a switch",
                lambda session: record_switch(session, moved),
            )
            return moved

    def _refuse_running(self, session_id: str, act: str) -> None:
        """The person's `act` waits until nothing runs, so no call is in flight
        across a switch, and no episode writes the worktree a handoff acts on."""
        conversation = self._conversations.get(session_id)
        if conversation is not None and conversation.running:
            raise EngineBusy(
                f"session {session_id} is {conversation.phase}; {act} between turns"
            )
        running = self._episodes.busy()
        if running is not None:
            raise EngineBusy(f"{running} is in flight on this engine; {act} when it ends")

    def _refuse_pinned(self, session_id: str, what: str) -> None:
        """A pinned (eval) session's models and caps are fixed by its manifest."""
        pinned = pin_of(*self._on_disk(session_id))
        if pinned is not None:
            raise ConfigError(
                f"session {session_id} is pinned ({pinned.why}): {what} cannot change"
            )

    def _wiring(self, moves: Iterable[ModelSwitch]) -> dict[str, str]:
        """Each rung's model now: the engine config's `[rungs]`, moved by `moves`."""
        return wired(self.config.rungs, moves)

    def _switches_of(self, session_id: str, *, strict: bool = False) -> tuple[ModelSwitch, ...]:
        """The session's switches, from its log on disk. Not `strict`, a log that cannot be
        read has none: composing such a session fails at its open, naming why, after the
        config's own checks."""
        try:
            return tuple(move for _, move in switches_of(*self._on_disk(session_id)))
        except SessionError:
            if strict:
                raise
            return ()

    def _session_config(self, moves: Sequence[ModelSwitch]) -> EngineConfig:
        """The engine config as the session runs it: its switches applied, validated
        as a load is. A switch the engine config no longer admits (its model since removed,
        say) is `ConfigError` naming it, and the session does not open until it is fixed."""
        if not moves:
            return self.config
        return self.config.rewired(self._wiring(moves))

    def _on_disk(self, session_id: str) -> tuple[SessionState, Payload]:
        """The session's fold and its payloads, from the log on disk: never Zeus's live fold,
        which its own thread is appending to. `SessionError` if it is missing or unreadable."""
        folds, _runs = self._caches_for(self._where(session_id))
        try:
            log, state = folds.get(session_id)
        except SessionMissing as exc:
            raise SessionError(f"session {session_id} does not exist") from exc
        except (SessionStoreError, OSError, ValueError) as exc:
            raise SessionError(f"session {session_id} cannot be read: {exc}") from exc
        return state, log.payload

    def _write(self, session_id: str, what: str, write: Callable[[Session], object]) -> None:
        """Write the person's line through the session's writer: its conversation's when it is
        open here and built, else the log opened here for the one line."""
        conversation = self._conversations.get(session_id)
        # A session open here writes through its own writer, once its build has ended; one
        # whose build failed has released the log before it says so, and is opened here.
        opened = conversation.opened() if conversation is not None else None
        try:
            if opened is not None:
                write(opened.session)
            else:
                root = sessions_root(self._where(session_id))
                with open_reclaimed(root, session_id) as session:
                    write(session)
        except SessionMissing as exc:
            raise SessionError(f"session {session_id} does not exist") from exc
        except SessionStoreError as exc:
            raise SessionError(f"session {session_id} cannot take {what}: {exc}") from exc

    def budget(self, session_id: str) -> SessionBudget:
        """The family's spend, the cap in force and where it came from. Computed per
        call from the records; adds nothing to them."""
        return self._budget(session_id)[0]

    def _budget(self, session_id: str) -> tuple[SessionBudget, dict[str | None, str]]:
        """`budget()`, and the member whose log holds each `cap` line in force, keyed by rung
        (`None` for the total): a line set from a fork caps its parent too, and the header
        names where it came from."""
        where = self._where(session_id)
        folds, runs = self._caches_for(where)
        # Every `cap` and `spend` line is fsynced before its writer returns, so the disk is
        # not behind on what this reports.
        state, _payload = self._on_disk(session_id)
        return family_budget(
            session_id, state, where, folds, runs, self.config.spend, self.config.rung_names
        )

    def _episode_inputs(self, session_id: str) -> EpisodeInputs:
        """The session's ledger and the engine's values, for an episode of `session_id`.
        Opens the session here, as a turn would, so its ledger is the one its turns charge."""
        conversation = self._conversations.open(session_id)
        opened = conversation.opened()
        if opened is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        # The family's caps and spend as the records say now, as before a turn: a cap
        # set, or a sibling's spend, since the session's last turn binds this episode too.
        opened.refresh()
        # The session's wiring, as its turns are composed from: a run started after a
        # `/rung` switch calls the rung's new model.
        session = opened.session
        with session.lock:
            moves = switches_of(session.state, lambda e: session.blob(e.payload_sha))
        config = self._session_config(tuple(m for _, m in moves))
        return EpisodeInputs(
            repo=self.repo,
            config=config,
            denylist=self._denylist,
            ledger=opened.ledger,
            backend=_select_backend(config.sandbox.backend),
            interaction=self._port(opened.session),
            session_id=session_id,
            session=opened.session,
            asking=partial(still_asking, session),
            prefs=self._prefs,
            commands=self._commands,
            beside=opened.jobs,
            live=conversation.show,
            accepted=self._accepted,
            runs=self._caches_for(_roots(self.repo))[1].readable,
            replied=partial(_team_replied, session, conversation.wake_due),
        )

    def _port(self, session: Session) -> GrantingPort:
        """The session's `InteractionPort`: the engine's own, wrapped so every question put to
        the person reaches the session log before anyone is asked, and its answer after
        (`context/questions.py`). Every asker a session has is handed this, never the bare
        port. A standing grant answers a confirmation it covers before anything is recorded,
        and the person's *always* is written down (`approval/grants.py`)."""
        # The grant check over the recorder: a granted action is one `action_granted` line,
        # never a question.
        return GrantingPort(
            RecordingPort(
                scoped(self.interaction, session.session_id),
                session,
                max_questions=self.config.interaction.max_questions_per_run,
            ),
            session,
            _roots(self.repo),
            never_ask=self.config.sandbox.never_ask,
        )

    def _prefs(self) -> Preferences:
        """The person's and the repository's approval preferences, read
        per ask so an edit lands at the next one. A file that cannot be used is the config's
        error, as any other config file's is."""
        try:
            return load_preferences(_roots(self.repo))
        except PreferenceError as exc:
            raise ConfigError(str(exc)) from exc

    def _open_progress(self, session: Session, docs: DesignDocs | None) -> WorkProgress | None:
        """Where the session's open work stands, or None with no open work:
        read from its runs, whichever session made them, and from the folder where it is
        written. One home for the dispatch and the frame."""
        with session.lock:
            work = open_work(session.state)
            tells = session.state.tells()
        if work is None or docs is None:
            return None
        return self._work_progresses().of(
            WorkKey(dir=work.dir, name=work.name),
            docs.folder_of(work),
            live=self._episodes.running,
            tells=tells,
        )

    def _work_progresses(self) -> Progresses:
        """The engine's one memo of where its works stand, over its one reader of the runs'
        requests (each read once: a run's config snapshot never changes) and its one fold of
        the journals."""
        with self._caches_lock:
            held = self._progresses
        if held is not None:
            return held
        where = _roots(self.repo)
        made = Progresses(JournalRuns(where, self._caches_for(where)[1]))
        with self._caches_lock:
            if self._progresses is None:
                self._progresses = made
            return self._progresses

    def _open_status(self, session: Session, docs: DesignDocs | None) -> str | None:
        """The open work's status as the frame carries it, or None."""
        found = self._open_progress(session, docs)
        return status_text(found) if found is not None and found.runs else None

    def _run_end(self, run_id: str) -> RunEndP | None:
        """A run's `run_end`, read from its journal: the terminal a result reports."""
        return self._projection(run_id).end

    def _projection(self, run_id: str) -> RunProjection:
        """Run `run_id`'s journal, folded. A damaged one cannot be read past its damage, and
        is never edited to parse: it is the session's error, naming the run, never the
        store's."""
        _folds, runs = self._caches_for(_roots(self.repo))
        try:
            return runs.projection(run_id)
        except JournalCorrupt as exc:
            raise SessionError(
                f"run {run_id}'s journal is damaged and cannot be read: {exc}"
            ) from exc

    def _sessions(self) -> Sessions:
        """The repository's sessions, for the facade's reads and creates. Holds nothing open
        between calls: the writer a session is open under is its conversation's."""
        try:
            where = _roots(self.repo)
        except NotARepository as exc:
            raise SessionError(
                f"{self.repo} is not a git repository; it has no sessions"
            ) from exc
        folds, _runs = self._caches_for(where)
        return Sessions(sessions_root(where), folds)

    def _where(self, session_id: str) -> Roots:
        try:
            return _roots(self.repo)
        except NotARepository as exc:
            raise SessionError(
                f"session {session_id} does not exist: {self.repo} is not a git repository"
            ) from exc

    def _caches_for(self, where: Roots) -> tuple[ReplayCache, RunProjections]:
        """The engine's one pair: every reader shares them, the ledger's
        refresh, `budget()`, `header()`, the run records and recall. The engine has one
        repository, so one pair serves every session in it."""
        with self._caches_lock:
            if self._caches is None:
                self._caches = (ReplayCache(sessions_root(where)), RunProjections(where))
            return self._caches

    def header(self, session_id: str) -> RunHeader:
        """Where the session runs, and every guarantee it runs without.

        Computed per call from the config, the session log and its episodes' journals, like
        `liveness()`: a stored header would be stale the moment an MCP server dropped
        mid-session. Adds no journal event. `handoff()` carries every line in `unverified`.
        """
        where = self._where(session_id)
        folds, runs = self._caches_for(where)
        try:
            budget, members = self._budget(session_id)
        except SessionError:
            budget, members = None, {}  # `_header` says why the session cannot be read
        uncapped, lifted = _spend_facts(budget, members, session_id)
        return _header(
            where,
            self.config,
            session_id,
            denylist=self._denylist,
            folds=folds,
            runs=runs,
            uncapped=uncapped,
            uncapped_rungs=lifted,
            commands=self._commands,
        )

    # ---- background jobs ---------------------------------------------

    def jobs(self, session_id: str) -> tuple[JobView, ...]:
        """The session's jobs, running and ended, in the order they started: from its `JOB`
        effects, so a finished job never shows as running."""
        return self._job_table(session_id).views()

    def kill_job(self, session_id: str, job_id: str) -> None:
        """The person's `/jobs kill`: signals and writes nothing; the job's
        waiter records the exit, which tells nobody. `SessionError` when it is not running."""
        if not self._job_table(session_id).kill(job_id, "person"):
            raise SessionError(f"no job {job_id} is running in session {session_id}")

    def work(self, session_id: str) -> WorkProgress | None:
        """Where the session's open work stands: each task of its approved
        plan across the work's runs, the one fold Zeus's status line reads, so a
        screen shows a task an earlier run did as done; None with no open work. `SessionError`
        when the session is not open on this engine."""
        opened = self._conversations.open(session_id).opened()
        if opened is None or opened.work is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        return opened.work()

    def _job_table(self, session_id: str) -> JobTable:
        opened = self._conversations.open(session_id).opened()
        if opened is None or opened.jobs is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        return opened.jobs

    # ---- undo by turn ------------------------------------------------

    # ---- the REPL's input ---------------------------------

    def files(self, session_id: str) -> tuple[str, ...]:
        """The paths `glob` would list where Zeus's view is now (`graph.ignore.listed`): what
        the REPL's `@` completes, so a completion never says an ignored or denied file
        exists. Opens the session here, as a turn would."""
        opened = self._conversations.open(session_id).opened()
        if opened is None or opened.root is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        return listed(opened.root(), self._denylist)

    def file_text(self, session_id: str, path: str) -> FileText:
        """`-> FileText`: one file `files()` lists, as text, or why it is not; the CODE
        pane's view of a file no task changed. `NotListed` for any other path."""
        opened = self._conversations.open(session_id).opened()
        if opened is None or opened.root is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        root = opened.root()
        return file_text(root, path, listed(root, self._denylist))

    def holds_secret(self, text: str) -> bool:
        """Whether `text` holds a secret value (`secrets.redact`): a line the REPL keeps out
        of its history file."""
        _redacted, found = redact_text(text)
        return bool(found)

    def undoable(self, session_id: str) -> tuple[Undoable, ...]:
        """The turns the person can take back, oldest first, each with the options it offers:
        code and conversation, conversation only, or code only."""
        session, _ = self._idle(session_id)
        return undo_module.undoable(session, self.repo)

    def undo(
        self,
        session_id: str,
        to_turn: int,
        *,
        code: bool,
        conversation: bool,
        commit: bool | None = None,
    ) -> UndoReport:
        """Take back turns `to_turn` onward. `commit` overrides `undo.on_your_branch`
        once, after `apply` or `squash`; in place nothing is committed. Conflicts with the
        person's own edits are asked through the `InteractionPort`."""
        session, records = self._idle(session_id)
        return undo_module.undo(
            session,
            records,
            self.repo,
            self.config.undo,
            self._port(session),
            to_turn=to_turn,
            code=code,
            conversation=conversation,
            commit=commit,
        )

    def redo(self, session_id: str) -> UndoReport:
        """Take the newest undo back, as far as it is still offered."""
        session, records = self._idle(session_id)
        return undo_module.redo(
            session, records, self.repo, self.config.undo, self._port(session)
        )

    def undo_task(self, session_id: str, run_id: str, task_id: str) -> Restored:
        """Roll a worktree run back to just before one task's commit (`/undo task`)."""
        session, records = self._idle(session_id)
        return undo_module.undo_task(
            session,
            records,
            self.repo,
            self.config.undo,
            self._port(session),
            run_id=run_id,
            task_id=task_id,
        )

    def undo_operation(
        self, session_id: str, operation_id: str, run_id: str | None = None
    ) -> Restored:
        """Restore one operation's declared paths from its snapshot ref (`/undo <operation>
        [<run>]`). `run_id` names the run when more than one journaled the
        id, since ids repeat across runs; without it a repeated id is refused, naming them."""
        session, records = self._idle(session_id)
        return undo_module.undo_operation(
            session, records, self.repo, operation_id, run_id=run_id
        )

    def fork(
        self,
        session_id: str,
        at_seq: int,
        reason: str,
        code: ForkChoice | None = None,
    ) -> Forked:
        """A new session that branches `session_id` at `at_seq`. The
        code goes back to the fork point (`restore`) or stays as it is (`keep`); with `code`
        unset the person is asked (default `keep`). Refused at an unsafe point, with
        nothing changed. Returns the child's id and what it was told on arrival."""
        session, records = self._idle(session_id)
        if code is None:
            code = undo_module.ask_fork_code(self._port(session), session_id, at_seq)
        try:
            return undo_module.fork(
                session,
                records,
                self.repo,
                self.config.undo,
                self._port(session),
                at_seq=at_seq,
                reason=reason,
                code=code,
            )
        except ForkRefused as exc:
            raise SessionError(str(exc)) from exc

    def _idle(self, session_id: str) -> tuple[Session, JournalRecords]:
        """The session's writer and its run records, refused while a turn runs: an undo is the
        person's act between turns, and a turn writes lines it could interleave with."""
        conversation = self._conversations.open(session_id)
        opened = conversation.opened()
        if opened is None:
            raise SessionError(f"session {session_id} is not open on this engine")
        if conversation.phase not in (Phase.IDLE, Phase.HELD):
            raise SessionError(f"session {session_id} is running a turn; undo between turns")
        where = self._where(session_id)
        _folds, runs = self._caches_for(where)
        return opened.session, JournalRecords(where, runs)

    def metrics(self, scope_id: str) -> RunMetrics:
        """`-> RunMetrics` (09) of one run, derived from its journal. A session's metrics have
        no aggregate the specs define yet, and are refused, never answered empty."""
        if not RUN_ID_RE.match(scope_id):
            raise NotImplementedError(
                f"{scope_id} is not a run: a session's metrics have no aggregate yet "
                f"(09-journal; a task not yet opened)"
            )
        return derive_metrics(self._projection(scope_id))

    def run_blob(self, run_id: str, sha: str) -> RunBlob:
        """`-> RunBlob`: a blob a record of run `run_id` names, as text, or
        why it is not: a brief, a report, a review's words, a command's output, a plan.
        `RunNotStarted` when the run has no journal; `NotNamed` when no record names `sha`."""
        return self._run_read(run_id, lambda records: run_blob(records, run_id, sha))

    def run_diff(
        self, run_id: str, *, task_id: str | None = None, operation_id: str | None = None
    ) -> RunDiff:
        """`-> RunDiff`: what task `task_id`, or step `operation_id`, of run
        `run_id` changed, between two trees its journal names. `NotReadable` when the run holds
        no such task or step."""
        return self._run_read(
            run_id,
            lambda records: run_diff(
                records, self.repo, run_id, task_id=task_id, operation_id=operation_id
            ),
        )

    def _run_read[T](self, run_id: str, read: Callable[[JournalRecords], T]) -> T:
        """`read` over the run's records, its damage said as `_projection()` says it."""
        where = _roots(self.repo)
        _folds, runs = self._caches_for(where)
        try:
            return read(JournalRecords(where, runs))
        except JournalCorrupt as exc:
            raise SessionError(
                f"run {run_id}'s journal is damaged and cannot be read: {exc}"
            ) from exc

    def finalize(self, session_id: str, choice: GitHandoffChoice | str) -> HandoffResult:
        """`-> HandoffResult`: the person's reviewed choice over the session's worktree. Never
        pushes, under any choice.

        The person's act: here between turns, or asked of the person at dispatch when Zeus's
        `worktree` tool proposes it (`_land_in_turn`). Made only while nothing
        runs (`EngineBusy`), and no run starts and no switch is made while it is under way
        (`_handing_off`, checked under `_starts`; the git work runs outside it). Under the
        repository WRITE lease, written ahead on the session log (`session_handoff`), and
        refused with `HandoffError`, nothing changed, when the session has no worktree or
        `session_handoff.refusal` refuses the choice: a run still paused in the worktree, or
        the worktree's own state. Afterwards the session's `workspace` line
        names the choice: after `apply`, `squash`, `discard` or `bring` the next `coder` call
        opens a new worktree, after `keep` it reattaches this one."""
        chosen = handoff_choice(choice)
        with self._starts:
            self._on_disk(session_id)
            self._refuse_handing_off(session_id)
            self._refuse_running(session_id, "hand off")
            self._handing_off.add(session_id)
        return self._hand_off(session_id, chosen)

    def _land_in_turn(self, session_id: str, choice: GitHandoffChoice) -> HandoffResult:
        """The same handoff as `finalize`, from inside the session's own turn: Zeus's
        `worktree` tool, after the person approved it at dispatch. The turn
        is the caller, so only an episode in flight refuses it; one landing, two callers."""
        with self._starts:
            self._refuse_handing_off(session_id)
            running = self._episodes.busy()
            if running is not None:
                raise EngineBusy(f"{running} is in flight on this engine; land it when it ends")
            self._handing_off.add(session_id)
        return self._hand_off(session_id, choice)

    def _hand_off(self, session_id: str, chosen: GitHandoffChoice) -> HandoffResult:
        """The handoff's git work and its lines, `_handing_off` already held for the session."""
        try:
            done: list[HandoffResult] = []
            _folds, runs = self._caches_for(self._where(session_id))
            self._write(
                session_id,
                "a handoff",
                lambda session: done.append(
                    finalize_session(
                        session,
                        self.repo,
                        chosen,
                        runs=runs,
                        work_commit=self.config.work.commit,
                        reindex=partial(
                            _reindex_held,
                            self.repo,
                            denylist=self._denylist,
                            sandbox=self._index_sandbox(),
                        ),
                    )
                ),
            )
            return done[0]
        finally:
            with self._starts:
                self._handing_off.discard(session_id)

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Releases connections and leases. Session history is process-independent and
        survives this — the engine is not a server. A turn still running is stopped
        hard; what it committed stays on the session log, and so do its words not yet
        delivered, as far as the log holds them. A `turn()` after this is a `SessionError`.

        A turn waiting on `ask_user` in a host's `InteractionPort.ask` is handed the stop's
        token: a port that honours it returns, and the call is answered
        `CANCELLED`. A port that ignores it is waited for.

        A run in flight is paused, not ended, and before the conversations close: their
        hard stop reaches the run through its `CoderHandle`, and would end it first."""
        self._episodes.stop()
        self._conversations.close_all()
        self._components.clear()


class NonInteractive:
    """The default `InteractionPort`: every ask point resolves to its declared default,
    so an embedding host gets unattended semantics unless it opts in.

    Raises on a question with no default rather than inventing one. A question the harness
    cannot answer alone must be *designed* with a default; discovering that at runtime and
    guessing is how an unattended run silently picks the wrong branch.

    `authorizations` are the host's explicit risk authorizations, by `operation_id`:
    the only thing that lets an unattended action through. `--yes` alone
    is not one."""

    def __init__(self, authorizations: Mapping[str, ActionAuthorization] | None = None) -> None:
        self._authorizations = dict(authorizations or {})

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        return Answer(question_id=q.id, choice=q.default, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        """Approved only where an authorization was granted for this very operation, still
        covers its concrete spec (`is_valid_for`), and is at least the tier the router
        computed now. Anything else is denied: unattended means no (Q3's default)."""
        granted = self._authorizations.get(approval.operation_id)
        if (
            granted is not None
            and granted.is_valid_for(approval.spec_sha)
            and TIER_ORDER.index(granted.tier) >= TIER_ORDER.index(approval.tier)
        ):
            return ApprovalDecision(
                action_id=approval.action_id,
                verdict="approve",
                source="explicit_authorization",
            )
        return ApprovalDecision(action_id=approval.action_id, verdict="deny", source="default")

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        """A stop's default: unattended, it is approved, and the decision says
        so. An action it leads to above T0 is still confirmed at dispatch, where unattended is
        a deny (`confirm_action`)."""
        del stop, cancel
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)


class ScriptedInteraction:
    """An `InteractionPort` that answers from a script: for tests, and for a
    host that knows its answers before it starts. Each method takes the next answer of its
    kind, as a person would give it; once a kind's script is spent it answers as
    `NonInteractive` does, so a script that runs short never hangs a run. What it was
    asked is kept, in order."""

    def __init__(
        self,
        *,
        answers: Sequence[str] = (),
        actions: Sequence[Literal["approve", "deny", "abort"]] = (),
        stops: Sequence[StopDecision] = (),
    ) -> None:
        self._answers = list(answers)
        self._actions = list(actions)
        self._stops = list(stops)
        self._unattended = NonInteractive()
        self.asked: list[Question] = []
        self.approvals: list[ActionApproval] = []
        self.stops: list[ApprovalStop] = []
        """What each `confirm_stop` was shown, in order."""

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.asked.append(q)
        if not self._answers:
            return self._unattended.ask(q, cancel=cancel)
        return Answer(question_id=q.id, choice=self._answers.pop(0), source=AnswerSource.HUMAN)

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        self.stops.append(stop)
        if not self._stops:
            return self._unattended.confirm_stop(stop, cancel=cancel)
        given = self._stops.pop(0)
        return (
            given
            if given.source is not None
            else given.model_copy(update={"source": AnswerSource.HUMAN})
        )

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self.approvals.append(approval)
        if not self._actions:
            return self._unattended.confirm_action(approval, cancel=cancel)
        verdict = self._actions.pop(0)
        return ApprovalDecision(action_id=approval.action_id, verdict=verdict, source="human")
