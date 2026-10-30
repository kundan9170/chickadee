"""One episode's parts, composed in one place.

The engine composes the long-lived parts once per episode (`RunParts`, `RoleSetting`,
`EpisodeSetting`); the episode is composed over them and its own journal here, and nowhere
else: the worker, the desk, Athena and the task loop that drives them, as `tests/box`'s rig
composes them, so what the tests exercise is what a real episode is built from.

What an episode gets from here, over the run's journal:
- each role's reads, over the workspace under the deny list, through a reader of its own (its
  handles and bytes are its own, as the scout's are), and the workspace's graph to
  navigate;
- one edit port over the worktree (`box/edits.WorktreeEdits`), the syntax of each written file
  judged by its own language's profile (`box/syntax.SyntaxChecker`);
- each role's box shell, bound to the run's journal, where a backend enforces its seal; a role
  on a machine with none is offered no shell;
- the commit (`box/commit.Projector`), and stop 2 asked through the session's port, bound to
  the run so the session-log pair names it;
- Athena's notes on the session log, under the approved work's sha;
- the episode's hooks: the set bound for it, over its journal, each role holding them
  with its name, its model, and where in the run it is.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final

from chickadee.adapters.base import select_blocks
from chickadee.approval.stops import (
    Reread,
    StopOutcome,
    StopWithdrawn,
    WorkspacePreview,
    ask_stop,
)
from chickadee.approval.tier import Preferences
from chickadee.box.athena import Athena, AthenaParts, Notes
from chickadee.box.carry import Carried
from chickadee.box.commit import Projector, Reindexer
from chickadee.box.desk import DeskParts, TaskDesk
from chickadee.box.edits import (
    ATHENA_FENCE,
    WORKER_FENCE,
    PlanDocs,
    WorkFence,
    WorkRule,
    WorktreeEdits,
)
from chickadee.box.loop import HeadParts, OneTaskPlan, TaskLoop
from chickadee.box.request import AttachedFile, CoderRequest
from chickadee.box.resume import Asking, never_asking
from chickadee.box.rung import BoxRung
from chickadee.box.shell import BoxShellFor
from chickadee.box.steps import Beside
from chickadee.box.syntax import Spawn, SyntaxChecker
from chickadee.box.tells import Tells, Told
from chickadee.box.worker import WorkerParts, WorkerRun, WorkerStopped
from chickadee.box.worker import instructions as worker_instructions
from chickadee.boxctx.athena import NumberedLog, design_text
from chickadee.boxctx.records import symbol_table
from chickadee.budget.manager import CostLedger
from chickadee.client.live import LiveSink
from chickadee.context.attachment import AttachCaps, CheckedFile, RefusedFile, check
from chickadee.context.changes import GitChanges
from chickadee.context.homes import ConversationHome, Now, Scratch
from chickadee.context.questions import for_run
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.terms import ReadRules, Vision
from chickadee.context.work_line import WorkLine
from chickadee.errors import HarnessBug
from chickadee.graph.queries import Graph
from chickadee.graph.servers import EpisodeServers
from chickadee.hooks.bind import HookConfig, ScriptRunner
from chickadee.hooks.builtins import default_set
from chickadee.hooks.fire import Hooks, HookSet, RoleHooks
from chickadee.hooks.sinks import RunHookSink
from chickadee.journal.errors import BlobMissing
from chickadee.journal.port import JournalPort
from chickadee.journal.projection import RunProjection
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile
from chickadee.schema import RetryParams
from chickadee.schema.interaction import ApprovalStop, InteractionPort
from chickadee.schema.journal import TellOutcome
from chickadee.secrets.denylist import Denylist
from chickadee.state.roots import roots as resolve_roots
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import SearchPort, ToolContext
from chickadee.tools.policy import RolePolicy
from chickadee.work import DEFAULT_DIR, design_files
from chickadee.zeus.tools.graphnav import GraphNavigator

__all__ = ["EpisodeSetting", "RoleSetting", "RunParts", "attachments_block", "compose_episode"]


@dataclass(frozen=True)
class RunParts:
    """What an episode is composed over and does not own: the engine's long-lived parts,
    bound to the workspace the run writes."""

    worktree: Path
    profiles: tuple[LanguageProfile, ...]
    """Every language the workspace holds: each written file is judged by its own."""
    denylist: Denylist
    graph: Graph
    indexer: Reindexer
    budget: CostLedger
    """The session's ledger: every run charges the same one, and a run's
    start resets only its run scope (`start_run`), never the session's."""
    interaction: InteractionPort
    """The session's port: stop 2 is asked through it, bound to the run."""
    shell_for: BoxShellFor | None = None
    """Each role's shell over a run's journal (`box/shell.shell_for`), which says whether it is
    sealed; `None` where no backend enforces its seal and `require_sandbox` is set, and then no
    role is offered `shell` and each prompt says why."""
    spawn: Spawn | None = None
    """The harness's own sandboxed process, for a syntax check that is a command; `None` and
    such a file is not judged (`syntax unchecked`)."""
    notes: NumberedLog | None = None
    """The session log Athena's notes are written to; `None`, and she has no `note`."""
    prefs: Callable[[], Preferences] | None = None
    """The person's approval.toml: the edit port refuses a path it denies, and each
    step puts back one a command changed. Read per call; `None` is no preferences."""
    languages: ProfileSet | None = None
    """The profile set the index read the workspace by, so a written file's
    language is the index's answer; `None`, `profiles` alone answer."""
    search: SearchPort | None = None
    """Text and path search over the worktree, for both roles: `find` searches the
    index, and a role without a shell has no other way to list files. `None`, no `search`."""
    work_dir: str = DEFAULT_DIR
    """`[work] dir`: the folder both roles' edit tools refuse (10)."""
    work_commit: bool = True
    """`[work] commit` (B7): false, and the work folder is never committed."""
    servers: EpisodeServers | None = None
    """The episode's language servers, which the box closes before it gives
    its lease back; `None`, and each index pass starts its own."""
    beside: Beside | None = None
    """The session's running jobs (`zeus/jobs.JobTable`): each step beside one
    takes a fresh "before" and says what it cannot separate; `None`, no job runs beside."""
    live: LiveSink | None = None
    """Where both roles' replies are shown as they stream: the session's running turn.
    `None`, nothing is shown live."""
    python_unsandboxed: bool = False
    """`[sandbox] allow_unsandboxed_python`: the worker's Python blocks may
    run in a plain child process where no sandbox backend enforces anything."""
    scripts: ScriptRunner | None = None
    """How a hook's script runs over this workspace, under the run's
    backend; `None` where nothing seals, and then a script entry is refused when bound."""
    runs: Callable[[str], RunProjection] | None = None
    """The engine's one fold of the run journals (C2), through which a run that continues a
    work reads the earlier runs (`box/carry.py`)."""
    attachment_bytes: Callable[[str], bytes] | None = None
    """The session's blobs, which hold the files the person sent: each the request
    names is checked again and handed to each role's reader. `None`, a run with no session,
    and its head says the files are not available to it."""
    attach_caps: AttachCaps | None = None
    """The caps a turn's files were checked under (`AttachCaps.of`), so the box checks them
    again under the same."""
    told: Callable[[], tuple[Told, ...]] | None = None
    """The session's tells, each with its text; `None`, no tell reaches the run."""
    told_token: threading.Event | None = None
    """Set when Zeus tells the run something (`BoxEpisode.told`): the `Tells` token."""
    replied: Callable[[str, TellOutcome, str], None] | None = None
    """Athena's reply to a tell, told to Zeus; `None`, nobody is told."""


@dataclass(frozen=True)
class RoleSetting:
    """One box role as the config resolves it: its rung, its tools, its limits, how its
    requests go out and how large they may be."""

    rung: BoxRung
    policy: RolePolicy
    limits: Mapping[str, int]
    """Its resolved rows by name (`limits.role_limits`), as its dispatcher's rules read them."""
    vision: Vision
    params: RetryParams
    budget: int
    """Tokens a request may hold: the rung's window less its output reserve."""
    keep_recent: int


@dataclass(frozen=True)
class EpisodeSetting:
    """The episode's own settings, from the config and the templates in force."""

    worker: RoleSetting
    athena: RoleSetting
    head: HeadParts
    """Athena's instructions and the plan and brief templates in force."""
    brief_template_sha: str
    workspace: WorkspacePreview
    """Where the work runs, for stop 2's display."""
    untracked_max_bytes: int
    task_seconds: int
    recut_cap: int
    one_task_max: int
    """`athena.one_task_max_tasks`."""
    design_layer: int
    note_cap: int
    hooks: HookConfig = field(default_factory=HookConfig)
    """What the config says of hooks: bound when the run starts (`episode.BoxRun`),
    with the run's script runner, never before."""


def compose_episode(
    parts: RunParts,
    setting: EpisodeSetting,
    journal: JournalPort,
    request: CoderRequest,
    *,
    cancel: threading.Event | None = None,
    pause: threading.Event | None = None,
    asking: Asking = never_asking,
    base: str | None = None,
    hooks: HookSet | None = None,
) -> TaskLoop:
    """The task loop of the episode `journal` records, for `request`'s approved work. `cancel`
    is the person's hard stop; `pause`, a soft stop, taken at the next task boundary;
    `asking`, whether a step's permission question still waits on the session log;
    `base`, the commit the run's changes are measured from,
    which each role's `read(at="@changed")` compares against (None, and it says none is set);
    `hooks`, the set the run bound, or the built-in defaults."""
    root = parts.worktree
    in_place = request.workspace == "in_place"
    fired = Hooks(hooks or default_set(setting.hooks.limits), RunHookSink(journal))
    navigator = GraphNavigator(parts.graph, parts.profiles)
    checker = SyntaxChecker(root, parts.profiles, parts.spawn, languages=parts.languages)

    def edits(policy: RolePolicy, fence: WorkRule | None) -> WorktreeEdits:
        return WorktreeEdits(
            root,
            parts.denylist,
            checker,
            edit_check=policy.edit_check,
            untracked_max_bytes=setting.untracked_max_bytes,
            # A line tool's anchors are read in the policy's style; without one, a
            # configured `replace_lines` would raise on its first call.
            style=policy.anchor_style if policy.anchors_lines(BUILTINS) else None,
            prefs=parts.prefs,
            work=None if fence is None else WorkFence(parts.work_dir, fence),
        )

    # The shell names the worker's task on every claim, and the worker is made after it.
    made: list[WorkerRun] = []
    w = setting.worker

    # A box role holds a shell where `shell_for` offers one and its policy enables it, so each
    # prompt says what it holds, never a tool the run withheld; held unsealed, it
    # says so, and withheld for want of a sandbox, why.
    def holds(policy: RolePolicy) -> frozenset[str]:
        if "shell" not in policy.enabled:
            return frozenset()
        if parts.shell_for is None:
            return frozenset({"no_sandbox"})
        return frozenset({"shell", "sealed" if parts.shell_for.sealed else "unsealed"})

    worker_reader = _reader(
        root,
        parts.denylist,
        w.vision,
        journal,
        parts.graph,
        base,
        ReadRules.of("worker", w.limits),
        now=lambda: made[0].now() if made else None,
    )
    attached = attachments_block(
        request.attachments, parts.attachment_bytes, parts.attach_caps, worker_reader
    )
    # One home for a tell's delivery and reply, shared by the three that read
    # it: the worker's round it holds, the desk that answers Athena, and her loop.
    tells = (
        Tells(
            journal.run_id,
            journal,
            parts.told,
            prior=request.continues.prior_runs if request.continues is not None else (),
            runs=parts.runs,
            token=parts.told_token,
            replied=parts.replied,
        )
        if parts.told is not None
        else None
    )
    worker = WorkerRun(
        w.rung,
        w.policy,
        ToolContext(
            role="worker",
            # The run's hard stop, which every call is handed: a command is reaped and a
            # permission question let go when it is set.
            cancel=cancel,
            constrained=frozenset(w.policy.constrain),
            files=worker_reader,
            handles=worker_reader,
            navigator=navigator,
            search=parts.search,
            edits=edits(w.policy, WORKER_FENCE),
            exec=None
            if parts.shell_for is None
            else parts.shell_for(journal, "worker", lambda: made[0].task if made else None),
            hooks=RoleHooks(
                fired,
                "worker",
                w.rung.model,
                run_id=journal.run_id,
                place=lambda: (made[0].task, made[0].round) if made else (None, None),
            ),
        ),
        journal,
        w.limits,
        WorkerParts(
            root=root,
            profiles=parts.profiles,
            graph=parts.graph,
            indexer=parts.indexer,
            params=w.params,
            budget=w.budget,
            keep_recent=w.keep_recent,
            untracked_max_bytes=setting.untracked_max_bytes,
            task_seconds=setting.task_seconds,
            prefs=parts.prefs,
            syntax=checker,
            blocks=holds(w.policy),
            work_dir=parts.work_dir,
            beside=parts.beside,
            python_unsandboxed=parts.python_unsandboxed,
        ),
        cancel=cancel,
        fixed=select_blocks(worker_instructions(work_dir=parts.work_dir), holds(w.policy))
        + attached,
        live=parts.live,
        tells=tells,
    )
    made.append(worker)
    port = for_run(parts.interaction, journal.run_id)

    def stops(stop: ApprovalStop, *, reread: Callable[[], Reread] | None = None) -> StopOutcome:
        try:
            return ask_stop(stop, port, reread=reread, cancel=cancel)
        except StopWithdrawn as exc:
            # The run's hard stop ended the wait: the same stop as one given
            # mid-task, so a dropped run ends and a leaving one is suspended, the stop to be
            # asked again on resume.
            raise WorkerStopped("stopped by the user") from exc

    desk = TaskDesk(
        journal,
        worker,
        Projector(root, parts.indexer, in_place=in_place, run_id=journal.run_id),
        stops,
        DeskParts(
            root=root,
            # Stop 2 shows where this request runs, whatever the setting was built for.
            workspace=replace(setting.workspace, mode=request.workspace),
            profiles=parts.profiles,
            untracked_max_bytes=setting.untracked_max_bytes,
            recut_cap=setting.recut_cap,
            one_task_max=setting.one_task_max,
            brief_template_sha=setting.brief_template_sha,
            design_layer=setting.design_layer,
            prefs=parts.prefs,
            work_dir=parts.work_dir,
            beside=parts.beside,
            # A design in its folder: her plan is its `plan/`, checked against the
            # templates in force; a host's design text, or a one-task plan, has none.
            work=_work_of(request),
            plan_template=setting.head.plan_template,
            brief_template=setting.head.brief_template,
            design_files=_design_files(request),
            design_tree=request.design.work.tree
            if request.design is not None and request.design.work is not None
            else None,
            work_commit=parts.work_commit,
            # The harness's own files of the folder: no role's fence, no model's check.
            records=WorktreeEdits(
                root,
                parts.denylist,
                checker,
                edit_check="off",
                untracked_max_bytes=setting.untracked_max_bytes,
            ),
            private=resolve_roots(root).repo_local / "work",
            carried=_carried(request, parts.runs, root),
        ),
        indexer=parts.indexer,
        # Taken at each delegation, over every indexed file, so a task's facts diff its symbols
        # against the tree it started from; never journaled, so None after a resume.
        symbols=lambda: symbol_table(parts.graph, parts.graph.indexed_paths()),
        pause=pause,
        asking=asking,
        hooks=RoleHooks(fired, "athena", setting.athena.rung.model, run_id=journal.run_id),
        tells=tells,
    )
    a = setting.athena
    # Her reader answers from her conversation, and she is made after it, as the worker is.
    hers: list[Athena] = []
    athena_reader = _reader(
        root,
        parts.denylist,
        a.vision,
        journal,
        parts.graph,
        base,
        ReadRules.of("athena", a.limits),
        now=lambda: hers[0].now() if hers else None,
    )
    athena = Athena(
        a.rung,
        a.policy,
        ToolContext(
            role="athena",
            cancel=cancel,
            constrained=frozenset(a.policy.constrain),
            files=athena_reader,
            handles=athena_reader,
            navigator=navigator,
            search=parts.search,
            edits=edits(a.policy, ATHENA_FENCE),
            # Her `doc` writes the open work's `plan/` through a port of its own, never gated by
            # the held task as her code edits are.
            docs=PlanDocs(root, edits(a.policy, None), _work_of(request)),
            exec=None
            if parts.shell_for is None
            else parts.shell_for(journal, "athena", lambda: desk.held),
            design_notes=None
            if parts.notes is None
            else Notes(parts.notes, request.work_sha, journal.run_id, setting.note_cap),
            hooks=RoleHooks(
                fired,
                "athena",
                a.rung.model,
                run_id=journal.run_id,
                place=lambda: (desk.held, None),
            ),
        ),
        journal,
        a.limits,
        desk,
        AthenaParts(
            params=a.params, budget=a.budget, keep_recent=a.keep_recent, blocks=holds(a.policy)
        ),
        cancel=cancel,
        live=parts.live,
        tells=tells,
    )
    hers.append(athena)
    # Her own block, from her own reader: an id is lengthened past a prefix another handle in
    # her home already holds, so the worker's ids are not promised to be hers.
    athena_attached = attachments_block(
        request.attachments, parts.attachment_bytes, parts.attach_caps, athena_reader
    )
    task = request.task
    return TaskLoop(
        athena,
        desk,
        journal,
        replace(
            setting.head,
            fixed=select_blocks(setting.head.fixed, holds(a.policy)) + athena_attached,
        ),
        root=root,
        params=a.params,
        untracked_max_bytes=setting.untracked_max_bytes,
        design=_design(request),
        one_task=OneTaskPlan(task.title, task.brief) if task is not None else None,
        indexer=parts.indexer,
    )


ATTACHED: Final = "Attachments the person sent with this work:"
NOT_SHOWN: Final = "not shown to you: your tool results are text"


def attachments_block(
    files: Sequence[AttachedFile],
    source: Callable[[str], bytes] | None,
    caps: AttachCaps | None,
    reader: Reader,
) -> str:
    """The files approved with the work, each checked again from `source`,
    the session's blobs, and kept by `reader` under a handle of its own, as a block for the
    role's instructions: what each is, and the call that reads it. Empty when there are none.
    A role's tool results are text, so an image is named and never offered as if it could be
    seen, and a PDF is offered as its text layer."""
    if not files:
        return ""
    caps = caps or AttachCaps.defaults()
    rows: list[str] = ["", "", ATTACHED]
    for f in files:
        checked = _checked_again(f, source, caps)
        if isinstance(checked, str):
            rows.append(f"- {f.id} {f.name}: {checked}")
            continue
        minted = reader.attach(checked, inline_bytes=caps.text_inline_bytes)
        label = f"- {f.id} {f.name} · {checked.kind} · {checked.size}"
        if checked.kind == "image":
            rows.append(f"{label} · {NOT_SHOWN}")
        elif checked.kind == "pdf" and minted.text_handle is not None:
            rows.append(f'{label} · text layer only: read(handle="{minted.text_handle.id}")')
        elif checked.kind == "pdf":
            rows.append(f"{label} · no text layer: {NOT_SHOWN}")
        else:
            rows.append(f'{label} · read(handle="{minted.handle.id}")')
    return "\n".join(rows)


def _checked_again(
    f: AttachedFile, source: Callable[[str], bytes] | None, caps: AttachCaps
) -> CheckedFile | str:
    if source is None:
        return "not available to this run: it has no session"
    try:
        data = source(f.sha)
    except BlobMissing:
        return "not available to this run: its bytes are gone from the session"
    checked = check(f.name, data, caps)
    return checked.why if isinstance(checked, RefusedFile) else checked


def _carried(
    request: CoderRequest, runs: Callable[[str], RunProjection] | None, root: Path
) -> Carried:
    """What the run carries from the work's earlier runs, read through the engine's fold."""
    if request.continues is None:
        return Carried.none()
    if runs is None:
        raise HarnessBug("a run that continues a work was composed with no reader of its runs")
    return Carried.of(request.continues, runs, root)


def _work_of(request: CoderRequest) -> WorkLine | None:
    """The work folder the approved design is in, as its pin names it; None for a host's
    design text or a one-task plan."""
    pin = request.design.work if request.design is not None else None
    return None if pin is None else WorkLine(name=pin.name, dir=pin.dir, state="open")


def _design_files(request: CoderRequest) -> tuple[tuple[str, str], ...]:
    design = request.design
    return design_files(design.text) if design is not None and design.work else ()


def _design(request: CoderRequest) -> str | None:
    """What Athena works from: a work folder's pinned README with its
    files by path, or a host's design text as it was approved."""
    design = request.design
    if design is None:
        return None
    if design.work is None:
        return design.text
    return design_text(design.text, design.work.folder, design_files(design.text))


def _reader(
    root: Path,
    denylist: Denylist,
    vision: Vision,
    journal: JournalPort,
    graph: Graph | None,
    base: str | None = None,
    rules: ReadRules | None = None,
    *,
    now: Callable[[], Now | None] | None = None,
) -> Reader:
    """A role's own reader over the workspace: its handles and bytes are its own. It
    reads this run's blobs too: the box shell stores a command's output there, and
    a long output's reference must open, where without the store it answered
    `HANDLE_EXPIRED{absent}`. Another run's blob is not this role's to read. `graph`, the
    worktree's index, gives each read its outline and its diagnostics; None, and a read says no
    index is attached. `base` is what `at="@changed"` compares
    the worktree against. `rules` are the role's own read limits, never
    Zeus's; None, the reader's defaults. `now` is the role's conversation as it stands, which
    its home answers a re-read from; None, and nothing is held."""
    own = Scratch() if now is None else ConversationHome(now)

    def run_blob(run_id: str, sha: str) -> bytes:
        if run_id != journal.run_id:
            raise BlobMissing(sha, run_id=run_id)
        return journal.get_blob(sha)

    resolver = HandleResolver(root, own.blob, run_blob=run_blob)
    languages = ProfileSet.load(repo=root)
    return Reader(
        own,
        resolver,
        vision=vision,
        denylist=denylist,
        languages=languages,
        index=graph,
        diagnostics=graph,
        changes=None if base is None else GitChanges(root, base),
        rules=rules,
    )
