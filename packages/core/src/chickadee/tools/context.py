"""The ports a tool may touch, and the context that carries them.

Data policy lives here, in what a role's tools are **given**, so a tool cannot break it
however it is written. Each port is a `Protocol`, implemented by its owner: the Reader is
`FileView`, the graph's renderers are `Navigator`, the session's blob store is
`HandleStore`. A port holds no rule of its own: the `files` port's owner
consults `read_denied`, the role's `FileRule` and, for a rule that hides tests, the
profile's `TestSpec.matches`; `AskPerson` consults the router through `approval`.

**Given per role, present or absent per scope.** A field is `None` when the role is not given
that port, or the scope lacks it (the scout's `exec_request` when escapes are not offered):
a tool needing an absent port is then not offered. A port that is present but has nothing
behind it answers its own failure, as a navigator answers `INDEX_UNAVAILABLE` without an
index: absence and unavailability stay different facts.

Only the ports something implements have a field. `mcp`, `mcp_request` and `box` get theirs
from the task that builds each, as an additive change; until then a tool needing one cannot
be offered (`has`). `search` and `exec_request` are the scout's.
"""

from __future__ import annotations

import threading
from collections.abc import Hashable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from chickadee.context.handle import Handle, HandleKind
from chickadee.context.results import CallCancelled, InvalidArgument, ToolFailure, ToolResult
from chickadee.edits.draft import DraftRefused
from chickadee.edits.formats.base import Targets
from chickadee.edits.lines import AnchorKind
from chickadee.edits.match import Region
from chickadee.edits.symbols import SymbolSpan
from chickadee.errors import HarnessBug
from chickadee.journal.effects import EffectLog
from chickadee.paths import Place
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import Answer as InteractionAnswer
from chickadee.schema.interaction import Question
from chickadee.schema.journal import Fresh, ReviewAnswer, TellOutcome
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Answer, Failure, Invocation, Port, Said
from chickadee.tools.validate import Arguments

if TYPE_CHECKING:
    from chickadee.graph.scoped import ScopedGraph
    from chickadee.hooks.fire import RoleHooks

__all__ = [
    "AgentsPort",
    "ApprovalPort",
    "Catalog",
    "DesignNotesPort",
    "DeskPort",
    "DraftCall",
    "DraftRefused",
    "ExecPort",
    "ExecRequestPort",
    "FileView",
    "HandleStore",
    "JobOutput",
    "JobsPort",
    "Locator",
    "Navigator",
    "NotesPort",
    "RecallPort",
    "Region",
    "SearchPort",
    "TeamPort",
    "ToolContext",
    "UserPort",
    "WebPort",
    "required",
]


class FileView(Protocol):
    """`read` and `grep_in` over the role's files. `context.Reader`
    is this, structurally: it applies the deny list and the caps, and bounds what it
    reads once, at source."""

    def read(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
        max_bytes: int | None = None,
        call_id: str | None = None,
    ) -> ToolResult | ToolFailure: ...

    def repeat_key(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
    ) -> Hashable | None:
        """What a read asks as a repetition counts it: the file, its bytes and the span;
        None when the read would fail."""
        ...

    def grep_in(
        self, handle: str, pattern: str, *, context: int = 2
    ) -> ToolResult | ToolFailure: ...

    def place(self, sent: str) -> Place | None:
        """Where a path a model sent lands in the repository: as sent when it names a file or
        directory there, else the one it most likely meant (`paths.suggest`), else None.
        Never a file the read rule refuses."""
        ...


class Navigator(Protocol):
    """The graph's answers, rendered as the text a model reads, or a typed failure. Never a
    file body (invariant 13)."""

    def atlas(self) -> str | ToolFailure: ...

    def symbol(self, name: str) -> str | ToolFailure: ...

    def neighbors(self, target: str, depth: int) -> str | ToolFailure: ...

    def impact(self, symbols: Sequence[str]) -> str | ToolFailure: ...

    def tests_for(self, paths: Sequence[str]) -> str | ToolFailure:
        """The test files that depend on `paths`, and what the graph cannot see."""
        ...


class HandleStore(Protocol):
    """The role's blob store: bytes that existed nowhere else, behind a
    handle. `Reader.store` is this."""

    def store(self, kind: HandleKind, data: bytes) -> Handle: ...


class ExecPort(Protocol):
    """A command, run under the role's authority and Isolation profile: Zeus's
    `Shell`, and the worker's with D74's stricter profile. Its answer is a
    result or the shell's typed refusal; an `InfraError` means nothing was spawned."""

    def run(
        self, args: Arguments, *, operation_id: str, cancel: threading.Event | None
    ) -> ToolResult | Failure: ...


class Locator(Protocol):
    """`locate`: pointers found by a bounded search agent on the cheap rung.
    It stops between rungs when `cancel` is set and returns what it found so far, marked
    partial; stopped before its search began, it is `CANCELLED{tool: locate}`."""

    def locate(
        self, question: str, *, cancel: threading.Event | None
    ) -> str | ToolFailure | CallCancelled: ...


@dataclass(frozen=True)
class JobOutput:
    """The end of a job's two streams as `JobsPort.output` reads them."""

    out: str
    err: str
    skipped: int
    """The bytes not read, of both streams together: 0 when all of it was. Before what was read
    for a running job; from the middle of its streams for an ended one (`kept`)."""
    kept: bool = False
    """An ended job's: what was kept of its output at the exit, both streams joined in `out`
    (`zeus/jobs.KEPT_END_BYTES`)."""


class JobsPort(Protocol):
    """Zeus's jobs (`zeus/jobs.JobTable`). Every answer is the session's:
    a job id another session started is unknown here."""

    @property
    def inline_bytes(self) -> int:
        """`zeus.shell_inline_bytes`: a tail's inline cap, as a shell result's."""
        ...

    @property
    def max_wait_s(self) -> int:
        """`zeus.job_max_wait_s`: the longest one `wait` blocks."""
        ...

    def known(self, job_id: str) -> bool: ...

    def is_run(self, job_id: str) -> bool:
        """Whether `job_id` names the session's run working in the background, as `listing`
        shows it beside the jobs: no job, so `kill` and `wait` refuse it, and
        `tail` shows its task board."""
        ...

    def run_board(self, job_id: str) -> str:
        """The task board of the run `job_id` names (`is_run`): each task it delegated so far,
        as its report will list them."""
        ...

    def listing(self) -> str:
        """Every job of the session, running or ended, one line each."""
        ...

    def output(self, job_id: str) -> JobOutput:
        """The end of its stdout and stderr as they are now, each read to a bound."""
        ...

    def await_exit(
        self, job_id: str, timeout_s: float, cancel: threading.Event | None
    ) -> str | None:
        """Its exit, worded, or None at `timeout_s` with it still running. Raises
        `process.supervisor.Cancelled` on the token, the job left running."""
        ...

    def kill(self, job_id: str, by: str) -> bool:
        """End it, naming who asked; False when it is not running."""
        ...


class RecallPort(Protocol):
    """What earlier sessions and runs recorded, as `recall` reads it: pointers into prior
    designs, runs and outcomes, never their bodies, bounded by the port."""

    def recall(self, query: str) -> ToolResult | ToolFailure: ...


class NotesPort(Protocol):
    """The role's typed notes to its own log: each returns the
    receipt naming what was recorded, or the refusal."""

    def note_concern(self, text: str, anchor: Sequence[str]) -> str | ToolFailure: ...

    def note_rejected(self, text: str, reason: str, *, at: int) -> str | ToolFailure: ...


class DesignNotesPort(Protocol):
    """Athena's notes on the design she is building: one
    `note` line on the session log, kept for the design's life. It can only append a note
    about the current design; it cannot write any other line, which is why it is not the
    `notes` port, whose concerns and rejections are Zeus's."""

    def note(self, text: str) -> str | ToolFailure: ...


class SkillsPort(Protocol):
    """The session's skills: a body loaded whole and pinned, or a
    typed refusal that loaded nothing. Loading only: no port method unloads."""

    def load(self, name: str, call_id: str) -> str | ToolFailure: ...

    def find(self, query: str) -> str:
        """The skills that best match `query`: name, kind, profile, what to
        pass, and a cost hint, best first. Never a body."""
        ...


class AgentsPort(Protocol):
    """The agents Zeus spawns and continues (Amendment 1 decision 2): one round of an
    agent, and its report as he reads it, or a refusal naming why nothing ran (an unknown
    name, a knowledge skill, an untrusted repository's agent skill, an agent still running or
    past its follow-ups)."""

    def spawn(
        self,
        skill: str | None,
        profile: str | None,
        task: str,
        handles: Sequence[str],
        *,
        call_id: str,
        cancel: threading.Event | None = None,
    ) -> str | ToolFailure: ...

    def follow_up(
        self,
        agent: str,
        message: str,
        *,
        call_id: str,
        cancel: threading.Event | None = None,
    ) -> str | ToolFailure: ...


class WebPort(Protocol):
    """The web (`external/web.Web`), one agent round's: every request through the
    external-call policy, each URL through `web_fetch`'s guards. What comes back is untrusted
    and framed as such; a refusal names the guard, and nothing was sent."""

    def fetch(
        self, url: str, offset: int, call_id: str, cancel: threading.Event | None = None
    ) -> str | ToolFailure:
        """`url`'s page as text from `offset` on, at most `web.fetch_max_chars` of it, saying
        how much is left; a redirect to another host is the answer, naming its target.
        `cancel`, the round's, withdraws any question the fetch would ask the person."""
        ...

    def search(self, query: str, source: str | None, call_id: str) -> str | ToolFailure:
        """At most `web.search_results` results for `query`, from `source` or the default
        sources, each a title, a URL and a snippet, never a page."""
        ...

    def saw(self, text: str) -> None:
        """The URLs in `text`, which the round was given, may be fetched: the agents layer
        passes the round's opening (`agents/spawn.py`)."""
        ...

    def fetched(self, url: str) -> str | None:
        """`url`'s text as the round fetched it, or as an earlier round of the agent was
        shown it, or None: what a web citation is checked against
        (`context/citations.check_web`)."""
        ...

    def carry(self, url: str, shown: Sequence[str]) -> None:
        """What an earlier round of the agent was shown of `url`: a follow-up's report may
        quote it still. The agents layer passes it (`agents/spawn.py`)."""
        ...


class SearchPort(Protocol):
    """Text or path search over the repository, including files the graph does not index:
    the scout's `Search`. It applies the deny list to every hit
    and bounds what it shows itself."""

    def __call__(
        self,
        query: str,
        *,
        kind: Literal["text", "path"] = "text",
        paths: Sequence[str] = (),
        cancel: threading.Event | None = None,
    ) -> str: ...


class ExecRequestPort(Protocol):
    """A requested command, granted or refused by a deterministic policy and run under
    DISCOVERY, never chosen: the scout's `Escapes`, for one `locate`
    call. What the model reads back, a refusal included, is the answer."""

    def request(self, command: str, reason: str) -> str: ...


class DraftCall(Protocol):
    """One edit call on the worker's draft: every change it makes stands, or, when it
    ends without `answer`, none does, so an `edits` list or a patch of several files is one
    call. A tool only resolves its anchor and calls `apply`, the draft's one write."""

    @property
    def targets(self) -> Targets: ...

    def text(self, path: str) -> str | DraftRefused:
        """A target's text as the draft holds it now, as it is matched (decoded)."""
        ...

    def crlf(self, path: str) -> bool:
        """Whether `path` is held decoded from a uniformly `\r\n` file, so an anchor written
        with its `\r\n` is matched as `\n` (`edits/match.find`'s `crlf`)."""
        ...

    def line(self, path: str, anchor: int | str, kind: AnchorKind) -> int | DraftRefused:
        """The 1-based line of `path` a line anchor names, in the draft's anchor style (`0`,
        the file's start, for an `after`), or `STALE_ANCHOR` with the lines there now."""
        ...

    def symbol(self, path: str, name: str) -> SymbolSpan | DraftRefused:
        """The one declaration of `path` that `name` names, in the draft as it now reads, its
        decorators included (`edits/symbols`), or why there is not one."""
        ...

    def apply(self, path: str, region: Region | None, content: str) -> DraftRefused | None:
        """`region` of `path`'s text replaced by `content`, or the whole file when `region` is
        None (a write, the only edit an absent create target takes)."""
        ...

    def delete(self, path: str) -> DraftRefused | None:
        """`path` removed, when the draft's `removable` lets it go; refused in a
        draft given none."""
        ...

    def move(self, source: str, target: str) -> DraftRefused | None:
        """`source` to `target`, which must not exist: a delete and a create in this call."""
        ...

    def answer(self) -> str | DraftRefused:
        """Keep the call's changes, and the answer naming them with each changed file's
        syntax; or `SYNTAX_FAIL`, when the call broke a file and is undone."""
        ...


EditScope = AbstractContextManager[DraftCall]
"""One edit call's scope: its changes stand when it is answered, or none do."""


class EditsPort(Protocol):
    """What every edit tool writes through: the box's `WorktreeEdits`
    (`box/edits.py`), whose draft is the task's worktree."""

    def edit(self) -> EditScope: ...


class DocsPort(Protocol):
    """What `doc` writes through: the edit port, scoped to the caller's
    part of the session's open work, Zeus's `design/` or Athena's `plan/`. Each call is
    recorded as the role records any write: Athena's as a step, Zeus's as his shell's are."""

    def resolve(self, file: str, title: str | None) -> str | DraftRefused:
        """`file`'s path from the repository root, in the caller's part, or why not: no work
        open and no `title` to open one, a path that leaves the part, a folder not the
        harness's. Zeus's first call opens the work, and his worktree with it (4b)."""
        ...

    def edit(self, call: Invocation, approved_sha: str) -> EditScope:
        """One `doc` call's draft, recorded as the caller's role records a write: under
        `call`'s operation, approving its arguments' sha."""
        ...

    def noted(self, answer: Answer) -> Answer:
        """The call's answer with what its record says, once the record is closed."""
        ...


class TeamPort(Protocol):
    """What Zeus's `pause` and `tell` reach: the session's run, handed over
    in the background, suspended now, or told something."""

    def pause(self) -> Said | InvalidArgument: ...

    def tell(self, text: str, sources: tuple[int, ...]) -> Said | InvalidArgument:
        """Zeus's message to the session's live or suspended run, citing the person's turns
        it came from; written on the session log and delivered at the run's next boundary."""
        ...


class WorktreePort(Protocol):
    """What Zeus's `worktree` moves the session through: into a worktree of
    its own and back, and landing or discarding it. The harness does every git step; a landing
    and a discard are asked of the person first, and so is entering a worktree the person's
    words did not name."""

    def enter(
        self, *, operation_id: str, cancel: threading.Event | None
    ) -> Said | DraftRefused: ...

    def leave(self) -> Said | DraftRefused: ...

    def land(
        self, how: str, *, operation_id: str, cancel: threading.Event | None
    ) -> Said | DraftRefused: ...

    def discard(
        self, *, operation_id: str, cancel: threading.Event | None
    ) -> Said | DraftRefused: ...


class FinishPort(Protocol):
    """Where the worker's `finish` hands over its report: the round ends on it, with
    the report as the worker's reply, for a model that ends its turns only by calling a tool."""

    def finish(self, report: str) -> str: ...


class DeskPort(Protocol):
    """Athena's three task tools: the harness
    side of her plan, her delegations and her reviews, `box/desk.TaskDesk`. Each answer is the
    text she reads next, or a refusal naming the rule. `call_id` is her call's: the desk writes
    it on the line it makes, so a call a crash cut off continues from that line."""

    def plan(self, text: str | None, *, call_id: str) -> str | ToolFailure:
        """The work folder's `plan/` (`text` None), or a plan's text where
        the work has no folder."""
        ...

    def delegate(
        self,
        title: str | None = None,
        brief: str | None = None,
        *,
        task: str | None = None,
        call_id: str,
    ) -> str | ToolFailure:
        """A task of the pinned plan by id, or a title and brief where the
        work has no folder."""
        ...

    def review(
        self,
        task: str,
        answer: ReviewAnswer,
        text: str | None,
        *,
        call_id: str,
        fresh: Fresh = None,
    ) -> str | ToolFailure:
        """Her answer to the open task's report: what happens next, and
        with `fresh` what is put back first (`True` every file, paths only those)."""
        ...

    def end_outgrown(self, *, call_id: str) -> str | ToolFailure:
        """A one-task plan ended `OUTGROWN`: refused in a design's episode
        and while a task is open; answered, her reply is the report."""
        ...

    def reply(
        self, tell: str, outcome: TellOutcome, text: str, *, call_id: str
    ) -> str | ToolFailure:
        """Her answer to a message from Zeus delivered to her, one per message."""
        ...


class Catalog(Protocol):
    """The registry as this role is offered it, for Zeus's `tools` tool (33 §3 row 8, §4 tier
    3): the specs, their description templates' text, a spec's schema as the model is shown
    it, and how many briefs one answer lists."""

    def offered(self) -> tuple[ToolSpec, ...]: ...

    def descriptions(self) -> Mapping[str, str]: ...

    def shown_schema(self, spec: ToolSpec) -> Mapping[str, object]: ...

    @property
    def entry_cap(self) -> int: ...


class UserPort(Protocol):
    """The person. `ask` is contract 6's typed question (`InteractionPort.ask`): a session with
    nobody attached answers the question's default. `confirm` is D164 decision 5b's: a
    role whose ceiling lacks this port cannot hold an `AskWhen` rule, so the worker is refused
    above T0 and never prompted. `ask`'s `cancel` is the call's D95 token, handed to
    the port so a hard stop never waits on a person."""

    def ask(
        self, question: Question, *, cancel: threading.Event | None = None
    ) -> InteractionAnswer: ...

    def confirm(self, question: str, tier: Tier) -> bool: ...


class ApprovalPort(Protocol):
    """The router, consulted and never copied: the tier this concrete call
    needs once the fact rows, the command table, the preferences and the grants have been
    asked (`approval/compute.py`). `T0` means nothing needs a person's yes."""

    def tier(self, spec: ToolSpec, arguments: Mapping[str, object]) -> Tier: ...


@dataclass(frozen=True)
class ToolContext:
    """One optional field per port that exists; `None` is *not given*. `role`, `cancel`,
    `effect_log`, `constrained` and `hooks` are context, not ports: what a call runs under, not
    what a tool touches."""

    role: str
    files: FileView | None = None
    navigator: Navigator | None = None
    graph: ScopedGraph | None = None
    """`16-graph.md` §5.1, per role."""
    handles: HandleStore | None = None
    exec: ExecPort | None = None
    locator: Locator | None = None
    journal: RecallPort | None = None
    notes: NotesPort | None = None
    search: SearchPort | None = None
    exec_request: ExecRequestPort | None = None
    catalog: Catalog | None = None
    user: UserPort | None = None
    approval: ApprovalPort | None = None
    edits: EditsPort | None = None
    skills: SkillsPort | None = None
    design_notes: DesignNotesPort | None = None
    desk: DeskPort | None = None
    finish: FinishPort | None = None
    docs: DocsPort | None = None
    jobs: JobsPort | None = None
    worktree: WorktreePort | None = None
    agents: AgentsPort | None = None
    team: TeamPort | None = None
    web: WebPort | None = None
    cancel: threading.Event | None = None
    effect_log: EffectLog | None = None
    """Where the `EffectLifecycle` wrapper writes a dispatcher-recorded lifecycle."""
    constrained: frozenset[str] = frozenset()
    """The registry names this run's model decodes under their grammar (`RolePolicy.constrain`):
    such a tool takes its grammar's language alone."""
    hooks: RoleHooks | None = None
    """The role's hooks: context, as `cancel` is, not a port a tool touches."""

    def has(self, port: Port) -> bool:
        """Whether `port` is given. A port with no field yet is never given."""
        return getattr(self, port, None) is not None


def required[T](port: T | None, name: Port) -> T:
    """A port a tool's `needs` names. A dispatcher offers a tool only when its ports are
    given, so a missing one here is a harness bug, never the model's."""
    if port is None:
        raise HarnessBug(f"a tool ran without the {name!r} port it needs")
    return port
