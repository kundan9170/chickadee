"""What each pane of the TUI shows: plain state, folded from
the wire by `feed.py` and drawn by the panes, never computed by them.

**Typed end to end.** An entry holds the wire record it came from (`chickadee_wire.models`),
not text made from it: what a line says is the pane's to draw, and a field the pane reads is
one the contract has.

**Absent is unknown, never clean** (invariant 7). A value the wire has not sent is `None`
or empty and is drawn as not known: no spend snapshot is not `$0`, a file no check read is not
"syntax ok", a test command that collected nothing is not a pass.

**State is a word.** Each state is an enum member; its glyph and colour are the theme's
(colour is never the only signal).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from chickadee_cli.render.board import RunView
from chickadee_cli.render.live import Replies
from chickadee_cli.tui.names import ROLE_WORDS
from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import (
    BudgetP,
    Call,
    CoderOutcome,
    CommandFactP,
    EffectCompletedP,
    HandoffSummary,
    JobView,
    Liveness,
    ModelWiring,
    PlanTaskP,
    RunDiff,
    RunEndP,
    RunHeader,
    SessionBudget,
    SessionInfo,
    TaskFactsP,
    TaskReportedP,
    TaskReviewedP,
    WireModel,
    WorkspaceRecord,
)
from chickadee_wire.questions import OpenQuestion, Questions

__all__ = [
    "ATHENA_TAIL",
    "AgentItem",
    "Approval",
    "Asked",
    "AthenaEntry",
    "AthenaKind",
    "AthenaPane",
    "BackgroundPane",
    "CarriedTask",
    "Connection",
    "DiffFile",
    "DiffScope",
    "DiffView",
    "FactsLine",
    "Header",
    "HeaderSpend",
    "Pane",
    "PlanState",
    "RungCall",
    "SpendLevel",
    "Stage",
    "StaleRoute",
    "TuiState",
    "WorkerEntry",
    "WorkerKind",
    "WorkerPane",
    "WorkerTask",
    "ZeusEntry",
    "ZeusKind",
    "ZeusPane",
    "call_of",
    "drawn_as_modal",
    "drawn_in_card",
    "facts_line",
]

ATHENA_TAIL: Final = 50
"""Athena's tail keeps her last calls and verdicts, not her whole run: the board holds the run,
and `/journal` every record."""


class Pane(StrEnum):
    """What a fold changed, so a screen redraws only that ."""

    HEADER = "header"
    ZEUS = "zeus"
    ATHENA = "athena"
    WORKER = "worker"
    STOPS = "stops"
    DIFF = "diff"
    """The session's committed diff (`/handoff`), which only the diff screen draws."""
    BACKGROUND = "background"
    """Zeus's agents and jobs."""


class StaleRoute(StrEnum):
    """A route whose answer a record has made stale; the feed names it, the screen reads it."""

    INFO = "info"
    """The session's info (`/session/{sid}`), read again when a turn of the person's lands
    while the session has no title: its title is that turn's first line, which the
    screen opened too early to see."""
    JOBS = "jobs"
    """`/jobs`, read again when a `shell_effect` record names a job's
    effect: one started or ended. Never on a timer."""
    HEADER = "header"
    """`/session/{sid}/header`: where the session runs and the checkout's uncommitted
    paths, a `git status` per read; read again when a record says the tree may have moved, when
    the terminal gives the screen its focus back, and on `HEADER_POLL_S` for what nothing says
    (the person's own commit in their terminal)."""
    WORK = "work"
    """`/session/{sid}/work`: where the open work stands, each task across its runs,
    read again when an `episode` or a `plan_written` lands and when a run ends."""
    HIDDEN = "hidden"
    """A client re-asks on each undo or redo record."""
    BUDGET = "budget"
    LIVENESS = "liveness"
    HANDOFF = "handoff"
    """The session's committed diff from its base (`HandoffSummary.diff`): read only while the
    diff screen is open, and again when a task's commit or the run's end lands then (each read
    runs git on the server)."""
    TASK_DIFF = "task_diff"
    """The shown task's diff (`/run/{run_id}/diff/task/{task_id}`): read
    again when its steps or reports move. Not a session route: the screen reads it for the
    task the worker pane shows."""
    STEP_DIFFS = "step_diffs"
    """Each step's diff (`/run/{run_id}/diff/step/{operation_id}`), read
    once a step that changed files lands: the line counts its card shows."""
    RUNG = "rung"
    """`/rung`, read again when a `model_switched` record says a rung's model moved."""


class Stage(StrEnum):
    """The header's stage strip, in its order."""

    DESIGN = "design"
    PLAN = "plan"
    WORK = "work"
    CHECK = "check"
    """Athena's final check: every task the approved plan lists has an end on the board, and
    the run has not (`plan_written.tasks`)."""
    HANDOFF = "handoff"


class SpendLevel(StrEnum):
    """Amber at the warning threshold, red at the cap, as the run's `budget` records say."""

    OK = "ok"
    WARNING = "warning"
    CAP = "cap"


class Connection(StrEnum):
    """The wire's state, set by whatever reads the streams."""

    LIVE = "live"
    RECONNECTING = "reconnecting"
    LOST = "lost"


# -- the header ------------------------------------------------------------------------------


@dataclass
class HeaderSpend:
    """The session family's spend against its cap. `total_usd` and `cap_usd` are the engine's
    (`/budget`); until the first answer they are unknown, not zero."""

    budget: SessionBudget | None = None
    level: SpendLevel = SpendLevel.OK
    capped: BudgetP | None = None
    """The run's `hard_cap` record, once the cap is crossed: whether the run stopped there
    (`terminal`) is its word, for the banner."""

    @property
    def total_usd(self) -> float | None:
        return None if self.budget is None else self.budget.spent_usd

    @property
    def cap_usd(self) -> float | None:
        """None when unknown or lifted; `lifted` tells the two apart."""
        return None if self.budget is None else self.budget.cap_usd

    @property
    def lifted(self) -> bool:
        """The person lifted the cap for this session family."""
        return self.budget is not None and self.budget.cap_usd is None


@dataclass
class Header:
    stage: Stage = Stage.DESIGN
    work_done: int = 0
    """Tasks the run has settled (committed, dropped or re-cut), of those delegated."""
    work_delegated: int = 0
    """Tasks delegated so far, less the re-cut ones. Not the plan's size: a task not yet
    delegated is in no record."""
    work_planned: int | None = None
    """How many tasks the approved plan lists, for "WORK 2/3" (`plan_written.tasks`); one for
    Zeus's one-task plan. None while no plan says, as a plan given as
    text, or a run recorded before the field: unknown, never "1/1" between two tasks."""
    spend: HeaderSpend = field(default_factory=HeaderSpend)
    workspace: WorkspaceRecord | None = None
    """The latest turn's workspace (worktree, branch, handoff), as the session records it."""


# -- Zeus ------------------------------------------------------------------------------------


class ZeusKind(StrEnum):
    YOU = "you"
    ZEUS = "zeus"
    """An assistant turn: his words, and the calls he made in it."""
    RESULT = "result"
    HARNESS = "harness"
    MARKER = "marker"
    """A line between turns: a question or stop answered, an episode, a cap, an undo, a redo,
    a fork, an action approved by a grant."""
    ERROR = "error"
    """A turn that ended on an error (the `error` progress frame): it may write no record."""
    UNREADABLE = "unreadable"
    OUTPUT = "output"
    """What this screen said, not a record: a command's lines, a question as it is asked,
    why a line was refused. The REPL's own lines (`commands`, `interaction`), as it prints
    them."""


@dataclass(frozen=True)
class ZeusEntry:
    seq: int | None
    """The record's seq; None for an error or an output, which no record carries."""
    kind: ZeusKind
    record: WireModel | None = None
    text: str | None = None
    """An error's reason, an unreadable record's kind, or an output's line (styled as the
    REPL styles it for a terminal)."""
    stop: str | None = None
    """For a stop's answer, which stop (`design` or `plan`): "design approved" is a marker."""
    frame: RecordFrame | None = None
    """The record itself, which the REPL's line renderer (`render/stream.py`) draws."""


@dataclass
class Approval:
    """An action someone was asked to approve, or a standing grant approved, and how it ended:
    what its card's tick says. Never who answered: the tick says how it ended."""

    asker: str
    """`zeus`, `athena` or `worker`."""
    command: str
    """What runs: the approval's `command`, else its `display` (all a grant's line has)."""
    call_id: str | None = None
    """Zeus's call it is about, when it is one (`{session}:{call id}`, `zeus/calls.py`'s
    operation id); None for another role's, whose card stands on its own."""
    task_id: str | None = None
    choice: str | None = None
    """`approve`, `deny` or `abort` as answered; None while it is open."""
    scope: str = "once"
    source: str | None = None
    """How it was decided: `human`, `default`, `preference`, `grant`, `never_ask`, or
    `withdrawn` when nobody did."""


@dataclass(frozen=True)
class RungCall:
    """A rung's latest call as its `spend` line records it: the model that served it and the
    prompt it was sent, in tokens (`usage.prompt_tokens`, cached tokens included)."""

    model: str
    tokens_in: int


@dataclass
class Asked:
    """A question one of his calls put to the person (`ask_user`: the question's id is the
    call's), and its answer once given: drawn in that call's card, never as lines of their
    own (the owner's call, 2026-10-26)."""

    text: str
    default: str
    """What an answer of none proceeds on: the question's `default`."""
    choice: str | None = None
    """The answer as recorded; None while it is open."""
    note: str | None = None
    source: str | None = None

    @property
    def defaulted(self) -> bool:
        """Answered with no words of the person's: Enter, a timeout, a withdrawal."""
        return self.choice is not None and (
            self.source != "human" or (self.choice == self.default and self.note is None)
        )


@dataclass
class ZeusPane:
    entries: list[ZeusEntry] = field(default_factory=list[ZeusEntry])
    results: dict[str, ZeusEntry] = field(default_factory=dict[str, ZeusEntry])
    """Each of his calls' result, by its call id: drawn in the call's card."""
    approvals: dict[str, Approval] = field(default_factory=dict[str, Approval])
    """Each action asked about or approved by a grant, by its action id."""
    questions: dict[str, Asked] = field(default_factory=dict[str, Asked])
    """Each question one of his calls asked, by the call's id."""
    calls: dict[str, Call] = field(default_factory=dict[str, Call])
    """The calls his turns made, by id: what a question, an agent or a job was asked with."""
    cards: int = 0
    """Moves when a result or an answer lands, so the pane refills its cards then, and never
    on a streamed token."""
    pending: list[str] = field(default_factory=list[str])
    """Words the person sent that no `turn` record holds yet: queued, shown dimmed."""
    busy: str | None = None
    """The latest progress kind of a turn in flight; a hint only."""
    cost_usd: float | None = None
    """His spend so far, summed from `spend` lines; None until one lands (absent is unknown,
    never $0)."""
    call: RungCall | None = None
    """His latest call, from the `spend` line naming his role (wire 4.9): the model that
    served it and its prompt (`TuiState.role_model`, `TuiState.context`). The scout and
    `residue` charge the session log too and may share his rung, so a line naming another
    role, or none (written before 4.9), is never read as his: unknown, not guessed."""
    hidden: frozenset[int] = frozenset()
    """The seqs the standing rewinds hide (`/hidden`), as the engine reports them."""

    def visible(self) -> list[ZeusEntry]:
        """The transcript less what an undo hides; an error or an output has no seq and is
        never hidden."""
        return [e for e in self.entries if e.seq is None or e.seq not in self.hidden]


# -- Athena ----------------------------------------------------------------------------------


class AthenaKind(StrEnum):
    CALL = "call"
    MODEL = "model"
    REVIEW = "review"
    STEP = "step"
    """A step of her own (no task), as the worker's are: an edit or a command she ran."""
    EXEC = "exec"
    HOOK = "hook"
    """A hook that fired for her: its verdict, error or bound reached."""
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class AthenaEntry:
    seq: int
    kind: AthenaKind
    record: WireModel | None = None
    text: str | None = None
    """An unreadable record's kind."""
    step: int | None = None
    """A step's number in the run, as `WorkerEntry.step`."""


class PlanState(StrEnum):
    APPROVED = "approved"
    REVISED = "revised"
    """A later plan, shown and never asked."""
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CarriedTask:
    """A task of the plan an earlier run of the work settled, in the board's words:
    `committed`, `kept` (in place), `dropped` or `re-cut`."""

    task_id: str
    title: str
    state: str
    run: str | None
    """The run that settled it."""


@dataclass
class AthenaPane:
    run_id: str | None = None
    board: RunView | None = None
    plan_version: int | None = None
    plan_state: PlanState | None = None
    plan_by: str | None = None
    """Who wrote the plan: `athena`, or `zeus` for a one-task plan, whose run
    skips the design and the planning."""
    tail: deque[AthenaEntry] = field(default_factory=lambda: deque(maxlen=ATHENA_TAIL))
    cost_usd: float | None = None
    """Her spend so far, summed from her `model_call`s; None until one lands."""
    rung: str | None = None
    """The rung her calls ran on, as each `model_call` names it."""
    model: str | None = None
    """The catalog model that served her latest call (`model_call.model`),
    never the configured one. None on a run recorded before the field: the
    pane title shows the rung."""
    tokens_in: int | None = None
    """The prompt her latest call was sent, in tokens (`model_call.tokens_in`)."""
    plan_tasks: tuple[PlanTaskP, ...] = ()
    """The tasks the approved plan lists (`plan_written.tasks`): their ids and titles."""
    carried: tuple[CarriedTask, ...] = ()
    """The plan's tasks an earlier run of the work settled, which this run's board does not
    hold: a run that continues a work carries them, and its journal never says so."""


# -- the worker ------------------------------------------------------------------------------


class WorkerKind(StrEnum):
    CALL = "call"
    STEP = "step"
    EXEC = "exec"
    HOOK = "hook"
    """A hook that fired for the task's worker: its verdict, error or bound."""
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class WorkerEntry:
    seq: int
    kind: WorkerKind
    record: WireModel | None = None
    step: int | None = None
    """A step's number in the run, for `⟲ step N`."""
    text: str | None = None
    """An unreadable record's kind."""


@dataclass(frozen=True)
class FactsLine:
    """The pinned facts block, from a round's facts: what ran, never that it is right."""

    files: int
    added: int
    removed: int
    tests: str | None
    """The last test command's outcome as the harness read it; None when no test ran."""
    failing: tuple[str, ...]
    syntax_ok: bool | None
    """None when no file's syntax was checked."""


def facts_line(facts: TaskFactsP) -> FactsLine:
    """A round's facts as the block says them: the latest round's, pinned in the worker's
    pane, and each round's, in a task's details."""
    tests = [c for c in facts.commands if c.test_outcome is not None]
    return FactsLine(
        files=len(facts.files),
        added=sum(f.added for f in facts.files),
        removed=sum(f.removed for f in facts.files),
        tests=tests[-1].test_outcome if tests else None,
        failing=tuple(tests[-1].failing) if tests else (),
        syntax_ok=all(s.ok for s in facts.syntax) if facts.syntax else None,
    )


@dataclass
class WorkerTask:
    task_id: str
    title: str = ""
    brief_sha: str | None = None
    """The brief it was delegated with, a run blob (`task_delegated.brief_sha`)."""
    round: int = 0
    delegated_at: str | None = None
    last_at: str | None = None
    """When the task's latest record was written: its time so far is read from the record,
    never the wall clock, so a replay and a skewed clock both read true."""
    rung: str | None = None
    model: str | None = None
    """The catalog model that served the worker's latest call, as `AthenaPane.model`."""
    tokens_in: int | None = None
    """The prompt the worker's latest call on this task was sent, in tokens."""
    cost_usd: float | None = None
    """The task's spend so far, from its `model_call`s; None until one lands."""
    entries: list[WorkerEntry] = field(default_factory=list[WorkerEntry])
    facts: TaskFactsP | None = None
    """The latest reported round's facts, pinned."""
    facts_round: int | None = None
    """Which round `facts` measured: a later round may be under way before it reports."""
    reports: list[TaskReportedP] = field(default_factory=list[TaskReportedP])
    """Each round's report, in order: how it stopped and what it measured (the `Enter` on a
    task); its words are a run blob (`report_sha`), read when shown."""
    reviews: list[TaskReviewedP] = field(default_factory=list[TaskReviewedP])
    """Athena's answer to each report, in order."""

    @property
    def facts_line(self) -> FactsLine | None:
        return None if self.facts is None else facts_line(self.facts)

    def last_step(self, path: str) -> int | None:
        """The last step that wrote `path` (`⟲ step N`), by its effect's observed surface; None
        when no step of this task is known to have."""
        for entry in reversed(self.entries):
            if (
                entry.step is not None
                and isinstance(entry.record, EffectCompletedP)
                and path in entry.record.observed_surface
            ):
                return entry.step
        return None


@dataclass
class WorkerPane:
    tasks: dict[str, WorkerTask] = field(default_factory=dict[str, WorkerTask])
    current: str | None = None
    """The task shown: the latest delegated (one task at a time)."""
    end: RunEndP | None = None
    outcome: CoderOutcome | None = None
    """The `coder` call's result for the run on screen, as Zeus's `tool` line types it: its
    terminal and why (a run that failed is a report, `BLOCKED` with its reason)."""

    @property
    def task(self) -> WorkerTask | None:
        return None if self.current is None else self.tasks.get(self.current)


# -- the diff (layout E's centre, and the Ctrl+D screen) ------------------------------------


class DiffScope(StrEnum):
    """What one diff view shows. The
    centre pane is the view at `TASK`; the `Ctrl+D` screen switches between the two."""

    TASK = "task"
    ALL = "all"
    """The run's tasks, each by `render/diff.py`'s rule (committed files, else its last
    round's): Part 6 builds it from the board, reusing that rule."""


@dataclass(frozen=True)
class DiffFile:
    path: str
    added: int | None = None
    removed: int | None = None
    """None while nothing measured it: no diff read yet, and no round's facts."""
    kind: str | None = None


@dataclass(frozen=True)
class DiffView:
    """A task's files, their lines, and its last failing test, derived from the worker pane and
    the task's diff as the server read it (`RunDiff`), never kept beside
    them. With no diff read (a server that does not build it), a file is its name and the
    round's counts."""

    scope: DiffScope = DiffScope.TASK
    task_id: str | None = None
    files: tuple[DiffFile, ...] = ()
    failing: CommandFactP | None = None
    steps: tuple[int, ...] = ()
    text: str | None = None
    """The task's unified diff, start to its latest step; None when none was read."""

    @property
    def lines(self) -> tuple[int, int] | None:
        """The files' added and removed lines together; None while any file's are unknown."""
        if not self.files or any(f.added is None or f.removed is None for f in self.files):
            return None
        return sum(f.added or 0 for f in self.files), sum(f.removed or 0 for f in self.files)


# -- the background -----------------------------------------------------------------


@dataclass
class AgentItem:
    """One spawned agent, folded from its `agent` records: running from a round's
    `agent_start` until its `agent_report`."""

    agent: str
    """Its id: `A1`."""
    profile: str
    call_id: str
    """The `spawn_agent` call that started it: what it was asked is that call's arguments."""
    running: bool = True
    calls: int = 0
    """Tool messages in its current round: the calls it has made."""
    status: str | None = None
    """Its last report's: `complete`, `partial`, `failed` or `cancelled`."""
    findings: int | None = None
    cost_usd: float | None = None


@dataclass
class BackgroundPane:
    """Zeus's work running past his turn: agents from the session's records, jobs from
    `/jobs`. The pane shows while one runs (`running`, the one rule)."""

    agents: dict[str, AgentItem] = field(default_factory=dict[str, AgentItem])
    jobs: tuple[JobView, ...] = ()
    """As `/jobs` last answered."""
    gone: set[str] = field(default_factory=set[str])
    """Items from an earlier burst: once nothing runs, what ran then is not drawn again."""

    @property
    def running(self) -> bool:
        return any(a.running for a in self.agents.values()) or any(j.running for j in self.jobs)

    @property
    def items(self) -> list[str]:
        """The rows, by id (`A1`, `job:<id>`), agents first, each kind as it started."""
        ids = [*self.agents, *(f"job:{j.job_id}" for j in self.jobs)]
        return [i for i in ids if i not in self.gone]

    def settle(self) -> None:
        """Called after each change: once nothing runs, every item so far is put away."""
        if not self.running:
            self.gone.update([*self.agents, *(f"job:{j.job_id}" for j in self.jobs)])


def call_of(zeus: ZeusPane, operation_id: str) -> Call | None:
    """His call an operation id names (`{session}:{call id}`, a step's or a job's)."""
    _scope, _sep, call_id = operation_id.rpartition(":")
    return zeus.calls.get(call_id)


# -- the whole screen ------------------------------------------------------------------------


def drawn_as_modal(asked: OpenQuestion) -> bool:
    """A stop and an action approval are drawn as modals over the screen
    ; a question Zeus asks, and an older log's design gate, are
     answered in his pane. One rule, read by the screen and by the
     REPL's question renderer it gives it to (`Interaction(draws=)`)."""
    return asked.stop is not None or asked.approval is not None


def drawn_in_card(asked: OpenQuestion, zeus: ZeusPane) -> bool:
    """A question one of his calls asks (`ask_user`, its id the call's) is drawn in that
    call's card in his pane: the question, then its answer. The REPL's question renderer
    prints it no lines (`Interaction(draws=)`)."""
    return asked.question is not None and asked.question.id in zeus.calls


@dataclass
class TuiState:
    header: Header = field(default_factory=Header)
    zeus: ZeusPane = field(default_factory=ZeusPane)
    athena: AthenaPane = field(default_factory=AthenaPane)
    worker: WorkerPane = field(default_factory=WorkerPane)
    background: BackgroundPane = field(default_factory=BackgroundPane)
    stops: Questions = field(default_factory=Questions)
    connection: Connection = Connection.LIVE
    info: SessionInfo | None = None
    run_header: RunHeader | None = None
    liveness: Liveness | None = None
    wiring: ModelWiring | None = None
    handoff: HandoffSummary | None = None
    """The session's handoff summary, whose `diff` is its committed change from its base; None
    until the diff screen reads it."""
    not_served: set[str] = field(default_factory=set[str])
    """Routes this server does not build (`501 NOT_BUILT`, a replay's `/handoff`): their
    answers are unknown, and a screen that needs one says it is not served here."""
    diffs: dict[str, RunDiff] = field(default_factory=dict[str, RunDiff])
    """Each task's diff, by its id, as last read (`StaleRoute.TASK_DIFF`)."""
    step_diffs: dict[str, RunDiff] = field(default_factory=dict[str, RunDiff])
    """Each step's diff, by its operation id (`StaleRoute.STEP_DIFFS`): a step never moves
    once it has landed, so each is read once."""
    stop_shown: dict[str, int] = field(default_factory=dict[str, int])
    """By stop id, how many times a stop of its kind has been shown since the last one was
    settled (approved or cancelled): a note or an edit sends the text back, and its author's
    next stop is the same work shown again. Counted from the records,
    since the engine's `round` restarts with each new stop (`feed.SENT_BACK`)."""
    stops_put_back: set[str] = field(default_factory=set[str])
    """The stops the person put back with `Esc`: a stop opens on the
    screen as it is asked; one put back waits as the banner, the run waiting on it, while they
    talk to Zeus, until `^S` opens it again. An id stays once its stop is settled: a settled
    stop is never open again, so its id is never read."""
    question_timeout_s: float | None = None
    """How long this server waits for an answer (`Hello`); None when it does
    not say."""
    asked_at: dict[int, str] = field(default_factory=dict[int, str])
    """When each question was asked, by its record's seq: its deadline's start."""
    live: Replies = field(default_factory=Replies)
    """The replies being written, any role's, folded from the live progress frames (W7 and W10):
    the REPL's own fold (`render/live.Replies`). A hint: each reply's record
    replaces it when it lands."""
    retrying: str | None = None
    """While `connection` is `RECONNECTING`: the try and why, as the stream's watch says it.
    """
    lost: dict[str, str] = field(default_factory=dict[str, str])
    """What could not be read, and why: a stream that ended on an error, a route that failed.
    Shown, never swallowed; a route the server does not build is not here."""

    @property
    def asking(self) -> OpenQuestion | None:
        """The question a modal shows now: the oldest open one, when it is drawn as a modal.
        One at a time, oldest first, as the REPL answers them (`Interaction.current`): a modal
        waits behind an older question asked in Zeus's pane. A stop the person put back waits
        as a banner (`waiting_stop`), and a question after it is not held behind it."""
        for open_now in self.stops.open:
            if open_now.stop is not None and open_now.id in self.stops_put_back:
                continue
            return open_now if drawn_as_modal(open_now) else None
        return None

    @property
    def waiting_stop(self) -> OpenQuestion | None:
        """The oldest stop the person put back, waiting for them: the banner's."""
        return next(
            (q for q in self.stops.open if q.stop is not None and q.id in self.stops_put_back),
            None,
        )

    def deadline(self, asked: OpenQuestion) -> str | None:
        """When `asked` answers itself if no one does, in local time (`HH:MM`): its record's
        time plus the server's timeout. None when either is unknown."""
        at = self.asked_at.get(asked.seq)
        if at is None or self.question_timeout_s is None:
            return None
        try:
            start = datetime.fromisoformat(at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        return (
            (start + timedelta(seconds=self.question_timeout_s)).astimezone().strftime("%H:%M")
        )

    def wired(self, role: str) -> tuple[str, str | None] | None:
        """The rung `role` runs on and the catalog model the wiring puts there, by the
        engine config's assignment (`ModelWiring.roles`); None before the
        wiring is read, or for a role it does not assign."""
        wiring = self.wiring
        if wiring is None or not wiring.roles or role not in wiring.roles:
            return None
        rung = wiring.roles[role]
        return rung, wiring.rungs.get(rung)

    def _latest(self, role: str) -> tuple[str | None, str | None, int | None]:
        """`role`'s latest call as the records say it: its rung, the model that served it,
        and its prompt's tokens."""
        if role == "zeus":
            wired = self.wired("zeus")
            call = self.zeus.call
            rung = wired[0] if wired is not None else None
            return rung, call.model if call else None, call.tokens_in if call else None
        if role == "athena":
            a = self.athena
            return a.rung, a.model, a.tokens_in
        task = self.worker.task
        if task is None:
            return None, None, None
        return task.rung, task.model, task.tokens_in

    def role_model(self, role: str) -> str | None:
        """The model `role` runs on: the one that served its latest call, else the one the
        wiring puts on its rung; None while neither is known (never guessed)."""
        served = self._latest(role)[1]
        if served is None and role == "worker":
            # Between tasks: the last task that was served.
            tasks = reversed(self.worker.tasks.values())
            served = next((t.model for t in tasks if t.model), None)
        if served is not None:
            return served
        wired = self.wired(role)
        return wired[1] if wired is not None else None

    def model_name(self, catalog: str) -> str:
        """The provider's name for catalog model `catalog` (`ModelWiring.models`'s `model_id`,
        wire 4.11: `claude-opus-4-1` for `frontier`), what a person knows the model by; the
        catalog name where the catalog does not say (an older server, an unknown name)."""
        models = self.wiring.models if self.wiring is not None else None
        known = models.get(catalog) if models else None
        return known.model_id if known is not None and known.model_id else catalog

    def context(self, role: str) -> tuple[int, int | None] | None:
        """How full `role`'s latest prompt was: its tokens and the window of the model that
        served it (`ModelWiring.models`); a record from before calls named their model reads its
        rung's model. The window is None where the catalog does not name that model: another
        model's window would be a guess. None before a call of the role's lands."""
        rung, served, tokens = self._latest(role)
        if tokens is None:
            return None
        wiring = self.wiring
        name = (
            served
            if served is not None
            else (wiring.rungs.get(rung) if wiring and rung else None)
        )
        models = wiring.models if wiring is not None else None
        window = models[name].context_window if models and name in models else None
        return tokens, window

    @property
    def landing(self) -> bool:
        """Nothing said and no run yet: the screen as it first
        opens, until the person's first words; a session with turns opens on its state."""
        said = any(e.kind in (ZeusKind.YOU, ZeusKind.ZEUS) for e in self.zeus.entries)
        return not said and not self.zeus.pending and self.athena.run_id is None

    @property
    def episode_open(self) -> bool:
        """A coder call's run is on screen and has not ended."""
        return self.athena.run_id is not None and self.worker.end is None

    @property
    def holding_for(self) -> str | None:
        """Whom Zeus waits on while he is silent ("holding until Athena reports"): the box
        reports to him through Athena, so while its run is open and he has no turn in
        flight, it is her."""
        if not self.episode_open or self.zeus.busy is not None:
            return None
        return f"the {ROLE_WORDS['athena'].lower()}"

    @property
    def task_diff(self) -> DiffView:
        """The diff at `TASK` scope: the task the worker pane shows."""
        current = self.worker.current
        return DiffView() if current is None else self.diff_of(current)

    def diff_of(self, task_id: str) -> DiffView:
        """Task `task_id`'s diff: as the server read it, else its latest round's facts, else
        the paths its steps touched; empty for a task the worker has not started."""
        task = self.worker.tasks.get(task_id)
        if task is None:
            return DiffView()
        steps = tuple(e.step for e in task.entries if e.step is not None)
        read = self.diffs.get(task.task_id)
        if read is not None:
            files = tuple(DiffFile(f.path, f.added, f.removed, f.kind) for f in read.files)
            commands = task.facts.commands if task.facts is not None else []
            failing = next((c for c in reversed(commands) if c.failing), None)
            return DiffView(DiffScope.TASK, task.task_id, files, failing, steps, read.text)
        if task.facts is not None:
            files = tuple(
                DiffFile(f.path, f.added, f.removed, f.kind) for f in task.facts.files
            )
            failing = next((c for c in reversed(task.facts.commands) if c.failing), None)
            return DiffView(DiffScope.TASK, task.task_id, files, failing, steps)
        touched: dict[str, None] = {}
        for entry in task.entries:
            if isinstance(entry.record, EffectCompletedP):
                touched.update(dict.fromkeys(entry.record.observed_surface))
        files = tuple(DiffFile(p) for p in touched)
        return DiffView(DiffScope.TASK, task.task_id, files, None, steps)
