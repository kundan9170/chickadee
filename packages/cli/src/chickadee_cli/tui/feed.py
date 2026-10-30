"""The TUI's one subscriber: every frame and every route answer
the screen receives, folded into `model.TuiState`.

**A fold of the wire, and nothing else**. Each record is read as its kind's type
(`chickadee_wire.records.payload`); one this client cannot read is an `unreadable` entry in the
pane it would have fed, never an error. A run's board is `render/board.py`'s `RunView`,
kept by `apply_record`, the fold the REPL's `/status` prints: one fold, never a second.

**Records are the truth, progress a hint**. A progress frame sets only Zeus's
`busy`; drop every one and the state is the same less that. **A record is folded once**: the
feed keeps the highest seq it has folded in each scope and drops any record at or below it, so
a catch-up read and a live stream may overlap.

**Routes are named, never called.** A record that makes a route's answer stale (an undo makes
`/hidden` stale, a charge `/budget`) is reported in `Changed.refresh`; whoever holds the client
reads the route and hands the answer to `snapshot`. The feed does no I/O.

**Not thread-safe.** One reader owns a `Feed`: the screen's event loop. A stream read on another
thread hands its frames to that loop rather than calling `apply` itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, assert_never

from pydantic import ValidationError

from chickadee_cli.render.board import COMMITTED, KEPT, RunView, apply_record
from chickadee_cli.render.stream import LIVE_PROGRESS, STEP_PREFIX
from chickadee_cli.tui.model import (
    AgentItem,
    Approval,
    Asked,
    AthenaEntry,
    AthenaKind,
    AthenaPane,
    CarriedTask,
    Connection,
    Pane,
    PlanState,
    RungCall,
    SpendLevel,
    Stage,
    StaleRoute,
    TuiState,
    WorkerEntry,
    WorkerKind,
    WorkerPane,
    WorkerTask,
    ZeusEntry,
    ZeusKind,
)
from chickadee_serve.frames import Hello, Hidden, ProgressFrame, RecordFrame
from chickadee_wire.events import Frame, Link, Reconnected, Retrying
from chickadee_wire.models import (
    ActionApproval,
    ActionGrantedRecord,
    AgentMessageP,
    AgentRecord,
    AgentReportP,
    AgentStartP,
    AssistantTurnP,
    BudgetP,
    CapRecord,
    EffectCompletedP,
    EpisodeRecord,
    ExecP,
    ForkRecord,
    HandoffSummary,
    HarnessTurnP,
    HookP,
    JobView,
    Liveness,
    ModelCallP,
    ModelSwitchedRecord,
    ModelWiring,
    PlanWrittenP,
    QuestionAnsweredRecord,
    QuestionAskedRecord,
    RedoRecord,
    RunDiff,
    RunEndP,
    RunHeader,
    RunStartP,
    SessionBudget,
    SessionInfo,
    ShellEffectRecord,
    SpendRecord,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
    TitleRecord,
    ToolCallP,
    ToolRecord,
    UndoRecord,
    UserTurnP,
    WireModel,
    WorkProgress,
    WorkspaceRecord,
)
from chickadee_wire.records import is_run_record, payload

__all__ = [
    "CARRIED_WORDS",
    "SENT_BACK",
    "SETTLED",
    "Changed",
    "Feed",
    "Jobs",
    "NotServed",
    "Recovered",
    "Snapshot",
    "Work",
]

SETTLED = frozenset({COMMITTED, KEPT, "dropped", "re-cut"})
"""A task the run has done with, in the board's words (`render/board.py`): `accepted` is not
yet, since its commit has not landed, nor, in place, its known-good record (`kept`)."""

_QUESTION_KINDS = frozenset({"question_asked", "question_answered"})

_MARKERS = (
    QuestionAnsweredRecord,
    EpisodeRecord,
    CapRecord,
    UndoRecord,
    RedoRecord,
    ForkRecord,
    ActionGrantedRecord,
)
"""The session records drawn as a line between turns, as the REPL's `normal` stream shows them
(`render/stream.py`)."""


@dataclass(frozen=True)
class NotServed:
    """A route this server does not build (`501 NOT_BUILT`): its answer stays unknown."""

    route: str


@dataclass(frozen=True)
class Jobs:
    """`/jobs`' answer: the session's jobs, running and ended."""

    jobs: tuple[JobView, ...]


@dataclass(frozen=True)
class Recovered:
    """A route whose read failed answered again: it is no longer said lost."""

    route: str


@dataclass(frozen=True)
class Work:
    """`/session/{sid}/work`'s answer: where the open work stands; None with none."""

    progress: WorkProgress | None


CARRIED_WORDS: Final = {
    "committed": COMMITTED,
    "unchecked": KEPT,
    "dropped": "dropped",
    "replaced": "re-cut",
}
"""Each state the work's fold settles a task in (`work_progress.DONE`), in the board's words:
`unchecked` is a task accepted in place, which the board calls `kept`. A task in any
other state is still to do, and waits as the plan lists it."""

type Snapshot = (
    SessionBudget | Hidden | Liveness | RunHeader | SessionInfo | ModelWiring | HandoffSummary
    | Hello | RunDiff | Jobs | Work | Recovered | NotServed
)  # fmt: skip
"""A route's answer the screen holds: `/budget`, `/hidden`, `/liveness`, `/header`, the
session's info, `/rung`, `/handoff`, `/jobs` and the server's own `GET /`; or that the server
does not build the route."""


@dataclass(frozen=True)
class Changed:
    """What a fold moved: the panes to redraw, the routes to read again."""

    panes: frozenset[Pane] = frozenset()
    refresh: frozenset[StaleRoute] = frozenset()

    def __or__(self, other: Changed) -> Changed:
        return Changed(self.panes | other.panes, self.refresh | other.refresh)

    def __bool__(self) -> bool:
        return bool(self.panes or self.refresh)


def _moved(*panes: Pane, refresh: tuple[StaleRoute, ...] = ()) -> Changed:
    return Changed(frozenset(panes), frozenset(refresh))


NOTHING = Changed()

_DIFF: Final = (StaleRoute.TASK_DIFF,)
BOUND: Final = "bound"
"""A `hook` record's decision for the set an episode bound at its start,
which is no firing: not drawn."""
"""A task's steps or reports moved: the shown task's diff is read again."""


SENT_BACK: Final = frozenset({"note", "edit"})
"""A stop's answers that send its text back to its author, who answers with a new stop: the
same work shown again. Any other answer (approve, cancel, a timeout's approve, a withdrawal)
settles it, and the next stop of its kind is shown for the first time."""


class Feed:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.state = TuiState()
        self._seen: dict[str, int] = {}
        """The highest record seq folded, per scope."""
        self._run: RunView | None = None
        self._steps = 0
        """Completed steps in the run on screen, so each has its `⟲ step N`."""
        self._stop_kinds: dict[str, str] = {}
        """Each stop's kind by its id: an answer names the id only, and the open-question fold
        forgets a stop once it is answered."""
        self._stop_streak: dict[str, int] = {}
        """Per stop kind, how many times one has been shown since the last was settled."""
        self._sent_back: dict[str, bool] = {}
        """Per stop kind, whether the last one's answer sent it back to its author (a note or
        an edit), so the next is the same work shown again."""

    # -- the three ways in -----------------------------------------------------------------

    def apply(self, frame: Frame) -> Changed:
        """Fold one frame of the session's stream or of a run's."""
        if isinstance(frame, ProgressFrame):
            return self._progress(frame)
        last = self._seen.get(frame.scope, 0)
        if frame.seq <= last:
            return NOTHING
        self._seen[frame.scope] = frame.seq
        if is_run_record(frame):
            changed = self._run_record(frame)
        elif frame.scope == self.session_id:
            changed = self._session_record(frame)
        else:
            return NOTHING
        return changed | self._restage()

    def snapshot(self, value: Snapshot) -> Changed:
        """Hold a route's answer."""
        state = self.state
        match value:
            case SessionBudget():
                spend = state.header.spend
                spend.budget = value
                # The engine's answer is the level now: a `/budget` that raised or lifted the
                # cap takes the header out of red, and its banner goes.
                spend.level = _level_of(value)
                if spend.level is not SpendLevel.CAP:
                    spend.capped = None
                return _moved(Pane.HEADER)
            case Hidden():
                state.zeus.hidden = frozenset(value.seqs)
                return _moved(Pane.ZEUS)
            case Liveness():
                return self._liveness(value)
            case RunHeader():
                state.run_header = value
                return _moved(Pane.HEADER)
            case SessionInfo():
                state.info = value
                return _moved(Pane.HEADER)
            case ModelWiring():
                state.wiring = value
                # The header names each role's model, and each pane's context reads its window.
                return _moved(Pane.HEADER, Pane.ZEUS, Pane.ATHENA, Pane.WORKER)
            case HandoffSummary():
                state.handoff = value
                return _moved(Pane.DIFF)
            case Jobs():
                state.background.jobs = value.jobs
                state.background.settle()
                return _moved(Pane.BACKGROUND)
            case Work():
                return self._carried(value.progress) | self._restage()
            case Recovered():
                return _moved(Pane.HEADER) if state.lost.pop(value.route, None) else NOTHING
            case NotServed():
                state.not_served.add(value.route)
                return _moved(Pane.DIFF)
            case Hello():
                state.question_timeout_s = value.question_timeout_s
                return _moved(Pane.STOPS)
            case RunDiff():
                return self._run_diff(value)
            case _:
                assert_never(value)

    def _run_diff(self, value: RunDiff) -> Changed:
        state = self.state
        if value.run_id != state.athena.run_id:
            return NOTHING  # an older run's, answered late
        if value.operation_id is not None:
            # A step's: its card's line counts, never the task's diff.
            state.step_diffs[value.operation_id] = value
            return _moved(Pane.ATHENA, Pane.WORKER)
        if value.task_id is None:
            return NOTHING
        state.diffs[value.task_id] = value
        return _moved(Pane.DIFF, Pane.WORKER)

    def lost(self, what: str, why: str, *, stream: bool) -> Changed:
        """`what` could not be read, and why. A stream's loss is the connection's: the screen
        says it is lost; a route's leaves its answer as it was."""
        self.state.lost[what] = why
        if stream:
            self.state.connection = Connection.LOST
        return _moved(Pane.HEADER)

    def link(self, link: Link) -> Changed:
        """The session stream's connection, as its watch says it: retrying
        is "reconnecting"; live again clears it. A lost wire stays lost."""
        state = self.state
        if state.connection is Connection.LOST:
            return NOTHING
        match link:
            case Retrying():
                state.connection = Connection.RECONNECTING
                state.retrying = f"try {link.failures}: {link.why}"
            case Reconnected():
                state.connection, state.retrying = Connection.LIVE, None
        return _moved(Pane.HEADER)

    def note(self, lines: Sequence[str]) -> Changed:
        """Lines this screen says in Zeus's pane, as the REPL prints them: a command's output,
        a question as it opens, why a line was refused. Not records: they carry no seq."""
        self.state.zeus.entries += [
            ZeusEntry(None, ZeusKind.OUTPUT, text=line) for line in lines
        ]
        return _moved(Pane.ZEUS)

    def unsent(self, text: str) -> Changed:
        """Words the server refused: no longer pending (the refusal is said by `note`)."""
        if text in self.state.zeus.pending:
            self.state.zeus.pending.remove(text)
        return _moved(Pane.ZEUS)

    def sent(self, text: str) -> Changed:
        """The person's words, sent: pending until a `turn` record holds them."""
        self.state.zeus.pending.append(text)
        return _moved(Pane.ZEUS)

    # -- progress ----------------------------------------------------------------------------

    def _progress(self, frame: ProgressFrame) -> Changed:
        zeus = self.state.zeus
        if frame.scope != self.session_id or frame.kind in ("alive", "context"):
            return NOTHING
        if frame.kind in LIVE_PROGRESS:
            return self._live(frame)
        if frame.kind == "error":
            for key in list(self.state.live.open):
                self.state.live.end(key)  # the turn ended on an error: nothing more streams
            # A turn that fails before replying writes no record: this may be all there is.
            text = frame.payload.get("text")
            zeus.entries.append(
                ZeusEntry(None, ZeusKind.ERROR, text=str(text) if text else None)
            )
            zeus.busy = None
            return _moved(Pane.ZEUS)
        if frame.kind == "answer":
            # His loop yields `answer` after it records the reply (`zeus/loop.py`), so it lands
            # after the record: the reply is in, and he is not still answering. Anything more
            # this turn sends sets him busy again.
            zeus.busy = None
            return _moved(Pane.ZEUS)
        zeus.busy = frame.kind
        return _moved(Pane.ZEUS)

    def _live(self, frame: ProgressFrame) -> Changed:
        """A reply as it streams: folded into `TuiState.live`, drawn by its role's pane.
        Only Zeus's own sets his `busy`: while the box writes, he waits on it."""
        got = self.state.live.take(frame)
        if got is None:
            return NOTHING
        reply, _moved_in = got
        if reply.role == "zeus":
            self.state.zeus.busy = frame.kind
            return _moved(Pane.ZEUS)
        return _moved(Pane.ATHENA if reply.role == "athena" else Pane.WORKER)

    # -- the session's records ---------------------------------------------------------------

    def _session_record(self, frame: RecordFrame) -> Changed:
        zeus = self.state.zeus
        changed = _moved(Pane.ZEUS) if zeus.busy is not None else NOTHING
        zeus.busy = None
        if frame.kind in _QUESTION_KINDS:
            changed |= self._question(frame)
        try:
            typed = payload(frame)
        except ValidationError:
            zeus.entries.append(
                ZeusEntry(frame.seq, ZeusKind.UNREADABLE, text=frame.kind, frame=frame)
            )
            return changed | _moved(Pane.ZEUS)
        if isinstance(typed, TitleRecord):
            # The session's name: the header reads its info again.
            return changed | _moved(Pane.HEADER, refresh=(StaleRoute.INFO,))
        return changed | self._session(frame, typed)

    def _question(self, frame: RecordFrame) -> Changed:
        try:
            moved = NOTHING
            if frame.kind == "question_asked":
                self.state.asked_at[frame.seq] = frame.at
                asked = QuestionAskedRecord.model_validate(frame.payload)
                if asked.approval is not None:
                    self._approval_asked(frame.scope, asked.approval)
                    moved = _moved(Pane.ZEUS)
                question = asked.question
                if question is not None and question.id in self.state.zeus.calls:
                    # His call's question: its card shows it, and then its answer.
                    self.state.zeus.questions[question.id] = Asked(
                        question.text, question.default
                    )
                    self.state.zeus.cards += 1
                    moved = _moved(Pane.ZEUS)
                stop = asked.stop
                if stop is not None:
                    self._stop_kinds[stop.id] = stop.kind
                    again = self._sent_back.get(stop.kind, False)
                    shown = self._stop_streak.get(stop.kind, 0) + 1 if again else 1
                    self._stop_streak[stop.kind] = shown
                    # The engine's own round counts an edit inside one asking; never less.
                    self.state.stop_shown[stop.id] = max(shown, stop.round)
            elif frame.kind == "question_answered":
                answer = QuestionAnsweredRecord.model_validate(frame.payload)
                said = self.state.zeus.questions.get(answer.id)
                if said is not None:
                    said.choice, said.note, said.source = (
                        answer.choice,
                        answer.note,
                        answer.source,
                    )
                    self.state.zeus.cards += 1
                    moved = _moved(Pane.ZEUS)
                approval = self.state.zeus.approvals.get(answer.id)
                if approval is not None:
                    approval.choice, approval.scope = answer.choice, answer.scope
                    approval.source = answer.source
                    self.state.zeus.cards += 1
                    moved = _moved(Pane.ZEUS)
                kind = self._stop_kinds.get(answer.id)
                if kind is not None:
                    self._sent_back[kind] = answer.choice in SENT_BACK
            self.state.stops.feed(frame)
        except ValidationError:
            return NOTHING  # the record itself is said unreadable in Zeus's pane
        return moved | _moved(Pane.STOPS, refresh=(StaleRoute.LIVENESS,))

    def _approval_asked(self, scope: str, asked: ActionApproval) -> None:
        """An action asked about, kept for its card: Zeus's calls carry the operation
        id `{session}:{call id}` (`zeus/calls.py`), so his approval names the call it is
        about; any other stands on its own."""
        zeus = self.state.zeus
        call_id = None
        if asked.asker == "zeus" and asked.operation_id.startswith(f"{scope}:"):
            call_id = asked.operation_id.removeprefix(f"{scope}:")
        zeus.approvals[asked.action_id] = Approval(
            asker=asked.asker,
            command=asked.command or asked.display,
            call_id=call_id,
            task_id=asked.task_id,
        )
        zeus.cards += 1

    def _session(self, frame: RecordFrame, typed: WireModel | None) -> Changed:
        state, zeus, seq = self.state, self.state.zeus, frame.seq
        match typed:
            case UserTurnP():
                if typed.text in zeus.pending:
                    zeus.pending.remove(typed.text)
                zeus.entries.append(ZeusEntry(seq, ZeusKind.YOU, typed, frame=frame))
                if state.info is None or state.info.title is None:
                    # The title is the first turn's first line: read it once it exists.
                    return _moved(Pane.ZEUS, refresh=(StaleRoute.INFO,))
            case AssistantTurnP():
                return self._zeus_turn(frame, typed)
            case HarnessTurnP():
                zeus.entries.append(ZeusEntry(seq, ZeusKind.HARNESS, typed, frame=frame))
            case ToolRecord():
                result = ZeusEntry(seq, ZeusKind.RESULT, typed, frame=frame)
                zeus.entries.append(result)
                zeus.results[typed.call_id] = result
                zeus.cards += 1
                if (
                    typed.outcome is not None
                    and frame.run_id is not None
                    and state.athena.run_id is not None
                ):
                    # A `coder` result answers the run its episode opened One with no
                    # run (a work already complete, a stop that approved nothing) is Zeus's
                    # answer alone: never the failure of the run on screen.
                    state.worker.outcome = typed.outcome
                    return _moved(Pane.ZEUS, Pane.WORKER)
            case SpendRecord():
                # The session log's charges are all his pane's spend; only his own
                # call says his model and his prompt (wire 4.9's `role`).
                zeus.cost_usd = (zeus.cost_usd or 0.0) + typed.cost_usd
                if typed.role == "zeus":
                    zeus.call = RungCall(typed.model, typed.tokens_in)
                return _moved(Pane.ZEUS, Pane.HEADER, refresh=(StaleRoute.BUDGET,))
            case WorkspaceRecord():
                state.header.workspace = typed
                return _moved(Pane.HEADER, refresh=(StaleRoute.HEADER,))
            case ModelSwitchedRecord():
                # A rung's model moved: `/rung` is read again.
                return _moved(Pane.ATHENA, Pane.WORKER, refresh=(StaleRoute.RUNG,))
            case AgentRecord():
                return self._agent(frame)
            case ShellEffectRecord():
                return self._shell_effect(frame)
            case _ if isinstance(typed, _MARKERS):
                return self._marker(frame, typed)
            case _:
                return NOTHING
        return _moved(Pane.ZEUS)

    def _agent(self, frame: RecordFrame) -> Changed:
        """A spawned agent's record: a round starts, makes a call, or reports. The
        record's `payload` is its event's type, read here by the event, as its writer wrote
        it (`context/record_lines.py`)."""
        event, inner = frame.payload.get("event"), frame.payload.get("payload")
        try:
            return self._agent_event(event, inner)
        except ValidationError:
            # Not its event's type: unreadable, as any record this client cannot read.
            self.state.zeus.entries.append(
                ZeusEntry(frame.seq, ZeusKind.UNREADABLE, text=f"agent {event}", frame=frame)
            )
            return _moved(Pane.ZEUS)

    def _agent_event(self, event: object, inner: object) -> Changed:
        background = self.state.background
        match event:
            case "agent_start":
                start = AgentStartP.model_validate(inner)
                agent = background.agents.get(start.agent)
                if agent is None:
                    agent = AgentItem(start.agent, start.profile, start.call_id)
                    background.agents[start.agent] = agent
                agent.running, agent.calls, agent.status = True, 0, None
                background.gone.discard(start.agent)
            case "agent_message":
                message = AgentMessageP.model_validate(inner)
                agent = background.agents.get(message.agent)
                if agent is None or message.side != "tool":
                    return NOTHING
                agent.calls += 1
            case "agent_report":
                report = AgentReportP.model_validate(inner)
                agent = background.agents.get(report.agent)
                if agent is None:
                    return NOTHING
                agent.running, agent.status = False, report.status
                agent.findings = report.findings
                agent.cost_usd = (agent.cost_usd or 0.0) + report.cost_usd
                agent.calls = report.tool_calls
            case _:
                return NOTHING
        background.settle()
        return _moved(Pane.BACKGROUND)

    def _shell_effect(self, frame: RecordFrame) -> Changed:
        """One of Zeus's shell effects: a job's (`job:` operation) starting or ending
        makes `/jobs` stale; a step's changes nothing here."""
        inner = frame.payload.get("payload")
        operation = inner.get("operation_id") if isinstance(inner, dict) else None
        if isinstance(operation, str) and operation.startswith("job:"):
            return _moved(refresh=(StaleRoute.JOBS,))
        return NOTHING

    def _marker(self, frame: RecordFrame, typed: WireModel) -> Changed:
        stop = (
            self._stop_kinds.get(typed.id)
            if isinstance(typed, QuestionAnsweredRecord)
            else None
        )
        self.state.zeus.entries.append(
            ZeusEntry(frame.seq, ZeusKind.MARKER, typed, stop=stop, frame=frame)
        )
        match typed:
            case ActionGrantedRecord():
                # Approved with nobody asked: a card's tick, as an answer is.
                scope = frame.scope
                call_id = None
                if typed.operation_id.startswith(f"{scope}:"):
                    call_id = typed.operation_id.removeprefix(f"{scope}:")
                self.state.zeus.approvals[typed.action_id] = Approval(
                    asker="zeus" if call_id is not None else "",
                    command=typed.display,
                    call_id=call_id,
                    choice="approve",
                    scope=typed.scope,
                    source="never_ask" if typed.scope == "never_ask" else "grant",
                )
                self.state.zeus.cards += 1
                return _moved(Pane.ZEUS)
            case EpisodeRecord():
                return self._episode(typed.run_id)
            case CapRecord():
                return _moved(Pane.ZEUS, refresh=(StaleRoute.BUDGET,))
            case UndoRecord() | RedoRecord():
                return _moved(Pane.ZEUS, refresh=(StaleRoute.HIDDEN, StaleRoute.HEADER))
            case _:
                return _moved(Pane.ZEUS)

    def _episode(self, run_id: str) -> Changed:
        """A coder call's run: the run the right of the screen now shows."""
        state = self.state
        if state.athena.run_id == run_id:
            return _moved(Pane.ZEUS)
        self._run = RunView(run_id)
        self._steps = 0
        state.athena = AthenaPane(run_id=run_id, board=self._run)
        state.worker = WorkerPane()
        # Task and operation ids repeat across runs: an older run's diffs are not this one's.
        state.diffs, state.step_diffs = {}, {}
        return _moved(
            Pane.ZEUS,
            Pane.ATHENA,
            Pane.WORKER,
            Pane.DIFF,
            refresh=(StaleRoute.LIVENESS, StaleRoute.WORK),
        )

    # -- a run's records ---------------------------------------------------------------------

    def _run_record(self, frame: RecordFrame) -> Changed:
        if self._run is None:
            self._episode(frame.scope)  # a run read before its episode: shown all the same
        if frame.scope != self.state.athena.run_id or self._run is None:
            return NOTHING  # an older run's record; the screen shows the latest episode
        committed = _committed(self._run)
        board = _moved(Pane.ATHENA) if apply_record(self._run, frame) else NOTHING
        if _committed(self._run) > committed:
            # A task's commit landed, or in place its tree was kept: the session's committed
            # diff, and the checkout's uncommitted paths, have moved.
            board = board | _moved(Pane.DIFF, refresh=(StaleRoute.HANDOFF, StaleRoute.HEADER))
        if frame.task_id is not None:
            task = self._task(frame.task_id)
            task.last_at = frame.at
            # The envelope names the round a record belongs to: a round under way is shown
            # before it reports (the envelope's `round`).
            task.round = max(task.round, frame.round or 0)
        try:
            typed = payload(frame)
        except ValidationError:
            return board | self._unreadable(frame)
        return board | self._run_typed(frame, typed)

    def _unreadable(self, frame: RecordFrame) -> Changed:
        if frame.task_id is not None:
            task = self._task(frame.task_id)
            task.entries.append(WorkerEntry(frame.seq, WorkerKind.UNREADABLE, text=frame.kind))
            return _moved(Pane.WORKER)
        self.state.athena.tail.append(
            AthenaEntry(frame.seq, AthenaKind.UNREADABLE, text=frame.kind)
        )
        return _moved(Pane.ATHENA)

    def _run_typed(self, frame: RecordFrame, typed: WireModel | None) -> Changed:
        athena, worker, seq = self.state.athena, self.state.worker, frame.seq
        match typed:
            case RunStartP():
                return _moved(Pane.ATHENA)
            case PlanWrittenP():
                athena.plan_version, athena.plan_by = typed.version, typed.by
                athena.plan_tasks = tuple(typed.tasks or ())
                # Zeus's one-task plan lists none: it is one task.
                planned = len(athena.plan_tasks) or (1 if typed.by == "zeus" else None)
                self.state.header.work_planned = planned
                athena.plan_state = (
                    PlanState.CANCELLED
                    if typed.outcome == "cancelled"
                    else PlanState.APPROVED
                    if typed.asked
                    else PlanState.REVISED
                )
                # A plan inherited from an earlier run: the tasks that run settled.
                return _moved(Pane.ATHENA, refresh=(StaleRoute.WORK,))
            case TaskDelegatedP():
                task = self._task(typed.task_id)
                task.title, task.delegated_at = typed.title, frame.at
                task.brief_sha = typed.brief_sha
                worker.current = typed.task_id
                return _moved(Pane.WORKER, refresh=_DIFF)
            case TaskReportedP():
                task = self._task(typed.task_id)
                task.round = max(task.round, typed.round)
                task.facts, task.facts_round = typed.facts, typed.round
                task.reports.append(typed)
                return _moved(Pane.WORKER, refresh=_DIFF)
            case TaskReviewedP():
                athena.tail.append(AthenaEntry(seq, AthenaKind.REVIEW, typed))
                self._task(typed.task_id).reviews.append(typed)
                # Its commit, or in place its tree kept, follows the review.
                return _moved(Pane.ATHENA, refresh=(StaleRoute.HEADER,))
            case ToolCallP():
                return self._call(frame, typed)
            case ModelCallP():
                return self._model_call(frame, typed)
            case EffectCompletedP() if typed.operation_id.startswith(STEP_PREFIX):
                return self._step(frame, typed)
            case ExecP():
                return self._exec(frame, typed)
            case HookP() if typed.decision != BOUND:
                return self._hook(frame, typed)
            case BudgetP():
                spend = self.state.header.spend
                level = SpendLevel.CAP if typed.kind == "hard_cap" else SpendLevel.WARNING
                if spend.level is not SpendLevel.CAP:
                    spend.level = level
                if typed.kind == "hard_cap":
                    spend.capped = typed
                return _moved(Pane.HEADER, refresh=(StaleRoute.BUDGET,))
            case RunEndP():
                worker.end = typed
                return _moved(
                    Pane.ATHENA,
                    Pane.WORKER,
                    Pane.DIFF,
                    refresh=(
                        StaleRoute.BUDGET,
                        StaleRoute.LIVENESS,
                        StaleRoute.HANDOFF,
                        StaleRoute.WORK,
                        StaleRoute.HEADER,
                    ),
                )
            case _:
                return NOTHING

    def _step(self, frame: RecordFrame, typed: EffectCompletedP) -> Changed:
        """A step: the worker's, or her own (no task), a card in her pane as the worker's
        are. One that changed files has its counts read once."""
        self._steps += 1
        # A step that changed files changed the tree the header counts.
        counts = (StaleRoute.STEP_DIFFS, StaleRoute.HEADER) if typed.observed_surface else ()
        if frame.task_id is None:
            self.state.athena.tail.append(
                AthenaEntry(frame.seq, AthenaKind.STEP, typed, step=self._steps)
            )
            return _moved(Pane.ATHENA, refresh=counts)
        self._task(frame.task_id).entries.append(
            WorkerEntry(frame.seq, WorkerKind.STEP, typed, step=self._steps)
        )
        return _moved(Pane.WORKER, refresh=(*_DIFF, *counts))

    def _exec(self, frame: RecordFrame, typed: ExecP) -> Changed:
        if frame.task_id is None:
            self.state.athena.tail.append(AthenaEntry(frame.seq, AthenaKind.EXEC, typed))
            return _moved(Pane.ATHENA)
        self._task(frame.task_id).entries.append(WorkerEntry(frame.seq, WorkerKind.EXEC, typed))
        return _moved(Pane.WORKER)

    def _zeus_turn(self, frame: RecordFrame, typed: AssistantTurnP) -> Changed:
        zeus = self.state.zeus
        self.state.live.finish("zeus", None)  # his record replaces what streamed
        zeus.entries.append(ZeusEntry(frame.seq, ZeusKind.ZEUS, typed, frame=frame))
        zeus.calls.update({call.id: call for call in typed.calls})
        if typed.calls:
            return _moved(Pane.ZEUS)
        # His turn ended: what his own commands wrote is in the tree.
        return _moved(Pane.ZEUS, refresh=(StaleRoute.HEADER,))

    def _liveness(self, value: Liveness) -> Changed:
        state = self.state
        state.liveness = value
        moved = _moved(Pane.HEADER, Pane.ZEUS)
        team = value.team_run
        if team is not None and team != state.athena.run_id:
            # The team works on a run the screen does not show: one Zeus's `coder`
            # resumed, whose call's `episode` names a run that never starts. Shown from its
            # first record, which the reader follows it from again.
            self._seen.pop(team, None)
            moved = moved | self._episode(team) | self._restage()
        return moved

    def _carried(self, progress: WorkProgress | None) -> Changed:
        """The plan's tasks an earlier run of the work settled, which this run's board does not
        hold: kept beside the board, in the board's words."""
        athena = self.state.athena
        board = athena.board
        here = {t.plan_task or t.task_id for t in board.tasks.values()} if board else set()
        carried = tuple(
            CarriedTask(t.id, t.title, CARRIED_WORDS[t.state], t.run)
            for t in (progress.tasks if progress is not None else ())
            if t.state in CARRIED_WORDS and t.run != athena.run_id and t.id not in here
        )
        if carried == athena.carried:
            return NOTHING
        athena.carried = carried
        return _moved(Pane.ATHENA)

    def _hook(self, frame: RecordFrame, typed: HookP) -> Changed:
        """A hook that fired: the task's, in its worker's stream where it fired,
        or with no task, Athena's, as a step is placed."""
        if frame.task_id is None:
            self.state.athena.tail.append(AthenaEntry(frame.seq, AthenaKind.HOOK, typed))
            return _moved(Pane.ATHENA)
        self._task(frame.task_id).entries.append(WorkerEntry(frame.seq, WorkerKind.HOOK, typed))
        return _moved(Pane.WORKER)

    def _call(self, frame: RecordFrame, typed: ToolCallP) -> Changed:
        if typed.role == "athena":
            self.state.athena.tail.append(AthenaEntry(frame.seq, AthenaKind.CALL, typed))
            return _moved(Pane.ATHENA)
        if frame.task_id is None:
            return NOTHING
        self._task(frame.task_id).entries.append(WorkerEntry(frame.seq, WorkerKind.CALL, typed))
        return _moved(Pane.WORKER)

    def _model_call(self, frame: RecordFrame, typed: ModelCallP) -> Changed:
        budget = (StaleRoute.BUDGET,)
        self.state.live.finish(typed.role, frame.task_id)  # the call's record replaces it
        if typed.role == "athena":
            athena = self.state.athena
            athena.cost_usd = (athena.cost_usd or 0.0) + typed.cost_usd
            athena.rung = typed.rung or athena.rung
            athena.model = typed.model or athena.model
            athena.tokens_in = typed.tokens_in
            athena.tail.append(AthenaEntry(frame.seq, AthenaKind.MODEL, typed))
            return _moved(Pane.ATHENA, Pane.HEADER, refresh=budget)
        if frame.task_id is None:
            return Changed(refresh=frozenset(budget))
        task = self._task(frame.task_id)
        task.cost_usd = (task.cost_usd or 0.0) + typed.cost_usd
        task.rung = typed.rung or task.rung
        task.model = typed.model or task.model
        task.tokens_in = typed.tokens_in
        return _moved(Pane.WORKER, Pane.HEADER, refresh=budget)

    def _task(self, task_id: str) -> WorkerTask:
        return self.state.worker.tasks.setdefault(task_id, WorkerTask(task_id))

    # -- the stage ---------------------------------------------------------------------------

    def _restage(self) -> Changed:
        header = self.state.header
        before = (header.stage, header.work_done, header.work_delegated)
        board = self.state.athena.board
        live = [] if board is None else [t for t in board.tasks.values() if t.state != "re-cut"]
        # A task an earlier run settled is done, and was delegated, there.
        carried = [t for t in self.state.athena.carried if t.state != "re-cut"]
        header.work_delegated = len(live) + len(carried)
        header.work_done = sum(t.state in SETTLED for t in live) + len(carried)
        header.stage = self._stage()
        after = (header.stage, header.work_done, header.work_delegated)
        return NOTHING if after == before else _moved(Pane.HEADER)

    def _stage(self) -> Stage:
        state = self.state
        asking = {o.stop.kind for o in state.stops.open if o.stop is not None}
        if "design" in asking:
            return Stage.DESIGN
        if "plan" in asking:
            return Stage.PLAN
        if state.episode_open:
            if state.athena.plan_state not in (PlanState.APPROVED, PlanState.REVISED):
                return Stage.PLAN
            # CHECK once every task the plan lists has an end and every one delegated has
            # too: between two tasks, or with the plan's size unknown, it is still WORK.
            header = state.header
            planned = header.work_planned
            done = header.work_done
            if planned and done >= planned and done == header.work_delegated:
                return Stage.CHECK
            return Stage.WORK
        workspace = state.header.workspace
        pending = workspace is not None and workspace.handoff == "pending"
        return Stage.HANDOFF if state.worker.end is not None and pending else Stage.DESIGN


def _level_of(budget: SessionBudget) -> SpendLevel:
    """The level by the engine's own rules (`budget.manager`): red once the family's spend, or
    a rung's, reaches its cap (reaching it exactly is crossing it); amber at or past the first
    warning level. A lifted cap stops and warns at nothing."""
    cap, levels = budget.cap_usd, budget.warning_fractions or []
    rungs = (budget.rungs or {}).values()
    if (cap is not None and budget.spent_usd >= cap) or any(
        r.cap_usd is not None and r.spent_usd >= r.cap_usd for r in rungs
    ):
        return SpendLevel.CAP
    if cap and levels and budget.spent_usd >= cap * min(levels):
        return SpendLevel.WARNING
    return SpendLevel.OK


def _committed(run: RunView) -> int:
    """How many of the run's tasks are committed, or kept in place, by the board's own fold."""
    return sum(1 for t in run.tasks.values() if t.state in (COMMITTED, KEPT))
