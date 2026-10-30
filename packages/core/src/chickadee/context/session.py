"""The session, folded.

`SessionState` is a pure fold of the session log. Replaying the log reproduces it exactly,
and nothing about a session lives anywhere else — invariant 3 holds here as it does for the
run journal. `Session` is the log's one writer, and every write follows the order,
**commit, then emit**:

    1 check the payload against the state a line the fold would refuse is never written
    2 write the line; flush the store's, or an exception and nothing else
    3 fold it only a durable line is ever folded
    4 only now return an id, render a line, or emit a notice

Step 1 is what the order needs and §5.1 leaves implicit: the one failure it cannot survive is
a durable line that replay refuses, because the session would never open again.

What each kind folds into:

- `turn` → `Turn`: a user turn; an assistant turn with its calls and findings; a notice.
- `tool` → `ToolLine`: one call's result exactly as the model read it, and its handle.
- `episode` → `EpisodeLine`: one `coder` dispatch.
- `workspace` → `WorkspaceRef`: where the session's turns run.
- `work` → `WorkLine`, kept in order and read by `work.py`.
- `compaction` → `CutLine`: one committed cut — boundary, block and standing.
- `fork` → `ForkLine`: a child's first line.
- `constraint`, `rejected` → `LedgerState`; `concern` → `ConcernState`.
- `packet_record`, `design_record`, `summary`, `note`, `box_compaction`, `limits` → `OtherLine`:
  records, compactions, limits and the kinds D196 retired (`box_call`,
  `plan_decision`), kept for older logs, never read here. So are Zeus's `prefix` and `bound`
  lines, and a Zeus `locate`'s `scout` records, which its spec reads.
- `frame` → `FrameLine`, typed for assembly; `spend`, `cap` and `agent` are kept in order
  for their owners to read: `budget/family.py` and `agents/`.
- `undo`, `redo` → a rewind: the fold re-folds without what it hides (`rewind.py`).

**A tool call and its result are a pair, and the log keeps them one**: a user or
assistant turn is refused while any call still waits for its result, so no line order the
fold accepts can put a turn between a `tool_use` and its `tool_result`. A notice may arrive
mid-exchange; it is rendered after the results, never between them.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType, NoneType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast, get_args

from pydantic import JsonValue

from chickadee.context import rewind
from chickadee.context.attachment import Minted
from chickadee.context.concerns import ConcernState
from chickadee.context.handle import Handle, HandleTable
from chickadee.context.kept_lines import AgentRecord, EffectRecord, OtherLine
from chickadee.context.ledger import Constraint, LedgerState
from chickadee.context.lines import (
    PARSED,
    PAYLOAD_CLASSES,
    SESSION_KINDS,
    AssistantTurnP,
    Attachment,
    Call,
    CapLine,
    Concern,
    ConstraintP,
    CutLine,
    Dismissed,
    EpisodeLine,
    ForkP,
    FrameLine,
    HarnessTurnP,
    Line,
    PayloadError,
    Promoted,
    Reconsideration,
    RedoLine,
    RejectedAlt,
    SessionKind,
    SpendLine,
    TellLine,
    ToolP,
    Touched,
    TurnP,
    UndoLine,
    UserTurnP,
    parse,
    render,
    reparse,
)
from chickadee.context.results import CoderOutcome, ImageRef, ToolResult
from chickadee.context.rewind import ForkRefused
from chickadee.context.running_jobs import RunningJobs
from chickadee.context.spans import Span
from chickadee.context.standing import NOTICE_TRIGGERS, Notice, NoticeTrigger
from chickadee.context.store import (
    LOG_NAME,
    SessionCorrupt,
    SessionEvent,
    SessionLog,
    SessionStore,
    session_dir,
)
from chickadee.context.work_line import WorkLine
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.errors import HarnessBug
from chickadee.roles import rung_name_error
from chickadee.schema import ToolCall
from chickadee.schema.adapter import Usage
from chickadee.schema.isolation import WorkspaceMode

if TYPE_CHECKING:
    from chickadee.context.project import RunRecords

__all__ = [
    "ASK_USER",
    "CODER",
    "ROLES",
    "ConstraintLine",
    "ForkLine",
    "OtherLine",
    "Record",
    "Role",
    "Session",
    "SessionState",
    "ToolLine",
    "Turn",
    "View",
    "WorkspaceLine",
    "parsed",
]

Role = Literal["user", "assistant", "harness"]
ROLES: Final[tuple[Role, ...]] = get_args(Role)

ASK_USER: Final = "ask_user"
"""The tool whose result is the user's own words: its answer is numbered with
`[turn N]` and citable by a constraint exactly as a user turn is."""

CODER: Final = "coder"
"""The tool that opens an episode."""

_DATE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# -- the records --------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    seq: int
    role: Role
    text: str
    calls: tuple[Call, ...] = ()
    findings: Mapping[str, str] = field(default_factory=lambda: MappingProxyType[str, str]({}))
    """`call_id → finding`: what this turn concluded from results it consumed. Written
    AFTER the result, in the turn that consumes it."""
    notice: Notice | None = None
    """Set iff `role == "harness"`."""
    workspace: WorkspaceMode | None = None
    """A user turn's `--worktree` or `--in-place`."""
    attachments: tuple[Attachment, ...] = ()
    """What the person sent with a user turn's words."""


@dataclass(frozen=True)
class ToolLine:
    """A call's result, exactly as the model read it."""

    seq: int
    call_id: str
    text: str
    ran: bool
    """Whether the call executed. A refused call did nothing, and nothing it declared
    happened."""
    handle: Handle | None
    view: tuple[Span, ...]
    """The file lines this result shows — the view it holds."""
    supersedes: tuple[str, ...]
    """Calls whose results this merged view absorbed. They render as a stub; their
    calls stay byte-identical."""
    image: ImageRef | None
    run_id: str | None
    outcome: CoderOutcome | None = None  # a `coder` result's terminal and why


@dataclass(frozen=True)
class WorkspaceLine:
    seq: int
    ref: WorkspaceRef


@dataclass(frozen=True)
class ForkLine:
    seq: int
    parent_id: str
    at_seq: int
    reason: str


@dataclass(frozen=True)
class ConstraintLine:
    constraint: Constraint
    supersedes: str | None


def _known_rung(rung: str) -> None:
    """A line names a rung by the one name rule. The log cannot know the config,
    so whether the rung is declared is its writer's check (the ledger hands out a view only
    for a declared rung, and `set_cap` refuses an undeclared one)."""
    why = rung_name_error(rung)
    if why is not None:
        raise PayloadError(why)


Record = (
    Turn
    | ToolLine
    | FrameLine
    | SpendLine
    | CapLine
    | EpisodeLine
    | WorkspaceLine
    | WorkLine
    | TellLine
    | CutLine
    | ForkLine
    | ConstraintLine
    | RejectedAlt
    | Reconsideration
    | Concern
    | Touched
    | Promoted
    | Dismissed
    | UndoLine
    | RedoLine
    | OtherLine
)


@dataclass(frozen=True)
class View:
    """File lines a result shows, still held by the session."""

    handle_id: str
    spans: tuple[Span, ...]
    call_id: str
    seq: int


# -- the fold -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Fold:
    """A parsed kind's check and apply: `SessionState._FOLDS` holds one per kind."""

    check: Callable[[SessionState, int, str | None, Any], Record]
    apply: Callable[[SessionState, Any], None]


class SessionState:
    """A pure fold of the session log. `check` reads a payload against the state and
    changes nothing; `apply` folds a line that is already durable."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.last_seq = 0
        self._lines: list[tuple[SessionEvent, Record]] = []
        self.hidden: frozenset[int] = frozenset()
        """Seqs a standing rewind hides (`rewind.hidden_seqs`). Everything below is folded
        from the other lines only, and refolded when an `undo` or a `redo` applies."""
        self._clear()

    def _clear(self) -> None:
        self.handles = HandleTable()
        self.ledger = LedgerState()
        self.concerns = ConcernState()
        self.noticed_stale: set[str] = set()
        """Handles whose view was already noticed stale (trigger 3), so the sweep never
        says it twice. A fresh view of the same handle clears it."""
        self._turns: dict[int, Turn] = {}
        self._calls: dict[str, Call] = {}
        self._results: dict[str, ToolLine] = {}
        self._findings: dict[str, str] = {}
        self._superseded: dict[str, str] = {}
        self._pending: dict[str, None] = {}
        self._episodes: list[EpisodeLine] = []
        self._workspaces: list[WorkspaceLine] = []
        self._cuts: list[CutLine] = []
        self._forks: list[ForkLine] = []
        self._frames: list[FrameLine] = []
        self.attachments: dict[str, Attachment] = {}
        """Every attachment of a turn no undo hides, by id (`A-1` on): what a constraint,
        a design or a role's head names. `next_attachment` counts the hidden ones too."""
        self._tells: list[TellLine] = []
        self.jobs = RunningJobs()  # D216: open `JOB` effects, the jobs running now

    # -- replay ---------------------------------------------------------------------------

    @staticmethod
    def replay(
        session_id: str,
        root: Path,
        events: Iterable[SessionEvent],
        payload: Callable[[SessionEvent], bytes],
    ) -> SessionState:
        """Fold every line in order, each parsed by its kind's type (`context.lines`) as the
        fold reaches it."""
        return SessionState.fold(session_id, root, (parsed(root, e, payload) for e in events))

    @staticmethod
    def fold(
        session_id: str, root: Path, lines: Iterable[tuple[SessionEvent, object | None]]
    ) -> SessionState:
        """Check and apply every parsed line from the first: the one fold, which `replay` and
        `ReplayCache` both use. A line it refuses is `SessionCorrupt` naming the
        log it lives in and its seq — never skipped, since a session that silently dropped a
        constraint is worse than one that will not open."""
        state = SessionState(session_id)
        for event, line in lines:
            try:
                record = state.check(event.kind, line, seq=event.seq, run_id=event.run_id)
            except PayloadError as exc:
                raise _corrupt(root, event, exc) from exc
            state.apply(event, record)
        return state

    # -- checking (pure) ------------------------------------------------------------------

    def check(
        self, kind: SessionKind, line: object | None, *, seq: int, run_id: str | None
    ) -> Record:
        """A parsed line against the state as it stands, and the record it folds into. `line`
        is `None` for a kind another package owns, which this fold keeps unread."""
        if seq != self.last_seq + 1:
            raise PayloadError(f"line {seq} does not follow {self.last_seq}")
        if kind not in SESSION_KINDS:
            raise PayloadError(f"no fold for line kind {kind!r}")
        if not isinstance(line, PAYLOAD_CLASSES.get(kind, NoneType)):
            raise HarnessBug(f"a {kind} line holds a {type(line).__name__}")
        if kind not in PARSED:
            return OtherLine(seq, kind, run_id)
        return self._FOLDS[kind].check(self, seq, run_id, line)

    def _check_turn(self, seq: int, run_id: str | None, line: TurnP) -> Turn:
        if isinstance(line, HarnessTurnP):
            if line.trigger not in NOTICE_TRIGGERS:
                raise PayloadError(f"{line.trigger} is not one of §5.11.2's notice triggers")
            try:
                n = Notice(NoticeTrigger(line.trigger), line.text, line.handle, line.carries)
            except ValueError as exc:
                raise PayloadError(str(exc)) from exc
            return Turn(seq, "harness", n.text, notice=n)
        if self._pending:
            waiting = next(iter(self._pending))
            raise PayloadError(f"call {waiting} is still waiting for its result")
        if isinstance(line, UserTurnP):
            self._check_attachments(line.attachments)
            return Turn(
                seq, "user", line.text, workspace=line.workspace, attachments=line.attachments
            )
        calls: list[Call] = []
        for c in line.calls:
            call = replace(c, seq=seq)
            if not call.call_id or not call.tool:
                raise PayloadError("a call has an id and a tool")
            if call.call_id in self._calls or any(x.call_id == call.call_id for x in calls):
                raise PayloadError(f"call id {call.call_id} is already used")
            calls.append(call)
        if sum(c.tool == CODER for c in calls) > 1:
            # `25-zeus.md` §1; and D123's `opened_at_seq` names a call only because of it.
            raise PayloadError("a turn makes at most one coder delegation")
        findings: dict[str, str] = {}
        for call_id, finding in line.findings.items():
            if call_id not in self._results:
                raise PayloadError(f"a finding for {call_id}, which has no result")
            if call_id in self._findings:
                raise PayloadError(f"{call_id} already has a finding")
            if not isinstance(finding, str) or not finding.strip():
                raise PayloadError(f"the finding for {call_id} says something")
            findings[call_id] = finding
        return Turn(seq, "assistant", line.text, tuple(calls), MappingProxyType(findings))

    def next_attachment(self) -> int:
        """The number the next attachment takes: one past every attachment
        the log holds, a turn an undo hides included, so a number once given is never given
        again and `A-2` names one file for the whole session."""
        return 1 + sum(
            len(record.attachments)
            for _event, record in self._lines
            if isinstance(record, Turn)
        )

    def _check_attachments(self, attachments: tuple[Attachment, ...]) -> None:
        """Numbered on from the session's last, never reused, each handle a
        stored one of kind `attachment` in the session's own store."""
        if not attachments:
            # Most turns carry none, and the count walks the whole log.
            return
        n = self.next_attachment() - 1
        for a in attachments:
            n += 1
            if a.id != f"A-{n}":
                raise PayloadError(f"attachment {a.id} is out of order; the next is A-{n}")
            for h in (a.handle, a.text_handle):
                if h is None:
                    continue
                if h.kind != "attachment" or h.run_id is not None:
                    raise PayloadError(f"{a.id}'s handle {h.id} is not a session attachment")
                held = self.handles.get(h.id)
                if held is not None and held.address != h.address:
                    raise PayloadError(f"handle {h.id} already names a different address")

    def _check_tool(self, seq: int, run_id: str | None, line: ToolP) -> ToolLine:
        if line.call_id not in self._pending:
            raise PayloadError(f"no call {line.call_id} is waiting for a result")
        call = self._calls[line.call_id]
        if call.tool == CODER:
            self._check_coder_result(call, run_id)
        if line.handle is not None:
            held = self.handles.get(line.handle.id)
            if held is not None and held.address != line.handle.address:
                raise PayloadError(f"handle {line.handle.id} already names a different address")
        if line.view and line.handle is None:
            raise PayloadError("a view is a view of a handle")
        for old in line.supersedes:
            if old not in self._results or old in self._superseded:
                raise PayloadError(f"{old} has no live result to supersede")
        return ToolLine(
            seq=seq,
            call_id=line.call_id,
            text=line.text,
            ran=line.ran,
            handle=line.handle,
            view=line.view,
            supersedes=line.supersedes,
            image=line.image,
            run_id=run_id,
            outcome=line.outcome,
        )

    def _check_episode(self, seq: int, run_id: str | None, line: EpisodeLine) -> EpisodeLine:
        line = replace(line, seq=seq)
        if any(e.run_id == line.run_id for e in self._episodes):
            raise PayloadError(f"run {line.run_id} already has an episode")
        if line.opened_at_seq is None:
            return line  # a host's run: no `coder` call of Zeus's to answer
        turn = self._turns.get(line.opened_at_seq)
        coder = next((c for c in turn.calls if c.tool == CODER), None) if turn else None
        if coder is None:
            raise PayloadError("an episode opens at the turn that called coder")
        if coder.call_id not in self._pending:
            raise PayloadError(
                f"coder call {coder.call_id} is answered; an episode opens at dispatch"
            )
        if self._episode_of(coder) is not None:
            raise PayloadError(f"coder call {coder.call_id} already opened an episode")
        return line

    def _check_coder_result(self, call: Call, run_id: str | None) -> None:
        """The episode is written at dispatch, so the result names its run. One naming no
        run, or another, would leave the episode — and every later unit — waiting for ever."""
        opened = self._episode_of(call)
        expected = opened.run_id if opened is not None else None
        if run_id == expected:
            return
        if expected is None:
            raise PayloadError(f"coder call {call.call_id} opened no episode, so no run")
        raise PayloadError(f"coder call {call.call_id} opened run {expected}; name that run")

    def _episode_of(self, call: Call) -> EpisodeLine | None:
        """The episode a `coder` call opened. One per turn at most, so the turn names it."""
        return next((e for e in self._episodes if e.opened_at_seq == call.seq), None)

    def _check_cut(self, seq: int, run_id: str | None, line: CutLine) -> CutLine:
        cut = replace(line, seq=seq)
        if cut.previous != self.boundary:
            raise PayloadError(f"a cut starts at the last boundary, {self.boundary}")
        if not cut.previous < cut.boundary <= seq:
            raise PayloadError(f"boundary {cut.boundary} is not after {cut.previous}")
        if not cut.block:
            raise PayloadError("a cut has a block")
        for s in cut.lifted:
            held = self.line(s)
            if held is None or not isinstance(held[1], ToolLine):
                raise PayloadError(f"lifted seq {s} is not a tool result")
        return cut

    def _check_frame(self, seq: int, run_id: str | None, line: FrameLine) -> FrameLine:
        # A frame opens a turn, so it waits for every result, as the turn it opens would.
        if self._pending:
            waiting = next(iter(self._pending))
            raise PayloadError(f"call {waiting} is still waiting for its result")
        if not _DATE_RE.match(line.date):
            raise PayloadError(f"a frame's date is YYYY-MM-DD, not {line.date!r}")
        if not line.text.strip():
            raise PayloadError("a frame says something")
        return replace(line, seq=seq)

    def _check_constraint(self, seq: int, run_id: str | None, line: ConstraintP) -> Record:
        constraint, supersedes = self.ledger.check_constraint(seq, line)
        for n in constraint.sources:
            if constraint.author == "zeus" and not self.is_user_words(n):
                raise PayloadError(f"constraint {constraint.id} cites seq {n}, not a user turn")
        return ConstraintLine(constraint, supersedes)

    def _check_tell(self, seq: int, run_id: str | None, line: TellLine) -> TellLine:
        want = f"tell-{len(self._tells) + 1}"
        if line.id != want:
            raise PayloadError(f"the session's next tell is {want}, not {line.id}")
        for n in line.sources:
            # D120's rule, as a constraint's: Zeus words it, and the person is its authority.
            if not self.is_user_words(n):
                raise PayloadError(f"{line.id} cites seq {n}, not the person's words")
        return replace(line, seq=seq)

    def _check_cap(self, seq: int, run_id: str | None, line: CapLine) -> CapLine:
        if line.hard_cost_usd == 0:
            raise PayloadError("a cap is above zero; lift it with null instead")
        return replace(line, seq=seq)

    def _check_spend(self, seq: int, run_id: str | None, line: SpendLine) -> SpendLine:
        if not line.model:
            raise PayloadError("a charge names the model that served it")
        return replace(line, seq=seq)

    def _check_fork(self, seq: int, run_id: str | None, line: ForkP) -> ForkLine:
        fork = ForkLine(seq, line.forked_from.session_id, line.forked_from.at_seq, line.reason)
        if fork.at_seq != self.last_seq:
            raise PayloadError("a fork line follows its parent's lines at the fork point")
        return fork

    # -- folding --------------------------------------------------------------------------

    def apply(self, event: SessionEvent, record: Record) -> None:
        self._lines.append((event, record))
        self.last_seq = event.seq
        if event.kind in PARSED:
            self._FOLDS[event.kind].apply(self, record)

    def _refold(self) -> None:
        """A rewind applied: fold every visible line again from nothing. The rewinds themselves
        are not re-applied, which is what ends the recursion; `hidden` already counts them."""
        self.hidden = rewind.hidden_seqs(self._lines)
        self._clear()
        for event, record in self._lines:
            if event.seq not in self.hidden and event.kind in PARSED.keys() - {"undo", "redo"}:
                self._FOLDS[event.kind].apply(self, record)

    def _apply_turn(self, turn: Turn) -> None:
        self._turns[turn.seq] = turn
        for a in turn.attachments:
            self.attachments[a.id] = a
            self.handles.add(a.handle)
            if a.text_handle is not None:
                self.handles.add(a.text_handle)
        for call in turn.calls:
            self._calls[call.call_id] = call
            self._pending[call.call_id] = None
        self._findings.update(turn.findings)
        n = turn.notice
        if n is not None and n.trigger is NoticeTrigger.STALE_VIEW and n.handle is not None:
            self.noticed_stale.add(n.handle)
        if n is not None and n.carries is not None:
            self.handles.add(n.carries)

    def _apply_tool(self, line: ToolLine) -> None:
        del self._pending[line.call_id]
        self._results[line.call_id] = line
        for old in line.supersedes:
            self._superseded[old] = line.call_id
        if line.handle is not None:
            self.handles.add(line.handle)
            if line.view:
                self.noticed_stale.discard(line.handle.id)

    # -- reading: lines -------------------------------------------------------------------

    def has_line(self, seq: int) -> bool:
        return 1 <= seq <= self.last_seq

    def line(self, seq: int) -> tuple[SessionEvent, Record] | None:
        """The line at `seq`, or None when there is none or a rewind hides it."""
        if not self.has_line(seq) or seq in self.hidden:
            return None
        return self._lines[seq - 1]

    def lines(
        self, lo: int = 1, hi: int | None = None, *, with_hidden: bool = False
    ) -> Iterator[tuple[SessionEvent, Record]]:
        """Visible lines with `lo <= seq <= hi`, in order; `with_hidden` for a reader of the
        log rather than of the conversation (a rewind's own checks)."""
        top = self.last_seq if hi is None else min(hi, self.last_seq)
        for seq in range(max(lo, 1), top + 1):
            if with_hidden or seq not in self.hidden:
                yield self._lines[seq - 1]

    def is_user_words(self, seq: int) -> bool:
        """A user turn, or an `ask_user` answer — the two lines a constraint may cite,
        and the two that carry `[turn N]`."""
        line = self.line(seq)
        if line is None:
            return False
        record = line[1]
        if isinstance(record, Turn):
            return record.role == "user"
        if isinstance(record, ToolLine):
            call = self._calls.get(record.call_id)
            return call is not None and call.tool == ASK_USER and record.ran
        return False

    # -- reading: exchanges ----------------------------------------------------------------

    def turn(self, seq: int) -> Turn | None:
        return self._turns.get(seq)

    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns.values())

    def call(self, call_id: str) -> Call | None:
        return self._calls.get(call_id)

    def result(self, call_id: str) -> ToolLine | None:
        return self._results.get(call_id)

    def finding(self, call_id: str) -> str | None:
        return self._findings.get(call_id)

    def superseded_by(self, call_id: str) -> str | None:
        return self._superseded.get(call_id)

    def pending_calls(self) -> tuple[str, ...]:
        """Calls still waiting for their result, oldest first."""
        return tuple(self._pending)

    def calls_upto(self, seq: int) -> tuple[Call, ...]:
        return tuple(c for c in self._calls.values() if c.seq <= seq)

    def ran(self, call_id: str) -> bool:
        line = self._results.get(call_id)
        return line is not None and line.ran

    def rereads(self, handle_id: str, after: int) -> int:
        """Results after seq `after` that returned the same handle — `rehydrated Nx`."""
        return sum(
            1
            for line in self._results.values()
            if line.seq > after and line.handle is not None and line.handle.id == handle_id
        )

    # -- reading: views ---------------------------------------------------------------------

    def views(self) -> tuple[View, ...]:
        """Every view whose result was not absorbed into a later merged view."""
        return tuple(
            View(line.handle.id, line.view, line.call_id, line.seq)
            for line in self._results.values()
            if line.handle is not None and line.view and line.call_id not in self._superseded
        )

    def expanded_views(self) -> tuple[View, ...]:
        """Views still EXPANDED in context: in the verbatim tail past the last cut,
        or re-emitted with a pinned body. A view collapsed to a receipt cannot be reasoned
        from, so only these can mislead."""
        lifted = set(self._cuts[-1].lifted) if self._cuts else set[int]()
        return tuple(v for v in self.views() if v.seq >= self.boundary or v.seq in lifted)

    # -- reading: episodes, workspace, cuts, lineage ----------------------------------------

    def tells(self) -> tuple[TellLine, ...]:
        """Zeus's messages to the team, in order: `tell-1` first."""
        return tuple(self._tells)

    def episodes(self) -> tuple[EpisodeLine, ...]:
        return tuple(self._episodes)

    def episodes_upto(self, seq: int) -> tuple[EpisodeLine, ...]:
        return tuple(e for e in self._episodes if e.seq <= seq)

    def episode(self, run_id: str) -> EpisodeLine | None:
        return next((e for e in self._episodes if e.run_id == run_id), None)

    def workspace_at(self, seq: int) -> WorkspaceRef | None:
        """The workspace as recorded at `seq` — what a fork at that seq starts from."""
        at = [w for w in self._workspaces if w.seq <= seq]
        return at[-1].ref if at else None

    def cuts(self) -> tuple[CutLine, ...]:
        return tuple(self._cuts)

    @property
    def boundary(self) -> int:
        """The sticky boundary: the last cut's, or 0 before any cut."""
        return self._cuts[-1].boundary if self._cuts else 0

    @property
    def forked_from(self) -> tuple[str, int] | None:
        """The immediate parent and fork seq, or None for a root session."""
        return (self._forks[-1].parent_id, self._forks[-1].at_seq) if self._forks else None

    def frames(self) -> tuple[FrameLine, ...]:
        return tuple(self._frames)

    def others(self, kind: SessionKind) -> tuple[OtherLine, ...]:
        return tuple(r for _, r in self.lines() if isinstance(r, OtherLine) and r.kind == kind)

    def _apply_constraint(self, record: ConstraintLine) -> None:
        self.ledger.apply(record.constraint, supersedes=record.supersedes)

    _FOLDS: ClassVar[Mapping[SessionKind, _Fold]] = {
        "turn": _Fold(_check_turn, _apply_turn),
        "tool": _Fold(_check_tool, _apply_tool),
        "episode": _Fold(_check_episode, lambda s, e: s._episodes.append(e)),
        "workspace": _Fold(
            lambda s, q, r, x: WorkspaceLine(q, x), lambda s, e: s._workspaces.append(e)
        ),
        "work": _Fold(lambda s, q, r, x: x, lambda s, e: None),
        "compaction": _Fold(_check_cut, lambda s, e: s._cuts.append(e)),
        "fork": _Fold(_check_fork, lambda s, e: s._forks.append(e)),
        "constraint": _Fold(_check_constraint, _apply_constraint),
        "rejected": _Fold(
            lambda s, q, r, x: s.ledger.check_rejected(q, x), lambda s, e: s.ledger.apply(e)
        ),
        "concern": _Fold(
            lambda s, q, r, x: s.concerns.check(q, r, x), lambda s, e: s.concerns.apply(e)
        ),
        "frame": _Fold(_check_frame, lambda s, e: s._frames.append(e)),
        "spend": _Fold(_check_spend, lambda s, e: None),
        "tell": _Fold(_check_tell, lambda s, e: s._tells.append(e)),
        "cap": _Fold(_check_cap, lambda s, e: None),
        "shell_effect": _Fold(
            lambda s, q, r, x: EffectRecord(q, "shell_effect", r, x),
            lambda s, e: s.jobs.apply(e),
        ),
        "agent": _Fold(lambda s, q, r, x: AgentRecord(q, "agent", r, x), lambda s, e: None),
        "undo": _Fold(lambda s, q, r, x: rewind.check_undo(s, q, x), lambda s, e: s._refold()),
        "redo": _Fold(lambda s, q, r, x: rewind.check_redo(s, q, x), lambda s, e: s._refold()),
    }


def parsed(
    root: Path, event: SessionEvent, payload: Callable[[SessionEvent], bytes]
) -> tuple[SessionEvent, object | None]:
    """A line and its payload, parsed by its kind's type; refused as `SessionCorrupt` at its
    seq, as the fold refuses a line."""
    try:
        return event, parse(event.kind, payload(event))
    except PayloadError as exc:
        raise _corrupt(root, event, exc) from exc


def _corrupt(root: Path, event: SessionEvent, exc: PayloadError) -> SessionCorrupt:
    return SessionCorrupt(session_dir(root, event.session_id) / LOG_NAME, event.seq, str(exc))


# -- the writer ---------------------------------------------------------------------------


class Session:
    """A session open for writing: its log, and the fold of it.

    Every write takes `lock`, a blob's too (its store has one writer, `durable/blobs.py`).
    Reads may run in parallel, as agents do, and each records its result, so
    two must never race for one seq. It is re-entrant: a constraint's notice is a write."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store
        self.lock = threading.RLock()
        self.state = SessionState.replay(
            store.session_id, store.root, store.events(), store.payload
        )

    # -- lifecycle ------------------------------------------------------------------------

    @staticmethod
    def create(root: Path, session_id: str) -> Session:
        return Session._wrap(SessionStore.create(root, session_id))

    @staticmethod
    def open(root: Path, session_id: str) -> Session:
        return Session._wrap(SessionStore.open(root, session_id))

    @staticmethod
    def _wrap(store: SessionStore) -> Session:
        try:
            return Session(store)
        except BaseException:
            store.close()
            raise

    @staticmethod
    def state_at(root: Path, session_id: str, at_seq: int) -> SessionState:
        """The session folded up to `at_seq`, read without its lock: what a fork starts from."""
        log = SessionLog.read(root, session_id, upto=at_seq)
        return SessionState.replay(session_id, root, log.events(), log.payload)

    @staticmethod
    def fork(
        root: Path,
        parent_id: str,
        at_seq: int,
        reason: str,
        child_id: str,
        *,
        code: Literal["restore", "keep"],
        records: RunRecords | None = None,
        kept_paths: Sequence[str] = (),
        runs: Sequence[str] = (),
    ) -> Session:
        """§5.1.1. The parent is read up to `at_seq` without its lock and never written;
        the child's first line is `fork` and every read walks through to the parent.

        `code` is the person's choice, with no default: the facade asks.
        The code itself is `undo.fork_code`'s; the child's prefix is the parent's, verbatim.
        What happens here is arrival (`rewind.arrive`): the effects no fork can undo, and on
        `keep` the code changed in turns the child cannot see, `kept_paths` by `runs`.

        A fork that fails before it has arrived leaves nothing behind. Kept, the child
        would reopen as a session that was never told, since arrival does not recur."""
        parent = Session.state_at(root, parent_id, at_seq)
        refusal = rewind.safe_point_refusal(parent, records)
        if refusal is not None:
            raise ForkRefused(parent_id, at_seq, refusal)
        store = SessionStore.create(root, child_id, fork=(parent_id, at_seq, reason))
        try:
            session = Session(store)
            kept = kept_paths if code == "keep" else ()
            rewind.arrive(session, parent_id, at_seq, reason, records, kept, runs)
        except BaseException:
            store.discard()
            raise
        return session

    def close(self) -> None:
        with self.lock:
            self._store.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- identity -------------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._store.session_id

    @property
    def root(self) -> Path:
        return self._store.root

    @property
    def next_seq(self) -> int:
        """The seq the next write takes. Valid under `lock` until that write lands; an id
        built from it is handed out only after."""
        return self._store.next_seq

    # -- the one write path -----------------------------------------------------------------

    def append(self, line: Line, *, run_id: str | None = None) -> SessionEvent:
        """Check, write, fold — the order. The line is rendered to the bytes the log will
        hold and parsed back through its type, so what is checked is exactly what replay
        will read, and the live fold and the replayed one cannot differ."""
        kind = line.KIND
        raw = render(line)
        parsed = reparse(line, raw)
        with self.lock:
            record = self.state.check(kind, parsed, seq=self._store.next_seq, run_id=run_id)
            event = self._store.append(kind, raw, run_id=run_id)
            self.state.apply(event, record)
            return event

    # -- recording ------------------------------------------------------------------------

    def record_user(
        self,
        text: str,
        workspace: WorkspaceMode | None = None,
        attachments: Sequence[Minted] = (),
    ) -> Turn:
        """The person's words, and what they attached: each arrives with its handle
        minted and its blobs stored (`Reader.attach`), and is numbered here, under the lock,
        after the session's last."""
        with self.lock:
            n = self.state.next_attachment()
            line = UserTurnP(
                text,
                workspace=workspace,
                attachments=tuple(m.numbered(f"A-{n + i}") for i, m in enumerate(attachments)),
            )
            return cast("Turn", self._record(line))

    def record_assistant(
        self,
        text: str,
        calls: Sequence[ToolCall] = (),
        *,
        findings: Mapping[str, str] | None = None,
    ) -> Turn:
        """One assistant response: its text, the calls it made, and the findings it drew from
        results it consumed. A call carries no purpose line."""
        line = AssistantTurnP(
            text,
            tuple(
                Call(
                    c.id,
                    c.name,
                    cast("Mapping[str, JsonValue]", MappingProxyType(dict(c.arguments))),
                )
                for c in calls
            ),
            MappingProxyType(dict(findings or {})),
        )
        with self.lock:
            return cast("Turn", self._record(line))

    def record_tool(
        self,
        call_id: str,
        result: ToolResult | str,
        *,
        ran: bool = True,
        run_id: str | None = None,
        outcome: CoderOutcome | None = None,
    ) -> ToolLine:
        """A call's result, as the model will read it. A `coder` result carries its run."""
        with self.lock:
            if isinstance(result, str):
                line = ToolP(call_id, result, ran, outcome=outcome)
            else:
                image = None
                if result.image is not None:
                    sha = self._store.put_blob(result.image.data)
                    image = ImageRef(sha, result.image.media_type)
                told = outcome if outcome is not None else result.outcome
                line = ToolP(call_id, result.render(), ran, result.handle, result.view,
                             result.supersedes, image, told)  # fmt: skip
            return cast("ToolLine", self._record(line, run_id=run_id))

    def record_notice(self, n: Notice) -> int:
        """One framed line, appended and never edited."""
        with self.lock:
            return self.append(
                HarnessTurnP(int(n.trigger), n.text, n.handle, carries=n.carries)
            ).seq

    def record_episode(self, line: EpisodeLine) -> EpisodeLine:
        with self.lock:
            return cast("EpisodeLine", self._record(line, run_id=line.run_id))

    def record_spend(
        self, model: str, usage: Usage, cost_usd: float, rung: str, role: str | None = None
    ) -> SpendLine:
        """One model call's charge, with the rung that made it and the role
        whose call it was. Fsynced as it is appended (`spend` is in `FSYNC_KINDS`)."""
        _known_rung(rung)
        line = SpendLine(
            model=model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            cached_in=usage.cached_prompt_tokens or 0,
            cache_writes=usage.cache_write_tokens or 0,
            cost_usd=cost_usd,
            estimated=usage.estimated,
            rung=rung,
            role=role,
        )
        with self.lock:
            return cast("SpendLine", self._record(line))

    def record_cap(self, hard_cost_usd: float | None, rung: str | None = None) -> CapLine:
        """The person's cap for the family, or `None` to lift it; for one rung
        beneath the total when `rung` names it. Fsynced, since `cap` is in
        `FSYNC_KINDS`; checked as a config load is, so a non-finite or non-positive cap, or a
        malformed rung name, is refused before it is written."""
        if rung is not None:
            _known_rung(rung)
        with self.lock:
            return cast("CapLine", self._record(CapLine(hard_cost_usd, rung)))

    def record_workspace(self, ref: WorkspaceRef) -> int:
        with self.lock:
            return self.append(ref).seq

    def record_tell(
        self, run_id: str, text: str, sources: tuple[int, ...], key: str | None = None
    ) -> TellLine:
        """Zeus's message to the team about run `run_id`, numbered next: its text a
        session blob, its citations checked by the fold, and synced, since a run reads it
        from another thread."""
        with self.lock:
            line = TellLine(
                id=f"tell-{len(self.state.tells()) + 1}",
                run_id=run_id,
                text_sha=self._store.put_blob(text.encode("utf-8")),
                sources=sources,
                key=key,
            )
            told = cast("TellLine", self._record(line))
            self.sync()
            return told

    def _record(self, line: Line, *, run_id: str | None = None) -> Record:
        seq = self.append(line, run_id=run_id).seq
        folded = self.state.line(seq)
        assert folded is not None  # the fold just applied it
        return folded[1]

    # -- blobs ----------------------------------------------------------------------------

    def put_blob(self, data: bytes) -> str:
        with self.lock:
            return self._store.put_blob(data)

    def blob(self, sha: str) -> bytes:
        """A session blob, never pruned (`gc` prunes runs): a miss is `SessionCorrupt`."""
        return self._store.get_blob(sha)

    def sync(self) -> None:
        with self.lock:
            self._store.sync()
