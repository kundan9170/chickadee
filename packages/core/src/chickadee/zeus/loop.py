"""The turn loop.

An ordinary loop — assemble, call the model, dispatch, collect, repeat until a response makes
no call — with the enforcement in the dispatcher, not in a planning graph.
**It owns no durable state.** What it knows is the session log, read through the fold, and
everything it does is a line appended to that log before anything is said about it (commit,
then emit). The events it streams are the live view and may be dropped without loss.

Every call a response makes is checked, in the order it was made, before any of them runs,
and each check in this order:

    the call cap BOUND_REACHED{tool_calls} from the (cap+1)th call of the turn
    the read cap BOUND_REACHED{result_tokens} once the turn has taken in its allowance
                        — asked again as each group is dispatched
    unknown tool UNKNOWN_TOOL{name, nearest} — never corrected to the nearest
    findings INVALID_ARGUMENTS — a finding names an earlier result
    the tool's schema INVALID_ARGUMENTS{tool, violations, schema}
    per response a second coder, and locate past the fan-out, are refused
    repetition served with `repetition{count, since_generation}` from N identical
                        calls against an unchanged workspace; refused from 2N

The checks are the one dispatcher's: `ROLES`' Zeus pipeline, its shared rules in that
order and Zeus's own three in its slots (`calls.py`). This module admits each response, writes
the turn's `bound` lines (`bounds.py`), and runs what was admitted. A call refused by a bound
is not dispatched and does not count. Every other answered call counts, an invalid one
included: it cost a round trip. Then the calls run by `ToolSpec.parallel` — navigate, read and
record together, shell and ask_user one at a time, coder alone — and each result is recorded
as it is answered, by Zeus's sink.

"Together" is a group, not a thread pool. A group's calls run one after another on the thread
that runs the turn: the graph's SQLite connection refuses any other thread, and the scout
behind `locate` holds one conversation. The group is still what the read bound is asked of,
and a stop still answers every call of it that has not started.

**The calls are recorded as they were sent** (invariant 19). A call's `findings` are read out of
its arguments and recorded beside them; the arguments themselves are never edited, and
validation is of a view with them set aside. No call carries a purpose line.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast

from chickadee.client.charge import charge_unanswered
from chickadee.client.errors import ClientError, ContextOverflow, RequestCancelled
from chickadee.client.live import (
    ContextUpdate,
    LiveEvent,
    LiveSink,
    Restarted,
    Speaker,
    TextUpdate,
    ThoughtUpdate,
    ToolUpdate,
    speaking,
)
from chickadee.client.request import ChatRequest, ModelClient, text_of
from chickadee.context.attachment import CheckedFile, Minted
from chickadee.context.compaction.cut import OVERFLOW_RETRIES, Compaction, TurnTooLarge
from chickadee.context.lines import FrameLine, render
from chickadee.context.request import assemble, est_text, est_tools, within_sight
from chickadee.context.results import Interrupted
from chickadee.context.session import OtherLine, ToolLine
from chickadee.context.standing import NoticeTrigger, hook_noted, moved_by_flag
from chickadee.errors import HarnessBug
from chickadee.hooks.failures import HookDenied, HookHalted
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.loop import RoleLoopHooks
from chickadee.hooks.payloads import BeforeStop, PromptSubmitted
from chickadee.hooks.points import HookPoint
from chickadee.hooks.stop import at_stop
from chickadee.hooks.verdicts import Combined, Deny
from chickadee.limits import ResolvedLimit, limit, resolved_value
from chickadee.roles import role
from chickadee.schema import StopReason, ToolCall, sha256_hex
from chickadee.schema.adapter import Usage
from chickadee.schema.isolation import WorkspaceMode
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.meta import is_line, split_meta
from chickadee.tools.policy import RolePolicy
from chickadee.tools.rules import BoundReached, Pending, Refuse
from chickadee.tools.spec import INLINE_BYTES_DEFAULT, RESULT_LINES_DEFAULT
from chickadee.tools.tool import Answer, Tool
from chickadee.zeus.bounds import BoundLines
from chickadee.zeus.calls import (
    RepetitionKey,
    ResponseFacts,
    ZeusSink,
    ZeusTools,
    repetition_key,
    zeus_dispatcher,
)
from chickadee.zeus.delegate import Delegation, OpenedEpisode
from chickadee.zeus.interrupt import Interrupts, Words
from chickadee.zeus.prefix import Prefix, record_prefix, store_texts
from chickadee.zeus.tools.act import Delegations

if TYPE_CHECKING:
    from chickadee.budget.manager import Ledger
    from chickadee.context.session import Session, SessionState
    from chickadee.context.terms import Vision
    from chickadee.schema.adapter import Completion, RetryParams

__all__ = [
    "LOCATE_FANOUT",
    "ZEUS_REPEAT_DEFAULT",
    "Event",
    "Interrupted",
    "LoopConfig",
    "ZeusLoop",
    "frame_due",
    "frame_text",
    "workspace_generation",
]

ZEUS_REPEAT_DEFAULT: Final = 3
"""D98's `zeus_repeat`: a notice at this many identical calls, a refusal at
twice it; `0` disables. `EngineConfig` has no key for it yet, so this is the value used."""

LOCATE_FANOUT: Final = 4
"""`locate` calls one response may run (past the fan-out the excess are
refused, the first N proceed). The spec names no number; a starting value, to be measured."""

EventKind = Literal[
    "thinking",
    "tool_call",
    "tool_result",
    "delegation",
    "answer",
    "error",
    "text",
    "thought",
    "tool_started",
    "context",
]
"""The turn's six events, each reflecting a line, and D212's live ones, which reflect a reply
still streaming, of any role (its text, its reasoning, a tool call being written), and Zeus's
context for the status line."""


# -- values ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """What the turn streams: the live view of one turn, never the record."""

    kind: EventKind
    seq: int
    """The session-log line the event reflects; 0 for a live one, which reflects none yet."""
    payload: str | ToolCall | ToolLine | Delegation | LiveEvent


def live_event(event: LiveEvent) -> Event:
    """A role's live update as a turn event: `text` (a new attempt clears the
    text, as `Restarted`), `thought` or `tool_started`."""
    match event.update:
        case ContextUpdate():
            kind: EventKind = "context"
        case ThoughtUpdate():
            kind = "thought"
        case ToolUpdate():
            kind = "tool_started"
        case TextUpdate() | Restarted():
            kind = "text"
    return Event(kind, 0, event)


@dataclass(frozen=True)
class LoopConfig:
    """The loop's three bounds. `0` disables any of them."""

    max_result_tokens: int
    """`zeus.max_cumulative_result_tokens`, resolved on the zeus rung's window."""
    max_tool_calls: int = int(limit("zeus.max_tool_calls").value)
    """Per turn."""
    repeat: int = ZEUS_REPEAT_DEFAULT
    locate_fanout: int = LOCATE_FANOUT
    agents_parallel: int = int(limit("agents.max_parallel").value)
    """`spawn_agent` calls in one response, counted apart from `locate`'s."""
    inline_bytes: int = INLINE_BYTES_DEFAULT
    """What a navigation answer shows inline, the rest behind its handle."""
    result_lines: int = RESULT_LINES_DEFAULT
    """The lines a line-bounded answer shows (`find`, `tools`)."""

    def limits(self) -> dict[str, int]:
        """The numbers Zeus's pipeline reads, by the keys its rules name."""
        return {
            "zeus.max_tool_calls": self.max_tool_calls,
            "zeus.max_cumulative_result_tokens": self.max_result_tokens,
            "zeus.repeat": self.repeat,
            "zeus.locate_fanout": self.locate_fanout,
            "agents.max_parallel": self.agents_parallel,
            "zeus.inline_bytes": self.inline_bytes,
            "zeus.result_lines": self.result_lines,
        }

    @staticmethod
    def from_limits(
        resolved: Sequence[ResolvedLimit],
        *,
        repeat: int = ZEUS_REPEAT_DEFAULT,
        agents_parallel: int = int(limit("agents.max_parallel").value),
    ) -> LoopConfig:
        """From `EngineConfig.resolved_limits`: the register's rows, on the zeus rung, and the
        harness-wide `agents.max_parallel` (`EngineConfig.harness_limit`)."""
        return LoopConfig(
            max_result_tokens=resolved_value(
                resolved, "zeus", "zeus.max_cumulative_result_tokens"
            ),
            max_tool_calls=resolved_value(resolved, "zeus", "zeus.max_tool_calls"),
            repeat=repeat,
            agents_parallel=agents_parallel,
        )


# -- the frame and the generation ----------------------------------------------------------


def frame_text(
    day: date, head: str | None, branch: str | None, status: str | None = None
) -> str:
    """D99's one harness line: today's date, and git's head and branch when there is one; then,
    while the session has open work, where it stands (`work_progress`)."""
    parts = [day.isoformat()]
    if head:
        parts.append(f"head {head[:12]}" + (f" on {branch}" if branch else ""))
    line = " · ".join(parts)
    return f"{line}\n{status}" if status else line


def frame_due(
    state: SessionState, day: date, branch: str | None, status: str | None = None
) -> bool:
    """A frame opens the session's first user turn, the first one after the date changes, the
    first one after a cut, which may have taken the last frame into its block, and the first
    one after Zeus's view changed workspace. `branch` is the view's branch,
    and it names the workspace: git never checks one branch out in two worktrees, so a move
    between the checkout and a worktree, or between worktrees, changes it. Never because
    `head` moved: every `coder` call moves it, and a frame per call would be noise. And the
    first one after the open work's status changed (`status`): a run moved it,
    or its design or plan was edited, so Zeus reads where it stands before he decides."""
    frames = state.frames()
    if not frames:
        return True
    last = frames[-1]
    return (
        last.date != day.isoformat()
        or last.seq < state.boundary
        or last.branch != branch
        or last.status != status
    )


def _prefix_tokens(prefix: Prefix) -> int:
    """The local estimate of what every request opens with: the system message, and its tools
    as every request's are weighed (`est_tools`), so a failed call on any rung is estimated by
    one rule."""
    return est_text(text_of(prefix.system())) + est_tools(prefix.tools)


def workspace_generation(state: SessionState) -> int:
    """D98's generation, as far as the log can tell it: it moves with each `coder` episode and
    each view the invalidation notices reported stale. Derived from the log, so replay agrees
    with live. The log cannot tell an episode that landed nothing from one that landed
    packets, so this default counts every episode — a refusal detector that under-counts is
    the safe one — and the engine passes the workspace's own counter where it has one, which
    is what lets BL-56's repeated delegation be the same key."""
    stale = sum(
        1
        for t in state.turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.STALE_VIEW
    )
    return len(state.episodes()) + stale


# -- the turn -------------------------------------------------------------------------------


class Compacting(Protocol):
    """What the loop asks of compaction (`context/compaction/session.SessionCompaction`):
    a cut before every request, one after a provider's overflow, and each
    answer's usage, which is the cache observation and the gap history a cut is decided on.
    `before` and `overflowed` may raise `TurnTooLarge`."""

    def before(self, session: Session, prefix_tokens: int) -> Compaction: ...

    def overflowed(self, session: Session, prefix_tokens: int) -> Compaction: ...

    def observed(self, usage: Usage) -> None: ...

    def model_changed(self) -> None: ...


class ZeusLoop:
    """One session's conversation loop.

    `compaction`, when given, runs before every request is assembled, and again, once, after a
    provider refuses one as too long. It holds what a cut needs that this loop does not: the
    endpoint's window and
    cache mode, the residue model, the run records. `head` returns git's head and branch for
    the frame; `today` the frame's date; `status` the open work's status, which the frame
    carries.
    `generation` is D98's workspace generation; `reconcile` answers a `coder` call a crash
    left waiting, from its run's journal; `tell_ended` tells how a run
    answered `PAUSED` ended since, resumed, in the same place: before anything else.

    `model` is the catalog model the zeus rung runs on (`[models.<name>]`): the name
    the ledger prices a call by. The rung's own name is not a price key — a rung can run on
    two models in one session — so a `ledger` without it is refused, and `reprefix` moves it
    with the prefix when `/rung zeus` switches the model."""

    def __init__(
        self,
        session: Session,
        prefix: Prefix,
        client: ModelClient,
        params: RetryParams,
        tools: ZeusTools,
        config: LoopConfig,
        *,
        interrupts: Interrupts | None = None,
        ledger: Ledger | None = None,
        model: str | None = None,
        today: Callable[[], date] = date.today,
        head: Callable[[], tuple[str | None, str | None]] = lambda: (None, None),
        status: Callable[[], str | None] = lambda: None,
        compaction: Compacting | None = None,
        reconcile: Callable[[str], str] | None = None,
        tell_ended: Callable[[], object] | None = None,
        generation: Callable[[SessionState], int] = workspace_generation,
        policy: RolePolicy | None = None,
        turn_ended: Callable[[], None] | None = None,
        turn_woken: Callable[[], None] | None = None,
        live: LiveSink | None = None,
        window: int | None = None,
        hooks: RoleHooks | None = None,
        unattended: bool = False,
        flagged: Callable[[WorkspaceMode], str | None] | None = None,
        attach: Callable[[CheckedFile], Minted] | None = None,
        vision: Vision | None = None,
    ) -> None:
        self._session = session
        self._prefix = prefix
        self._prefix_tokens = _prefix_tokens(prefix)
        self._client = client
        self._params = params
        self._tools = tools
        self._config = config
        self._sink = ZeusSink(session)
        self._delegations = Delegations(session, tools.box)
        self._opened: dict[str, OpenedEpisode | Answer] = {}
        """`coder` calls opened and not yet run, by call id: `_delegate` to `_delegated`."""
        self.interrupts = interrupts if interrupts is not None else Interrupts()
        if ledger is not None and not model:
            raise HarnessBug(
                "a ledger charges the catalog model the zeus rung runs on (D128); pass model="
            )
        self._ledger = ledger
        self._model = model or ""
        self._today = today
        self._head = head
        self._status = status
        """The open work's status for the frame (`work_progress.status_text`), or None."""
        self._compaction = compaction
        self._reconcile = reconcile
        self._tell_ended = tell_ended
        self._generation = generation
        self._turn_ended = turn_ended
        """Called once each user turn's conversation ends, however it ends: the turn's code
        snapshots' `after`."""
        self._turn_woken = turn_woken
        """Called as a wake turn begins: what it writes is its person's turn's next snapshot
        part."""
        self._speaker = None if live is None else Speaker(live, "zeus")
        self._attach = attach
        # What the rung's model takes now: a part sent before a `/rung` switch to a model that
        # takes less is put as a line, never sent to be refused.
        self._vision = vision
        """Keeps a file the person sent in the session's store, under its handle, and says how
        his rung is sent it (`Reader.attach`); None, and a turn with one is a bug."""
        """His replies, shown as they stream; never journaled."""
        self._shown = live if window is not None else None
        self._window = window or 0
        """His rung's context window: what the status line measures his context against."""
        with session.lock:
            self._read_before = session.next_seq
        """The first seq the last request the model answered did not carry: what Zeus has
        read. From the open, everything already on the log counts as read,
        so a notice from before it never wakes (decision 4); written and read on the
        conversation's worker."""
        default = role("zeus").tool_defaults
        if policy is None and default is None:
            raise HarnessBug("Zeus has no default tool policy")
        self._policy = cast("RolePolicy", policy if policy is not None else default)
        """Which tools, under which names (`[tools.zeus]`). The prefix lists
        what it enables, so the two are chosen together (`EngineConfig.tool_policy`)."""
        self._hooks = hooks or RoleHooks.none("zeus")
        """His hooks: his turn's points here, his calls' through his dispatcher."""
        self._loop_hooks = RoleLoopHooks(
            self._hooks, lambda shown: self._policy.registry_name(shown)
        )
        self._unattended = unattended
        self._flagged = flagged
        """The person's `--in-place` or `--worktree` as their turn opens:
        `zeus/worktree.WorktreeMoves.follow_flag`, what it moved said as a notice."""
        """Whether nobody answers at the end of his turn (a driven session): a stop hook's
        input says so."""

    @property
    def read_before(self) -> int:
        """`_read_before`: a notice at or after it reached no answered request."""
        return self._read_before

    # -- the person's side ------------------------------------------------------------------

    def type(self, text: str, workspace: WorkspaceMode | None = None) -> None:
        """Input while a turn runs: queued, and delivered at the next turn boundary."""
        self.interrupts.type(text, workspace)

    def interrupt(self, hard: bool = False) -> None:
        self.interrupts.interrupt(hard)

    def reprefix(self, prefix: Prefix, model: str, policy: RolePolicy | None = None) -> None:
        """`/rung zeus` moved the model, so the prefix is reassembled; from the next
        call, with its sha on the session log. `model` is the catalog name now behind the
        rung, and the next call is charged at its price row — a call in flight has already
        been charged at the old one. `policy` is `[tools.zeus]` for that model (its `models`
        section), the policy the new prefix lists; the next turn dispatches
        it. Without one the policy stays, so the prefix must list what it enables."""
        self._prefix = prefix
        self._prefix_tokens = _prefix_tokens(prefix)
        self._model = model
        if policy is not None:
            self._policy = policy
        if self._compaction is not None:
            self._compaction.model_changed()

    # -- a turn -----------------------------------------------------------------------------

    def turn(
        self, text: str | None = None, workspace: WorkspaceMode | None = None
    ) -> Iterator[Event]:
        """One user turn, streamed; then every input typed while it ran, in order, each as the
        next turn. A stop ends the stream and leaves what is still queued queued.

        With no `text`, a **driver's** turn: the oldest queued word, and only it,
        as one turn. A driver that queues the person's words with `type()` and calls this
        once per word delivers them in the order they were typed, and can do what it must
        between them (the engine restores its ledger before each). The stop is the driver's:
        this never clears it, and while one is in force it delivers nothing and leaves the
        queue as it was. A stop between two words therefore holds; the driver clears it with
        `interrupts.begin_turn()` when the person starts a turn."""
        if text is None:
            yield from self._queued_turn()
            return
        self.interrupts.begin_turn()
        yield from self._deliver([*self._drain(), Words(text, workspace)], more=True)

    def wake(self) -> Iterator[Event]:
        """A turn a job's exit opens: no user line, since the harness never
        puts words in the person's mouth; the `JOB_EXITED` notices already on the log open it.
        Otherwise a turn like any other: repaired, gated by the ledger, bounded,
        and snapshotted at its end, as its person's turn's next part."""
        if self._turn_woken is not None:
            self._turn_woken()
        yield from self._repair()
        self._record_prefix()
        said = self._submitted("", wake=True)
        if isinstance(said.decided, Deny):
            yield Event("error", self._session.state.last_seq, self._refused(said))
            self._end_turn()
            return
        self._noted(said.note)
        try:
            yield from self._converse()
        except BaseException as exc:
            try:
                self._end_turn()
            except Exception as also:
                exc.add_note(f"the turn's code snapshot also failed: {also!r}")
            raise
        self._end_turn()

    def _queued_turn(self) -> Iterator[Event]:
        if self.interrupts.stopping():
            return
        words = self.interrupts.next_input()
        if words is None:
            return
        yield from self._deliver([words], more=False)

    def _deliver(self, pending: list[Words], *, more: bool) -> Iterator[Event]:
        """Each of `pending` as a turn, in order, and with `more` every word typed meanwhile
        after them. A stop is checked before each word is opened, so one that lands during
        the repair, or between two words, delivers nothing more: a word is opened only to
        be answered."""
        try:
            yield from self._repair()
            self._record_prefix()
            while pending and not self.interrupts.stopping():
                said = self._submitted(pending[0].text, wake=False)
                if isinstance(said.decided, Deny):
                    # A hook refused the words: nothing is recorded or sent.
                    del pending[0]
                    yield Event("error", self._session.state.last_seq, self._refused(said))
                    continue
                self._open(pending[0])
                self._noted(said.note)
                # Delivered once its user line is on the log, and not before.
                del pending[0]
                try:
                    yield from self._converse()
                except BaseException as exc:
                    # A turn that wrote and then failed or was stopped still gets its `after`,
                    # and a snapshot that fails too is a note on why the turn ended, never
                    # raised in its place (as `given_back_on_failure` treats an abandon).
                    try:
                        self._end_turn()
                    except Exception as also:
                        exc.add_note(f"the turn's code snapshot also failed: {also!r}")
                    raise
                self._end_turn()
                if more and not self.interrupts.stopping():
                    pending.extend(self._drain())
        finally:
            # A stop pauses the conversation, and an error or a reader that stops reading
            # ends the turn; in every case the words not yet delivered go back to the front
            # of the queue, older than anything typed since, in the order they were typed
            self.interrupts.requeue(pending)

    def _end_turn(self) -> None:
        if self._turn_ended is not None:
            self._turn_ended()

    def _drain(self) -> list[Words]:
        out: list[Words] = []
        while (nxt := self.interrupts.next_input()) is not None:
            out.append(nxt)
        return out

    def _record_prefix(self) -> None:
        """The prefix's sha to the log, once per prefix: at session start, and after D128's
        switch. A resumed session whose prefix is unchanged writes nothing."""
        state = self._session.state
        wanted = sha256_hex(render(self._prefix.line()))
        last = None
        for event, record in state.lines():
            if isinstance(record, OtherLine) and record.kind == "prefix":
                last = event.payload_sha
        if last != wanted:
            record_prefix(self._session, self._prefix)
        else:
            # A session begun in older versions has the line but not the texts.
            store_texts(self._session, self._prefix)

    def _repair(self) -> Iterator[Event]:
        """Answer every call a crash left waiting, before anything else is said. A
        `coder` call's answer is its run's reconciliation, which only the journal knows."""
        state = self._session.state
        for call_id in state.pending_calls():
            call = state.call(call_id)
            if call is None:
                raise HarnessBug(f"pending call {call_id} has no call")
            episode = next((e for e in state.episodes() if e.opened_at_seq == call.seq), None)
            tool = self._registered(call.tool)
            if tool is not None and tool.streams and episode is not None:
                # The episode line is written before the run does any work, so an open run
                # may have done nothing or everything; only its journal can say which.
                if self._reconcile is not None:
                    why = "the session ended while the run was open"
                    text = Interrupted(call.tool, why).render()
                    text = f"{text}\n{self._reconcile(episode.run_id)}"
                else:
                    why = (
                        f"the session ended while run {episode.run_id} was open; what it did "
                        "is unknown until its journal is reconciled"
                    )
                    text = Interrupted(call.tool, why).render()
                line = self._session.record_tool(call_id, text, ran=True, run_id=episode.run_id)
            else:
                why = "the session ended before its result was recorded; what it did is unknown"
                # `ran: false` is a refusal's, and an interrupted command was not refused:
                # it may have reached the network, so a later fork's effects notice must still
                # list it. A question nobody answered stays out of the person's words.
                ran = tool is not None and tool.spec.mutates == "environment"
                line = self._session.record_tool(
                    call_id, Interrupted(call.tool, why).render(), ran=ran
                )
            yield Event("tool_result", line.seq, line)
        if self._tell_ended is not None:
            # After every call is answered: a notice never sits between a call and its result.
            self._tell_ended()

    def _open(self, words: Words) -> None:
        """A user turn's lines: the frame when one is due, then what the person said."""
        session = self._session
        day = self._today()
        head, branch = self._head()
        status = self._status()
        if frame_due(session.state, day, branch, status):
            session.append(
                FrameLine(
                    day.isoformat(),
                    frame_text(day, head, branch, status),
                    head,
                    branch,
                    status=status,
                )
            )
        minted = ()
        if words.attachments:
            if self._attach is None:
                raise HarnessBug("a turn carried attachments to a loop that cannot keep them")
            minted = tuple(self._attach(c) for c in words.attachments)
        session.record_user(words.text, words.workspace, minted)
        if words.workspace is not None and self._flagged is not None:
            moved = self._flagged(words.workspace)
            if moved:
                session.record_notice(moved_by_flag(moved))

    def _converse(self) -> Iterator[Event]:
        """Call the model and dispatch what it asks for until a response makes no call."""
        session = self._session
        turn = self._open_turn()
        while not self.interrupts.stopping():
            if self._ledger is not None and not self._ledger.may_start():
                # The zeus rung's view names the cap that stopped it.
                cap = self._ledger.stopped_by() or "the session's spend cap is reached"
                why = f"BUDGET_CAP: {cap}"
                yield Event("error", session.state.last_seq, why)
                return
            completion = self._send()
            if isinstance(completion, str):
                yield Event("error", session.state.last_seq, completion)
                return
            if not completion.tool_calls:
                answer = session.record_assistant(completion.text)
                yield Event("answer", answer.seq, completion.text)
                if completion.stop_reason is StopReason.MAX_TOKENS:
                    yield Event("error", answer.seq, "the answer was cut at max_tokens")
                    return
                if not self._go_on(completion.text, turn):
                    return
                continue
            yield from self._respond(completion, turn)
            halted = self._halted(completion, turn)
            if halted is not None:
                # A hook ended his tool use: the turn ends as a stop ends it.
                yield Event("error", session.state.last_seq, halted)
                return

    def _open_turn(self) -> _Turn:
        """A user turn's dispatcher, offering what the prefix lists, with its counters and its
        `bound` lines starting afresh (the bounds are per turn)."""
        facts = ResponseFacts()
        self._opened.clear()
        dispatcher = zeus_dispatcher(
            self._tools,
            self._sink,
            facts,
            offered=[t.name for t in self._prefix.tools],
            limits=self._config.limits(),
            generation=self._generation,
            cancel=self.interrupts.cancel,
            executors={name: self._delegated for name in _streaming()},
            policy=self._policy,
            hooks=self._hooks,
        )
        return _Turn(dispatcher, facts, BoundLines(self._session.append))

    # -- his hooks -------------------------------------------------------------------

    def _submitted(self, text: str, *, wake: bool) -> Combined:
        """`prompt_submitted`, before the turn's words are recorded: a `Deny` refuses them."""
        return self._hooks.fire(
            HookPoint.PROMPT_SUBMITTED, PromptSubmitted(prompt=text, wake=wake)
        )

    @staticmethod
    def _refused(said: Combined) -> str:
        denied = cast("Deny", said.decided)
        return HookDenied(said.by or "hooks", denied.reason, denied.do).render()

    def _noted(self, text: str | None) -> None:
        """A hook's words as a harness notice (trigger 9), which `assemble` holds until any
        waiting call's result is in."""
        if text:
            self._session.record_notice(hook_noted(text))

    def _go_on(self, reply: str, turn: _Turn) -> bool:
        """`before_stop` at his answer: whether his turn goes on, his hooks' words a notice
        after it. The bound is the turn's: a crash ends a turn, and the next starts afresh."""
        stop = BeforeStop(
            reply=reply,
            tools_open=turn.dispatcher.stopped is None,
            unattended=self._unattended,
        )
        said = at_stop(self._hooks, stop, used=turn.continues)
        words = said.words
        if words is None:
            self._noted(said.note)
            return False
        turn.continues += 1
        self._noted(words)
        return True

    def _halted(self, completion: Completion, turn: _Turn) -> str | None:
        """`after_batch` once a response's calls are answered, its note a notice; and why a
        hook ended his tool use, at a call or after the batch, or None."""
        said = self._loop_hooks.after_batch(tuple(completion.tool_calls))
        self._noted(said.note)
        if said.halt is not None:
            return HookHalted(said.halt.by, said.halt.reason).render()
        stopped = turn.dispatcher.stopped
        if stopped is not None and isinstance(stopped.failure, HookHalted):
            return stopped.failure.render()
        return None

    def _charge_failed(self, held: int, exc: ClientError) -> str:
        """A call that failed after it was sent, charged what it is known to have cost
        (`client/charge.py`): the request's input, estimated here as the
        prefix (the system message and its tool schemas) and `held`, what `assemble` counted
        of the rest, tool calls and images included, at P-2's local rate.

        Returns what the turn's error must add: nothing, or that the charge's `spend` line
        could not be written (a full disk). The call's own error is the one the person is
        told, and the disk's is told beside it, never in its place. The ledger counted the
        charge before it wrote the line, so the cap still holds it in this process; only the
        line, and so a restart's memory of it, is lost."""
        ledger = self._ledger
        if ledger is None:
            return ""
        try:
            charge_unanswered(
                lambda usage: ledger.charge_usage("", usage, self._model),
                lambda: self._prefix_tokens + held,
                exc,
            )
        except OSError as unwritten:
            return (
                f"; and its estimated charge could not be written to the session log "
                f"({type(unwritten).__name__}: {unwritten})"
            )
        return ""

    def _show_context(self, usage: Usage) -> None:
        """His context against his window, for the status line."""
        if self._shown is not None:
            used = usage.prompt_tokens + usage.completion_tokens
            update = ContextUpdate(used, self._window, self._model)
            self._shown(LiveEvent("zeus", None, "context", update))

    def _send(self) -> Completion | str:
        compaction = self._compaction
        if compaction is not None:
            try:
                compaction.before(self._session, self._prefix_tokens)
            except TurnTooLarge as exc:
                # §5.12 step 12b: nothing is sent. The cut, if one committed, stays.
                return f"TURN_TOO_LARGE: {exc}"
        refusals = 0
        while True:
            # One read of the log: what is assembled is every line before `carried`.
            with self._session.lock:
                carried = self._session.next_seq
                assembly = assemble(self._session.state, self._session.blob)
            with speaking(self._speaker) as live:
                messages = assembly.messages
                if self._vision is not None:
                    messages = within_sight(messages, self._vision)
                request = ChatRequest(
                    messages=(self._prefix.system(), *messages),
                    params=self._params,
                    tools=self._prefix.tools,
                    session_key=self._session.session_id,
                    cancel=self.interrupts.cancel,
                    live=live,
                )
                halt = self._loop_hooks.before_request(request)
                if halt is not None:
                    return HookHalted(halt.by, halt.reason).render()
                try:
                    completion = self._client.send(request)
                    break
                except RequestCancelled as exc:
                    unwritten = self._charge_failed(assembly.tokens, exc)
                    if exc.attempts == 0:
                        return "CANCELLED: the request was stopped before it was sent"
                    # In flight or in a backoff alike, and never how many attempts: a retry is
                    # silent to the person.
                    return f"CANCELLED: the request was stopped before its reply{unwritten}"
                except ContextOverflow as exc:
                    unwritten = self._charge_failed(assembly.tokens, exc)
                    refused = f"{type(exc).__name__}: {exc}"
                    if compaction is None:
                        return f"{refused}{unwritten}"
                    if refusals >= OVERFLOW_RETRIES:
                        return f"{refused} (after {refusals} compact-and-retry){unwritten}"
                    # One cut on the provider's word, then the same turn again.
                    refusals += 1
                    try:
                        done = compaction.overflowed(self._session, self._prefix_tokens)
                    except TurnTooLarge as too_large:
                        return f"{refused}; TURN_TOO_LARGE: {too_large}{unwritten}"
                    if done.cut is None:
                        why = done.aborted or "compaction had nothing left to cut"
                        return f"{refused}; {why}{unwritten}"
                except ClientError as exc:
                    unwritten = self._charge_failed(assembly.tokens, exc)
                    return f"{type(exc).__name__}: {exc}{unwritten}"
        return self._received(completion, carried)

    def _received(self, completion: Completion, carried: int) -> Completion | str:
        """A reply's bookkeeping: what Zeus has now read, his context, the cut's observation
        and the charge; and the reply, or why nothing of it is recorded."""
        compaction = self._compaction
        # A notice that landed while this request was in flight is after `carried`: unread.
        self._read_before = carried
        self._show_context(completion.usage)
        if compaction is not None:
            compaction.observed(completion.usage)
        if self._ledger is not None:
            self._ledger.charge_usage("", completion.usage, self._model)
        unnamed = [c.id or "(no id)" for c in completion.tool_calls if not c.name]
        if unnamed:
            # A call that names no tool cannot be recorded (the log refuses it) nor answered
            # `UNKNOWN_TOOL`, which names what was asked. The response is malformed, as a body
            # that is not a completion is: nothing of it is recorded, and the turn ends.
            return (
                "MALFORMED_RESPONSE: the model sent a tool call with no name "
                f"({', '.join(unnamed)}); nothing of this response was recorded"
            )
        return completion

    # -- a response -------------------------------------------------------------------------

    def _respond(self, completion: Completion, turn: _Turn) -> Iterator[Event]:
        calls = self._labelled(completion.tool_calls)
        delegations = [c for c in calls if (t := self._registered(c.name)) and t.streams]
        # The fold keeps a turn to one `coder` call, so a
        # second one in a response is its own assistant line, answered after the first.
        extra = {id(c) for c in delegations[1:]}
        first = [c for c in calls if id(c) not in extra]
        yield from self._exchange(completion.text, first, turn)
        for call in delegations[1:]:
            yield from self._exchange("", [call], turn, excess_coder=True)

    def _labelled(self, calls: Sequence[ToolCall]) -> list[ToolCall]:
        """A call's id is the transport's token for pairing it with its result, and the log
        refuses one that is empty or already used. Some servers send none, or number each
        response's calls from zero, so such a call is given a harness id before it is
        recorded. That relabels the call (invariant 19 permits it); its name and arguments
        are recorded exactly as sent."""
        state = self._session.state
        taken = {c.id for c in calls if c.id}
        kept: set[str] = set()
        out: list[ToolCall] = []
        n = 0
        for c in calls:
            if c.id and c.id not in kept and state.call(c.id) is None:
                kept.add(c.id)
                out.append(c)
                continue
            n += 1
            while f"call_h{n}" in taken or state.call(f"call_h{n}") is not None:
                n += 1
            cid = f"call_h{n}"
            taken.add(cid)
            kept.add(cid)
            out.append(c.model_copy(update={"id": cid}))
        return out

    def _exchange(
        self, text: str, calls: list[ToolCall], turn: _Turn, *, excess_coder: bool = False
    ) -> Iterator[Event]:
        session = self._session
        # No call carries a purpose line: its findings alone are read beside it.
        findings, bad = self._findings(calls, [split_meta(c.arguments)[1] for c in calls])
        assistant = session.record_assistant(text, calls, findings=findings)
        if text:
            yield Event("thinking", assistant.seq, text)
        for call in calls:
            yield Event("tool_call", assistant.seq, call)
        turn.facts.excess_coder = excess_coder
        turn.facts.bad_findings = bad
        pending = turn.dispatcher.admit(calls, at=assistant.seq)
        self._bound_lines(pending, turn)
        yield from self._dispatch(pending, assistant.seq, turn)

    def _findings(
        self, calls: Sequence[ToolCall], raws: Sequence[object]
    ) -> tuple[dict[str, str], dict[str, str]]:
        """The findings a response writes — for each earlier result it relies on, what that
        result established — and, per call, why its own were refused. A finding the fold
        would refuse is never recorded: the fold is the record's integrity."""
        state = self._session.state
        good: dict[str, str] = {}
        bad: dict[str, str] = {}
        for call, raw in zip(calls, raws, strict=True):
            if raw is None:
                continue
            if not isinstance(raw, Mapping):
                bad[call.id] = "an object: call id to what that result established"
                continue
            mine: dict[str, str] = {}
            for key, value in cast("Mapping[object, object]", raw).items():
                cid = str(key)
                if not is_line(value):
                    bad[call.id] = f"the finding for {cid} says something"
                elif state.result(cid) is None:
                    bad[call.id] = f"{cid} has no result to draw a finding from"
                elif state.finding(cid) is not None or cid in good or cid in mine:
                    bad[call.id] = f"{cid} already has a finding"
                else:
                    mine[cid] = cast("str", value)
            if call.id not in bad:
                good.update(mine)
        return good, bad

    def _bound_lines(
        self, pending: Sequence[Pending], turn: _Turn, *, notices: bool = True
    ) -> None:
        """The turn's `bound` lines for what admit (or, without `notices`,
        dispatch) decided, in call order, each resource and outcome once a turn."""
        repeat = self._config.repeat
        for p in pending:
            decided = p.decided
            if isinstance(decided, Refuse) and isinstance(decided.failure, BoundReached):
                bound = decided.failure
                key = self._key(p) if bound.resource == "repetition" else None
                turn.lines.refused(bound, key)
            elif notices and p.notes:
                # Repetition is the only rule of Zeus's that notes a call. Its streak runs
                # 1, 2,... within a turn, so the turn's first notice, the one its line
                # records, is always at the Nth identical call.
                turn.lines.noticed(repeat, repeat, self._key(p))

    def _key(self, pending: Pending) -> RepetitionKey:
        return repetition_key(pending, self._generation(self._session.state))

    # -- dispatch ---------------------------------------------------------------------------

    def _dispatch(self, pending: Sequence[Pending], at: int, turn: _Turn) -> Iterator[Event]:
        """By `ToolSpec.parallel` (`Dispatcher.schedule`): a run of `any` and `capped` calls
        together, anything else alone, in the order the calls were made; a call admit
        answered is answered in its place. A stop is honoured between calls: what has not
        started is answered `CANCELLED`, and what has started finishes."""
        for batch in turn.dispatcher.schedule(pending):
            head = batch.pending[0]
            if head.decided is not None:
                yield self._answered(turn.dispatcher.execute(head))
            elif batch.parallel == "any":
                yield from self._group(batch.pending, turn)
            else:
                yield from self._alone(head, at, turn)

    def _group(self, group: Sequence[Pending], turn: _Turn) -> Iterator[Event]:
        """ "Together" is a group: its calls run one after another on the thread that runs
        the turn, but for those whose spec says `threads` (`spawn_agent`), which run at once
        beside them, each on its own (`Dispatcher.execute_group`). Answers
        come in call order. The read bound is asked of the whole group before any of it
        runs."""
        dispatcher = turn.dispatcher
        if self.interrupts.stopping():
            for p in group:
                yield self._answered(dispatcher.cancel(p))
            return
        dispatched = dispatcher.dispatch(group)
        self._bound_lines(dispatched, turn, notices=False)
        for recorded in dispatcher.execute_group(dispatched, stopping=self.interrupts.stopping):
            yield self._answered(recorded)

    def _alone(self, pending: Pending, at: int, turn: _Turn) -> Iterator[Event]:
        dispatcher = turn.dispatcher
        if self.interrupts.stopping():
            yield self._answered(dispatcher.cancel(pending))
            return
        (pending,) = dispatcher.dispatch((pending,))
        self._bound_lines((pending,), turn, notices=False)
        if pending.decided is not None or pending.tool not in _streaming():
            yield self._answered(dispatcher.execute(pending))
            return
        yield from self._delegate(pending, at, dispatcher)

    def _delegate(
        self, pending: Pending, at: int, dispatcher: Dispatcher[ToolLine]
    ) -> Iterator[Event]:
        """A streaming call: the episode opens and its line is written, the delegation is
        streamed, and then the dispatcher runs it through `_delegated`, which waits for the
        report. A refusal before anything opened is the call's answer."""
        if pending.arguments is None:
            raise HarnessBug(f"call {pending.call.id} was dispatched unvalidated")
        opened = self._delegations.open(
            pending.arguments,
            opened_at_seq=at,
            cancel=self.interrupts.cancel,
        )
        self._opened[pending.call.id] = opened
        if not isinstance(opened, OpenedEpisode):
            yield self._answered(dispatcher.execute(pending))
            return
        asked = False
        try:
            yield Event("delegation", opened.line.seq, opened.delegation)
            asked = True
            line = dispatcher.execute(pending)
        finally:
            if not asked:
                # The stream ended at the delegation: the report that would release what
                # `open` took will never be asked for.
                del self._opened[pending.call.id]
                opened.run.abandon()
        yield self._answered(line)

    def _delegated(self, pending: Pending) -> Answer:
        """The executor of a streaming tool (`RoleExecutors`): the run `_delegate` opened,
        run to its report, or the refusal it was opened with."""
        opened = self._opened.pop(pending.call.id)
        if isinstance(opened, OpenedEpisode):
            return Delegations.finish(opened)
        return opened

    def _registered(self, shown: str) -> Tool | None:
        """The tool a call names, by the name the model is shown (`[tools.zeus]`'s
        `shown_as`), or None when the policy enables none so named."""
        name = self._policy.registry_name(shown)
        return BUILTINS.get(name) if name is not None else None

    @staticmethod
    def _answered(line: ToolLine) -> Event:
        return Event("tool_result", line.seq, line)


# -- helpers --------------------------------------------------------------------------------


@dataclass
class _Turn:
    """One user turn's dispatcher, what it is told of each response, and its `bound`
    lines."""

    dispatcher: Dispatcher[ToolLine]
    facts: ResponseFacts
    lines: BoundLines
    continues: int = 0
    """The hooks' `Continue`s taken this turn: what `hooks.max_continues` bounds."""


def _streaming() -> frozenset[str]:
    """The registered tools that stream (`Tool.streams`): `coder`, which the loop runs."""
    return frozenset(n for n in BUILTINS.names() if (t := BUILTINS.get(n)) and t.streams)
