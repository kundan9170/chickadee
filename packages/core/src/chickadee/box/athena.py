"""Athena in the task loop: one
episode's request, beside the worker's `box/worker.py`.

Her request is the whole episode: one `tools.loop.ask` over her `JournaledTranscript`, from
her head (`boxctx/athena.athena_head`) to her final reply, which is her report to Zeus and is
never parsed. Every decision the harness acts on is one of her tool calls: `plan`,
`delegate` and `review` over the desk (`box/desk.TaskDesk`); the reads, `atlas`, `recall` and
`note`; the edit tools, always offered and refused by the desk unless she holds a task;
her shell. Her writing calls are recorded under the task she holds, or under none,
by the desk's `around`.

Her caps refuse past their bound rather than end her tool use, and spare the task tools
(`roles.ATHENA_TASK`); at each task tool's answer her request scope is opened again, so each
stretch of her work between two of them has the whole allowance and a long investigation
never strands an open task. Each model call is charged through her rung's ledger and gated by
its `may_start` before it is made, and lined by `BoxSink`, at the task she holds.

`AthenaPorts` moved here from `planner/client.py` (one home), which imports it back until
The task loop retires it. The packet session's own notes and seen-files ports stay
there: they keep `AthenaSession`'s state, which the task loop does not have. Her task-loop note
is `Notes`, the
same `note` line through `boxctx/athena.write_note`.

Composed by `box/compose.compose_episode`; `box/loop.TaskLoop` runs her.

**Concurrency.** One `Athena` per episode, on its thread; the worker runs inside her
`delegate` and `review` calls, on the same thread.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final, Literal

from chickadee.adapters.base import (
    BOX_DESCRIPTIONS,
    SHARED_TEMPLATES,
    load_template,
    tool_descriptions,
)
from chickadee.box.desk import TaskDesk
from chickadee.box.resume import Owed
from chickadee.box.rung import BoxRung
from chickadee.box.sink import BoxServed, BoxSink
from chickadee.box.tells import RoundToken, Tells, to_athena
from chickadee.boxctx.athena import NumberedLog, write_note
from chickadee.boxctx.conversation import BoxConversation, ContextFull, JournaledTranscript
from chickadee.boxctx.sent import SentRequests
from chickadee.client.live import LiveSink, Speaker
from chickadee.client.request import Message
from chickadee.context.homes import Now
from chickadee.context.results import ToolFailure
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.loop import RoleLoopHooks
from chickadee.hooks.payloads import BeforeStop, OpenTask
from chickadee.hooks.points import HookPoint
from chickadee.hooks.stop import at_stop, continued
from chickadee.hooks.verdicts import Continue
from chickadee.journal.port import JournalPort
from chickadee.schema import RetryParams
from chickadee.schema.adapter import Completion, ParseError, ToolCall
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import (
    ExecRequestPort,
    FileView,
    HandleStore,
    Navigator,
    RecallPort,
    ToolContext,
)
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.loop import AskResult, End, Reply, ask, kept
from chickadee.tools.meta import shown
from chickadee.tools.policy import RolePolicy
from chickadee.tools.validate import resident

__all__ = [
    "Athena",
    "AthenaAnswer",
    "AthenaEnd",
    "AthenaParts",
    "AthenaPorts",
    "Notes",
    "instructions",
]

ROLE: Final = "athena"
REQUEST: Final = "episode"
"""What her `model_call` lines name as the request: the whole episode, one request."""


def instructions() -> str:
    """Her task-loop instructions, the head's first part: tied to no model family."""
    return load_template(SHARED_TEMPLATES, "athena.md")


@dataclass(frozen=True)
class AthenaPorts:
    """What her tools touch (`roles.ATHENA_CEILING`). An absent port is a tool not offered."""

    files: FileView
    navigator: Navigator
    handles: HandleStore | None = None
    journal: RecallPort | None = None
    exec_request: ExecRequestPort | None = None


class Notes:
    """Her `design_notes` port in the task loop: one `note` line on the session log under the
    approved work's sha, durable before it is shown (`boxctx/athena.write_note`). The note is
    already in her conversation as her own call, so nothing else holds it."""

    def __init__(self, log: NumberedLog, design_sha: str, run_id: str, cap: int) -> None:
        self._log = log
        self._design_sha = design_sha
        self._run_id = run_id
        self._cap = cap

    def note(self, text: str) -> str | ToolFailure:
        written = write_note(self._log, self._design_sha, self._run_id, self._cap, text)
        return written if isinstance(written, ToolFailure) else written.id


@dataclass(frozen=True)
class AthenaParts:
    """How large her request may be, and how it goes out."""

    params: RetryParams
    budget: int
    """Tokens a request may hold: her rung's window less its output reserve."""
    keep_recent: int
    """`athena.keep_recent`: the newest exchanges a fit never evicts."""
    blocks: frozenset[str] = frozenset()
    """The optional blocks her prompt was rendered with (`compose.holds`): her tool
    descriptions are rendered with the same, so both say one thing about her shell."""


AthenaEnd = End | Literal["context_full", "paused"]
"""How her request ended: `tools.loop.End`; `context_full`, a request that did not fit after
eviction, which makes the episode `PARTIAL`; or `paused`, at a task boundary (the desk's
`paused`), to be resumed."""


@dataclass(frozen=True)
class AthenaAnswer:
    report: str | None
    """Her final reply, verbatim; None when her request ended without one."""
    end: AthenaEnd


class Athena:
    """One episode's Athena."""

    def __init__(
        self,
        rung: BoxRung,
        policy: RolePolicy,
        ctx: ToolContext,
        journal: JournalPort,
        limits: Mapping[str, int],
        desk: TaskDesk,
        parts: AthenaParts,
        *,
        cancel: threading.Event | None,
        live: LiveSink | None = None,
        tells: Tells | None = None,
    ) -> None:
        """`live` shows each of her replies as it streams. `tells` brings Zeus's
        messages to her before her next model call when no worker round is open."""
        # Beneath any client a role wraps around it: what the provider got is recorded.
        self._sent = SentRequests(rung.client)
        self._rung = replace(rung, client=self._sent)
        # The desk's step recorder joins whatever the policy already wraps calls with.
        around = (*policy.pipeline.around, desk)
        self._policy = replace(policy, pipeline=replace(policy.pipeline, around=around))
        edits = desk.edits(ctx.edits) if ctx.edits is not None else None
        self._ctx = replace(ctx, desk=desk, edits=edits)
        self._journal = journal
        self._limits = limits
        self._desk = desk
        self._parts = parts
        self._cancel = cancel
        self._tells = tells
        self._speaker = None if live is None else Speaker(live, ROLE, lambda: desk.held)
        self._sink = BoxSink(ROLE, journal, rung.name, rung.model)
        self._sink.sent = self._sent
        self._live: JournaledTranscript | None = None
        """Her request's transcript while it runs, for her reader's home (`now`)."""

    def now(self) -> Now | None:
        """Her running request's conversation, for her reader; None
        between requests."""
        live = self._live
        return None if live is None else live.now(ROLE)

    def start(self, head: tuple[Message, ...]) -> AthenaAnswer:
        """Her request from its head (`athena_head`), the head journaled first."""
        conversation = BoxConversation(
            role=ROLE, head=head, messages=(), params=self._parts.params
        )
        transcript = self._transcript(conversation)
        transcript.begin()
        return self._run(transcript)

    def resume(self, conversation: BoxConversation, owed: tuple[Owed, ...]) -> AthenaAnswer:
        """Her request continued from her conversation as journaled, each call the crash left
        unanswered answered first, or dispatched again (`box/resume.athena_owed`): a desk call
        then continues from the desk's own line for it."""
        return self._run(self._transcript(conversation), owed)

    def _transcript(self, conversation: BoxConversation) -> JournaledTranscript:
        return JournaledTranscript(
            conversation,
            self._journal,
            budget=self._parts.budget,
            keep_recent=self._parts.keep_recent,
            keep_tools=frozenset(
                self._policy.shown(n)
                for n in self._policy.enabled
                if (t := BUILTINS.get(n)) is not None and t.spec.family == "edit"
            ),
            # Her lines name no task, the ones she writes while she holds one included: her
            # conversation is one, and `rebuilt` reads it whole (task None).
            task_id=None,
            round=None,
        )

    def _run(
        self, transcript: JournaledTranscript, owed: tuple[Owed, ...] = ()
    ) -> AthenaAnswer:
        self._live = transcript
        try:
            return self._ran(transcript, owed)
        finally:
            self._live = None

    def _ran(self, transcript: JournaledTranscript, owed: tuple[Owed, ...]) -> AthenaAnswer:
        tells = self._tells
        # Her own command is cut by a tell as the worker's is.
        ctx = (
            self._ctx
            if tells is None
            else replace(self._ctx, cancel=RoundToken(self._cancel, tells.token))
        )
        dispatcher = Dispatcher[BoxServed](
            BUILTINS,
            self._policy,
            ctx,
            self._sink,
            limits=self._limits,
            operation_id=self._desk.operation_id,
        )
        dispatcher.open_scope("request")
        offered = dispatcher.offered()
        transcript.conversation = replace(
            transcript.conversation,
            tools=shown(
                resident(
                    offered,
                    tool_descriptions(offered, *BOX_DESCRIPTIONS, present=self._parts.blocks),
                )
            ),
            commands=frozenset(
                self._policy.shown(s.name) for s in offered if s.mutates == "environment"
            ),
        )
        ledger = self._rung.ledger
        model = self._rung.model
        role = _TaskRole(dispatcher, self._desk, self._sink)
        for item in owed:
            # In call order: an answer is added; a call is dispatched again, its answer added.
            if isinstance(item, Message):
                transcript.add(item)
            else:
                reply = role.call(item, self._cancel)
                transcript.add(
                    Message(role="tool", content=reply.content, tool_call_id=item.id)
                )
                if reply.ends:
                    # Her tool use ended (a cancel at stop 2, a stop): every later call is
                    # answered with the end's words, so her next turn is her reply.
                    role.close()

        def asking() -> AskResult[str]:
            # A message from Zeus with no worker round open is hers before her next model call
            # and her request goes on after it.
            before: AskResult[str] | None = None
            while True:
                said = asked_once()
                said = said if before is None else said.after(before)
                if said.end != "held" or tells is None:
                    return said
                due = tells.deliver()
                if due:
                    transcript.add(Message(role="user", content=to_athena(due)))
                before = said

        def asked_once() -> AskResult[str]:
            return ask(
                self._rung.client,
                transcript,
                role,
                reasks=1,
                charge=lambda usage: (
                    0.0
                    if ledger is None
                    # Journaled as her `model_call`, so never on the session too.
                    else ledger.charge_usage(self._held(), usage, model, journaled=True)
                ),
                # A failed call has no `model_call` line: its charge is a session `spend` line.
                charge_failed=lambda usage: (
                    0.0 if ledger is None else ledger.charge_usage(self._held(), usage, model)
                ),
                # A pause is read before every model call, beside the spend gate.
                may_start=lambda: (
                    not self._desk.paused and (ledger is None or ledger.may_start())
                ),
                cancel=self._cancel,
                on_call=lambda completion, cost, refusal: self._line(completion, cost, refusal),
                speaker=self._speaker,
                hooks=RoleLoopHooks(self.hooks, self._policy.registry_name),
                hold=tells.token if tells is not None else None,
            )

        first: str | None = None

        def add(said: AskResult[str], words: str) -> None:
            # Her stop's hooks, `owed_review` first among them (she replied
            # with a task reported and never reviewed, and the run ended `PARTIAL` with nothing
            # committed). Journaled, so a resume finds it said; her reply is kept, and her turn
            # goes on with her tools open.
            nonlocal first
            first, reply = said.answer, kept(said)
            if reply is not None:
                transcript.add(reply)
            transcript.add(Message(role="user", content=words))

        try:
            asked = continued(asking(), words=self._go_on, add=add, again=lambda _: asking())
        except ContextFull:
            return AthenaAnswer(first, "answered" if first is not None else "context_full")
        if self._desk.paused and asked.end == "spend_cap":
            return AthenaAnswer(None, "paused")
        if asked.answer is None and first is not None:
            # The reminder's request gave no reply: her first one is still her report, already
            # in her conversation. A failed request is no reason to lose it; a stop or the cap
            # still ends the episode as they say.
            failed = asked.end in ("failed", "unparsed", "kept_calling")
            return AthenaAnswer(first, "answered" if failed else asked.end)
        reply = kept(asked)
        if reply is not None:
            # The loop keeps no final answer (`tools.loop.kept`); her conversation does, so a
            # resume finds her request answered rather than continuing it.
            transcript.add(reply)
        return AthenaAnswer(asked.answer, asked.end)

    def _held(self) -> str:
        return self._desk.held or ""

    def _go_on(self, asked: AskResult[str]) -> str | None:
        """What her request goes on with, if her stop's hooks say it does not end yet: only
        after an answered reply. How many were taken is the journal's, so a resume
        counts the ones said before it; her continues count toward the episode's steering."""
        hooks = self.hooks
        if asked.answer is None or asked.end != "answered":
            return None
        if not hooks.has(HookPoint.BEFORE_STOP):
            return None
        task = self._desk.book().open()
        stop = BeforeStop(
            reply=asked.answer,
            tools_open=not asked.tools_ended,
            open_task=None
            if task is None
            else OpenTask(id=task.id, title=task.title, state=task.state),
            ended=self._desk.ended is not None,
        )
        used = hooks.used(HookPoint.BEFORE_STOP, Continue)
        return at_stop(hooks, stop, used=used, room=hooks.steering_room()).words

    @property
    def hooks(self) -> RoleHooks:
        return self._ctx.hooks or RoleHooks.none("athena")

    def _line(self, completion: Completion, cost: float, refusal: ParseError | None) -> None:
        self._sink.task_id = self._desk.held
        self._sink.model_call(REQUEST, completion, cost, refusal)


class _TaskRole:
    """`tools.loop.Role` for her: each call through her dispatcher, at the task she holds; a
    task tool's answer opens her request scope again; a cancel at stop 2 ends her tool use.
    Her reply with no tool call is her report, kept whole and never parsed."""

    def __init__(
        self, dispatcher: Dispatcher[BoxServed], desk: TaskDesk, sink: BoxSink
    ) -> None:
        self._dispatcher = dispatcher
        self._desk = desk
        self._sink = sink
        self._closed = False

    def close(self) -> None:
        """End her tool use before `ask` begins: a resume whose re-dispatched call ended it."""
        self._closed = True

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        del cancel  # the context carries the run's token
        if self._closed:
            return Reply(self.after_end(), ends=True)
        self._sink.task_id = self._desk.held
        served = self._dispatcher.call(call)
        tool = BUILTINS.get(call.name)
        if tool is not None and tool.spec.family == "task":
            self._dispatcher.open_scope("request")
        ends = self._dispatcher.stopped is not None or self._desk.ended is not None
        return Reply(served.text, ends=ends)

    def after_end(self) -> str:
        return "your tool use has ended; reply with your report now"

    def parse(self, text: str) -> str | ParseError:
        return text

    def reask(self, error: ParseError) -> str:
        return error.hint

    def cut(self) -> ParseError:
        return ParseError(
            kind="truncated", hint="your report hit max_tokens; send it again, shorter"
        )
