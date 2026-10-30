"""The task worker: one task's rounds, each a `tools.loop.ask` to its end.

A round opens its conversation from `boxctx/worker.task_head` (`start`), continues it with the
reviewer's feedback as the next user message (`cont`, so the provider's cache is
kept), or continues an open one `rebuilt` from the journal, what the crash cut off reconciled
first (`resume`, `box/resume.py`). Its tools run through the
role's dispatcher: the reads, the edit tools and the box shell over the run's worktree, every
writing call a recorded step (`box/steps.StepRecord`), every call a `tool_call` line and every
model call a `model_call` line (`box/sink.BoxSink`), charged through the rung's ledger and
gated by its `may_start` before it is made, as Athena's are.

A reply with no tool call is the report: kept as the worker's words, never parsed. Why the
round stopped is read from the loop and the dispatcher, never from the report: `replied`; the
call cap or the deadline, `limit`; a repetition stop, `looped`; a conversation that does not
fit, `context_full`; the client failing, `model_error`; the harness raising, `harness_error`.
A person's stop or the spend cap is not a round's end: `WorkerStopped` is raised, the task
stays open, and resume continues it. A message from Zeus holds the round at its next tool
boundary (`box/tells.py`): a shell command in flight is cut and its step put back,
the calls after it are answered `not run`, no next model call is made, and the round reports
`held`, with no report and its facts.

At the round's end the facts are measured (`box/facts.measure`) and the round's
`task_reported` line is written with them.

The role's hooks are asked before each model call and after each turn's calls
(`hooks/loop.py`), and where its answer is accepted (`hooks/stop.at_stop`): a `Continue` there
is the reply and the hook's words, journaled in that order, and the round goes on with its tools
open. A round cut off between the two resumes with the words the journal holds
(`resume.owed_continuation`). A hook's `Halt` ends the round `halted`.

Where a hook is bound for the worker at `after_batch` or `before_stop`, each call is tagged as
the model reads its answer (`box/progress.TaskProgress`) and the hooks are told the
round's progress; a process that did not run the task's rounds rebuilds it from the journal.
While the progress watchdog is bound, the worker's `Repetition` refuses repeated calls instead
of ending the round, so the watchdog's nudge and report come first.

Composed once per episode by `box/compose.compose_episode`.

**Concurrency.** One `WorkerRun` per episode, serving one round at a time on its thread.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from chickadee import pycall
from chickadee.adapters.base import (
    BOX_DESCRIPTIONS,
    SHARED_TEMPLATES,
    load_template,
    tool_descriptions,
)
from chickadee.approval.tier import Preferences
from chickadee.box.commit import Reindexer
from chickadee.box.facts import measure, ran_of
from chickadee.box.progress import StepSeen, TaskProgress
from chickadee.box.resume import (
    NOT_RUN,
    Asking,
    Owed,
    never_asking,
    open_round,
    owed_continuation,
    owed_reply,
    task_trees,
    unanswered,
)
from chickadee.box.rung import BoxRung
from chickadee.box.sink import BoxServed, BoxSink, task_counters
from chickadee.box.steps import Beside, StepRecord, StepSnapshot, StepTrees
from chickadee.box.syntax import SyntaxChecker
from chickadee.box.tells import RoundToken, Tells
from chickadee.boxctx.conversation import (
    BoxConversation,
    ContextFull,
    JournaledTranscript,
    rebuilt,
)
from chickadee.boxctx.records import SymbolGraph, SymbolTable, TaskRecord
from chickadee.boxctx.sent import SentRequests
from chickadee.boxctx.worker import task_head
from chickadee.client.live import LiveSink, Speaker
from chickadee.client.request import Message, ModelClient
from chickadee.context.homes import Now
from chickadee.executor.types import ExecResult
from chickadee.graph.diagnostics import SinceStart
from chickadee.hooks.failures import HookHalted
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.loop import RoleLoopHooks
from chickadee.hooks.payloads import BeforeStop, DiagnosticsSeen, WorkerProgress
from chickadee.hooks.points import HookPoint
from chickadee.hooks.stop import at_stop, continued
from chickadee.hooks.verdicts import Continue
from chickadee.hooks.watchdog import REPORT_HEAD
from chickadee.ids import step_id
from chickadee.journal.effects import on_journal
from chickadee.journal.port import JournalPort
from chickadee.profiles.model import LanguageProfile
from chickadee.schema import RetryParams
from chickadee.schema.adapter import Completion, ParseError, ToolCall, Usage
from chickadee.schema.journal import EventKind, ExecP, RoundEnd, TaskFactsP, TaskReportedP
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import Repetition
from chickadee.tools.context import ToolContext
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.loop import AskResult, Reply, Role, ask, kept
from chickadee.tools.meta import shown
from chickadee.tools.policy import RolePolicy
from chickadee.tools.rules import BoundReached, Rule
from chickadee.tools.validate import resident
from chickadee.work import DEFAULT_DIR, with_work_dir
from chickadee.workspace import current_tree

__all__ = ["RoundReport", "WorkerParts", "WorkerRun", "WorkerStopped", "instructions"]

ROLE = "worker"

_log = logging.getLogger(__name__)


def instructions(*, work_dir: str) -> str:
    """The task worker's fixed instructions, the head's first message: the same bytes for
    every task, so the provider's cache holds them. `work_dir` is `[work] dir`, the
    folder it is told to read and never write."""
    return with_work_dir(load_template(SHARED_TEMPLATES, "worker.md"), work_dir)


@dataclass(frozen=True)
class RoundReport:
    """One round's end: why it stopped, the worker's words, and the facts measured."""

    task: str
    round: int
    stopped: RoundEnd
    report: str | None
    """The worker's words; None when it did not reply."""
    facts: TaskFactsP


class WorkerStopped(Exception):
    """A round ended by a person's stop or the spend cap, not by the worker: the task stays
    open, its round not reported, for resume. `by` says which, for the episode's
    outcome (`box/loop.TaskLoop`: `ABORTED` or `BUDGET_CAP`)."""

    def __init__(self, why: str, by: Literal["person", "spend_cap"] = "person") -> None:
        super().__init__(why)
        self.why = why
        self.by: Literal["person", "spend_cap"] = by


@dataclass(frozen=True)
class WorkerParts:
    """What the worker reads of its run: the worktree and its languages, the graph, how large
    a request may be, and the round's bounds."""

    root: Path
    profiles: tuple[LanguageProfile, ...]
    graph: SymbolGraph | None
    indexer: Reindexer | None
    params: RetryParams
    budget: int
    """Tokens a request may hold: the rung's window less its output reserve."""
    keep_recent: int
    untracked_max_bytes: int
    task_seconds: int
    prefs: Callable[[], Preferences] | None = None
    """The person's approval.toml: a step puts back a path its `deny` globs name."""
    syntax: SyntaxChecker | None = None
    """The edit tool's own check, run again over each changed file for the task's facts."""
    blocks: frozenset[str] = frozenset()
    """The optional blocks its prompt was rendered with (`compose.holds`): its tool
    descriptions are rendered with the same, so both say one thing about its shell."""
    work_dir: str | None = None
    """`[work] dir`: a command's write into it is put back."""
    beside: Beside | None = None
    """The session's running jobs: each step beside one says so."""
    python_unsandboxed: bool = False
    """`[sandbox] allow_unsandboxed_python`, read when the policy's
    `calling` is `python`."""


class WorkerRun:
    """One episode's task worker."""

    def __init__(
        self,
        rung: BoxRung,
        policy: RolePolicy,
        ctx: ToolContext,
        journal: JournalPort,
        limits: Mapping[str, int],
        parts: WorkerParts,
        *,
        cancel: threading.Event | None,
        fixed: str | None = None,
        live: LiveSink | None = None,
        tells: Tells | None = None,
    ) -> None:
        """`live` shows each reply as it streams, labelled with the task. `tells` holds
        a round for a message from Zeus; None where none reaches the run."""
        # Beneath any client a role wraps around it: what the provider got is recorded.
        self._sent = SentRequests(rung.client)
        self._rung = replace(rung, client=self._sent)
        self._policy = policy
        self._ctx = ctx
        self._journal = journal
        self._limits = limits
        self._parts = parts
        # No configured folder (`work_dir` None, no fence) is the default's name, never a gap.
        told = parts.work_dir or DEFAULT_DIR
        self._fixed = instructions(work_dir=told) if fixed is None else fixed
        self._cancel = cancel
        self._tells = tells
        self._shown = live
        self._sink = BoxSink("worker", journal, rung.name, rung.model)
        self._sink.sent = self._sent
        self._before: dict[str, SymbolTable | None] = {}
        self._trees: dict[str, StepTrees] = {}
        self._live: JournaledTranscript | None = None
        """The round's transcript while it runs, for its reader's home (`now`)."""
        self._spent: dict[str, float] = {}
        """Seconds each task's rounds have run in this process: `worker.task_seconds` bounds
        the task, not a round."""
        self._progress: dict[str, TaskProgress] = {}
        """Each task's progress record, while a hook is told it."""

    @property
    def task(self) -> str | None:
        """The task whose round is running, for the box shell's claims; None between rounds."""
        return self._sink.task_id

    @property
    def round(self) -> int | None:
        """The round running, with `task`: where its hooks' lines are written."""
        return self._sink.round

    @property
    def hooks(self) -> RoleHooks:
        return self._ctx.hooks or RoleHooks.none("worker")

    def calls_left(self, task: str) -> int | None:
        """The task's tool calls left for another round (`worker.max_tool_calls` is per task,
        its count replayed into each round); None where nothing bounds them."""
        cap = self._limits.get("worker.max_tool_calls")
        if cap is None:
            return None
        made = task_counters(self._journal.projection(), ROLE, task).calls
        return max(cap - made, 0)

    def room(self, task: str) -> bool:
        """Whether another round of `task` could make a call: calls and seconds left."""
        calls = self.calls_left(task)
        seconds = self._parts.task_seconds - self._spent.get(task, 0.0)
        return (calls is None or calls > 0) and seconds > 0

    def now(self) -> Now | None:
        """The running round's conversation, for its reader: one per task,
        so a reader's memory of it ends with the task. None between rounds."""
        task, live = self._sink.task_id, self._live
        return None if task is None or live is None else live.now(task)

    # -- the three ways a round begins -----------------------------------------------------

    def start(
        self,
        task: str,
        brief: str,
        start_tree: str,
        *,
        done: tuple[TaskRecord, ...] = (),
        design_layer: int = 4000,
        before: SymbolTable | None = None,
        learned: str | None = None,
    ) -> RoundReport:
        """The task's first round, from its head: the fixed instructions, "done so far", what
        the attempt it replaces learned (`facts.last_attempt`), the brief.
        `before` is the symbol table taken at delegation, for the facts."""
        self._before[task] = before
        self._trees[task] = StepTrees(
            start_tree, self._parts.untracked_max_bytes, run_id=self._journal.run_id, task=task
        )
        told = brief if learned is None else f"{learned}\n\n{brief}"
        head = task_head(self._fixed, done, told, design_layer)
        conversation = BoxConversation(
            role="worker", head=head, messages=(), params=self._parts.params
        )
        transcript = self._transcript(task, 1, conversation)
        transcript.begin()
        return self._ask(task, 1, transcript)

    def cont(self, task: str, feedback: str | None, rounds: int) -> RoundReport:
        """The next round of the same conversation, the reviewer's feedback its next user
        message, or none after a held round (`continue`: the worker's next model
        call is made on the conversation as it stands, its calls left and seconds what the
        held round left): rebuilt from the journal, so it is what the worker saw. What the
        last round left owed comes first: its reply, if a crash kept it from the conversation,
        and an answer to each call it ended without answering (a call that raised, or calls
        after the end), since a provider refuses a call with no result."""
        run = self._journal.projection()
        # The next round's first step starts from the tree as it is now, never a tree kept
        # from the last round: between the two, Athena may have changed it (her steps under
        # the task she took over, or her shell outside one), and a step's pre-image that
        # predates those would put them back if the step were cut off.
        start, _ = task_trees(run, task)
        now, _ = current_tree(
            self._parts.root, untracked_max_bytes=self._parts.untracked_max_bytes
        )
        self._before.setdefault(task, None)
        self._trees[task] = StepTrees(
            start,
            self._parts.untracked_max_bytes,
            run_id=self._journal.run_id,
            task=task,
            before=now,
        )
        conversation = self._rebuilt(task)
        nxt = rounds + 1
        transcript = self._transcript(task, nxt, conversation)
        reply = owed_reply(run, task, conversation, self._journal.get_blob)
        if reply is not None:
            transcript.add(reply)
        transcript.add(
            *(
                Message(role="tool", content=NOT_RUN, tool_call_id=c)
                for c in unanswered(transcript.conversation)
            ),
            *((Message(role="user", content=feedback),) if feedback is not None else ()),
        )
        return self._ask(task, nxt, transcript)

    def resume(self, task: str, *, asking: Asking = never_asking) -> RoundReport | None:
        """`task`'s open round continued from its conversation as journaled, what the crash
        cut off reconciled first (`box/resume.open_round`), a call whose permission question
        was still waiting (`asking`) dispatched again; None when no round is open."""
        conversation = self._rebuilt(task)
        found = open_round(
            self._journal.projection(),
            task,
            conversation,
            self._parts.root,
            self._journal,
            self._parts.indexer,
            untracked_max_bytes=self._parts.untracked_max_bytes,
            asking=asking,
        )
        if found is None:
            return None
        self._before.setdefault(task, None)
        self._trees[task] = StepTrees(
            found.start_tree,
            self._parts.untracked_max_bytes,
            run_id=self._journal.run_id,
            task=task,
            before=found.last_tree,
        )
        transcript = self._transcript(task, found.round, conversation)
        run = self._journal.projection()
        go_on = owed_continuation(run, task, found.round, conversation, self.hooks)
        if go_on is not None:
            # Cut off between a `Continue`'s reply and its words.
            transcript.add(Message(role="user", content=go_on))
        return self._ask(task, found.round, transcript, owed=found.answers)

    # -- one round ---------------------------------------------------------------------------

    def _rebuilt(self, task: str) -> BoxConversation:
        return rebuilt(
            self._journal.projection(),
            "worker",
            task,
            self._journal.get_blob,
            params=self._parts.params,
        )

    def _transcript(
        self, task: str, round_: int, conversation: BoxConversation
    ) -> JournaledTranscript:
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
            task_id=task,
            round=round_,
        )

    def _ask(
        self,
        task: str,
        round_: int,
        transcript: JournaledTranscript,
        *,
        owed: tuple[Owed, ...] = (),
    ) -> RoundReport:
        self._sink.task_id, self._sink.round = task, round_
        self._live = transcript
        try:
            return self._asked(task, round_, transcript, owed)
        finally:
            self._sink.task_id, self._sink.round = None, None
            self._live = None

    def _asked(
        self, task: str, round_: int, transcript: JournaledTranscript, owed: tuple[Owed, ...]
    ) -> RoundReport:
        steps, token = self._steps(task, round_)
        tells = self._tells
        hooks = self.hooks
        progress = self._watched(task)
        if progress is not None:
            # A round `continue` opened goes on from the held one.
            held = [
                p.payload.stopped
                for p in self._journal.projection().tasks.get(task, ())
                if isinstance(p.payload, TaskReportedP)
            ][-1:] == ["held"]
            progress.begin_round(round_, self.calls_left(task), held=held)
        policy = self._round_policy(steps)
        # `finish` hands its report here, and the round ends on it.
        finished = _Finished()
        ctx = replace(self._ctx, finish=finished)
        if token is not None:
            ctx = replace(ctx, cancel=token)
        dispatcher = Dispatcher[BoxServed](
            BUILTINS,
            policy,
            ctx,
            self._sink,
            limits=self._limits,
            operation_id=lambda call_id: step_id(task, round_, call_id),
        )
        dispatcher.open_scope("task")
        offered = dispatcher.offered()
        tools = shown(
            resident(
                offered,
                tool_descriptions(offered, *BOX_DESCRIPTIONS, present=self._parts.blocks),
            )
        )
        commands = frozenset(
            self._policy.shown(s.name) for s in offered if s.mutates == "environment"
        )
        transcript.conversation = replace(
            transcript.conversation, tools=tools, commands=commands
        )
        for item in owed:
            # In call order: an answer is added; a call is dispatched again, its answer added.
            if isinstance(item, Message):
                transcript.add(item)
            else:
                served = dispatcher.call(item)
                transcript.add(Message(role="tool", content=served.text, tool_call_id=item.id))
        deadline = _Deadline(self._parts.task_seconds - self._spent.get(task, 0.0))
        ledger = self._rung.ledger
        reply: Message | None = None
        client, role = self._round_role(_WorkerRole(dispatcher, finished), policy)
        if progress is not None:
            # A Python block makes several calls, which the call cap counts one by one.
            made = (
                (lambda: task_counters(self._journal.projection(), ROLE, task).calls)
                if policy.calling == "python"
                else None
            )
            role = _Observed(role, progress, steps, made, self._ran_of)
        loop_hooks = RoleLoopHooks(
            hooks, policy.registry_name, progress=progress.now if progress is not None else None
        )

        def called(completion: Completion, cost: float, refusal: ParseError | None) -> None:
            if progress is not None:
                progress.respond()
            self._sink.model_call("task", completion, cost, refusal)

        def asking() -> AskResult[str]:
            finished.used = False
            return ask(
                client,
                transcript,
                role,
                reasks=1,
                charge=self._charge(task),
                # No `model_call` line records a failed call, so its charge is a `spend` line.
                charge_failed=self._charge(task, journaled=False),
                # The round's deadline is read before every model call, beside the spend
                # gate: a round past it ends as `limit`, never as the person's stop.
                may_start=lambda: (
                    not deadline.passed() and (ledger is None or ledger.may_start())
                ),
                cancel=self._cancel,
                speaker=None
                if self._shown is None
                else Speaker(self._shown, "worker", lambda: task, policy.calling),
                force_tool=policy.force_tool_call,
                on_call=called,
                hooks=loop_hooks,
                hold=tells.token if tells is not None else None,
            )

        def add(said: AskResult[str], words: str) -> None:
            # The reply, then the hook's words: what the conversation shows, and what a resume
            # finds. A fresh `ask` follows, so tool use is open.
            reply = kept(said)
            told = Message(role="user", content=words)
            transcript.add(*((reply, told) if reply is not None else (told,)))

        try:
            asked = continued(
                asking(),
                words=lambda said: self._go_on(
                    task, round_, said, dispatcher, finished, progress
                ),
                add=add,
                again=lambda _: asking(),
            )
            stopped = self._stopped(asked, dispatcher, deadline)
            report, reply = asked.answer, kept(asked)
        except ContextFull:
            stopped, report = "context_full", None
        except WorkerStopped:
            raise
        except Exception as exc:  # the harness's own failure is the round's end, said
            # Named by its type, since an `AssertionError` says nothing else, and logged whole:
            # the report is one line, and the traceback is the only place the cause is.
            _log.error("%s round %d stopped on a harness error", task, round_, exc_info=exc)
            stopped = "harness_error"
            report = f"the harness failed: {type(exc).__name__}: {exc}"
        finally:
            self._spent[task] = self._spent.get(task, 0.0) + deadline.elapsed()
        reported = self._report(task, round_, stopped, report, steps.last)
        if reply is not None:
            # The loop keeps no final answer (`tools.loop.kept`); the worker's conversation
            # does, so the next round continues after its reply. Journaled after the report,
            # so a round with a reply is never open on resume (a crash between:
            # `resume.owed_reply`).
            transcript.add(reply)
        return reported

    def _steps(self, task: str, round_: int) -> tuple[StepRecord, RoundToken | None]:
        """The round's step record, and the token its tools see: set by the run's stop or,
        while a command runs, by a tell, which cuts that command and only that."""
        trees = self._trees.get(task)
        if trees is None:
            raise ValueError(f"{task} has no step trees: start or resume it first")
        self._keep_start(task)
        tells = self._tells
        token = RoundToken(self._cancel, tells.token) if tells is not None else None
        steps = StepRecord(
            self._parts.root,
            on_journal(self._journal, task_id=task, round=round_),
            trees,
            self._parts.indexer,
            prefs=self._parts.prefs,
            work_dir=self._parts.work_dir,
            beside=self._parts.beside,
            cut=token,
        )
        return steps, token

    def _round_policy(self, steps: StepRecord) -> RolePolicy:
        """The worker's policy for one round: its steps recorded around each writing call,
        and its `Repetition` refusing, not ending the round, while the progress watchdog is
        bound."""
        pipeline = self._policy.pipeline
        admit = pipeline.admit
        if self.hooks.uses("progress_watchdog"):
            admit = _refusing(admit)
        return replace(self._policy, pipeline=replace(pipeline, admit=admit, around=(steps,)))

    def _round_role(self, role: Role[str], policy: RolePolicy) -> tuple[ModelClient, Role[str]]:
        """The rung's client and `role`, both through the Python arm when it is chosen."""
        client: ModelClient = self._rung.client
        pycall.warn_retired_flags()
        if policy.calling == "python":  # chosen in the config
            client, role = pycall.python_calls(
                client,
                role,
                root=self._parts.root,
                refused=pycall.traced(self._sink),
                noted=pycall.traced_fixes(self._sink),
                allow_unsandboxed=self._parts.python_unsandboxed,
            )
        return client, role

    def _go_on(
        self,
        task: str,
        round_: int,
        asked: AskResult[str],
        dispatcher: Dispatcher[BoxServed],
        finished: _Finished,
        progress: TaskProgress | None,
    ) -> str | None:
        """What the round goes on with, if its hooks say it does not stop yet."""
        hooks = self.hooks
        if asked.end != "answered" or asked.answer is None:
            return None
        if not hooks.has(HookPoint.BEFORE_STOP):
            return None
        if progress is not None and hooks.uses("evidence_check"):
            seen = self._diagnosed(task, hooks.limits.recheck_wait_s)
            if seen is not None:
                progress.diagnosed(seen)
        stop = BeforeStop(
            reply=asked.answer,
            via="finish" if finished.used else "reply",
            tools_open=not asked.tools_ended and dispatcher.stopped is None,
            facts=self._facts(task, None),
            progress=progress.now() if progress is not None else None,
            handed_up=any(
                hooks.text(line).startswith(REPORT_HEAD)
                for line in hooks.select(HookPoint.AFTER_BATCH, task_id=task, round=round_)
            ),
        )
        used = hooks.used(HookPoint.BEFORE_STOP, Continue, task_id=task, round=round_)
        return at_stop(hooks, stop, used=used).words

    def _keep_start(self, task: str) -> None:
        """Each index pass from here keeps `task`'s start; the next task's rounds
        name theirs."""
        trees = self._trees.get(task)
        if isinstance(self._parts.indexer, _KeepsStart):
            self._parts.indexer.keeping = task
            self._parts.indexer.keeping_from = trees.start if trees is not None else None

    def _diagnosed(self, task: str, wait_s: float) -> DiagnosticsSeen | None:
        """`task`'s files against their kept start, the re-check given its short wait first;
        None with no index that keeps one."""
        indexer = self._parts.indexer
        if not isinstance(indexer, _KeepsStart):
            return None
        indexer.settle(wait_s)
        found = indexer.since_start(task)
        return DiagnosticsSeen(
            new_errors=tuple(str(f) for f in found.new_errors),
            new_warnings=tuple(str(f) for f in found.new_warnings),
            already=tuple(str(f) for f in found.already),
            no_record=found.no_record,
            unchecked=found.unchecked,
        )

    def progress(self, task: str, round_: int) -> WorkerProgress | None:
        """`task`'s round `round_` as its progress record holds it, for the reader of
        its report (`round_ended`): the record kept in this process when it holds that round,
        else that round rebuilt from the journal; None for a round the journal does not hold."""
        found = self._progress.get(task)
        if found is not None and found.round == round_:
            return found.now()
        rounds: dict[int, WorkerProgress] = {}
        TaskProgress.rebuilt(
            self._journal.projection(),
            task,
            self._journal.get_blob,
            self._parts.root,
            profiles=self._parts.profiles,
            call_cap=self._limits.get("worker.max_tool_calls"),
            registry_name=self._policy.registry_name,
            on_round=lambda r, p, kept=rounds: kept.update({r: p}),
        )
        return rounds.get(round_)

    def _watched(self, task: str) -> TaskProgress | None:
        """`task`'s progress record, where a hook bound for the worker is told it: kept from
        the task's earlier rounds in this process, else rebuilt from the journal. None
        otherwise, so an unwatched worker pays nothing for it."""
        hooks = self.hooks
        if not (hooks.has(HookPoint.AFTER_BATCH) or hooks.has(HookPoint.BEFORE_STOP)):
            return None
        found = self._progress.get(task)
        if found is None:
            found = TaskProgress.rebuilt(
                self._journal.projection(),
                task,
                self._journal.get_blob,
                self._parts.root,
                profiles=self._parts.profiles,
                call_cap=self._limits.get("worker.max_tool_calls"),
                registry_name=self._policy.registry_name,
            )
            self._progress[task] = found
        return found

    def _ran_of(self, operation_id: str) -> ExecResult | None:
        """A step's command with its whole output, from its `exec` line: what the progress
        record reads a test run from, as a rebuild of it does."""
        lines = self._journal.projection().lines(ExecP)
        found = next(
            (p.payload for p in reversed(lines) if str(p.payload.operation_id) == operation_id),
            None,
        )
        return None if found is None else ran_of(found, self._journal.get_blob)

    def _facts(self, task: str, last: StepSnapshot | None) -> TaskFactsP:
        """The task's facts as the journal and the trees hold them now."""
        facts = measure(
            self._parts.root,
            self._journal.projection(),
            task,
            self._parts.profiles,
            self._parts.graph,
            self._before.get(task),
            self._journal.get_blob,
            syntax=self._parts.syntax,
        )
        if last is not None and last.skipped:
            facts = facts.model_copy(update={"skipped": last.skipped})
        return facts

    def _report(
        self,
        task: str,
        round_: int,
        stopped: RoundEnd,
        report: str | None,
        last: StepSnapshot | None,
    ) -> RoundReport:
        """The round's end: its facts measured and its `task_reported` line written."""
        facts = self._facts(task, last)
        report_sha = self._journal.put_blob(report.encode()) if report is not None else None
        self._journal.append(
            EventKind.TASK_REPORTED,
            TaskReportedP(
                task_id=task, round=round_, stopped=stopped, report_sha=report_sha, facts=facts
            ),
            task_id=task,
        )
        return RoundReport(task, round_, stopped, report, facts)

    def _stopped(
        self,
        asked: AskResult[str],
        dispatcher: Dispatcher[BoxServed],
        deadline: _Deadline,
    ) -> RoundEnd:
        """Why the round ended, from the loop and the dispatcher. A person's stop and the spend
        cap are not the round's: they raise, and the task stays open."""
        if asked.end == "cancelled":
            raise WorkerStopped("stopped by the user")
        if asked.end == "held":
            return "held"
        if asked.end == "spend_cap":
            if deadline.passed():
                return "limit"
            raise WorkerStopped("the spend cap stopped the work", "spend_cap")
        if asked.end == "failed":
            return "model_error"
        stop = dispatcher.stopped
        if asked.halted is not None or (
            stop is not None and isinstance(stop.failure, HookHalted)
        ):
            return "halted"
        if stop is not None:
            failure = stop.failure
            if isinstance(failure, BoundReached) and failure.resource in (
                "repetition",
                "print_only",
            ):
                return "looped"
            return "limit"
        if asked.answer is not None:
            return "replied"
        return "model_error"

    def _charge(self, task: str, *, journaled: bool = True) -> Callable[[Usage], float]:
        """The ledger's charge for `task`'s calls: `journaled` for an answered one, which its
        `model_call` line records."""
        ledger = self._rung.ledger
        model = self._rung.model

        def charge(usage: Usage) -> float:
            return (
                0.0
                if ledger is None
                else ledger.charge_usage(task, usage, model, journaled=journaled)
            )

        return charge


class _Finished:
    """`FinishPort` for one round: the report `finish` handed over, once it has."""

    def __init__(self) -> None:
        self.report: str | None = None
        self.used = False
        """`finish` was called since the request began: the stop's `via`."""

    def finish(self, report: str) -> str:
        self.report = report
        self.used = True
        return "reported: the task ends here"


class _WorkerRole:
    """`tools.loop.Role` for the worker: its calls through the dispatcher; its report is its
    `finish` call's, or its reply with no tool call, kept whole and never parsed."""

    def __init__(self, dispatcher: Dispatcher[BoxServed], finished: _Finished) -> None:
        self._dispatcher = dispatcher
        self._finished = finished

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        del cancel  # the context carries the run's token
        served = self._dispatcher.call(call)
        report, self._finished.report = self._finished.report, None
        stopped = self._dispatcher.stopped is not None
        return Reply(
            served.text, ends=stopped, final=report, data=served.data, refused=served.refused
        )

    def after_end(self) -> str:
        return "your tool calls for this round have ended; reply with your report now"

    def parse(self, text: str) -> str | ParseError:
        return text

    def reask(self, error: ParseError) -> str:
        return error.hint

    def cut(self) -> ParseError:
        return ParseError(
            kind="truncated", hint="your report hit max_tokens; send it again, shorter"
        )


@runtime_checkable
class _KeepsStart(Protocol):
    """An index that keeps a task's start (`graph.indexer.Indexer`)."""

    keeping: str | None
    keeping_from: str | None

    def settle(self, timeout_s: float) -> bool: ...

    def since_start(self, task: str) -> SinceStart: ...


class _Observed:
    """The worker's role, each call tagged as the model reads its answer: the tree its
    step left, when it made one, read from the round's step record."""

    def __init__(
        self,
        role: Role[str],
        progress: TaskProgress,
        steps: StepRecord,
        made: Callable[[], int] | None = None,
        ran: Callable[[str], ExecResult | None] | None = None,
    ) -> None:
        """`made` counts the task's tool calls, where one call of the role's may make
        several (a Python block's); `ran` reads a step's command and whole output."""
        self._role = role
        self._progress = progress
        self._steps = steps
        self._made = made
        self._ran = ran

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        before = self._steps.last
        counted = self._made() if self._made is not None else 0
        reply = self._role.call(call, cancel)
        weight = self._made() - counted if self._made is not None else 1
        after = self._steps.last
        step = (
            StepSeen(after.tree, after.changed, bool(self._steps.last_beside))
            if after is not None and after is not before
            else None
        )
        op = self._steps.last_operation
        ran = self._ran(op) if step is not None and op is not None and self._ran else None
        self._progress.observe(
            call, reply.content, refused=reply.refused, step=step, weight=weight, ran=ran
        )
        return reply

    def after_end(self) -> str:
        return self._role.after_end()

    def parse(self, text: str) -> str | ParseError:
        return self._role.parse(text)

    def reask(self, error: ParseError) -> str:
        return self._role.reask(error)

    def cut(self) -> ParseError:
        return self._role.cut()


def _refusing(admit: tuple[Rule, ...]) -> tuple[Rule, ...]:
    """`admit` with each `Repetition` that ends the round refusing instead: the progress
    watchdog ends a looping round, after its nudge and its report."""
    return tuple(
        replace(r, on_cap="refuse") if isinstance(r, Repetition) and r.on_cap == "stop" else r
        for r in admit
    )


class _Deadline:
    """What is left of the task's `worker.task_seconds`, from now: a round past it ends as
    `limit`. A restart starts the task's clock again; the spend cap still holds."""

    def __init__(self, seconds: float) -> None:
        self._start = time.monotonic()
        self._end = self._start + seconds

    def passed(self) -> bool:
        return time.monotonic() > self._end

    def elapsed(self) -> float:
        return time.monotonic() - self._start
