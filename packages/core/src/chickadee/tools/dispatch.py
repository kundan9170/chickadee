"""The dispatcher: the stages' runner, the same code for every role.

    admit → dispatch → around(run) → result → record → observe

It only runs a role's pipeline over its tools, context and sink: every difference between
roles is a rule, an order, a sink or an executor, never a branch here (D164's costs table: a
per-role flag in this file is a review finding).

- **Admit before execute.** Every call of a response is admitted (and counted) before any
  runs, and the dispatch rules are asked again as each is executed (Zeus's shape).
- **Scopes** are the register's levels. `open_scope` starts a level's counters from
  the sink's replay (decision 5e) and fixes the offered set for it (decision 9).
- **A stop ends the loop**: the call that stopped it, and every later one until a scope
  reopens, is answered and never run (the scout's caps).

**Concurrency, stated where the state is.** A dispatcher, its `ScopeState` and its offered set
belong to one loop on one thread and are never shared; each scout `locate` gets its own.
State shared across loops lives in ports, under the port's own lock. The one exception is a
call whose spec says `threads` (`execute_group`): its run, and only its run, is on a thread
of its own, so the tool's run and the pipeline's `around` wrappers it passes through must
touch no state of this dispatcher's. Its rules, record and observers stay on the loop's.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace

from chickadee.context.result_data import Completed, ResultData
from chickadee.context.results import CallCancelled, ToolResult
from chickadee.errors import HarnessBug
from chickadee.schema.adapter import ToolCall
from chickadee.tools.catalogue import Valid
from chickadee.tools.context import ToolContext
from chickadee.tools.policy import RolePolicy
from chickadee.tools.registry import Registry
from chickadee.tools.rules import (
    AskGate,
    AskPerson,
    CallOutcome,
    Declined,
    Level,
    Pass,
    Pending,
    Refuse,
    ScopeState,
    Stop,
    Verdict,
)
from chickadee.tools.sink import NO_EXECUTORS, RoleExecutors, ToolSink
from chickadee.tools.spec import Parallel, ToolSpec
from chickadee.tools.tool import Answer, Invocation, Port, TextAnswer

__all__ = ["Batch", "Dispatcher"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Batch:
    """Calls that may run together, by parallel class (33 §3): a run of `any` and `capped`
    calls, or one call alone."""

    parallel: Parallel
    pending: tuple[Pending, ...]


class Dispatcher[R]:
    def __init__(
        self,
        registry: Registry,
        policy: RolePolicy,
        ctx: ToolContext,
        sink: ToolSink[R],
        *,
        limits: Mapping[str, int],
        executors: RoleExecutors = NO_EXECUTORS,
        operation_id: Callable[[str], str] = lambda call_id: call_id,
    ) -> None:
        pipeline = policy.pipeline
        if pipeline.slots:
            raise HarnessBug(f"{policy.role}'s pipeline has unfilled slots {pipeline.slots}")
        if not any(isinstance(r, Valid) for r in pipeline.admit):
            raise HarnessBug(f"{policy.role}'s pipeline runs tools it never validates")
        tools = (registry.get(name) for name in policy.enabled)
        specs = [t.spec for t in tools if t is not None]
        missing = sorted(pipeline.keys(specs) - set(limits))
        if missing:
            raise HarnessBug(f"{policy.role}'s pipeline reads limits it lacks: {missing}")
        self._registry = registry
        self._policy = policy
        self._ctx = ctx
        self._sink = sink
        self._executors = executors
        self._operation_id = operation_id
        self.state = ScopeState(limits)
        self.state.place = ctx.files.place if ctx.files is not None else None
        self.state.read_key = ctx.files.repeat_key if ctx.files is not None else None
        self.state.hooks = ctx.hooks
        self._offered: tuple[str, ...] = ()
        self.stopped: Stop | None = None
        """Set when a call ended the role's loop; cleared when a scope opens."""

    # -- scopes ------------------------------------------------------------------------------

    def open_scope(self, level: Level, ports: frozenset[Port] | None = None) -> None:
        """Start `level` from the sink's replay, and fix what this scope offers: the enabled
        tools whose ports are given (and, with `ports`, present for this scope)."""
        counters = self._sink.replay(level)
        self.state.reset(level, calls=counters.calls, tokens=counters.result_tokens)
        self.stopped = None
        offered: list[str] = []
        for name in self._policy.enabled:
            tool = self._registry.get(name)
            if tool is None or (tool.streams and name not in self._executors):
                continue
            if all(self._ctx.has(p) and (ports is None or p in ports) for p in tool.needs):
                offered.append(name)
        self._offered = tuple(offered)
        self.state.offered = tuple(self._policy.shown(n) for n in offered)
        self.state.specs = {
            name: replace(self._policy.resolved(tool.spec), name=self._policy.shown(name))
            for name in self._offered
            if (tool := self._registry.get(name)) is not None
        }

    def offered(self) -> tuple[ToolSpec, ...]:
        """The specs this scope offers, under the names the model sees. Fixed until the scope
        reopens, so a provider prefix never changes mid-session."""
        return tuple(self.state.specs.values())

    # -- the stages --------------------------------------------------------------------------

    def admit(self, calls: Sequence[ToolCall], *, at: int = 0) -> tuple[Pending, ...]:
        """One response's calls, each through the admit rules in order, and counted, before
        any runs."""
        self.state.reset("response")
        return tuple(self._admit_one(call, at) for call in calls)

    def schedule(self, pending: Sequence[Pending]) -> tuple[Batch, ...]:
        """In the order the calls were made: a run of `any` and `capped` calls together,
        anything else alone. A call a verdict answered runs nothing, so it is alone."""
        out: list[Batch] = []
        group: list[Pending] = []
        for p in pending:
            parallel: Parallel = p.spec.parallel if p.spec is not None else "serial"
            if p.decided is None and parallel in ("any", "capped"):
                group.append(p)
                continue
            if group:
                out.append(Batch("any", tuple(group)))
                group = []
            out.append(Batch(parallel, (p,)))
        if group:
            out.append(Batch("any", tuple(group)))
        return tuple(out)

    def dispatch(self, batch: Sequence[Pending]) -> tuple[Pending, ...]:
        """The dispatch rules, asked of every call of a batch before any of them runs (Zeus's
        read bound is asked of a whole group). `execute` does not ask them again."""
        return tuple(self._dispatch(p) for p in batch)

    def execute(self, pending: Pending) -> R:
        """Dispatch (unless `dispatch` already did), run inside the wrappers, the result
        rules, the sink's record, then the observers."""
        pending = self._dispatch(pending)
        decided = pending.decided
        spec = pending.spec
        if decided is not None or spec is None:
            stop = decided if isinstance(decided, Stop) else None
            return self._finish(CallOutcome(pending, None, stop))
        return self._executed(pending, spec)

    def execute_group(
        self, dispatched: Sequence[Pending], *, stopping: Callable[[], bool]
    ) -> Iterator[R]:
        """A group `dispatch` has asked, answered in call order. The calls whose spec says
        `threads` start first, at once, each on a thread of its own; the rest run here
        meanwhile, one after another, and one that `stopping`
        reaches before it starts is cancelled.

        A call run here is finished and recorded as it ends, inside the sink's hold, as
        `execute` records it: what the hold guards spans its run and its record (Zeus's read
        family's merge decision and its line, `ZeusSink.executing`), so the next call never
        runs before it is recorded. The threaded calls are recorded after them, in call order,
        and the refused and the cancelled in their place as they are answered. So the record
        is the same whichever run ended first, though a threaded call's line may follow a
        later call's; each result is paired with its call by id (`request._tail`).

        A run that raises is raised once every other call's answer is recorded: an agent that
        fails never takes its siblings' reports, paid for, with it. A call here that raises
        stops the calls here after it, which neither run nor are recorded, as `execute` would
        leave them."""
        # By position, not by call id: nothing makes a model's ids unique within a response.
        threaded = [
            n
            for n, p in enumerate(dispatched)
            if p.decided is None and p.spec is not None and p.spec.threads
        ]
        if not threaded:
            for p in dispatched:
                yield self.cancel(p) if p.decided is None and stopping() else self.execute(p)
            return
        failed: Exception | None = None
        with ThreadPoolExecutor(len(threaded), thread_name_prefix="tool") as pool:
            running: dict[int, Future[Answer]] = {
                n: pool.submit(self._held, dispatched[n]) for n in threaded
            }
            # What each call run here was recorded as; None for one a stop reached first.
            here: dict[int, R | None] = {}
            for n, p in enumerate(dispatched):
                if n in running or p.decided is not None:
                    continue
                if stopping():
                    here[n] = None
                    continue
                spec = p.spec
                if spec is None:
                    raise HarnessBug(f"call {p.call.id} reached its run with no spec")
                try:
                    here[n] = self._executed(p, spec)
                except Exception as exc:  # raised below, once the rest are recorded
                    failed = exc
                    break
            for n, p in enumerate(dispatched):
                future = running.get(n)
                if future is not None:
                    try:
                        answer = future.result()
                    except Exception as exc:  # raised below, once the rest are recorded
                        failed = failed or exc
                        continue
                    yield self._finish(CallOutcome(p, answer))
                elif p.decided is not None:
                    yield self.execute(p)
                elif n in here:
                    got = here[n]
                    yield self.cancel(p) if got is None else got
        if failed is not None:
            raise failed

    def cancel(self, pending: Pending) -> R:
        """Answer a call the person stopped before it started: `CANCELLED`, never run, and
        recorded and accounted like any answered call. Which stop reaches which call is
        the loop's (a soft stop is only its own), so the loop asks for this."""
        if pending.decided is not None:
            raise HarnessBug(f"call {pending.call.id} was answered; a stop cannot reach it")
        stopped = replace(pending, dispatched=True)
        return self._finish(CallOutcome(stopped, CallCancelled(pending.sent)))

    def call(self, call: ToolCall, *, at: int = 0) -> R:
        """Admit and execute one call, for a role that takes them one at a time."""
        (pending,) = self.admit((call,), at=at)
        return self.execute(pending)

    # -- internals ---------------------------------------------------------------------------

    def _admit_one(self, call: ToolCall, at: int) -> Pending:
        name = self._policy.registry_name(call.name)
        tool = self._registry.get(name) if name in self._offered else None
        pending = Pending(
            call=Invocation(call.id, self._operation_id(call.id), at),
            sent=call.name,
            raw=call.arguments,
            tool=name if tool is not None else None,
            spec=self._policy.resolved(tool.spec) if tool is not None else None,
        )
        if self.stopped is not None:
            return pending.with_verdict(self.stopped)
        for rule in self._policy.pipeline.admit:
            verdict = self._asked(pending, rule.check(pending, self.state))
            pending = pending.with_verdict(verdict)
            if isinstance(verdict, Stop):
                self.stopped = verdict
            if pending.decided is not None:
                break
        if pending.counts:
            levels = self._policy.pipeline.counts_at
            for level in levels:
                self.state.count(level)
            pending = replace(pending, counted=levels)
        return pending

    def _dispatch(self, pending: Pending) -> Pending:
        if pending.decided is not None or pending.dispatched:
            return pending
        pending = replace(pending, dispatched=True)
        for rule in self._policy.pipeline.dispatch:
            verdict = self._asked(pending, rule.check(pending, self.state))
            pending = pending.with_verdict(verdict)
            if isinstance(verdict, Stop):
                self.stopped = verdict
            decided = pending.decided
            if decided is None:
                continue
            if isinstance(decided, Refuse) and not decided.counts:
                # Counted when admitted, refused as it was dispatched: a refusal that does
                # not count is uncounted, as every bound refusal is.
                for level in pending.counted:
                    self.state.count(level, -1)
                pending = replace(pending, counted=())
            break
        return pending

    def _asked(self, pending: Pending, verdict: Verdict) -> Verdict:
        """An `AskPerson` is asked here, through the `user` port: yes continues, no (or nobody
        to ask) is a refusal that counts."""
        if not isinstance(verdict, AskPerson):
            return verdict
        # A hook may refuse before the person is asked, never answer for them.
        gate: AskGate | None = self.state.hooks
        denied = None if gate is None else gate.ask_refused(pending, verdict)
        if denied is not None:
            return denied
        user = self._ctx.user
        if user is not None and user.confirm(verdict.question, verdict.tier):
            return Pass()
        return Refuse(Declined(pending.sent, verdict.tier))

    def _executed(self, pending: Pending, spec: ToolSpec) -> R:
        """`_run` and the record of what it answered, both inside the sink's hold."""
        with self._sink.executing(spec):
            return self._finish(CallOutcome(pending, self._run(pending)))

    def _held(self, pending: Pending) -> Answer:
        """`_run` inside the sink's hold, on the thread that calls it: a threaded call's, whose
        record waits for the loop's thread."""
        spec = pending.spec
        if spec is None:
            raise HarnessBug(f"call {pending.call.id} reached its run with no spec")
        with self._sink.executing(spec):
            return self._run(pending)

    def _run(self, pending: Pending) -> Answer:
        name = pending.tool
        tool = self._registry.get(name) if name is not None else None
        if tool is None:
            raise HarnessBug(f"{pending.sent!r} was admitted with no registered tool")
        arguments = pending.arguments
        if arguments is None:
            raise HarnessBug(f"{tool.spec.name} reached its run unvalidated")
        executor = self._executors.get(tool.spec.name)

        def run() -> Answer:
            if executor is not None:
                return executor(pending)
            return tool.run(pending.call, arguments, self._ctx)

        wrapped: Callable[[], Answer] = run
        for wrapper in reversed(self._policy.pipeline.around):
            wrapped = _wrap(wrapper.around, pending, wrapped, self._ctx)
        return wrapped()

    def _finish(self, outcome: CallOutcome) -> R:
        made = outcome.answer
        for rule in self._policy.pipeline.result:
            outcome = rule.apply(outcome, self.state, self._ctx)
        outcome = _data_as_shown(made, outcome)
        if outcome.stop is not None:
            self.stopped = outcome.stop
        recorded = self._sink.record(outcome.pending.call, outcome)
        for observer in self._policy.pipeline.observe:
            try:
                observer.observe(outcome)
            except Exception:
                # An observer watches; it cannot fail the call it watched.
                _log.exception("tool observer %r failed and was dropped", observer)
        return recorded


def _data_as_shown(made: Answer | None, outcome: CallOutcome) -> CallOutcome:
    """The data an answer was made from, kept only while what it shows is
    unchanged: a result rule that cut, redacted or framed the text narrows it to what no such
    rule touches, so the data never holds more than the model was shown. One place for every
    rule, present and future."""
    answer = outcome.answer
    if not isinstance(answer, TextAnswer | ToolResult) or answer.data is None:
        return outcome
    if _shown(made) == _shown(answer):
        return outcome
    return replace(outcome, answer=replace(answer, data=_narrowed(answer.data)))


def _narrowed(data: ResultData) -> ResultData | None:
    """What of `data` stands once the text it was made from changed: a command's exit code,
    which its header still shows, and nothing of its output; nothing of rows the text may no
    longer show."""
    if isinstance(data, Completed):
        return replace(data, stdout=None, stderr=None)
    return None


def _shown(answer: Answer | None) -> str | None:
    if isinstance(answer, TextAnswer):
        return answer.text
    if isinstance(answer, ToolResult):
        return answer.content
    return None


def _wrap(
    around: Callable[[Pending, Callable[[], Answer], ToolContext], Answer],
    pending: Pending,
    inner: Callable[[], Answer],
    ctx: ToolContext,
) -> Callable[[], Answer]:
    return lambda: around(pending, inner, ctx)
