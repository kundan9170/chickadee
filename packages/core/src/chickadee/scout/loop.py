"""Layer 2: the scout loop — `Scout.run(question, snapshot, sink,
cancel)`.

A turn is read from `stop_reason` and `tool_calls`, never from text. A turn's calls are
dispatched in order and their results return together — several calls per turn is what makes
a small model competitive here — but not on threads: the graph's SQLite connection
belongs to the thread that opened it.

Every call runs the loop: nothing here reads names out of the question and answers from the
graph alone. A name Zeus can write, Zeus looks up itself.

The loop ends when the model answers, and in four other ways, each of which says so:

- **a cap** — `locate.max_tool_calls`, or the cumulative result tokens. The call past
  the cap is answered `BOUND_REACHED`, the model gets one more turn to report what it has,
  and the report is `truncated: true` — never a silent partial;
- **no progress** — three consecutive calls answered from the dedup cache, handled the same
  way;
- **cancellation** — checked between calls; no further model call is made, and the
  report is `CANCELLED`, partial;
- **the session's spend cap** — no further model call starts; the crossing is the
  session's to report.

A scout that cannot search — no rung, a model without function calling, a rung that fails
before any draft exists — returns `UNAVAILABLE{rung, reason}`, never a snapshot-only
report: Zeus's own ladder is the fallback, and weak candidates shaped like findings are not.

There is no rung above this one. A report that failed validation or ran out of calls
goes back to Zeus as it is: the bigger model on a localization question is Zeus.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

from chickadee.client.errors import ClientError
from chickadee.client.request import Message, Part, TextPart
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.loop import RoleLoopHooks
from chickadee.hooks.payloads import BeforeStop
from chickadee.hooks.points import HookPoint
from chickadee.hooks.stop import at_stop, continued
from chickadee.schema import sha256_hex
from chickadee.schema.adapter import ParseError, ToolCall, Usage
from chickadee.schema.findings import Findings
from chickadee.schema.journal import EventKind, ScoutReportP, ScoutStartP
from chickadee.scout.findings import (
    Checker,
    Draft,
    Outcome,
    Report,
    Unavailable,
    gate,
    parse_draft,
)
from chickadee.scout.rung import ScoutRung, nudge, prompt
from chickadee.scout.sink import ScoutSink
from chickadee.scout.snapshot import Snapshot
from chickadee.scout.tools import LocateCaps, ScoutTools
from chickadee.tools.loop import AskResult, Messages, Reply, ask

if TYPE_CHECKING:
    from chickadee.budget.manager import Ledger

__all__ = ["Scout"]

REASKS = 1
"""One format nudge, and the turn after it is the report."""


@dataclass
class _Run:
    """One call's account; `Scout` itself holds none between calls."""

    question: str
    tokens: int = 0
    cost: float = 0.0
    searched: list[str] = field(default_factory=list[str])
    notes: list[str] = field(default_factory=list[str])


@dataclass
class _Searching:
    """The scout's half of the shared loop (`tools.loop.Role`): its calls go through the
    `locate` pipeline, its answer is the findings draft."""

    tools: ScoutTools
    sink: ScoutSink
    offer_exec: bool
    no_progress: int
    run: _Run

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        """The caps, the escape room and the dedup streak are the `locate` pipeline's:
        a call past them ends the search, and every later one is answered `the
        search ended`."""
        answer = self.tools.call(
            call.name, call.arguments, self.sink, offer_exec=self.offer_exec, cancel=cancel
        )
        if answer.searched is not None:
            self.run.searched.append(answer.searched)
        if answer.ended == "no_progress":
            self.run.notes.append(
                f"{self.no_progress} consecutive calls repeated earlier ones and "
                "were answered from the dedup cache: no progress"
            )
        # An image the reader returned rides beside the text that names it: the model was
        # told it was sent, so it is.
        content: str | tuple[Part, ...] = (
            (TextPart(answer.text), answer.image) if answer.image is not None else answer.text
        )
        return Reply(content, ends=answer.ended is not None)

    def after_end(self) -> str:
        return self.tools.search_ended()

    def parse(self, text: str) -> Draft | ParseError:
        return parse_draft(text)

    def reask(self, error: ParseError) -> str:
        return f"{nudge()}\n{error.hint}"

    def cut(self) -> ParseError:
        return ParseError(
            kind="truncated",
            hint="your output hit max_tokens; send the findings JSON alone, shorter",
        )


class Scout:
    """One scout, parallel calls within it — never a committee. It may live for a
    session: `run` starts each `locate` call with a fresh count, dedup cache, escape budget and
    line cache, and keeps only what the session knows."""

    def __init__(
        self,
        rung: ScoutRung | None,
        tools: ScoutTools,
        checker: Checker,
        caps: LocateCaps,
        *,
        ledger: Ledger | None = None,
        system: str | None = None,
        hooks: RoleHooks | None = None,
    ) -> None:
        self.rung = rung
        self.tools = tools
        self.checker = checker
        self.caps = caps
        self.ledger = ledger
        self._system = system
        self.hooks = hooks or RoleHooks.none("locate")
        """Its hooks: `before_request` and `after_batch` as `ask` asks them, its stop
        here; its calls' points are its tools' (`ScoutTools`)."""

    def run(
        self,
        question: str,
        snapshot: Snapshot,
        sink: ScoutSink,
        cancel: threading.Event | None = None,
    ) -> Report:
        self.tools.begin(self.caps)
        self.checker.forget()
        model = self.rung.model if self.rung is not None else "none"
        sink.record(
            EventKind.SCOUT_START,
            ScoutStartP(rung=model, goal_sha=sha256_hex(question), snapshot_sha=snapshot.sha),
        )
        if self.rung is None:
            return self._unavailable(
                sink, question, _Run(question), "unreachable", "no scout rung is configured"
            )
        if not self.rung.native_tools:
            # The TOOL text protocol carries only the coder's tools, so this model would
            # search with nothing and answer from the snapshot alone — a guess, not a search.
            return self._unavailable(
                sink,
                question,
                _Run(question),
                "no_tool_calling",
                f"{self.rung.model} is configured native_tools = false, and the TOOL text "
                "protocol does not carry the scout's tools",
            )
        return self._loop(self.rung, question, snapshot, sink, cancel)

    # -- layer 2 ------------------------------------------------------------------------

    def _loop(
        self,
        rung: ScoutRung,
        question: str,
        snapshot: Snapshot,
        sink: ScoutSink,
        cancel: threading.Event | None,
    ) -> Report:
        escapes = self.tools.escapes
        offer_exec = escapes is not None and escapes.runnable()
        system = self._system if self._system is not None else prompt()
        conversation = Messages(
            [
                Message(role="system", content=system),
                Message(role="user", content=f"{snapshot.render()}\n\n## QUESTION\n{question}"),
            ],
            rung.params(),
            self.tools.schemas(offer_exec=offer_exec),
        )
        state = _Run(question)
        searching = _Searching(self.tools, sink, offer_exec, self.caps.no_progress, state)
        ledger = self.ledger
        loop_hooks = RoleLoopHooks(self.hooks, self.tools.registry_name)
        charge = None if ledger is None else partial(self._charge, ledger, rung)

        def asking() -> AskResult[Draft]:
            return ask(
                rung.client,
                conversation,
                searching,
                reasks=REASKS,
                charge=charge,
                charge_failed=charge,
                may_start=None if ledger is None else ledger.may_start,
                cancel=cancel,
                hooks=loop_hooks,
            )

        taken = 0

        def add(said: AskResult[Draft], words: str) -> None:
            # Its findings, then the hook's words, and a fresh `ask` with its tools open.
            nonlocal taken
            conversation.add(
                Message(role="assistant", content=_drafted(said)),
                Message(role="user", content=words),
            )
            taken += 1

        asked = continued(
            asking(),
            words=lambda said: self._go_on(said, taken),
            add=add,
            again=lambda said: asking().after(said),
        )
        state.tokens, state.cost = asked.tokens, asked.cost_usd
        if asked.end == "failed" and asked.failure is not None:
            return self._failed(sink, question, state, asked.model_calls, asked.failure)
        if asked.end == "spend_cap":
            state.notes.append("the session's spend cap is crossed; no further call started")
        elif asked.end == "unparsed" and asked.error is not None:
            state.notes.append(
                f"the findings did not parse ({asked.error.kind}); none were taken"
            )
        elif asked.end == "kept_calling":
            state.notes.append(
                "the model kept calling tools after the search ended; no findings"
            )
        if (halted := asked.halted) is not None:
            state.notes.append(f"hook {halted.by} ended the search: {halted.reason}")
        cancelled = asked.end == "cancelled"
        truncated = asked.tools_ended or asked.end in ("spend_cap", "halted")
        findings, dropped_p, dropped_f = gate(
            asked.answer or Draft(), self.checker, truncated=truncated or cancelled
        )
        outcome: Outcome = (
            "cancelled" if cancelled else ("found" if findings.targets else "unresolved")
        )
        return self._finish(
            sink, question, findings, outcome, state, dropped=(dropped_p, dropped_f)
        )

    def _go_on(self, asked: AskResult[Draft], taken: int) -> str | None:
        """`before_stop` at its findings: what the search goes on with, if its hooks say it
        does not stop yet. One `locate` is one call, never resumed, so the count is its own."""
        if asked.end != "answered" or not self.hooks.has(HookPoint.BEFORE_STOP):
            return None
        stop = BeforeStop(reply=_drafted(asked), tools_open=not asked.tools_ended)
        return at_stop(self.hooks, stop, used=taken).words

    def _failed(
        self, sink: ScoutSink, question: str, state: _Run, model_calls: int, exc: ClientError
    ) -> Report:
        """The rung failed. Before any model call it was never reachable; after one, no draft
        exists yet — the draft is the final turn — so nothing validated is left to return, and
        an empty report would read as a search that finished and found nothing."""
        if model_calls == 0:
            return self._unavailable(sink, question, state, "unreachable", str(exc))
        return self._unavailable(
            sink,
            question,
            state,
            "failed_mid_search",
            f"after {model_calls} model call(s) and {self.tools.calls} tool call(s): {exc}",
        )

    @staticmethod
    def _charge(ledger: Ledger, rung: ScoutRung, usage: Usage) -> float:
        return ledger.charge_usage("", usage, rung.model)

    # -- the end ------------------------------------------------------------------------

    def _unavailable(
        self, sink: ScoutSink, question: str, state: _Run, why: Unavailable, detail: str
    ) -> Report:
        """The typed failure, paired with a `scout_report` so no discovery record is
        left open. The record says `truncated`: its frozen payload has no outcome field, and
        zero targets with `truncated: false` would record a search that finished and found
        nothing — the reading `UNAVAILABLE` exists to prevent."""
        return self._finish(
            sink,
            question,
            Findings(truncated=True),
            "unavailable",
            state,
            dropped=(0, 0),
            why=why,
            detail=detail,
        )

    def _finish(
        self,
        sink: ScoutSink,
        question: str,
        findings: Findings,
        outcome: Outcome,
        state: _Run,
        *,
        dropped: tuple[int, int],
        why: Unavailable | None = None,
        detail: str = "",
    ) -> Report:
        escapes = self.tools.escapes
        report = Report(
            question=question,
            findings=findings,
            outcome=outcome,
            model=self.rung.model if self.rung is not None else "none",
            calls=self.tools.calls,
            escapes=escapes.used if escapes is not None else 0,
            tokens=state.tokens,
            cost_usd=state.cost,
            dropped_pointers=dropped[0],
            dropped_facts=dropped[1],
            searched=tuple(dict.fromkeys(state.searched)),
            notes=tuple(state.notes),
            unavailable=why,
            detail=detail,
        )
        sink.record(
            EventKind.SCOUT_REPORT,
            ScoutReportP(
                targets=len(findings.targets),
                facts=len(findings.facts),
                unresolved=len(findings.unresolved),
                truncated=findings.truncated,
                tokens=state.tokens,
                cost_usd=state.cost,
            ),
        )
        return report


def _drafted(asked: AskResult[Draft]) -> str:
    """The findings the scout answered with, as it wrote them back to itself."""
    return asked.answer.model_dump_json() if asked.answer is not None else ""
