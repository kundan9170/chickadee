"""A box role's half of each tool call and model call (`tools/sink.ToolSink`): one `tool_call`
line per call and one `model_call` line per answered model call, with previews, at the
task they are about.

Moved from `planner/client._AthenaSink`, with its role, so Athena and the worker are
lined alike. Both always have a journal: the session log's `box_call` lines, for Athena's
calls before her run had one, retired with the packet loop.
"""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chickadee.boxctx.sent import SentRequests, record_sent
from chickadee.context.result_data import ResultData
from chickadee.context.results import ToolResult
from chickadee.journal.metrics import python_note
from chickadee.journal.previews import args_preview, preview
from chickadee.journal.projection import RunProjection
from chickadee.schema.journal import BoxRoleName, EventKind, ModelCallP, ToolCallP
from chickadee.tools.catalogue import answer_text, args_sha, refused, result_tokens
from chickadee.tools.sink import Counters
from chickadee.tools.tool import TextAnswer

if TYPE_CHECKING:
    from chickadee.journal.port import JournalPort
    from chickadee.schema.adapter import Completion, ParseError
    from chickadee.tools.rules import CallOutcome, Level
    from chickadee.tools.spec import ToolSpec
    from chickadee.tools.tool import Invocation

__all__ = ["BoxServed", "BoxSink"]


@dataclass(frozen=True)
class BoxServed:
    """One call's text as the role reads it, and the data it was made from when a rule left
    the text as made, for the worker's Python blocks."""

    text: str
    data: ResultData | None = None
    refused: bool = False
    """The call was refused or failed (`catalogue.refused`), as its `denied_reason` says."""


class BoxSink:
    """`ToolSink[BoxServed]`: the role holds nothing while a tool runs. Its lines go to the run
    journal, at the task they are about. Each request starts its counts at zero."""

    def __init__(
        self,
        role: BoxRoleName,
        journal: JournalPort,
        rung: str | None = None,
        model: str | None = None,
    ) -> None:
        self.role: BoxRoleName = role
        self.journal: JournalPort = journal
        self.rung = rung
        """The rung the role's model calls are charged to, on each `model_call` line."""
        self.model = model
        """The catalog model the rung calls (`BoxRung.model`), on each `model_call` line.
        A role's rung is fixed for its life: a `/rung` switch takes effect
        from the next episode, which builds the role again."""
        self.sent: SentRequests | None = None
        """The role's transport recorder: each `model_call` names what it sent."""
        self.task_id: str | None = None
        """The task loop's task the role's lines are about; None outside one."""
        self.round: int | None = None

    def line(self, payload: ToolCallP | ModelCallP) -> None:
        """One record of the role's, at the task it is about."""
        kind = EventKind.TOOL_CALL if isinstance(payload, ToolCallP) else EventKind.MODEL_CALL
        self.journal.append(
            kind,
            payload,
            task_id=self.task_id,
            round=self.round if self.task_id is not None else None,
        )

    def model_call(
        self,
        request: str,
        completion: Completion,
        cost: float,
        refusal: ParseError | None,
    ) -> None:
        """One answered model call, at the task it was made in, with the error its answer was
        refused on when it was."""
        self.line(
            ModelCallP(
                role=self.role,
                request=request,
                tokens_in=completion.usage.prompt_tokens,
                tokens_out=completion.usage.completion_tokens,
                cost_usd=cost,
                stop_reason=completion.stop_reason.value,
                tools=tuple(c.name for c in completion.tool_calls),
                text_preview=preview(completion.text) if completion.text else None,
                refusal=None if refusal is None else preview(f"{refusal.kind}: {refusal.hint}"),
                rung=self.rung,
                sent_sha=self._sent(),
                model=self.model,
            )
        )

    def _sent(self) -> str | None:
        request = None if self.sent is None else self.sent.take()
        return None if request is None else record_sent(self.journal, request)

    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        del spec
        return nullcontext()

    def render(self, outcome: CallOutcome) -> str | ToolResult:
        return answer_text(outcome)

    def record(self, call: Invocation, outcome: CallOutcome) -> BoxServed:
        del call
        text = answer_text(outcome)
        pending = outcome.pending
        denied = refused(outcome.answer)
        self.line(
            ToolCallP(
                role=self.role,
                tool=pending.sent,
                args_sha=args_sha(pending.raw),
                outcome="served",
                result_tokens=0 if denied else result_tokens(outcome),
                result_sha=None if denied else hashlib.sha256(text.encode("utf-8")).hexdigest(),
                denied_reason=text if denied else None,
                args_preview=args_preview(pending.raw),
                # A refusal is its `denied_reason`, whole; not written twice.
                result_preview=preview(text) if text and not denied else None,
            )
        )
        answer = outcome.answer
        data = answer.data if isinstance(answer, TextAnswer | ToolResult) else None
        return BoxServed(text, None if denied else data, denied)

    def replay(self, level: Level) -> Counters:
        """A `task` scope starts from the task's `tool_call` lines of this role, so a resumed
        round keeps its count and the cap holds across a crash, as the spend cap does; every
        other scope starts at zero, as Athena's request always has."""
        if level != "task" or self.task_id is None:
            return Counters()
        return task_counters(self.journal.projection(), self.role, self.task_id)


def task_counters(run: RunProjection, role: str, task: str) -> Counters:
    """What `role` spent of `task`'s call budget, from its `tool_call` lines: the one count,
    which the dispatcher's `task` scope starts from (`BoxSink.replay`) and a hook is told
    (`WorkerRun.calls_left`)."""
    mine = [
        p.payload
        for p in run.tasks.get(task, ())
        if isinstance(p.payload, ToolCallP)
        and p.payload.role == role
        and not python_note(p.payload)
    ]
    return Counters(calls=len(mine), result_tokens=sum(c.result_tokens for c in mine))
