"""The around stage's seams: how a call runs. Each wraps the run
itself, composed in pipeline order with the first outermost, and none is in any role's
default today: the shell and the box record their own lifecycles, and each tool keeps its own
timeout (decision 5).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chickadee.errors import HarnessBug, InfraError
from chickadee.journal.effects import EffectKind, Intent, perform
from chickadee.tools.catalogue import args_sha, failed
from chickadee.tools.rules import Around, Pending
from chickadee.tools.tool import Answer, CallTimedOut

if TYPE_CHECKING:
    from chickadee.tools.context import ToolContext

__all__ = ["EffectLifecycle", "RetryIdempotent", "Timeout"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectLifecycle:
    """D164 decision 5a: a `dispatcher`-effect tool runs inside `journal.effects.perform`,
    keyed on `Invocation.operation_id`, with the spec's `idempotent` as the recovery rule. A
    tool that records its own lifecycle (`self`) or none passes through."""

    kind: EffectKind

    def around(self, pending: Pending, run: Around, ctx: ToolContext) -> Answer:
        spec = pending.spec
        if spec is None or spec.effects != "dispatcher":
            return run()
        if ctx.effect_log is None:
            raise HarnessBug(f"{spec.name} records its effect and the role gave no journal")
        intent = Intent(
            operation_id=pending.call.operation_id,
            kind=self.kind,
            approved_sha=args_sha(pending.raw),
            preconditions=(),
            idempotent=spec.idempotent,
            surface=(spec.name,),
        )
        with perform(ctx.effect_log, intent) as effect:
            answer = run()
            effect.complete("succeeded" if not failed(answer) else "failed")
            return answer


@dataclass(frozen=True)
class Timeout:
    """A wall-clock bound: when it runs out the cancel event is set, and
    a run that stops because of it answers `TIMED_OUT`. The tool honours the event; nothing
    here kills a process (the supervisor does)."""

    seconds: float

    def around(self, pending: Pending, run: Around, ctx: ToolContext) -> Answer:
        cancel = ctx.cancel
        if cancel is None:
            raise HarnessBug("a Timeout needs the context's cancel event")
        fired = threading.Event()

        def expire() -> None:
            fired.set()
            cancel.set()

        timer = threading.Timer(self.seconds, expire)
        timer.start()
        try:
            answer = run()
        finally:
            timer.cancel()
        if fired.is_set():
            return CallTimedOut(pending.sent, self.seconds)
        return answer


@dataclass(frozen=True)
class RetryIdempotent:
    """Run again on `InfraError`, at most `times` more, and only for a spec marked
    `idempotent`: a non-idempotent effect is never blindly retried."""

    times: int

    def around(self, pending: Pending, run: Around, ctx: ToolContext) -> Answer:
        spec = pending.spec
        if spec is None or not spec.idempotent:
            return run()
        for _ in range(self.times):
            try:
                return run()
            except InfraError:
                _log.warning("retrying %s after an infrastructure failure", spec.name)
        return run()
