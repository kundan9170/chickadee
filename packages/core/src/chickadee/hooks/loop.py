"""`before_request` and `after_batch`: the role's hooks as `tools/loop.ask`
asks them (`tools.loop.LoopHooks`), so the loop knows hooks only as two questions.

A `Halt` at `after_batch` says its reason on the batch's last answer, beside any note (D233
amends D227 decision 7): the role's next turn is its answer, and it is told why.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chickadee.client.request import ChatRequest
from chickadee.context.request import request_tokens
from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.payloads import AfterBatch, BeforeRequest, WorkerProgress
from chickadee.hooks.points import HookPoint
from chickadee.hooks.verdicts import Combined, Halt
from chickadee.schema.adapter import ToolCall
from chickadee.tools.loop import BatchSaid, Halted

__all__ = ["RoleLoopHooks"]


@dataclass(frozen=True)
class RoleLoopHooks:
    """`LoopHooks` over one role's hooks. `registry_name` reads a call's name as the dispatcher
    does, so a hook is told the tool, whatever the model wrote; a name no tool has is
    told as it was written."""

    hooks: RoleHooks
    registry_name: Callable[[str], str | None]
    progress: Callable[[], WorkerProgress] | None = None
    """The worker's round so far (`box/progress.TaskProgress.now`), for `after_batch`."""

    def before_request(self, request: ChatRequest) -> Halted | None:
        if not self.hooks.has(HookPoint.BEFORE_REQUEST):
            return None
        payload = BeforeRequest(
            messages=len(request.messages), tokens_est=request_tokens(request)
        )
        return _halted(self.hooks.fire(HookPoint.BEFORE_REQUEST, payload))

    def after_batch(self, calls: tuple[ToolCall, ...]) -> BatchSaid:
        if not self.hooks.has(HookPoint.AFTER_BATCH):
            return BatchSaid()
        payload = AfterBatch(
            tools=tuple(self.registry_name(c.name) or c.name for c in calls),
            progress=self.progress() if self.progress is not None else None,
        )
        said = self.hooks.fire(HookPoint.AFTER_BATCH, payload)
        halt = _halted(said)
        told = (said.note, halt.reason if halt is not None else None)
        note = "\n".join(t for t in told if t) or None
        return BatchSaid(note=note, halt=halt)


def _halted(said: Combined) -> Halted | None:
    """`said`'s halt, naming the hook that gave it, as a tool point's `HOOK_HALTED` does."""
    decided = said.decided
    return Halted(said.by or "hooks", decided.reason) if isinstance(decided, Halt) else None
