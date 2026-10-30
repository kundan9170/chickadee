"""The level-2 points: what the desk and the episode ask the hooks,
and the bounds on what they may steer.

- `round_ended`: a worker round's report, before Athena reads it. A `RetryTask` is taken only
  after a round that ended `replied`, `limit` or `looped`, while the task has calls and seconds
  left, under `hooks.max_retries` for the task and `hooks.max_steering` for the episode; past
  any of these it is journaled as bounded and the report goes to Athena as it would with no
  hook. The desk then writes Athena's own `feedback` line for it (`retry_line`), and acts on it
  as on hers.
- `hook_review`: the retry line a hook's retry wrote for a round, recognised by the id
  `retry_line` gives it, so the module that writes the id is the one that reads it.
- `harness_notes`: every note a hook gave about one round, read back from the journal, so
  Athena reads them whether the round ran now or before a crash.

The desk is Athena's, so its points are about her role: her stops and the retries a hook gave
count toward one episode's steering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.payloads import RoundEnded
from chickadee.hooks.points import HookPoint
from chickadee.hooks.verdicts import KINDS, HookVerdict, Note, RetryTask
from chickadee.journal.projection import RunProjection
from chickadee.schema.journal import RoundEnd, TaskReviewedP

__all__ = ["RETRYABLE", "harness_notes", "hook_review", "retry_line", "round_ended"]

RETRYABLE: Final[frozenset[RoundEnd]] = frozenset({"replied", "limit", "looped"})
"""The ends a retry can change. After `context_full` the conversation is still full, and
`model_error`, `harness_error` and `halted` are said as they are (decision 7)."""

_NOTED_AT: Final = (HookPoint.BEFORE_STOP, HookPoint.ROUND_ENDED)
"""Where a hook's note about a round is given: the worker's stop, and the round's end."""


def round_ended(
    hooks: RoleHooks, payload: RoundEnded, *, room: bool
) -> tuple[str, RetryTask] | None:
    """The retry the hooks gave `payload`'s round, and the hook that gave it, if one was taken.
    `room` is whether the task has calls and seconds left for another round.

    A round its hooks already answered (a resume, the desk acting again on a reported round)
    is read back from its lines, never asked again. A retry whose line was written and whose
    review line was not is taken from its words with `fresh` False: the line does not keep
    `fresh`, and keeping the work loses nothing a put-back would not."""
    if not hooks.has(HookPoint.ROUND_ENDED):
        return None
    task, round_ = payload.task, payload.round
    fired = hooks.select(HookPoint.ROUND_ENDED, task_id=task, round=round_)
    if fired:
        taken = [s for s in fired if s.decision == KINDS[RetryTask] and not s.bounded]
        return (taken[0].name, RetryTask(hooks.text(taken[0]), fresh=False)) if taken else None
    limits = hooks.limits
    used = hooks.used(HookPoint.ROUND_ENDED, RetryTask, task_id=task)
    may = payload.stopped in RETRYABLE and room and used < limits.max_retries
    steers = hooks.steering_room()
    left = max(limits.max_retries - used, 0) if may and steers else 0
    told = payload.model_copy(update={"retries_used": used, "retries_left": left})
    bounded: frozenset[type[HookVerdict]] = frozenset() if may else frozenset({RetryTask})
    steering: frozenset[type[HookVerdict]] = frozenset() if steers else frozenset({RetryTask})
    said = hooks.fire(
        HookPoint.ROUND_ENDED,
        told,
        task_id=task,
        round=round_,
        bounded=bounded,
        steering=steering,
    )
    decided = said.decided
    if isinstance(decided, RetryTask):
        return said.by or "hooks", decided
    return None


def _retry_id(by: str, task: str, round_: int) -> str:
    return f"hook:{by}:{task}:{round_}"


def retry_line(
    by: str, task: str, round_: int, text_sha: str, retry: RetryTask
) -> TaskReviewedP:
    """Athena's own `feedback`, as the harness writes it for a hook's retry (decision 5): its
    `call_id` derived from the round it answers, so a resume finds it (`hook_review`), and
    `by` naming the hook. `fresh` is kept as her review keeps it (`desk._fresh`)."""
    return TaskReviewedP(
        task_id=task,
        call_id=_retry_id(by, task, round_),
        answer="feedback",
        text_sha=text_sha,
        fresh=retry.fresh,
        by=f"hook:{by}",
    )


def hook_review(run: RunProjection, task: str, round_: int) -> TaskReviewedP | None:
    """The `feedback` a hook's retry wrote for `task`'s `round_` (`retry_line`), if one was:
    its id rebuilt from its `by`, never matched by a suffix."""
    return next(
        (
            line
            for p in run.tasks.get(task, ())
            if isinstance(line := p.payload, TaskReviewedP)
            and line.by is not None
            and line.by.startswith("hook:")
            and line.call_id == _retry_id(line.by.removeprefix("hook:"), task, round_)
        ),
        None,
    )


def harness_notes(hooks: RoleHooks, *, task: str, round_: int) -> tuple[str, ...]:
    """Every note a hook gave about `task`'s `round_`: at the worker's stop and at its end."""
    lines: Sequence[str] = [
        hooks.text(said)
        for point in _NOTED_AT
        for said in hooks.select(point, task_id=task, round=round_)
        if said.decision == KINDS[Note]
    ]
    return tuple(lines)
