"""`before_stop`: the one stop path, for the worker, Athena, the scout
and Zeus, where each accepts its answer.

- `at_stop` asks the role's hooks. A `Continue` is offered only while the role may still call
  tools, and only up to `hooks.max_continues` at that stop: past either, it is journaled as
  bounded and not taken, and the role stops as it would with no hook. The caller says how many
  were already taken at this stop, read from the journal where the role's stop outlives a crash
  (`RoleHooks.used`).
- `StopSaid.words` is what the role is sent when it goes on: the one spelling of a stop's
  `Continue` and its notes together, live or read back after a crash (`stopped_at`).
- `continued` is the one loop over a role's `ask`: while its stop says go on, the reply and the
  words are added, in that order, and the role is asked again with its tools open. Zeus's turn
  is not an `ask`, and takes `words` alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from chickadee.hooks.fire import RoleHooks
from chickadee.hooks.payloads import BeforeStop
from chickadee.hooks.points import HookPoint
from chickadee.hooks.verdicts import Combined, Continue, HookVerdict
from chickadee.tools.loop import AskResult

__all__ = ["StopSaid", "at_stop", "continued", "stopped_at"]


@dataclass(frozen=True)
class StopSaid:
    """What the hooks said at a stop: the role's next message, if it goes on, and a note for
    whoever reads next."""

    go_on: str | None = None
    note: str | None = None

    @classmethod
    def of(cls, said: Combined) -> StopSaid:
        if isinstance(said.decided, Continue):
            return cls(go_on=said.decided.text, note=said.note)
        return cls(note=said.note)

    @property
    def words(self) -> str | None:
        """The role's next message when it goes on: the `Continue`, then the stop's notes."""
        if self.go_on is None:
            return None
        return self.go_on if self.note is None else f"{self.go_on}\n\n{self.note}"


NOTHING = StopSaid()


def at_stop(hooks: RoleHooks, stop: BeforeStop, *, used: int, room: bool = True) -> StopSaid:
    """What the role's hooks say at `stop`. `used` is how many `Continue`s were already taken
    at this stop; `room`, whether the run may still steer itself (Athena's `max_steering`,
    decision 8), which holds only the hooks that steer (`BoundHook.steers`)."""
    if not hooks.has(HookPoint.BEFORE_STOP):
        return NOTHING
    left = max(hooks.limits.max_continues - used, 0) if stop.tools_open else 0
    told = left if room else 0
    payload = stop.model_copy(update={"continues_used": used, "continues_left": told})
    bounded: frozenset[type[HookVerdict]] = frozenset() if left > 0 else frozenset({Continue})
    steering: frozenset[type[HookVerdict]] = frozenset() if room else frozenset({Continue})
    said = hooks.fire(HookPoint.BEFORE_STOP, payload, bounded=bounded, steering=steering)
    return StopSaid.of(said)


def stopped_at(
    hooks: RoleHooks, *, task: str, round_: int, between: tuple[int, int]
) -> StopSaid:
    """What a stop whose lines were written `between` two seqs said, read back: the words a
    round cut off before them is owed (`resume.owed_continuation`)."""
    lines = hooks.select(HookPoint.BEFORE_STOP, task_id=task, round=round_, between=between)
    return StopSaid.of(hooks.combined(lines))


def continued[A](
    asked: AskResult[A],
    *,
    words: Callable[[AskResult[A]], str | None],
    add: Callable[[AskResult[A], str], None],
    again: Callable[[AskResult[A]], AskResult[A]],
) -> AskResult[A]:
    """`asked`, asked again while the role's stop says go on. `words` is the role's stop (its
    `at_stop(...).words`), `add` puts its reply and the words on its conversation, and `again`
    asks once more, given the ask it continues."""
    while (said := words(asked)) is not None:
        add(asked, said)
        asked = again(asked)
    return asked
