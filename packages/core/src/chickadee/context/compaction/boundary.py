"""Where a cut falls: the boundary, the recent turns kept,
and the fallback when no completed unit allows one. No cut falls between a call and its result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from chickadee.context.project import completed_units, unit_episodes
from chickadee.errors import HarnessBug

if TYPE_CHECKING:
    from chickadee.context.project import Unit
    from chickadee.context.session import SessionState

__all__ = [
    "MIN_COMPLETED_UNITS",
    "RETAIN_RECENT_TURNS",
    "Boundary",
    "Decline",
    "choose_boundary",
    "fallback_boundary",
    "recent_floor",
]


RETAIN_RECENT_TURNS: Final = 2
"""§7.6.4 rule 4. The newest completed unit is protected by `choose_boundary`; this guards the
in-flight tail. A turn here is a user or assistant line."""


MIN_COMPLETED_UNITS: Final = 2
"""One unit gives no safe boundary."""


class Decline(StrEnum):
    NO_SAFE_BOUNDARY = "no_safe_boundary"
    """One long unbroken task: a legitimate outcome, and layer 3's fallback path."""
    BOUNDARY_TOO_RECENT = "boundary_too_recent"
    """The boundary would touch the recent turns."""


@dataclass(frozen=True)
class Boundary:
    turn: int
    """The first seq kept verbatim: the start of the newest completed unit, which the user
    may still be reacting to."""
    units: tuple[Unit, ...]
    """Every completed unit before it, projected in this one cut."""


def recent_floor(state: SessionState) -> int | None:
    """The oldest seq a cut must keep: the `RETAIN_RECENT_TURNS`-th newest user or assistant
    turn. None when there are not that many — everything is recent."""
    seqs = [t.seq for t in state.turns() if t.role != "harness"]
    return seqs[-RETAIN_RECENT_TURNS] if len(seqs) >= RETAIN_RECENT_TURNS else None


def choose_boundary(state: SessionState, previous: int) -> Boundary | Decline:
    """§5.5. `previous` is the last cut's boundary, or 0: a unit wholly before it is already
    projected, and projecting it again would feed a projection back through the projector.

    The boundary this returns is sticky: the fold keeps it until the next cut commits, since
    a boundary that advanced every turn would be a fresh rebuild every turn."""
    units = completed_units(state, after=previous)
    if len(units) < MIN_COMPLETED_UNITS:
        return Decline.NO_SAFE_BOUNDARY
    boundary = units[-1].turn_start
    floor = recent_floor(state)
    if floor is None or boundary > floor:
        return Decline.BOUNDARY_TOO_RECENT
    _assert_no_live_unit(state, boundary)
    _assert_paired(state, boundary)
    return Boundary(boundary, units[:-1])


def fallback_boundary(state: SessionState, previous: int) -> int | None:
    """Job A's boundary: everything older than the recent turns. It is a user or
    assistant turn, and the fold refuses one of those while a call waits, so no call before
    it has its result after it. None when nothing past `previous` is old enough to cut."""
    floor = recent_floor(state)
    if floor is None or floor <= max(previous, 1):
        return None
    _assert_paired(state, floor)
    return floor


def _assert_paired(state: SessionState, boundary: int) -> None:
    """X-29, an assert rather than a comment: a provider rejects a `tool_use` without its
    `tool_result` for the rest of the session, and the market has shipped that bug."""
    for call in state.calls_upto(boundary - 1):
        result = state.result(call.call_id)
        if result is None or result.seq >= boundary:
            raise HarnessBug(f"a cut at seq {boundary} would separate call {call.call_id}")


def _assert_no_live_unit(state: SessionState, boundary: int) -> None:
    """§7.6.6 rule 1: no in-flight unit begins before the boundary."""
    ended = {u.run_id: u.seq_end for u in completed_units(state)}
    for episode in unit_episodes(state):
        opened = episode.opened_at_seq
        assert opened is not None  # unit_episodes keeps Zeus's alone
        if opened < boundary and ended.get(episode.run_id, boundary) >= boundary:
            raise HarnessBug(f"run {episode.run_id} is in flight across seq {boundary}")
