"""The one motion on the screen: `●`, a model turn in flight, pulses
once a second, and nothing else animates (a frame budget over ssh).

**Never a spinner forever**. The pulse runs only while the record says a turn is in
flight, and stops once the session is not live: liveness `abandoned` or `unknown` ends it,
whatever the last record was. A dimmed `●` keeps its word beside it (colour is never the
only signal).
"""

from __future__ import annotations

from typing import Final

from textual.app import App
from textual.timer import Timer
from textual.widgets import Static

from chickadee_cli.tui.model import TuiState

__all__ = ["HALF_PERIOD_S", "PULSE_DIM", "Pulse", "PulseDot", "in_flight"]

HALF_PERIOD_S: Final = 0.5
"""Lit then dim, so one pulse a second."""
PULSE_DIM: Final = "-pulse-dim"
"""The screen's class while the pulse is in its dim half."""

_NOT_LIVE: Final = frozenset({"abandoned", "unknown"})


def in_flight(state: TuiState) -> bool:
    """A model turn is in flight: Zeus's, or the box's while its run is open and working."""
    live = state.liveness
    if live is not None and live.kind in _NOT_LIVE:
        return False
    if state.zeus.busy is not None:
        return True
    return state.episode_open and (live is None or live.kind == "working")


class PulseDot(Static):
    """`●` beside the word for what is in flight; the pulse dims it every other half second.
    Parts 3 and 4 put one beside the worker's timer and Zeus's busy line."""

    DEFAULT_CSS = """
    PulseDot { width: auto; }
    Screen.-pulse-dim PulseDot { text-style: dim; }
    """


class Pulse:
    """The app's one timer, running only while `in_flight` holds."""

    def __init__(self, app: App[None]) -> None:
        self._app = app
        self._timer: Timer | None = None
        self._dim = False
        self._ticks = 0

    @property
    def running(self) -> bool:
        return self._timer is not None

    @property
    def ticks(self) -> int:
        """Half-periods since the app started: how a test sees it pulse without a clock."""
        return self._ticks

    def follow(self, state: TuiState) -> None:
        """Start or stop with the state."""
        if in_flight(state) and self._timer is None:
            self._timer = self._app.set_interval(HALF_PERIOD_S, self._tick)
        elif not in_flight(state) and self._timer is not None:
            self._timer.stop()
            self._timer = None
            self._set(dim=False)

    def _tick(self) -> None:
        self._ticks += 1
        self._set(dim=not self._dim)

    def _set(self, *, dim: bool) -> None:
        self._dim = dim
        self._app.screen.set_class(dim, PULSE_DIM)
