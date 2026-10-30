"""Ctrl-C while a turn runs: stop at the next boundary, or now.

The first press asks the engine to stop the turn at its next boundary
(`client.interrupt`): what it committed stays, and words already queued are kept. A second
press within `HARD_WINDOW_S` stops it hard (`hard=True`). The engine owns what either
means; this module only counts presses.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final, Literal

from chickadee_wire.client import Client
from chickadee_wire.errors import WireError

__all__ = ["HARD_WINDOW_S", "Interrupter", "stop_if_working"]

HARD_WINDOW_S: Final = 2.0
"""A second Ctrl-C this soon after the first is a hard stop."""


class Interrupter:
    def __init__(
        self,
        client: Client,
        session_id: str,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._sid = session_id
        self._clock = clock
        self._last: float | None = None

    def press(self) -> Literal["boundary", "hard"]:
        """One Ctrl-C during a turn, sent; which stop it asked for."""
        now = self._clock()
        hard = self._last is not None and now - self._last <= HARD_WINDOW_S
        self._last = None if hard else now
        self._client.interrupt(self._sid, hard=hard)
        return "hard" if hard else "boundary"


def stop_if_working(
    client: Client, session_id: str, interrupter: Interrupter
) -> Literal["boundary", "hard"] | None:
    """One stop key, pressed (the REPL's Ctrl-C, the TUI's Esc): sent only while the session
    works, a turn waiting on a question included; which stop it asked for, or None when
    nothing was running. The one home of the rule, so both keys stop the same thing."""
    try:
        working = client.liveness(session_id).kind in ("working", "awaiting")
    except WireError:
        working = False
    return interrupter.press() if working else None
