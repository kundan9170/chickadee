"""The machine's sleep, as a model call sees it: the one home of the clock that counts it, of
noticing that it happened, and of the connections it leaves dead.

A laptop closed mid-request leaves the request's connection dead: the server, or something
between, drops it while the machine sleeps, and the client is never told. Two things then kept
a session waiting long after the person was back (2026-10-23):

- **The silences counted only time awake.** `time.monotonic()` stops while the machine sleeps
  (Linux's `CLOCK_MONOTONIC`, and the Mach clock Python uses on macOS), and so do the queue
  waits built on it. A request in flight at sleep still had its whole first-token or token-gap
  limit to wait out after waking. `WAITED` counts the sleep, so the limits measure what the
  person waits: after a long sleep the dead request is past its limit at once, and retried.
- **A connection kept for reuse may be dead after it.** `ModelConnections` replaces a client's
  connections the first time it is used after a sleep, so the next request opens a fresh one
  instead of waiting on one nobody will answer.

A platform with no clock that counts sleep falls back to `time.monotonic()`: no sleep is ever
seen there, and nothing changes from before.
"""

from __future__ import annotations

import functools
import sys
import threading
import time
from collections.abc import Callable
from typing import Final

import httpx2

__all__ = ["SLEPT_S", "WAITED", "ModelConnections", "SleepWatch"]


def _sleep_counting() -> Callable[[], float]:
    if sys.platform == "linux" and hasattr(time, "CLOCK_BOOTTIME"):
        return functools.partial(time.clock_gettime, time.CLOCK_BOOTTIME)
    if sys.platform == "darwin" and hasattr(time, "CLOCK_MONOTONIC"):
        # Darwin's CLOCK_MONOTONIC counts sleep; `time.monotonic()` there reads the Mach
        # absolute time, which does not.
        return functools.partial(time.clock_gettime, time.CLOCK_MONOTONIC)
    return time.monotonic


WAITED: Final[Callable[[], float]] = _sleep_counting()
"""Seconds that keep counting while the machine sleeps: what a person waiting has waited."""

SLEPT_S: Final = 5.0
"""How far `WAITED` must run ahead of the awake clock to count as a sleep: above the drift two
clocks show in a busy second, below any lid closed long enough to drop a connection."""


class SleepWatch:
    """Whether the machine slept since the last look, by how far `waited` ran ahead of `awake`.
    Not shared between threads: each `ModelConnections` holds its own under its lock."""

    def __init__(
        self,
        waited: Callable[[], float] = WAITED,
        awake: Callable[[], float] = time.monotonic,
    ) -> None:
        self._waited = waited
        self._awake = awake
        self._at = (waited(), awake())

    def slept(self) -> bool:
        now = (self._waited(), self._awake())
        ahead = (now[0] - self._at[0]) - (now[1] - self._at[1])
        self._at = now
        return ahead > SLEPT_S


class ModelConnections:
    """A model client's HTTP connections, made anew the first time they are asked for after a
    sleep. The old ones are closed: any request still on them is past its silence by then, so
    its wait has already ended or is about to (`client/stream.py`)."""

    def __init__(
        self, make: Callable[[], httpx2.Client], watch: SleepWatch | None = None
    ) -> None:
        self._make = make
        self._watch = watch if watch is not None else SleepWatch()
        self._lock = threading.Lock()
        self._http = make()

    def get(self) -> httpx2.Client:
        with self._lock:
            if self._watch.slept():
                stale, self._http = self._http, self._make()
                stale.close()
            return self._http

    def close(self) -> None:
        with self._lock:
            self._http.close()
