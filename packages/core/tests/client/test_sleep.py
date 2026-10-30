"""The machine's sleep, as a model call sees it (`client/sleep.py`, 2026-10-23).

A laptop closed mid-request left the session waiting a whole first-token limit, five minutes,
after it was opened again: the limits ran on a clock that stops in sleep, and the request's
connection was dead. The limits now count the sleep, and the connections are made anew after
one.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx2
import pytest

from chickadee.client.sleep import SLEPT_S, ModelConnections, SleepWatch
from chickadee.client.stream import Silences, Silent, read_stream


class _Hung:
    """A stream whose server never answers: its iteration waits until it is closed."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[object]:
        self.closed.wait(30)
        return iter(())

    def close(self) -> None:
        self.closed.set()


class _Clock:
    """A clock a test moves: the machine's sleep, as `WAITED` counts it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_sleep_longer_than_the_limit_ends_the_wait_soon_after_waking() -> None:
    """The request was in flight when the lid closed; the clock that counts sleep jumps past
    the first-token limit, and the wait ends within a wake check, not after the limit again."""
    hung, clock = _Hung(), _Clock()

    @contextmanager
    def open_stream() -> Iterator[_Hung]:
        yield hung

    def slept() -> None:
        time.sleep(0.2)
        clock.now += 3600.0  # an hour with the lid closed

    threading.Thread(target=slept, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(Silent) as ended:
        read_stream(
            open_stream,
            pieces=lambda _event: (),
            final=lambda _stream: None,
            silences=Silences(first_token_s=300.0, token_gap_s=60.0),
            cancel=None,
            clock=clock,
        )
    assert ended.value.kind == "first_token"
    assert time.monotonic() - started < 3.0, "waited out a limit after the sleep"
    assert hung.closed.is_set(), "the dead request's stream was not closed"


def test_a_sleep_is_seen_by_the_clock_that_counts_it_running_ahead() -> None:
    waited, awake = _Clock(), _Clock()
    watch = SleepWatch(waited, awake)
    waited.now += 1.0
    awake.now += 1.0
    assert not watch.slept(), "time awake is not a sleep"
    waited.now += SLEPT_S + 1.0
    assert watch.slept()
    assert not watch.slept(), "a sleep is seen once"


def test_connections_are_made_anew_after_a_sleep_and_the_stale_ones_closed() -> None:
    waited, awake = _Clock(), _Clock()
    made: list[httpx2.Client] = []

    def make() -> httpx2.Client:
        made.append(httpx2.Client())
        return made[-1]

    pool = ModelConnections(make, SleepWatch(waited, awake))
    first = pool.get()
    assert pool.get() is first, "no sleep, the same connections"
    waited.now += 600.0
    fresh = pool.get()
    assert fresh is not first and first.is_closed and not fresh.is_closed
    pool.close()
    assert fresh.is_closed
