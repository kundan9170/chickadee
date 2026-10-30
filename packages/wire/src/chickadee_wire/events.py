"""A scope's frames as an iterator that survives its connection.

**Live** reads `/events?from=<seq>`: records, then progress and keep-alives as work happens.
**Polling** (`poll=True`) reads `/journal?from=<seq>` to its end, waits, and reads again: the
same records in the same order, and no progress, which a correct client may drop anyway.

**Every record once, in order**. The iterator keeps `last_seq`, the highest record seq
it has yielded. On a dropped connection, or a live stream the server closed, it reconnects
from that seq and drops any record at or below it, so a gap in attendance costs nothing and a
record is never yielded twice. The server keeps no cursor: a client that stops and comes back
later starts a new iterator from its `last_seq`.

**Reconnecting is bounded.** `Reconnect` says how many tries in a row and how long between
them; a spent budget raises `Disconnected` rather than ending the iteration as if the scope
were done. A route's own refusal (`WireError`: no such session, no token) is raised at once.

**Reconnecting is visible**. A `watch` callback is told `Retrying` before
each wait, and `Reconnected` when a frame arrives after one: a frontend draws "reconnecting"
from the stream itself, since no server can say it is unreachable. Called on the reading
thread.

`Events` and `AsyncEvents` are the same loop over `Position` and `Reconnect`, once per kind of
I/O.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterator
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from dataclasses import dataclass

import httpx

from chickadee_serve.frames import ProgressFrame, RecordFrame, decode_frame
from chickadee_wire.calls import Stream
from chickadee_wire.errors import Disconnected

__all__ = [
    "AsyncEvents",
    "Events",
    "Frame",
    "Link",
    "Position",
    "Reconnect",
    "Reconnected",
    "Retrying",
    "Watch",
]

type Frame = RecordFrame | ProgressFrame

type Lines = Callable[[Stream, int], AbstractContextManager[Iterator[str]]]
"""Opens `stream` from a seq: the response's lines while it lasts (`Client._lines`)."""

type AsyncLines = Callable[[Stream, int], AbstractAsyncContextManager[AsyncIterator[str]]]


@dataclass
class Position:
    """Where a reader is in one scope: the highest record seq it has been given."""

    last_seq: int = 0

    def take(self, line: str) -> Frame | None:
        """The frame on `line`, or None for a blank line or a record already given."""
        if not line.strip():
            return None
        frame = decode_frame(line)
        if isinstance(frame, RecordFrame):
            if frame.seq <= self.last_seq:
                return None
            self.last_seq = frame.seq
        return frame


@dataclass(frozen=True)
class Reconnect:
    """How hard a stream tries to come back: `attempts` tries in a row, waiting `backoff_s`
    doubled each time up to `max_wait_s`. A policy only, so one may be shared: each stream
    counts its own failures (`_Tries`)."""

    attempts: int = 5
    backoff_s: float = 0.2
    max_wait_s: float = 5.0


@dataclass(frozen=True)
class Retrying:
    """The stream dropped, or the server closed it: try `failures` of the budget, after
    `wait_s`, because of `why`."""

    failures: int
    wait_s: float
    why: str


@dataclass(frozen=True)
class Reconnected:
    """The stream is reading again after a `Retrying`."""


type Link = Retrying | Reconnected
"""What a stream's `watch` is told about its connection."""

type Watch = Callable[[Link], None]


@dataclass
class _Tries:
    """One stream's failures in a row under its `Reconnect`. A frame received resets it, and
    `watch` hears each change."""

    policy: Reconnect
    watch: Watch | None = None
    failures: int = 0

    def ok(self) -> None:
        if self.failures and self.watch is not None:
            self.watch(Reconnected())
        self.failures = 0

    def failed(self, why: str) -> float:
        """The wait before the next try; `Disconnected` when the budget is spent."""
        self.failures += 1
        if self.failures > self.policy.attempts:
            raise Disconnected(f"gave up after {self.policy.attempts} reconnects: {why}")
        wait = min(self.policy.backoff_s * 2 ** (self.failures - 1), self.policy.max_wait_s)
        if self.watch is not None:
            self.watch(Retrying(self.failures, wait, why))
        return wait


_DROPPED = (httpx.TransportError, httpx.StreamError)
"""What a dropped connection raises, as opposed to a route's refusal."""


class Events(Iterator[Frame]):
    """`stream`'s frames from `after`, reconnecting as `reconnect` allows. Polling waits
    `interval_s` between reads."""

    def __init__(
        self,
        lines: Lines,
        stream: Stream,
        *,
        after: int = 0,
        poll: bool = False,
        interval_s: float = 0.5,
        reconnect: Reconnect | None = None,
        watch: Watch | None = None,
    ) -> None:
        self._lines = lines
        self._stream = stream
        self._poll = poll
        self._interval_s = interval_s
        self._tries = _Tries(reconnect if reconnect is not None else Reconnect(), watch)
        self.position = Position(after)
        self._frames = self._run()

    @property
    def last_seq(self) -> int:
        return self.position.last_seq

    def __next__(self) -> Frame:
        return next(self._frames)

    def close(self) -> None:
        self._frames.close()

    def _run(self) -> Generator[Frame]:
        while True:
            try:
                with self._lines(self._stream, self.position.last_seq) as lines:
                    for line in lines:
                        frame = self.position.take(line)
                        if frame is not None:
                            self._tries.ok()
                            yield frame
            except _DROPPED as exc:
                time.sleep(self._tries.failed(str(exc)))
                continue
            if self._poll:
                self._tries.ok()
                time.sleep(self._interval_s)
            else:
                time.sleep(self._tries.failed("the server closed the stream"))


class AsyncEvents(AsyncIterator[Frame]):
    """`Events`, for an `AsyncClient`."""

    def __init__(
        self,
        lines: AsyncLines,
        stream: Stream,
        *,
        after: int = 0,
        poll: bool = False,
        interval_s: float = 0.5,
        reconnect: Reconnect | None = None,
        watch: Watch | None = None,
    ) -> None:
        self._lines = lines
        self._stream = stream
        self._poll = poll
        self._interval_s = interval_s
        self._tries = _Tries(reconnect if reconnect is not None else Reconnect(), watch)
        self.position = Position(after)
        self._frames = self._run()

    @property
    def last_seq(self) -> int:
        return self.position.last_seq

    async def __anext__(self) -> Frame:
        return await anext(self._frames)

    async def aclose(self) -> None:
        await self._frames.aclose()

    async def _run(self) -> AsyncGenerator[Frame]:
        while True:
            try:
                async with self._lines(self._stream, self.position.last_seq) as lines:
                    async for line in lines:
                        frame = self.position.take(line)
                        if frame is not None:
                            self._tries.ok()
                            yield frame
            except _DROPPED as exc:
                await asyncio.sleep(self._tries.failed(str(exc)))
                continue
            if self._poll:
                self._tries.ok()
                await asyncio.sleep(self._interval_s)
            else:
                await asyncio.sleep(self._tries.failed("the server closed the stream"))
