"""The record stream: a seek on `seq`, never a replay buffer.

A client says where it got to (`?from=<seq>`) and the scope is read from there through the
facade (`journal(scope, cursor)`), `RECORD_PAGE` at a time, in order. At the end the stream
either closes (`/journal`, the poll-friendly half) or stays attached and reads again every
`RECORD_POLL_S` (`/events`). The **only** state a stream holds is its cursor, two integers in
this generator's frame: a slow, crashed or hostile client costs the server nothing it has to
size, and dropping the connection frees it. Reconnecting is the same call with the highest
seq the client saw.

`/events` interleaves the session's progress and a keep-alive into the same stream. The
loop is one coroutine: it reads records, then waits up to `RECORD_POLL_S` for progress, so a
progress frame goes out at once and records are at most one poll behind their commit.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Final

import anyio.to_thread

from chickadee.scopes import Cursor, Page, ScopeRecord
from chickadee_serve.frames import KEEPALIVE_S, ProgressFrame, RecordFrame
from chickadee_serve.progress import Subscription

__all__ = ["RECORD_PAGE", "RECORD_POLL_S", "ReadPage", "record_frame", "stream"]

RECORD_PAGE: Final = 256
"""Records one read returns: a read size, not a bound a client can observe."""

RECORD_POLL_S: Final = 0.05
"""How long an attached stream waits before reading its scope again. Records are at most
this far behind their commit; progress is not delayed by it."""

ReadPage = Callable[[Cursor], Page]
"""The facade's `journal` bound to one scope: the next page past a cursor."""


def record_frame(record: ScopeRecord) -> RecordFrame:
    """A scope's record as the wire carries it: the stored payload, unchanged."""
    return RecordFrame(
        scope=record.scope,
        seq=record.seq,
        at=record.at,
        kind=record.kind,
        run_id=record.run_id,
        payload=dict(record.payload),
        task_id=record.task_id,
        round=record.round,
    )


async def stream(
    read: ReadPage,
    first: Page,
    *,
    scope: str,
    follow: bool,
    progress: Subscription | None = None,
    gone: Callable[[], Awaitable[bool]] | None = None,
    stopping: Callable[[], bool] | None = None,
    poll_s: float = RECORD_POLL_S,
    keepalive_s: float = KEEPALIVE_S,
) -> AsyncIterator[RecordFrame | ProgressFrame]:
    """Every record past the client's cursor, in order and once, starting from `first` (the
    page the route read before answering, so a missing scope is a 404 and not a broken
    stream). With `follow`, then live: new records as they commit, `progress`'s frames as they
    arrive, and a keep-alive when nothing else was sent for `keepalive_s`. `gone` says the
    client has disconnected; it is asked at every idle turn, so an attached stream ends when
    its client does rather than when a write to it next fails. `stopping` says the server is
    shutting down, asked at the same turn: the stream ends, and its client reconnects to
    whatever serves next, from its last seq."""
    page = first
    sent = time.monotonic()
    while True:
        for record in page.records:
            yield record_frame(record)
        if page.records:
            sent = time.monotonic()
            page = await anyio.to_thread.run_sync(read, page.cursor)
            continue
        if not follow or (stopping is not None and stopping()):
            return
        if gone is not None and await gone():
            return
        if progress is not None:
            async for frame in _drain(progress, poll_s):
                sent = time.monotonic()
                yield frame
        else:
            await asyncio.sleep(poll_s)
        if time.monotonic() - sent >= keepalive_s:
            sent = time.monotonic()
            yield ProgressFrame(scope=scope, kind="alive")
        page = await anyio.to_thread.run_sync(read, page.cursor)


async def _drain(sub: Subscription, wait_s: float) -> AsyncIterator[ProgressFrame]:
    """What is queued now, else whatever arrives within `wait_s`."""
    with contextlib.suppress(TimeoutError):
        yield await asyncio.wait_for(sub.queue.get(), timeout=wait_s)
    while not sub.queue.empty():
        yield sub.queue.get_nowait()
