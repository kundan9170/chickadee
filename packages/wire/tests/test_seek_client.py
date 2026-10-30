"""Every record exactly once across a disconnect, from the client's side:
a dropped connection re-seeks from the last record seq yielded, a record sent again is not
yielded again, and a spent reconnect budget raises instead of ending as if the scope were
done."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
from engines import answer
from hosting import SID, hosted, until

from chickadee_serve.frames import ProgressFrame, RecordFrame, encode_frame
from chickadee_wire import calls
from chickadee_wire.errors import Disconnected
from chickadee_wire.events import AsyncEvents, Events, Link, Reconnect, Reconnected, Retrying

STREAM = calls.session_events(SID)
FAST = Reconnect(attempts=3, backoff_s=0.0)


def _record(seq: int) -> str:
    frame = RecordFrame(scope=SID, seq=seq, at="t", kind="turn", payload={"n": seq})
    return encode_frame(frame).decode()


ALIVE = encode_frame(ProgressFrame(scope=SID, kind="alive")).decode()

DROP = object()
"""A connection that ends in a transport error rather than closing."""


class Connections:
    """Scripted connections, one per open: each its lines, then a drop or a close. Records
    which seq each open asked from."""

    def __init__(self, *scripts: list[object]) -> None:
        self._scripts = list(scripts)
        self.asked_from: list[int] = []

    @contextmanager
    def open(self, stream: calls.Stream, after: int) -> Iterator[Iterator[str]]:
        assert stream == STREAM
        self.asked_from.append(after)
        script = self._scripts.pop(0) if self._scripts else [DROP]

        def lines() -> Iterator[str]:
            for item in script:
                if item is DROP:
                    raise httpx.ReadError("connection reset")
                yield str(item)

        yield lines()

    @asynccontextmanager
    async def open_async(
        self, stream: calls.Stream, after: int
    ) -> AsyncIterator[AsyncIterator[str]]:
        with self.open(stream, after) as lines:

            async def each() -> AsyncIterator[str]:
                for line in lines:
                    yield line

            yield each()


def _seqs(events: Events, n: int) -> list[int]:
    out: list[int] = []
    while len(out) < n:
        frame = next(events)
        if isinstance(frame, RecordFrame):
            out.append(frame.seq)
    return out


def test_a_dropped_stream_re_seeks_and_yields_every_record_once() -> None:
    conns = Connections(
        [_record(1), _record(2), _record(3), DROP],
        [_record(2), _record(3), _record(4), ALIVE],  # the server closed it: reconnect
        [_record(5)],
    )
    events = Events(conns.open, STREAM, reconnect=FAST)
    assert _seqs(events, 5) == [1, 2, 3, 4, 5]
    assert conns.asked_from == [0, 3, 4], "each reconnect asks from the last seq yielded"
    assert events.last_seq == 5


def test_the_async_stream_re_seeks_as_the_sync_one_does() -> None:
    conns = Connections(
        [_record(1), _record(2), DROP],
        [_record(2), _record(3), ALIVE],
        [_record(4)],
    )

    async def read() -> list[int]:
        events = AsyncEvents(conns.open_async, STREAM, reconnect=FAST)
        out: list[int] = []
        async for frame in events:
            if isinstance(frame, RecordFrame):
                out.append(frame.seq)
            if len(out) == 4:
                break
        await events.aclose()
        return out

    assert asyncio.run(read()) == [1, 2, 3, 4]
    assert conns.asked_from == [0, 2, 3]


def test_a_reader_resumes_from_where_it_stopped() -> None:
    conns = Connections([_record(4), _record(5)])
    events = Events(conns.open, STREAM, after=3, reconnect=FAST)
    assert _seqs(events, 2) == [4, 5]
    assert conns.asked_from == [3]


def test_a_spent_reconnect_budget_raises() -> None:
    conns = Connections([_record(1), DROP])
    events = Events(conns.open, STREAM, reconnect=FAST)
    assert _seqs(events, 1) == [1]
    with pytest.raises(Disconnected, match="gave up after 3 reconnects"):
        next(events)
    assert conns.asked_from == [0, 1, 1, 1]


def test_a_live_session_read_in_two_attendances_is_read_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch, [answer("one"), answer("two")]) as host, host.client() as client:
        client.turn(SID, "first")
        until(client, "turn", count=2)
        first = client.session_events(SID)
        seen = _seqs(first, len(client.session_journal(SID)))
        first.close()
        client.turn(SID, "second")
        until(client, "turn", count=4)
        later = client.session_events(SID, after=seen[-1])
        total = len(client.session_journal(SID))
        seen += _seqs(later, total - len(seen))
        later.close()
    assert seen == list(range(1, total + 1))


def test_a_shared_reconnect_policy_counts_each_streams_failures_apart() -> None:
    """One `Reconnect` handed to two streams: each has its own budget."""
    policy = Reconnect(attempts=1, backoff_s=0.0)
    first = Events(Connections([_record(1), DROP], [DROP]).open, STREAM, reconnect=policy)
    second = Events(Connections([_record(1)], [_record(2)]).open, STREAM, reconnect=policy)
    assert _seqs(first, 1) == [1]
    with pytest.raises(Disconnected):
        next(first)  # the first stream spends its one reconnect, and a second fails
    assert _seqs(second, 2) == [1, 2], "the second stream's budget is its own"


def test_a_watcher_hears_each_retry_and_the_stream_coming_back() -> None:
    """What a frontend draws as "reconnecting", from the stream itself."""
    heard: list[Link] = []
    conns = Connections([_record(1), DROP], [_record(2)])
    events = Events(conns.open, STREAM, reconnect=FAST, watch=heard.append)
    assert _seqs(events, 2) == [1, 2]
    assert heard == [Retrying(1, 0.0, "connection reset"), Reconnected()]


def test_a_watcher_hears_nothing_while_the_stream_stays_up() -> None:
    heard: list[Link] = []
    events = Events(Connections([_record(1), _record(2)]).open, STREAM, watch=heard.append)
    assert _seqs(events, 2) == [1, 2]
    assert heard == []
