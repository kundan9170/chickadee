"""Polling alone is correct, from the client's side: a session read with
`poll=True`, never attaching to the live channel, yields the same records in the same order as
the live stream does, and no progress."""

from __future__ import annotations

import pytest
from engines import answer, call, calls
from hosting import SID, hosted, until

from chickadee_serve.frames import ProgressFrame, RecordFrame
from chickadee_wire.events import Events


def _records(events: Events, through: int) -> tuple[list[RecordFrame], int]:
    records: list[RecordFrame] = []
    progress = 0
    while not records or records[-1].seq < through:
        frame = next(events)
        if isinstance(frame, ProgressFrame):
            progress += 1
        else:
            records.append(frame)
    events.close()
    return records, progress


def test_polling_yields_what_the_live_stream_yields(monkeypatch: pytest.MonkeyPatch) -> None:
    script = [calls(call("read_file", "c1", path="README.md")), answer("read it")]
    with hosted(monkeypatch, script) as host, host.client() as client:
        client.turn(SID, "look")
        last = until(client, "turn", count=2).seq
        live, _ = _records(client.session_events(SID), last)
        polling = client.session_events(SID, poll=True, interval_s=0.01)
        polled, progress = _records(polling, last)
    assert [r.seq for r in polled] == list(range(1, last + 1))
    assert polled == live
    assert progress == 0, "the journal carries records only"
