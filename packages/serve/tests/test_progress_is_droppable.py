"""The admission test as code: drop progress frames and nothing a client holds changes.

The records a client ends with, and so everything it renders from them, are the same whether
it received every progress frame or almost none.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from demo_session import QUESTION_ID, SESSION_ID
from serve_helpers import live, recorded_lines, replay

from chickadee_serve.frames import ProgressFrame, RecordFrame, decode_frame
from chickadee_serve.progress import ProgressHub


def _attend(tmp: Path, depth: int | None) -> tuple[list[tuple[int, str]], int]:
    """Watch `/events` until every record has arrived; the records, and progress frames seen."""
    records: list[tuple[int, str]] = []
    progress = 0
    total = len(recorded_lines())
    events = f"/session/{SESSION_ID}/events"
    with (
        replay(tmp, depth=depth, start=False) as h,
        live(h.served) as client,
        client.stream("GET", events, headers=h.headers()) as r,
    ):
        assert r.status_code == 200
        h.facade.start()
        for line in r.iter_lines():
            if not line:
                continue
            frame = decode_frame(line)
            if isinstance(frame, ProgressFrame):
                progress += 1
                continue
            assert isinstance(frame, RecordFrame)
            records.append((frame.seq, frame.kind))
            if frame.kind == "question_asked":
                client.post(
                    f"/session/{SESSION_ID}/answer/{QUESTION_ID}",
                    json={"choice": "yes"},
                    headers=h.headers(),
                )
            if len(records) == total:
                break
    return records, progress


def test_records_are_the_same_with_and_without_progress(tmp_path: Path) -> None:
    full, seen = _attend(tmp_path / "full", depth=None)
    starved, _ = _attend(tmp_path / "starved", depth=1)
    assert seen > 0, "the full run carried progress, so the comparison means something"
    assert full == starved == [(d["seq"], d["kind"]) for d in recorded_lines()]


def test_a_full_queue_drops_and_never_waits() -> None:
    async def scenario() -> tuple[int, int]:
        hub = ProgressHub(depth=2)
        hub.bind(asyncio.get_running_loop())
        async with hub.subscribe("s") as sub:
            for _ in range(10):
                hub.publish(ProgressFrame(scope="s", kind="thinking"))
            await asyncio.sleep(0.05)  # let the loop run the offers
            return sub.queue.qsize(), sub.dropped

    held, dropped = asyncio.run(scenario())
    assert (held, dropped) == (2, 8)


def test_progress_with_nobody_listening_is_dropped() -> None:
    hub = ProgressHub()
    hub.publish(ProgressFrame(scope="s", kind="thinking"))  # no loop, no subscriber: no error
    assert hub.subscribers("s") == 0


def test_a_live_piece_says_when_its_role_calls_by_python() -> None:
    """The mode rides each piece, so a viewer knows it from the first."""
    from chickadee.client.live import LiveEvent, TextUpdate
    from chickadee.zeus.loop import live_event
    from chickadee_serve.progress import describe

    python = LiveEvent("worker", "T1", "w1", TextUpdate(0, "x"), "python")
    native = LiveEvent("worker", "T1", "w1", TextUpdate(0, "x"))
    assert describe("S", live_event(python)).payload["calls"] == "python"
    assert "calls" not in describe("S", live_event(native)).payload
