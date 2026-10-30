"""The real test: a client that never opens the live channel, and only polls `/journal`, ends
holding exactly what the session wrote. It is unresponsive, never wrong."""

from __future__ import annotations

import time
from pathlib import Path

from demo_session import QUESTION_ID, SESSION_ID
from serve_helpers import recorded_lines, replay


def test_polling_alone_reaches_the_whole_session(tmp_path: Path) -> None:
    held: list[int] = []
    answered = False
    deadline = time.monotonic() + 10
    with replay(tmp_path) as h:
        while len(held) < len(recorded_lines()):
            assert time.monotonic() < deadline, f"polling stalled at {held}"
            for record in h.journal(held[-1] if held else 0):
                held.append(record.seq)
                # The pending question is a record, so a poller answers it like anyone.
                if record.kind == "question_asked" and not answered:
                    r = h.client.post(
                        f"/session/{SESSION_ID}/answer/{QUESTION_ID}",
                        json={"choice": "no"},
                        headers=h.headers(),
                    )
                    assert r.status_code == 200
                    answered = True
            time.sleep(0.02)
    assert held == [d["seq"] for d in recorded_lines()]
