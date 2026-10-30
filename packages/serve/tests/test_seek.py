"""Reconnection is a seek. From 0, from the middle, and across a gap in attendance, every
record arrives exactly once and in order, and the server keeps nothing per client."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from demo_session import QUESTION_ID, SESSION_ID
from serve_helpers import Harness, recorded_lines, replay


def _until(h: Harness, seq: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while h.facade.horizon < seq:
        assert time.monotonic() < deadline, f"the replay never reached seq {seq}"
        time.sleep(0.01)


def _answer(h: Harness) -> None:
    r = h.client.post(
        f"/session/{SESSION_ID}/answer/{QUESTION_ID}",
        json={"choice": "yes"},
        headers=h.headers(),
    )
    assert r.status_code == 200, r.text


def test_from_zero_middle_and_after_a_gap(tmp_path: Path) -> None:
    total = len(recorded_lines())
    with replay(tmp_path) as h:
        _until(h, 4)  # the replay holds at the question
        first = h.journal(0)
        assert [r.seq for r in first] == [1, 2, 3, 4]
        _answer(h)
        _until(h, total)
        # The client was away while 5..total were written; it seeks from what it held.
        rest = h.journal(first[-1].seq)
        assert [r.seq for r in rest] == list(range(5, total + 1))
        # From the middle, and again from the end: nothing twice, nothing missing.
        assert [r.seq for r in h.journal(7)] == list(range(8, total + 1))
        assert h.journal(total) == []


def test_records_are_the_recording(tmp_path: Path) -> None:
    """What arrives is the log as stored: the kinds, seqs and stamps of every line."""
    with replay(tmp_path) as h:
        _until(h, 4)
        _answer(h)
        _until(h, len(recorded_lines()))
        got = [(r.seq, r.kind, r.at, r.run_id) for r in h.journal(0)]
    want = [(d["seq"], d["kind"], d["at"], d["run_id"]) for d in recorded_lines()]
    assert got == want


def test_a_cursor_past_the_end_is_empty_not_an_error(tmp_path: Path) -> None:
    with replay(tmp_path) as h:
        assert h.journal(10_000) == []


@pytest.mark.parametrize("bad", ["-1", "1.5", "x", "\u00b2", "\u0661"])
def test_a_bad_cursor_is_a_typed_422(tmp_path: Path, bad: str) -> None:
    """Non-ASCII digits too: `str.isdigit` admits a superscript two, which `int` refuses
    (a 500), and an Arabic-Indic one, which a cursor that names a seq has no use for."""
    with replay(tmp_path) as h:
        r = h.client.get(
            f"/session/{SESSION_ID}/journal", params={"from": bad}, headers=h.headers()
        )
        assert r.status_code == 422
        assert r.json()["code"] == "BAD_REQUEST"
