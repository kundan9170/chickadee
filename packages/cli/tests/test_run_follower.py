"""Following a run the session opened (`runs.py`): its records from the
`episode` record that names it to its `run_end`, over a hosted engine; once per run. A run that
has not started is waited for while its episode is live and never called lost; only a
stream that started and then broke is. What settles a run is its end being said: the
coder call's result with its outcome, or a trigger-4 notice naming it; a background hand-over's
`RUNNING`, which has no outcome, is not, so a planned run whose journal appears only after it
is still followed to its end."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from hosting import SID, hosted
from recordings import install_run

from chickadee_cli.runs import RunFollower
from chickadee_serve.frames import JsonValue, RecordFrame
from chickadee_wire.events import Frame

RUN_ID = "2026-10-10-0123abcd"


def _episode(run_id: str, seq: int = 9) -> RecordFrame:
    # As the engine writes it since §12.3: no tier.
    body: dict[str, JsonValue] = {"run_id": run_id, "path": "exact"}
    return RecordFrame(scope=SID, seq=seq, at="t", kind="episode", payload=body)


def _result(run_id: str, seq: int = 11) -> RecordFrame:
    """The coder call's result for `run_id`, as the session records it once the run ended or
    never started: its outcome typed. The episode settled."""
    body: dict[str, JsonValue] = {
        "call_id": "z-1",
        "ran": True,
        "text": f"BLOCKED · run {run_id} · $0.0000",
        "outcome": {"terminal": "BLOCKED", "why": ["the stop was not approved"]},
    }
    return RecordFrame(scope=SID, seq=seq, at="t", kind="tool", payload=body, run_id=run_id)


def _handed_over(run_id: str, seq: int = 11) -> RecordFrame:
    """The coder call answered at once, the run handed over in the background: its
    `tool` line names the run and has no outcome, since the run has not ended."""
    body: dict[str, JsonValue] = {
        "call_id": "z-1",
        "ran": True,
        "text": f"RUNNING · run {run_id} · mode NEW: the team works on it on its own.",
    }
    return RecordFrame(scope=SID, seq=seq, at="t", kind="tool", payload=body, run_id=run_id)


def _run_ended(run_id: str, seq: int = 14) -> RecordFrame:
    """A background run's end on the session's log: a trigger-4 notice naming it."""
    body: dict[str, JsonValue] = {
        "role": "harness",
        "text": f"BLOCKED · run {run_id} · $0.0000",
        "trigger": 4,
        "handle": run_id,
    }
    return RecordFrame(scope=SID, seq=seq, at="t", kind="turn", payload=body)


def _recorded(state_dir: Path) -> None:
    """The recorded task run (`recordings.RUN_FIXTURE`) in `state_dir`'s runs."""
    install_run(state_dir / "runs", RUN_ID)


def test_a_run_is_followed_from_its_episode_to_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Frame] = []
    lock = threading.Lock()

    def emit(frame: Frame) -> None:
        with lock:
            seen.append(frame)

    with hosted(monkeypatch) as host, host.client() as client:
        _recorded(host.state_dir)
        follower = RunFollower(client, emit)
        follower.feed(_episode(RUN_ID))
        follower.feed(_episode(RUN_ID, seq=10))  # the same run again: not followed twice
        assert follower.wait(20), "the run reached its run_end"
        follower.close()
    records = [f for f in seen if isinstance(f, RecordFrame)]
    assert records[0].kind == "run_start" and records[-1].kind == "run_end"
    assert [f.seq for f in records] == list(range(1, len(records) + 1)), "each record once"
    assert all(f.scope == RUN_ID for f in records)


def test_a_run_that_never_starts_is_not_lost_and_its_follower_ends_with_its_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#542: a run blocked before it started writes no journal. The route says not found, the
    follower waits quietly while the episode is live, and ends with no "lost" line once the
    coder call's result for it is recorded."""
    never = "2026-10-13-be6bf9eb"
    lost: list[tuple[str, str]] = []
    with hosted(monkeypatch) as host, host.client() as client:
        follower = RunFollower(
            client, lambda _f: None, lambda run, why: lost.append((run, why))
        )
        follower.feed(_episode(never))
        assert not follower.wait(0.6), "still waiting while the episode is live"
        follower.feed(_result(never))
        assert follower.wait(10)
    assert lost == []


def test_a_newer_episode_settles_an_unstarted_one(monkeypatch: pytest.MonkeyPatch) -> None:
    lost: list[tuple[str, str]] = []
    with hosted(monkeypatch) as host, host.client() as client:
        follower = RunFollower(
            client, lambda _f: None, lambda run, why: lost.append((run, why))
        )
        follower.feed(_episode("2026-10-13-00000001"))
        follower.feed(_episode("2026-10-13-00000002", seq=10))
        follower.feed(_result("2026-10-13-00000002"))
        assert follower.wait(10)
    assert lost == []


def test_a_run_that_appears_late_is_still_followed_to_its_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's journal can appear well after its episode, while its workspace is made: no
    deadline ends the wait while its episode is live."""
    seen: list[Frame] = []
    lost: list[tuple[str, str]] = []
    with hosted(monkeypatch) as host, host.client() as client:
        follower = RunFollower(client, seen.append, lambda run, why: lost.append((run, why)))
        follower.feed(_episode(RUN_ID))
        time.sleep(1.0)
        _recorded(host.state_dir)
        assert follower.wait(20)
        follower.close()
    assert lost == []
    records = [f for f in seen if isinstance(f, RecordFrame)]
    assert records and records[-1].kind == "run_end"


class _Breaks:
    """A client whose run stream starts and then breaks."""

    def run_events(self, run_id: str, after: int = 0, **_kw: Any) -> Iterator[Frame]:
        yield RecordFrame(scope=run_id, seq=1, at="t", kind="run_start", payload={})
        raise ConnectionError("the connection dropped")


def test_a_stream_that_started_and_broke_is_lost() -> None:
    lost: list[tuple[str, str]] = []
    follower = RunFollower(
        cast("Any", _Breaks()), lambda _f: None, lambda run, why: lost.append((run, why))
    )
    follower.feed(_episode(RUN_ID))
    assert follower.wait(10)
    assert lost == [(RUN_ID, "the connection dropped")]


def test_other_session_records_start_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch) as host, host.client() as client:
        follower = RunFollower(client, lambda _f: None)
        follower.feed(RecordFrame(scope=SID, seq=1, at="t", kind="turn", payload={}))
        assert follower.wait(0.1)


class _GatedRuns:
    """`run_events` for two runs: the older one hands out a record, then waits on `gate`."""

    def __init__(self, older: str) -> None:
        self.older = older
        self.gate = threading.Event()
        self.first = threading.Event()

    def run_events(self, run_id: str, _after: int) -> Iterator[Frame]:
        for seq in range(1, 4):
            if run_id == self.older and seq == 2:
                self.first.set()
                self.gate.wait(5)
            kind = "run_end" if seq == 3 else "note"
            yield RecordFrame(scope=run_id, seq=seq, at="t", kind=kind, payload={})


def test_latest_only_stops_an_earlier_run_once_a_newer_one_is_followed() -> None:
    """The TUI draws the latest episode (`latest_only`): a session read from its start does not
    replay every past run's journal; the REPL, which prints each run, follows every one."""
    older, newer = "2026-10-13-00000001", "2026-10-13-00000002"
    runs = _GatedRuns(older)
    seen: list[tuple[str, int]] = []
    lock = threading.Lock()

    def emit(frame: Frame) -> None:
        assert isinstance(frame, RecordFrame)
        with lock:
            seen.append((frame.scope, frame.seq))

    follower = RunFollower(cast(Any, runs), emit, latest_only=True)
    follower.feed(_episode(older))
    assert runs.first.wait(5)
    follower.feed(_episode(newer, seq=10))
    runs.gate.set()
    assert follower.wait(5)
    follower.close()
    assert [s for r, s in seen if r == newer] == [1, 2, 3]
    assert [s for r, s in seen if r == older] == [1], "stopped once the newer run was followed"


def test_a_run_handed_over_in_the_background_is_followed_once_its_journal_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug T-5.39 fixes: the hand-over's `RUNNING` was taken for the run's end, so a
    planned run, whose journal waits for Athena's plan, was given up before it wrote a line,
    and the screen's Athena and worker panes stayed empty."""
    seen: list[Frame] = []
    lost: list[tuple[str, str]] = []
    with hosted(monkeypatch) as host, host.client() as client:
        follower = RunFollower(client, seen.append, lambda run, why: lost.append((run, why)))
        follower.feed(_episode(RUN_ID))
        follower.feed(_handed_over(RUN_ID))
        assert not follower.wait(1.5), "still waiting: a hand-over is no end"
        _recorded(host.state_dir)
        assert follower.wait(20)
        follower.close()
    assert lost == []
    records = [f for f in seen if isinstance(f, RecordFrame)]
    assert records and records[0].kind == "run_start" and records[-1].kind == "run_end"


def test_a_background_run_s_end_notice_settles_one_that_never_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never = "2026-10-13-be6bf9eb"
    lost: list[tuple[str, str]] = []
    with hosted(monkeypatch) as host, host.client() as client:
        follower = RunFollower(
            client, lambda _f: None, lambda run, why: lost.append((run, why))
        )
        follower.feed(_episode(never))
        follower.feed(_handed_over(never))
        assert not follower.wait(0.6)
        follower.feed(_run_ended("2026-10-13-00000009"))  # another run's end: not this one's
        assert not follower.wait(0.6)
        follower.feed(_run_ended(never))
        assert follower.wait(10)
    assert lost == []


class _Paused:
    """`run_events` for two runs: the older one's first read hands out a record and waits on
    `gate`, a run paused under it; any later read of it runs to its end."""

    def __init__(self, older: str) -> None:
        self.older = older
        self.gate = threading.Event()
        self.first = threading.Event()
        self.asked: list[tuple[str, int]] = []

    def run_events(self, run_id: str, after: int = 0, **_kw: Any) -> Iterator[Frame]:
        again = (run_id, after) in self.asked
        self.asked.append((run_id, after))
        for seq in range(after + 1, 4):
            if run_id == self.older and seq == 2 and not again:
                self.first.set()
                self.gate.wait(5)
            kind = "run_end" if seq == 3 else "note"
            yield RecordFrame(scope=run_id, seq=seq, at="t", kind=kind, payload={})


def test_a_run_a_newer_one_stopped_is_read_again_when_asked() -> None:
    """Zeus's `coder` resumed a paused run, which goes on under its first id while
    the call's `episode` names another; the screen asks for the resumed one again."""
    older, newer = "2026-10-13-00000001", "2026-10-13-00000002"
    runs = _Paused(older)
    seen: list[tuple[str, int]] = []
    lock = threading.Lock()

    def emit(frame: Frame) -> None:
        assert isinstance(frame, RecordFrame)
        with lock:
            seen.append((frame.scope, frame.seq))

    follower = RunFollower(cast(Any, runs), emit, latest_only=True)
    follower.feed(_episode(older))
    assert runs.first.wait(5)
    follower.feed(_episode(newer, seq=10))  # stops the older one's stream
    runs.gate.set()
    follower.follow(older)
    assert follower.wait(5)
    assert runs.asked.count((older, 0)) == 2, "read again, from its first record"
    assert [s for r, s in seen if r == older] == [1, 1, 2, 3]
    follower.follow(older)
    assert follower.wait(5)
    assert runs.asked.count((older, 0)) == 2, "read to its end: never read again"
    follower.close()
