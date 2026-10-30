"""The progress watchdog replayed over a recorded run: where it would have nudged and
handed a round up, from the journal alone, whether or not it was bound when the run ran.

What these pin: a round of refused calls recorded with no hook bound replays to a nudge, a
report and a backstop at the calls the live ladder gives them, with the calls made after the
hand-up counted; another threshold is tried with `--set`; the command prints each round and
what they add up to, and says so when nothing was recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from box_helpers import delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.box.progress_replay import main, replay_run
from chickadee.hooks.watchdog import WatchdogParams

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


def _nope(n: int) -> list[Any]:
    return [calls(call("nope", f"n{i}", why=str(i))) for i in range(1, n + 1)]


def _recorded(root: Path) -> Any:
    """A round of 14 refused calls, recorded with no hook bound, ended by the worker's reply."""
    worker, journal, _ = worker_rig(root, [*_nope(14), answer("gave up")])
    worker.start("T1", "Do it.", delegate(journal, root))
    journal.close()
    return journal


def test_a_run_recorded_without_the_watchdog_replays_to_its_ladder(root: Path) -> None:
    journal = _recorded(root)
    run = journal.projection()
    [row] = replay_run(
        run,
        journal.run_id,
        journal.get_blob,
        root,
        WatchdogParams(looping=4),  # the numbers below are D231's looping
        call_cap=40,
    )

    assert (row.task, row.round, row.calls, row.ended) == ("T1", 1, 14, "replied")
    assert row.rungs == (
        ("nudge", "looping", 4),
        ("report", "looping", 8),
        ("backstop", "looping", 11),
    )
    assert row.after_hand_up == 6
    assert row.calls_left == 40


def test_another_threshold_is_tried_without_running_anything(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _recorded(root)
    assert main([str(root), "--set", "looping=6", "--cap", "200"]) == 0
    out = capsys.readouterr().out
    assert "T1 round 1: 14 calls, ended replied" in out
    assert "nudge (looping) after call 6; report (looping) after call 12" in out
    assert "1 nudged, 1 handed up, 2 call(s) made after a hand-up point" in out


def test_a_repository_with_no_runs_says_so(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(root)]) == 1
    assert "no worker rounds recorded" in capsys.readouterr().out
