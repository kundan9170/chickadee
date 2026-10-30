"""`CoderHandle.events()` tails the run's journal (`T-3.8` item 8): read from another thread,
it is the journal replay reads, line for line."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo, one_task, solved
from zeus_turn_helpers import WAIT_S

from chickadee.engine import PacketEngine
from chickadee.journal.reader import read_run
from chickadee.schema import JournalEvent
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def test_a_watcher_on_another_thread_sees_what_replay_reads(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    handle = engine.coder(one_task(), session_id=SID)
    seen: list[JournalEvent] = []
    watcher = threading.Thread(target=lambda: seen.extend(handle.events()))
    watcher.start()  # events() starts the run

    result = handle.result()
    watcher.join(WAIT_S)

    assert not watcher.is_alive(), "the stream did not end with the run"
    assert result.outcome == "COMPLETED"
    replayed = list(read_run(resolve_roots(repo), handle.run_id))
    assert [e.seq for e in seen] == [e.seq for e in replayed]
    assert seen == replayed
    assert seen[0].kind == "run_start" and seen[-1].kind == "run_end"
