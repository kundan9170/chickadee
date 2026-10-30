"""#542: a run's journal asked for before it has one is `RunNotStarted`, never a missing
session; a session someone named with a run-shaped id is still served from the session."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import build, make_repo

from chickadee.engine import PacketEngine
from chickadee.errors import RunNotStarted


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def test_a_run_with_no_journal_is_not_started(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, [])
    engines.append(engine)
    with pytest.raises(RunNotStarted, match="2026-10-13-be6bf9eb has no journal yet"):
        engine.journal("2026-10-13-be6bf9eb")


def test_a_session_named_like_a_run_is_still_a_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, [])
    engines.append(engine)
    named = "2026-10-13-0000abcd"
    engine.create_session(named)
    page = engine.journal(named)
    assert isinstance(page.records, tuple)
