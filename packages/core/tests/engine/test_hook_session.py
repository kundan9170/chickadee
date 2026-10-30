"""The session's points through the engine: `[[hooks]]` in the config,
bound as the session opens, `session_start` told how it opened, its note a harness notice
Zeus's first request reads, and `session_end` told as the session closes.

The hook is a built-in of the test's own, registered for its length, so the whole path runs:
the config, `bind_session`, the session's `bound` line, the fire site and the notice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import SID, WAIT_S, answer, config_for, start, state, with_sessions

from chickadee.context.standing import NoticeTrigger
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import PacketEngine
from chickadee.hooks import builtins
from chickadee.hooks.builtins import Builtin
from chickadee.hooks.payloads import HookEvent, SessionStart
from chickadee.hooks.points import HookPoint, spec_of
from chickadee.hooks.verdicts import HookVerdict, Note, Proceed, Warn
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


def _probe(
    monkeypatch: pytest.MonkeyPatch,
    raw: dict[str, Any],
    **at: Callable[[HookEvent], HookVerdict],
) -> None:
    """Each of `at` as a built-in of its point, configured in `raw`."""
    rows: list[dict[str, str]] = []
    for point, run in at.items():
        returns = frozenset({Proceed, Note, Warn}) & spec_of(HookPoint(point)).accepts
        probe = Builtin(point, frozenset({HookPoint(point)}), returns, lambda params, r=run: r)
        monkeypatch.setitem(builtins.BUILTINS, point, probe)  # type: ignore[arg-type]
        rows.append({"event": point, "name": point, "use": point})
    raw["hooks"] = rows


def _hook_lines(repo: Path) -> list[dict[str, Any]]:
    log = SessionLog.read(sessions_root(resolve_roots(repo)), SID)
    return [json.loads(log.payload(e))["hook"] for e in log.events() if e.kind == "hook"]


def test_session_start_is_told_how_it_opened_and_its_note_is_read_first(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    opened: list[str] = []

    def hello(event: HookEvent) -> HookVerdict:
        assert isinstance(event.payload, SessionStart)
        opened.append(event.payload.source)
        return Note("main is red since 09:00")

    _probe(monkeypatch, raw, session_start=hello)
    engine, model = start(monkeypatch, repo, config_for(raw), [answer("noted")])
    engines.append(engine)
    engine.turn(SID, "hi").wait(timeout=WAIT_S)

    assert opened == ["open"]
    notices = [
        t.text
        for t in state(engine).turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.HOOK
    ]
    assert notices == ["main is red since 09:00"]
    assert "main is red since 09:00" in str(model.requests[0].messages)


def test_session_end_is_told_as_the_session_closes(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    _probe(monkeypatch, raw, session_end=lambda e: Warn("1 turn, nothing committed"))
    engine, _ = start(monkeypatch, repo, config_for(raw), [answer("hi")])
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    engine.close()

    ended = [line for line in _hook_lines(repo) if line["event"] == "session_end"]
    assert [(line["decision"], line["message"]) for line in ended] == [
        ("warn", "1 turn, nothing committed")
    ]
