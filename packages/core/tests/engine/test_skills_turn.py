"""The skills index reaches Zeus, and a load reaches context.

`PacketEngine.turn()` with a scripted rung: the system message carries an indexed skill's name
and description and never its body; a `load_skill` call answers the body, framed, and the
session log records the load against that call. Lives beside the engine's turn helpers
rather than in `tests/skills/`, whose suite builds no engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import (
    SID,
    WAIT_S,
    call,
    calls,
    config_for,
    result,
    start,
    state,
    with_sessions,
)

from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError, SessionError

BODY = "SECRET-SAUCE: backfill in batches of 500, with a 2s sleep between them.\n"
SKILL = f"---\nname: migrations\ndescription: how migrations are written here\n---\n{BODY}"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`$HOME` and `$XDG_CONFIG_HOME` both point at small directories, never the developer's:
    the user and external skill roots live under them."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = with_sessions(init_repo(tmp_path / "repo", commit=None), SID)
    skills = path / ".chickadee" / "skills"
    skills.mkdir(parents=True)
    (skills / "migrations.md").write_text(SKILL, encoding="utf-8")
    return path


def test_the_index_is_in_the_system_message_and_the_body_is_not(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    system = model.requests[0].messages[0].content
    assert isinstance(system, str)
    assert "- migrations · project · how migrations are written here" in system
    assert "SECRET-SAUCE" not in system


def test_a_load_answers_the_body_and_is_recorded_against_its_call(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [calls(call("load_skill", "k1", name="migrations"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    engine.turn(SID, "write a migration").wait(timeout=WAIT_S)
    assert BODY.rstrip() in result(engine, "k1")
    loads = [e for e, _ in state(engine).lines() if e.kind == "skill_loaded"]
    assert len(loads) == 1


def test_a_skill_named_for_a_built_in_refuses_the_session_naming_the_file(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    bad = repo / ".chickadee" / "skills" / "status.md"
    bad.write_text("---\nname: status\ndescription: d\n---\nx\n", encoding="utf-8")
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    # The build runs off the call, so the refusal reaches the turn's `wait` as a
    # `SessionError` whose cause is the config's.
    with pytest.raises(SessionError, match="status") as refused:
        engine.turn(SID, "hi").wait(timeout=WAIT_S)
    assert isinstance(refused.value.__cause__, ConfigError)
