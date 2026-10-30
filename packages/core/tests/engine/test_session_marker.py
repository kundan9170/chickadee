"""Every process a session spawns names that session.

`CHICKADEE_SESSION` tells two sessions' children apart. The binding
was process-wide and nothing in production set it, so every spawn carried an empty value. Each
conversation now binds its own worker thread before its build runs, and one engine's two open
sessions each name their own children.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import (
    WAIT_S,
    Step,
    answer,
    call,
    calls,
    config_for,
    start,
    with_sessions,
)

import chickadee.engine as engine_module
from chickadee.engine import PacketEngine
from chickadee.isolation.none import Passthrough
from chickadee.process.supervisor import scrub_env


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), "S-one", "S-two")


@pytest.fixture
def engines() -> Iterator[list[PacketEngine]]:
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()


def marked(sid: str) -> Step:
    """A response that runs one shell command writing the session it sees to a file. The
    model is asked again after the command's result, so each is followed by an answer."""
    return calls(
        call(
            "shell",
            f"s-{sid}",
            command=f'printf %s "$CHICKADEE_SESSION" > {sid}.seen',
        )
    )


def test_each_session_s_shell_names_its_own_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Two sessions open in one engine at once, each on its own worker: each command sees its
    own session, and the caller's thread, which runs no session, sees none. The turns run one
    after the other only so the scripted model answers them in a known order: both sessions
    stay open throughout, which a process-wide binding could not tell apart."""
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    engine, _ = start(
        monkeypatch,
        repo,
        config_for(raw),
        [marked("S-one"), answer(), marked("S-two"), answer()],
    )
    engines.append(engine)

    engine.turn("S-one", "go").wait(timeout=WAIT_S)
    engine.turn("S-two", "go").wait(timeout=WAIT_S)

    assert (repo / "S-one.seen").read_text(encoding="utf-8") == "S-one"
    assert (repo / "S-two.seen").read_text(encoding="utf-8") == "S-two"
    assert scrub_env()["CHICKADEE_SESSION"] == ""


def test_a_session_s_spawns_keep_its_name_turn_after_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The binding is the worker thread's own, so a later turn's spawn still carries it."""
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    later = calls(
        call(
            "shell",
            "s-later",
            command='printf %s "$CHICKADEE_SESSION" > later.seen',
        )
    )
    engine, _ = start(
        monkeypatch, repo, config_for(raw), [marked("S-one"), answer(), later, answer()]
    )
    engines.append(engine)

    engine.turn("S-one", "go").wait(timeout=WAIT_S)
    engine.turn("S-one", "again").wait(timeout=WAIT_S)

    assert (repo / "later.seen").read_text(encoding="utf-8") == "S-one"
