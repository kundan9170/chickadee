"""A code file's read through the real composition: Zeus's reader over the checkout,
and the worker's over the episode's worktree, each over its root's index.

Pinned before the readers were given the graph as their index: a read of `app.py` said it had
no index, `at=` was refused, and the worker's `at="@changed"` said nothing tracked changes.
Moved with T-3.185, listed in its Log: `app.py` fits the inline budget, so it is read whole and
says so; `at="go"` resolves the function, whose lines that read already holds; the worker's
`@changed` after its edit returns the line it changed, against the run's base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, athena_accepts, box_call, build, make_repo
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

from chickadee.context.parts import WHOLE_FILE
from chickadee.engine import PacketEngine

NO_INDEX = "no index for this file, so no outline"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def test_zeus_reads_app_py_whole_and_resolves_a_function_by_name(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(call("read", "r1", path="app.py"), call("read", "r2", path="app.py", at="go")),
        answer("read"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "read it").wait(timeout=WAIT_S)
    said = {}
    for cid in ("r1", "r2"):
        line = state(engine, SID).result(cid)
        assert line is not None
        said[cid] = line.text
    assert NO_INDEX not in said["r1"]
    assert WHOLE_FILE.format(n=2) in said["r1"]
    assert "1│def go() -> int:" in said["r1"]
    # `go` resolves; its lines are the ones the whole-file read already sent, so the
    # answer is D214's pointer to them (it was the outline and "already hold view").
    assert "INDEX_UNAVAILABLE" not in said["r2"]
    assert "[you already hold lines 1-2" in said["r2"]


def test_the_workers_at_changed_returns_the_line_it_edited(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    work = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}
    script = [
        calls(call("coder", "c1", **work)),
        calls(box_call("edit", "w1", path="app.py", old="return 1", new="return 2")),
        calls(box_call("read", "w2", path="app.py", at="@changed")),
        answer("T1: edited app.py."),
        *athena_accepts(),
        answer("done"),
    ]
    engine, model = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
    tool_texts = [
        str(message.content)
        for request in model.requests
        for message in request.messages
        if message.role == "tool"
    ]
    assert not any("@changed needs a base" in t for t in tool_texts), tool_texts
    assert any("app.py — view 2-2 of 2" in t and "2│    return 2" in t for t in tool_texts), (
        tool_texts
    )
