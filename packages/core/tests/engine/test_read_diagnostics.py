"""What a read's diagnostics line says through the real composition: Zeus's reader over the
checkout, and the worker's over the episode's worktree.

Pinned before the readers were given the graph, when every read said no index was attached.
Moved with T-3.170, listed in its Log: each reader is given its root's
index, so a read says what the index holds. `app.py` is clean, so it reads `none` where a
server judged it, and `unavailable` where none could run (no server installed, or none sealed
under `require_sandbox`); a text file no server judges is `unavailable` everywhere. Never
`none` for a file nothing judged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, athena_accepts, box_call, build, make_repo, worker_edit
from repos import git
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

from chickadee.engine import PacketEngine

DETACHED = "[diagnostics: unavailable — no diagnostics index is attached to this read]"
"""What every read said before T-3.170."""
CLEAN_PY = {"[diagnostics: none]", "[diagnostics: unavailable — no analyzer verdict for .py]"}
TEXT = "[diagnostics: unavailable — no analyzer verdict for .txt]"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = make_repo(tmp_path / "repo")
    (root / "notes.txt").write_text("a note\n", encoding="utf-8")
    git(root, "add", "notes.txt")
    git(root, "commit", "-q", "-m", "a note")
    return root


def _diagnostics(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("[diagnostics:")]


def test_zeus_reads_say_what_the_checkouts_index_holds(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(
            call("read", "r1", path="app.py", depth="full"),
            call("read", "r2", path="notes.txt", depth="full"),
        ),
        answer("read"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "read them").wait(timeout=WAIT_S)
    said = {}
    for cid in ("r1", "r2"):
        line = state(engine, SID).result(cid)
        assert line is not None
        [said[cid]] = _diagnostics(line.text)
    assert said["r1"] in CLEAN_PY
    assert said["r2"] == TEXT


def test_the_workers_read_says_what_the_worktrees_index_holds(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    work = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}
    script = [
        calls(call("coder", "c1", **work)),
        calls(box_call("read", "w0", path="app.py", depth="full")),
        *worker_edit("return 1", "return 2"),
        *athena_accepts(),
        answer("done"),
    ]
    engine, model = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
    seen = {
        line
        for request in model.requests
        for message in request.messages
        if message.role == "tool"
        for line in _diagnostics(str(message.content))
    }
    assert seen and seen <= CLEAN_PY
    assert DETACHED not in seen
