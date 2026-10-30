"""What the REPL's input reads from the engine: `files`, the paths
`glob` would list where Zeus's view is, never an ignored or denied file, for `@`; and
`holds_secret`, whether a line holds a secret value, which the REPL keeps out of its history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import SID, config_for, start, with_sessions

from chickadee.engine import PacketEngine


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = init_repo(tmp_path / "repo", commit=None)
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("print('hi')\n")
    (root / ".env").write_text("TOKEN=x\n")
    (root / "build").mkdir()
    (root / "build" / "out.bin").write_text("x")
    (root / ".gitignore").write_text("build/\n")
    return with_sessions(root, SID)


def test_files_lists_what_glob_would_and_never_a_denied_or_ignored_one(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine: PacketEngine
    engine, _model = start(monkeypatch, repo, config_for(raw), [])
    try:
        files = engine.files(SID)
    finally:
        engine.close()
    assert "src/app.py" in files
    assert ".env" not in files, "a denied file is never suggested"
    assert "build/out.bin" not in files, "an ignored file is never suggested"


def test_a_line_holding_a_key_is_a_secret_and_words_are_not(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine: PacketEngine
    engine, _model = start(monkeypatch, repo, config_for(raw), [])
    try:
        assert engine.holds_secret("use sk-ant-" + "a" * 24 + " for this")
        assert not engine.holds_secret("fix the retry in src/app.py")
    finally:
        engine.close()
