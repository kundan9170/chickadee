"""Fixtures for the context suite. Plain helpers live in `context_helpers.py`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from context_helpers import new_session

from chickadee.context.session import Session


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """The directory a session reads. Not a git worktree, so the fs-watchdog has nothing to
    guard, and not named like one for the same reason."""
    path = tmp_path / "ws"
    path.mkdir()
    return path


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    opened = new_session(tmp_path)
    yield opened
    opened.close()
