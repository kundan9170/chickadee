"""Fixtures for the journal suite."""

from __future__ import annotations

import pathlib

import pytest

from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def worktree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A working tree with a git directory."""
    (tmp_path / ".git" / "info").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def repo(worktree: pathlib.Path) -> Roots:
    """The resolved `Roots` a journal is written under.

    Returns `Roots`, not a path, and that is the migration in one line: `Journal.create`
    takes the roots someone already resolved, so it can no longer be handed a worktree that
    dies with the episode. The fixture keeps its name because every test here means "the
    place journals go", and that place is now typed.
    """
    return resolve_roots(worktree)


@pytest.fixture
def run_id() -> str:
    return "2026-08-18-a1b2c3d4"
