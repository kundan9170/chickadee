"""A command that assumes a prepared repository refuses before it starts."""

from __future__ import annotations

import pathlib

import pytest

from chickadee.init import InitError, init, publish, require_initialised


def test_run_without_init_fails_with_the_instruction(repo: pathlib.Path) -> None:
    with pytest.raises(InitError, match="run `chickadee init` first"):
        require_initialised(repo)


def test_a_partially_published_generation_is_not_initialised(repo: pathlib.Path) -> None:
    """The marker is written last precisely so this reads as "not ready" rather than as a
    generation a run would then consume half of."""
    init(repo)
    publish.state_path(repo).write_text(
        '{"layout_version": 1, "generation": "g", "complete": false}'
    )
    with pytest.raises(InitError, match="no complete generation"):
        require_initialised(repo)


def test_after_init_the_guard_passes(repo: pathlib.Path) -> None:
    init(repo)
    require_initialised(repo)
