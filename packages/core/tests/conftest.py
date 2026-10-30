"""Shared fixtures for the core suite.

The `fs-watchdog` arrived with `T-2.13` under D61 and is armed below for the whole suite.
The remaining heavyweight fixtures the specs call for — scripted models, fixture
repositories — still arrive with the tasks that need them, so that a fixture and the thing
it guards land in the same review.
"""

from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    """The workspace root, for tests that read committed docs, profiles or goldens."""
    return REPO_ROOT


# --- fs-watchdog ---------------------------------

WORKTREE_FIXTURES = ("worktree", "repo", "empty_repo", "work", "source", "root")
"""Fixture names that may hold a worktree. Candidates only — `is_worktree` decides, so a
fixture returning a plain temporary directory is not guarded."""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "writes_worktree(*paths): this test deliberately writes these paths inside a "
        "worktree without the Applier (D61). It declares a set, it does not switch the "
        "guard off — any write outside the declared paths still fails. A bare marker with "
        "no paths is rejected, because 'I write something somewhere' is not a declaration.",
    )


@pytest.fixture(autouse=True)
def _fs_watchdog(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    """Invariant 1, asserted for every test that touches a real worktree.

    Depends on nothing and resolves fixtures lazily, so it runs **after** the fixtures that
    build the repository — their setup writes are arrangement, not simulation, and tripping
    on them would say nothing about invariant 1.
    """
    from watchdog import fs_watchdog, is_worktree

    marker = request.node.get_closest_marker("writes_worktree")
    declared: tuple[str, ...] = tuple(marker.args) if marker else ()
    if marker is not None and not declared:
        pytest.fail(
            "writes_worktree needs the paths it writes, e.g. "
            'writes_worktree("app.py"). A bare marker disables the guard for the whole '
            "test, which is what D61 exists to prevent."
        )

    guarded: pathlib.Path | None = None
    for name in WORKTREE_FIXTURES:
        if name not in request.fixturenames:
            continue
        value = request.getfixturevalue(name)
        if is_worktree(value):
            guarded = value
            break

    if guarded is None:
        yield
        return
    with fs_watchdog(guarded, allow=declared):
        yield
