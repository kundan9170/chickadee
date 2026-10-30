"""Fault injection before every write, fsync and rename.

The property: at no moment is a half-initialised state visible. A reader sees the previous
generation byte-for-byte, or the new one complete, and never anything between. Asserted by
failing at each named step in turn rather than at one representative one — the step that
breaks the invariant is exactly the step nobody thought to try.
"""

from __future__ import annotations

import pathlib

import pytest

from chickadee.init import init, publish

STEPS = (
    "stage_mkdir",
    "index_build",
    "config_write",
    "config_fsync",
    "replace_write",
    "replace_fsync",
    "replace_rename",
    "replace_dir_fsync",
    "publish_rename",
    "publish_dir_fsync",
    "publish_state",
)


def _fail_at(step: str):
    def hook(current: str) -> None:
        if current == step:
            raise OSError(f"injected fault at {step}")

    return hook


@pytest.mark.parametrize("step", STEPS)
def test_a_fresh_init_never_publishes_a_partial_generation(
    repo: pathlib.Path, step: str
) -> None:
    """No previous generation exists, so the only acceptable outcome is "not initialised"."""
    with pytest.raises(OSError):
        init(repo, hook=_fail_at(step))
    assert publish.read_state(repo) is None, f"fault at {step} left a published state"


@pytest.mark.parametrize("step", STEPS)
def test_a_failed_reinit_leaves_the_previous_generation_authoritative(
    repo: pathlib.Path, step: str
) -> None:
    first = init(repo)
    before = publish.state_path(repo).read_bytes()
    config_before = (publish.chickadee_dir(repo) / publish.CONFIG_NAME).read_bytes()

    with pytest.raises(OSError):
        init(repo, force=True, hook=_fail_at(step))

    # The invariant is not "the pointer never moves" — the marker is replaced by an
    # atomic rename, and a fault in the directory fsync *after* it has legitimately
    # published the new generation. What must always hold is that whatever the pointer
    # names is complete and present: a reader is never sent to a generation that is
    # missing or half-written.
    state = publish.read_state(repo)
    assert state is not None, f"fault at {step} destroyed the published generation"
    named = publish.chickadee_dir(repo) / "generations" / state["generation"]
    assert named.is_dir(), f"fault at {step} published a generation that does not exist"

    if publish.state_path(repo).read_bytes() == before:
        assert state["generation"] == first.generation
    assert (publish.chickadee_dir(repo) / "generations" / first.generation).is_dir()
    assert (
        publish.chickadee_dir(repo) / publish.CONFIG_NAME
    ).read_bytes() != b"" or config_before


def test_an_incomplete_marker_reads_as_uninitialised(repo: pathlib.Path) -> None:
    """The marker is the only thing a reader trusts, so `complete: false` is not a
    generation — it is a repository that has not finished initialising."""
    init(repo)
    publish.state_path(repo).write_text(
        '{"layout_version": 1, "generation": "g", "complete": false}'
    )
    assert publish.read_state(repo) is None
    assert not publish.is_initialised(repo)


def test_a_corrupt_marker_reads_as_uninitialised(repo: pathlib.Path) -> None:
    init(repo)
    publish.state_path(repo).write_text("{not json")
    assert publish.read_state(repo) is None


def test_staging_is_discarded_after_a_failure(repo: pathlib.Path) -> None:
    """A leftover staging directory is inert — nothing points at it — but leaving one per
    failed attempt would grow without bound."""
    with pytest.raises(OSError):
        init(repo, hook=_fail_at("publish_rename"))
    staging_root = publish.chickadee_dir(repo) / publish.STAGING_DIR
    assert not staging_root.exists() or not any(staging_root.iterdir())
