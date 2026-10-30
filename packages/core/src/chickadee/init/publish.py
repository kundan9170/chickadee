"""Generation staging and atomic publication.

The property this file exists for: **at no moment is a half-initialised state visible.**
A reader either sees the previous generation byte-for-byte, or the new one complete, and
never anything between. That is why the order is fixed rather than convenient —

    stage → fsync files → rename directory → fsync parent → publish state.json

— and why `state.json` carries `complete: true` and is written last. It is the only thing
a reader consults, so until it names the new generation the new generation does not exist,
however much of it is already on disk. A fresh init has no marker at all, so a concurrent
reader reports "initialization incomplete" rather than consuming partial state.

`fsync` on the parent directory is the step that is easy to omit and silently wrong: a
rename is not durable until the directory entry is, so a crash after rename and before that
fsync can leave the pointer swinging at a directory that no longer exists.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from chickadee.durable import Fault, make_dir, rename_durably, write_atomic

CHICKADEE_DIR = ".chickadee"
STAGING_DIR = ".init"
GENERATIONS_DIR = "generations"
STATE_NAME = "state.json"
CONFIG_NAME = "config.toml"
LAYOUT_VERSION = 1

GIT_EXCLUDE_LINE = f"{CHICKADEE_DIR}/"

# Set by tests to fail at a named point. Production passes `None`; every call site names
# its step so a fault can be injected before each write, fsync and rename.
FaultHook = Callable[[str], None] | None


def fault(hook: FaultHook, step: str) -> None:
    """Fire the hook at a named step. Public because `init` names steps of its own between
    publication's, the index build among them."""
    if hook is not None:
        hook(step)


def chickadee_dir(root: Path) -> Path:
    return root / CHICKADEE_DIR


def state_path(root: Path) -> Path:
    return chickadee_dir(root) / STATE_NAME


def read_state(root: Path) -> dict[str, Any] | None:
    """The published generation, or None when the repository is not initialised.

    A `complete: false` marker reads as None: a partially published generation is not a
    generation, and treating it as one is the failure this whole module prevents.
    """
    try:
        state = json.loads(state_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if state.get("complete") is True else None


def is_initialised(root: Path) -> bool:
    return read_state(root) is not None


def stage(root: Path, generation: str, hook: FaultHook = None) -> Path:
    """Create the staging directory for a new generation."""
    staging = chickadee_dir(root) / STAGING_DIR / generation
    fault(hook, "stage_mkdir")
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def _steps(hook: FaultHook, names: Mapping[Fault, str]) -> Callable[[Fault], None] | None:
    """Name `durable`'s fault points with init's step names, so a
    test fails init at the step it names. A point with no step here fires nothing."""
    if hook is None:
        return None
    named = hook

    def fire(point: Fault) -> None:
        step = names.get(point)
        if step is not None:
            named(step)

    return fire


_REPLACE_STEPS: Mapping[Fault, str] = {
    Fault.BEFORE_WRITE: "replace_write",
    Fault.AFTER_WRITE: "replace_fsync",
    Fault.AFTER_FSYNC: "replace_rename",
    Fault.AFTER_RENAME: "replace_dir_fsync",
}


def write_file(
    path: Path, content: str, hook: FaultHook = None, *, step: str = "write"
) -> None:
    """Write one staged file, whole and synced, through `durable.write_atomic`."""
    write_atomic(
        path,
        content.encode(),
        fault=_steps(
            hook, {Fault.BEFORE_WRITE: f"{step}_write", Fault.AFTER_WRITE: f"{step}_fsync"}
        ),
    )


def publish(root: Path, staging: Path, generation: str, hook: FaultHook = None) -> Path:
    """Move a complete staging directory into place, then publish the pointer.

    The pointer is written last and is the only thing a reader trusts, so a crash anywhere
    before it leaves the previous generation authoritative and the new one merely present
    and unreferenced.
    """
    generations = chickadee_dir(root) / GENERATIONS_DIR
    make_dir(generations)
    target = generations / generation

    # A rename is only durable once the directory holding the new entry is synced: until
    # then a crash can leave the pointer naming a directory that is not there.
    rename_durably(
        staging,
        target,
        fault=_steps(
            hook, {Fault.AFTER_FSYNC: "publish_rename", Fault.AFTER_RENAME: "publish_dir_fsync"}
        ),
    )

    marker = (
        json.dumps(
            {"layout_version": LAYOUT_VERSION, "generation": generation, "complete": True},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    fault(hook, "publish_state")
    write_atomic(state_path(root), marker.encode(), fault=_steps(hook, _REPLACE_STEPS))
    return target


def discard_staging(root: Path, generation: str) -> None:
    """Remove an unreferenced staging generation after a failure.

    Best effort by design: what must hold is that the *previous* generation is untouched,
    and it is, because nothing above writes outside staging until the final rename. A
    leftover staging directory is inert — no `state.json` points at it.
    """
    import shutil

    staging = chickadee_dir(root) / STAGING_DIR / generation
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)


def ensure_git_excluded(exclude_file: Path, hook: FaultHook = None) -> bool:
    """Add `.chickadee/` to the resolved exclude file, once. True if it was added.

    Never the user's tracked `.gitignore`: that is their file, it is committed, and
    editing it puts our metadata in their next diff.
    """
    existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
    if any(line.strip() == GIT_EXCLUDE_LINE for line in existing.splitlines()):
        return False
    updated = existing if existing.endswith("\n") or not existing else existing + "\n"
    write_atomic(
        exclude_file,
        (updated + GIT_EXCLUDE_LINE + "\n").encode(),
        fault=_steps(hook, _REPLACE_STEPS),
    )
    return True
