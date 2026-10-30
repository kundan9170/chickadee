"""The `fs-watchdog` — invariant 1, asserted rather than trusted.

Invariant 1 says only the Applier mutates the worktree. The risk this guards is not
production code (which is reviewed) but *tests*: a verifier or orchestrator test that
hand-writes a file into a worktree to simulate an apply is testing that component against a
state the Applier could never have produced, and every gate the Applier would have run is
skipped silently.

**Armed for the test body, after fixtures have built their state.** A fixture that creates a
repository has to write files; that is setup, not simulation. `conftest.py` arms this around
the body only, so arrangement done in a fixture is never the thing that trips it.

**Scope: this is a test-suite rule, not a runtime mechanism.** It is never imported by the
harness, never installed, and has no effect on anyone running Chickadee. It exists inside a
`pytest` session, for tests that use a git-repo fixture, for the duration of the test body.
The *runtime* rule — only the Applier and Executor change files — holds because no other
component contains code that writes; this stops our own tests from faking around it.

**Known limitation: it only sees writes that go through Python.** The wrappers replace
Python-level functions, so a test that shells out — `subprocess.run(["rm", "file"])`, or a
`git checkout` — mutates the worktree in a separate process where none of them exist, and
nothing fires. Closing that would mean sandboxing the test process itself, which is a much
larger thing than a fixture. Stated here because a guard whose blind spot is undocumented is
one people will assume covers more than it does. D61 §"What this does not cover" is the
record.

**A deliberate write is declared, not forbidden**. Some components — the effect
bracket, the graph indexer — measure the filesystem and must touch it. Those tests carry
`@pytest.mark.writes_worktree`. The goal was never to make such writes impossible, only to
make them impossible to do *silently*: the marker is the complete, greppable list of places
the suite mutates a worktree outside the Applier.
"""

from __future__ import annotations

import builtins
import fnmatch
import inspect
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

APPLIER_PACKAGE = os.path.join("chickadee", "applier")
EDIT_PORT = os.path.join("chickadee", "box", "edits.py")
"""The box's edit port: the task loop's one home of model-authored writes,
beside the Applier until `T-3.157` retires it (`test_invariant_spawn_and_writes.WRITERS`)."""

F_GETPATH = 50
"""macOS `fcntl` command for "what path is this descriptor". Not exposed by the `fcntl`
module, so it is named here rather than left as a bare 50 at the call site."""

NOT_THE_WORKTREE = (".git", ".chickadee")
"""Directories inside a worktree that the Applier does **not** own.

Invariant 1 is about the source tree a run mutates, not about the harness's own bookkeeping.
`.chickadee/` holds journals, blobs, the graph database and run worktrees; `.git/` is git's.
The journal writer, the graph store and `init` all write there by design, and guarding them
would make the watchdog fire on 96 tests that are doing exactly what they should."""


class WorktreeWriteError(AssertionError):
    """A test wrote inside a worktree without going through the Applier."""


def dir_of(fd: int) -> Path | None:
    """The directory a descriptor refers to, or None if it cannot be resolved.

    The `*at`-style calls take a path **relative to a descriptor**, not to the working
    directory, so `os.open("a.txt",..., dir_fd=worktree_fd)` writes inside the worktree
    while the string says nothing about where that is. Without this the containment check
    compares `a.txt` against the cwd and finds no worktree at all.
    """
    try:
        if sys.platform == "darwin":
            import fcntl

            raw = fcntl.fcntl(fd, F_GETPATH, b"\0" * 1024)
            return Path(raw.split(b"\0", 1)[0].decode())
        return Path(os.readlink(f"/proc/self/fd/{fd}"))
    except (OSError, ValueError, ImportError):  # pragma: no cover - platform dependent
        return None


def _through_applier() -> bool:
    """Whether any frame above us belongs to the applier package or the box's edit port.

    Frames rather than a flag: a flag can be set by the very code that should not be
    writing, while the stack cannot be faked by the caller it is being read on behalf of.
    """
    return any(
        APPLIER_PACKAGE in frame.filename or frame.filename.endswith(EDIT_PORT)
        for frame in inspect.stack()
    )


@contextmanager
def fs_watchdog(worktree: Path, allow: tuple[str, ...] = ()) -> Iterator[list[str]]:
    """Fail on any write under `worktree` that did not come through the Applier.

    `allow` is the set of paths a test has **declared** it writes. Everything else still
    raises, so a declaration permits exactly what it names rather than switching the guard
    off — a marked test doing one extra undeclared write is still caught.

    Patches every primitive that can mutate a path, not just the convenient ones.

    The first version patched `Path.write_*` and `os.open` and believed that was enough. It
    was not: CPython's `builtins.open` reaches the OS through C rather than through the
    Python-level `os.open`, so the single most common way to write a file went straight
    past the guard — and no delete was intercepted at all, so a test could remove a file
    from the worktree unobserved. A watchdog with holes is worse than none, because the
    suite is then trusted on the strength of a check that was not running.
    """
    root = Path(os.path.realpath(worktree))
    seen: list[str] = []
    real_write_bytes = Path.write_bytes
    real_write_text = Path.write_text
    real_open = os.open
    real_replace = os.replace
    real_builtin_open = builtins.open
    real_unlink = os.unlink
    real_remove = os.remove
    real_path_unlink = Path.unlink
    real_rename = os.rename
    real_rmdir = os.rmdir

    def as_path(target: object) -> Path | None:
        """Normalise anything the OS accepts as a path.

        `str(target)` was wrong for two of them: a bytes path stringifies to `"b\'/x\'"`,
        which resolves nowhere near the worktree, so every `os.open(b"...")` slipped past
        unchecked. An integer is a file descriptor and names no path at all.
        """
        if isinstance(target, int):
            return None
        try:
            return Path(os.fsdecode(os.fspath(target)))  # type: ignore[arg-type]
        except (TypeError, ValueError):  # pragma: no cover - not a path-like
            return None

    def under_root(target: object) -> str | None:
        """The worktree-relative path, or None if this is not a worktree entry.

        Checks the path **both** as written and fully resolved, and a hit on either counts.
        Resolving alone missed an in-worktree symlink: `unlink(worktree/link)` resolves to
        whatever the link points at, so removing an entry that lives in the worktree was
        classified as touching something outside it.
        """
        path = as_path(target)
        if path is None:
            return None
        try:
            # Resolve the *directory chain* but not the final entry. `abspath` alone was not
            # enough: `root` is realpath'd, and on macOS `/tmp` is a symlink to
            # `/private/tmp`, so an unresolved path never matched it. Keeping the last
            # component unresolved is what makes an in-worktree symlink count as an entry
            # in the worktree rather than as whatever it points at.
            entry = Path(os.path.realpath(path.parent)) / path.name
            resolved = Path(os.path.realpath(path))
        except (OSError, ValueError):  # pragma: no cover - defensive
            return None
        lexical = entry

        for candidate in (lexical, resolved):
            if candidate != root and not str(candidate).startswith(str(root) + os.sep):
                continue
            rel = candidate.relative_to(root)
            if any(part in NOT_THE_WORKTREE for part in rel.parts):
                return None
            return str(rel)
        return None

    def guard(target: object, how: str) -> None:
        rel = under_root(target)
        if rel is None:
            return
        seen.append(rel)
        if any(fnmatch.fnmatch(rel, pattern) for pattern in allow):
            return
        if not _through_applier():
            caller = next((f for f in inspect.stack() if "watchdog.py" not in f.filename), None)
            where = f"{caller.filename}:{caller.lineno}" if caller else "unknown"
            raise WorktreeWriteError(
                f"{where} wrote {rel} inside the worktree via {how} without the Applier. "
                "Invariant 1: only the Applier mutates. If the write is deliberate, declare "
                f'it: @pytest.mark.writes_worktree("{rel}"). Declared paths so far: '
                f"{list(allow) or 'none'}."
            )

    def patched_write_bytes(self: Path, data: object) -> int:  # type: ignore[override]
        guard(self, "Path.write_bytes")
        return real_write_bytes(self, data)  # type: ignore[arg-type]

    def patched_write_text(self: Path, data: str, *a: object, **kw: object) -> int:  # type: ignore[override]
        guard(self, "Path.write_text")
        return real_write_text(self, data, *a, **kw)  # type: ignore[arg-type]

    def at(path: object, fd_key: str, kw: dict[str, object]) -> object:
        """Rebase a descriptor-relative path onto the directory the descriptor names."""
        fd = kw.get(fd_key)
        resolved = as_path(path)
        if not isinstance(fd, int) or resolved is None or resolved.is_absolute():
            return path
        base = dir_of(fd)
        return base / resolved if base is not None else path

    def patched_open(path: object, flags: int, *a: object, **kw: object) -> int:
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            guard(at(path, "dir_fd", kw), "os.open")
        return real_open(path, flags, *a, **kw)  # type: ignore[arg-type]

    def patched_builtin_open(file: object, mode: str = "r", *a: object, **kw: object):  # type: ignore[no-untyped-def]
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            guard(file, f"open(mode={mode!r})")
        return real_builtin_open(file, mode, *a, **kw)  # type: ignore[arg-type]

    def patched_unlink(path: object, *a: object, **kw: object) -> None:
        guard(at(path, "dir_fd", kw), "os.unlink")
        real_unlink(path, *a, **kw)  # type: ignore[arg-type]

    def patched_remove(path: object, *a: object, **kw: object) -> None:
        guard(at(path, "dir_fd", kw), "os.remove")
        real_remove(path, *a, **kw)  # type: ignore[arg-type]

    def patched_path_unlink(self: Path, *a: object, **kw: object) -> None:  # type: ignore[override]
        guard(self, "Path.unlink")
        real_path_unlink(self, *a, **kw)  # type: ignore[arg-type]

    def patched_rename(src: object, dst: object, *a: object, **kw: object) -> None:
        # Both ends. Guarding only the destination let a file be moved *out* of the
        # worktree — the entry disappears and nothing raised, because the destination was
        # somewhere else entirely.
        guard(at(src, "src_dir_fd", kw), "os.rename (source)")
        guard(at(dst, "dst_dir_fd", kw), "os.rename")
        real_rename(src, dst, *a, **kw)  # type: ignore[arg-type]

    def patched_rmdir(path: object, *a: object, **kw: object) -> None:
        guard(at(path, "dir_fd", kw), "os.rmdir")
        real_rmdir(path, *a, **kw)  # type: ignore[arg-type]

    def patched_replace(src: object, dst: object, *a: object, **kw: object) -> None:
        # The destination, not the temp file. `write_atomic` writes to a same-directory
        # temp and renames onto the target, so without this the log records
        # `.app.py.k3n2.tmp` and never the path anyone cares about.
        guard(at(src, "src_dir_fd", kw), "os.replace (source)")
        guard(at(dst, "dst_dir_fd", kw), "os.replace")
        real_replace(src, dst, *a, **kw)  # type: ignore[arg-type]

    # Everything else that can change a worktree entry. Content is only one kind of
    # mutation: `chmod` alters what a file *is* (the applier had to be fixed for exactly
    # that), `utime` alters what the effect bracket measures, and `mkdir`/`symlink`/`link`
    # create entries with no write call at all. Wrapped generically so the list is a list
    # rather than eight near-identical functions.
    SINGLE_PATH = ("mkdir", "chmod", "chown", "utime", "mkfifo", "mknod", "truncate")
    TWO_PATH = ("symlink", "link")

    originals: dict[str, object] = {}

    def wrap_single(name: str):  # type: ignore[no-untyped-def]
        real = getattr(os, name)

        def patched(path: object, *a: object, **kw: object):  # type: ignore[no-untyped-def]
            guard(at(path, "dir_fd", kw), f"os.{name}")
            return real(path, *a, **kw)

        return real, patched

    def wrap_two(name: str):  # type: ignore[no-untyped-def]
        real = getattr(os, name)

        def patched(src: object, dst: object, *a: object, **kw: object):  # type: ignore[no-untyped-def]
            # The *destination* is the entry being created; the source is only read.
            guard(at(dst, "dst_dir_fd", kw), f"os.{name}")
            return real(src, dst, *a, **kw)

        return real, patched

    for name in SINGLE_PATH + TWO_PATH:
        if not hasattr(os, name):  # pragma: no cover - platform dependent
            continue
        real, patched = (wrap_single if name in SINGLE_PATH else wrap_two)(name)
        originals[name] = real
        setattr(os, name, patched)

    Path.write_bytes = patched_write_bytes  # type: ignore[method-assign]
    Path.write_text = patched_write_text  # type: ignore[method-assign]
    os.open = patched_open  # type: ignore[assignment]
    os.replace = patched_replace  # type: ignore[assignment]
    builtins.open = patched_builtin_open  # type: ignore[assignment]
    os.unlink = patched_unlink  # type: ignore[assignment]
    os.remove = patched_remove  # type: ignore[assignment]
    Path.unlink = patched_path_unlink  # type: ignore[method-assign]
    os.rename = patched_rename  # type: ignore[assignment]
    os.rmdir = patched_rmdir  # type: ignore[assignment]
    try:
        yield seen
    finally:
        for name, real in originals.items():
            setattr(os, name, real)
        Path.write_bytes = real_write_bytes  # type: ignore[method-assign]
        Path.write_text = real_write_text  # type: ignore[method-assign]
        os.open = real_open  # type: ignore[assignment]
        os.replace = real_replace  # type: ignore[assignment]
        builtins.open = real_builtin_open  # type: ignore[assignment]
        os.unlink = real_unlink  # type: ignore[assignment]
        os.remove = real_remove  # type: ignore[assignment]
        Path.unlink = real_path_unlink  # type: ignore[method-assign]
        os.rename = real_rename  # type: ignore[assignment]
        os.rmdir = real_rmdir  # type: ignore[assignment]


def is_worktree(path: Path) -> bool:
    """Whether this directory is a real worktree, i.e. a git repository.

    Identified by what it *is* rather than by which folder the test lives in. A bare
    temporary directory is not a worktree, and guarding one would fail tests that have
    nothing to do with invariant 1.
    """
    if not isinstance(path, Path) or not path.is_dir():
        return False
    marker = path / ".git"
    return marker.is_dir() or marker.is_file()
