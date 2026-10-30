"""Isolation is wrap input. It is not a policy class.

`SandboxLevel` is **re-exported, not redefined** — `schema/execution.py` owns it, because
`ExecResult.sandbox` carries it on a G0-frozen row. Keeping the name importable from here is
what lets this package read naturally without there being two classes.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chickadee.schema.execution import SandboxLevel as SandboxLevel


@dataclass(frozen=True)
class Isolation:
    """What wrap actually needs. Who chose the argv is the caller's problem."""

    argv: tuple[str, ...]
    cwd: Path
    network_off: bool = True
    repo_writable: bool = False
    writable: tuple[Path, ...] = ()
    """Paths bound writable beside `cwd`, and only when `repo_writable` is: a linked worktree's
    git state (HEAD, its index, the refs) lives in the repository's common git directory,
    outside the worktree, so a declared `git switch` there needs it. Ignored for a read-only
    wrap, so a caller cannot open a write through it
    alone."""
    read_only: tuple[Path, ...] = ()
    """Bound read-only over a writable workspace, after it: the box shell's worktree
    `.git` (a file in a linked worktree, a directory in place), so a command cannot rewrite
    git's hooks or config. bwrap: `--ro-bind`; Seatbelt: a `deny file-write*` row; container:
    a read-only mount. Each must exist; the caller names only what does."""
    deny_read: tuple[Path, ...] = ()
    """Absolute paths the wrap must make UNREADABLE — a read fails with a permission
    error, never returns empty. Resolved by the caller from `secrets/denylist.py`
    `resolve(roots)` over the workspace and $HOME; `()` under `allow_secret_read`.
    Isolation does not know the deny list, the same way it does not know PolicyClass:
    it is told paths, and it masks them."""
    deny_patterns: tuple[str, ...] | None = None
    """The bare-name half of the resolved deny list, as globs: the patterns with
    no `/` (`*.pem`, `id_rsa*`, a project's `*.kdbx`), which match a file by its name
    wherever it sits. A backend that can match names anywhere on disk renders them
    (Seatbelt's regex rows); the directory-shaped half (`.ssh/**`) reaches the wrap as
    resolved paths in `deny_read`.

    A caller that resolved a list sets it from `Denylist.name_patterns()`, which is `()`
    when the list is off (`allow_secret_read`) or keeps no bare-name pattern. An empty
    `deny_read` does not mean the list is off: a removal can leave nothing concrete to mask
    while `*.pem` and the rest stay in force, and these patterns are what cover a key that
    appears mid-session or outside the scanned roots. `None` keeps the reading callers
    relied on before, the built-in rows while `deny_read` is non-empty, for a caller that
    resolved no list."""
    spared_dirs: tuple[str, ...] = ()
    """Directories, as path tails (`site-packages/certifi`), whose files no certificate pattern
    masks: `Denylist.trust_stores().dirs`, the deny list's own statement."""
    spared_paths: tuple[Path, ...] = ()
    """Paths at or below which no certificate pattern masks: the system's
    certificate stores and the CA bundles the environment names (`trust_stores().paths`)."""
    spared_patterns: tuple[str, ...] = ()
    """The name patterns those two spare: the list's certificate patterns (`*.pem`, `*.crt`,
    `*.cer`). Every other pattern masks its files wherever they sit. Seatbelt renders all three
    as exceptions to those rows; bubblewrap masks only `deny_read`, which `resolve` already
    leaves them out of."""
    scratch: tuple[Path, ...] = ()
    """Directories writable in every wrap, read-only or not: the repository's
    tool cache, where a package manager keeps what it downloads. The harness's own scratch,
    never the person's state: not recorded, never undone. Each must exist."""
    tmp: Path | None = None
    """The session's temporary directory, writable in every wrap; bubblewrap
    binds it over `/tmp` as well, so a file one command leaves there is there for the next.
    `None` keeps a private, empty `/tmp` per command. Must exist."""


def harness_cache() -> Path:
    """The harness's own cache directory: `$XDG_CACHE_HOME/chickadee`, or `~/.cache/chickadee`.
    Never `/tmp`, which bubblewrap replaces, so nothing kept here vanishes under a wrap."""
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "chickadee"


def tool_cache(repository_id: str) -> Path:
    """The repository's tool cache, created on first use: kept across sessions
    of one repository (its id is the same in every linked worktree, `init.detect`) and never
    shared between repositories."""
    path = harness_cache() / "tool-cache" / repository_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_tmp(session_id: str) -> Path:
    """The session's temporary directory, created on first use and removed when
    the session closes (`remove_session_tmp`)."""
    path = harness_cache() / "tmp" / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_session_tmp(session_id: str) -> None:
    """The session's temporary directory gone, with what its commands left there: a directory
    a command left read-only is made writable to empty it, and anything that still cannot go
    stays for `sweep_session_tmp`."""

    def writable_then_again(
        fn: Callable[[str], object], path: str, _err: BaseException
    ) -> None:
        try:
            os.chmod(os.path.dirname(path), 0o700)
            if os.path.isdir(path) and not os.path.islink(path):
                os.chmod(path, 0o700)
            fn(path)
        except OSError:
            pass

    shutil.rmtree(harness_cache() / "tmp" / session_id, onexc=writable_then_again)


SESSION_TMP_MAX_AGE_S: Final = 7 * 24 * 3600
"""A session's temporary directory untouched this long is a crashed session's."""


def sweep_session_tmp(max_age_s: float = SESSION_TMP_MAX_AGE_S) -> None:
    """Remove the temporary directories of sessions that ended without closing (a crash, a
    kill): every one not changed for `max_age_s`. A session still open touches its own each
    command, so it is never one of them in practice; scratch either way."""
    root = harness_cache() / "tmp"
    if not root.is_dir():
        return
    now = time.time()
    for entry in root.iterdir():
        try:
            stale = entry.is_dir() and now - entry.stat().st_mtime > max_age_s
        except OSError:
            continue
        if stale:
            remove_session_tmp(entry.name)


def mask_entries() -> tuple[Path, Path]:
    """The two mode-000 scratch entries a mask binds over a denied path:
    `mask-file` for a file, `mask-dir` for a directory.

    Under the harness's own cache directory — never `/tmp`, which bubblewrap replaces
    with a private tmpfs, so a source there would vanish under the wrap. Created on first
    use and kept: the entries are contentless and the mode is the whole point, so there is
    nothing to refresh. Mode 000 is re-applied every call, because a chmod outside the
    sandbox would silently turn the mask into a readable empty file — exactly the
    "never returns empty" failure the design forbids.
    """
    base = harness_cache() / "mask"
    base.mkdir(parents=True, exist_ok=True)
    file_ = base / "mask-file"
    dir_ = base / "mask-dir"
    if not file_.exists():
        file_.touch()
    dir_.mkdir(exist_ok=True)
    for entry in (file_, dir_):
        entry.chmod(0)
    return file_, dir_


def mask_for(path: Path) -> Path:
    """Which scratch entry masks this path — a directory gets the directory mask so that
    `ls` fails the same way a `cat` does, rather than showing an empty listing."""
    file_, dir_ = mask_entries()
    return dir_ if path.is_dir() else file_
