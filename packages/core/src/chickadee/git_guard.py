"""The files a tool outside the sandbox runs, held across a command that could write them.

git reads `config`, `config.worktree` and `info/`, and runs `hooks/`, on every later call,
the harness's and the person's, outside any sandbox. The hooks folder `core.hooksPath` names
(`.husky/` and the like) is run the same way, and an editor runs what `.vscode/` and `.idea/`
define when the folder is opened. A command that changes one has planted something that runs
where nothing contains it. So a shell that lets a command reach them holds
their bytes before it runs and puts back whatever it changed: Zeus's recorded writes, where a
git command's state is writable (`zeus.writes.ShellWrites`), the box shell, whose approved
command runs outside the sandbox that kept `.git` read-only (`box.shell.BoxShell`), and a job,
across its run (`zeus.jobs`). Inside the sandbox they are read-only as well.

One change is kept: a branch's upstream, to a remote the config already names. It is what an
approved `git push -u` writes, and it names where a push goes only among
remotes the person already has.
"""

from __future__ import annotations

import base64
import json
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Final

from chickadee.durable.files import write_atomic
from chickadee.process import run_git

__all__ = [
    "Held",
    "dump",
    "git_dirs",
    "guarded",
    "held",
    "load",
    "put_back",
    "tampered_note",
]

Held = dict[Path, tuple[bytes, int]]
"""Each guarded file's bytes and mode, by path."""


def git_dirs(root: Path) -> tuple[Path, Path] | None:
    """The repository's common git directory and `root`'s own, resolved; None when `root` is
    not in a repository git can name."""
    done = run_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir", "--git-dir")
    lines = done.stdout.decode("utf-8", "replace").splitlines()
    if done.exit_code != 0 or done.timed_out or len(lines) != 2:
        return None
    return Path(lines[0]).resolve(), Path(lines[1]).resolve()


_CONFIGS = frozenset({"config", "config.worktree"})

_UPSTREAM = re.compile(r"branch\..+\.(remote|merge)")
"""A branch's upstream: the two keys `git push -u` and `git branch --set-upstream-to` write."""


_RUN_BY_TOOLS: Final = (".husky", ".vscode", ".idea")
"""The worktree's folders a tool outside the sandbox runs: husky's hooks,
which git runs once `core.hooksPath` names them, and the editors' tasks and run settings,
which an editor runs when the folder is opened. The same three Claude Code's sandbox refuses."""


def guarded(root: Path) -> tuple[Path, ...]:
    """What a command must not change, wherever it lies, as it exists: git's `config`, the
    worktree's `config.worktree`, `hooks/` and `info/`; the hooks folder `core.hooksPath`
    names; and `_RUN_BY_TOOLS`. `info/` holds the excludes that decide what a snapshot and an
    undo can see, and the attributes and grafts git applies unasked. What `held`
    holds, and what a sandboxed writing command sees read-only."""
    found = git_dirs(root)
    if found is None:
        return ()
    common, own = found
    paths = [common / "config", own / "config.worktree", common / "hooks", common / "info"]
    # The person's own setting: the harness's git sets `core.hooksPath` itself, so it
    # is read without that override, which `hooks=True` leaves off. Reading runs no hook.
    hooks = run_git(root, "config", "--get", "core.hooksPath", hooks=True)
    named = hooks.stdout.decode("utf-8", "replace").strip()
    if hooks.exit_code == 0 and named:
        where = Path(named).expanduser()
        paths.append(where if where.is_absolute() else root / where)
    paths.extend(root / name for name in _RUN_BY_TOOLS)
    found_paths: dict[Path, None] = {}
    for p in paths:
        if p.exists():
            found_paths[p.resolve()] = None
    return tuple(found_paths)


def held(root: Path, paths: tuple[Path, ...] | None = None) -> Held:
    """The bytes and mode of every file `guarded` names: what a command must not change.
    `paths` is `guarded(root)` already found, so a step finds it once for both its holds."""
    kept: Held = {}
    for path in guarded(root) if paths is None else paths:
        files = (p for p in path.rglob("*") if not p.is_dir()) if path.is_dir() else (path,)
        for f in files:
            if f.is_file():
                kept[f] = (f.read_bytes(), stat.S_IMODE(f.stat().st_mode))
    return kept


def put_back(before: Held, after: Held) -> tuple[str, ...]:
    """Every guarded file the command changed, put back as it was; returns their paths. A
    config whose only change is a branch's upstream is kept."""
    changed: list[str] = []
    for path in sorted({*before, *after}):
        was = before.get(path)
        now = after.get(path)
        if now == was:
            continue
        if was is not None and now is not None and _upstream_only(path, was, now):
            continue
        changed.append(str(path))
        if was is None:
            path.unlink(missing_ok=True)
            continue
        data, mode = was
        write_atomic(path, data)
        path.chmod(mode)
    return tuple(changed)


def _upstream_only(path: Path, was: tuple[bytes, int], now: tuple[bytes, int]) -> bool:
    """Whether a config's change, read by git's own parser, only sets or unsets a branch's
    upstream: `remote` naming a remote the config had already, or `.`, and `merge` a branch."""
    if path.name not in _CONFIGS or was[1] != now[1]:
        return False
    before = _entries(path.parent, was[0])
    after = _entries(path.parent, now[0])
    if before is None or after is None:
        return False
    remotes = {k[len("remote.") : -len(".url")] for k, _ in before if _remote_url(k)}
    added = Counter(after) - Counter(before)
    removed = Counter(before) - Counter(after)
    if not added and not removed:
        return False  # bytes moved with no entry changed: not a change git makes
    for key, value in added:
        if _UPSTREAM.fullmatch(key) is None:
            return False
        if key.endswith(".remote") and value not in remotes and value != ".":
            return False
        if key.endswith(".merge") and not value.startswith("refs/heads/"):
            return False
    return all(_UPSTREAM.fullmatch(key) is not None for key, _ in removed)


def _remote_url(key: str) -> bool:
    return key.startswith("remote.") and key.endswith(".url")


def _entries(cwd: Path, data: bytes) -> list[tuple[str, str]] | None:
    """A config's entries as git reads them, or None when git cannot read it."""
    done = run_git(cwd, "config", "--file", "-", "--list", "-z", stdin=data)
    if done.exit_code != 0 or done.timed_out:
        return None
    out: list[tuple[str, str]] = []
    for item in done.stdout.decode("utf-8", "replace").split("\0"):
        if item:
            key, _, value = item.partition("\n")
            out.append((key, value))
    return out


def dump(kept: Held) -> bytes:
    """`kept` as bytes a session blob holds, so resume can put back what a lost job changed."""
    rows = {
        str(path): [base64.b64encode(data).decode("ascii"), mode]
        for path, (data, mode) in kept.items()
    }
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load(data: bytes) -> Held:
    """What `dump` wrote."""
    rows: dict[str, list[str | int]] = json.loads(data)
    return {
        Path(path): (base64.b64decode(str(row[0])), int(row[1])) for path, row in rows.items()
    }


def tampered_note(tampered: tuple[str, ...], put_back: tuple[str, ...]) -> str:
    """What a result says of a command that changed git's own files."""
    return (
        f"[effect check: changed {', '.join(tampered)}, which git runs outside any sandbox; "
        f"the call failed, and every path it changed was put back from its snapshot "
        f"({', '.join(put_back) or 'none'})]"
    )
