"""Repository WRITE lease, worktree isolation and in-place snapshots.

Three jobs, in the order they must happen: take a repository-wide lease so no other engine
process mutates this repository at the same time; give the run somewhere isolated to write;
and — for the one mode that writes into the user's own checkout — snapshot what was already
there, durably, before anything touches it.

**In-place is a choice, never a fallback.** D39 makes the leased worktree the default for
autonomous multi-packet work and `--in-place` the person's explicit option, on any job since
turn snapshots gave a run in place its undo point. So a worktree
that cannot be created raises rather than degrading: a run that silently continued in the
user's checkout would be writing somewhere they did not agree to, and the reduced isolation
would appear nowhere. That is the failure invariant 6 exists to prevent, applied to
workspaces.
"""

from __future__ import annotations

import contextlib
import difflib
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Final, Literal

from chickadee.config import WorkspacePolicy
from chickadee.durable import write_new
from chickadee.errors import ConfigError, InfraError
from chickadee.graph.ignore import ALWAYS_SKIP
from chickadee.ids import operation_component
from chickadee.init.detect import RepoIdentity, resolve
from chickadee.init.lease import RepoLease, take_write_lease
from chickadee.paths import within
from chickadee.process import DIFF_FLAGS, GIT_TIMEOUT_S, Completed, git, run_git
from chickadee.schema.isolation import IsolationSpec, WorkspaceMode

WORKTREES_DIR = "chickadee-worktrees"
"""Worktrees live under the **git common directory**, not under `.chickadee/` in the
working tree. Nesting a checkout inside the repository it belongs to means every
tree-walking thing has to remember to skip it: the graph indexer, the packer, A3(a)'s
untracked-file bracket, and — found by `test_writes_in_the_worktree_do_not_touch_the
_source_checkout` — `snapshot()`'s own `git add -A`, which would otherwise fold a
previous run's entire worktree into the next run's in-place snapshot. Under the common
dir git reports none of it as untracked, and `git worktree list` still finds it."""
SNAPSHOT_REF_PREFIX = "refs/chickadee/snapshots"
TURNS_REF_PREFIX = "refs/chickadee/turns"
"""A turn's snapshot of each workspace it wrote in,
`refs/chickadee/turns/<session>/<turn>/<key>/<before|after>`."""
OPS_REF_PREFIX = "refs/chickadee/ops"
"""One ref per DECLARED EFFECT, `refs/chickadee/ops/<scope>/<operation_id>`, taken
before `effect_started` so the operation is an undo unit like a packet. `<scope>` is the
branch the operation ran on, so a handoff can retire exactly its own session's refs and
leave another session's alone; `keep` leaves them with the branch they explain."""
STEPS_REF_PREFIX = "refs/chickadee/steps"
"""A task's step chain: `<prefix>/<run>/<task>`, each step's tree a commit
on it, so `gc` keeps every one; swept like the turn refs (`sweep_snapshots`)."""
PINS_REF_PREFIX = "refs/chickadee/pins"
"""A stop's approved part of a work folder: `<prefix>/<scope>/<name>`, the
part's tree committed there so `gc` keeps it and a reader finds it by name (`design`,
`plan-v<n>`); what every later read of it is answered from."""
BRANCH_PREFIX = "chickadee"

# `git hash-object -t tree /dev/null` — the tree with no entries. Git defines it as a
# constant, so an empty repository still has a real object to name as its pre-image
# rather than an empty string the contract would reject.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

# `commit-tree` needs an author, and taking one from ambient config makes the snapshot
# depend on whether the user — or CI — happens to have `user.name` set. It does not: a
# machine with no identity configured fails with "Author identity unknown", which would
# mean the safety net silently does not exist exactly where it is least supervised.
# The identity is ours because the commit is ours; it is never attributed to the user.
SNAPSHOT_IDENTITY = (
    "-c",
    "user.name=chickadee",
    "-c",
    "user.email=chickadee@localhost",
)

_SLUG_MAX = 40


class _GitFailed(InfraError):
    """A git command the workspace depends on did not succeed.

    `InfraError` rather than a private class because `18-engine` §5 already assigns this
    exact situation — a required tool failing — to `InfraError`, and it is never the
    model's fault.
    """


def _run_git(
    root: Path,
    *args: str,
    extra_env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
) -> Completed:
    done = run_git(root, *args, extra_env=extra_env, stdin=stdin)
    if done.timed_out:
        raise InfraError(f"git {args[0]} timed out after {GIT_TIMEOUT_S}s in {root}")
    return done


def _text(done: Completed) -> str:
    return done.stdout.decode("utf-8", "replace").strip()


def _git(root: Path, *args: str, hooks: bool = False) -> str:
    return git(root, *args, on_error=_GitFailed, hooks=hooks)


def head_commit(root: Path) -> str | None:
    """The current commit, or None on an unborn HEAD (`git init` with no commit yet)."""
    done = _run_git(root, "rev-parse", "HEAD")
    return _text(done) if done.exit_code == 0 else None


def head_and_branch(repo: Path) -> tuple[str | None, str | None]:
    """Git's head and branch for D99's turn frame, through the supervisor. `None` for
    an unborn head, and for the branch when HEAD is detached. Moved out of the facade
    unchanged, which binds it for Zeus's loop."""
    head = run_git(repo, "rev-parse", "--verify", "-q", "HEAD")
    branch = run_git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    sha = head.stdout.decode("utf-8", "replace").strip()
    name = branch.stdout.decode("utf-8", "replace").strip()
    return (
        sha if head.exit_code == 0 and sha else None,
        name if branch.exit_code == 0 and name and name != "HEAD" else None,
    )


def run_branch_name(goal: str, run_id: str) -> str:
    """`chickadee/<goal-slug>-<run_id8>` — descriptive and PR-ready.

    The scheme comes from the 2026-08-13 run-branch decision. That decision was superseded
    on the question of *whether* to isolate, and D39 then restored isolation via worktrees;
    the naming argument it made was never what got overturned. A branch list full of
    `run-<hex>` tells the user nothing about which run did what, so the goal leads and the
    run id disambiguates.
    """
    return f"{BRANCH_PREFIX}/{slug(goal, 'run')}-{run_id[-8:]}"


def slug(text: str, fallback: str) -> str:
    """`text` as a name's lowercase words joined by `-`, at most `_SLUG_MAX` characters, or
    `fallback` when it has none: a run branch's (above) and a work folder's (`work.py`)."""
    out = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(out) > _SLUG_MAX:
        cut = out[:_SLUG_MAX]
        # Prefer a word boundary, but never return an empty slug for a long first word.
        out = cut.rsplit("-", 1)[0] if "-" in cut.strip("-") else cut
    return out.strip("-") or fallback


def _write_tree(
    source: Path, what: str, *, untracked_max_bytes: int | None = None
) -> tuple[str, tuple[str, ...]]:
    """The tree of the working tree as it is — staged, unstaged and untracked — written
    through a throwaway index so the user's own index is never touched, and the untracked
    paths it left out. The one mechanism the pre-run, operation and turn snapshots use.

    `untracked_max_bytes` is a turn snapshot's bound: an untracked file
    larger than it is dropped from the throwaway index and named. Tracked files are always
    kept, since git already holds them; the pre-run and operation snapshots pass no bound,
    because their job is to protect work the person already had.

    The throwaway index starts as a copy of the worktree's own (fix 1), so `add -A`
    re-reads only the files whose stat changed rather than hashing the whole tree: the same
    tree, sooner (`_seed_index`)."""
    oversized = _oversized_untracked(source, untracked_max_bytes)
    left_out = set(oversized)
    caches = tuple(n for n in _untracked_skipped(source) if n not in left_out)
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index"
        extra = {"GIT_INDEX_FILE": str(index)}
        _seed_index(source, index, extra, what)
        add = _run_git(source, "add", "-A", extra_env=extra)
        if add.exit_code != 0:
            raise _GitFailed(
                f"{what} could not stage the tree: "
                f"{add.stderr.decode('utf-8', 'replace').strip()}"
            )
        # By name, never as a pattern: `[id].js` as a pathspec also matches `i.js` and `d.js`.
        for start in range(0, len(oversized) + len(caches), _FLAG_BATCH):
            batch = (*oversized, *caches)[start : start + _FLAG_BATCH]
            literal = [f":(literal){name}" for name in batch]
            _git_in_index(source, extra, what, "rm", "--cached", "--quiet", "--", *literal)
        tree = _run_git(source, "write-tree", extra_env=extra)
        if tree.exit_code != 0:
            detail = tree.stderr.decode("utf-8", "replace").strip()
            raise _GitFailed(f"{what} could not write a tree: {detail}")
        return _text(tree), oversized


UNTRACKED_SKIP: frozenset[str] = frozenset(ALWAYS_SKIP - {".git", "dist", "build"})
"""Directories whose untracked files no snapshot holds: caches and environments a
command makes as it runs (`__pycache__` from `python -m py_compile`, `.venv`, `node_modules`),
never a task's work. The indexer's skip list less `dist` and `build`, which may hold real files;
and only untracked ones, so a tracked file there is snapshotted and committed as any other.
A restore never deletes under them either (`_sweep_extras`): the snapshot left them out, so it
cannot prove they were not there."""


def _untracked_skipped(source: Path) -> tuple[str, ...]:
    """The untracked files under an `UNTRACKED_SKIP` directory, each by name, so removing
    them never touches a tracked file beside them. Files the person's own ignore rules cover
    are not listed: `add -A` never stages them."""
    listed = _run_git(source, "ls-files", "--others", "--exclude-standard", "-z")
    if listed.exit_code != 0:
        detail = listed.stderr.decode("utf-8", "replace").strip()
        raise _GitFailed(f"listing the untracked files failed: {detail}")
    names = listed.stdout.decode("utf-8", "surrogateescape").split("\0")
    return tuple(n for n in names if n and UNTRACKED_SKIP & set(PurePosixPath(n).parts))


_FLAG_BATCH = 500
"""Paths per `update-index` call, so a repository with many flagged files stays under the
argument limit."""


def _own_index(source: Path) -> Path | None:
    """The worktree's own index file's path (a linked worktree has its own), or `None` when
    git cannot say."""
    found = _run_git(source, "rev-parse", "--git-path", "index")
    if found.exit_code != 0:
        return None
    own = Path(_text(found))
    return own if own.is_absolute() else source / own


def _seed_index(source: Path, index: Path, extra: Mapping[str, str], what: str) -> None:
    """`index` as a copy of the worktree's own index, with no flag that hides a change: every
    entry flagged assume-unchanged or skip-worktree whose file is on disk is unflagged, so
    `add -A` reads it as an empty index would. A skip-worktree entry whose file is absent (a
    sparse checkout) keeps its flag, and the tree keeps its index content rather than reading
    it as deleted. With no index yet (an unborn repository), the copy starts empty."""
    own = _own_index(source)
    if own is None or not own.is_file():
        return
    # `copy2`, keeping the index's mtime: git reads an entry no later than it as racy and
    # hashes its file again. Stamped now, a same-size edit in the index's second matched its
    # stat under `core.checkStat=minimal` and was not seen.
    shutil.copy2(own, index)
    listed = _run_git(source, "ls-files", "-v", "-z", extra_env=extra)
    if listed.exit_code != 0:
        detail = listed.stderr.decode("utf-8", "replace").strip()
        raise _GitFailed(f"{what} could not read its index's flags: {detail}")
    assumed: list[str] = []
    skipped: list[str] = []
    for entry in listed.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if len(entry) < 3:
            continue
        tag, path = entry[0], entry[2:]
        if not os.path.lexists(source / path):
            continue
        if tag.islower():
            assumed.append(path)
        if tag in ("S", "s"):
            skipped.append(path)
    # One flag per call: given both, git clears only assume-unchanged.
    for flag, paths in (("--no-assume-unchanged", assumed), ("--no-skip-worktree", skipped)):
        for start in range(0, len(paths), _FLAG_BATCH):
            batch = paths[start : start + _FLAG_BATCH]
            _git_in_index(source, extra, what, "update-index", flag, "--", *batch)


def _git_in_index(source: Path, extra: Mapping[str, str], what: str, *args: str) -> None:
    done = _run_git(source, *args, extra_env=extra)
    if done.exit_code != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise _GitFailed(f"{what} could not {args[0]}: {detail}")


def _oversized_untracked(source: Path, limit: int | None) -> tuple[str, ...]:
    """The untracked files over `limit`, sorted; untracked as the person's own index says,
    which is what makes them the person's and not git's. Ignored files are not listed: `add -A`
    never stages them anyway."""
    if limit is None:
        return ()
    listed = _run_git(source, "ls-files", "--others", "--exclude-standard", "-z")
    if listed.exit_code != 0:
        detail = listed.stderr.decode("utf-8", "replace").strip()
        raise _GitFailed(f"listing the untracked files failed: {detail}")
    names = [n for n in listed.stdout.decode("utf-8", "surrogateescape").split("\0") if n]
    over = [n for n in names if (source / n).is_file() and (source / n).stat().st_size > limit]
    return tuple(sorted(over))


def _ops_scope(root: Path) -> str:
    """The branch an operation ran on, or `detached` — the middle segment of its ref."""
    done = _run_git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    return _text(done) if done.exit_code == 0 and _text(done) else "detached"


def ops_ref(root: Path, operation_id: str, *, run_id: str | None = None) -> str:
    """The operation's ref. `run_id` keys it by the run as well: operation ids are unique
    within a run, not across runs, and a session's runs share one branch,
    so `<branch>/<operation_id>` alone would name two runs' snapshots alike. Still one ref
    component, `<run_id>/<operation_id>` percent-encoded, so `gc` groups it by branch as
    before. Without `run_id`, the name a ref had before runs shared a branch."""
    name = operation_id if run_id is None else f"{run_id}/{operation_id}"
    return f"{OPS_REF_PREFIX}/{_ops_scope(root)}/{operation_component(name)}"


def snapshot_operation(root: Path, operation_id: str, *, run_id: str | None = None) -> str:
    """The tree of `root` as it is, parked under `refs/chickadee/ops/<scope>/<id>`
    before a declared effect starts. Returns the tree sha, which `effect_intended` carries.

    The same `write-tree` as `snapshot()` — the user's index untouched, untracked files
    included, a ref so `gc` cannot collect it. A second call with the same id is refused:
    an operation snapshots once, before `effect_started`, and a repeat is a sequencing bug,
    not a refresh. `run_id`: the run the operation belongs to (`ops_ref`).
    """
    ref = ops_ref(root, operation_id, run_id=run_id)
    if _run_git(root, "show-ref", "--verify", "--quiet", ref).exit_code == 0:
        raise InfraError(
            f"operation {operation_id!r} already has a snapshot at {ref}; an operation "
            "snapshots once, before effect_started"
        )
    tree_sha, _ = _write_tree(root, f"operation {operation_id}")
    commit = _git(
        root,
        *SNAPSHOT_IDENTITY,
        "commit-tree",
        tree_sha,
        "-m",
        f"chickadee operation snapshot {operation_id}",
    )
    _git(root, "update-ref", ref, commit)
    return tree_sha


def restore_operation(
    root: Path, operation_id: str, paths: tuple[str, ...], *, run_id: str | None = None
) -> tuple[str, ...]:
    """Put the DECLARED paths back as the operation's snapshot had them: a path the
    tree holds is restored from it, a path it does not hold was created by the operation
    and is deleted. Returns the paths it touched. `run_id`: the run that journaled it, whose
    ref is looked for first; a ref taken before refs were keyed by run is found by its
    operation id alone (`ops_ref`).

    Bounded to the paths given — the declared surface — never the whole tree, because a
    restore that reached past the declaration would undo work the person did meanwhile.
    Through `git restore --source=<tree> --worktree`, so the user's index is untouched here
    too. A path outside the worktree is refused with the applier's containment rule.

    **A declared DIRECTORY is restored exactly, contents included.** `git restore` only ever
    puts back what the tree holds; it has no instruction to remove what the tree does not,
    so on its own it left every file the operation *created* inside a declared directory
    sitting there — a directory that is neither its before state nor its after, which is the
    half-undo D107 exists to prevent. The snapshot is a full `write-tree` of the worktree,
    untracked files included, so it already knows exactly which paths existed: `_sweep_extras`
    asks it, and deletes what it does not name. That the tree is complete is what makes this
    answerable rather than a guess.
    """
    refs = [ops_ref(root, operation_id, run_id=run_id)]
    if run_id is not None:
        refs.append(ops_ref(root, operation_id))
    ref = next((r for r in refs if ref_exists(root, r)), None)
    if ref is None:
        raise InfraError(f"operation {operation_id!r} has no snapshot at {refs[0]}")
    return restore_tree(root, _git(root, "rev-parse", f"{ref}^{{tree}}"), paths)


def restore_tree(root: Path, tree: str, paths: tuple[str, ...]) -> tuple[str, ...]:
    """**the one restore every undo goes through** — turn, packet and
    operation undo, redo, and a fork that restores the code. Each path the tree holds is put
    back from it; a path it lacks was created since and is deleted. Returns the paths it
    touched. Bounded to `paths`, through `git restore --worktree` so the person's index is
    untouched; a path outside the worktree, or the worktree root itself, is refused; a
    directory is restored exactly (`_sweep_extras`). D107's reasons for each rule are on
    `restore_operation`, whose restore this was."""
    real_root = Path(os.path.realpath(root))
    touched: list[str] = []
    for rel in paths:
        target = within(root, rel)
        if target is None:
            raise InfraError(
                f"{rel!r} resolves outside the worktree {root}; refusing to restore it"
            )
        if target == real_root:
            # Sweeping extras under the root would delete everything created since the
            # snapshot, the person's own work included — `/undo <operation>` becoming
            # `reset the repository`. The bound on the declared surface is the whole
            # safety argument, and a root declaration erases it.
            raise InfraError(
                f"{rel!r} is the worktree root; refusing to restore it. An operation "
                "declares the paths it writes, and undoing one is bounded to them — "
                "restoring the root would delete every file created since the snapshot"
            )
        in_tree = _run_git(root, "cat-file", "-e", f"{tree}:{rel}").exit_code == 0
        if in_tree:
            _git(root, "restore", f"--source={tree}", "--worktree", "--", rel)
            if target.is_dir():
                _sweep_extras(root, tree, rel, target)
            touched.append(rel)
        elif target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            touched.append(rel)
    return tuple(touched)


def _sweep_extras(root: Path, tree: str, rel: str, target: Path) -> None:
    """Delete everything under `target` that `tree` does not hold.

    The tree is the authority on what existed, so the question is answered by asking it
    rather than guessed from timestamps.

    **Files only.** A directory left empty by the sweep stays, because git cannot represent
    one and so the snapshot cannot tell a directory the operation created from a directory
    the person made and the operation happened to fill. Removing it would be deleting on a
    guess; leaving it costs an empty directory git itself does not track. The rule
    throughout is to delete only what the snapshot proves was not there.
    """
    listed = _git(root, "ls-tree", "-r", "-z", "--name-only", tree, "--", rel)
    kept = {entry for entry in listed.split("\0") if entry}
    base = PurePosixPath(Path(rel).as_posix())
    for path in target.rglob("*"):
        # Relative to the declared path rather than to `root`: `target` is a realpath, so a
        # repository reached through a symlink is not literally under `root` and
        # `relative_to(root)` would raise. `ls-tree` prints repo-relative paths, which is
        # what this rebuilds.
        below = path.relative_to(target)
        if ".git" in below.parts:
            continue  # never reach into a nested repository's own metadata
        if UNTRACKED_SKIP & set(below.parts) or UNTRACKED_SKIP & set(PurePosixPath(rel).parts):
            continue  # a snapshot leaves these out, so it cannot prove they were not there
        if path.is_file() and str(base / below.as_posix()) not in kept:
            path.unlink()


def snapshot(source: Path, run_id: str) -> tuple[str, str]:
    """Durably record the working tree *without disturbing it*.

    Returns `(pre_image_sha, dirty_snapshot_sha)`: the commit the run starts from, and a
    commit object holding everything that was already in the tree — staged, unstaged, and
    **untracked**.

    Three properties, each of which a simpler implementation loses:

    - **It does not touch the index or the working tree.** The write goes through a
      throwaway `GIT_INDEX_FILE`, so the user's staged changes are still staged afterwards.
      `git stash push` would have moved their work; `git stash create` leaves the tree
      alone but cannot see untracked files at all.
    - **It includes untracked files.** They are pre-existing changes too, and a run that
      creates a file where one already sat would otherwise destroy it with no way back.
    - **It survives `git gc`.** The snapshot gets a ref under `refs/chickadee/snapshots/`.
      An unreferenced commit is unreachable by definition, and unreachable objects are what
      `gc` collects — so a snapshot without a ref is a safety net with a timer on it.
    """
    head = head_commit(source)
    tree_sha, _ = _write_tree(source, "snapshot")

    parents = ["-p", head] if head else []
    commit = _git(
        source,
        *SNAPSHOT_IDENTITY,
        "commit-tree",
        tree_sha,
        *parents,
        "-m",
        f"chickadee pre-run snapshot for {run_id}",
    )
    _git(source, "update-ref", f"{SNAPSHOT_REF_PREFIX}/{run_id}", commit)
    return head or EMPTY_TREE, commit


# -- turn snapshots and their retention -------------------------------------------

DAY_S = 86_400

Phase = Literal["before", "after"]


@dataclass(frozen=True)
class SnapshotSite:
    """Where a turn snapshot is written from, and nothing that holds the lease: what a session
    line can record and a later turn can find again."""

    root: Path
    mode: WorkspaceMode
    key: str
    """The workspace's name within a turn: the run's id for a worktree, `in-place` for
    the person's checkout, so two in-place runs in one turn share a pair."""
    pre_run_tree: str | None = None
    """In place, the tree the D39 pre-run snapshot has just written: a turn's `before` records
    it rather than staging the checkout a second time."""


IN_PLACE_KEY = "in-place"


@dataclass(frozen=True)
class TurnSnapshot:
    tree: str
    ref: str
    head: str | None
    """The workspace's commit: the worktree branch's, or the checkout's `HEAD` in place. `None`
    on an unborn head."""
    skipped: tuple[str, ...]
    """Untracked paths over the bound, left out and named."""


def turn_ref(session_id: str, turn: int, key: str, phase: Phase, part: int = 0) -> str:
    """Every part a legal ref component, through the one encoder D107's refs use. A wake
    turn's `part` names its phase `<phase>-<part>`, under its turn, so the sweep keeps
    and prunes it with the turn."""
    named = phase if part == 0 else f"{phase}-{part}"
    return (
        f"{TURNS_REF_PREFIX}/{operation_component(session_id)}/{turn}/"
        f"{operation_component(key)}/{named}"
    )


def snapshot_turn(
    site: SnapshotSite,
    *,
    session_id: str,
    turn: int,
    phase: Phase,
    untracked_max_bytes: int,
    part: int = 0,
) -> TurnSnapshot:
    """D162 decision 2, D179. The tree of `site` parked under its turn ref, so `gc` cannot
    collect it. Once per ref: a second snapshot of one phase of one workspace in one turn's
    `part` is a sequencing bug, not a refresh."""
    ref = turn_ref(session_id, turn, site.key, phase, part)
    if _run_git(site.root, "show-ref", "--verify", "--quiet", ref).exit_code == 0:
        raise InfraError(f"turn {turn} already has a {phase} snapshot of {site.key} at {ref}")
    if phase == "before" and site.pre_run_tree is not None:
        tree, skipped = site.pre_run_tree, ()
    else:
        tree, skipped = _write_tree(
            site.root,
            f"turn {turn}'s {phase} snapshot",
            untracked_max_bytes=untracked_max_bytes,
        )
    message = f"chickadee turn snapshot {session_id} turn {turn} {site.key} {phase}"
    return _park(site.root, tree, skipped, ref, message)


def redo_ref(session_id: str, undo_seq: int, key: str) -> str:
    """Where an undo parks a workspace's tree before restoring it: under the
    session's turn refs, as `undo-<seq>`, which the sweep groups and orders like a turn."""
    return (
        f"{TURNS_REF_PREFIX}/{operation_component(session_id)}/undo-{undo_seq}/"
        f"{operation_component(key)}"
    )


def snapshot_for_redo(
    root: Path, *, session_id: str, undo_seq: int, key: str, untracked_max_bytes: int
) -> TurnSnapshot:
    """The tree of `root` as it stands before an undo restores anything, parked at its redo
    ref: what a redo puts back. With a turn snapshot's bound, since only the paths the harness
    wrote are ever restored from it, and those are never skipped."""
    ref = redo_ref(session_id, undo_seq, key)
    if _run_git(root, "show-ref", "--verify", "--quiet", ref).exit_code == 0:
        raise InfraError(f"undo {undo_seq} already has a redo snapshot of {key} at {ref}")
    tree, skipped = _write_tree(
        root, f"undo {undo_seq}'s redo snapshot", untracked_max_bytes=untracked_max_bytes
    )
    message = f"chickadee redo snapshot {session_id} undo {undo_seq} {key}"
    return _park(root, tree, skipped, ref, message)


def _park(
    root: Path, tree: str, skipped: tuple[str, ...], ref: str, message: str
) -> TurnSnapshot:
    """`tree`, committed on the workspace's head and named by `ref` so `gc` keeps it."""
    head = head_commit(root)
    parents = ["-p", head] if head else []
    commit = _git(root, *SNAPSHOT_IDENTITY, "commit-tree", tree, *parents, "-m", message)
    _git(root, "update-ref", ref, commit)
    return TurnSnapshot(tree, ref, head, skipped)


def step_ref(run_id: str, task: str) -> str:
    """A task's step chain, every part a legal ref component, through the one encoder."""
    return f"{STEPS_REF_PREFIX}/{operation_component(run_id)}/{operation_component(task)}"


def park_step(root: Path, run_id: str, task: str, tree: str, message: str) -> str:
    """`tree`, committed on the task's step chain and the chain moved to it, so `gc` keeps
    every step as it keeps a turn's: the chain's last step is the parent, the head the first's.
    Returns the commit."""
    ref = step_ref(run_id, task)
    found = _run_git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    parent = _text(found) if found.exit_code == 0 and _text(found) else head_commit(root)
    parents = ["-p", parent] if parent else []
    commit = _git(root, *SNAPSHOT_IDENTITY, "commit-tree", tree, *parents, "-m", message)
    _git(root, "update-ref", ref, commit)
    return commit


def pin_tree(root: Path, tree: str, folder: str, scope: str, name: str) -> None:
    """`tree`, the version of `folder` a stop approved (`folder_tree`, as the person saw it),
    parked as `name` under `scope`. Never re-read from the disk: an edit
    after the answer is not what was approved."""
    message = f"chickadee pin {folder}"
    commit = _git(root, *SNAPSHOT_IDENTITY, "commit-tree", tree, "-m", message)
    _git(root, "update-ref", _pin_ref(scope, name), commit)


def pin_of(root: Path, scope: str, name: str) -> str | None:
    """The tree pinned as `name` under `scope`, or None when none is."""
    done = _run_git(
        root, "rev-parse", "--verify", "--quiet", f"{_pin_ref(scope, name)}^{{tree}}"
    )
    return _text(done) if done.exit_code == 0 and _text(done) else None


def _pin_ref(scope: str, name: str) -> str:
    return f"{PINS_REF_PREFIX}/{operation_component(scope)}/{operation_component(name)}"


def folder_tree(root: Path, folder: str) -> str | None:
    """`folder`'s tree as it stands, untracked files included and unbounded, parked nowhere:
    whether a folder shown at a stop has changed since. None when absent.
    Through a throwaway index holding the folder alone, so a stop's every answer hashes the
    folder, never the whole repository and its untracked artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        extra = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        # A pathspec that matches nothing fails: the folder is absent.
        add = _run_git(root, "add", "-A", "--", f":(literal){folder}", extra_env=extra)
        if add.exit_code != 0:
            return None
        tree = _run_git(root, "write-tree", extra_env=extra)
        if tree.exit_code != 0:
            detail = tree.stderr.decode("utf-8", "replace").strip()
            raise _GitFailed(f"{folder} could not be written as a tree: {detail}")
        found = _run_git(root, "rev-parse", "--verify", "--quiet", f"{_text(tree)}:{folder}")
    return _text(found) if found.exit_code == 0 and _text(found) else None


def ignored(root: Path, path: str) -> bool:
    """Whether git ignores `path` in `root`, by the person's rules: no snapshot holds such a
    file (`_write_tree` stages with `add -A`), so no undo can bring it back."""
    return _run_git(root, "check-ignore", "-q", "--", path).exit_code == 0


def pinned_text(root: Path, tree: str, path: str) -> str | None:
    """The file `path` of the pinned `tree` (`pin_tree`), or None when it holds none."""
    done = _run_git(root, "cat-file", "blob", f"{tree}:{path}")
    return done.stdout.decode("utf-8") if done.exit_code == 0 else None


DIFF_SHOWN_LINES: Final = 80
"""How much of a part's diff a stop shows beside its README (`tree_diff`): the stat is whole."""


def tree_diff(root: Path, before: str, after: str, label: str) -> str | None:
    """What changed from the pinned tree `before` to `after`, as a stop shows it beside the
    part's README: which files changed, then the diff itself, cut at
    `DIFF_SHOWN_LINES` with the rest counted. None when nothing did. `label` names `before`."""
    if before == after:
        return None
    stat = _text(_run_git(root, "diff", "--stat", *DIFF_FLAGS, before, after))
    patch = _text(_run_git(root, "diff", *DIFF_FLAGS, before, after)).splitlines()
    shown = patch[:DIFF_SHOWN_LINES]
    more = len(patch) - len(shown)
    tail = [f"… {more} more line(s) of the diff; the files are the whole of it"] if more else []
    return "\n".join([f"Changed since {label}:", stat, "", *shown, *tail])


def text_diff(before: str, after: str, label: str) -> str | None:
    """`tree_diff`'s for two texts (a one-task plan's brief, amended): the
    diff, cut at `DIFF_SHOWN_LINES` with the rest counted. None when nothing changed."""
    if before == after:
        return None
    patch = list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), "before", "after", lineterm=""
        )
    )
    shown = patch[:DIFF_SHOWN_LINES]
    more = len(patch) - len(shown)
    tail = [f"… {more} more line(s) of the diff"] if more else []
    return "\n".join([f"Changed since {label}:", *shown, *tail])


def pinned_paths(root: Path, tree: str) -> tuple[str, ...]:
    """Every file of the pinned `tree`, by its path from it."""
    return tuple(_text(_run_git(root, "ls-tree", "-r", "--name-only", tree)).splitlines())


def current_tree(root: Path, *, untracked_max_bytes: int | None) -> tuple[str, tuple[str, ...]]:
    """The tree of `root` as it stands, through a throwaway index, with a turn snapshot's
    bound, and the untracked paths it left out: what a redo compares against its target.
    `None` is no bound, as an operation snapshot has: what a declared write is compared
    against, so a large file it created is seen and put back."""
    return _write_tree(root, "the current tree", untracked_max_bytes=untracked_max_bytes)


def turn_refs(root: Path, session_id: str) -> frozenset[str]:
    """Every turn and redo snapshot ref the session still has, in one `for-each-ref`: what a
    reader checking many snapshots asks once, rather than once per ref."""
    prefix = f"{TURNS_REF_PREFIX}/{operation_component(session_id)}/"
    return frozenset(ref for ref, _ in _refs_with_dates(root, prefix))


def ref_exists(root: Path, ref: str) -> bool:
    return _run_git(root, "show-ref", "--verify", "--quiet", ref).exit_code == 0


def commit_of(root: Path, ref: str) -> str:
    return _git(root, "rev-parse", f"{ref}^{{commit}}")


def tree_of(root: Path, ref: str) -> str:
    return _git(root, "rev-parse", f"{ref}^{{tree}}")


def ref_tips(root: Path) -> tuple[str, ...]:
    """What the repository's history is before a command runs: `HEAD` and every branch, tag
    and remote-tracking ref. Never the harness's own refs, whose snapshots hold whatever the
    tree held, a model's writes included. What `history_head` is asked against."""
    # A tag is peeled to what it names; one naming a tree or a blob is no commit to walk from.
    shape = (
        "%(if)%(*objectname)%(then)%(*objecttype) %(*objectname)"
        "%(else)%(objecttype) %(objectname)%(end)"
    )
    out = _git(
        root, "for-each-ref", f"--format={shape}", "refs/heads", "refs/tags", "refs/remotes"
    )
    commits = {
        sha for kind, sha in (line.split() for line in out.splitlines()) if kind == "commit"
    }
    head = head_commit(root)
    return tuple(sorted({*commits, *((head,) if head else ())}))


def history_head(root: Path, before: Sequence[str], started: float) -> str | None:
    """The commit `HEAD` is on, when git's history accounts for it: reachable from `before`
    (`ref_tips`, taken before the command), or fetched, every commit it adds reachable from
    a remote-tracking ref and committed before `started`. `None` when the command made one
    of those commits itself (a `git commit`, a merge, a rebase), and on an unborn `HEAD`.

    The date is what tells a fetched commit from one the command made and then fetched into a
    remote-tracking ref of its own (`git fetch. HEAD:refs/remotes/x`); a commit in the
    second `started` began counts as made, the cautious answer."""
    head = head_commit(root)
    if head is None:
        return None
    unreachable = "".join(f"^{tip}\n" for tip in before).encode()
    added = _run_git(root, "rev-list", "--timestamp", "HEAD", "--stdin", stdin=unreachable)
    if added.exit_code != 0:
        return None
    stamps = [int(line.split()[0]) for line in _text(added).splitlines() if line.strip()]
    if not stamps:
        return head
    if max(stamps) >= int(started):
        return None
    unfetched = _run_git(
        root, "rev-list", "-n1", "HEAD", "--not", "--remotes", "--stdin", stdin=unreachable
    )
    return head if unfetched.exit_code == 0 and not _text(unfetched) else None


def index_bytes(root: Path) -> bytes | None:
    """git's index file as it stands, or `None` when there is none: what a
    command's index changes are measured against, held whole so the entries before can be read
    only when it changed."""
    own = _own_index(root)
    try:
        return own.read_bytes() if own is not None else None
    except FileNotFoundError:
        return None


IndexEntries = Mapping[str, tuple[str, ...]]
"""A path's index entries, each `<mode> <object> <stage>` as `ls-files -s` prints it: one at
stage 0, or the stages of a conflict."""


def index_entries(root: Path, held: bytes | None = None, *, now: bool = False) -> IndexEntries:
    """Every path's index entries: of the index as it stands when `now`, else of `held`, an
    index file's bytes (`index_bytes`), read through a throwaway copy; `{}` for no index."""
    if now:
        return _entries(_run_git(root, "ls-files", "-s", "-z"))
    if held is None:
        return {}
    with tempfile.TemporaryDirectory(prefix="chickadee-index-") as tmp:
        copy = Path(tmp) / "index"
        write_new(copy, held)
        done = _run_git(root, "ls-files", "-s", "-z", extra_env={"GIT_INDEX_FILE": str(copy)})
    return _entries(done)


def _entries(done: Completed) -> IndexEntries:
    if done.exit_code != 0:
        detail = done.stderr.decode("utf-8", "replace").strip() or f"exit {done.exit_code}"
        raise InfraError(f"git ls-files -s failed: {detail}")
    entries: dict[str, list[str]] = {}
    for record in done.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if record:
            entry, _, path = record.partition("\t")
            entries.setdefault(path, []).append(entry)
    return {path: tuple(found) for path, found in entries.items()}


def commit_entries(root: Path, commit: str, paths: Sequence[str]) -> IndexEntries:
    """What `commit` holds at `paths`, as the stage-0 index entry it would check out; a path
    it lacks is absent."""
    if not paths:
        return {}
    out = _run_git(root, "ls-tree", "-z", commit, "--", *paths)
    entries: dict[str, tuple[str, ...]] = {}
    for record in out.stdout.decode("utf-8", "surrogateescape").split("\0"):
        if record:
            meta, _, path = record.partition("\t")
            mode, _, sha = meta.split()
            entries[path] = (f"{mode} {sha} 0",)
    return entries


def restore_index(root: Path, before: IndexEntries, paths: Sequence[str]) -> None:
    """Put `paths`' index entries back to `before`'s, conflict stages included, a path absent
    from it removed, touching no file. Each path is first removed (a mode-0
    entry), as `update-index --index-info` needs before it takes stages."""
    null = "0" * (64 if _git(root, "rev-parse", "--show-object-format") == "sha256" else 40)
    records: list[str] = []
    for path in paths:
        records.append(f"0 {null}\t{path}")
        records.extend(f"{entry}\t{path}" for entry in before.get(path, ()))
    feed = "".join(f"{r}\0" for r in records).encode("utf-8", "surrogateescape")
    done = _run_git(root, "update-index", "-z", "--index-info", stdin=feed)
    if done.exit_code != 0:
        detail = done.stderr.decode("utf-8", "replace").strip() or f"exit {done.exit_code}"
        raise InfraError(f"git update-index --index-info failed in {root}: {detail}")


def move_branch(root: Path, commit: str) -> None:
    """Move the worktree's branch, and its index with it, to `commit`, touching no file (a
    mixed reset). Only for a run's own worktree branch, which is the harness's: an undo
    before the handoff moves it back, and its redo forward, and `restore_tree` then decides
    what the files hold."""
    _git(root, "reset", "--quiet", "--mixed", commit)


def commit_paths(
    root: Path, paths: Sequence[str], message: str, *, own_unit: bool = False
) -> str:
    """Commit exactly `paths` as the working tree holds them, whatever else is staged, with the
    harness's identity (as a squash handoff commits). Returns the commit. The commit an undo
    makes on the person's branch, under `undo.on_your_branch = "commit"`,
    and a declared Zeus write's in the session's worktree.

    `own_unit`: the commit is the harness's rollback unit in its own worktree, as a task's is
    (`box/commit.Projector.commit`): unsigned, and past the person's hooks, which were written
    for their commits."""
    tracked = {p for p in _git(root, "ls-files", "-z", "--", *paths).split("\0") if p}
    # A path neither on disk nor tracked was never committed and is gone: nothing to record,
    # and git refuses a pathspec that matches nothing.
    present = [p for p in paths if p in tracked or (root / p).exists()]
    if not present:
        raise InfraError("an undo commit needs at least one path git can see")
    _git(root, "add", "--all", "--", *present)
    own = ("-c", "commit.gpgsign=false") if own_unit else ()
    skip = ("--no-verify",) if own_unit else ()
    # The person's hooks run on a commit made for them on their branch, never on the
    # harness's own unit.
    _git(
        root,
        *SNAPSHOT_IDENTITY,
        *own,
        "commit",
        "--quiet",
        *skip,
        "--only",
        "-m",
        message,
        "--",
        *present,
        hooks=not own_unit,
    )
    return _git(root, "rev-parse", "HEAD")


def unmerged_paths(root: Path) -> tuple[str, ...]:
    """The paths git holds as unmerged in `root`'s index, sorted: a merge, rebase, cherry-pick
    or revert that stopped on conflicts. Empty when git cannot say."""
    done = _run_git(root, "diff", "--name-only", "--diff-filter=U", "-z")
    if done.exit_code != 0 or done.timed_out:
        return ()
    out = done.stdout.decode("utf-8", "surrogateescape")
    return tuple(sorted({p for p in out.split("\0") if p}))


def dirty_paths(root: Path) -> tuple[str, ...]:
    """Every path `git status` lists as changed against `HEAD`, untracked included, sorted: what
    a commit of the working tree would take. A rename lists both its names."""
    # Raw, never stripped: the first entry's status column can begin with a space. No optional
    # lock: a status refreshes the index's stat cache when it may, and a reader that must not
    # write (the run header) asks it the same question. Skipping the refresh
    # never changes the answer.
    done = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        extra_env={"GIT_OPTIONAL_LOCKS": "0"},
    )
    if done.exit_code != 0 or done.timed_out:
        raise _GitFailed(f"git status failed in {root}")
    out = done.stdout.decode("utf-8", "surrogateescape")
    names: set[str] = set()
    entries = iter(out.split("\0"))
    for entry in entries:
        if not entry:
            continue
        names.add(entry[3:])
        if entry[0] in "RC":
            names.add(next(entries, ""))
    names.discard("")
    return tuple(sorted(names))


def sweep_snapshots(
    root: Path, session_id: str, *, keep_days: int, keep_turns: int, now: float
) -> tuple[str, ...]:
    """**the one place a snapshot ref is deleted.** A group of refs goes only
    when it is both older than `keep_days` and outside the newest `keep_turns` groups. Returns
    the refs it deleted.

    - The session's turn refs group by their turn component: a turn's seq, or a later
      `undo-<seq>` redo snapshot, ordered by that number.
    - D107's operation refs follow the same rule per scope (the branch they ran on), each ref
      its own group: they are not a session's, so every scope is swept.
    - A task's step chains group by their run, which is no session's either, so
      every run's are swept, ordered by age.
    - The pre-run snapshots under `refs/chickadee/snapshots/` are not touched: `reattach`
      needs them, and D162 gives them no rule."""
    cutoff = now - keep_days * DAY_S
    groups: dict[tuple[str, str], list[tuple[str, int]]] = {}
    prefix = f"{TURNS_REF_PREFIX}/{operation_component(session_id)}/"
    for ref, at in _refs_with_dates(root, prefix):
        groups.setdefault(("turns", ref[len(prefix) :].split("/", 1)[0]), []).append((ref, at))
    for ref, at in _refs_with_dates(root, f"{OPS_REF_PREFIX}/"):
        groups.setdefault((ref.rsplit("/", 1)[0], ref), []).append((ref, at))
    for ref, at in _refs_with_dates(root, f"{STEPS_REF_PREFIX}/"):
        run = ref[len(STEPS_REF_PREFIX) + 1 :].split("/", 1)[0]
        groups.setdefault(("steps", run), []).append((ref, at))
    deleted: list[str] = []
    for scope in {s for s, _ in groups}:
        mine = {g: refs for g, refs in groups.items() if g[0] == scope}
        order = sorted(mine, key=lambda g: _position(g[1], mine[g]), reverse=True)
        for g in order[keep_turns:]:
            if max(at for _, at in mine[g]) < cutoff:
                for ref, _ in mine[g]:
                    _git(root, "update-ref", "-d", ref)
                    deleted.append(ref)
    return tuple(sorted(deleted))


def _position(name: str, refs: list[tuple[str, int]]) -> tuple[int, int]:
    """A group's place in its scope, newest last: a turn's seq or an undo's, else its age."""
    number = name.removeprefix("undo-")
    newest = max(at for _, at in refs)
    return (int(number), newest) if number.isdigit() else (newest, newest)


def _refs_with_dates(root: Path, prefix: str) -> list[tuple[str, int]]:
    out = _git(root, "for-each-ref", "--format=%(refname) %(committerdate:unix)", prefix)
    pairs: list[tuple[str, int]] = []
    for line in out.splitlines():
        ref, _, at = line.strip().rpartition(" ")
        if ref:
            pairs.append((ref, int(at)))
    return pairs


# -- what an undo would do ------------------------------------------


def changed_paths(root: Path, a: str, b: str) -> tuple[str, ...]:
    """**The one tree diff**: the paths that differ between trees `a` and
    `b`, sorted. No rename detection, so a rename is its old path and its new one, both of
    which an undo puts back."""
    out = _diff_tree(root, a, b, "--name-only")
    return tuple(sorted({p for p in out.split("\0") if p}))


DELETED: Final = "0" * 40
"""`tree_change`'s blob for a path tree `b` does not hold."""


def tree_change(root: Path, a: str, b: str) -> tuple[tuple[str, str], ...]:
    """`changed_paths`' diff with each path's blob in `b` (`DELETED` for one it lacks), sorted
    by path: a state the progress watchdog compares by equality, so a file
    written back to what it was reads as the state it was."""
    if a == b:
        return ()
    parts = [p for p in _diff_tree(root, a, b, "--raw").split("\0") if p]
    # `--raw -z` gives `:<mode> <mode> <blob> <blob> <status>` then the path, as two fields.
    found = {parts[i + 1]: parts[i].split()[3] for i in range(0, len(parts) - 1, 2)}
    return tuple(sorted(found.items()))


def _diff_tree(root: Path, a: str, b: str, form: str) -> str:
    """The one `git diff-tree` under chickadee: no renames, so a rename is its old path and
    its new one; `form` is the output git gives (`--name-only`, `--name-status`,
    `--numstat`, `--raw`)."""
    return _git(root, "diff-tree", "-r", "--no-renames", form, "-z", a, b)


@dataclass(frozen=True)
class PathDiff:
    """One path's change between two trees, as `diff_stats` counts it."""

    path: str
    kind: Literal["created", "edited", "deleted"]
    added: int
    removed: int
    """Lines; 0 and 0 for a binary file, which git does not count."""


def diff_stats(root: Path, a: str, b: str) -> tuple[PathDiff, ...]:
    """`changed_paths`' diff, with each path's kind and line counts (the task facts): the same
    diff, no renames, sorted by path."""
    status = _diff_tree(root, a, b, "--name-status")
    parts = [p for p in status.split("\0") if p]
    kinds = {parts[i + 1]: parts[i] for i in range(0, len(parts) - 1, 2)}
    counts: dict[str, tuple[int, int]] = {}
    numstat = _diff_tree(root, a, b, "--numstat")
    for row in numstat.split("\0"):
        fields = row.split("\t")
        if len(fields) == 3:
            added, removed, path = fields
            counts[path] = (
                int(added) if added.isdigit() else 0,
                int(removed) if removed.isdigit() else 0,
            )
    word: dict[str, Literal["created", "edited", "deleted"]] = {"A": "created", "D": "deleted"}
    return tuple(
        PathDiff(path, word.get(kinds[path], "edited"), *counts.get(path, (0, 0)))
        for path in sorted(kinds)
    )


@dataclass(frozen=True)
class TurnTrees:
    """One turn's snapshot pair of one workspace, as an undo reads it."""

    turn: int
    before: str
    after: str
    skipped: tuple[str, ...] = ()
    """Untracked paths either snapshot left out: unknown to it, not absent."""


def harness_paths(root: Path, turns: Sequence[TurnTrees]) -> tuple[str, ...]:
    """The paths the harness wrote in `turns`: each turn's `before` to its `after`, less every
    path a snapshot skipped, which neither snapshot can speak for."""
    unknown = {p for t in turns for p in t.skipped}
    wrote = {p for t in turns for p in changed_paths(root, t.before, t.after)}
    return tuple(sorted(wrote - unknown))


@dataclass(frozen=True)
class UndoPlan:
    """What taking back `turns` would do to one workspace, before anything is written."""

    target: str
    """The tree to return to: the first turn's `before`."""
    restore: tuple[str, ...]
    """The paths the harness wrote: the only paths an undo touches."""
    person: tuple[str, ...]
    """The paths the person edited between those turns or since the last one."""
    conflicts: tuple[str, ...]
    """What the harness wrote that the person edited too, or that the current tree cannot
    read: never overwritten without the person deciding."""
    skipped: tuple[str, ...]
    """Every untracked path a snapshot left out, so an undo can say what it cannot protect."""


def plan_undo(root: Path, turns: Sequence[TurnTrees], *, untracked_max_bytes: int) -> UndoPlan:
    """D162 decision 3, per workspace. `turns` are one workspace's snapshot pairs from
    turn N onward, oldest first. Writes nothing to the workspace: the tree it reads now goes
    through a throwaway index, with the bound turn snapshots use, so a large untracked file is
    left out on both sides rather than read as an edit."""
    if not turns:
        raise InfraError("an undo plan needs at least one snapshotted turn")
    current, now_skipped = _write_tree(
        root, "the undo plan's current tree", untracked_max_bytes=untracked_max_bytes
    )
    unknown = {p for t in turns for p in t.skipped} | set(now_skipped)
    restore = harness_paths(root, turns)
    edited: set[str] = set()
    for done, nxt in pairwise(turns):
        edited.update(changed_paths(root, done.after, nxt.before))
    edited.update(changed_paths(root, turns[-1].after, current))
    person = tuple(sorted(edited - unknown))
    # A path the harness wrote whose state now cannot be read (it grew past the bound since)
    # may hold the person's edit: it is asked about, never overwritten on a guess.
    unreadable_now = set(now_skipped) & set(restore)
    return UndoPlan(
        target=turns[0].before,
        restore=restore,
        person=person,
        conflicts=tuple(sorted(set(restore) & (set(person) | unreadable_now))),
        skipped=tuple(sorted(unknown)),
    )


@dataclass(frozen=True)
class Workspace:
    """Where this run's writes land, and what isolation it actually got."""

    spec: IsolationSpec
    root: Path
    """The directory packets write into. The worktree in `worktree` mode, the user's own
    checkout in `in_place`."""
    source_root: Path
    identity: RepoIdentity
    degraded: tuple[str, ...] = ()
    """Plain sentences naming every way this run is less isolated than the default. Empty
    for `worktree`. Read by the run header and the handoff — never inferred from `mode`,
    because a reader that has to derive the warning is a reader that can forget to."""
    _lease: RepoLease | None = field(default=None, repr=False)

    @property
    def isolated(self) -> bool:
        return self.spec.mode == "worktree"

    def snapshot_site(self, *, fresh: bool) -> SnapshotSite:
        """Where a turn snapshot of this workspace is written from. `fresh`: acquired just now,
        so an in-place workspace's pre-run snapshot is the checkout as it stands and a turn's
        `before` can reuse it; a reattached one's is older than that, and is not offered."""
        if self.spec.mode == "worktree":
            key = Path(self.spec.worktree_path or self.root).name
            return SnapshotSite(self.root, "worktree", key)
        dirty = self.spec.dirty_snapshot_sha
        tree = _git(self.root, "rev-parse", f"{dirty}^{{tree}}") if fresh and dirty else None
        return SnapshotSite(self.root, "in_place", IN_PLACE_KEY, tree)

    def release(self) -> None:
        """Release the lease. The worktree itself outlives this — what happens to it is the
        user's choice at handoff (apply/squash/keep/discard), not ours."""
        if self._lease is not None:
            self._lease.release()

    def __enter__(self) -> Workspace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def acquire(
    source: Path,
    *,
    run_id: str,
    goal: str,
    policy: WorkspacePolicy | None = None,
    start: str | None = None,
    session: str | None = None,
) -> Workspace:
    """Take the repository WRITE lease and prepare the run's workspace.

    `policy`: where the run writes. The person's configured mode is in place by default, and
    turn snapshots give a run there its undo point; the
    caller passes it. With no policy, a worktree: what a host's own `CoderRequest` asks for and
    what a fork's child needs, since neither is a person's conversation.

    `start`: the commit a worktree's branch begins at, instead of the source's `HEAD`. A fork
    makes its child's worktree this way, from a turn snapshot: the one
    worktree constructor, never a second. In place has no branch to start, so it is refused.

    A repository with no commits gets an empty first commit to branch from (`_first_commit`),
    which a seed (below) then builds on.

    `session`: the session that will own the worktree, which names it
    instead of the run: `<session>`, or `<session>-<n>` for the session's next worktree after
    a handoff left the first in place. A checkout with uncommitted changes is seeded
    (`_seed_commit`): the worktree starts with them, read under
    the lease so nothing writes the checkout between the read and the worktree.
    """
    policy = policy or WorkspacePolicy(mode="worktree")
    identity = resolve(source)

    if not policy.require_lease:
        # Contract 11 validates `lease_id` as non-empty for *every* mode, and D39 says the
        # lease is "what makes isolation real rather than advisory". There is therefore no
        # lease-free workspace this function could construct and still return a valid
        # IsolationSpec. Refused loudly rather than quietly ignored — see the task log.
        raise ConfigError(
            "workspace.require_lease=false has no implementation: contract 11 requires a "
            "lease_id in every isolation mode and D39 requires the lease itself. Remove "
            "the override, or raise an Issue to change the contract."
        )

    lease = take_write_lease(identity)
    try:
        if policy.mode == "in_place":
            if start is not None or session is not None:
                raise ConfigError("an in-place run has no branch of its own to start elsewhere")
            return _in_place(source, identity, lease, run_id=run_id)
        name = run_id
        if session is not None:
            name = _free_name(identity, session, goal)
            if start is None:
                start = _seed_commit(identity.root, session)
        return _worktree(source, identity, lease, run_id=name, goal=goal, start=start)
    except BaseException:
        lease.release()  # never leave the repository locked by a setup that failed
        raise


def _lease_id(identity: RepoIdentity) -> str:
    return f"{identity.repository_id}:{os.getpid()}"


SEED_TRAILER: Final = "Chickadee-Seed"
"""The trailer on a worktree's first commit when it holds the checkout's uncommitted changes;
its value is the session. `handoff.seed_of` finds the commit by
it, so a landing writes only what came after."""

SEED_MESSAGE: Final = (
    "chickadee: your uncommitted changes, carried into the worktree\n\n"
    "The checkout's work as it was when the worktree was made, untracked files included, so\n"
    "the work here starts where yours was. A landing writes only what came after this commit."
)


def _seed_commit(root: Path, session: str) -> str | None:
    """The checkout's uncommitted changes as one commit on `HEAD`, for a session's worktree to
    start from; None for a clean checkout. The tree is read as a
    turn snapshot reads it (`_write_tree`: the person's index untouched, untracked files
    included, ignored ones and caches left out), so a tree equal to `HEAD`'s is clean too. A
    repository with no commits gets its empty first commit first, to be the seed's parent."""
    if not dirty_paths(root):
        return None
    if head_commit(root) is None:
        _first_commit(root)
    tree, _ = _write_tree(root, "the worktree's seed")
    if tree == _git(root, "rev-parse", "HEAD^{tree}"):
        return None
    message = f"{SEED_MESSAGE}\n\n{SEED_TRAILER}: {session}"
    return _git(root, *SNAPSHOT_IDENTITY, "commit-tree", tree, "-p", "HEAD", "-m", message)


def chickadee_worktrees(source: Path) -> tuple[Path, ...]:
    """Every worktree the harness made for `source`'s repository, any session's: where a work
    folder another session opened lives until it lands."""
    folder = resolve(source).git_common_dir / WORKTREES_DIR
    return tuple(sorted(p for p in folder.iterdir() if p.is_dir())) if folder.is_dir() else ()


def _free_name(identity: RepoIdentity, session: str, goal: str) -> str:
    """The session's first worktree is `<session>`; a later one, after a handoff left the
    earlier in place, takes the next `-<n>` whose directory and branch are
    both free. Never an existing path: `_worktree` refuses one, since it may hold work."""
    n = 1
    while True:
        name = session if n == 1 else f"{session}-{n}"
        taken_path = (identity.git_common_dir / WORKTREES_DIR / name).exists()
        branch = run_branch_name(goal, name)
        found = _run_git(
            identity.root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"
        )
        if not taken_path and found.exit_code != 0:
            return name
        n += 1


def _in_place(
    source: Path,
    identity: RepoIdentity,
    lease: RepoLease,
    *,
    run_id: str,
) -> Workspace:
    pre_image, dirty = snapshot(source, run_id)
    return _in_place_workspace(identity, lease, run_id=run_id, pre_image=pre_image, dirty=dirty)


def _in_place_workspace(
    identity: RepoIdentity, lease: RepoLease, *, run_id: str, pre_image: str, dirty: str
) -> Workspace:
    return Workspace(
        spec=IsolationSpec(
            mode="in_place",
            lease_id=_lease_id(identity),
            pre_image_sha=pre_image,
            dirty_snapshot_sha=dirty,
        ),
        root=identity.root,
        source_root=identity.root,
        identity=identity,
        degraded=(
            "in-place mode: this run writes directly into your checkout, so packet "
            "commits and rollbacks touch the branch you are on.",
            f"pre-existing changes are snapshotted at {SNAPSHOT_REF_PREFIX}/{run_id} "
            f"({dirty[:12]}) and can be restored with `git checkout {dirty[:12]} -- .`",
        ),
        _lease=lease,
    )


FIRST_COMMIT_MESSAGE: Final = (
    "chickadee: an empty first commit, so the work has a base to branch from"
)


def _first_commit(root: Path) -> str:
    """An empty root commit on the unborn branch HEAD names, so a repository with no commits
    has a base for the worktree. The empty tree through `commit-tree`, under
    the harness's identity because the commit is ours, then `update-ref` expecting no commit
    yet, so a commit made meanwhile is never overwritten; neither the index nor a file moves.
    Its message says why it is there. Returns the commit."""
    # The empty tree from an empty file, so no stdin and no hash written in for SHA-1 alone.
    empty = _git(root, "hash-object", "-t", "tree", "-w", os.devnull)
    commit = _git(root, *SNAPSHOT_IDENTITY, "commit-tree", empty, "-m", FIRST_COMMIT_MESSAGE)
    # An empty old value: the branch must still have no commit, or nothing is written.
    _git(root, "update-ref", "HEAD", commit, "")
    return commit


def _worktree(
    source: Path,
    identity: RepoIdentity,
    lease: RepoLease,
    *,
    run_id: str,
    goal: str,
    start: str | None = None,
) -> Workspace:
    if head_commit(identity.root) is None and start is None:
        _first_commit(identity.root)
    branch = run_branch_name(goal, run_id)
    path = identity.git_common_dir / WORKTREES_DIR / run_id
    path.parent.mkdir(parents=True, exist_ok=True)

    # Refuse a path that is already there rather than write into it or, worse, clean it up
    # on failure. A directory at this location is a previous run's worktree, and it may hold
    # work the user has not saved anywhere else — `git worktree add` fails on a non-empty
    # path, so the unconditional cleanup below would have deleted data this run never
    # created. Naming the collision is both safer and more useful than either.
    if path.exists():
        raise InfraError(
            f"a worktree for run {run_id} already exists at {path} — remove it with "
            f"`git worktree remove {path}` once you no longer need what is in it, or use "
            "a different run id."
        )
    try:
        _git(identity.root, "worktree", "add", "-b", branch, str(path), start or "HEAD")
    except InfraError:
        # Deliberately no fallback to in_place. See the module docstring: degrading here
        # would move the user's run into their own checkout without their say-so.
        # Safe to remove unconditionally now: the guard above proves this invocation is
        # the only thing that could have created it.
        shutil.rmtree(path, ignore_errors=True)
        # `worktree add` can register administrative data under the common dir before it
        # fails. Removing only the directory leaves that entry behind, so `git worktree
        # list` reports a broken worktree and the next run reusing this id fails with
        # "already registered" instead of whatever actually went wrong. Errors from the
        # prune are swallowed so it cannot mask the original failure.
        with contextlib.suppress(OSError, InfraError):
            run_git(identity.root, "worktree", "prune")
        raise
    return _worktree_workspace(identity, lease, path=path, branch=branch)


def _worktree_workspace(
    identity: RepoIdentity, lease: RepoLease, *, path: Path, branch: str
) -> Workspace:
    return Workspace(
        spec=IsolationSpec(
            mode="worktree",
            lease_id=_lease_id(identity),
            branch=branch,
            worktree_path=str(path),
        ),
        root=path,
        source_root=identity.root,
        identity=identity,
        _lease=lease,
    )


class WorktreeGone(InfraError):
    """A session's recorded worktree is gone, and so is its branch: nothing of it is left to
    reattach, so its next call opens a new one."""


def reattach_worktree(source: Path, worktree: Path, *, branch: str | None = None) -> Workspace:
    """Take the repository WRITE lease again over a session's recorded worktree, for its next
    `coder` call (the session owns the worktree, a call only the lease).
    Found by the path the session's `workspace` line names, so a fork's child worktree is
    found as its own session's is; `reattach` finds a run's by its id.

    A directory that is gone while its `branch` still exists is added back at the same path
    from that branch: the branch holds the session's work, which a new worktree from `HEAD`
    would leave behind without a word. `WorktreeGone` when the branch is gone too."""
    identity = resolve(source)
    lease = take_write_lease(identity)
    try:
        if not worktree.is_dir():
            if branch is None or not ref_exists(identity.root, f"refs/heads/{branch}"):
                raise WorktreeGone(f"the session's worktree at {worktree} is gone")
            # A directory deleted by hand leaves git's registration behind, and `worktree
            # add` refuses a registered path; prune drops only entries whose directory is gone.
            _git(identity.root, "worktree", "prune")
            _git(identity.root, "worktree", "add", str(worktree), branch)
        return _found_worktree(identity, lease, worktree)
    except BaseException:
        lease.release()
        raise


def abandon_worktree(source: Path, workspace: Workspace) -> None:
    """Remove a worktree and its branch that this call created and nothing recorded, so a
    failure between `acquire` and the session's `workspace` line leaves no orphan. Only for
    a worktree just created: it holds no commit of its own yet. Best effort, so it never
    hides the failure that called it."""
    path, branch = workspace.spec.worktree_path, workspace.spec.branch
    if workspace.spec.mode != "worktree" or path is None:
        return
    root = resolve(source).root
    with contextlib.suppress(InfraError, OSError):
        _git(root, "worktree", "remove", "--force", path)
    with contextlib.suppress(InfraError, OSError):
        if branch:
            _git(root, "branch", "-D", branch)


def is_ancestor(root: Path, older: str, newer: str) -> bool:
    """Whether `older` is `newer` or one of its ancestors. `InfraError` when git cannot say."""
    done = _run_git(root, "merge-base", "--is-ancestor", older, newer)
    if done.exit_code in (0, 1) and not done.timed_out:
        return done.exit_code == 0
    raise _GitFailed(f"git merge-base --is-ancestor {older} {newer} failed")


def _found_worktree(identity: RepoIdentity, lease: RepoLease, path: Path) -> Workspace:
    branch = _git(path, "branch", "--show-current")
    return _worktree_workspace(identity, lease, path=path, branch=branch)


def reattach(source: Path, *, run_id: str, worktree: Path | None = None) -> Workspace:
    """Take the repository WRITE lease again for a run that already has its workspace, to
    continue it. Nothing is created: a worktree run finds its worktree and branch, an
    in-place run finds the snapshot of the person's work that `acquire` took. The mode
    is read from what exists, never assumed; a run with neither has nothing to continue.

    `worktree`: the session's worktree the run wrote in, which its session's
    `workspace` line names. A run's own worktree, and an in-place run's snapshot, are keyed by
    its id and found first, so a run recorded before sessions owned worktrees resumes too."""
    identity = resolve(source)
    lease = take_write_lease(identity)
    try:
        path = identity.git_common_dir / WORKTREES_DIR / run_id
        if path.is_dir():
            return _found_worktree(identity, lease, path)
        ref = f"{SNAPSHOT_REF_PREFIX}/{run_id}"
        found = _run_git(identity.root, "rev-parse", "--verify", "--quiet", ref)
        if found.exit_code == 0:
            dirty = _text(found)
            parent = _run_git(identity.root, "rev-parse", "--verify", "--quiet", f"{dirty}^")
            pre_image = _text(parent) if parent.exit_code == 0 else EMPTY_TREE
            return _in_place_workspace(
                identity, lease, run_id=run_id, pre_image=pre_image, dirty=dirty
            )
        if worktree is not None and worktree.is_dir():
            return _found_worktree(identity, lease, worktree)
        raise InfraError(
            f"run {run_id} has no workspace to continue: no worktree at {path} and no "
            f"snapshot at {ref}"
        )
    except BaseException:
        lease.release()
        raise
