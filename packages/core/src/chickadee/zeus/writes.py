"""A Zeus command that may write.

Every Zeus command that is not a pure read runs with the session's worktree writable (D199
amendment 1; it was only one that declared `effects=`), and this module is everything around
that one spawn:

- **An undo unit.** The tree is parked under the operation's ref before anything starts,
  keyed by the session as a run keys its own (`workspace.ops_ref`), and `/undo <operation_id>`
  restores the declared paths from it.
- **D40's lifecycle, on the session log.** Zeus's commands belong to no run,
  so the intent, the start and the completion are `shell_effect` lines, each one
  journal record (`RecordLine`), written through `journal.effects.perform` under its own kind,
  `EffectKind.SHELL_WRITE`.
- **Source stays the coder's.** A file an installed language
  profile owns that the command changed or created is put back from the snapshot (a new one
  deleted), and the answer says so, unless git's own history brought it
  (`CommandTable.git_line`): every source change of a line that is only git working on its
  history and commands that write nothing (`cd x && git switch y | tail`), and, of a line that
  runs git beside something that may write (`git switch y && pytest`), a file that now holds
  what the commit git left `HEAD` on holds, when that commit is history the repository had or
  fetched (`workspace.history_head`). A source path's entry in git's index is held to the
  same rule: a commit takes source from the index, so one the command staged that
  history does not hold is put back too. A line whose git would record the write into history
  is refused before it runs (`CommandTable.git_refusal`); one whose `HEAD` a program it
  ran moved unseen is read as git beside a write, and a commit it made is named. A path the
  person's `deny` globs name is put back the same way.
- **git's own files guarded.** `git_guard.held` watches git's `config`, `hooks` and `info`,
  which git reads or runs later outside any sandbox; a command that changed one fails, and
  every path it changed is put back.
- **Recorded as what it wrote.** Its completion names the paths it changed and kept, which are
  what is committed and what `/undo` restores. A command that exits unsuccessfully keeps what
  it wrote, recorded the same way: its result says it failed.
- **Under the repository WRITE lease.** A run's worktree holds the lease for each `coder`
  call; a write meanwhile would land between its packets uncounted, so it is refused.
- **Committed in the session's worktree.** A write that left changes is committed on its own,
  under the harness's identity and the operation's trailer, so no later packet folds it into
  its commit and no packet undo reverts it silently (the task's design, decision 3). In the
  person's checkout, before the session has a worktree, nothing is committed: it is their tree.
- **A branch change follows the record.** A write that leaves the session's worktree on another
  branch appends a `workspace` line naming it, which the next `coder` call,
  the handoff and Zeus's view follow.

**Concurrency.** One `ShellWrites` per root of a session's view; Zeus's tool calls run one at a
time on the conversation's worker, so nothing here is shared between threads.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from chickadee.approval.commands import GitLine
from chickadee.approval.prefs import denied
from chickadee.approval.tier import Preferences
from chickadee.box.commit import commit_message
from chickadee.box.steps import Beside, beside_note, beside_precondition
from chickadee.context.record_lines import SHELL_EFFECT, ShellEffectLine
from chickadee.context.session import Session, SessionState
from chickadee.context.store import SessionEvent
from chickadee.context.view import latest_workspace, still_written
from chickadee.errors import HarnessBug, InfraError
from chickadee.executor.effects import SKIP_DIRS
from chickadee.executor.types import ExecRequest, ExecResult
from chickadee.git_guard import dump, git_dirs, held, put_back, tampered_note
from chickadee.git_guard import guarded as guarded_paths
from chickadee.init.detect import resolve
from chickadee.init.lease import take_write_lease
from chickadee.journal.effects import EFFECT_EVENTS, EffectKind, Intent, failed_if, perform
from chickadee.paths import os_junk, within
from chickadee.profiles.model import LanguageProfile
from chickadee.schema import sha256_hex
from chickadee.schema.journal import EffectCompletedP, EffectIntendedP, EventKind, P
from chickadee.shell import Spawn, Yielded
from chickadee.work import in_work
from chickadee.workspace import (
    EMPTY_TREE,
    changed_paths,
    commit_entries,
    commit_paths,
    current_tree,
    dirty_paths,
    head_and_branch,
    history_head,
    index_bytes,
    index_entries,
    ref_tips,
    restore_index,
    restore_tree,
    snapshot_operation,
    unmerged_paths,
)

__all__ = [
    "SHELL_EFFECT",
    "ShellEffect",
    "ShellEffectLine",
    "ShellWrites",
    "guarded_before",
    "not_gated",
    "shell_effects",
]

_ROOT: Final = "root="
"""The precondition naming where a recorded write ran, read back by `shell_effects`."""

_GUARDED: Final = "guarded="
"""The precondition naming the blob of what `git_guard` held before the command ran: what
resume puts back after a job a crash cut off."""


def guarded_before(preconditions: tuple[str, ...]) -> str | None:
    """The blob of a step's `git_guard` hold, as its intent names it, or `None`."""
    return next((p[len(_GUARDED) :] for p in preconditions if p.startswith(_GUARDED)), None)


_LINE: Final = TypeAdapter(ShellEffectLine)


@dataclass(frozen=True)
class _Before:
    """What a line that is not only git's history is checked against, taken before it runs:
    the repository's history (`ref_tips`), the commit `HEAD` was on, when it started, and git's
    index file."""

    tips: tuple[str, ...]
    head: str | None
    started: float
    index: bytes | None


@dataclass(frozen=True)
class _After:
    """git as a command left it, read against `_Before`. `line` is the line
    as its effect shows it: `BESIDE` when it ran git, or a program it ran moved `HEAD`. `head`
    is the commit `HEAD` is on when that is history (`history_head`), what kept source and
    index entries must hold; `made`, a commit the command made itself that `HEAD` is on."""

    line: GitLine
    head: str | None = None
    made: str | None = None


@dataclass(frozen=True)
class _SourceBack:
    """The source paths a command changed that are put back. Each comes back
    from the step's tree before, except `from_head`: a path whose committed bytes git moved
    (a branch switch) comes back from `head`, the commit git left `HEAD` on, so the file holds
    what git brought rather than what the branch it left held."""

    paths: tuple[str, ...] = ()
    head: str | None = None
    from_head: tuple[str, ...] = ()


class SessionEffects:
    """`EffectLog` over the session log: each record a `shell_effect` line, durable before
    `record` returns, as an effect's intent must be before its command is spawned."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, kind: EventKind, payload: P) -> int:
        if kind not in EFFECT_EVENTS:
            raise HarnessBug(f"a shell_effect line is a D40 record, not {kind.value!r}")
        event = self._session.append(ShellEffectLine(kind, payload))
        self._session.sync()
        return event.seq


@dataclass(frozen=True)
class ShellEffect:
    """One effect on the session log, read back from its `shell_effect` lines: a shell step
    (`shell_write`) or a job (`shell_job`). Every reader reads `kind`."""

    operation_id: str
    kind: str
    """`effect_kind` as intended."""
    preconditions: tuple[str, ...]
    surface: tuple[str, ...]
    """What it declared: paths and globs."""
    written: tuple[str, ...]
    """What it wrote inside the declaration, as its completion observed: the paths its undo
    restores. On a failed write, what was put back."""
    events: tuple[str, ...]
    outcome: str | None
    post_state: str | None
    """The commit that holds it, in the session's worktree; None in the checkout or when it
    changed nothing."""
    root: Path | None
    """Where it wrote: the session's worktree or the person's checkout."""
    result: str | None = None
    """The completion's `result_sha`: a step's stdout blob, a job's `JobReport` blob."""

    @property
    def is_job(self) -> bool:
        return self.kind == EffectKind.JOB.value


def shell_effects(
    state: SessionState, payload: Callable[[SessionEvent], bytes]
) -> tuple[ShellEffect, ...]:
    """Every declared write on the session's log, in the order their intents were written.
    `payload` reads a line's bytes (`session.blob` by the event's sha); the caller holds the
    session's lock if the log is live."""
    found: dict[str, ShellEffect] = {}
    for event, _ in state.lines():
        if event.kind != SHELL_EFFECT:
            continue
        line = _LINE.validate_json(payload(event))
        record = line.payload
        if isinstance(record, EffectIntendedP):
            root = next(
                (Path(p[len(_ROOT) :]) for p in record.preconditions if p.startswith(_ROOT)),
                None,
            )
            found[record.operation_id] = ShellEffect(
                record.operation_id,
                record.effect_kind,
                record.preconditions,
                record.declared_surface,
                (),
                (line.event.value,),
                None,
                None,
                root,
            )
            continue
        op = getattr(record, "operation_id", None)
        seen = found.get(op) if isinstance(op, str) else None
        if seen is None:
            continue
        events = (*seen.events, line.event.value)
        if isinstance(record, EffectCompletedP):
            found[seen.operation_id] = replace(
                seen,
                written=record.observed_surface,
                events=events,
                outcome=record.outcome,
                post_state=record.post_state_sha,
                result=record.result_sha,
            )
        else:
            found[seen.operation_id] = replace(seen, events=events)
    return tuple(found.values())


def not_gated(effect: ShellEffect) -> str:
    """The handoff's line for a shell write committed in the session's worktree, or for what a
    job wrote that no step recorded: by its
    operation id, never counted as gated."""
    if effect.is_job:
        return (
            f"{effect.operation_id}: {', '.join(effect.written)} written by a background job, "
            "not recorded, not gated"
        )
    return (
        f"{effect.operation_id}: {', '.join(effect.written)} edited by a shell command, "
        "not gated"
    )


class ShellWrites:
    """The recorded writes of one session's view at `root` (`shell.RecordedWrites`).

    `commits`: `root` is the session's worktree, whose writes are committed on their own; the
    person's checkout is never committed to. `profiles` are the installed languages whose files
    are source; `prefs` the person's approval.toml, read per
    command, whose
    `deny` globs a command's change is put back from."""

    def __init__(
        self,
        session: Session,
        root: Path,
        *,
        commits: bool,
        profiles: tuple[LanguageProfile, ...] = (),
        prefs: Callable[[], Preferences] | None = None,
        work_dir: Callable[[], str] | None = None,
        beside: Beside | None = None,
    ) -> None:
        self._session = session
        self._beside = beside
        """The session's running jobs: a step beside one says so."""
        self._root = root.resolve()
        self._commits = commits
        self._profiles = profiles
        self._prefs = prefs
        self._work_dir = work_dir
        """`[work] dir`, read at each command since the person may choose
        another folder mid-session: a command's write into it is put back,
        as source's is; `doc` writes it (`recorded`)."""

    @property
    def where(self) -> str:
        """Where a write lands, for the confirmation, and what its undo cannot
        reach: the snapshot is `git add -A`'s, so a file git ignores is in no tree (confirmed
        against `workspace._write_tree`)."""
        unrecorded = "; files git ignores are not recorded"
        if self._commits:
            _, branch = head_and_branch(self._root)
            where = f"the session's worktree ({branch or 'detached'}), committed on its own"
            return where + unrecorded
        return "your checkout, left uncommitted for you" + unrecorded

    def git_dirs(self) -> tuple[Path, ...]:
        """What a git command in a linked worktree writes outside `root` (design
        4a): the repository's objects, refs and reflogs, and the worktree's own git directory
        (its HEAD and index). Never the whole common directory: its `config` and `hooks` run
        later, unsandboxed, in every git call the person makes. Nothing for the main worktree,
        whose git directory is inside `root` already; `git_guard.held` watches it there."""
        found = git_dirs(self._root)
        if found is None:
            return ()
        common, own = found
        if within(self._root, common) is not None:
            return ()
        wanted = (common / "objects", common / "refs", common / "logs", own)
        return tuple(d for d in wanted if d.is_dir())

    def run(
        self,
        request: ExecRequest,
        operation_id: str,
        cancel: threading.Event | None,
        spawn: Spawn,
        *,
        git: GitLine,
    ) -> tuple[ExecResult | Yielded, tuple[str, ...]]:
        """`RepoLeaseBusy` while a run holds the repository: a write between a run's packets
        would land in its worktree uncounted. `git`: how much of the line's source
        changes git's history accounts for, and so keeps. A command that
        yields ends its step there, as one that exited would: its put-backs
        and commit are made on the tree at the yield, and the lease is released as the job
        runs on."""
        line = request.shell_line
        if line is None:
            raise HarnessBug("a recorded write is a shell line")
        with take_write_lease(resolve(self._root)):
            return self._run(request, line, operation_id, cancel, spawn, git)

    @contextmanager
    def recorded(
        self, operation_id: str, approved_sha: str, label: str, *, commit: bool = True
    ) -> Generator[list[str]]:
        """A write that is not a command, recorded as a command's is (`doc`):
        under the WRITE lease, parked first as an undo unit, its D40 lifecycle on the session
        log, what it changed committed on its own in the session's worktree unless `commit` is
        false (`work.commit = false`, B7). The caller writes inside; the notes it is given say
        where the write went."""
        with take_write_lease(resolve(self._root)):
            tree = snapshot_operation(self._root, operation_id, run_id=self._session.session_id)
            intent = Intent(
                operation_id,
                EffectKind.SHELL_WRITE,
                approved_sha=approved_sha,
                preconditions=(
                    "profile=zeus",
                    "worktree" if self._commits else "checkout",
                    f"{_ROOT}{self._root}",
                ),
                idempotent=False,
                surface=(label,),
                preimages=tree,
            )
            notes: list[str] = []
            with perform(
                SessionEffects(self._session), intent, on_raise=failed_if(InfraError)
            ) as fx:
                yield notes
                now, _ = current_tree(self._root, untracked_max_bytes=None)
                written = changed_paths(self._root, tree, now) if now != tree else ()
                made = (
                    self._commit(operation_id, written, label)
                    if self._commits and commit
                    else None
                )
                if made is not None:
                    notes.append(
                        f"[committed as {made[:8]} on its own; /undo {operation_id} puts it "
                        "back]"
                    )
                fx.complete("succeeded", observed=written, post_state=made)
                if self._beside is not None:
                    # What it wrote is its own, never a job's that ran beside it.
                    self._beside.stepped(self._root, operation_id, written, now)

    def _run(
        self,
        request: ExecRequest,
        line: str,
        operation_id: str,
        cancel: threading.Event | None,
        spawn: Spawn,
        git: GitLine,
    ) -> tuple[ExecResult | Yielded, tuple[str, ...]]:
        # Parked before `effect_started`, so the operation is an undo unit and the
        # intent can name a tree that is already durable.
        tree = snapshot_operation(self._root, operation_id, run_id=self._session.session_id)
        before = (
            None
            if git is GitLine.HISTORY
            else _Before(
                ref_tips(self._root),
                head_and_branch(self._root)[0],
                time.time(),
                index_bytes(self._root),
            )
        )
        where = guarded_paths(self._root)
        guarded = held(self._root, where)
        # Durable with the intent: a job this command becomes may be cut off by a crash, and
        # resume puts back from this what it changed.
        kept = self._session.put_blob(dump(guarded))
        running = self._beside.running_in(self._root) if self._beside is not None else ()
        intent = Intent(
            operation_id,
            EffectKind.SHELL_WRITE,
            approved_sha=sha256_hex(line),
            preconditions=(
                "profile=zeus",
                "worktree" if self._commits else "checkout",
                f"{_ROOT}{self._root}",
                f"{_GUARDED}{kept}",
                *beside_precondition(running),
            ),
            idempotent=request.idempotent,
            surface=(),
            preimages=tree,
        )
        notes: list[str] = []
        with perform(
            SessionEffects(self._session), intent, on_raise=failed_if(InfraError)
        ) as fx:
            result = spawn(request, operation_id, cancel)
            # git's settings first, before any git of the harness's runs: a planted
            # `core.fsmonitor` or hook would run in the very call that measures the tree,
            # outside any sandbox.
            tampered = put_back(guarded, held(self._root, where))
            # Unbounded, as the snapshot is: a large file the command created is seen.
            now, _ = current_tree(self._root, untracked_max_bytes=None)
            changed = changed_paths(self._root, tree, now)
            after = self._after(git, before)
            if tampered:
                restored = restore_tree(self._root, tree, changed)
                notes.append(tampered_note(tampered, restored))
                notes.extend(self._git_notes(after, before))
                fx.complete("failed", observed=tuple(sorted({*restored, *tampered})))
                written: tuple[str, ...] = ()
                back = set(restored)
            else:
                put = self._put_back_source(changed, after, before, now)
                source = put.paths
                refused = self._denied(changed)
                documents = self._documents(changed)
                kept_back = tuple(sorted({*source, *refused, *documents}))
                from_tree = tuple(p for p in kept_back if p not in put.from_head)
                back: set[str] = (
                    set(restore_tree(self._root, tree, from_tree)) if from_tree else set()
                )
                if put.head is not None and put.from_head:
                    back.update(restore_tree(self._root, put.head, put.from_head))
                notes.extend(self._git_notes(after, before))
                if source:
                    notes.append(
                        f"[put back {', '.join(source)}: source is coder's to write, so a "
                        "command's change to it, a new file included, is undone (D199, D202)]"
                    )
                if refused:
                    notes.append(
                        f"[put back {', '.join(refused)}: your approval.toml denies writing "
                        "there]"
                    )
                if documents:
                    notes.append(
                        f"[put back {', '.join(documents)}: the work folder is written with "
                        "doc (D204)]"
                    )
                written = tuple(
                    p
                    for p in changed
                    if p not in back
                    and p not in kept_back
                    and not SKIP_DIRS.intersection(p.split("/"))
                )
                commit = self._commit(operation_id, written) if self._commits else None
                if commit is not None:
                    notes.append(
                        f"[committed as {commit[:8]} on its own; /undo {operation_id} puts it "
                        "back]"
                    )
                fx.complete(
                    "succeeded",
                    observed=written,
                    post_state=commit,
                    result=result.stdout_sha if isinstance(result, ExecResult) else None,
                )
            if self._beside is not None:
                # The tree the step left: `now` less what it put back, measured again only then.
                left = current_tree(self._root, untracked_max_bytes=None)[0] if back else now
                self._beside.stepped(self._root, operation_id, written, left)
        notes.extend(beside_note(running))
        notes.extend(self._branch())
        return result, tuple(notes)

    def _source(self, changed: tuple[str, ...]) -> tuple[str, ...]:
        """The changed paths that are source: a file of a
        language an installed profile owns, tracked or new, so an empty repository's first
        files are the coder's too. Not under an always-skipped directory: an install's
        `node_modules/**/*.js` is its output, not source anyone wrote."""
        return tuple(
            p
            for p in changed
            if not SKIP_DIRS.intersection(p.split("/"))
            and any(profile.owns(p) for profile in self._profiles)
        )

    def _after(self, git: GitLine, before: _Before | None) -> _After:
        """What git's history accounts for after the command (`_After`). A line with no git it
        shows, whose `HEAD` stayed, is `NONE`, costing one `rev-parse`."""
        if before is None:
            return _After(GitLine.HISTORY)
        now_head = head_and_branch(self._root)[0]
        moved = now_head != before.head
        if git is GitLine.NONE and not moved:
            return _After(GitLine.NONE)
        head = history_head(self._root, before.tips, before.started)
        return _After(GitLine.BESIDE, head, now_head if moved and head is None else None)

    def _put_back_source(
        self, changed: tuple[str, ...], after: _After, before: _Before | None, now: str
    ) -> _SourceBack:
        """The source paths to put back: none of a line that is only git's
        history; of a line that ran git, those that do not hold what the commit git left `HEAD`
        on holds, or all of them when that commit is not history (`history_head`); of any
        other line, all of them. A file deleted where that commit has none matches it."""
        if after.line is GitLine.HISTORY:
            return _SourceBack()
        source = self._source(changed)
        head = after.head
        if not source or before is None or head is None:
            return _SourceBack(source)
        differs = set(changed_paths(self._root, head, now))
        put = tuple(p for p in source if p in differs)
        if before.head == head:
            return _SourceBack(put)
        moved = set(changed_paths(self._root, before.head or EMPTY_TREE, head))
        return _SourceBack(put, head, tuple(p for p in put if p in moved))

    def _put_back_index(self, after: _After, before: _Before | None) -> tuple[str, ...]:
        """The source paths whose index entries the command changed and git's history does not
        account for, put back to their entries before: a commit takes source
        from the index, so a model's write staged there is the write. Kept: what the commit
        `HEAD` is on holds, when that is history. Nothing to read when the index file's bytes
        did not change."""
        if before is None or after.line is GitLine.HISTORY:
            return ()
        if index_bytes(self._root) == before.index:
            return ()
        was = index_entries(self._root, before.index)
        now = index_entries(self._root, now=True)
        source = self._source(
            tuple(sorted(p for p in {*was, *now} if was.get(p) != now.get(p)))
        )
        if source and after.head is not None:
            held = commit_entries(self._root, after.head, source)
            source = tuple(p for p in source if now.get(p) != held.get(p))
        if source:
            restore_index(self._root, was, source)
        return source

    def _git_notes(self, after: _After, before: _Before | None) -> tuple[str, ...]:
        """The index put back and a commit the command made, as the answer says them."""
        notes: list[str] = []
        staged = self._put_back_index(after, before)
        if staged:
            notes.append(
                f"[put back the staged {', '.join(staged)}: source is coder's to write, and a "
                "commit takes it from git's index (D229)]"
            )
        if after.made is not None:
            was = before.head if before is not None else None
            back = (
                f"`git reset --soft {was[:8]}` moves the branch back"
                if was is not None
                else "the branch had no commit before it"
            )
            notes.append(
                f"[HEAD moved to {after.made[:8]}, a commit this command made, which nothing "
                "could refuse before it ran: the source it holds was put back on disk and in "
                f"the index (D229). Do: tell the person; {back} if the commit should not stay]"
            )
        conflicted = unmerged_paths(self._root) if after.line is not GitLine.NONE else ()
        if conflicted:
            notes.append(
                f"[git stopped on conflicts in {', '.join(conflicted)}: source is the coder's "
                "to resolve (D230 decision 8). Do: send coder a one-task plan to resolve the "
                "conflicts in these files keeping both sides' behaviour, and conclude the "
                "merge with a git-only line once it lands; or `git merge --abort` (or "
                "`--abort` of the operation that stopped) to go back; ask the person when the "
                "two sides contradict]"
            )
        return tuple(notes)

    def _documents(self, changed: tuple[str, ...]) -> tuple[str, ...]:
        """The changed paths in the work folder: `doc` writes it. Not a
        file the operating system wrote there (`os_junk`): `doc` never writes one,
        so a command may delete it."""
        if self._work_dir is None:
            return ()
        work_dir = self._work_dir()
        return tuple(p for p in changed if in_work(p, work_dir) and not os_junk(p))

    def _denied(self, changed: tuple[str, ...]) -> tuple[str, ...]:
        """The changed paths the person's `deny` globs name."""
        if self._prefs is None:
            return ()
        prefs = self._prefs()
        return tuple(p for p in changed if denied(prefs, p) is not None)

    def _commit(
        self, operation_id: str, written: tuple[str, ...], label: str = "shell"
    ) -> str | None:
        """What the command wrote inside its declaration and HEAD does not already hold: a
        cache an earlier run's checks left (`__pycache__`) is dirty too, and is not this
        write's, and a `git switch` changes files that are clean on the branch it reaches. Not
        a file the operating system wrote meanwhile (`os_junk`): it is no command's.
        Its deletion is, since no file browser deletes one, so that is committed."""
        dirty = set(dirty_paths(self._root))
        paths = [
            p for p in written if p in dirty and not (os_junk(p) and (self._root / p).exists())
        ]
        if not paths:
            return None
        message = commit_message(f"chickadee: {label} {operation_id}", operation_id)
        return commit_paths(self._root, paths, message, own_unit=True)

    def _branch(self) -> tuple[str, ...]:
        """A branch the write left the session's worktree on, recorded."""
        if not self._commits:
            return ()
        _, branch = head_and_branch(self._root)
        with self._session.lock:
            ref = still_written(latest_workspace(self._session.state))
        if ref is None or ref.worktree_path is None:
            return ()
        if Path(ref.worktree_path).resolve() != self._root or branch == ref.branch:
            return ()
        if branch is None:
            return (
                "[the session's worktree is now on a detached HEAD; its record still "
                f"names {ref.branch}]",
            )
        self._session.record_workspace(replace(ref, branch=branch))
        return (
            f"[the session's worktree is now on {branch}; the next coder call, the "
            "handoff and your view follow it]",
        )
