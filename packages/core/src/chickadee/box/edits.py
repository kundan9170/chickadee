"""The box's edit port: the one write path for every model-authored change to the
worktree, the task worker's and Athena's.

`WorktreeEdits` is `tools/context.EditsPort` over a worktree. Each call opens a fresh
`CandidateDraft` over the files as they are on disk now, since a shell command may have changed
them since the last call, and the draft keeps every rule it held for the packet worker: the one
write, an elision refused, a call all or none, each changed file's syntax reported. This module
adds what a live worktree needs:

- **Which paths a call may take on** (`refusal`): inside the worktree, links and
  `..` resolved, not inside a nested repository or `.git` (`paths.contained`), and not a secret
  path (`secrets.denylist.write_denied`). Anything else is taken on as it is on disk: an edit
  for a file that exists, a create for one that does not.
- **What may be deleted or moved** (`removable`): only a file undo could bring
  back from the snapshot: tracked, or untracked, not ignored, and within the snapshot's cap.
  Anything else is refused, naming why, and pointed at the shell, where the person is asked.
- **A path that is not there** (`_meant`): its refusal names the file it most
  likely meant, from what `glob` lists, so a mount's absolute path is answered with the
  repository's.
- **Empty content over a non-empty file** is refused (`edits/draft.emptied`), since no
  Applier gate follows these writes.
- **The write** (`on_answered`): each changed file through `durable.write_atomic`, which keeps
  a replaced file's mode; a moved file keeps its source's mode; a deleted one is unlinked
  durably. The step around the call (`box/steps.py`) records the trees before and after.

Composed per task by the episode (`box/compose.compose_episode`).

**Concurrency.** One `WorktreeEdits` per episode, serving one call at a time on its thread;
each call's draft is its own.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from chickadee.approval.prefs import denied, path_rule
from chickadee.approval.tier import Preferences
from chickadee.box.syntax import SyntaxChecker
from chickadee.context.work_line import WorkLine
from chickadee.durable.files import unlink_durably, write_atomic
from chickadee.edits.draft import CandidateDraft, DraftMode, DraftRefused, SyntaxCheck
from chickadee.edits.lines import AnchorStyle, Decoded, decode
from chickadee.graph.ignore import listed
from chickadee.paths import contained, suggest, suggest_under, within
from chickadee.process import run_git
from chickadee.secrets.denylist import Denylist, read_denied, write_denied
from chickadee.tools.context import DraftCall, EditScope
from chickadee.tools.tool import Answer, Invocation
from chickadee.work import in_work, resolve
from chickadee.workspace import ignored

__all__ = ["ATHENA_FENCE", "WORKER_FENCE", "PlanDocs", "WorkFence", "WorkRule", "WorktreeEdits"]

_BY_SHELL = "if it must go, delete it with the shell, which asks the person"
_LEAVE = "leave this file alone; if the task needs it changed, say so in your report"


@dataclass(frozen=True)
class WorkRule:
    """What a role is told when its edit tools reach the work folder: why, and what instead."""

    why: str
    do: str


@dataclass(frozen=True)
class WorkFence:
    """The work folder an edit port refuses, and what the role is told:
    Athena that `doc` writes the plan, the worker that the folder is read-only to it."""

    dir: str
    fence: WorkRule


ATHENA_FENCE: Final = WorkRule(
    "the plan is written with doc", "write it with doc, naming the file from plan/"
)
WORKER_FENCE: Final = WorkRule(
    "the work folder is read-only to you",
    "leave it; if the design or the plan is wrong, say so in your report",
)


class PlanDocs:
    """`DocsPort` for Athena: her `plan/` of the open work, through an edit
    port of its own, so `doc` is not gated by the held task as her code edits are.
    Her `doc` call is a writing call, so her step records it as it records an edit."""

    def __init__(self, root: Path, edits: WorktreeEdits, work: WorkLine | None) -> None:
        self._root = root
        self._edits = edits
        self._work = work

    def resolve(self, file: str, title: str | None) -> str | DraftRefused:
        # Her plan is written into the work Zeus opened and named, so a title names nothing:
        # left unread, as Zeus's own is on a later call (`zeus/docs.py`). Refused, it cost a
        # call every episode, and its instruction told her never to write the README again.
        del title
        return resolve(self._root, self._work, "plan", file)

    def edit(self, call: Invocation, approved_sha: str) -> EditScope:
        del call, approved_sha  # her step records the call (`box/steps.StepRecord`)
        return self._edits.edit()

    def noted(self, answer: Answer) -> Answer:
        return answer


_SEEN: ContextVar[set[str] | None] = ContextVar("chickadee_edits_seen", default=None)
"""The paths written in the current `writes_seen` block, `None` outside one. A context
variable, so a write on another thread, a job's or another call's, is never counted."""


@contextmanager
def writes_seen() -> Generator[set[str]]:
    """The paths every `WorktreeEdits` writes or removes inside this block: what one call wrote
    itself, which a step beside a running job tells apart from the job's writes
    (`box/steps.StepRecord`)."""
    paths: set[str] = set()
    token = _SEEN.set(paths)
    try:
        yield paths
    finally:
        _SEEN.reset(token)


def wrote(paths: Iterable[str]) -> None:
    """Note `paths` as written by the call in the current `writes_seen` block, if any."""
    seen = _SEEN.get()
    if seen is not None:
        seen.update(paths)


class WorktreeEdits:
    """`EditsPort` over the worktree at `root`."""

    def __init__(
        self,
        root: Path,
        denylist: Denylist,
        checker: SyntaxChecker,
        *,
        edit_check: Literal["reject", "report", "off"] = "report",
        untracked_max_bytes: int,
        style: AnchorStyle | None = None,
        prefs: Callable[[], Preferences] | None = None,
        work: WorkFence | None = None,
    ) -> None:
        self._root = root
        self._work = work
        """The work folder this port refuses, and why; None for `doc`'s own
        port, which `work.resolve` scopes instead."""
        self._real = Path(os.path.realpath(root))
        """`read_denied` judges a link's landing against the workspace's real path."""
        self._denylist = denylist
        self._checker = checker
        self._edit_check: Literal["reject", "report", "off"] = edit_check
        self._cap = untracked_max_bytes
        self._style = style
        """The role's anchor style when it offers a line tool, else None."""
        self._prefs = prefs
        """The person's approval.toml, read per path: its `deny` globs refuse a write.
        """

    @contextmanager
    def edit(self) -> Generator[DraftCall]:
        """One edit call, on a draft over the worktree as it is now."""

        def answered(files: Mapping[str, bytes | None]) -> None:
            # Told after the call is answered, while its moves are still the draft's.
            self._write(files, draft.moves)

        draft = CandidateDraft(
            None,
            self._read,
            self.refusal,
            judge=self._judged,
            edit_check=self._edit_check,
            opens=self._opens,
            on_answered=answered,
            refuse_emptied=True,
            removable=self.removable,
            style=self._style,
            languages=self._checker.language_of,
            meant=self._meant,
        )
        with draft.call() as call:
            yield call

    def refusal(self, path: str) -> DraftRefused | None:
        """Why `path` may not be written, in the edit tool's words, and what to do instead; or
        None."""
        why = contained(path, self._root)
        if why is not None:
            return _denied(
                why,
                "name the file by its path from the worktree's root, with no '..', outside "
                ".git and outside any nested repository",
            )
        secret = write_denied(path, self._denylist)
        if secret is not None:
            return _denied(f"{path} is a secret path ({secret.pattern})", _LEAVE)
        # By its name above; by where it lands here, so a link to a secret is neither shown
        # nor written through.
        landed = read_denied(path, self._real / path, self._denylist, workspace=self._real)
        if landed is not None:
            return _denied(f"{path} leads to a secret path ({landed.pattern})", _LEAVE)
        if self._work is not None and self._in_work(path):
            fence = self._work.fence
            return _denied(
                f"{path} is in the work folder {self._work.dir}: {fence.why}", fence.do
            )
        if self._prefs is not None:
            prefs = self._prefs()
            glob = denied(prefs, path)
            if glob is not None:
                return _denied(f"{path} is denied by {path_rule(prefs, glob)}", _LEAVE)
        return None

    def _meant(self, path: str) -> str | None:
        """The file a refused `path` most likely meant, from what `glob` lists: for one that is
        not in the worktree, or lands outside it (a mount's absolute path),
        never one refused for anything else. One that names no file yet is a create's: its
        directory, with its name."""
        landing = contained(path, self._root)
        if landing is None and (self.refusal(path) is not None or (self._root / path).exists()):
            return None
        files = listed(self._root, self._denylist)
        found = suggest(path, files)
        if found is None and landing is not None:
            found = suggest_under(path, files)
        return found

    def _in_work(self, path: str) -> bool:
        """By its name, and by where it lands: a link outside `work.dir` into it is the folder
        too, so the read-only design and plan are never written through one."""
        if self._work is None:
            return False
        if in_work(path, self._work.dir):
            return True
        landed = within(self._real, path)
        if landed is None:
            return False
        return in_work(landed.relative_to(self._real).as_posix(), self._work.dir)

    def removable(self, path: str) -> DraftRefused | None:
        """Why the file at `path` may not be deleted or moved, and what to do instead; or None:
        undo must be able to bring it back from the snapshot."""
        refused = self.refusal(path)
        if refused is not None:
            return refused
        target = self._root / path
        if target.is_dir():
            return _denied(f"{path} is a directory", _BY_SHELL)
        if not target.is_file():
            return _denied(f"{path} is not a file", _BY_SHELL)
        if self._git("ls-files", "--error-unmatch", "--", f":(literal){path}") == 0:
            return None
        if ignored(self._root, path):
            return _denied(f"{path} is git-ignored, so undo couldn't bring it back", _BY_SHELL)
        size = target.stat().st_size
        if size > self._cap:
            return _denied(
                f"{path} is {size} bytes, over the snapshot's {self._cap}-byte cap for an "
                "untracked file, so undo couldn't bring it back",
                _BY_SHELL,
            )
        return None

    def _judged(self, path: str, content: bytes) -> SyntaxCheck | None:
        """`content`'s syntax as `path` would be written. A check that is a command reads the
        file on disk, and the draft judges a call before its answer writes it, so `content` is
        put in place for the command and what was there put back after: the call's own write
        is the answer's (`_write`), or none if the call is undone. A crash between leaves the
        call's step open, and resume puts the step's paths back (`box/resume.py`)."""
        profile = self._checker.on_disk(path)
        if profile is None:
            return self._checker.check(path, content)
        target = self._root / path
        was = target.read_bytes() if target.is_file() else None
        made = [d for d in target.parents if d != self._root and not d.exists()]
        try:
            write_atomic(target, content)
            return self._checker.command(path, profile)
        finally:
            if was is not None:
                write_atomic(target, was)
            else:
                unlink_durably(target)
                for directory in made:  # deepest first, as `parents` gives them
                    directory.rmdir()

    def _opens(self, path: str) -> DraftMode | DraftRefused:
        target = self._root / path
        if target.exists() and not target.is_file():
            return _denied(f"{path} is not a file", "name a file; glob lists them")
        return "edit" if target.is_file() else "create"

    def _read(self, path: str) -> Decoded | None:
        try:
            return decode((self._root / path).read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError):
            return None

    def _write(self, files: Mapping[str, bytes | None], moves: Mapping[str, str]) -> None:
        """The answered call's files on disk: writes first, so a moved file's source still has
        its mode when its target is written, then removals. Each is the call's own (`wrote`)."""
        wrote(files)
        for path, data in files.items():
            if data is None:
                continue
            source = moves.get(path)
            mode = (
                stat.S_IMODE((self._root / source).stat().st_mode)
                if source is not None and (self._root / source).is_file()
                else None
            )
            write_atomic(self._root / path, data, mode=mode)
        for path, data in files.items():
            if data is None:
                unlink_durably(self._root / path)

    def _git(self, *args: str) -> int:
        return run_git(self._root, *args).exit_code


def _denied(why: str, do: str) -> DraftRefused:
    return DraftRefused("DENIED", why, do=do)
