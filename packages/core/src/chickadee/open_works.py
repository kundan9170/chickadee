"""The repository's open work.

Every session's log already records the work it opened (its `work` line) and the worktree it
writes (its `workspace` line). This module reads them across every session of the clone and
answers which pieces of work are still open, the newest first, so a new session can continue
one instead of starting beside it.

**A projection, never a second record.** The session logs are the truth, and this is read
from them each time it is asked, so it cannot drift from them. The one write is `adopt`: a new
session's own copy of the lines it continues, through the session's own methods. A work
continued by a later session is that session's from then on, because the newest line about a
work decides what it is.

**Open** means: its newest `work` line is `open` or `kept` (leaving is `keep`), or, for a
worktree with no work folder, its newest `workspace` line still names a worktree
the session writes (`still_written`). It is **not** open when:

- its work was landed or dropped (`landed`, `dropped`), or its worktree handed off;
- it has reached the person's branch anyway: a worktree branch merged into `HEAD`
  (`handoff.landed`), or an in-place work whose `report.md` is committed there, which is how
  `handoff.record_works` already reads a work as landed;
- the session that holds it is open in another process, which is still working on it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from chickadee.context.session import Session, SessionState, WorkspaceLine
from chickadee.context.view import still_written
from chickadee.context.work_line import WorkLine
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.handoff import landed
from chickadee.process import ProcessSupervisor, run_git
from chickadee.schema.isolation import WorkspaceMode
from chickadee.work import record_work, title_of
from chickadee.work_records import REPORT

__all__ = ["OpenWork", "OpenWorkInfo", "adopt", "open_works", "work_key"]

_OPEN_STATES = frozenset({"open", "kept"})


@dataclass(frozen=True)
class OpenWork:
    """One piece of open work: what a new session continues."""

    key: str
    """What names it across sessions: `work:<dir>/<name>` for a work folder, else
    `worktree:<path>`."""
    session_id: str
    """The session whose line about it is the newest: the one that holds it now."""
    at: str
    """When that line was written."""
    title: str | None
    """The work's title, from its README's first heading; None when it has none."""
    work: WorkLine | None
    """Its work folder's line, None for a worktree with no work folder."""
    workspace: WorkspaceRef | None
    """The worktree it is written in, None in place."""

    @property
    def mode(self) -> WorkspaceMode:
        return "in_place" if self.workspace is None else "worktree"

    def info(self) -> OpenWorkInfo:
        """What a frontend is told of it: never the session's own line types."""
        return OpenWorkInfo(
            key=self.key,
            title=self.title,
            work=self.work.name if self.work is not None else None,
            mode=self.mode,
            worktree_path=self.workspace.worktree_path if self.workspace is not None else None,
            session_id=self.session_id,
            at=self.at,
        )


@dataclass(frozen=True)
class OpenWorkInfo:
    """One piece of open work as a frontend reads it: what the start header
    says once, and what the person picks from when several are open."""

    key: str
    """Passed back to `create_session(adopt=…)` to continue it."""
    title: str | None
    work: str | None
    """The work folder's `<date>-<slug>`, None for a worktree with no work folder."""
    mode: WorkspaceMode
    worktree_path: str | None
    session_id: str
    """The session that holds it now."""
    at: str
    """When it was last written."""


def adopt(session: Session, work: OpenWork) -> None:
    """Continue `work` in `session`, a new session: its worktree's line, then
    its work folder's line opened, appended to the new session's own log, so every reader of a
    session's work (`open_work`, `still_written`) finds it there and the newest line about it
    is this session's. Nothing is moved or committed."""
    if work.workspace is not None:
        session.record_workspace(work.workspace)
    if work.work is not None:
        line = work.work
        record_work(session, WorkLine(name=line.name, dir=line.dir, state="open"))


def work_key(line: WorkLine) -> str:
    return f"work:{line.dir}/{line.name}"


def _worktree_key(ref: WorkspaceRef) -> str:
    return f"worktree:{ref.worktree_path}"


@dataclass(frozen=True)
class _Seen:
    """The newest a session says about one key."""

    at: str
    order: int
    is_open: bool
    session_id: str
    work: WorkLine | None
    workspace: WorkspaceRef | None


def open_works(
    repo: Path,
    sessions: Iterable[tuple[str, SessionState]],
    *,
    live: Callable[[str], bool] = lambda _sid: False,
    supervisor: ProcessSupervisor | None = None,
) -> tuple[OpenWork, ...]:
    """The repository's open work, the newest first. `sessions` is every session's id and
    fold; `live` says whether a session is open in another process, whose work is not free to
    continue."""
    sup = supervisor or ProcessSupervisor()
    newest: dict[str, _Seen] = {}
    order = 0
    for session_id, state in sessions:
        for seen_key, seen in _said(session_id, state):
            order += 1
            ranked = _Seen(seen.at, order, seen.is_open, session_id, seen.work, seen.workspace)
            held = newest.get(seen_key)
            if held is None or (ranked.at, ranked.order) > (held.at, held.order):
                newest[seen_key] = ranked
    found = [
        OpenWork(
            key=key,
            session_id=seen.session_id,
            at=seen.at,
            title=_title(repo, seen),
            work=seen.work,
            workspace=seen.workspace,
        )
        for key, seen in newest.items()
        if seen.is_open and not live(seen.session_id) and not _reached(repo, seen, sup)
    ]
    found.sort(key=lambda w: w.at, reverse=True)
    return tuple(found)


def _said(session_id: str, state: SessionState) -> list[tuple[str, _Seen]]:
    """What one session last said about each work and the worktree it holds: each work's
    newest `work` line, placed where the session was when it opened that work, and its
    newest worktree's line when no open work of its own is written there."""
    works: dict[str, tuple[str, WorkLine, WorkspaceRef | None]] = {}
    here: WorkspaceRef | None = None
    last_tree: tuple[str, WorkspaceRef] | None = None
    trees: dict[str, tuple[str, WorkspaceRef]] = {}
    for event, record in state.lines():
        if event.session_id != session_id:
            continue  # a fork's read-through lines are its parent's to say
        if isinstance(record, WorkspaceLine):
            here = record.ref
            if record.ref.mode == "worktree" and record.ref.worktree_path:
                # The worktree it holds, whatever lines naming the checkout came after: a
                # session that left its worktree keeps it (`held_worktree`).
                last_tree = (event.at, record.ref)
                trees[record.ref.worktree_path] = last_tree
        elif isinstance(record, WorkLine):
            key = work_key(record)
            known = works.get(key)
            if record.state == "open" or known is None:
                # Where it is written: the session's place when it opened it, never a
                # worktree it entered or left later.
                place = here if here is not None and here.mode == "worktree" else None
            else:
                place = known[2]
            works[key] = (event.at, record, place)
    out: list[tuple[str, _Seen]] = []
    claimed: set[str] = set()
    for key, (at, line, place) in works.items():
        is_open = line.state in _OPEN_STATES
        held: WorkspaceRef | None = None
        when = at
        if is_open and place is not None and place.worktree_path:
            tree_at, tree = trees.get(place.worktree_path, (at, place))
            held = still_written(tree)
            if held is not None:
                claimed.add(place.worktree_path)
                when = max(at, tree_at)
        out.append((key, _Seen(when, 0, is_open, session_id, line, held)))
    if last_tree is not None:
        at, ref = last_tree
        path = ref.worktree_path
        if path is not None and path not in claimed:
            # A worktree with no open work folder: open while still written, closed once
            # handed off; a worktree that holds an open work is that work's, said above.
            written = still_written(ref)
            out.append(
                (
                    _worktree_key(ref),
                    _Seen(at, 0, written is not None, session_id, None, written),
                )
            )
    return out


def _reached(repo: Path, seen: _Seen, sup: ProcessSupervisor) -> bool:
    """Whether the work reached the person's branch without a handoff that says so."""
    ref = seen.workspace
    if ref is not None and ref.branch is not None:
        tip = run_git(repo, "rev-parse", "--verify", "--quiet", ref.branch, supervisor=sup)
        if tip.exit_code != 0:
            return False
        head = tip.stdout.decode("utf-8", "replace").strip()
        return landed(repo, ref.source_head, head, supervisor=sup) is not None
    line = seen.work
    if line is None:
        return False
    report = f"{line.dir}/{line.name}/{REPORT}"
    done = run_git(repo, "log", "-1", "--format=%H", "--", report, supervisor=sup)
    return done.exit_code == 0 and bool(done.stdout.strip())


def _title(repo: Path, seen: _Seen) -> str | None:
    """The work's title: its README's first heading, where the work is written."""
    line = seen.work
    if line is None:
        return None
    ref = seen.workspace
    root = Path(ref.worktree_path) if ref is not None and ref.worktree_path else repo
    readme = root / line.dir / line.name / "README.md"
    try:
        return title_of(readme.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None
