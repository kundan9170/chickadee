"""Zeus's `docs` port: his `design/` of the session's open work.

- **The open work.** His first `doc` call with none open carries the work's `title`: the
  folder `<date>-<slug>` is named (never one that exists), its marked README written, and the
  session's `work` line opens it. Later calls write into it; a `title` then is not asked for.
  The handoff closes it; in place, where there is none, a `title` once its run reported closes
  it as landed and opens the next.
- **Where it lives (4b).** The folder is written where the work will run. In a worktree
  session whose view is still the checkout, the first call opens the session's worktree
  through the one constructor `coder` uses (`delegation.session_worktree`), so a dirty checkout
  seeds it as for `coder`, and the view follows it there as after a `coder` call.
  In place (`[workspace] mode = "in_place"`), the folder is the checkout's.
- **Never over the person's files (4c).** The harness writes only what it
  makes in `work.dir`: a new folder no existing one is named as, marked. A file or folder of
  the person's there is never written and never refused on; only a `work.dir` that cannot be a
  folder, or that git ignores, is (`work.owned_error`), and then the harness first offers the
  person a free folder for this repository.
- **Recorded as his writes are (decision 9).** Each call is one `recorded` write of the view's
  root (`zeus/writes.ShellWrites`): under the WRITE lease, an undo unit, its lifecycle on the
  session log, committed on its own in the worktree, never in the checkout. The work's own
  README is written in the opening call's draft, through the edit port, never beside it.

**Concurrency.** Zeus's tool calls run one at a time on the conversation's worker; nothing here
is shared between threads.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from chickadee.context.session import Session
from chickadee.context.view import held_worktree
from chickadee.context.work_line import WorkLine
from chickadee.edits.draft import DraftRefused
from chickadee.errors import HarnessBug
from chickadee.schema import sha256_hex
from chickadee.tools.context import DraftCall, EditsPort
from chickadee.tools.tool import Answer, Invocation, noted
from chickadee.work import (
    folder_name,
    open_work,
    owned_error,
    record_work,
    resolve,
    work_readme,
    works,
)
from chickadee.work_records import REPORT
from chickadee.workspace import chickadee_worktrees
from chickadee.zeus.writes import ShellWrites

__all__ = ["DesignDocs", "Opening"]


_PERSONS = (
    "tell the person what the reason says: only they can change it, by removing the file or "
    "un-ignoring the folder, or by choosing another folder with two lines in "
    '~/.config/chickadee/config.toml: [work] then dir = "notes/designs"; no design is written '
    "until then"
)
"""A work folder the harness may not write is the person's to fix, never the model's: since
D230 Amendment 1, only one that is not a folder, or that git ignores."""


@dataclass(frozen=True)
class Opening:
    """A work named by this call and not yet written: its folder and README are written inside
    the call's recorded write, so they are committed and undone with it."""

    line: WorkLine
    title: str
    closes: WorkLine | None = None
    """In place, the open work that already ran: closed as landed when this one opens."""


class DesignDocs:
    """`DocsPort` for Zeus. `root` is his view's root now (the checkout, or the session's
    worktree once one is open); `open_worktree` opens the session's worktree and moves the view
    there, saying so when it moved, or says why not; `edits(root)` is an edit port over a root,
    never the work fence."""

    def __init__(
        self,
        session: Session,
        root: Callable[[], Path],
        checkout: Path,
        *,
        work_dir: str,
        in_place: Callable[[], bool],
        commit: bool = True,
        open_worktree: Callable[[str], DraftRefused | str | None],
        edits: Callable[[Path], EditsPort],
        writes: Callable[[Path], ShellWrites],
        relocate: Callable[[Path, str, str], str | None] = lambda _r, _d, _w: None,
        running: Callable[[], str | None] = lambda: None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._session = session
        self._root = root
        self._checkout = checkout.resolve()
        self._work_dir = work_dir
        self._in_place = in_place
        """Whether work runs in place where the session is now (`context/view.session_mode`):
        asked at each call, since the person's flag and the session's moves change it."""
        self._commit = commit
        """`[work] commit` (B7): false, and the work folder's writes are never committed."""
        self._open_worktree = open_worktree
        self._relocate = relocate
        """The engine's offer of a folder for this repository when `[work] dir` cannot be used
        here: asked of the person, never decided by a model. Takes the root,
        the unusable folder and why; the folder the person chose, or None."""
        self._edits = edits
        self._writes = writes
        self._today = today
        self._running = running
        """What episode runs on the engine, or None: `design/` is frozen while one does (B5)."""
        self._opening: Opening | None = None
        self._notes: tuple[str, ...] = ()

    def resolve(self, file: str, title: str | None) -> str | DraftRefused:
        # An opening named by an earlier call that was refused after naming it is not this
        # call's: only the call that names it writes it.
        self._opening = None
        running = self._running()
        if running is not None:
            # B5: what Athena and the worker read is the pinned design; a change is a new stop.
            return DraftRefused(
                "DENIED",
                f"design/ is frozen while {running} builds it",
                do="change it once that run ends, with a new coder call",
            )
        with self._session.lock:
            work = open_work(self._session.state)
        if work is None or self._finished(work, title):
            # In place there is no handoff to close a work (4a): the one that already ran is
            # closed as landed, its files in the checkout for the person's commit, once this
            # call's title has opened the next (`edit`), never by a call refused before then.
            # Its index row is the next session's heal (B3).
            opened = self._open(title, closes=work)
            if isinstance(opened, DraftRefused):
                return opened
            work = opened
        return resolve(self._root(), work, "design", file)

    def _finished(self, work: WorkLine, title: str | None) -> bool:
        """In place, a `title` given once the open work's run wrote its `report.md`: a new
        design. In a worktree the handoff closes the work, never a `doc` call."""
        if not self._in_place() or title is None or not title.strip():
            return False
        return (self._root() / work.dir / work.name / REPORT).is_file()

    def _open(self, title: str | None, *, closes: WorkLine | None) -> WorkLine | DraftRefused:
        """The work this call opens (4a), its root made ready (4b, 4c); written by `edit`."""
        if title is None or not title.strip():
            return DraftRefused(
                "DENIED",
                "no work is open",
                do="send the call again with title: the first doc call of a design gives it",
            )
        if not self._in_place() and self._root().resolve() == self._checkout:
            moved = self._open_worktree(title)
            if isinstance(moved, DraftRefused):
                return moved
            if moved is not None:
                self._notes = (*self._notes, moved)
        root = self._root()
        refused = owned_error(root, self._work_dir)
        if refused is not None:
            chosen = self._relocate(root, self._work_dir, refused)
            if chosen is None:
                return DraftRefused("DENIED", refused, do=_PERSONS)
            self._notes = (
                *self._notes,
                f"the work folder is {chosen} in this repository from now on, as the user "
                f"chose: {refused}",
            )
            self._work_dir = chosen
        name = folder_name(self._today(), title, lambda n: self._taken(root, n))
        line = WorkLine(name=name, dir=self._work_dir, state="open")
        self._opening = Opening(line, title.strip(), closes)
        return line

    def _taken(self, root: Path, name: str) -> bool:
        """A name this session used, or one in the person's checkout, this root or any
        session's worktree of the repository (4a): a second session's work never reuses it."""
        if name in {w.name for w in works(self._session.state)}:
            return True
        roots = {root, self._checkout, *chickadee_worktrees(self._checkout)}
        return any((r / self._work_dir / name).exists() for r in roots)

    @contextmanager
    def edit(self, call: Invocation, approved_sha: str) -> Generator[DraftCall]:
        root = self._root()
        opening, self._opening = self._opening, None
        writes = self._writes(root)
        with (
            writes.recorded(
                call.operation_id, approved_sha, "doc", commit=self._commit
            ) as notes,
            self._edits(root).edit() as draft,
        ):
            if opening is not None:
                # The work's own README, in the same draft: one write path, one record.
                line = opening.line
                readme = f"{self._work_dir}/{line.name}/README.md"
                refused = draft.apply(readme, None, work_readme(line.name, opening.title))
                if refused is not None:
                    raise HarnessBug(f"the work's README was refused: {refused.reason}")
            yield draft
        self._notes = (*self._notes, *notes)
        # Opened only once its README landed: a call the draft refused writes none of it, so
        # the next call opens the work again rather than writing into a folder never made.
        if opening is not None and (root / self._work_dir / opening.line.name).is_dir():
            if opening.closes is not None:
                record_work(self._session, replace(opening.closes, state="landed"))
            record_work(self._session, opening.line)

    def folder_of(self, work: WorkLine) -> Path:
        """The root `work`'s folder is under: the view's, or the session's worktree when the
        view left it for the checkout and the folder is there, so a design
        written in the worktree still runs there."""
        root = self._root()
        if (root / work.dir / work.name).is_dir():
            return root
        with self._session.lock:
            held = held_worktree(self._session.state)
        if held is not None and held.worktree_path is not None:
            tree = Path(held.worktree_path)
            if (tree / work.dir / work.name).is_dir():
                return tree
        return root

    def write_readme(self, work: WorkLine, text: str) -> None:
        """`design/README.md` as the harness writes it at `coder(work)` (its Request and
        Constraints sections): one recorded write, as a `doc` call is, so
        what the stop pins is on the record and `/undo` reaches it. Unchanged, nothing."""
        root = self.folder_of(work)
        path = f"{work.dir}/{work.name}/design/README.md"
        if (root / path).is_file() and (root / path).read_text(encoding="utf-8") == text:
            return
        operation = f"{self._session.session_id}:sections-{self._session.next_seq}"
        with (
            self._writes(root).recorded(
                operation, sha256_hex(text), "doc", commit=self._commit
            ),
            self._edits(root).edit() as draft,
        ):
            refused = draft.apply(path, None, text)
            if refused is not None:
                raise HarnessBug(f"the design's sections were refused: {refused.reason}")
            said = draft.answer()
            if isinstance(said, DraftRefused):
                raise HarnessBug(f"the design's sections were refused: {said.reason}")

    def noted(self, answer: Answer) -> Answer:
        notes, self._notes = self._notes, ()
        for note in notes:
            answer = noted(answer, note)
        return answer
