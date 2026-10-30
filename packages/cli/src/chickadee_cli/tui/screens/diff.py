"""The diff screen: `^D` or `/diff`. The
files on the left, grouped by task, and the hunks of the chosen file on the right; the same
diff the centre CODE pane shows, at its full size (one diff view).

**Two scopes** (`t`):
- **This task**, the task the worker pane shows (`TuiState.task_diff`), as the CODE pane: its
  files, their lines added and removed, and its hunks, start to its latest step, as the
  server read them (`/run/{id}/diff/task/{task}`). A server that does not
  build it says `render/diff.NO_LINES`.
- **All tasks**, the run's tasks from the board, each with the files `render/diff.task_files`
  names (the one rule, `/diff`'s), and the hunks of the session's **committed** change from
  its base (`HandoffSummary.diff`, read only while this screen is open), titled as exactly
  that. A file no commit holds says `NO_LINES`; a server that does not build `/handoff`
  (a replay's) says so.

**What it says is the record's.** A file's counts are its task's latest facts, absent until
they measure it; the total is unknown while any count is; committed or not is the board's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.events import Resize
from textual.message import Message
from textual.widgets import Static

from chickadee_cli.render.diff import NO_LINES, task_files
from chickadee_cli.tui.diffs import FileHunks, split_diff
from chickadee_cli.tui.feed import SETTLED
from chickadee_cli.tui.keys import ScreenView, diff_bound
from chickadee_cli.tui.model import DiffScope, StaleRoute, TuiState
from chickadee_cli.tui.panes import border_title
from chickadee_cli.tui.panes.diff import hunk_lines, live
from chickadee_cli.tui.panes.worker import syntax_word, tests_word
from chickadee_cli.tui.statusbar import key_hint
from chickadee_cli.tui.theme import TASK_GLYPH, Glyphs

__all__ = ["NOT_READ", "NOT_SERVED", "ChangedFile", "FullDiff", "changed_files"]

NOT_SERVED: Final = "the session's line diff is not served here"
NOT_READ: Final = "reading the session's diff…"
_HINT: Final = (("n", "next file"), ("left_square_bracket", "hunk"))


@dataclass(frozen=True)
class ChangedFile:
    """One file in the left list: its task, its path, its counts (None: not measured)."""

    task_id: str
    path: str
    added: int | None
    removed: int | None
    committed: bool


def changed_files(state: TuiState, scope: DiffScope) -> list[ChangedFile]:
    """The files the scope shows, in the list's order."""
    if scope is DiffScope.TASK:
        view = state.task_diff
        task = state.worker.task
        board = state.athena.board
        shown = board.tasks.get(task.task_id) if board is not None and task else None
        committed = shown is not None and shown.state == "committed"
        return [
            ChangedFile(view.task_id or "", f.path, f.added, f.removed, committed)
            for f in view.files
        ]
    board = state.athena.board
    out: list[ChangedFile] = []
    for task in board.tasks.values() if board is not None else ():
        files, committed = task_files(task)
        worker = state.worker.tasks.get(task.task_id)
        facts = worker.facts if worker is not None else None
        counts = {f.path: (f.added, f.removed) for f in facts.files} if facts else {}
        for path in files:
            added, removed = counts.get(path, (None, None))
            out.append(ChangedFile(task.task_id, path, added, removed, committed))
    return out


class FullDiff(Horizontal):
    """The diff screen. `show(state)` draws it; its keys are the diff bar's (`DIFF_BAR`)."""

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(k.key, k.action, show=False) for k in diff_bound() if k.action is not None
    ]

    DEFAULT_CSS = """
    FullDiff { height: 1fr; display: none; }
    FullDiff > #changed {
        width: 30%; border: round $worker; padding: 0 1; scrollbar-size-vertical: 1;
    }
    FullDiff > #hunks {
        width: 1fr; border: double $worker; padding: 0 1; scrollbar-size-vertical: 1;
    }
    FullDiff Static { height: auto; }
    FullDiff .code, FullDiff #list { text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, colours: dict[str, str], glyphs: Glyphs) -> None:
        super().__init__(id="fulldiff")
        self._colours, self._glyphs = colours, glyphs
        self.scope = DiffScope.TASK
        """the mockup opens on "this task"."""
        self.chosen = 0
        """The chosen file, by its place in the list."""
        self.hunk = 0
        """The chosen hunk of that file."""
        self.state: TuiState | None = None
        self._parsed: tuple[str, dict[str, FileHunks]] | None = None
        self.changed = VerticalScroll(id="changed")
        self.hunks = VerticalScroll(id="hunks")
        self._list = Static("", id="list")
        self._body = Static("", id="body", classes="code")
        self._foot = Static("", id="foot")

    def compose(self) -> ComposeResult:
        with self.changed:
            yield self._list
        with self.hunks:
            yield self._body
            yield self._foot

    # -- what it shows ---------------------------------------------------------------------

    def files(self) -> list[ChangedFile]:
        return changed_files(self.state, self.scope) if self.state is not None else []

    def file_hunks(self) -> FileHunks | None:
        """The chosen file's committed hunks, at "all tasks"; None where there are none."""
        state, files = self.state, self.files()
        if state is None or not files:
            return None
        if self.scope is DiffScope.TASK:
            text = state.task_diff.text  # the task's own lines, as the server read them
        else:
            text = state.handoff.diff if state.handoff is not None else None
        if text is None:
            return None
        if self._parsed is None or self._parsed[0] != text:
            self._parsed = (text, split_diff(text))
        return self._parsed[1].get(files[min(self.chosen, len(files) - 1)].path)

    @property
    def hunks_shown(self) -> bool:
        hunks = self.file_hunks()
        return hunks is not None and bool(hunks.hunks)

    def show(self, state: TuiState) -> None:
        self.state = state
        files = self.files()
        self.chosen = min(self.chosen, max(len(files) - 1, 0))
        self._list.update(self._list_text(state, files))
        chosen = files[self.chosen] if files else None
        self._body.update(self._hunks_text(state, chosen))
        self._foot.update(self._foot_text(state, chosen))
        self._retitle(chosen)

    def _list_text(self, state: TuiState, files: list[ChangedFile]) -> Content:
        c, g = self._colours, self._glyphs
        on, off = f"bold {c['ink']} on {c['fg']}", c["dim"]
        switch = Content.assemble(
            (" all tasks ", on if self.scope is DiffScope.ALL else off),
            " ",
            (" this task ", on if self.scope is DiffScope.TASK else off),
        )
        lines = [switch, Content("")]
        if not files:
            lines.append(Content.styled("no file changed yet", c["dim"]))
        board = state.athena.board
        shown_task: str | None = None
        for i, f in enumerate(files):
            if f.task_id != shown_task:
                shown_task = f.task_id
                task = board.tasks.get(f.task_id) if board is not None else None
                glyph = getattr(g, TASK_GLYPH.get(task.state, "running")) if task else g.running
                title = task.title if task else ""
                lines.append(Content.styled(f"{glyph} {f.task_id} {title}", f"bold {c['fg']}"))
            counts = _counts(f.added, f.removed, g, c)
            row = Content.assemble(f"  {f.path}  ", counts)
            if i == self.chosen:
                row = Content.styled(row.plain, f"bold {c['ink']} on {c['worker']}")
            lines.append(row)
        lines += [Content(""), self._hint(state)]
        return Content("\n").join(lines)

    def _hint(self, state: TuiState) -> Content:
        return key_hint(
            state, self._colours, _HINT, view=ScreenView.DIFF, hunks=self.hunks_shown
        )

    def _hunks_text(self, state: TuiState, chosen: ChangedFile | None) -> Content:
        c = self._colours
        if chosen is None:
            return Content.styled("no file changed yet", c["dim"])
        if self.scope is DiffScope.TASK and state.task_diff.text is None:
            return Content.styled(NO_LINES, c["dim"])  # a server that does not build it
        if self.scope is DiffScope.ALL and state.handoff is None:
            not_served = StaleRoute.HANDOFF.value in state.not_served
            return Content.styled(NOT_SERVED if not_served else NOT_READ, c["dim"])
        hunks = self.file_hunks()
        if hunks is None or not hunks.hunks:
            notes = list(hunks.notes) if hunks is not None else []
            gone = "no line of it changed" if self.scope is DiffScope.TASK else "not committed"
            return Content.styled("\n".join([*notes, f"{gone}: {NO_LINES}"]), c["dim"])
        self.hunk = min(self.hunk, len(hunks.hunks) - 1)
        lines = [Content.styled(n, c["dim"]) for n in hunks.notes]
        for i, h in enumerate(hunks.hunks):
            lines += hunk_lines(h, self._colours, chosen=i == self.hunk)
        return Content("\n").join(lines)

    def _foot_text(self, state: TuiState, chosen: ChangedFile | None) -> Content:
        """The chosen task's facts, and whether it is committed: the board's word, never the
        change's verdict (invariant 7)."""
        c = self._colours
        if chosen is None:
            return Content("")
        worker = state.worker.tasks.get(chosen.task_id)
        facts = worker.facts_line if worker is not None else None
        lines: list[Content] = [Content.styled("─" * 40, c["faint"])]
        if facts is not None:
            lines.append(
                Content.assemble(
                    ("facts  ", c["dim"]),
                    syntax_word(facts, c),
                    ("  ·  ", c["dim"]),
                    tests_word(facts, c),
                )
            )
        board = state.athena.board
        task = board.tasks.get(chosen.task_id) if board is not None else None
        if chosen.committed:
            lines.append(Content.styled(f"{chosen.task_id} is committed.", c["dim"]))
        elif task is not None and task.state in SETTLED:
            # Settled without a commit (dropped, re-cut): none of it is coming, so no "yet".
            text = f"{chosen.task_id} was {task.state}: nothing of it is committed."
            lines.append(Content.styled(text, c["dim"]))
        else:
            running = live(state) and state.worker.current == chosen.task_id
            where = "This is the live worktree. " if running else ""
            text = f"{where}Nothing is committed for {chosen.task_id} yet."
            lines.append(Content.styled(text, c["dim"]))
        return Content("\n").join(lines)

    def _retitle(self, chosen: ChangedFile | None) -> None:
        c, g = self._colours, self._glyphs
        files = self.files()
        known = [f for f in files if f.added is not None and f.removed is not None]
        total = _counts(
            sum(f.added or 0 for f in known) if len(known) == len(files) and files else None,
            sum(f.removed or 0 for f in known) if len(known) == len(files) and files else None,
            g,
            c,
        )
        self.changed.border_title = border_title(
            Content.styled("CHANGED", f"bold {c['worker']}"),
            total,
            self.changed.outer_size.width,
            double=False,
        )
        if chosen is None:
            self.hunks.border_title = Content.styled("no file", c["dim"])
            return
        where = "committed, from base" if self.scope is DiffScope.ALL else chosen.task_id
        left = Content.assemble((chosen.path, f"bold {c['worker']}"), (f" · {where}", c["fg"]))
        right = _counts(chosen.added, chosen.removed, g, c)
        worker = self.state.worker.tasks.get(chosen.task_id) if self.state else None
        step = worker.last_step(chosen.path) if worker is not None else None
        if step is not None:
            right = Content.assemble(right, (f" · step «{step}", c["dim"]))
        self.hunks.border_title = border_title(
            left, right, self.hunks.outer_size.width, double=True
        )

    def on_resize(self, _event: Resize) -> None:
        if self.state is not None:
            self.show(self.state)

    # -- its keys --------------------------------------------------------------------------

    def action_next_file(self) -> None:
        files = self.files()
        if files:
            self.chosen = (self.chosen + 1) % len(files)
            self.hunk = 0
            self._redraw()

    def action_hunk(self, step: int) -> None:
        hunks = self.file_hunks()
        if hunks is None or not hunks.hunks:
            return
        self.hunk = (self.hunk + step) % len(hunks.hunks)
        self._redraw()

    def action_toggle_scope(self) -> None:
        self.scope = DiffScope.ALL if self.scope is DiffScope.TASK else DiffScope.TASK
        self.chosen = self.hunk = 0
        self._redraw()

    def action_close_diff(self) -> None:
        self.post_message(self.Closed())

    def _redraw(self) -> None:
        if self.state is not None:
            self.show(self.state)
        self.post_message(self.Moved())

    class Moved(Message):
        """The choice moved: the bar redraws, since `[ ]` applies only where hunks show."""

    class Closed(Message):
        """`Esc`: back to layout E."""


def _counts(
    added: int | None, removed: int | None, glyphs: Glyphs, colours: dict[str, str]
) -> Content:
    if added is None or removed is None:
        return Content("")
    return Content.assemble(
        (f"+{added}", colours["worker"]), " ", (f"{glyphs.minus}{removed}", colours["bad"])
    )
