"""The diff: one view
of what a task, or the run, changed, at two sizes (one concept, one home).

- **The centre pane** is the view at `DiffScope.TASK`, compact and live: the task the worker
  pane shows, its files with their lines added and removed, its last failing test.
- **The `Ctrl+D` screen** (Part 6) is the same view full screen, with the file list and the
  "this task / all tasks" switch; "all tasks" reuses `render/diff.py`'s rule for which files a
  task shows, not a second one.

A file is its name, its counts and its hunks, as the server read the task's diff; a server that
does not build it leaves the counts the round's facts and says
`render/diff.py`'s `NO_LINES`. `hunk_lines` draws a hunk for both sizes.

**As the screen first opens** (while nothing
is said and no run is open, `TuiState.landing`) the pane is the screen's front page: the bird,
the name, and the server's recent sessions; where it runs and the models are the header's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Final

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.events import Click, Resize
from textual.message import Message
from textual.style import Style
from textual.widgets import Input, Static

from chickadee_cli import __version__
from chickadee_cli.prompt import ranked_paths
from chickadee_cli.render.board import STATES
from chickadee_cli.render.diff import NO_LINES, shown_line
from chickadee_cli.tui.brand import front_mark
from chickadee_cli.tui.diffs import DiffHunk, split_diff
from chickadee_cli.tui.feed import SETTLED
from chickadee_cli.tui.keys import CODE_KEYS, CODE_OFF_KEYS, FRONT_KEYS, lit
from chickadee_cli.tui.model import DiffFile, DiffView, TuiState
from chickadee_cli.tui.panes import RoleFrame
from chickadee_cli.tui.panes.athena import waiting
from chickadee_cli.tui.statusbar import key_hint, pane_hint
from chickadee_cli.tui.theme import TASK_GLYPH, UNICODE, UNKNOWN, Glyphs
from chickadee_wire.models import FileText, SessionInfo

__all__ = [
    "FOUND_SHOWN",
    "RECENT_SHOWN",
    "DiffFrame",
    "FindBox",
    "StripTask",
    "chosen_file",
    "diff_body",
    "file_lines",
    "file_tabs",
    "found_lines",
    "launch_body",
    "live",
    "strip_tasks",
    "task_strip",
]


_HINT: Final = (("ctrl+d", "full diff"), ("ctrl+u", "undo task"))
"""The CODE pane's keys (`ui/` `codeRun`), drawn from the bar's table, lit or dimmed."""
_RULES: Final = 8
"""The idle pane's faint ruled lines, where the diff will be."""
_PREV: Final = "\u2039"
_NEXT: Final = "\u203a"
"""The tab row's ends, which say `←`/`→` move between the files."""
FOUND_SHOWN: Final = 6
FOUND_WIDTH: Final = 52
"""The search's matches shown over the diff."""
RECENT_SHOWN: Final = 3
"""The front page's recent sessions, as the mockup lists them; `/resume` lists all."""


def front_sessions(
    state: TuiState, recent: tuple[SessionInfo, ...] | None
) -> list[SessionInfo]:
    """The sessions the front page lists, and the ones Up and Down choose among: the
    recent sessions other than this one, newest first, `RECENT_SHOWN` at most. A session with
    no title, neither a name nor words said, is left out:
    every screen makes one as it opens, continuing the open work where there is some, so
    earlier launches' would fill the rows with nothing to go back to, and the screen's own is
    one."""
    if recent is None:
        return []
    here = state.info.session_id if state.info is not None else None
    others = [s for s in recent if s.session_id != here and s.title is not None]
    return sorted(others, key=lambda s: s.updated_at, reverse=True)[:RECENT_SHOWN]


FRONT_META: Final = "front_session"
"""The meta key a front-page row carries its session's id under, for `DiffFrame.on_click`.
Plain meta, not `@click`: Textual draws a `@click` span as a link, in its own colour
and underlined, over the row's."""


def launch_body(
    state: TuiState,
    colours: dict[str, str],
    *,
    recent: tuple[SessionInfo, ...] | None,
    pictures: bool,
    width: int,
    height: int = 0,
    chosen: int | None = None,
    glyphs: Glyphs = UNICODE,
) -> Content:
    """The front page: the bird and the name as `width` columns fit them
    (`brand.front_mark`; `pictures` false where the half blocks cannot be drawn), then the
    recent sessions, newest first, centred top to bottom in `height` where it fits.
    Side to side, the mark is centred on `width`, and the sessions and the hint one block,
    centred and left-aligned: a session wider than the mark no longer pushes the mark
    left of centre. The row `chosen` (`front_sessions`' index) is marked with `glyphs.call`
    and its id in Zeus's colour; every row sits two columns in, so a choice moves nothing.
    A click on a row opens it. Where the session runs, its tree, its
    sandbox and each role's model are the header's, said once."""
    dim, fg = colours["dim"], colours["fg"]
    mark = front_mark(colours, width=width, pictures=pictures)
    lines = [Content(""), Content.styled("RECENT", dim)]
    newest = front_sessions(state, recent)
    if recent is None:
        lines.append(Content.styled("reading the sessions…", dim))
    elif not newest:
        lines.append(Content.styled("none yet", dim))
    for at, info in enumerate(newest):
        when = info.updated_at.replace("T", " ")[:16] or UNKNOWN
        picked = at == chosen
        row = Content.assemble(
            (f"{glyphs.call} " if picked else "  ", colours["zeus"]),
            (f"{info.session_id}  ", f"bold {colours['zeus']}" if picked else fg),
            (f"{info.title} · {when}", dim),
        )
        lines.append(row.stylize(Style.from_meta({FRONT_META: info.session_id})))
    hint = key_hint(state, colours, (("slash", "commands"), ("question_mark", "help")))
    if newest:
        keys = Content.assemble(
            ("Up/Down" if glyphs.mark.isascii() else "↑↓", colours["zeus"]),
            (" choose · ", dim), ("Enter", colours["zeus"]), (" open", dim),
        )  # fmt: skip
        hint = Content.styled(" · ", dim).join([keys, hint])
    lines += [Content(""), hint]
    left = max((width - max(line.cell_length for line in lines)) // 2, 0)
    # The mark's rows come centred on its widest (`brand.front_mark`): that one is centred here.
    centre = max((width - max(row.cell_length for row in mark)) // 2, 0)
    rows = [Content(" " * centre) + row for row in mark]
    rows += [Content(" " * left) + line for line in lines]
    top = max((height - len(rows)) // 2, 0)
    return Content("\n").join([Content("")] * top + rows)


def diff_body(
    view: DiffView,
    glyphs: Glyphs,
    colours: dict[str, str],
    state: TuiState | None = None,
    *,
    path: str | None = None,
) -> Content:
    """Below the strip and the tabs: the chosen file's hunks (`path`, else the task's first),
    `render/diff.py`'s sentence where the wire carries none, the last failing test and its
    first line of output, and the steps to go back to (`ui/` `codeRun`)."""
    dim, bad = colours["dim"], colours["bad"]
    if not view.files:
        return Content.styled("no file changed yet", dim)
    shown = chosen_file(view, path)
    lines: list[Content] = []
    hunks = split_diff(view.text).get(shown) if view.text is not None else None
    if hunks is None:
        lines.append(Content.styled(NO_LINES, dim))
    else:
        lines += [Content.styled(n, dim) for n in hunks.notes]
        for hunk in hunks.hunks:
            lines += hunk_lines(hunk, colours)
    if view.failing is not None:
        first = next((ln for ln in view.failing.tail.splitlines() if ln.strip()), "")
        lines += [Content("")]
        lines += [
            Content.styled(f"{glyphs.dropped} {name}", bad) for name in view.failing.failing
        ]
        if first:
            lines.append(Content.styled(first.strip(), dim))
    if view.steps:
        lines += [Content(""), Content.styled(" · ".join(f"«{n}" for n in view.steps), dim)]
    lines.append(key_hint(state, colours, _HINT))
    return Content("\n").join(lines)


def chosen_file(view: DiffView, path: str | None) -> str:
    """The file shown: `path` while the task has it, else its first."""
    paths = [f.path for f in view.files]
    return path if path in paths else paths[0]


def file_tabs(
    view: DiffView, glyphs: Glyphs, colours: dict[str, str], path: str | None
) -> Content:
    """The open task's files as tabs, between angle marks when there is more than one to move
    to: the chosen one marked, each by its name and its lines when measured."""
    if not view.files:
        return Content("")
    dim, ink = colours["dim"], colours["ink"]
    shown = chosen_file(view, path)
    parts: list[tuple[str, str] | str] = [(f"{_PREV} ", dim)] if len(view.files) > 1 else []
    for f in view.files:
        name = f.path.rsplit("/", 1)[-1]
        on = f.path == shown
        parts.append(
            (f" {name} ", f"bold {ink} on {colours['worker']}" if on else colours["fg"])
        )
        if f.added is not None and f.removed is not None:
            parts += [
                " ",
                (f"+{f.added}", colours["worker"]),
                " ",
                (f"{glyphs.minus}{f.removed}", colours["bad"]),
            ]
        parts.append(" ")
    if len(view.files) > 1:
        parts.append((_NEXT, dim))
    return Content.assemble(*parts)


@dataclass(frozen=True)
class StripTask:
    """One task on the CODE pane's strip: the board's, or the plan's not yet delegated."""

    task_id: str
    title: str
    glyph: str
    started: bool
    """Delegated: it has a diff to open. One the plan lists and no delegation ran has not."""


def strip_tasks(state: TuiState, glyphs: Glyphs) -> list[StripTask]:
    """The run's tasks as the Planner's pane lists them: the board's, then the plan's waiting
    ones (`athena.waiting`), one list for both panes."""
    board = state.athena.board
    tasks = [
        StripTask(t.task_id, t.title, getattr(glyphs, TASK_GLYPH[t.state]), True)
        for t in (board.tasks.values() if board is not None else ())
    ]
    tasks += [StripTask(p.id, p.title, glyphs.waiting, False) for p in waiting(state)]
    return tasks


_SHORT: Final = 12
"""The most of another task's title the strip keeps; the open task's is whole."""


def task_strip(
    state: TuiState, glyphs: Glyphs, colours: dict[str, str], open_task: str | None, width: int
) -> Content:
    """`✔ T1 Build ▶ T2 Range query ○ T3 Use it…`: every task, the open one whole and
    marked, the others' titles cut to fit `width` (their ids always shown)."""
    tasks = strip_tasks(state, glyphs)
    if not tasks:
        return Content("")
    words = {t.task_id: f" {t.glyph} {t.task_id} " for t in tasks}
    for t in tasks:
        if t.task_id == open_task:
            words[t.task_id] = f" {t.glyph} {t.task_id} {t.title} "
    room = width - sum(len(w) + 1 for w in words.values())
    for t in tasks:
        if t.task_id == open_task or room <= 1 or not t.title:
            continue
        short = t.title if len(t.title) <= _SHORT else t.title[: _SHORT - 1] + "…"
        short = short[: room - 1]
        if short:
            words[t.task_id] = f" {t.glyph} {t.task_id} {short} "
            room -= len(short) + 1
    parts: list[tuple[str, str] | str] = []
    for t in tasks:
        if t.task_id == open_task:
            style = f"bold {colours['ink']} on {colours['worker']}"
        else:
            style = colours["fg"] if t.started else colours["dim"]
        parts += [(words[t.task_id], style), " "]
    return Content.assemble(*parts)


def file_lines(read: FileText, colours: dict[str, str]) -> Content:
    """A file no task changed, read only: each line numbered and drawn safe, as a
    diff's are; or why there is no text."""
    dim = colours["dim"]
    if read.text is None:
        return Content.styled(f"{read.path}: {read.withheld or 'no text'}", dim)
    lines = read.text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return Content("\n").join(
        Content.assemble((f"{n:>{NUM_W}}   ", dim), (shown_line(text), colours["fg"]))
        for n, text in enumerate(lines, start=1)
    )


def found_lines(
    found: list[str],
    at: int,
    changed: dict[str, tuple[str, DiffFile]],
    glyphs: Glyphs,
    colours: dict[str, str],
    width: int,
    typed: str = "",
) -> Content:
    """The search's matches over the diff: a file a task changed with its lines and task, any
    other `unchanged`; `at` the one `Enter` opens, and `typed` marked where a name holds it."""
    dim = colours["dim"]
    if not found:
        return Content.styled(" no file matches", dim)
    lines: list[Content] = []
    for i, path in enumerate(found):
        task = changed.get(path)
        if task is not None:
            task_id, f = task
            counts = f"+{f.added} {glyphs.minus}{f.removed}  " if f.added is not None else ""
            tag = Content.assemble((counts, colours["worker"]), (task_id, dim))
        else:
            tag = Content.styled("unchanged", dim)
        room = max(width - cell_len(tag.plain) - 3, 8)
        name = path if len(path) <= room else "…" + path[-(room - 1) :]
        pad = " " * max(width - cell_len(name) - cell_len(tag.plain) - 2, 1)
        style = f"on {colours['btn']}" if i == at else ""
        line = Content.assemble((f" {name}{pad}", f"{colours['fg']} {style}".strip()), tag, " ")
        hit = name.lower().rfind(typed.lower()) if typed else -1
        if hit >= 0:
            line = line.stylize(f"bold {colours['zeus']}", 1 + hit, 1 + hit + len(typed))
        lines.append(line.stylize(style) if style else line)
    lines.append(Content.styled(" ↑↓ choose · Enter open · Esc close", dim))
    return Content("\n").join(lines)


NUM_W: Final = 4
"""A hunk line's number column."""


def hunk_lines(
    hunk: DiffHunk, colours: dict[str, str], *, chosen: bool = False
) -> list[Content]:
    """One hunk as both diff views draw it: its header, then each line numbered as git counts
    it, added and removed on their backgrounds; `chosen` marks the hunk `[`/`]` is on."""
    c = colours
    mark = "▸ " if chosen else "  "
    lines = [Content.styled(f"{mark}{hunk.header}", c["athena"] if chosen else c["dim"])]
    for ln in hunk.lines:
        number = ln.new if ln.new is not None else ln.old
        num = f"{number:>{NUM_W}}" if number is not None else " " * NUM_W
        if ln.kind == "added":
            lines.append(
                Content.styled(f"{num} + {ln.text}", f"{c['worker']} on {c['add_bg']}")
            )
        elif ln.kind == "removed":
            lines.append(Content.styled(f"{num} - {ln.text}", f"{c['bad']} on {c['del_bg']}"))
        elif ln.kind == "context":
            lines.append(Content.assemble((f"{num}   ", c["dim"]), (ln.text, c["fg"])))
        else:
            lines.append(Content.styled(f"{' ' * NUM_W}   {ln.text}", c["dim"]))
    return lines


def live(state: TuiState) -> bool:
    """A task is in flight: its run is open and the board has not settled it."""
    task = state.worker.task
    board = state.athena.board
    if task is None or board is None or not state.episode_open:
        return False
    shown = board.tasks.get(task.task_id)
    return shown is not None and shown.state in STATES - SETTLED


class FindBox(Input):
    """The CODE pane's search box: `↑`/`↓` move the match `Enter` opens."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
    ]

    @dataclass
    class Moved(Message):
        step: int

    def action_move(self, step: int) -> None:
        self.post_message(self.Moved(step))


class DiffFrame(RoleFrame):
    role = "diff"
    colour = "worker"
    KEYS = CODE_KEYS
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(k.key, k.action, show=False)
        for k in (*CODE_KEYS, *CODE_OFF_KEYS, *FRONT_KEYS)
        if k.action is not None
    ]

    @dataclass
    class TaskChosen(Message):
        """The person opened another task: its diff is read if it is not yet."""

        task_id: str

    @dataclass
    class SessionOpened(Message):
        """Enter on the front page's chosen session, with the pane focused."""

        session_id: str

    @dataclass
    class FileChosen(Message):
        """The person chose a file no task changed: the screen reads it."""

        path: str

    DEFAULT_CSS = """
    DiffFrame { border: round $worker; layers: base above; }
    DiffFrame.-idle { border: round $faint; }
    DiffFrame:focus, DiffFrame.-idle:focus, DiffFrame:focus-within { border: double $worker; }
    DiffFrame > #code-head { dock: top; height: auto; display: none; }
    DiffFrame #code-row { height: 1; }
    DiffFrame #code-strip, DiffFrame #code-tabs { text-wrap: nowrap; text-overflow: ellipsis; }
    DiffFrame #code-tabs { width: 1fr; }
    DiffFrame #code-find-hint { width: auto; color: $dim; }
    DiffFrame FindBox {
        width: 24; height: 1; border: none; padding: 0 1; display: none;
        background: $btn; color: $fg;
    }
    DiffFrame FindBox:focus { border: none; background: $btn; }
    DiffFrame FindBox > .input--cursor { background: $zeus; color: $ink; }
    DiffFrame FindBox > .input--placeholder { color: $dim; }
    DiffFrame #code-rule { color: $faint; height: 1; }
    DiffFrame > #code-found {
        layer: above; position: absolute; height: auto;
        border: round $zeus; background: $ink; display: none;
    }
    """

    def __init__(self, colours: dict[str, str], glyphs: Glyphs) -> None:
        super().__init__(colours, glyphs)
        self.recent: tuple[SessionInfo, ...] | None = None
        """The server's sessions, read once for the front page; None until
        answered."""
        self.front_chosen: int | None = None
        """The front page's row Up and Down chose (`front_sessions`' index), if any."""
        self.pictures = False
        """Whether the terminal draws half blocks: the front page's bird and wordmark."""
        self.task_id: str | None = None
        """The task the person opened (`Tab`); None follows the worker's."""
        self.path: str | None = None
        """The file the person chose (`←`/`→`); None is the open task's first."""
        self.opened: FileText | None = None
        """A file no task changed, read for the person; None shows the open task."""
        self.paths: Callable[[], tuple[str, ...]] = tuple
        """The session's files as last read (the `@` list, `prompt.Files.cached`)."""
        self.found: list[str] = []
        """The search's matches while it is open; `found_at` the one `Enter` opens."""
        self.found_at = 0
        self._run_id: str | None = None
        """The run the open task belongs to: a new run drops the person's choice."""
        self.head = Vertical(id="code-head")
        self.strip = Static("", id="code-strip")
        self.tabs = Static("", id="code-tabs")
        self.find_hint = Static("", id="code-find-hint")
        self.find = FindBox(placeholder="find a file", id="code-find")
        self.found_view = Static("", id="code-found")

    def compose(self) -> ComposeResult:
        with self.head:
            yield self.strip
            with Horizontal(id="code-row"):
                yield self.tabs
                yield self.find_hint
                yield self.find
            yield Static("─" * 400, id="code-rule")
        yield from super().compose()
        yield self.found_view

    # -- what is open ----------------------------------------------------------------------

    def open_task(self, state: TuiState) -> str | None:
        """The task shown: the one the person opened while it has started, else the
        worker's."""
        if self.task_id is not None and self.task_id in state.worker.tasks:
            return self.task_id
        return state.worker.current

    def view(self, state: TuiState) -> DiffView:
        task = self.open_task(state)
        return DiffView() if task is None else state.diff_of(task)

    @property
    def finding(self) -> bool:
        return self.find.display

    def takes_escape(self) -> bool:
        """`Esc` is the pane's while its search or a read-only file is open: it closes that,
        and the app's interrupt steps aside (`ChickadeeApp.check_action`)."""
        return self.finding or self.opened is not None

    # -- drawing ---------------------------------------------------------------------------

    def idle(self, state: TuiState) -> bool:
        return state.worker.task is None and self.opened is None

    def show(self, state: TuiState) -> None:
        super().show(state)
        if state.athena.run_id != self._run_id:
            # Another run: its tasks are not the last one's, whatever their ids.
            self._run_id, self.task_id, self.path = state.athena.run_id, None, None
        working = not self.idle(state) or self.finding
        self.head.display = working
        if not working:
            return
        width = max(self.scrollable_content_region.width, 20)
        self.strip.update(
            task_strip(state, self._glyphs, self._colours, self._open_id(state), width)
        )
        if self.opened is not None:
            self.tabs.update(
                Content.assemble(
                    (
                        f" {self.opened.path} ",
                        f"bold {self._colours['ink']} on {self._colours['dim']}",
                    ),
                    ("  unchanged · read only", self._colours["dim"]),
                )
            )
        else:
            self.tabs.update(
                file_tabs(self.view(state), self._glyphs, self._colours, self.path)
            )
        hint = Content.assemble(("⌕ find ", self._colours["dim"]), ("/", self._colours["zeus"]))
        self.find_hint.update(hint)
        self.find_hint.display = not self.finding
        self._draw_found(state)

    def _open_id(self, state: TuiState) -> str | None:
        return None if self.opened is not None else self.open_task(state)

    def title(self, state: TuiState) -> Content:
        c = self._colours
        if self.idle(state) and state.landing:
            return Content.styled("CHICKADEE", f"bold {c['zeus']}")
        if self.idle(state):
            return Content.styled("CODE · idle", c["dim"])
        if self.opened is not None:
            return Content.assemble(
                ("CODE", f"bold {c['worker']}"), (f" · {self.opened.path}", c["fg"])
            )
        task = self.open_task(state)
        word = "live" if live(state) and task == state.worker.current else "last task"
        if task != state.worker.current:
            word = "chosen"
        return Content.assemble(
            ("CODE", f"bold {c['worker']}"), (f" · {task} · {word}", c["fg"])
        )

    def label(self, state: TuiState) -> Content:
        c = self._colours
        if self.idle(state) and state.landing:
            return Content.styled(f"v{__version__}", c["dim"])
        if self.opened is not None:
            return Content.styled("read only", c["dim"])
        lines = self.view(state).lines
        if lines is None:
            return Content("")
        added, removed = lines
        return Content.assemble(
            (f"+{added}", c["worker"]), " ", (f"{self._glyphs.minus}{removed}", c["bad"])
        )

    def content(self, state: TuiState) -> Content:
        c = self._colours
        if self.opened is not None:
            back = Content.assemble(
                ("←→", c["zeus"]), (" back to the task's files · ", c["dim"]),
                ("Esc", c["zeus"]), (" back to the diff", c["dim"]),
            )  # fmt: skip
            return Content("\n").join([file_lines(self.opened, c), Content(""), back])
        body = diff_body(self.view(state), self._glyphs, c, state, path=self.path)
        if self.has_focus:
            body = Content("\n").join([body, pane_hint(state, c, self.KEYS)])
        return body

    def idle_text(self, state: TuiState) -> Content:
        c = self._colours
        if state.landing:
            region = self.scrollable_content_region

            def page(width: int) -> Content:
                return launch_body(
                    state,
                    c,
                    recent=self.recent,
                    pictures=self.pictures,
                    width=width,
                    height=region.height,
                    chosen=self.front_chosen,
                    glyphs=self._glyphs,
                )

            body = page(region.width)
            if (
                len(body.plain.splitlines()) > region.height
                and not self.show_vertical_scrollbar
            ):
                # It will scroll, and the scrollbar takes its column from the art's.
                body = page(region.width - self.styles.scrollbar_size_vertical)
            return body
        lines = [
            Content.styled("No changes yet.", c["dim"]),
            Content(""),
            Content.styled(
                "When a task starts, the worker's live diff shows up here, file by file.",
                c["dim"],
            ),
            Content(""),
            *[Content.styled("┈" * 36, c["faint"]) for _ in range(_RULES)],
            Content(""),
            key_hint(state, c, (("ctrl+d", "diff"), ("ctrl+u", "undo task"))),
        ]
        return Content("\n").join(lines)

    def choices(self) -> list[SessionInfo]:
        """The front page's sessions to choose among."""
        return front_sessions(self.state, self.recent)

    def choose(self, move: int) -> None:
        """Move the front page's choice by `move`: Down from none chooses the newest,
        Up from the newest chooses none again, each end holds; 0 drops the choice."""
        rows = len(self.choices())
        at = self.front_chosen
        if move == 0 or rows == 0:
            at = None
        elif at is None:
            at = 0 if move > 0 else None
        else:
            at = at + move
            at = None if at < 0 else min(at, rows - 1)
        if at != self.front_chosen:
            self.front_chosen = at
            self.show(self.state)

    def chosen_session(self) -> str | None:
        """The id of the front page's chosen session, if one is chosen."""
        rows = self.choices()
        at = self.front_chosen
        return rows[at].session_id if at is not None and at < len(rows) else None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        # The front page's keys apply only there: past it, Up, Down and Enter are not taken.
        for key in FRONT_KEYS:
            if key.action is not None and key.action.split("(")[0] == action:
                return lit(key, self.state) and bool(self.choices())
        return super().check_action(action, parameters)

    def action_front(self, move: int) -> None:
        self.choose(move)

    def action_front_open(self) -> None:
        chosen = self.chosen_session()
        if chosen is not None:
            self.post_message(self.SessionOpened(chosen))

    def on_click(self, event: Click) -> None:
        """A click on a front-page row opens its session, read from the cell's meta."""
        session_id = event.style.meta.get(FRONT_META)
        if isinstance(session_id, str) and self.state.landing:
            event.stop()
            self.post_message(self.SessionOpened(session_id))

    def on_resize(self, _event: Resize) -> None:
        # The front page's art is laid out for the pane's width, so a new width draws it
        # again; so is the strip, which cuts titles to fit. `RoleFrame.on_resize` runs too.
        if not self.idle(self.state) or self.state.landing:
            self.show(self.state)

    # -- keys ------------------------------------------------------------------------------

    def action_next_task(self, step: int) -> None:
        state = self.state
        started = [t.task_id for t in strip_tasks(state, self._glyphs) if t.started]
        started = [t for t in started if t in state.worker.tasks]
        if not started:
            return
        here = self.open_task(state)
        at = started.index(here) if here in started else 0
        self.task_id = started[(at + step) % len(started)]
        self.path, self.opened = None, None
        self.post_message(self.TaskChosen(self.task_id))
        self.show(state)

    def action_next_file(self, step: int) -> None:
        if self.opened is not None:
            self.opened = None  # ←/→ goes back to the task's files
            self.show(self.state)
            return
        view = self.view(self.state)
        paths = [f.path for f in view.files]
        if not paths:
            return
        here = chosen_file(view, self.path)
        self.path = paths[(paths.index(here) + step) % len(paths)]
        self.scroll_home(animate=False)
        self.show(self.state)

    def action_find(self) -> None:
        self.find.display = True
        self.find.value = ""
        self.find.focus()
        self._search("")
        self.show(self.state)

    def action_close(self) -> None:
        """`Esc`: the search closes first, then a read-only file."""
        if self.finding:
            self.find.display = False
            self.found = []
            self.focus()
        elif self.opened is not None:
            self.opened = None
        self.show(self.state)

    def show_file(self, read: FileText) -> None:
        """A file no task changed, as the screen read it."""
        self.opened = read
        self.scroll_home(animate=False)
        self.show(self.state)

    # -- the search ---------------------------------------------------------------

    def changed(self, state: TuiState) -> dict[str, tuple[str, DiffFile]]:
        """Each file a started task changed, by path, to that task and its counts (a later
        task's wins: it is the file as it now stands)."""
        out: dict[str, tuple[str, DiffFile]] = {}
        for task in strip_tasks(state, self._glyphs):
            if task.started:
                for f in state.diff_of(task.task_id).files:
                    out[f.path] = (task.task_id, f)
        return out

    def _search(self, text: str) -> None:
        changed = self.changed(self.state)
        listed = self.paths()
        known = set(listed)
        paths = (*listed, *(p for p in changed if p not in known))
        self.found = (
            ranked_paths(text, paths)[:FOUND_SHOWN] if text else list(changed)[:FOUND_SHOWN]
        )
        self.found_at = 0

    def _draw_found(self, state: TuiState) -> None:
        shown = self.finding
        self.found_view.display = shown
        if shown:
            # Under the search box, against the pane's right edge, as wide as fits.
            inner = self.scrollable_content_region.width
            width = min(FOUND_WIDTH, max(inner - 1, 20))
            self.found_view.styles.width = width
            self.found_view.styles.offset = (max(inner - width, 0), 2)
            self.found_view.update(
                found_lines(
                    self.found,
                    self.found_at,
                    self.changed(state),
                    self._glyphs,
                    self._colours,
                    width - 2,
                    self.find.value,
                )
            )

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input is self.find:
            self._search(event.value)
            self._draw_found(self.state)

    def on_find_box_moved(self, event: FindBox.Moved) -> None:
        if self.found:
            self.found_at = (self.found_at + event.step) % len(self.found)
            self._draw_found(self.state)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input is not self.find or not self.found:
            return
        path = self.found[self.found_at]
        self.action_close()
        task = self.changed(self.state).get(path)
        if task is not None:
            # A changed file opens its task on that file's tab.
            self.task_id, self.path, self.opened = task[0], path, None
            self.post_message(self.TaskChosen(task[0]))
            self.show(self.state)
        else:
            self.post_message(self.FileChosen(path))
