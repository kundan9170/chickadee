"""The header: the brand, the session, where
it runs, each role's model, the stage strip, and the spend against its cap. Five rows beside the
bird; a short screen keeps three (`SHORT_BELOW`). What one role spent and how full its prompt is
are its pane's title (`panes.spend_label`), not the header's.

**Full screen** (`^F`, `/fullscreen`) keeps one line and the rule: the name, where it
runs, its uncommitted paths, the stage, the spend (`full_screen_line`). The banner is drawn
under it as ever, so a cap or a lost wire is never hidden by it.

Each line is built by a plain function of the state (`stage_strip`, `spend_lines`,
`where_line`), so what it says is tested without a terminal; the widget only places them.
**What the wire does not carry is left out, never filled in**: no spend yet reads `—`, not
`$0`; the plan's size is `?` until its approved plan lists its tasks; a lifted cap says
so.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.widgets import Static

from chickadee_cli import __version__
from chickadee_cli.tui.brand import NAME, SPACED, bird
from chickadee_cli.tui.modals.stop import stop_name
from chickadee_cli.tui.model import (
    Connection,
    Header,
    HeaderSpend,
    SpendLevel,
    Stage,
    TuiState,
)
from chickadee_cli.tui.names import ROLE_WORDS
from chickadee_cli.tui.theme import STAGE_WORDS, UNKNOWN, Glyphs, meter
from chickadee_wire.models import IndexProgress, RunHeader

__all__ = [
    "BAR_CELLS",
    "INDEX_CELLS",
    "SHORT_BELOW",
    "BrandHeader",
    "banner_line",
    "full_screen_line",
    "header_rows",
    "index_progress",
    "lost_line",
    "models_line",
    "session_line",
    "spend_lines",
    "stage_strip",
    "tree_fact",
    "where_line",
    "where_lines",
]

BAR_CELLS: Final = 20
"""The spend bar's width, as the mockup's."""
INDEX_CELLS: Final = 10
"""The index build's bar, on the repository's row beside its facts: half the spend bar."""
SHORT_BELOW: Final = 28
"""Rows below which the header keeps three of its five (the name, where, the stages) and drops
the bird, the session and the models: a short screen's rows are the panes'."""
_TALL: Final = 6
"""The header's rows on a screen `SHORT_BELOW` or taller: five lines and the rule under them."""
_SHORT: Final = 4
"""On a shorter one: three lines and the rule."""
_FULL_SCREEN: Final = 2
"""In full screen: one line and the rule."""
_LABEL: Final = 9
"""The width of a header row's label (`session`, `repo`, `models`)."""
_ROLES: Final = ("zeus", "athena", "worker")

_ORDER: Final = (Stage.DESIGN, Stage.PLAN, Stage.WORK, Stage.CHECK, Stage.HANDOFF)


def header_rows(screen_height: int, *, full_screen: bool = False) -> int:
    """How many rows the header takes on a screen `screen_height` rows tall, in full screen
    or not: what an overlay leaves above it, so the stage and the spend stay in view."""
    if full_screen:
        return _FULL_SCREEN
    return _SHORT if screen_height < SHORT_BELOW else _TALL


def stage_strip(state: TuiState, glyphs: Glyphs, colours: dict[str, str]) -> Content:
    """`● Design ━━ ● Plan ━━ ▶ WORK 1/? ┈┈ ○ Check ┈┈ ○ Hand off`: done stages green and joined
    by a heavy line, the current one inverted, the rest dim. A one-task plan (Zeus's) skipped
    design and planning."""
    header, athena = state.header, state.athena
    current = _ORDER.index(header.stage)
    one_task = athena.plan_by == "zeus" and athena.plan_state is not None
    green, dim, faint = (colours[k] for k in ("worker", "dim", "faint"))
    parts: list[tuple[str, str] | str] = []
    for i, stage in enumerate(_ORDER):
        word = STAGE_WORDS[stage]
        if i:
            # A join is heavy when the stage before it was done, not skipped.
            done = i <= current and not (one_task and i - 1 <= _ORDER.index(Stage.PLAN))
            parts += [
                " ",
                (glyphs.done_join, green) if done else (glyphs.next_join, faint),
                " ",
            ]
        if i == current:
            parts.append(_chip(header, glyphs, colours))
        elif one_task and stage in (Stage.DESIGN, Stage.PLAN):
            parts.append((f"{glyphs.skipped} {word}", dim))
        elif i < current:
            parts.append((f"{glyphs.in_flight} {word}", green))
        else:
            parts.append((f"{glyphs.waiting} {word}", dim))
    return Content.assemble(*parts)


def _chip(header: Header, glyphs: Glyphs, colours: dict[str, str]) -> tuple[str, str]:
    """The current stage as the strip and the full-screen line both draw it: ` ▶ WORK 1/3 `."""
    word = STAGE_WORDS[header.stage]
    return (
        f" {glyphs.running} {_current(word, header)} ",
        f"bold {colours['ink']} on {colours['zeus']}",
    )


def _current(word: str, header: Header) -> str:
    if header.stage is not Stage.WORK:
        return word.upper()
    planned = "?" if header.work_planned is None else str(header.work_planned)
    return f"{word.upper()} {header.work_done}/{planned}"


def spend_lines(spend: HeaderSpend, glyphs: Glyphs, colours: dict[str, str]) -> list[Content]:
    """`SPEND`, `$0.72 / $5.00 14%` and the bar; amber at a warning, red at the cap."""
    dim, faint = colours["dim"], colours["faint"]
    tone = {SpendLevel.OK: colours["fg"], SpendLevel.WARNING: colours["zeus"]}.get(
        spend.level, colours["bad"]
    )
    bar_colour = colours["bad"] if spend.level is SpendLevel.CAP else colours["zeus"]
    title = Content.styled("SPEND", dim)
    total = spend.total_usd
    if total is None:
        return [
            title,
            Content.styled("—", dim),
            Content.styled(glyphs.bar_empty * BAR_CELLS, faint),
        ]
    if spend.lifted:
        line = Content.assemble((f"${total:.2f}", tone), (" / no cap", dim))
        return [title, line, Content("")]
    cap = spend.cap_usd or 0.0
    share = min(total / cap, 1.0) if cap > 0 else 1.0
    line = Content.assemble((f"${total:.2f}", tone), (f" / ${cap:.2f}  {share:.0%}", dim))
    return [title, line, meter(share, BAR_CELLS, glyphs, bar_colour, faint)]


def where_line(state: TuiState, repo: str | None) -> str:
    """`project-chickadee wt/s-41c branch ccr-a7d2fbb1`, of what is known: the repository's
    name from the host that started the screen, the worktree and branch from the session's
    workspace record."""
    workspace = state.header.workspace
    parts = [repo] if repo else []
    if workspace is not None and workspace.worktree_path:
        parts.append(f"wt/{PurePosixPath(workspace.worktree_path).name}")
    if workspace is not None and workspace.branch:
        parts.append(f"branch {workspace.branch}")
    return "  ".join(parts)


def tree_fact(header: RunHeader, colours: dict[str, str]) -> tuple[str, str]:
    """The checkout's state, from the header's uncommitted paths: its words
    and their tone, as the header's rows and its full-screen line say it."""
    dirty = len(header.uncommitted or [])
    if not dirty:
        return "clean", colours["fg"]
    return f"{dirty} uncommitted path{'s' if dirty != 1 else ''}", colours["zeus"]


def _row(label: str, *value: tuple[str, str] | Content, colours: dict[str, str]) -> Content:
    return Content.assemble((f"{label:<{_LABEL}}", colours["dim"]), *value)


def session_line(state: TuiState, colours: dict[str, str]) -> Content:
    """`session Range-sum queries s-41c`: its title and id, as `/sessions` lists them."""
    info = state.info
    if info is None:
        return _row("session", (UNKNOWN, colours["dim"]), colours=colours)
    return _row(
        "session",
        (info.title or "(untitled)", colours["fg"]),
        (f"  {info.session_id}", colours["dim"]),
        colours=colours,
    )


def where_lines(
    state: TuiState, repo: str | None, glyphs: Glyphs, colours: dict[str, str]
) -> Content:
    """`repo project-chickadee wt/s-41c branch ccr-a7d2 · 2 uncommitted paths · sandbox
    bwrap`, then what could not be read (`lost_line`)."""
    dim, faint = colours["dim"], colours["faint"]
    header = state.run_header
    parts: list[tuple[str, str]] = [(where_line(state, repo) or UNKNOWN, colours["fg"])]
    if header is not None:
        tree, tone = tree_fact(header, colours)
        parts += [(" · ", faint), (tree, tone)]
        if header.sandbox:
            parts += [(" · ", faint), ("sandbox ", dim), (header.sandbox, colours["fg"])]
    live = state.liveness
    if live is not None and live.index_progress is not None:
        # Built beside the person: how far, until it is done.
        return _row(
            "repo",
            *parts,
            (" · ", faint),
            index_progress(live.index_progress, glyphs, colours),
            (lost_line(state), colours["bad"]),
            colours=colours,
        )
    if live is not None and live.indexing is not None:
        # Said with no progress: the build failed, and why, until `chickadee init` is run.
        parts += [(" · ", faint), (live.indexing, colours["bad"])]
    return _row("repo", *parts, (lost_line(state), colours["bad"]), colours=colours)


def index_progress(progress: IndexProgress, glyphs: Glyphs, colours: dict[str, str]) -> Content:
    """`indexing ████░░░░░░ 45% · 820/1830 files` while the files are read; the stage's word
    either side of it, where a share would sit at 0 or 100% while work goes on."""
    gold, dim = colours["gold"], colours["dim"]
    if progress.stage == "preparing":
        return Content.assemble(("indexing", gold), (" · preparing…", dim))
    if progress.stage == "linking":
        return Content.assemble(
            ("indexing", gold), (f" · linking {progress.total} files…", dim)
        )
    share = progress.done / progress.total if progress.total else 0.0
    return Content.assemble(
        ("indexing ", gold),
        meter(share, INDEX_CELLS, glyphs, gold, colours["faint"]),
        (f" {share:.0%} · {progress.done}/{progress.total} files", dim),
    )


def models_line(state: TuiState, colours: dict[str, str]) -> Content:
    """`models designer claude-opus-4-1 · planner claude-opus-4-1 · worker qwen3-coder`, each
    in its role's colour: the model that served the role's latest call, else the one its rung
    is wired to (`TuiState.role_model`), by its provider's name (`TuiState.model_name`), not
    the config's catalog name; `—` for a role neither names yet."""
    dim, faint = colours["dim"], colours["faint"]
    parts: list[tuple[str, str]] = []
    for role in _ROLES:
        if parts:
            parts.append((" · ", faint))
        model = state.role_model(role)
        model = state.model_name(model) if model is not None else None
        parts += [
            (f"{ROLE_WORDS[role].lower()} ", dim),
            (model, colours[role]) if model else (UNKNOWN, dim),
        ]
    return _row("models", *parts, colours=colours)


def full_screen_line(
    state: TuiState, repo: str | None, glyphs: Glyphs, colours: dict[str, str]
) -> Content:
    """`CHICKADEE · project-chickadee wt/s-41c branch ccr-1 · 2 uncommitted paths · ▶ WORK
    1/3`: what stays in full screen, the spend beside it; what was lost, after."""
    faint = colours["faint"]
    parts: list[tuple[str, str]] = [
        (NAME, f"bold {colours['gold']}"),
        (" · ", faint),
        (where_line(state, repo) or UNKNOWN, colours["fg"]),
    ]
    header = state.run_header
    if header is not None and header.uncommitted:
        tree, tone = tree_fact(header, colours)
        parts += [(" · ", faint), (tree, tone)]
    parts += [(" · ", faint), _chip(state.header, glyphs, colours)]
    return Content.assemble(*parts, (lost_line(state), colours["bad"]))


def lost_line(state: TuiState) -> str:
    """What could not be read, said beside where the session runs: a stream lost, or a
    route that failed. Part 7 draws the full notice; this is never left unsaid."""
    if not state.lost:
        return ""
    what, why = next(reversed(state.lost.items()))
    more = f" (+{len(state.lost) - 1} more)" if len(state.lost) > 1 else ""
    return f"  · lost {what}: {why}{more}"


def banner_line(state: TuiState, colours: dict[str, str]) -> Content | None:
    """The banner across the header, or None: the wire lost (the panes below are frozen,
    and what is typed is kept), or the cap crossed (the run's own `hard_cap` record says
    whether it stopped there; `/budget` raises the cap only by the person's command).
    A lost wire is said first: it is why nothing else would move."""
    bad, zeus = colours["bad"], colours["zeus"]
    if state.connection is Connection.LOST:
        what, why = next(reversed(state.lost.items()), ("the wire", "lost"))
        return Content.assemble(
            (" LOST ", f"bold {colours['ink']} on {bad}"),
            (f" {what}: {why}. ", bad),
            ("Nothing more arrives here; what you type is kept. ", colours["fg"]),
            ("chickadee tui --continue opens the session again.", colours["dim"]),
        )
    if state.connection is Connection.RECONNECTING:
        return Content.assemble(
            (" RECONNECTING ", f"bold {colours['ink']} on {zeus}"),
            (f" the session's stream: {state.retrying or 'retrying'}. ", zeus),
            ("It resumes from the last record it read; nothing is missed.", colours["dim"]),
        )
    waiting = state.waiting_stop
    if waiting is not None and waiting.stop is not None:
        # A stop the person put back waits as a banner while they talk to Zeus; the run
        # waits on it, with no timeout while they are here.
        return Content.assemble(
            (f" {stop_name(waiting.stop)} ", f"bold {colours['ink']} on {zeus}"),
            (" waiting for you. ", zeus),
            ("^S opens it; the team waits on it, and Zeus is yours meanwhile.", colours["dim"]),
        )
    capped = state.header.spend.capped
    if capped is None:
        return None
    stopped = (
        "the run stopped there and kept a handoff you can resume"
        if capped.terminal
        else "the run goes on to its next boundary"
    )
    scope = f" for {capped.rung}" if capped.rung else ""
    then = " raises it, then /resume continues" if capped.terminal else " raises it"
    return Content.assemble(
        (" CAP ", f"bold {colours['ink']} on {bad}"),
        (f" ${capped.spent_usd:.2f} of ${capped.cap_usd:.2f}{scope}: {stopped}. ", bad),
        ("/budget", zeus),
        (then, colours["dim"]),
    )


class BrandHeader(Horizontal):
    """Five rows and a rule: the bird, the brand and its lines, and the spend.

    **Narrow or short, it gives way in a fixed order** (the app's `-narrow`, `-single` and
    `-short` classes): below 100 columns the bird goes; below 80 the spend's bar goes and its
    numbers stay (`compact`); below `SHORT_BELOW` rows the session and the models go, and the
    bird with them. No line wraps: what does not fit is cut and marked `…` (invariant 16)."""

    DEFAULT_CSS = """
    BrandHeader { border-bottom: solid $faint; }  /* its height: the app's `header_rows` */
    BrandHeader #bird { width: 16; padding-left: 1; }  /* the bird's 14 and a gap */
    BrandHeader #brand { width: 1fr; }
    BrandHeader #brand Static { text-wrap: nowrap; text-overflow: ellipsis; }
    BrandHeader #line { display: none; }
    BrandHeader #spend { width: 27; }
    """
    # The narrow rules (`-narrow`, `-single`) are the app's CSS: they read the screen's class,
    # which a widget's own (scoped) CSS cannot.

    def __init__(self, glyphs: Glyphs, colours: dict[str, str], *, ascii_only: bool) -> None:
        super().__init__()
        self._glyphs, self._colours, self._ascii = glyphs, colours, ascii_only
        self.compact = False
        """Below 80 columns: the spend's numbers without its bar."""
        self.full_screen = False
        """One line (`full_screen_line`), the app's `-fullscreen` class hiding the rest."""

    def compose(self) -> ComposeResult:
        mark = [Content.styled(self._glyphs.mark, self._colours["zeus"])]
        yield Static(Content("\n").join(mark if self._ascii else bird()), id="bird")
        with Vertical(id="brand"):
            yield Static("", id="line")
            yield Static(
                Content.assemble(
                    (SPACED, f"bold {self._colours['gold']}"),
                    (f"   v{__version__}", self._colours["dim"]),
                ),
                classes="row",
            )
            yield Static("", id="session", classes="row tall")
            yield Static("", id="where", classes="row")
            yield Static("", id="models", classes="row tall")
            yield Static("", id="stages", classes="row")
        yield Static("", id="spend")

    def show(self, state: TuiState, repo: str | None) -> None:
        colours, glyphs = self._colours, self._glyphs
        self.query_one("#session", Static).update(session_line(state, colours))
        self.query_one("#where", Static).update(where_lines(state, repo, glyphs, colours))
        self.query_one("#models", Static).update(models_line(state, colours))
        self.query_one("#stages", Static).update(stage_strip(state, glyphs, colours))
        self.query_one("#line", Static).update(full_screen_line(state, repo, glyphs, colours))
        lines = spend_lines(state.header.spend, glyphs, colours)
        if self.full_screen:
            lines = lines[1:2]  # the numbers, beside the one line
        elif self.compact:
            lines = lines[:2]
        self.query_one("#spend", Static).update(Content("\n").join(lines))
