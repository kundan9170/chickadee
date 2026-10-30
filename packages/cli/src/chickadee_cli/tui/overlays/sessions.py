"""The sessions (`/resume`): this server's sessions, each with its
spend and state, and a preview of the chosen one's last run.

**Read as it opens, never held by the screen.** The list is `sessions()`; each row's spend
and state are its own `/budget` and `/liveness`, read once per row and unknown until answered;
the preview is the chosen session's last run folded by the board's own fold (`fold_run`, the
same as Athena's board), read when it is first chosen. When a row was written is its record's
time (`updated_at`), never an age reckoned from this machine's clock.

**`Enter` on this session continues it** (`/resume`, the REPL's); **on another, the screen
reopens on that session**, from where `/resume` continues it. `f` (fork) is
dimmed: a fork needs a seq, a reason and whether to restore the code, which `chickadee fork`
asks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Final, Protocol

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal
from textual.content import Content
from textual.widgets import Input, Static

from chickadee_cli.render.board import RunView, fold_run
from chickadee_cli.tui.overlays import (
    CLOSE,
    Host,
    Overlay,
    OverlayKey,
    bindings,
    fit,
    money,
    row,
)
from chickadee_cli.tui.theme import TASK_GLYPH
from chickadee_wire.models import Liveness, SessionBudget, SessionInfo

__all__ = ["READING", "SessionList", "SessionsHost", "when"]

READING: Final = "reading…"
_LIST_SHARE: Final = 0.6


class SessionsHost(Host, Protocol):
    def switch_session(self, session_id: str) -> None:
        """Reopen the screen on `session_id`."""
        ...


def when(stamp: str) -> str:
    """A record's time as the list shows it: its date and minute (`2026-10-23 10:04`)."""
    return stamp.replace("T", " ")[:16] if stamp else "unknown"


class SessionList(Overlay):
    WORD = "SESSIONS"
    COMMAND = "/resume"
    WIDTH = 83
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("up", "down"), "Up/Down", "choose", ("move(-1)", "move(1)")),
        OverlayKey(("enter",), "Enter", "resume", ("resume",)),
        OverlayKey(("f",), "f", "fork", builds_in="chickadee fork"),
        OverlayKey(("slash",), "/", "filter", ("filter",)),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def __init__(self, host: SessionsHost) -> None:
        super().__init__(host)
        self.switcher = host
        self.listed: tuple[SessionInfo, ...] | None = None
        self.failed: str | None = None
        self.budgets: dict[str, SessionBudget | str] = {}
        """By session, its `/budget`, or why it could not be read."""
        self.lives: dict[str, Liveness | str] = {}
        self.runs: dict[str, RunView | str] = {}
        """By run, its board, or why its journal could not be read."""

    def head(self) -> ComposeResult:
        """the mockup's `> filter` row, always drawn; `/` puts the keys in it."""
        with Horizontal(id="prompt"):
            yield Static(">")  # the composer's own prompt mark, as the palette's
            yield Input(placeholder="filter", id="filter")

    def on_mount(self) -> None:
        super().on_mount()
        self.host.read(lambda c: c.sessions(), self.loaded, self.list_failed)

    def right(self) -> str:
        return "" if self.listed is None else f"{len(self.listed)} on this server"

    # -- what it reads ---------------------------------------------------------------------

    def loaded(self, sessions: tuple[SessionInfo, ...]) -> None:
        self.listed = tuple(sorted(sessions, key=lambda s: s.updated_at, reverse=True))
        for info in self.listed:
            sid = info.session_id
            self.host.read(
                lambda c, s=sid: c.budget(s),
                self._into(self.budgets, sid),
                self._why(self.budgets, sid),
            )
            self.host.read(
                lambda c, s=sid: c.liveness(s),
                self._into(self.lives, sid),
                self._why(self.lives, sid),
            )
        self._preview()
        self.redraw()

    def list_failed(self, why: str) -> None:
        self.failed = why
        self.redraw()

    def _into[T](self, held: dict[str, T | str], key: str) -> Callable[[T], None]:
        def put(value: T) -> None:
            held[key] = value
            self.redraw()

        return put

    def _why[T](self, held: dict[str, T | str], key: str) -> Callable[[str], None]:
        def put(why: str) -> None:
            held[key] = why
            self.redraw()

        return put

    def _preview(self) -> None:
        chosen = self.chosen()
        if chosen is None or chosen.last_run_id is None or chosen.last_run_id in self.runs:
            return
        run_id = chosen.last_run_id
        self.runs[run_id] = READING
        self.host.read(
            lambda c: fold_run(run_id, c.run_journal(run_id)),
            self._into(self.runs, run_id),
            self._why(self.runs, run_id),
        )

    # -- what it shows ---------------------------------------------------------------------

    def shown(self) -> tuple[SessionInfo, ...]:
        if self.listed is None:
            return ()
        typed = (
            self.query_one("#filter", Input).value.strip().lower() if self.is_mounted else ""
        )
        return tuple(
            s
            for s in self.listed
            if typed in s.session_id.lower() or typed in (s.title or "").lower()
        )

    def rows(self) -> int:
        return len(self.shown())

    def chosen(self) -> SessionInfo | None:
        shown = self.shown()
        return shown[self.cursor] if 0 <= self.cursor < len(shown) else None

    def action_move(self, step: int) -> None:
        super().action_move(step)
        self._preview()

    def cursor_line(self) -> int:
        return self.cursor * 3

    def body(self, width: int) -> Content:
        c = self.colours
        if self.listed is None:
            return Content.styled(self.failed or READING, c["bad"] if self.failed else c["dim"])
        list_w = max(int(width * _LIST_SHARE), 24)
        left = self._list(list_w)
        right = self._preview_lines(max(width - list_w - 3, 10))
        lines: list[Content] = []
        for at in range(max(len(left), len(right))):
            on_left = left[at] if at < len(left) else Content("")
            # Every row padded to the list's width, so the preview's rule stays one column.
            on_left = Content.assemble(on_left, " " * max(list_w - cell_len(on_left.plain), 0))
            on_right = right[at] if at < len(right) else Content("")
            lines.append(Content.assemble(on_left, (" │ ", c["faint"]), on_right))
        return Content("\n").join(lines)

    def _list(self, width: int) -> list[Content]:
        c = self.colours
        here = self.host.session_id
        lines: list[Content] = []
        for at, info in enumerate(self.shown()):
            chosen = at == self.cursor
            title = Content.assemble(
                (
                    f"{info.session_id}  ",
                    f"bold {c['zeus']}" if info.session_id == here else c["fg"],
                ),
                fit(info.title or "(untitled)", max(width - len(info.session_id) - 2, 4)),
            )
            budget = self.budgets.get(info.session_id, READING)
            spend = money(budget.spent_usd if isinstance(budget, SessionBudget) else None)
            state = self._state(info.session_id)
            detail = Content.assemble(
                (f"{when(info.updated_at)} · {spend} · ", c["dim"]), state
            )
            lines.append(row(title, width, c, chosen=chosen))
            lines.append(
                row(
                    Content(fit(detail.plain, width)) if chosen else detail,
                    width,
                    c,
                    chosen=chosen,
                )
            )
            lines.append(Content(" " * width))
        if not lines:
            lines.append(Content.styled(fit("no session matches", width), c["dim"]))
        return lines

    def _state(self, sid: str) -> Content:
        c = self.colours
        live = self.lives.get(sid, READING)
        if not isinstance(live, Liveness):
            return Content.styled("state unknown", c["dim"])
        word = live.kind if live.outcome is None else f"{live.kind}: {live.outcome}"
        colour = {"working": c["worker"], "awaiting": c["zeus"], "abandoned": c["bad"]}.get(
            live.kind, c["dim"]
        )
        return Content.styled(word, colour)

    def _preview_lines(self, width: int) -> list[Content]:
        c, glyphs = self.colours, self.host.glyphs
        chosen = self.chosen()
        if chosen is None:
            return []
        lines = [
            Content.assemble((chosen.session_id, f"bold {c['zeus']}"), (" preview", c["dim"]))
        ]
        lines.append(Content(""))
        if chosen.last_run_id is None:
            lines.append(Content.styled("no run yet", c["dim"]))
        else:
            view = self.runs.get(chosen.last_run_id, READING)
            lines.append(Content.styled("tasks", c["dim"]))
            if isinstance(view, str):
                lines.append(Content.styled(fit(view, width), c["dim"]))
            else:
                for task in view.tasks.values():
                    glyph = getattr(glyphs, TASK_GLYPH.get(task.state, "waiting"))
                    lines.append(
                        Content.assemble(
                            (f"{glyph} ", c["worker"]),
                            fit(f"{task.task_id} {task.title}", width - 2),
                        )
                    )
                if not view.tasks:
                    lines.append(Content.styled("none delegated", c["dim"]))
        live = self.lives.get(chosen.session_id)
        if isinstance(live, Liveness) and (live.outcome or live.why):
            lines += [
                Content(""),
                Content.styled(fit(live.outcome or live.kind, width), c["zeus"]),
            ]
            if live.why:
                lines.append(Content.styled(live.why, c["fg"]))
        return lines

    # -- keys ------------------------------------------------------------------------------

    def applies(self, action: str) -> bool:
        if action == "resume":
            return self.chosen() is not None
        return True

    def action_filter(self) -> None:
        self.query_one("#filter", Input).focus()

    def on_input_changed(self, _event: Input.Changed) -> None:
        self.cursor = 0
        self._preview()
        self.redraw()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self.scroll.focus()  # back to the list's keys, the filter kept

    def action_resume(self) -> None:
        chosen = self.chosen()
        if chosen is None:
            return
        here = self.host.session_id
        self.dismiss()
        if chosen.session_id == here:
            self.host.command("/resume")
        else:
            self.switcher.switch_session(chosen.session_id)
