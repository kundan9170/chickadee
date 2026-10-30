"""The budget (`/budget`): the session family's spend against its
cap, by rung and by task, and the person's cap commands.

**Every number is a record's or a route's.** The total, the cap and each rung's share are
`/budget`'s, read again as the overlay opens; each task's cost
is the board's (`fold_run`). Unknown is said, never $0. The warning levels are marked under the
bar (`SessionBudget.warning_fractions`), and the spend is shared by role
(`roles`, decision 6): Zeus's conversation (the design), Athena's planning and reviews, and
the worker's tasks. `w` would change the levels, which needs a config route (not built): drawn
dimmed.

**The cap is the person's**. `c` and `r` take an amount and send `/budget`'s own
line, so its refusals and its question (a cap below what is spent asks first) are the REPL's;
`l` lifts the cap only when pressed twice, since a lifted cap no longer stops a run.
"""

from __future__ import annotations

from typing import ClassVar, Final

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.content import Content
from textual.widgets import Input, Static

from chickadee_cli.tui.model import StaleRoute
from chickadee_cli.tui.overlays import (
    CLOSE,
    Host,
    Overlay,
    OverlayKey,
    bindings,
    chip,
    fit,
    money,
)
from chickadee_cli.tui.theme import TASK_GLYPH, meter

__all__ = ["BAR_W", "LIFT_AGAIN", "Budget"]

BAR_W: Final = 40
_ROLE_BAR_W: Final = 14
LIFT_AGAIN: Final = "l again lifts the cap for this session family; its spend is still counted"
_ROLE_WORDS: Final = {
    "zeus": "the design, talking",
    "athena": "planning, reviews",
    "worker": "the tasks",
}
"""What each role's share paid for (the cast): the picture's "design and planning"."""
_ROLE_ORDER: Final = {"zeus": 0, "athena": 1, "worker": 2}
_ASKS: Final = {
    "cap": "the family's cap in $, or off",
    "rung": "a rung and its cap in $, or off (mid-param 1.50)",
}


class Budget(Overlay):
    WORD = "BUDGET"
    COMMAND = "/budget"
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("c",), "c", "set cap", ("ask('cap')",)),
        OverlayKey(("r",), "r", "rung cap", ("ask('rung')",)),
        OverlayKey(("l",), "l", "lift for this session", ("lift",)),
        OverlayKey(("w",), "w", "warn levels", builds_in="a later release"),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    DEFAULT_CSS = """
    Budget #actions { height: auto; border-top: solid $faint; }
    Budget #amount { display: none; }
    """

    def __init__(self, host: Host) -> None:
        super().__init__(host)
        self.asking: str | None = None
        """Which amount the input takes (`cap`, `rung`), while it is open."""
        self.lifting = False
        """`l` was pressed once: a second press lifts the cap."""
        self.actions = Static("", id="actions")
        self.amount = Input(id="amount")

    def tail(self) -> ComposeResult:
        yield self.actions
        yield self.amount

    def on_mount(self) -> None:
        super().on_mount()
        self.host.refresh_routes({StaleRoute.BUDGET})

    def right(self) -> str:
        return f"session family {self.host.session_id}"

    # -- what it shows ---------------------------------------------------------------------

    def body(self, width: int) -> Content:
        c = self.colours
        budget = self.state.header.spend.budget
        lines: list[Content] = []
        if budget is None:
            why = (
                "not served here"
                if StaleRoute.BUDGET in self.state.not_served
                else "reading /budget…"
            )
            lines.append(Content.assemble(("TOTAL    ", f"bold {c['fg']}"), (why, c["dim"])))
        else:
            if budget.cap_usd is None:
                cap = "no cap: lifted for this session family, still counted"
            else:
                cap = f"of ${budget.cap_usd:.2f} hard cap (from {budget.cap_source})"
            lines.append(
                Content.assemble(
                    ("TOTAL    ", f"bold {c['fg']}"),
                    (money(budget.spent_usd), f"bold {c['fg']}"),
                    f" {cap}",
                )
            )
            lines.append(self._bar(budget.spent_usd, budget.cap_usd, BAR_W, c["zeus"]))
            if budget.cap_usd is not None:
                lines.append(self._marks(budget.warning_fractions or []))
        lines += [Content(""), *self._by_role(), Content(""), *self._by_rung()]
        lines += [Content(""), *self._by_task(width)]
        lines += [
            Content(""),
            Content.styled(
                "At the cap the run stops and keeps a handoff you can resume.", c["dim"]
            ),
        ]
        return Content("\n").join(lines)

    def _bar(self, spent: float, cap: float | None, width: int, colour: str) -> Content:
        glyphs = self.host.glyphs
        if cap is None or cap <= 0:
            return Content.styled(glyphs.bar_empty * width, self.colours["faint"])
        return meter(spent / cap, width, glyphs, colour, self.colours["faint"])

    def _marks(self, fractions: list[float]) -> Content:
        """The scale under the bar: `0`, each warning level where it falls, then `cap`."""
        c = self.colours
        scale = [" "] * BAR_W
        scale[0] = "0"
        scale[BAR_W - 3 :] = list("cap")
        for fraction in fractions:
            label = f"{round(fraction * 100)}%"
            at = min(max(round(BAR_W * fraction) - len(label) // 2, 2), BAR_W - 4 - len(label))
            if all(ch == " " for ch in scale[at - 1 : at + len(label) + 1]):
                scale[at : at + len(label)] = list(label)
        text = "".join(scale)
        return Content.assemble((text[:-3], c["dim"]), (text[-3:], c["bad"]))

    def _by_role(self) -> list[Content]:
        c = self.colours
        budget = self.state.header.spend.budget
        lines = [Content.styled("BY ROLE", f"bold {c['fg']}")]
        roles = (budget.roles or {}) if budget is not None else {}
        if not roles:
            lines.append(Content.styled("not served", c["dim"]))
        for role, spent in sorted(roles.items(), key=lambda kv: _ROLE_ORDER.get(kv[0], 9)):
            said = _ROLE_WORDS.get(role, role)
            lines.append(
                Content.assemble(
                    (fit(role, 8), c.get(role, c["fg"])),
                    (fit(said, 22), c["dim"]),
                    money(spent),
                )
            )
        return lines

    def _by_rung(self) -> list[Content]:
        c = self.colours
        budget = self.state.header.spend.budget
        lines = [
            Content.assemble(
                ("BY RUNG", f"bold {c['fg']}"),
                (" " * (_ROLE_BAR_W + 6), ""),
                ("SPENT      CAP", c["dim"]),
            )
        ]
        rungs = (budget.rungs or {}) if budget is not None else {}
        if not rungs:
            lines.append(Content.styled("not served", c["dim"]))
        for name, rung in sorted(rungs.items()):
            # A rung with no cap of its own is drawn against the family's.
            cap_usd = rung.cap_usd
            if cap_usd is None and budget is not None:
                cap_usd = budget.cap_usd
            cap = "none" if rung.cap_usd is None else f"${rung.cap_usd:.2f}"
            lines.append(
                Content.assemble(
                    (fit(name, 12), c["fg"]),
                    self._bar(rung.spent_usd, cap_usd, _ROLE_BAR_W, c["athena"]),
                    f"  {fit(money(rung.spent_usd), 10)} ",
                    (cap, c["dim"]),
                )
            )
        return lines

    def _by_task(self, width: int) -> list[Content]:
        c, glyphs = self.colours, self.host.glyphs
        lines = [Content.styled("BY TASK", f"bold {c['fg']}")]
        board = self.state.athena.board
        if board is None or not board.tasks:
            lines.append(Content.styled("no task yet", c["dim"]))
            return lines
        for task in board.tasks.values():
            glyph = getattr(glyphs, TASK_GLYPH.get(task.state, "waiting"))
            so_far = " so far" if task.state in ("running", "taken over") else ""
            title = fit(f"{task.task_id} {task.title}", max(min(width - 22, 40), 8))
            lines.append(
                Content.assemble(
                    (f"{glyph} ", c["worker"]), title, f"  ${task.cost_usd:.2f}{so_far}"
                )
            )
        return lines

    def foot_keys(self) -> tuple[OverlayKey, ...]:
        return (CLOSE,)  # the cap's keys are the buttons above it

    def redraw(self) -> None:
        super().redraw()
        if not self.is_mounted:
            return
        c = self.colours
        chips = [chip(k.shown, k.label, c, on=self.lit(k)) for k in self.KEYS[:-1]]
        line = Content("  ").join(chips)
        if self.lifting:
            line = Content.assemble(line, "\n", (LIFT_AGAIN, c["zeus"]))
        self.actions.update(line)

    # -- the cap ---------------------------------------------------------------------------

    def applies(self, action: str) -> bool:
        budget = self.state.header.spend.budget
        if action in ("ask", "lift") and self.asking is not None:
            return False
        if action == "lift":
            return budget is not None and budget.cap_usd is not None
        if action == "ask":
            return budget is not None
        return True

    def action_ask(self, what: str) -> None:
        self.asking = what
        self.lifting = False
        self.amount.placeholder = _ASKS[what]
        self.amount.value = ""
        self.amount.display = True
        self.amount.focus()
        self.redraw()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        words = event.value.split()
        wanted = 1 if self.asking == "cap" else 2
        if len(words) != wanted:
            self.amount.value = ""  # its placeholder says again what it takes
            return
        self.dismiss()
        self.host.command(f"/budget {' '.join(words)}")

    def action_close(self) -> None:
        if self.asking is not None:  # Esc closes the input first
            self.asking = None
            self.amount.display = False
            self.redraw()
            return
        super().action_close()

    def action_lift(self) -> None:
        if not self.lifting:
            self.lifting = True
            self.redraw()
            return
        self.dismiss()
        self.host.command("/budget off")
