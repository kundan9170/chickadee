"""Hand off (`/handoff`, and on leaving): the session's worktree
against its base, and the five choices over it (`bring`). Never pushes.

**The REPL's review, drawn.** What it says is `/handoff`'s (`HandoffSummary`), read again as
the overlay opens: the worktree, its commits, the files and lines of its committed diff
(`diffs.split_diff`), what bears on the choice (`notes`), and each choice in `CHOICES`' order,
said in the engine's words (`offers`), one the engine refuses dimmed with why
(`render.diff.offered`, so a choice that would then fail is never offered). As the REPL's
review does, it leaves out the record's `unverified` lines: the owner's call (2026-10-13),
though the picture draws them.

**Choosing carries it out** (`finalize`), and the result is `/handoff <choice>`'s own lines
(`render.diff.finalize_lines`), shown here and in Zeus's pane. `discard` deletes the worktree,
so it asks again: a second `Enter`.

**On leaving**: the screen opens it before it closes, when there is work
not handed off (`render.diff.asked_on_leaving`, the REPL's rule). `Esc` leaves the worktree
pending, as the REPL's Enter does; either way, what happened is printed once the screen has
gone (`Host.leave`).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import ClassVar, Final, Protocol

from textual.binding import BindingType
from textual.content import Content

from chickadee_cli.commands import LEAVE_PENDING
from chickadee_cli.render.diff import CHOICES, finalize_lines, offered
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.tui.diffs import split_diff
from chickadee_cli.tui.model import StaleRoute
from chickadee_cli.tui.overlays import Host, Overlay, OverlayKey, bindings, fit, row
from chickadee_cli.tui.panes.zeus import ansi
from chickadee_wire.models import HandoffResult, HandoffSummary

__all__ = ["DISCARD_AGAIN", "Handoff", "HandoffHost"]

DISCARD_AGAIN: Final = "discard deletes the worktree: Enter again to discard it, Esc to keep it"
_STYLE: Final = Style(tty=True)


class HandoffHost(Host, Protocol):
    def note(self, lines: list[str]) -> None:
        """REPL-styled lines in Zeus's pane."""
        ...

    def open_diff(self) -> None: ...

    def leave(self, lines: list[str]) -> None:
        """Close the screen, and print `lines` in the terminal once it has gone."""
        ...


class Handoff(Overlay):
    WORD = "HAND OFF"
    COMMAND = "/handoff"
    WIDTH = 72
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("up", "down"), "Up/Down", "choose", ("move(-1)", "move(1)")),
        OverlayKey(("enter",), "Enter", "choose", ("choose",)),
        OverlayKey(("d",), "d", "show diff", ("diff",)),
        OverlayKey(("escape",), "Esc", "decide later", ("close",)),
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def __init__(self, host: HandoffHost, *, leaving: bool = False) -> None:
        super().__init__(host)
        self.handing = host
        self.leaving = leaving
        self.discarding = False
        """`Enter` was pressed once on `discard`: a second carries it out."""
        self.sending = False
        self.done: list[str] | None = None
        """What the choice did, as `/handoff <choice>` prints it."""
        self.refusal: str | None = None

    def on_mount(self) -> None:
        super().on_mount()
        self.host.refresh_routes({StaleRoute.HANDOFF})

    def right(self) -> str:
        return "never pushes"

    @property
    def summary(self) -> HandoffSummary | None:
        return self.state.handoff

    def rows(self) -> int:
        return len(CHOICES)

    # -- what it shows ---------------------------------------------------------------------

    def body(self, width: int) -> Content:
        c = self.colours
        summary = self.summary
        if self.done is not None:
            more = "Enter or Esc leaves" if self.leaving else "Esc closes"
            return Content("\n").join(
                [ansi(self.done), Content(""), Content.styled(more, c["dim"])]
            )
        if summary is None:
            served = StaleRoute.HANDOFF not in self.state.not_served
            return Content.styled(
                "reading /handoff…" if served else "not served here", c["dim"]
            )
        git = summary.git
        if git is None or summary.workspace is None:
            return Content.styled("no worktree to hand off", c["dim"])
        lines = [self._against(summary), *self._commits(summary)]
        lines.append(Content.styled("─" * width, c["faint"]))
        lines += [Content.styled(note, c["bad"]) for note in summary.notes]
        if summary.notes:
            lines.append(Content(""))
        lines += self._choices(summary, width)
        if self.discarding:
            lines += [Content(""), Content.styled(DISCARD_AGAIN, c["bad"])]
        if self.sending:
            lines += [Content(""), Content.styled("handing off…", c["dim"])]
        if self.refusal is not None:
            lines += [Content(""), Content.styled(self.refusal, c["bad"])]
        return Content("\n").join(lines)

    def _against(self, summary: HandoffSummary) -> Content:
        c, git = self.colours, summary.git
        assert git is not None
        files = split_diff(summary.diff)
        added = sum(f.added for f in files.values())
        removed = sum(f.removed for f in files.values())
        name = (
            PurePosixPath(git.worktree_path).name
            if git.worktree_path
            else git.branch or "the worktree"
        )
        onto = git.source_branch or git.base[:12]
        return Content.assemble(
            f"wt/{name} against {onto}   ",
            (f"{len(files)} file{'' if len(files) == 1 else 's'} ", f"bold {c['fg']}"),
            (f"+{added} ", c["worker"]),
            (f"{self.host.glyphs.minus}{removed}", c["bad"]),
        )

    def _commits(self, summary: HandoffSummary) -> list[Content]:
        git = summary.git
        assert git is not None
        if not git.commits:
            return [Content.styled("no commits yet", self.colours["dim"])]
        return [
            Content.assemble(
                ("commits  ", self.colours["dim"]), "   ".join(sha[:7] for sha in git.commits)
            )
        ]

    def _choices(self, summary: HandoffSummary, width: int) -> list[Content]:
        c = self.colours
        live = offered(summary)
        refused = summary.refused or {}
        # The engine's one wording per choice (`HandoffSummary.offers`), as the REPL's
        # review prints it; a server one minor older sends none, and the name stands alone.
        words = summary.offers or {}
        lines: list[Content] = []
        for at, choice in enumerate(CHOICES):
            chosen = at == self.cursor
            mark = "*" if chosen else " "
            said = words.get(choice, "")
            if choice == "discard":
                said = f"{said}. Asks again." if said else "Asks again."
            text = f"({mark}) {fit(choice, 9)}{said}"
            if choice not in live:
                why = refused.get(choice, "not offered")
                lines.append(
                    row(
                        # Why it is refused, in the words' place: what it would do is moot.
                        Content.styled(
                            fit(f"({mark}) {fit(choice, 9)}refused: {why}", width), c["dim"]
                        ),
                        width,
                        c,
                        chosen=chosen,
                    )
                )
            else:
                lines.append(row(Content(text), width, c, chosen=chosen))
        return lines

    def cursor_line(self) -> int:
        return self.cursor + 3

    # -- keys ------------------------------------------------------------------------------

    def chosen(self) -> str:
        return CHOICES[self.cursor % len(CHOICES)]

    def applies(self, action: str) -> bool:
        summary = self.summary
        if action == "choose":
            if self.done is not None:
                return self.leaving
            return (
                summary is not None and not self.sending and self.chosen() in offered(summary)
            )
        if action == "diff":
            return self.done is None and not self.sending
        if action == "close":
            return not self.sending  # a choice in flight is seen through to its result
        return True

    def action_move(self, step: int) -> None:
        self.discarding = False
        self.refusal = None
        super().action_move(step)

    def action_choose(self) -> None:
        if self.done is not None:
            self.handing.leave(self.done)
            return
        choice = self.chosen()
        if choice == "discard" and not self.discarding:
            self.discarding = True
            self.redraw()
            return
        self.discarding, self.sending, self.refusal = False, True, None
        sid = self.host.session_id
        self.host.read(lambda c: c.finalize(sid, choice), self.handed, self.refused)
        self.redraw()

    def handed(self, result: HandoffResult) -> None:
        self.sending = False
        self.done = finalize_lines(result, _STYLE)
        self.handing.note(self.done)
        self.host.refresh_routes({StaleRoute.HANDOFF})
        self.redraw()

    def refused(self, why: str) -> None:
        self.sending = False
        self.refusal = f"/handoff: {why}"
        self.redraw()

    def action_diff(self) -> None:
        self.dismiss()
        self.handing.open_diff()

    def action_close(self) -> None:
        if self.discarding:
            self.discarding = False
            self.redraw()
            return
        if self.leaving:
            self.handing.leave(self.done or _STYLE.lines(LEAVE_PENDING, Tone.DIM))
            return
        super().action_close()
