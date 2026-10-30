"""A task's details ("`Enter` on a task opens its brief, the worker's
last report and Athena's review verdict"), from Athena's pane.

**What the records hold of it, and no more**. The board's line (its state,
rounds, cost, how its last round stopped, its files by `/diff`'s rule) and, round by round,
how the round stopped, what its facts measured, and Athena's answer to it. The brief, each
report's words and each review's are run blobs, read as the overlay shows them
(`/run/{run_id}/blob/{sha}`).
"""

from __future__ import annotations

from typing import ClassVar, Final

from textual.binding import BindingType
from textual.content import Content

from chickadee_cli.render.diff import task_files
from chickadee_cli.tui.model import facts_line
from chickadee_cli.tui.names import ROLE_WORDS
from chickadee_cli.tui.overlays import CLOSE, Blobs, Host, Overlay, OverlayKey, bindings, money
from chickadee_cli.tui.panes.worker import syntax_word, tests_word
from chickadee_cli.tui.theme import TASK_GLYPH

__all__ = ["TaskDetail", "blob_lines"]

NONE_GIVEN: Final = "none given"


class TaskDetail(Overlay):
    WORD = "TASK"
    COMMAND = "Enter on a task"
    WIDTH = 72
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("up", "down"), "Up/Down", "scroll", ("scroll(-1)", "scroll(1)")),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def __init__(self, host: Host, task_id: str) -> None:
        super().__init__(host)
        self.task_id = task_id
        self.blobs = Blobs(host, host.state.athena.run_id, self.redraw)

    def heading(self) -> str:
        return self.task_id

    def right(self) -> str:
        return "what the records hold"

    def body(self, width: int) -> Content:
        c, glyphs = self.colours, self.host.glyphs
        state = self.state
        board = state.athena.board
        task = board.tasks.get(self.task_id) if board is not None else None
        if task is None:
            return Content.styled(f"{self.task_id} is not on the board", c["dim"])
        glyph = getattr(glyphs, TASK_GLYPH.get(task.state, "waiting"))
        stopped = f" · last round stopped: {task.stopped}" if task.stopped else ""
        lines = [
            Content.assemble((f"{glyph} {task.task_id}  ", f"bold {c['athena']}"), task.title),
            Content.styled(
                f"{task.state} · {task.rounds} round(s) · {money(task.cost_usd)}{stopped}",
                c["dim"],
            ),
            Content(""),
            Content.styled("BRIEF", f"bold {c['fg']}"),
        ]
        worker = state.worker.tasks.get(self.task_id)
        brief = self.blobs.said(worker.brief_sha if worker is not None else None)
        lines += blob_lines(brief, c, indent="  ")
        lines += [Content(""), Content.styled("ROUNDS", f"bold {c['fg']}")]
        reports = worker.reports if worker is not None else []
        reviews = worker.reviews if worker is not None else []
        if not reports:
            lines.append(Content.styled("no round has reported yet", c["dim"]))
        for at, report in enumerate(reports):
            facts = facts_line(report.facts)
            review = reviews[at] if at < len(reviews) else None
            answer = (
                review.answer.replace("_", " ") if review is not None else "not reviewed yet"
            )
            lines.append(
                Content.assemble(
                    (f"round {report.round}  ", c["fg"]),
                    (f"{report.stopped} · {facts.files} file(s) ", c["dim"]),
                    (f"+{facts.added} ", c["worker"]),
                    (f"{glyphs.minus}{facts.removed}", c["bad"]),
                )
            )
            lines.append(
                Content.assemble("  ", tests_word(facts, c), ("  ", ""), syntax_word(facts, c))
            )
            lines += blob_lines(self.blobs.said(report.report_sha), c, indent="  │ ")
            lines.append(
                Content.assemble((f"  {ROLE_WORDS['athena']} → ", c["athena"]), answer)
            )
            if review is not None:
                lines += blob_lines(self.blobs.said(review.text_sha), c, indent="  │ ")
        files, committed = task_files(task)
        lines += [Content(""), Content.styled("FILES", f"bold {c['fg']}")]
        word = "committed" if committed else "its last round changed"
        lines.append(Content.styled(f"{word}: {', '.join(files) or 'none'}", c["dim"]))
        return Content("\n").join(lines)

    def action_scroll(self, step: int) -> None:
        self.scroll.scroll_relative(y=step, animate=False)


def blob_lines(text: str | None, colours: dict[str, str], *, indent: str) -> list[Content]:
    """A blob's text, each line under `indent`, verbatim (whitespace is meaning); `none given`
    for no blob."""
    if text is None:
        return [Content.styled(f"{indent}{NONE_GIVEN}", colours["dim"])]
    return [Content.styled(f"{indent}{line}", colours["fg"]) for line in text.splitlines()]
