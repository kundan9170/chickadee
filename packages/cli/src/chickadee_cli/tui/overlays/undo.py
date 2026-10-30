"""Undo (`^U`, `/undo`): what can be taken back, and what each would
put back, before it is done.

**Three kinds, each the REPL's own undo**:
- **a turn**, from `/undoable` (read as the overlay opens), with the halves the engine offers
  for it (`Undoable.options`: code and conversation, conversation only, code only), `Space`
  stepping through them: `/undo <turn> <halves>`;
- **a task** of the run on screen, from the board: `/undo task <run> <task>`;
- **a step** of a task, from the worker pane's `⟲ step N`, by its effect's operation id:
  `/undo <operation> <run>`, its paths put back as they were before it.

**Never silent.** `Enter` sends the line and the REPL prints what was restored, kept and not
undone, in Zeus's pane; a conflict with the person's own edits is asked there as the undo runs.
"What happens" says only what the records say: which turn, which task's files, which step's
paths. Whether it would clash is not known before it runs (no route previews an undo), and is
said so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from textual.binding import BindingType
from textual.content import Content

from chickadee_cli.render.diff import task_files
from chickadee_cli.tui.overlays import CLOSE, Host, Overlay, OverlayKey, bindings, fit, row
from chickadee_wire.models import EffectCompletedP, Undoable

__all__ = ["HALVES", "UNKNOWN_CLASH", "TaskSteps", "Undo", "UndoRow", "choices"]

HALVES: Final = {
    "both": "code and conversation",
    "conversation": "conversation only",
    "code": "code only",
}
UNKNOWN_CLASH: Final = "Conflicts with your own edits are asked about as it runs."
_READING: Final = "reading what can be taken back…"


@dataclass(frozen=True)
class UndoRow:
    """One row that can be taken back, and the REPL line that does it."""

    kind: str
    """`turn`, `task` or `step`."""
    label: str
    line: str
    """The command, without a turn's halves (`Undo.line` adds them)."""
    happens: tuple[str, ...]
    undoable: Undoable | None = None
    indent: int = 0


@dataclass(frozen=True)
class TaskSteps:
    """A task of the run on screen: its files by `/diff`'s rule, and its steps."""

    task_id: str
    title: str
    files: tuple[str, ...]
    steps: tuple[tuple[int, EffectCompletedP], ...]
    """Each `⟲ step N` and the effect it completed, oldest first."""


def choices(
    undoable: tuple[Undoable, ...], run_id: str | None, tasks: list[TaskSteps]
) -> list[UndoRow]:
    """The rows, in the order drawn: the turns, newest first, then each task and its steps,
    newest step first."""
    out = [
        UndoRow(
            "turn",
            f"turn {u.turn}  {u.prompt}",
            f"/undo {u.turn}",
            (f"Takes back from turn {u.turn}.",),
            u,
        )
        for u in sorted(undoable, key=lambda u: u.turn, reverse=True)
    ]
    if run_id is None:
        return out
    for task in tasks:
        task_id, named = task.task_id, ", ".join(task.files) or "none the records name"
        out.append(
            UndoRow(
                "task",
                f"{task_id}  {task.title}",
                f"/undo task {run_id} {task_id}",
                (f"{task_id} goes back: its files {named}.",),
            )
        )
        for step, effect in reversed(task.steps):
            paths = ", ".join(effect.observed_surface) or "no path recorded"
            out.append(
                UndoRow(
                    "step",
                    f"step {step}  {paths}",
                    f"/undo {effect.operation_id} {run_id}",
                    (f"{paths} goes back to before step {step}.",),
                    indent=2,
                )
            )
    return out


class Undo(Overlay):
    WORD = "UNDO"
    COMMAND = "/undo"
    WIDTH = 75
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("up", "down"), "Up/Down", "choose", ("move(-1)", "move(1)")),
        OverlayKey(("space",), "Space", "scope", ("scope",)),
        OverlayKey(("enter",), "Enter", "undo", ("undo",)),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def __init__(self, host: Host) -> None:
        super().__init__(host)
        self.undoable: tuple[Undoable, ...] | None = None
        self.failed: str | None = None
        self.halves: dict[int, str] = {}
        """By turn, the halves chosen: the first the engine offers, until `Space`."""

    def on_mount(self) -> None:
        super().on_mount()
        sid = self.host.session_id
        self.host.read(lambda c: c.undoable(sid), self.loaded, self.read_failed)

    def loaded(self, undoable: tuple[Undoable, ...]) -> None:
        self.undoable = undoable
        self.redraw()

    def read_failed(self, why: str) -> None:
        self.failed = why
        self.undoable = ()
        self.redraw()

    def right(self) -> str:
        return "never silent: what it did is said"

    # -- what it offers --------------------------------------------------------------------

    def offered(self) -> list[UndoRow]:
        state = self.state
        tasks: list[TaskSteps] = []
        board = state.athena.board
        for task in board.tasks.values() if board is not None else ():
            files, _committed = task_files(task)
            worker = state.worker.tasks.get(task.task_id)
            steps = tuple(
                (e.step, e.record)
                for e in (worker.entries if worker is not None else [])
                if e.step is not None and isinstance(e.record, EffectCompletedP)
            )
            tasks.append(TaskSteps(task.task_id, task.title, files, steps))
        return choices(self.undoable or (), state.athena.run_id, tasks)

    def rows(self) -> int:
        return len(self.offered())

    def chosen(self) -> UndoRow | None:
        offered = self.offered()
        return offered[self.cursor] if 0 <= self.cursor < len(offered) else None

    def half(self, choice: UndoRow) -> str | None:
        if choice.undoable is None or not choice.undoable.options:
            return None
        return self.halves.get(choice.undoable.turn, choice.undoable.options[0])

    def line(self, choice: UndoRow) -> str:
        half = self.half(choice)
        return choice.line if half is None else f"{choice.line} {half}"

    # -- what it shows ---------------------------------------------------------------------

    def body(self, width: int) -> Content:
        c = self.colours
        offered = self.offered()
        if self.undoable is None:
            return Content.styled(_READING, c["dim"])
        lines = [Content.styled("Take back…", c["fg"]), Content("")]
        if self.failed:
            lines.append(Content.styled(f"the turns: {self.failed}", c["bad"]))
        if not offered:
            lines.append(Content.styled("nothing to take back", c["dim"]))
        for at, choice in enumerate(offered):
            text = Content(fit(" " * choice.indent + choice.label, width))
            colour = (
                c["worker"]
                if choice.kind == "task"
                else c["fg"]
                if choice.kind == "turn"
                else c["dim"]
            )
            drawn = text if at == self.cursor else Content.styled(text.plain, colour)
            lines.append(row(drawn, width, c, chosen=at == self.cursor))
        chosen = self.chosen()
        if chosen is not None:
            lines += [Content(""), Content.styled("WHAT HAPPENS", f"bold {c['fg']}")]
            lines += [Content(h) for h in chosen.happens]
            if chosen.undoable is not None and chosen.undoable.no_code:
                lines.append(Content.styled(chosen.undoable.no_code, c["dim"]))
            lines.append(Content.styled(UNKNOWN_CLASH, c["dim"]))
            half = self.half(chosen)
            if chosen.undoable is not None and half is not None:
                lines += [Content(""), Content.styled("UNDO", f"bold {c['fg']}")]
                radios = [
                    Content.styled(
                        f"({'*' if o == half else ' '}) {HALVES[o]}",
                        c["zeus"] if o == half else c["dim"],
                    )
                    for o in chosen.undoable.options
                ]
                lines.append(Content("   ").join(radios))
                lines += [
                    Content(""),
                    Content.styled("/redo takes the last undo back.", c["dim"]),
                ]
        return Content("\n").join(lines)

    def cursor_line(self) -> int:
        return self.cursor + 2

    # -- keys ------------------------------------------------------------------------------

    def applies(self, action: str) -> bool:
        chosen = self.chosen()
        if action == "undo":
            return chosen is not None
        if action == "scope":
            return (
                chosen is not None
                and chosen.undoable is not None
                and len(chosen.undoable.options) > 1
            )
        return True

    def action_scope(self) -> None:
        chosen = self.chosen()
        if chosen is None or chosen.undoable is None:
            return
        options: list[str] = list(chosen.undoable.options)
        half = self.half(chosen) or options[0]
        self.halves[chosen.undoable.turn] = options[(options.index(half) + 1) % len(options)]
        self.redraw()

    def action_undo(self) -> None:
        chosen = self.chosen()
        if chosen is not None:
            self.dismiss()
            self.host.command(self.line(chosen))
