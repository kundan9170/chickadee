"""The worker's pane: right, bottom. The current task's calls, steps
and commands as cards (`panes/cards.box_cards`, the same card as Zeus's and the
Planner's; an edit its files and their line counts, a command `$` and the command), newest at
the bottom and followed, and its facts pinned below; read-only.

**The facts say what ran, never that the change is right** (invariant 7): files, lines,
the tests' outcome as the harness read it (a run that collected nothing is never a pass), the
syntax check, and whether a commit exists.

What the wire does not carry is left out rather than guessed: a step's line counts until its
diff is read (`TuiState.step_diffs`; a server that does not build the route shows the paths
alone), a command's output past its result's first line, and what the worker is doing between
records (the box streams no progress). `worker_line` says an entry in one line, for the narrow
layout's ticker. Its right label is the worker's spend and how full its latest prompt was
(`panes.spend_label`); the model that served it is the header's.

**A run that failed opens its report here**: the `coder` result's own terminal and
why lines, word for word (`CoderOutcome`, never read out of its text), and what comes next by
the records: a run whose journal never ended is resumed (`/resume`), one that ended
(`ABORTED` with its handoff) is handed off.

**`Enter` on a command** asks the screen for what the record holds of it
(`overlays/command.py`): with the pane focused, `↑`/`↓` choose among the task's commands, the
latest first chosen, its card marked."""

from __future__ import annotations

import time
from datetime import datetime
from typing import ClassVar, Final

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from chickadee_cli.render.stream import preview
from chickadee_cli.tui.keys import WORKER_KEYS
from chickadee_cli.tui.model import FactsLine, TuiState, WorkerEntry, WorkerKind, WorkerTask
from chickadee_cli.tui.names import ROLE_WORDS
from chickadee_cli.tui.panes import Following, RoleFrame, live_lines, spend_label
from chickadee_cli.tui.panes.cards import CardText, box_cards, hook_tick, show_cards
from chickadee_cli.tui.statusbar import pane_hint
from chickadee_cli.tui.theme import Glyphs
from chickadee_wire.models import EffectCompletedP, ExecP, HookP, ToolCallP

__all__ = [
    "FAILED",
    "WorkerFrame",
    "elapsed",
    "facts_lines",
    "failure_lines",
    "syntax_word",
    "tests_word",
    "worker_after",
    "worker_body",
    "worker_cards",
    "worker_line",
]

_TOOL_TONE: Final = {
    "read": "dim",
    "search": "dim",
    "symbol": "dim",
    "edit": "worker",
    "shell": "shell",
}
"""A call's dot by what it does (`ui/` `workerRun`: a read dim, an edit green, a command blue);
the tool's name is always drawn beside it."""


def elapsed(task: WorkerTask) -> str | None:
    """`m:ss` from the task's delegation to its latest record, both the records' own times."""
    if task.delegated_at is None or task.last_at is None:
        return None
    seconds = int(
        (
            datetime.fromisoformat(task.last_at) - datetime.fromisoformat(task.delegated_at)
        ).total_seconds()
    )
    return f"{max(seconds, 0) // 60}:{max(seconds, 0) % 60:02d}"


FAILED: Final = frozenset({"BLOCKED", "ABORTED"})
"""The terminals a run's failure ends on (`BLOCKED` with the harness's reason, or a
`HarnessBug`'s `ABORTED`); `PARTIAL` and `COMPLETED` are a run's own ends."""


def failure_lines(state: TuiState, colours: dict[str, str]) -> list[Content]:
    """The failed run's report, or nothing for a run that did not fail."""
    worker = state.worker
    outcome = worker.outcome
    if outcome is None or outcome.terminal not in FAILED:
        return []
    bad, dim = colours["bad"], colours["dim"]
    lines = [Content.styled(f"RUN {outcome.terminal}", f"bold {bad}")]
    lines += [Content.styled(why, colours["fg"]) for why in outcome.why]
    if worker.end is None:
        lines.append(Content.styled("its journal did not end: /resume continues it", dim))
    else:
        lines.append(
            Content.styled(f"it ended {worker.end.outcome}: /handoff shows what it kept", dim)
        )
    return lines


def worker_body(state: TuiState, colours: dict[str, str]) -> Content:
    """The task's head line: its id, its round and its time so far; a failed run's report when
    there is no task to show."""
    task = state.worker.task
    if task is None:
        return Content("\n").join(failure_lines(state, colours))
    dim = colours["dim"]
    head = f"{task.task_id} · round {task.round or 1}"
    took = elapsed(task)
    return Content.assemble((head, dim), (f"  {took}" if took else "", dim))


def worker_cards(state: TuiState, colours: dict[str, str]) -> list[CardText]:
    """The task's calls, steps and commands as cards, in order."""
    task = state.worker.task
    if task is None:
        return []
    return box_cards(task.entries, step_diffs=state.step_diffs, colours=colours)


def worker_after(state: TuiState, glyphs: Glyphs, colours: dict[str, str]) -> list[Content]:
    """Under the cards: the reply being written, then a failed run's report."""
    task = state.worker.task
    if task is None:
        return []
    lines: list[Content] = []
    now = time.monotonic()
    for reply in state.live.open.values():
        if reply.role == "worker" and reply.task == task.task_id:
            lines += live_lines(reply, glyphs, colours, now=now)  # as it is written
    failed = failure_lines(state, colours)
    if failed:
        lines += [Content(""), *failed]
    return lines


def worker_line(entry: WorkerEntry, glyphs: Glyphs, colours: dict[str, str]) -> Content:
    dim, bad, faint = colours["dim"], colours["bad"], colours["faint"]
    match entry.record:
        case ToolCallP() as call if call.denied_reason is not None:
            said = preview(call.result_preview or call.denied_reason, whole=False)
            return Content.styled(f"✗ {call.tool} refused: {said}", bad)
        case ToolCallP() as call:
            args = preview(call.args_preview, whole=False) if call.args_preview else ""
            tone = colours[_TOOL_TONE.get(call.tool, "fg")]
            return Content.assemble(
                (glyphs.in_flight, tone), (f" {call.tool}", colours["fg"]), (f" {args}", dim)
            )
        case EffectCompletedP() as step:
            what = ", ".join(step.observed_surface) or "nothing changed"
            tone = dim if step.outcome == "succeeded" else bad
            return Content.assemble(
                ("│ ", faint), (f"{glyphs.step} step {entry.step}  {what}", tone)
            )
        case ExecP() as ran:
            late = " (timed out)" if ran.timed_out else ""
            tone = dim if ran.exit_code == 0 and not ran.timed_out else bad
            return Content.assemble(
                ("│ ", faint), (f"{ran.label}: exit {ran.exit_code}{late}", tone)
            )
        case HookP() as hook:
            return Content.assemble(
                (f"{glyphs.hook} ", colours["zeus"]),
                (f"{hook.name} ", colours["fg"]),
                hook_tick(hook, colours),
            )
        case _:
            return Content.styled(f"unreadable record ({entry.text})", bad)


def facts_lines(state: TuiState, glyphs: Glyphs, colours: dict[str, str]) -> Content:
    """The pinned facts block: what ran, from the round's facts and the board."""
    task = state.worker.task
    facts = task.facts_line if task is not None else None
    dim, fg = colours["dim"], colours["fg"]
    if task is None or facts is None:
        return Content.styled("facts once the round reports", dim)
    board = state.athena.board
    shown = board.tasks.get(task.task_id) if board is not None else None
    committed = shown is not None and shown.state == "committed"
    return Content("\n").join(
        [
            Content.assemble(
                (f"{facts.files} file(s) ", fg),
                (f"+{facts.added}", colours["worker"]),
                " ",
                (f"{glyphs.minus}{facts.removed}", colours["bad"]),
                (f"  after round {task.facts_round}", dim),
            ),
            tests_word(facts, colours),
            syntax_word(facts, colours),
            Content.styled("committed" if committed else "no commit yet", dim),
        ]
    )


def tests_word(facts: FactsLine, colours: dict[str, str]) -> Content:
    if facts.tests is None:
        return Content.assemble(("tests: ", colours["fg"]), ("none run", colours["dim"]))
    if facts.failing:
        word, tone = f"{len(facts.failing)} failing", "bad"
    elif facts.tests == "not_collected":
        word, tone = "no tests collected", "bad"
    else:
        word, tone = (
            facts.tests.replace("_", " "),
            "worker" if facts.tests == "passed" else "bad",
        )
    return Content.assemble(("tests: ", colours["fg"]), (word, colours[tone]))


def syntax_word(facts: FactsLine, colours: dict[str, str]) -> Content:
    if facts.syntax_ok is None:
        return Content.assemble(("syntax: ", colours["fg"]), ("not checked", colours["dim"]))
    word, tone = ("ok", "worker") if facts.syntax_ok else ("fails", "bad")
    return Content.assemble(("syntax: ", colours["fg"]), (word, colours[tone]))


class WorkerFrame(RoleFrame):
    role = "worker"
    colour = "worker"
    KEYS = WORKER_KEYS
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(k.key, k.action, show=False) for k in WORKER_KEYS if k.action is not None
    ]

    DEFAULT_CSS = """
    WorkerFrame { border: round $worker; }
    WorkerFrame.-idle { border: round $faint; }
    WorkerFrame:focus, WorkerFrame.-idle:focus { border: double $worker; }
    WorkerFrame { overflow-y: hidden; }
    WorkerFrame > .facts { height: auto; border-top: solid $faint; }
    WorkerFrame .body { text-wrap: nowrap; text-overflow: ellipsis; }
    WorkerFrame .cards, WorkerFrame .after { height: auto; }
    """

    def compose(self) -> ComposeResult:
        # The stream scrolls and follows; the facts stay pinned below it, outside the scroll,
        # so they never cover its newest lines (a failed run's report among them).
        self.stream = Following()
        self.cards = Vertical(classes="cards")
        self.after = Static("", classes="after")
        with self.stream:
            yield from super().compose()
            yield self.cards
            yield self.after
        self.facts = Static("", classes="facts")
        yield self.facts

    def show(self, state: TuiState) -> None:
        super().show(state)
        idle = self.idle(state)
        chosen = self.chosen()
        show_cards(
            self.cards,
            [] if idle else worker_cards(state, self._colours),
            int(chosen) if chosen is not None and self.has_focus else None,
        )
        after = [] if idle else worker_after(state, self._glyphs, self._colours)
        if not idle and self.has_focus:
            after.append(pane_hint(state, self._colours, self.KEYS))
        self.after.update(Content("\n").join(after))
        self.after.display = bool(after)
        self.facts.display = state.worker.task is not None
        self.facts.update(facts_lines(state, self._glyphs, self._colours))

    def idle(self, state: TuiState) -> bool:
        return state.worker.task is None and not failure_lines(state, self._colours)

    def title(self, state: TuiState) -> Content:
        return Content.styled(ROLE_WORDS["worker"].upper(), f"bold {self._colours['worker']}")

    def label(self, state: TuiState) -> Content | tuple[Content, ...]:
        """The worker's spend over every task, and how full the shown task's latest prompt
        was; the model is the header's (`header.models_line`)."""
        costs = [t.cost_usd for t in state.worker.tasks.values() if t.cost_usd is not None]
        return spend_label(
            sum(costs) if costs else None,
            state.context("worker"),
            self._colours["worker"],
            self._glyphs,
            self._colours,
        )

    def idle_text(self, state: TuiState) -> Content:
        return Content.styled("idle\n\nStarts once the plan is approved.", self._colours["dim"])

    def content(self, state: TuiState) -> Content:
        return worker_body(state, self._colours)

    def rows(self, state: TuiState) -> list[str]:
        task = state.worker.task
        entries = task.entries if task is not None else []
        return [str(e.seq) for e in entries if e.kind is WorkerKind.EXEC]
