"""The two modals over the screen: a stop (`stop.py`) and a command's
confirmation (`confirm.py`). The rule for which question gets one is the model's
(`model.drawn_as_modal`); the app opens it as the question becomes the oldest open one, and
closes it when the record that answers it lands, whoever answered.

**No second approval mechanism**. A modal answers through the REPL's own rules
(`approval.parse_stop`, `render.question.choices`) and the REPL's own send
(`Interaction.send`); it only puts keys over them. **The record is the truth**: sending does
not close the modal, the `question_answered` record does, and who answered is the token's to
say, never the modal's.

**One table binds and draws a modal's keys** (the status bar's rule, `keys.py`): a key whose
answer the wire cannot carry yet is drawn dimmed and bound to nothing. An answer this question
does not offer (an *always* with no grant to give) is not drawn: it answers another question.

**An answer is chosen, then confirmed**. A modal opens on its
own, often while the person is typing to Zeus, so no single key answers it: an answer's letter,
`←`/`→` or `Tab` only move the highlight, and `Enter` submits the highlighted answer (the
`⏎ confirm` button). For `ARM_S` after it opens its keys do nothing and are drawn dimmed, so
keys already on their way when it appeared reach nothing.

**While a modal is open, the panes keep updating beneath it**: it is drawn over the
screen, never instead of it, on the app's `$backdrop` (dimmed where colours blend).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Final

from textual.binding import Binding, BindingType
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from chickadee_wire.questions import OpenQuestion

__all__ = ["ARM_S", "Answer", "QuestionModal", "Send", "answer_bindings"]

ARM_S: Final = 1.0
"""How long a modal's keys stay inert after it opens: long enough for keys already typed to
reach nothing, short enough not to keep a person who has read the modal waiting."""

_CHOOSING: Final = frozenset({"answer", "move", "confirm", "default"})
"""The actions that choose or submit an answer: none runs before the modal is armed, after an
answer is sent, or while a note is typed."""

type Send = Callable[[OpenQuestion, str, str | None, str | None, Callable[[], None]], None]
"""How a modal sends: `(asked, choice, note, text, failed)`, the app's `answer`; `failed` is
called on the app's loop when the server refused it, so the keys work again."""


@dataclass(frozen=True)
class Answer:
    key: str
    """Textual's name for it (`a`, `y`)."""
    choice: str | None
    """What it answers on the wire; None for an answer the wire cannot carry yet (drawn
    dimmed, bound to nothing)."""
    wire_request: str | None = None
    """For an answer not built, the wire request that would build it."""


def answer_bindings(answers: tuple[Answer, ...]) -> list[BindingType]:
    """A modal's bindings, from its table: every built answer, none of the rest. A letter only
    selects its answer; `Enter` submits it."""
    return [Binding(a.key, f"answer('{a.key}')", show=False) for a in answers if a.choice]


_NAVIGATION: Final[list[BindingType]] = [
    # Priority, so a scrolling text in the modal does not take them; while a note is typed
    # their actions do not apply (`check_action`), and the key reaches the input instead.
    Binding("left", "move(-1)", show=False, priority=True),
    Binding("right", "move(1)", show=False, priority=True),
    Binding("tab", "move(1)", show=False, priority=True),
    Binding("shift+tab", "move(-1)", show=False, priority=True),
    Binding("enter", "confirm", show=False, priority=True),
]


class QuestionModal(ModalScreen[None]):
    """One open question over the screen, until the record that answers it lands."""

    ANSWERS: ClassVar[tuple[Answer, ...]]
    ARM_S: ClassVar[float] = ARM_S
    BINDINGS: ClassVar[list[BindingType]] = _NAVIGATION

    DEFAULT_CSS = """
    QuestionModal { align: center middle; }
    QuestionModal > #box { border: double $zeus; background: ansi_default; padding: 0 1; }
    QuestionModal .dim { color: $dim; }
    QuestionModal .bad { color: $bad; }
    QuestionModal #why { color: $bad; height: auto; }
    """

    TIMES_OUT_AS: ClassVar[str] = "answered by its default"
    """What happens to it when no one answers in time, for its deadline line."""

    def __init__(
        self,
        asked: OpenQuestion,
        colours: dict[str, str],
        send: Send,
        *,
        deadline: str | None = None,
    ) -> None:
        """`deadline`: when it answers itself (`TuiState.deadline`)."""
        super().__init__()
        self.asked = asked
        self._colours = colours
        self._send = send
        self.sent = False
        """An answer is on its way: the keys do nothing until its record closes the modal, or
        the server refuses it."""
        self.armed = False
        """`ARM_S` has passed since the modal opened: its keys work."""
        self.selected: str | None = self.first_selected()
        """The highlighted answer's key: what `Enter` submits."""
        self.why = Static("", id="why")
        said = f"Unanswered at {deadline}, it is {self.TIMES_OUT_AS}." if deadline else ""
        self.deadline = Static(Content.styled(said, colours["dim"]), id="deadline")
        """Drawn under the answers; empty when the deadline is unknown."""

    def on_mount(self) -> None:
        if self.ARM_S > 0:
            self.set_timer(self.ARM_S, self.arm)
        else:
            self.arm()

    def arm(self) -> None:
        """Its keys work from now: `ARM_S` after it opened."""
        self.armed = True
        self.redraw()

    # -- choosing --------------------------------------------------------------------------

    def first_selected(self) -> str | None:
        """What is highlighted when it opens: nothing, unless it has a safe default."""
        return None

    def order(self) -> tuple[str, ...]:
        """The answers' keys in the order they are drawn: what `←`/`→` step through."""
        return tuple(a.key for a in self.ANSWERS)

    def offered(self, key: str) -> bool:
        """Whether `key`'s answer is one this question takes now."""
        answer = next((a for a in self.ANSWERS if a.key == key), None)
        return answer is not None and answer.choice is not None

    def typing(self) -> bool:
        """Words are being typed in the modal (a stop's note): its keys are their letters."""
        return False

    def ready(self) -> bool:
        return self.armed and not self.sent and not self.typing()

    def lit(self, key: str) -> bool:
        return self.ready() and self.offered(key)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in _CHOOSING:
            if not self.ready():
                return False
            if action == "answer":
                return self.offered(str(parameters[0]))
        return True

    def action_answer(self, key: str) -> None:
        """An answer's letter: highlight it, never submit it."""
        self._select(key)

    def action_move(self, step: int) -> None:
        keys = [k for k in self.order() if self.offered(k)]
        if not keys:
            return
        if self.selected in keys:
            at = (keys.index(self.selected) + step) % len(keys)
        else:
            at = 0 if step > 0 else len(keys) - 1
        self._select(keys[at])

    def action_confirm(self) -> None:
        """`Enter`: submit the highlighted answer; with none highlighted, nothing."""
        if self.selected is not None and self.offered(self.selected):
            self.submit(self.selected)

    def _select(self, key: str) -> None:
        self.selected = key
        self.why.update("")
        self.redraw()

    def submit(self, key: str) -> None:
        raise NotImplementedError

    def chip(self, key: str, label: str) -> Content:
        """An answer as the mockup draws it: the key on amber and the label on its button;
        the highlighted one on teal, marked; one that does not apply now faint."""
        c = self._colours
        if not self.lit(key):
            mark = "▸" if key == self.selected else " "
            return Content.assemble(
                (f"{mark}{key} ", f"{c['dim']} on {c['faint']}"), (f" {label} ", c["dim"])
            )
        if key == self.selected:
            style = f"bold {c['ink']} on {c['athena']}"
            return Content.assemble((f"▸{key} ", style), (f"{label} ", style))
        return Content.assemble(
            (f" {key} ", f"{c['ink']} on {c['zeus']}"),
            (f"{label} ", f"{c['fg']} on {c['btn']}"),
        )

    def confirm_chip(self) -> Content:
        """`⏎ confirm`, lit when `Enter` would submit something, with how to choose."""
        c = self._colours
        hint = ("  ←/→ or a letter to choose", c["dim"])
        if self.ready() and self.selected is not None and self.offered(self.selected):
            key, label = f"{c['ink']} on {c['zeus']}", f"{c['fg']} on {c['btn']}"
            return Content.assemble((" ⏎ ", key), (" confirm ", label), hint)
        return Content.assemble(
            (" ⏎ ", f"{c['dim']} on {c['faint']}"), (" confirm ", c["dim"]), hint
        )

    # -- sending ---------------------------------------------------------------------------

    def send(self, choice: str, note: str | None = None, text: str | None = None) -> None:
        self.sent = True
        self.why.update("")
        self.redraw()
        self._send(self.asked, choice, note, text, self._failed)

    def _failed(self) -> None:
        self.sent = False
        self.redraw()

    def refuse(self, why: str) -> None:
        """Why what was pressed answers nothing: the question stays open."""
        self.why.update(why)

    def redraw(self) -> None:
        """Draw the answers again (lit, dimmed, sent)."""
        raise NotImplementedError
