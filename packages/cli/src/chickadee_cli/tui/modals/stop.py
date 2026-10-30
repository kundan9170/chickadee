"""A stop as a modal: stop 1 Zeus's design, stop 2 Athena's
plan, the markdown shown in full and rendered, and the four answers.

**What is approved is what was shown** (`approval.py`): approving binds this text by its sha,
which the title names so a text shown again reads as new. **Nothing is highlighted when it
opens**, so a stray `Enter` approves nothing. **It opens as it is asked** (amending the
banner-first stop), and `Esc` puts it back as a banner, the run waiting on it, answering
nothing, while the person talks to Zeus; `^S` opens it again. `Esc` once did nothing, since
a stop left open was approved by its timeout; with the person here a stop has no timeout, so
hiding it costs nothing and the banner still says it waits. An answer is chosen (its
letter, `←`/`→`) and then confirmed with `Enter`.

**The REPL's rules, not a copy.** Each answer is read by `approval.parse_stop` from the line
the REPL would have typed (`a`, `e`, `n <words>`, `c [reason]`), so a note still needs words
and an unchanged edit still sends nothing, said in the modal.

**No task count at stop 2:** the stop carries the plan's text, which lists its tasks, and
counting them here would read a fact out of prose. The count is recorded once the plan is
approved (`plan_written.tasks`); a count frozen on the stop could go stale,
since an edit or a note shows the stop again with new text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar, Final, Literal

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import Resize
from textual.widgets import Input, Markdown, Static

from chickadee_cli.approval import Editor, Invalid, parse_stop, stop_head, stop_where
from chickadee_cli.tui.modals import Answer, QuestionModal, Send, answer_bindings
from chickadee_cli.tui.panes import border_title
from chickadee_wire.models import ApprovalStop
from chickadee_wire.questions import OpenQuestion

__all__ = ["STOP_ANSWERS", "StopModal", "stop_name", "stop_title"]

STOP_ANSWERS: Final = (
    Answer("a", "approve"),
    Answer("e", "edit"),
    Answer("n", "note"),
    Answer("c", "cancel"),
)
"""The `[a] approve [e] edit in $EDITOR [n] note to … [c] cancel`: `STOP_CHOICES`."""

_NUMBER: Final = {"design": 1, "plan": 2}
_SHA_SHOWN: Final = 4
"""The sha's head the title names (the `design 7c1e…`)."""

type Words = Literal["n", "c"]


def stop_name(stop: ApprovalStop) -> str:
    """`STOP 2 · PLAN`: the modal's title, and the banner a stop put back is."""
    return f"STOP {_NUMBER.get(stop.kind, '?')} · {stop.kind.upper()}"


def stop_title(stop: ApprovalStop, shown: int) -> tuple[str, str]:
    """`STOP 1 · DESIGN`, `shown 2 times`, and the right label `zeus · 7c1e…`. `shown` is how
    many times a stop of this kind has been shown since the last was settled (`Feed`'s count,
    `TuiState.stop_shown`): the engine's own `round` restarts when a note sends the text back
    to its author, who answers with a new stop."""
    times = "1 time" if shown == 1 else f"{shown} times"
    head = f"{stop_name(stop)} · shown {times}"
    return head, f"{stop.author} · {stop.sha[:_SHA_SHOWN]}…"


class StopModal(QuestionModal):
    TIMES_OUT_AS: ClassVar[str] = "approved, as the server's timeout answers a stop"
    ANSWERS: ClassVar[tuple[Answer, ...]] = STOP_ANSWERS
    BINDINGS: ClassVar[list[BindingType]] = [
        *answer_bindings(STOP_ANSWERS),
        Binding("escape", "back", show=False),
    ]

    DEFAULT_CSS = """
    StopModal > #box { width: 80%; height: 80%; }
    StopModal #text { height: 1fr; }
    StopModal #text > Markdown { margin: 0; padding: 0; background: transparent; }
    StopModal #answers { height: auto; margin-top: 1; }
    StopModal #words { display: none; border: none; height: 1; padding: 0; }
    StopModal #words.-open { display: block; }
    StopModal Static { height: auto; }
    """

    def __init__(
        self,
        asked: OpenQuestion,
        colours: dict[str, str],
        send: Send,
        edit: Editor,
        *,
        shown: int = 1,
        deadline: str | None = None,
        later: Callable[[], None] = lambda: None,
    ) -> None:
        if asked.stop is None:
            raise ValueError("a stop modal shows a stop")
        super().__init__(asked, colours, send, deadline=deadline)
        self.stop = asked.stop
        self._later = later
        """`Esc` on the stop itself: back to its banner, unanswered (the app's)."""
        self.shown = shown
        self._edit = edit
        self._words: Words | None = None
        self.answers = Static("", id="answers")
        self.words = Input(id="words")

    def compose(self) -> ComposeResult:
        stop = self.stop
        colour = self._colours[stop.author]
        with Vertical(id="box") as box:
            box.styles.border = ("double", colour)
            yield Static(Content(stop_head(stop)), classes="dim")
            yield Static(Content(stop_where(stop)), classes="dim")
            with VerticalScroll(id="text"):
                yield Markdown(stop.text)
            if stop.refused:
                yield Static(Content(stop.refused), classes="bad")
            yield self.answers
            yield self.words
            yield self.why
            yield self.deadline

    def on_mount(self) -> None:
        super().on_mount()
        self.redraw()
        self.query_one("#text").focus()

    def on_resize(self, _event: Resize) -> None:
        self._retitle()

    def _retitle(self) -> None:
        box = self.query_one("#box")
        head, right = stop_title(self.stop, self.shown)
        c = self._colours
        left = Content.assemble((f" {head} ", f"bold {c['ink']} on {c[self.stop.author]}"))
        box.border_title = border_title(
            left, Content.styled(right, c["dim"]), box.outer_size.width, double=True
        )

    def redraw(self) -> None:
        author = self.stop.author.capitalize()
        labels = {
            "a": "approve",
            "e": "edit in $EDITOR",
            "n": f"note to {author}",
            "c": "cancel",
        }
        chips = [self.chip(a.key, labels[a.key]) for a in STOP_ANSWERS]
        self.answers.update(Content("\n").join([Content(" ").join(chips), self.confirm_chip()]))
        if self.is_mounted:
            self._retitle()

    def typing(self) -> bool:
        """While a note or a reason is typed, the keys are its letters."""
        return self._words is not None

    # -- answering -------------------------------------------------------------------------

    def submit(self, key: str) -> None:
        """The highlighted answer, confirmed: a note or a reason asks for its words first."""
        if key in ("n", "c"):
            self._ask_words("n" if key == "n" else "c")
            return
        self._reply(key)

    def _ask_words(self, words: Words) -> None:
        self._words = words
        author = self.stop.author.capitalize()
        self.words.placeholder = (
            f"note to {author}: what to change (Enter sends, Esc back)"
            if words == "n"
            else "cancel: why, if you like (Enter sends, Esc back)"
        )
        self.words.value = ""
        self.words.add_class("-open")
        self.words.focus()
        self.redraw()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._words is None or self.sent:
            return
        line = f"{self._words} {event.value}"
        if self._reply(line):
            self._close_words()

    def action_back(self) -> None:
        """Esc: back out of a note or a reason; on the stop itself, back to its banner, with
        nothing answered."""
        if self._words is not None:
            self._close_words()
        elif not self.sent:
            self._later()

    def _close_words(self) -> None:
        self._words = None
        self.words.remove_class("-open")
        self.query_one("#text").focus()
        self.redraw()

    def _reply(self, line: str) -> bool:
        """`line` read as the REPL reads it at a stop, and sent; False when it answers
        nothing, said in the modal."""
        reply = parse_stop(self.stop, line, self._edit)
        if isinstance(reply, Invalid):
            self.refuse(reply.why)
            return False
        self.send(reply.choice, reply.note, reply.text)
        return True
