"""The REPL's half of a question: the renderer over
`chickadee_wire.questions`, never an `InteractionPort`. The port is the server's; here a
`question_asked` record is shown, and the person's next line is its answer.

**One question at a time, oldest first.** Records open and close questions (`Questions`);
an answer given by another client, a timeout or a withdrawal closes one here too, and the
next open one is shown. A frontend that draws some questions itself (the TUI's modals) says
which (`draws`): those are not printed, and are answered through `send`.

**A call that asks** (`answering`): an undo, a redo or a fork blocks in the engine while it
asks the person (a conflict, restore or keep), so it runs on a thread while the asking thread
answers each question it raises. The one way a command waits on a call that may ask.

**A stop is answered by `approval.py`**: its markdown is shown exactly, and the line
is `a`, `e`, `n <note>` or `c [reason]` rather than a choice. A design gate an older log left
open (retired with the switch) is closed by any line, as a cancel.

**An answer is a choice, then an optional note**: `main` or `2 because it is the default
branch`. Enter takes the default. A line that names no choice is not sent: the options are
shown again. A question with no options takes the whole line as its answer. Who answered is
the token's to say, never this module's.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import click

from chickadee_cli.approval import (
    Editor,
    Invalid,
    parse_stop,
    stop_head,
    stop_lines,
)
from chickadee_cli.render.question import approval_head, choices, default_of, question_lines
from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import ErrorCode, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.questions import OpenQuestion, Questions

__all__ = ["Interaction", "answering", "open_in_editor", "parse_answer"]

RETIRED_GATE: Final = (
    "a design gate from before the rebuild is open (D187, retired): any line closes it"
)

WAITING_MAX: Final = 160
"""The most of a question the toolbar shows: one line, so the toolbar keeps its height."""

POLL_S: Final = 0.05
"""How often `answering` looks for a question while its call runs."""


def parse_answer(asked: OpenQuestion, line: str) -> tuple[str, str | None] | None:
    """`line` as `(choice, note)`, or None when it names no choice. A choice is an option's
    key (any case), or its number; an empty line is the default. A question with no options
    asks for words: the whole line is the choice."""
    text = line.strip()
    if not text:
        return default_of(asked), None
    if not choices(asked):
        return text, None
    head, _, rest = text.partition(" ")
    note = rest.strip() or None
    options = choices(asked)
    if head.isdigit() and 1 <= int(head) <= len(options):
        return options[int(head) - 1].key, note
    for option in options:
        if option.key.lower() == head.lower():
            return option.key, note
    return None


class Interaction:
    """The open questions of one session, shown as they open and answered from the REPL's
    lines. `feed` runs on the events thread; `answer` on the REPL's: one lock between."""

    def __init__(
        self,
        client: Client,
        session_id: str,
        style: Style,
        out: Callable[[str], None],
        edit: Editor | None = None,
        *,
        draws: Callable[[OpenQuestion], bool] | None = None,
        answers: Callable[[OpenQuestion], bool] | None = None,
    ) -> None:
        """`draws`: the questions the frontend draws itself, which are not printed here.
        `answers`: the ones it also answers itself, through `send` (the TUI's modals), which a
        typed line never answers; a question it only draws (in Zeus's card) still takes the
        next line."""
        self._client = client
        self._sid = session_id
        self._style = style
        self._out = out
        self._edit = edit if edit is not None else open_in_editor
        self._draws = draws
        self._answers = answers
        self._questions = Questions()
        self._shown: str | None = None
        self._lock = threading.Lock()

    @property
    def current(self) -> OpenQuestion | None:
        """The oldest question still open."""
        with self._lock:
            open_now = self._questions.open
        return open_now[0] if open_now else None

    @property
    def typed(self) -> OpenQuestion | None:
        """The question a typed line answers: the oldest open one the frontend does not answer
        itself (`answers`). One it answers (the TUI's stop, waiting as a banner while the
        person talks to Zeus) goes through `send`, so a line typed beside it is words
        or a command, never its answer: an `a` meant for Zeus must not approve a plan."""
        with self._lock:
            open_now = self._questions.open
        answers = self._answers
        return next((q for q in open_now if answers is None or not answers(q)), None)

    def fold(self, frames: Iterable[RecordFrame]) -> None:
        """Apply what the session already holds, then show the question still open, if any.
        Only the end state is shown: a question asked and closed earlier in the history (a
        stop approved, a permission a stopped run left and that was withdrawn) is over, and
        shown again it reads as open, so the next line would answer a question nobody waits
        on and go to the session as words instead."""
        with self._lock:
            self._questions.feed_all(frames)
        self._show_head()

    def feed(self, frame: RecordFrame) -> None:
        """Apply a record; when the question to answer changes, show the new one."""
        with self._lock:
            self._questions.feed(frame)
        self._show_head()

    def _show_head(self) -> None:
        with self._lock:
            open_now = self._questions.open
            head = open_now[0] if open_now else None
            if head is None or head.id == self._shown:
                return
            self._shown = head.id
        if self._draws is not None and self._draws(head):
            return
        for line in self._lines(head):
            self._out(line)

    def _lines(self, asked: OpenQuestion) -> list[str]:
        if asked.gate is not None:
            return self._style.lines(RETIRED_GATE, Tone.DIM)
        if asked.stop is not None:
            return stop_lines(asked.stop, self._style)
        return question_lines(asked, self._style)

    def waiting(self) -> str | None:
        """The open question as one line, for the prompt's toolbar, or None: printed lines can
        scroll away or be lost to a terminal's redraw (2026-10-22), and a question nobody sees
        holds the turn, so the toolbar keeps it on screen until it is answered."""
        asked = self.current
        if asked is None:
            return None
        if asked.question is not None:
            what = asked.question.text
        elif asked.approval is not None:
            what = approval_head(asked.approval)
        elif asked.stop is not None:
            what = stop_head(asked.stop)
        else:
            what = "a retired gate; any line closes it"
        what = " ".join(what.split())
        if len(what) > WAITING_MAX:
            what = what[: WAITING_MAX - 1] + "…"
        return f"? waiting for your answer: {what}"

    def note(self, text: str) -> None:
        """A dim line beside the questions: something the person should know about one."""
        for line in self._style.lines(text, Tone.DIM):
            self._out(line)

    def answer(self, line: str) -> str | None:
        """Send `line` as the answer to the question a typed line answers (`typed`), or show the
        options again. The id of the question it closed (answered here, or already elsewhere),
        or None when the line named no option and the question is still open."""
        asked = self.typed
        if asked is None:
            return None
        if asked.gate is not None:
            return self.send(asked, "cancel", line.strip() or None)
        if asked.stop is not None:
            return self._answer_stop(asked, line)
        parsed = parse_answer(asked, line)
        if parsed is None:
            self._out(self._style.lines(f"not an option: {line.strip()}", Tone.BAD)[0])
            for text in self._lines(asked):
                self._out(text)
            return None
        choice, note = parsed
        return self.send(asked, choice, note)

    def _answer_stop(self, asked: OpenQuestion, line: str) -> str | None:
        stop = asked.stop
        if stop is None:
            raise ValueError("not a stop")
        parsed = parse_stop(stop, line, self._edit, open_folder_in_editor)
        if isinstance(parsed, Invalid):
            self._out(self._style.lines(parsed.why, Tone.BAD)[0])
            return None
        return self.send(asked, parsed.choice, parsed.note, text=parsed.text)

    def send(
        self,
        asked: OpenQuestion,
        choice: str,
        note: str | None = None,
        *,
        text: str | None = None,
    ) -> str:
        """Send `choice` (with `note`, and a stop's edited `text`) as the answer to `asked`; the
        id it closed. One already answered elsewhere, or no longer asked, is said and counted
        closed; any other refusal is raised."""
        try:
            self._client.answer(self._sid, asked.id, choice, note, text=text)
        except WireError as exc:
            if exc.code in (ErrorCode.ANSWERED, ErrorCode.NOT_FOUND):
                self._out(self._style.lines(f"{asked.id}: {exc.message}", Tone.DIM)[0])
                return asked.id
            raise
        return asked.id


def open_in_editor(text: str) -> str | None:
    """`text` in `$EDITOR` (or `$VISUAL`, or the platform's default), as the person saved it;
    None when nothing was saved or no editor could be opened: a stop's edit is then not sent."""
    try:
        return click.edit(text, extension=".md")
    except click.ClickException:
        return None


def open_folder_in_editor(folder: Path) -> bool:
    """Every file of `folder` in `$EDITOR` at once, saved in place; whether
    an editor opened."""
    files = sorted(str(p) for p in folder.rglob("*") if p.is_file())
    try:
        click.edit(filename=files)
    except click.ClickException:
        return False
    return True


def answering[T](
    call: Callable[[], T], questions: Interaction, read: Callable[[str], str]
) -> T:
    """`call()`, run on a thread while each question it raises is answered with `read("? ")`.
    Its result, or what it raised. A question is asked once: its answer's record closes it,
    and until then it is not asked again. A line read for a question that closed meanwhile
    (another client answered it, or it timed out) is not sent, so it cannot land on the next
    question, and the person is told."""
    result: list[T] = []
    raised: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        try:
            result.append(call())
        except BaseException as exc:  # handed back to the caller's thread below
            raised.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, name="cli-answering", daemon=True).start()
    asked: set[str] = set()
    while not done.wait(POLL_S):
        question = questions.current
        if question is None or question.id in asked:
            continue
        line = read("? ")
        now = questions.current
        if now is None or now.id != question.id:
            asked.add(question.id)
            questions.note(f"{question.id} was settled meanwhile; that line was not used")
            continue
        closed = questions.answer(line)
        if closed is not None:
            asked.add(closed)
    if raised:
        raise raised[0]
    return result[0]
