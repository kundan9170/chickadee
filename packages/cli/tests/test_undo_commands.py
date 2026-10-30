"""`/undo`, `/redo` and `chickadee fork` at the terminal: each prompt, the report in words, the
turn's words put back, a redo after a new turn
refused with its reason, and the fork's restore-or-keep question answered while the fork waits
on it. Ctrl-C at an undo's question is no answer: nothing is taken back."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Literal, cast

import pytest
from engines import answer, scripted
from hosting import SID, repository, short_root
from repl_driver import CtrlC, Typed, printed, repl

from chickadee.context.questions import open_reclaimed
from chickadee.context.store import sessions_root
from chickadee.state.roots import roots
from chickadee_cli import fork as forking
from chickadee_cli.commands import Context, NoAnswer, dispatch
from chickadee_cli.interaction import Interaction, answering
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.repl import NO_ANSWER
from chickadee_cli.surface import DONE, USAGE
from chickadee_wire.host import host_in_process
from chickadee_wire.models import Restored, Undoable, UndoReport
from chickadee_wire.questions import OpenQuestion

TURN = 3
"""The first user turn's seq, its number for `/undo`: the turn frame is line 1 and the prefix
line 2."""


def _count(out: list[str], text: str) -> int:
    return sum(text in line for line in out)


def _turn_then(*typed: Typed) -> list[Typed]:
    """One turn, its answer seen, then `typed`, then `/exit`."""
    return [Typed("hello there"), *typed, Typed("/exit")]


def test_undo_lists_asks_and_puts_the_turns_words_back(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = _turn_then(
        Typed("/undo", after=printed("hi back")),
        Typed(str(TURN), after=printed(f"turn {TURN}: hello there")),
    )
    with repl(monkeypatch, [answer("hi back")], lines) as (source, _code):
        out = "\n".join(source.out)
    assert f"  turn {TURN}: hello there" in out
    assert "no turn from here on changed code" in out, "why only the conversation is offered"
    assert f"undone from turn {TURN}: conversation" in out
    assert source.defaults == ["hello there"], "the turn's words are put back to edit"


def test_a_redo_after_a_new_turn_is_refused_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _turn_then(
        Typed(f"/undo {TURN}", after=printed("hi back")),
        Typed("/redo", after=printed(f"undone from turn {TURN}")),
        Typed(f"/undo {TURN}", after=printed(f"redone from turn {TURN}: conversation")),
        Typed(
            "and now this",
            after=lambda out: _count(out, "no turn from here on changed code") == 2,
        ),
        Typed("/redo", after=printed("second answer")),
    )
    script = [answer("hi back"), answer("second answer")]
    with repl(monkeypatch, script, lines) as (source, _code):
        out = " ".join(source.out)  # a pipe wraps at 80
    assert "/redo: " in out and "its conversation moved on at seq" in out


def test_undo_task_and_operation_reach_the_engine_and_say_its_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both go over the wire to the engine, whose refusal (no such run, no such
    operation) comes back as its reason, the operation's run included. A malformed
    `/undo task` or `/undo <operation>` prints its usage."""
    lines = _turn_then(
        Typed("/undo task R-none T1", after=printed("hi back")),
        Typed("/undo op-none", after=printed("/undo: ")),
        Typed("/undo task R-none", after=lambda out: _count(out, "/undo: ") == 2),
        Typed("/undo op-none R-none", after=printed("/undo task <run> <task>")),
        Typed("/undo op-none R-none extra", after=lambda out: _count(out, "/undo: ") == 3),
        Typed("/undo", after=printed("/undo <operation> [<run>]")),
        Typed("never mind", after=printed(f"turn {TURN}: hello there")),
    )
    with repl(monkeypatch, [answer("hi back")], lines) as (source, _code):
        out = "\n".join(source.out)
    joined = " ".join(source.out)  # a pipe wraps at 80
    assert "not on the wire" not in out
    assert "/undo: no run R-none in this session" in joined, out
    assert "/undo: no run of this session journaled operation op-none" in joined, out
    assert joined.count("/undo: no run R-none in this session") == 2, out
    assert "undo cancelled" in out
    assert source.defaults == []


class _Terminal:
    """The person at a terminal, answering each question in turn."""

    def __init__(self, answers: Sequence[str]) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def read(self, prompt_text: str, default: str = "") -> str:
        self.asked.append(prompt_text)
        if not self.answers:
            raise EOFError  # Ctrl-D
        return self.answers.pop(0)


def _fork(
    monkeypatch: pytest.MonkeyPatch, source: _Terminal | None, at: int | None = None
) -> tuple[int, str]:
    """Fork `SID`, whose history is one answered turn written before the host starts: a fork
    is made between turns, and nothing here is running one."""
    said: list[str] = []

    def say(text: str, tone: Tone) -> None:
        said.append(text)

    repo = repository(monkeypatch, short_root())
    with open_reclaimed(sessions_root(roots(repo)), SID) as session:
        session.record_user("hello there")
        session.record_assistant("hi back")
    scripted(monkeypatch, ())
    with host_in_process(repo) as host, host.client() as client:
        code = forking.run(
            client, SID, at=at, reason=None, source=source, say=say, style=Style(tty=False)
        )
    return code, " ".join(said)


def test_fork_asks_restore_or_keep_and_shows_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    person = _Terminal(["restore"])
    code, said = _fork(monkeypatch, person)
    assert code == DONE, said
    assert person.asked == ["? "], "asked once, while the fork waited on the answer"
    assert "put the code back as it was then, or keep it" in said
    assert "forked: " in said and "(code restore)" in said


def test_fork_off_a_terminal_keeps_the_code_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, said = _fork(monkeypatch, None)
    assert code == DONE, said
    assert forking.NOBODY_TO_ASK in said and "(code keep)" in said


def test_fork_at_an_unsafe_point_is_refused_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, said = _fork(monkeypatch, None, at=9999)
    assert code == USAGE
    assert said.startswith(forking.NOBODY_TO_ASK) and "chickadee fork: " in said


def test_ctrl_d_at_the_fork_question_takes_its_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fork is under way on its thread: ending the input answers `keep`, as Enter would,
    rather than leaving the fork to run under a host that is shutting down."""
    code, said = _fork(monkeypatch, _Terminal([]))
    assert code == DONE, said
    assert "(code keep)" in said


# -- the edges, without an engine -------------------------------------------------------------


class _SettledElsewhere:
    """`Interaction`'s surface `answering` reads: one open question, closed elsewhere while
    the person is typing."""

    def __init__(self) -> None:
        self.question: OpenQuestion | None = OpenQuestion("q1", 4, None, None)
        self.sent: list[str] = []
        self.noted: list[str] = []

    @property
    def current(self) -> OpenQuestion | None:
        return self.question

    def answer(self, line: str) -> str | None:
        self.sent.append(line)
        return "q1"

    def note(self, text: str) -> None:
        self.noted.append(text)


def test_a_line_for_a_question_settled_meanwhile_is_not_sent_and_is_said() -> None:
    asked = _SettledElsewhere()
    gate = threading.Event()

    def read(_prompt: str) -> str:
        asked.question = None  # another client answered it while this one typed
        gate.set()
        return "main"

    def call() -> str:
        assert gate.wait(5)
        return "done"

    assert answering(call, cast(Interaction, asked), read) == "done"
    assert asked.sent == [], "the line never lands on a question it was not typed for"
    assert asked.noted == ["q1 was settled meanwhile; that line was not used"]


class _Undo:
    """A `Context` whose session has one turn of two lines to take back."""

    def __init__(self) -> None:
        self.said: list[str] = []
        self.back: list[str] = []
        self.client = self

    session_id = SID
    style = Style(tty=False)

    def undoable(self, _sid: str) -> list[Undoable]:
        return [Undoable(turn=3, prompt="first line", options=["conversation"])]

    def undo(self, _sid: str, _turn: int, **_how: object) -> UndoReport:
        return UndoReport(
            seq=9, to_turn=3, code=False, conversation=True, prompt="first line\n  second"
        )

    def say(self, text: str, tone: Tone = Tone.PLAIN, *, indent: str = "") -> None:
        self.said.append(text)

    def answering(self, call: Callable[[], UndoReport]) -> UndoReport:
        return call()

    def put_back(self, text: str) -> None:
        self.back.append(text)


def test_a_turn_of_several_lines_comes_back_as_one_and_says_so() -> None:
    ctx = _Undo()
    dispatch(cast(Context, ctx), "/undo 3")
    assert ctx.back == ["first line second"]
    assert "(its 2 lines are joined into one to edit)" in ctx.said


class _Restore(_Undo):
    """A `Context` whose engine restored one task's file and kept the person's edit."""

    def undo_task(self, _sid: str, _run: str, _task: str) -> Restored:
        return Restored(root="/wt", restored=["app.py"], kept=["mine.py"])


def test_a_task_undo_reads_in_words() -> None:
    ctx = _Restore()
    dispatch(cast(Context, ctx), "/undo task R-1 T1")
    assert ctx.said == ["restored in /wt", "restored: app.py", "kept: mine.py"]


class _Halves(_Undo):
    """A `Context` whose turn offers all three halves, recording what it is asked and sent."""

    def __init__(self, options: list[Literal["both", "conversation", "code"]]) -> None:
        super().__init__()
        self.options = options
        self.asked: list[str] = []
        self.how: dict[str, object] | None = None

    def undoable(self, _sid: str) -> list[Undoable]:
        return [Undoable(turn=3, prompt="first line", options=self.options)]

    def undo(self, _sid: str, _turn: int, **how: object) -> UndoReport:
        self.how = how
        return UndoReport(seq=9, to_turn=3, code=True, conversation=False)

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        return ""


def test_a_turn_given_with_its_halves_is_not_asked_them() -> None:
    """The TUI's undo screen chose the halves already and sends one line."""
    ctx = _Halves(["both", "conversation", "code"])
    dispatch(cast(Context, ctx), "/undo 3 code")
    assert ctx.asked == [], "no question"
    assert ctx.how == {"code": True, "conversation": False, "commit": None}
    assert "undone from turn 3: code" in ctx.said


def test_halves_a_turn_does_not_offer_are_refused_by_name() -> None:
    ctx = _Halves(["conversation"])
    dispatch(cast(Context, ctx), "/undo 3 code")
    assert ctx.how is None, "nothing undone"
    assert "turn 3 offers conversation, not code" in ctx.said


class _NoAnswerHalves(_Halves):
    """Ctrl-C or Ctrl-D at the halves question: no answer at all (`commands.NoAnswer`)."""

    def ask(self, prompt: str) -> str:
        self.asked.append(prompt)
        raise NoAnswer


def test_no_answer_at_the_halves_question_undoes_nothing() -> None:
    """An empty line takes the default, `both`; Ctrl-C is no line, so nothing is taken back."""
    ctx = _NoAnswerHalves(["both", "conversation", "code"])
    with pytest.raises(NoAnswer):
        dispatch(cast(Context, ctx), "/undo 3")
    assert ctx.asked, "it was asked"
    assert ctx.how is None, "nothing undone"


def test_ctrl_c_at_an_undo_question_says_nothing_was_sent_and_the_repl_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _turn_then(
        Typed("/undo", after=printed("hi back")),
        Typed(CtrlC(), after=printed(f"turn {TURN}: hello there")),
    )
    with repl(monkeypatch, [answer("hi back")], lines) as (source, code):
        out = "\n".join(source.out)
    assert NO_ANSWER in out and "undone" not in out
    assert code == 0, "the REPL went on to /exit"
