"""The REPL against a hosted engine: words become turns and the
answer streams back; a question is shown and answered by the next line; words sent while a
turn runs are queued; and the slash commands answer over the wire."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from engines import Step, answer, call, calls
from hosting import SID, hosted
from repl_driver import CtrlC, Script, Typed, printed, repl

from chickadee.skills.discover import RESERVED_NAMES
from chickadee_cli.commands import COMMANDS
from chickadee_cli.render.style import Style
from chickadee_cli.repl import Repl
from chickadee_serve.frames import ErrorBody, ErrorCode
from chickadee_wire.client import Client
from chickadee_wire.errors import Unanswered, WireError


def test_words_are_a_turn_and_the_answer_streams_back(monkeypatch: pytest.MonkeyPatch) -> None:
    script: list[Step] = [calls(call("read_file", "c1", path="README.md")), answer("all good")]
    lines = [Typed("is the readme fine?"), Typed("/exit", after=printed("all good"))]
    with repl(monkeypatch, script, lines) as (source, code):
        out = "\n".join(source.out)
    assert code == 0
    assert "> is the readme fine?" not in out, "the person's words show once, on their own line"
    assert '→ read_file {"path": "README.md"}\n' in out, (
        "the tool call, at normal verbosity, with no purpose line (D208)"
    )
    assert "all good" in out


def test_a_question_is_answered_by_the_next_line(monkeypatch: pytest.MonkeyPatch) -> None:
    script: list[Step] = [
        calls(call("ask_user", "q1", question="which branch?")),
        answer("on it"),
    ]
    lines = [
        Typed("go"),
        Typed("main, please", after=printed("? which branch?")),
        Typed("/exit", after=printed("on it")),
    ]
    with repl(monkeypatch, script, lines) as (source, _code):
        out = "\n".join(source.out)
    assert "answer in words" in out, "a question with no options asks for words"
    assert "answered q1: main, please, by human" in out, "the token's class says who answered"


def test_words_sent_during_a_turn_are_queued(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(_request: object) -> object:
        started.set()
        release.wait(10)
        return answer("first done")

    script: list[Step] = [slow, answer("second done")]  # type: ignore[list-item]
    lines = [
        Typed("first"),
        Typed("second", after=lambda _out: started.is_set()),
        Typed(
            "/exit",
            after=lambda out: release.set() is None and any("second done" in x for x in out),
        ),
    ]
    with repl(monkeypatch, script, lines) as (source, _code):
        out = "\n".join(source.out)
    assert "(queued: delivered at the running turn's next boundary)" in out
    assert "first done" in out and "second done" in out


def test_ctrl_c_during_a_turn_stops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release = threading.Event(), threading.Event()

    def slow(_request: object) -> object:
        started.set()
        release.wait(10)
        return answer("finished anyway")

    lines = [
        Typed("work"),
        Typed(CtrlC(), after=lambda _out: started.is_set()),
        Typed("/exit", after=lambda out: release.set() is None and printed("stopping")(out)),
    ]
    with repl(monkeypatch, [slow], lines) as (source, _code):  # type: ignore[list-item]
        out = "\n".join(source.out)
    assert "stopping at the next boundary" in out


def test_ctrl_c_at_an_idle_prompt_only_reminds(monkeypatch: pytest.MonkeyPatch) -> None:
    with repl(monkeypatch, [], [Typed(CtrlC()), Typed("/exit")]) as (source, _code):
        assert any("Ctrl-D or /exit" in line for line in source.out)


def test_the_commands_answer_over_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        Typed(text)
        for text in (
            "/status",
            "/budget",
            "/verbose v",
            "/handoff",
            "/diff",
            "/replay R-1",
            "/nope",
            "/help",
        )
    ]
    # One turn first, so a rung has spent something and /budget has a rung to show.
    lines = [Typed("hello"), Typed(lines[0].text, after=printed("hi there")), *lines[1:]]
    lines.append(Typed("/exit"))
    with repl(monkeypatch, [answer("hi there")], lines) as (source, _code):
        out = "\n".join(source.out)
    assert f"session {SID}" in out and "state      idle" in out
    assert "spend      $" in out
    assert not any(line.lstrip().startswith("{") for line in source.out), "no raw JSON"
    assert any("high-param" in line and "no cap of its own" in line for line in source.out), (
        "a rung's line: what it spent, and that it has no cap of its own"
    )
    assert "verbosity v" in out
    assert "no worktree to hand off" in out, "/handoff before any coder call (T-5.5)"
    assert "no run yet" in out, "/diff before the session has run anything"
    assert "not on the wire yet" in out
    assert "no command /nope" in out
    assert "/budget [rung] [usd|off]" in out


def test_the_commands_are_section_6s_table_and_reserved_against_skills() -> None:
    assert set(COMMANDS) == RESERVED_NAMES


def test_a_free_text_answer_may_start_with_a_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    script: list[Step] = [
        calls(call("ask_user", "q1", question="which directory?")),
        answer("on it"),
    ]
    lines = [
        Typed("go"),
        Typed("/etc/app", after=printed("? which directory?")),
        Typed("/exit", after=printed("on it")),
    ]
    with repl(monkeypatch, script, lines) as (source, _code):
        out = "\n".join(source.out)
    assert "answered q1: /etc/app, by human" in out
    assert "no command" not in out


def test_a_refused_answer_is_said_and_the_repl_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise WireError(422, ErrorBody(code=ErrorCode.BAD_REQUEST, message="not a choice"))

    monkeypatch.setattr(Client, "answer", refuse)
    script: list[Step] = [calls(call("ask_user", "q1", question="which branch?"))]
    lines = [
        Typed("go"),
        Typed("main", after=printed("? which branch?")),
        Typed("/exit", after=printed("the answer was refused")),
    ]
    with repl(monkeypatch, script, lines) as (source, code):
        out = "\n".join(source.out)
    assert "the answer was refused: not a choice" in out
    assert code == 0


def test_ctrl_c_stops_a_turn_waiting_on_a_question(monkeypatch: pytest.MonkeyPatch) -> None:
    script: list[Step] = [calls(call("ask_user", "q1", question="which branch?"))]
    lines = [
        Typed("go"),
        Typed(CtrlC(), after=printed("? which branch?")),
        Typed("/exit", after=printed("stopping at the next boundary")),
    ]
    with repl(monkeypatch, script, lines) as (source, _code):
        out = "\n".join(source.out)
    assert "Ctrl-D or /exit" not in out, "a waiting turn is a turn to stop, not an idle prompt"


def test_words_another_client_sent_are_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only this REPL's own words are left to its prompt line; another client's are news."""
    source = Script([Typed("/exit", after=printed("from elsewhere done"))])
    with (
        hosted(monkeypatch, [answer("from elsewhere done")]) as host,
        host.client() as mine,
        host.client() as theirs,
    ):
        session = Repl(
            mine, SID, source=source, write=source.out.append, style=Style(tty=False)
        )
        reader = threading.Thread(target=session.run)
        reader.start()
        theirs.turn(SID, "from another client")
        reader.join(30)
    assert "> from another client" in source.out


def test_a_turn_the_server_does_not_answer_is_said_and_the_repl_carries_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `ReadTimeout` on the first turn crashed the REPL. It is said, the word is not
    counted as sent, and the next line is read."""
    real = Client.turn
    refused: list[str] = []

    def once(self: Client, sid: str, text: str, **kw: Any) -> object:
        if not refused:
            refused.append(text)
            raise Unanswered("POST", f"/session/{sid}/turn", "within 30 s")
        return real(self, sid, text, **kw)

    monkeypatch.setattr(Client, "turn", once)
    script: list[Step] = [answer("here now")]
    lines = [
        Typed("hello?"),
        Typed("again", after=printed("the server did not answer")),
        Typed("/exit", after=printed("here now")),
    ]
    with repl(monkeypatch, script, lines) as (source, code):
        out = "\n".join(source.out)
    assert refused == ["hello?"]
    assert "the server did not answer POST /session/" in out
    assert "here now" in out
    assert code == 0


def test_a_turn_that_may_have_arrived_is_not_called_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timeout after the words were sent is not known to have failed. The REPL says
    to check before resending, and keeps the words noted as sent, so their record, if it
    comes, shows as this person's rather than another client's."""
    real = Client.turn
    late: list[str] = []

    def arrives_late(self: Client, sid: str, text: str, **kw: Any) -> object:
        real(self, sid, text, **kw)
        late.append(text)
        raise Unanswered(
            "POST", f"/session/{sid}/turn", "no answer came within 30 s", maybe_delivered=True
        )

    monkeypatch.setattr(Client, "turn", arrives_late)
    script: list[Step] = [answer("got it anyway")]
    lines = [Typed("hello?"), Typed("/exit", after=printed("got it anyway"))]
    with repl(monkeypatch, script, lines) as (source, code):
        out = "\n".join(source.out)
    assert late == ["hello?"]
    assert "check before sending it again" in out
    assert "got it anyway" in out
    assert code == 0
