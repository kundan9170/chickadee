"""An open question stays on screen until it is answered: the prompt's toolbar holds
it as one line, and the prompt reads `? ` while it waits.

Was (2026-10-22): a question was shown only as the lines printed when it opened. On a person's
terminal those lines were lost to the redraw (the transcript held them, the screen did not), and
the turn waited on an answer nobody saw: the session looked stopped.
"""

from __future__ import annotations

from typing import Any, cast

from chickadee_cli.interaction import WAITING_MAX, Interaction
from chickadee_cli.render.style import Style
from chickadee_cli.repl import ASKING_PROMPT, PROMPT, InputSource, Repl
from chickadee_serve.frames import RecordFrame
from chickadee_wire.client import Client

PIPE = Style(tty=False)
TTY = Style(tty=True)
TEXT = "When a fighter gets hit hard, do you want a real knockdown?"


def _question(text: str = TEXT, qid: str = "q1") -> dict[str, Any]:
    return {
        "question": {
            "id": qid,
            "kind": "clarify_goal",
            "text": text,
            "options": [],
            "default": "no answer: proceed on the default the question states",
            "blocking": True,
            "evidence": [],
        }
    }


def _frame(seq: int, kind: str, payload: dict[str, Any]) -> RecordFrame:
    return RecordFrame(
        scope="S", seq=seq, at="2026-10-22T13:26:33Z", kind=kind, payload=payload
    )


def _asked(seq: int, payload: dict[str, Any]) -> RecordFrame:
    return _frame(seq, "question_asked", payload)


def _answered(seq: int, qid: str) -> RecordFrame:
    payload = {"id": qid, "choice": "x", "elapsed_ms": 1, "source": "person"}
    return _frame(seq, "question_answered", payload)


def _interaction() -> Interaction:
    return Interaction(cast("Client", object()), "S", PIPE, lambda _line: None)


def test_nothing_waits_until_a_question_opens_and_after_it_is_answered() -> None:
    asking = _interaction()
    assert asking.waiting() is None
    asking.feed(_asked(1, _question()))
    assert asking.waiting() == f"? waiting for your answer: {TEXT}"
    asking.feed(_answered(2, "q1"))
    assert asking.waiting() is None


def test_a_long_question_is_one_line_cut_to_the_most_the_toolbar_shows() -> None:
    asking = _interaction()
    asking.feed(_asked(1, _question("first line\n\nsecond " + "x" * 400)))
    shown = asking.waiting()
    assert shown is not None and "\n" not in shown
    what = shown.removeprefix("? waiting for your answer: ")
    assert len(what) == WAITING_MAX and what.startswith("first line second ")
    assert what.endswith("…")


def test_a_stop_waits_as_its_head() -> None:
    stop = {
        "id": "stop-1",
        "kind": "plan",
        "author": "athena",
        "text": "# Plan\n",
        "sha": "a" * 64,
        "round": 1,
        "workspace": "worktree",
    }
    asking = _interaction()
    asking.feed(_asked(1, {"stop": stop}))
    assert asking.waiting() == "? waiting for your answer: athena asks you to approve the plan"


def _repl() -> Repl:
    return Repl(
        cast("Client", object()),
        "S",
        source=cast("InputSource", object()),
        write=lambda _line: None,
        style=TTY,
    )


def test_the_toolbar_leads_with_the_waiting_question_and_the_prompt_turns() -> None:
    repl = _repl()
    assert repl.prompt_mark() == PROMPT
    assert "waiting for your answer" not in repl.toolbar()
    repl._interaction.feed(_asked(1, _question()))  # pyright: ignore[reportPrivateUsage]
    assert repl.prompt_mark() == ASKING_PROMPT
    first = repl.toolbar().split("\n")[0]
    assert f"? waiting for your answer: {TEXT}" in first
    repl._interaction.feed(_answered(2, "q1"))  # pyright: ignore[reportPrivateUsage]
    assert repl.prompt_mark() == PROMPT
    assert "waiting for your answer" not in repl.toolbar()


def test_a_stop_from_zeus_still_offers_the_other_size() -> None:
    """`stop_head` came out of `stop_lines`; the line below the head still reads the kind."""
    from chickadee_cli.approval import stop_lines
    from chickadee_wire.models import ApprovalStop

    stop = ApprovalStop.model_validate(
        {
            "id": "stop-2",
            "kind": "plan",
            "author": "zeus",
            "text": "# Plan\n",
            "sha": "b" * 64,
            "round": 1,
            "workspace": "worktree",
        }
    )
    lines = stop_lines(stop, PIPE)
    assert "zeus asks you to approve the one-task plan" in lines[0]
    assert "n <words> asks for a design the team plans instead" in lines[1]
