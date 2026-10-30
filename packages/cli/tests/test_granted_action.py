"""An action a standing grant approved is one line in the REPL and never a question: no menu is
drawn, nothing waits for an answer, and a line typed afterwards is a chat
line, as the person meant it."""

from __future__ import annotations

from chickadee_cli.interaction import Interaction
from chickadee_cli.render.stream import Verbosity, frame_lines
from chickadee_cli.render.style import Style
from chickadee_serve.frames import JsonValue, RecordFrame

PIPE = Style(tty=False)
GRANTED: dict[str, JsonValue] = {
    "action_id": "shell-f8bdd3406e78",
    "operation_id": "S:c3",
    "display": "ls -la",
    "scope": "repo",
    "cls": "read_only",
    "prefix": ["ls"],
}


def frame(kind: str, payload: dict[str, JsonValue]) -> RecordFrame:
    return RecordFrame(scope="S-1", seq=7, at="t", kind=kind, payload=payload)


def test_a_granted_action_reads_as_one_line() -> None:
    assert frame_lines(frame("action_granted", GRANTED), Verbosity.NORMAL, PIPE) == [
        "  ls -la — approved by your repo grant"
    ]
    assert frame_lines(frame("action_granted", GRANTED), Verbosity.QUIET, PIPE) == []


def test_a_granted_action_opens_no_question() -> None:
    shown: list[str] = []
    questions = Interaction(object(), "S-1", PIPE, shown.append)  # type: ignore[arg-type]
    questions.feed(frame("action_granted", GRANTED))
    assert questions.current is None and shown == []


def test_an_action_never_ask_ran_says_so_and_names_no_grant() -> None:
    """D199 item 8b: `never_ask` is the person's setting, not a grant they gave."""
    never = {**GRANTED, "scope": "never_ask", "cls": "unbound", "prefix": []}
    assert frame_lines(frame("action_granted", never), Verbosity.NORMAL, PIPE) == [
        "  ls -la — run unasked (sandbox.never_ask)"
    ]
