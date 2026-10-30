"""Pins of what the REPL showed in older versions, kept as the record of what moved.

1. A reply's words appear only once the turn's record arrives: no progress frame carries text.
2. A long `create` shows nothing until its tool line: a progress frame names a tool, never its
   arguments or how far they have streamed.
3. Progress frames carry Zeus's six turn events and the keep-alive only.

Each pin is moved by D212 and asserted here in its new form beside the old one, so the move is
read in one place."""

from __future__ import annotations

from typing import cast

from chickadee_cli.render.live import LiveView
from chickadee_cli.render.stream import Verbosity, frame_lines
from chickadee_cli.render.style import Style
from chickadee_cli.repl import InputSource, Repl
from chickadee_serve.frames import PROGRESS_KINDS, JsonValue, ProgressFrame, RecordFrame
from chickadee_wire.client import Client

TTY = Style(tty=True)
BEFORE_D212 = ("alive", "thinking", "tool_call", "tool_result", "delegation", "answer", "error")


def _record(kind: str, payload: dict[str, JsonValue]) -> RecordFrame:
    return RecordFrame(scope="S", seq=1, at="t", kind=kind, payload=payload)


def test_pin_1_the_words_arrive_with_the_turns_record() -> None:
    """Before D212, nothing but the `turn` record printed the assistant's words."""
    thinking = ProgressFrame(scope="S", kind="thinking", payload={"seq": 1})
    for verbosity in Verbosity:
        assert not any("all good" in line for line in frame_lines(thinking, verbosity, TTY))
    turn = _record("turn", {"role": "assistant", "text": "all good"})
    assert any("all good" in line for line in frame_lines(turn, Verbosity.NORMAL, TTY))


def test_pin_2_a_tool_call_frame_names_the_tool_only() -> None:
    """Before D212, a call's progress named its tool, at `-v` only."""
    call = ProgressFrame(
        scope="S", kind="tool_call", payload={"tool": "create", "call_id": "c"}
    )
    assert frame_lines(call, Verbosity.NORMAL, TTY) == []
    assert frame_lines(call, Verbosity.VERBOSE, TTY) == ["\x1b[2m  … tool_call create\x1b[0m"]


def test_pin_3_the_progress_kinds() -> None:
    """Before D212: the keep-alive and Zeus's six turn events. D212 adds to them, never
    removes one."""
    assert set(BEFORE_D212) <= set(PROGRESS_KINDS)


# -- each pin, moved by D212 ------------------------------------------------------------------


def test_pin_1_moved_the_words_arrive_as_they_are_written() -> None:
    """A `text` frame carries the words, and the live view prints each finished line
    before the record arrives; the record then prints only the rest."""
    out: list[str] = []
    view = LiveView(out.append, TTY, lambda: Verbosity.NORMAL)
    piece: dict[str, JsonValue] = {"role": "zeus", "task": None, "reply": "z", "offset": 0}
    view.progress(ProgressFrame(scope="S", kind="text", payload={**piece, "piece": "all\n"}))
    assert out == ["all"]
    rest = view.record(_record("turn", {"role": "assistant", "text": "all\ngood"}))
    assert frame_lines(rest, Verbosity.NORMAL, TTY) == ["good"]


def test_pin_2_moved_a_call_being_written_says_its_path_and_size() -> None:
    """`tool_started` names the tool, the path once it can be read, and the size so far;
    never the arguments."""
    frame = ProgressFrame(
        scope="S",
        kind="tool_started",
        payload={
            "role": "zeus",
            "reply": "z",
            "index": 0,
            "tool": "create",
            "path": "a.py",
            "size": 9,
        },
    )
    assert set(frame.payload) == {"role", "reply", "index", "tool", "path", "size"}
    assert frame_lines(frame, Verbosity.VERBOSE, TTY) == [], "drawn live, never as a line"


def test_pin_3_moved_the_progress_kinds() -> None:
    assert set(PROGRESS_KINDS) == {*BEFORE_D212, "text", "thought", "tool_started", "context"}


def test_a_pipe_draws_nothing_live() -> None:
    """Piped, the final records print as before, and no toolbar is drawn."""
    repl = Repl(
        cast("Client", object()),
        "S",
        source=cast("InputSource", object()),
        write=lambda _line: None,
        style=Style(tty=False),
    )
    assert repl.toolbar() == ""
