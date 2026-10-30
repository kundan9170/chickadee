"""The REPL's live view, against recorded frames: Zeus's
words printed line by line and never twice; a gap marked and filled; the thinking line; a tool
call's live line past half a second; a retry and a stop; the box's words in the toolbar; the
working indicator; and nothing live to a pipe."""

from __future__ import annotations

from chickadee_cli.render.live import GAP, STOPPED, TOOL_AFTER_S, LiveView
from chickadee_cli.render.stream import Verbosity, frame_lines
from chickadee_cli.render.style import Style
from chickadee_serve.frames import JsonValue, ProgressFrame, RecordFrame

PLAIN = Style(tty=True)


class _ViewClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _view(verbosity: Verbosity = Verbosity.NORMAL) -> tuple[LiveView, list[str], _ViewClock]:
    out: list[str] = []
    clock = _ViewClock()
    return LiveView(out.append, PLAIN, lambda: verbosity, clock), out, clock


def _text(
    offset: int, piece: str, *, role: str = "zeus", task: str | None = None, reply: str = "z1"
) -> ProgressFrame:
    payload: dict[str, JsonValue] = {
        "role": role,
        "task": task,
        "reply": reply,
        "offset": offset,
        "piece": piece,
        "tokens": (offset + len(piece)) // 4,
    }
    return ProgressFrame(scope="S", kind="text", payload=payload)


def _assistant(text: str) -> RecordFrame:
    return RecordFrame(
        scope="S", seq=9, at="t", kind="turn", payload={"role": "assistant", "text": text}
    )


def _printed(view: LiveView, record: RecordFrame) -> list[str]:
    return frame_lines(view.record(record), Verbosity.NORMAL, PLAIN)


def test_zeus_lines_print_as_they_finish_and_the_record_prints_only_the_rest() -> None:
    view, out, _ = _view()
    view.progress(_text(0, "first line\nsec"))
    assert out == ["first line"]
    assert any("zeus · sec" in line for line in view.bottom()), "the line being written"
    view.progress(_text(14, "ond line\nthird"))
    assert out == ["first line", "second line"]
    rest = _printed(view, _assistant("first line\nsecond line\nthird"))
    assert rest == ["third"], "nothing printed twice"
    assert not any("zeus · third" in line for line in view.bottom())


def test_a_lost_frame_is_marked_and_the_record_is_printed_whole() -> None:
    view, out, _ = _view()
    view.progress(_text(0, "one\n"))
    view.progress(_text(10, "three\n"))  # 4..10 lost
    assert out == ["one", f"{GAP}three"]
    whole = _printed(view, _assistant("one\ntwo...\nthree\n"))
    assert whole == ["one", "two...", "three"], "the record fills the gap"


def test_a_late_frame_of_a_finished_reply_is_not_drawn_again() -> None:
    view, out, _ = _view()
    view.progress(_text(0, "done\n"))
    _printed(view, _assistant("done\n"))
    view.progress(_text(0, "done\n"))
    assert out == ["done"]


def test_thinking_is_one_line_and_its_reasoning_shows_at_v_only() -> None:
    thought = ProgressFrame(
        scope="S",
        kind="thought",
        payload={
            "role": "zeus",
            "task": None,
            "reply": "z1",
            "elapsed_s": 12.0,
            "offset": 0,
            "piece": "weigh it\n",
            "tokens": 3,
        },
    )
    quiet, out, _ = _view()
    quiet.progress(thought)
    assert out == []
    assert any("zeus · thinking… 12s" in line for line in quiet.bottom())
    loud, out_v, _ = _view(Verbosity.VERBOSE)
    loud.progress(thought)
    assert out_v == ["\x1b[2m  weigh it\x1b[0m"]


def _tool(size: int, *, reply: str = "z1") -> ProgressFrame:
    return ProgressFrame(
        scope="S",
        kind="tool_started",
        payload={
            "role": "zeus",
            "task": None,
            "reply": reply,
            "index": 0,
            "tool": "create",
            "path": "core/ai.py",
            "size": size,
            "tokens": size // 4,
        },
    )


def test_a_tool_call_shows_its_live_line_after_half_a_second_and_grows() -> None:
    view, _, clock = _view()
    view.progress(_tool(100))
    assert not any("→ create" in line for line in view.bottom()), "a short call never flashes"
    clock.now += TOOL_AFTER_S
    view.progress(_tool(4200))
    assert any("→ create core/ai.py · writing… 4.1 KB" in line for line in view.bottom())
    view.record(_assistant(""))
    assert not any("→ create" in line for line in view.bottom()), "the tool line replaces it"


def test_a_retry_clears_the_live_text() -> None:
    view, _, _ = _view()
    view.progress(_text(0, "half a th"))
    view.progress(
        ProgressFrame(
            scope="S",
            kind="text",
            payload={"role": "zeus", "task": None, "reply": "z1", "restart": True},
        )
    )
    assert not any("half a th" in line for line in view.bottom())


def test_a_stop_leaves_what_was_shown_marked() -> None:
    view, out, _ = _view()
    view.progress(_text(0, "line\nhalf"))
    view.stopped()
    assert out == ["line", f"half{STOPPED}"]


def test_the_box_speaks_in_the_toolbar_and_its_record_ends_it() -> None:
    view, out, _ = _view()
    view.progress(_text(0, "looking at the tests\nnow", role="worker", task="T1", reply="w1"))
    assert out == [], "the box's words are never printed as lines"
    assert any("worker (T1) · now" in line for line in view.bottom())
    model_call = RecordFrame(
        scope="R-1",
        seq=4,
        at="t",
        kind="model_call",
        run_id="R-1",
        task_id="T1",
        payload={
            "role": "worker",
            "request": "task",
            "tokens_in": 1,
            "tokens_out": 1,
            "cost_usd": 0.0,
            "stop_reason": "end_turn",
            "model": "m",
            "rung": "worker",
        },
    )
    view.record(model_call)
    assert not any("worker (T1) · now" in line for line in view.bottom())


def test_the_working_indicator_names_who_and_counts_time_and_tokens() -> None:
    view, _, clock = _view()
    view.progress(_text(0, "x" * 40, reply="z1"))
    clock.now += 72
    (indicator,) = [line for line in view.bottom() if "working" in line]
    assert "zeus · writing · working 1m 12s · 10 tokens · Ctrl-C to stop" in indicator
    view.progress(ProgressFrame(scope="S", kind="answer", payload={"seq": 3}))
    assert not any("working" in line for line in view.bottom())


def test_zeus_context_alone_never_starts_the_working_indicator() -> None:
    """A `context` frame (a compaction's start, say) is the status line's: on its own it is
    no turn running."""
    view, _, _ = _view()
    view.progress(
        ProgressFrame(
            scope="S",
            kind="context",
            payload={"used": 1, "window": 10, "model": "m", "compacting": True},
        )
    )
    assert view.bottom() == []
