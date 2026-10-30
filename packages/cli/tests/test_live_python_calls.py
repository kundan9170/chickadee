"""The live view under Python tool calling (`chickadee/pycall.py`): a worker's prose printed
whole as its words, as Zeus's are, and its block printed as its call; nothing printed twice,
and nothing changed without the mode."""

from __future__ import annotations

import pytest

from chickadee_cli.render.live import LiveView
from chickadee_cli.render.stream import Verbosity, frame_lines
from chickadee_cli.render.style import Style
from chickadee_serve.frames import JsonValue, ProgressFrame, RecordFrame

TTY = Style(tty=True)
DIM = "\x1b[2m{}\x1b[0m"

REPLY = (
    "I will read the file first.\n"
    "Then I can edit the return value.\n"
    "```python\n"
    'read(path="app.py")\n'
    "```\n"
)


def _view(python_calls: bool = True) -> tuple[LiveView, list[str]]:
    out: list[str] = []
    view = LiveView(out.append, TTY, lambda: Verbosity.NORMAL, python_calls=python_calls)
    return view, out


def _text(offset: int, piece: str) -> ProgressFrame:
    payload: dict[str, JsonValue] = {
        "role": "worker",
        "task": "T1",
        "reply": "w1",
        "offset": offset,
        "piece": piece,
        "tokens": 1,
    }
    return ProgressFrame(scope="S", kind="text", payload=payload)


def _model_call(preview: str | None, tools: tuple[str, ...] = ("python",)) -> RecordFrame:
    return RecordFrame(
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
            "stop_reason": "tool_use",
            "tools": list(tools),
            "text_preview": preview,
            "model": "m",
            "rung": "worker",
        },
    )


def _stream(view: LiveView, text: str, step: int = 7) -> None:
    for i in range(0, len(text), step):
        view.progress(_text(i, text[i : i + step]))


def test_the_prose_prints_whole_as_it_is_written_and_the_block_as_the_call() -> None:
    view, out = _view()
    _stream(view, REPLY)
    assert out == [
        DIM.format("worker (T1):"),
        "I will read the file first.",
        "Then I can edit the return value.",
    ], "the prose, line by line, and none of the block"
    assert any("→ python · writing…" in line for line in view.bottom())
    printed = frame_lines(
        view.record(_model_call("I will read the file first. Then I can edit…")),
        Verbosity.NORMAL,
        TTY,
    )
    assert out[3:] == [DIM.format("    → python"), DIM.format('        read(path="app.py")')]
    assert not any("I will read" in line for line in printed), "never printed twice"
    assert any("worker (T1) · task" in line for line in printed), "the call's own line stays"
    assert all("working" in line for line in view.bottom()), "only the turn's indicator"


def test_a_last_line_with_no_newline_is_printed_when_the_reply_ends() -> None:
    view, out = _view()
    _stream(view, "Done, reporting.")
    assert out == []
    view.record(_model_call("Done, reporting."))
    assert out == [DIM.format("worker (T1):"), "Done, reporting."]


def test_a_reply_that_is_only_a_block_prints_no_label() -> None:
    view, out = _view()
    _stream(view, '```python\nfinish(report="r")\n```')
    printed = frame_lines(view.record(_model_call(None)), Verbosity.NORMAL, TTY)
    assert out == [DIM.format("    → python"), DIM.format('        finish(report="r")')]
    assert any("worker (T1) · task" in line for line in printed)


def test_without_the_mode_the_workers_words_stay_in_the_toolbar() -> None:
    view, out = _view(python_calls=False)
    _stream(view, REPLY)
    assert out == []
    native = _model_call("I will read…", tools=("read",))
    printed = frame_lines(view.record(native), Verbosity.NORMAL, TTY)
    assert out == []
    assert any("I will read" in line for line in printed), "the preview, as before"


@pytest.mark.parametrize(
    "reply",
    [
        REPLY,
        "```Python\nread(path='a')\n```",
        "~~~py3\nread(path='a')\n~~~",
        "Two.\n```python\nread(path='a')\n```\n```python\nread(path='b')\n```",
        "```\nread(path='a')\n```",
        "```bash\nls\n```\n```\nread(path='a')\n```",
        "```python\nread(path='a')",
        "No block here.",
        'grep(pattern="TODO")',
        "Run it:\n```bash\nls\n```\nThen:\n```python\nread(path='a')\n```\nDone.",
    ],
)
def test_the_repl_reads_the_blocks_the_engine_reads(reply: str) -> None:
    """The REPL never imports the engine, so its reading of a reply's blocks and words is held
    equal to the engine's here."""
    from chickadee import pycall
    from chickadee_cli.render.live import _split  # pyright: ignore[reportPrivateUsage]

    prose, codes = _split(reply, whole=True)
    engine_prose, engine_codes = pycall.split(reply, {"read", "grep"})
    assert codes == [b.rstrip() for b in engine_codes]
    assert prose.strip() == engine_prose


def test_the_mode_is_learnt_from_the_engines_model_call() -> None:
    """The config is the engine's: the REPL learns the mode from the first
    worker `model_call` whose tool is `python`, and prints the worker's words from then on."""
    view, out = _view(python_calls=False)
    _stream(view, REPLY)
    view.record(_model_call("I will read the file first. Then I can edit…"))
    assert DIM.format("    → python") in out
    out.clear()
    view.progress(
        ProgressFrame(
            scope="S",
            kind="text",
            payload={
                "role": "worker",
                "task": "T1",
                "reply": "w2",
                "offset": 0,
                "piece": "Next, the edit.\n",
                "tokens": 1,
            },
        )
    )
    assert out == [DIM.format("worker (T1):"), "Next, the edit."]


def test_a_block_written_with_an_indent_is_still_the_call() -> None:
    """The instructions' examples are indented, and a model copies the indent: the block is
    still its call, never printed as its words."""
    view, out = _view()
    _stream(view, 'Reading.\n    ```python\n    read(path="a.py")\n    ```\n')
    view.record(_model_call("Reading."))
    assert out == [
        DIM.format("worker (T1):"),
        "Reading.",
        DIM.format("    → python"),
        DIM.format('        read(path="a.py")'),
    ]


def test_words_after_a_block_and_around_another_fence_are_kept() -> None:
    reply = (
        "Run it like this:\n```bash\nmake test\n```\nThen I read.\n"
        "```python\nread(path='a')\n```\nAnd after.\n"
    )
    view, out = _view()
    _stream(view, reply)
    view.record(_model_call("Run it like this…"))
    assert out == [
        DIM.format("worker (T1):"),
        "Run it like this:",
        "```bash",
        "make test",
        "```",
        "Then I read.",
        "And after.",
        DIM.format("    → python"),
        DIM.format("        read(path='a')"),
    ]


def test_a_reply_that_is_only_code_with_no_fence_is_the_call() -> None:
    """The engine runs a fenceless reply that is Python calling a tool as a block; its words
    are not printed as it streams, since its first line is code."""
    view, out = _view()
    _stream(view, 'grep(pattern="TODO")\n')
    assert out == []
    view.record(_model_call(None))
    assert out == [DIM.format("    → python"), DIM.format('        grep(pattern="TODO")')]


def test_a_reply_whose_fence_comes_later_streams_its_words() -> None:
    """`Next` alone compiles as Python; once a fence is seen, the reply is not fenceless."""
    from chickadee_cli.render.live import _may_be_code  # pyright: ignore[reportPrivateUsage]

    assert _may_be_code("Next")
    assert not _may_be_code("Next\n```python\nread(path='a')")
    view, out = _view()
    _stream(view, "Next\n```python\nread(path='a')\n```\n")
    assert out == [DIM.format("worker (T1):"), "Next"]


def test_the_first_reply_is_shown_as_python_from_its_first_piece() -> None:
    """The engine says `calls: "python"` on every live piece, so the
    session's first reply prints its words live, before any record has said the mode."""
    view, out = _view(python_calls=False)
    for i in range(0, len(REPLY), 7):
        frame = _text(i, REPLY[i : i + 7])
        view.progress(
            frame.model_copy(update={"payload": {**frame.payload, "calls": "python"}})
        )
    assert out == [
        DIM.format("worker (T1):"),
        "I will read the file first.",
        "Then I can edit the return value.",
    ]


def test_a_report_with_a_fence_shows_it_as_words_not_a_call() -> None:
    """At report time the engine makes no call: the record names none, so a fence in the
    report is shown as its words."""
    view, out = _view()
    _stream(view, "All done.\n```python\nprint('example')\n```\n")
    view.record(_model_call("All done.", tools=()))
    assert DIM.format("    → python") not in out
    assert "    print('example')" in out


def test_a_fenceless_reply_with_words_left_is_words() -> None:
    """The engine left the reply's words in its record: it was no block (`sorted(x)` calls no
    offered tool), so it is not shown as a call."""
    view, out = _view()
    _stream(view, "sorted(x)\n")
    view.record(_model_call("sorted(x)"))
    assert DIM.format("    → python") not in out
    assert "sorted(x)" in out
