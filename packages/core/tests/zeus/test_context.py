"""Context discipline (invariants 13 and 14).

No tool returns bulk content: every answer is bounded by the tool that produced it and the
rest is behind a handle that holds all of it. Source text enters only through an explicit
`read` — a file's body reaches the model when a call asks for its lines, and not because some
other tool happened to include them."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.client.request import text_of
from chickadee.context.results import ToolFailure, UnresolvedSearch
from chickadee.schema import ToolCall
from chickadee.tools.spec import (
    INLINE_BYTES_DEFAULT,
    RESULT_LINES_DEFAULT,
    SHELL_INLINE_BYTES_DEFAULT,
)

from .zeus_helpers import ScriptedGraph, ScriptedPerson, call, calls, rig

MARKER = "SECRET_SAUCE_BODY_7f3a"
SOURCE = (
    "".join(f"FILLER_{n} = {n}\n" for n in range(1, 100))
    + f"def outer():\n    return '{MARKER}'\n"
    + "".join(f"TAIL_{n} = {n}\n" for n in range(1, 100))
)
"""Two hundred lines with the body in the middle: past the size at which a preview is the
whole file, so a read with no span shows its edges."""

BIG = "".join(f"hit {n}: src/module_{n}.py:{n} some context\n" for n in range(5000))


@pytest.mark.parametrize(
    ("tool", "args", "cap"),
    [
        ("atlas", {}, INLINE_BYTES_DEFAULT),
        ("symbol", {"name": "x"}, INLINE_BYTES_DEFAULT),
        ("neighbors", {"target": "x"}, INLINE_BYTES_DEFAULT),
        ("impact", {"symbols": ["x"]}, INLINE_BYTES_DEFAULT),
    ],
)
def test_a_navigate_answer_is_bounded_and_the_rest_is_behind_its_handle(
    tmp_path: Path, tool: str, args: dict[str, object], cap: int
) -> None:
    asked = ToolCall(id="n1", name=tool, arguments=dict(args))
    r = rig(tmp_path, [calls(asked)], graph=ScriptedGraph(big=BIG))
    r.run("map it")
    line = r.session.state.result("n1")
    assert line is not None and line.handle is not None
    assert len(line.text.encode()) < cap + 400, "bounded by the tool that produced it"
    assert "[showing lines 1-" in line.text and line.handle.id in line.text
    assert r.session.blob(line.handle.content_sha).decode() == BIG, "the handle holds all of it"


def test_a_line_bounded_answer_keeps_result_lines_and_names_the_total(tmp_path: Path) -> None:
    r = rig(tmp_path, [calls(call("grep", "f1", pattern="hit"))], graph=ScriptedGraph(big=BIG))
    r.run("find hits")
    line = r.session.state.result("f1")
    assert line is not None
    shown = [t for t in line.text.splitlines() if t.startswith("hit ")]
    assert len(shown) == RESULT_LINES_DEFAULT
    assert "of 5,000 matches not shown; narrow the pattern" in line.text


def test_shell_output_is_capped_and_the_rest_is_behind_its_handle(tmp_path: Path) -> None:
    big = tmp_path / "ws" / "big.txt"
    big.parent.mkdir()
    big.write_text(BIG, encoding="utf-8")
    r = rig(tmp_path, [calls(call("shell", "s1", command="cat big.txt"))], shell=True)
    r.run("show it")
    line = r.session.state.result("s1")
    assert line is not None and line.handle is not None
    assert len(line.text.encode()) < SHELL_INLINE_BYTES_DEFAULT + 400
    assert "[showing lines 1-" in line.text and "next: read {" in line.text
    assert r.session.blob(line.handle.content_sha).decode() == BIG


def test_small_shell_output_returns_untouched(tmp_path: Path) -> None:
    r = rig(tmp_path, [calls(call("shell", "s1", command="echo hello"))], shell=True)
    r.run("say hello")
    line = r.session.state.result("s1")
    assert line is not None
    assert line.text.splitlines()[1:] == ["hello"], "no marker, no overhead under the cap"
    assert "[showing lines" not in line.text
    # Unwrapped, nothing held the repository read-only, and the header does not claim it.
    header = line.text.splitlines()[0]
    assert "sandbox none" in header and "repo not protected" in header
    assert "repo read-only" not in header


def test_source_enters_only_through_an_explicit_read(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text(SOURCE, encoding="utf-8")
    everything_else = [
        call("atlas", "t1"),
        call("grep", "t2", pattern="outer"),
        call("symbol", "t3", name="outer"),
        call("neighbors", "t4", target="a.py"),
        call("impact", "t5", symbols=["outer"]),
        call("tools", "t6", query="read"),
        call("recall", "t7", query="outer"),
        call("note_concern", "t8", text="outer returns a constant", anchor=["a.py"]),
        call("note_rejected", "t9", text="inline outer", reason="it is public"),
        call("ask_user", "t10", question="keep outer? default: yes"),
        call("read", "t11", path="a.py"),  # no span: the outline and a handle, not the body
    ]
    script = [calls(*everything_else), calls(call("read", "r1", path="a.py", lines="99-101"))]
    r = rig(tmp_path, script, workspace=ws, person=ScriptedPerson("yes"))
    r.run("what does outer return?")
    before = r.model.requests[1]
    assert all(MARKER not in text_of(m) for m in before.messages), "no tool leaked the body"
    after = r.model.requests[2]
    carrying = [m for m in after.messages if MARKER in text_of(m)]
    assert [(m.role, m.tool_call_id) for m in carrying] == [("tool", "r1")]


def test_a_record_is_one_line_naming_what_it_recorded(tmp_path: Path) -> None:
    concern = call("note_concern", "k", text="retry gives up after 2", anchor=["api/client.py"])
    rejected = call("note_rejected", "j", text="a global lock", reason="it serialises readers")
    blank = call("note_concern", "b", text="nowhere", anchor=["   "])
    r = rig(tmp_path, [calls(concern, rejected, blank)])
    r.run("note things")
    state = r.session.state
    for cid in ("k", "j"):
        text = state.result(cid).text  # type: ignore[union-attr]
        assert len(text.splitlines()) == 1 and text.endswith(" recorded")
    assert [c.text for c in state.concerns.live()] == ["retry gives up after 2"]
    assert state.result("b").text.startswith("INVALID_ARGUMENTS{field: anchor")  # type: ignore[union-attr]
    assert not state.ran("b")


def test_a_shell_call_with_a_why_or_an_intent_is_refused_naming_what_it_takes(
    tmp_path: Path,
) -> None:
    """D208, moved: the shell takes no purpose line (it took `why`), so one sent is
    refused as any argument the tool does not name is, never run."""
    r = rig(
        tmp_path,
        [
            calls(call("shell", "w1", command="echo hi", intent="list")),
            calls(call("shell", "s1", command="echo hi", why="say hi")),
        ],
        shell=True,
    )
    r.run("say hi")
    for cid, key in (("w1", "intent"), ("s1", "why")):
        got = r.session.state.result(cid)
        assert got is not None and not got.ran, cid
        assert got.text.startswith("INVALID_ARGUMENTS: shell") and f"- {key}:" in got.text, cid


def test_a_search_with_no_hits_ran(tmp_path: Path) -> None:
    """#571: an empty search is a result, not a refusal; it was recorded `ran: false`, and the
    REPL showed `(not run)`. A refused call still did not run. (`symbol`'s, since `find`'s
    search moved into it, D200.)"""

    class Nothing(ScriptedGraph):
        def symbol(self, name: str) -> str | ToolFailure:
            return UnresolvedSearch(name, searched=("src",), uncovered=())

    r = rig(
        tmp_path,
        [calls(call("symbol", "f1", name="SegmentTree"), call("nosuch", "n1"))],
        graph=Nothing(),
    )
    r.run("look for it")
    found, unknown = r.session.state.result("f1"), r.session.state.result("n1")
    assert found is not None and found.ran and found.text.startswith("UNRESOLVED")
    assert unknown is not None and not unknown.ran
