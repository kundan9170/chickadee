"""What a result was made from: `grep` and `glob` hand the rows their text
shows, typed, beside it; a result rule that changes the text drops them, so the data never
holds more than the model was shown; a native call's text is unchanged."""

from __future__ import annotations

from pathlib import Path

from chickadee.context.result_data import Match, Matches, Paths, Structured
from chickadee.scout.search import Search
from chickadee.tools.builtin.grep import SPEC as GREP
from chickadee.tools.catalogue import Redact
from chickadee.tools.dispatch import _data_as_shown  # pyright: ignore[reportPrivateUsage]
from chickadee.tools.rules import CallOutcome, Pending
from chickadee.tools.tool import Invocation, TextAnswer, as_text


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\nreturn_value = 2\n")
    (tmp_path / "b.py").write_text("return_value = 3\n")
    (tmp_path / ".env").write_text("return_value=secret\n")
    return tmp_path


def test_grep_hands_exactly_the_rows_its_text_shows(tmp_path: Path) -> None:
    found = Search(root=_repo(tmp_path), rg=None)("return_value")
    assert isinstance(found, Structured) and isinstance(found.data, Matches)
    rows = found.data.rows
    assert {(m.path, m.line) for m in rows} == {("a.py", 2), ("b.py", 1)}
    for m in rows:
        assert f"{m.path}:{m.line}: {m.text}" in found
    assert all(m.path != ".env" for m in rows), "a denied file is in neither"


def test_a_hit_past_the_bound_is_in_neither(tmp_path: Path) -> None:
    found = Search(root=_repo(tmp_path), rg=None, max_hits=1)("return_value")
    assert isinstance(found, Structured) and isinstance(found.data, Matches)
    [only] = found.data.rows
    assert "[truncated: 1 of 2 hits not shown" in found
    assert f"{only.path}:{only.line}:" in found


def test_glob_hands_its_paths(tmp_path: Path) -> None:
    found = Search(root=_repo(tmp_path))("*.py", kind="path")
    assert isinstance(found, Structured) and isinstance(found.data, Paths)
    assert sorted(found.data.paths) == ["a.py", "b.py"]


def test_no_hits_hand_no_rows(tmp_path: Path) -> None:
    found = Search(root=_repo(tmp_path), rg=None)("nowhere")
    assert isinstance(found, Structured) and found.data == Matches(())


def test_the_answer_carries_the_data_and_the_same_text(tmp_path: Path) -> None:
    found = Search(root=_repo(tmp_path), rg=None)("return_value")
    answer = as_text(GREP, found)
    assert isinstance(answer, TextAnswer)
    assert type(answer.text) is str and answer.text == found, "a native call's text unchanged"
    assert isinstance(answer.data, Matches)


def _outcome(text: str) -> CallOutcome:
    pending = Pending(
        call=Invocation("c1", "op-c1", 0), sent="grep", raw={}, tool="grep", spec=GREP
    )
    data = Matches((Match("a.py", 1, "token=abc"),))
    return CallOutcome(pending, TextAnswer(text, "lines", data))


def test_a_rule_that_changes_the_text_drops_the_data() -> None:
    made = _outcome("a.py:1: key = AKIAIOSFODNN7EXAMPLE")
    redacted = Redact().apply(made, None, None)  # type: ignore[arg-type]
    assert redacted.answer != made.answer, "the redaction changed the text"
    kept = _data_as_shown(made.answer, redacted)
    assert isinstance(kept.answer, TextAnswer) and kept.answer.data is None


def test_a_rule_that_leaves_the_text_keeps_the_data() -> None:
    made = _outcome("a.py:1: x = 1")
    kept = _data_as_shown(made.answer, Redact().apply(made, None, None))  # type: ignore[arg-type]
    assert isinstance(kept.answer, TextAnswer) and kept.answer.data is not None


def test_a_cut_drops_the_data() -> None:
    """The worker's cut (`Clip`) is a result rule like any other: one place drops the data."""
    made = _outcome("a.py:1: x = 1\n" * 500)
    assert isinstance(made.answer, TextAnswer)
    cut = CallOutcome(
        made.pending, TextAnswer("a.py:1: x = 1\n[cut]", "lines", made.answer.data)
    )
    kept = _data_as_shown(made.answer, cut)
    assert isinstance(kept.answer, TextAnswer) and kept.answer.data is None


def test_a_changed_shell_text_keeps_its_returncode() -> None:
    """The exit code is in the header the model still sees; only the output goes."""
    from chickadee.context.handle import Handle
    from chickadee.context.result_data import Completed
    from chickadee.context.results import ToolResult

    handle = Handle.__new__(Handle)
    made = ToolResult(handle=handle, content="out", data=Completed(1, "out", ""))
    pending = Pending(
        call=Invocation("c1", "op-c1", 0), sent="shell", raw={}, tool="shell", spec=GREP
    )
    changed = CallOutcome(pending, ToolResult(handle=handle, content="o[cut]", data=made.data))
    kept = _data_as_shown(made, changed)
    assert isinstance(kept.answer, ToolResult)
    assert kept.answer.data == Completed(1, None, None)
