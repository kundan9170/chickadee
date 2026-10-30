"""#544, from the owner's first live session: a malformed structured argument is answered
with a minimal valid example of its shape (`ShowExample`, from `ToolSpec.examples`). And D200:
a call to `find`, retired, is told what replaced it."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chickadee.context.results import InvalidArgument
from chickadee.tools.catalogue import ShowExample
from chickadee.tools.rules import CallOutcome, Pending
from chickadee.tools.spec import SURFACE

from .zeus_helpers import answer, call, calls, rig

CODER = next(s for s in SURFACE if s.name == "coder")


def test_a_missing_brief_is_answered_with_the_briefs_shape(tmp_path: Path) -> None:
    """A one-task plan's title sent without its brief: the answer shows the brief to send."""
    r = rig(tmp_path, [calls(call("coder", "c1", title="Add a limiter")), answer()])
    r.run("edit")
    text = r.session.state.result("c1").text  # type: ignore[union-attr]
    assert text.startswith("INVALID_ARGUMENTS{field: brief")
    assert f"example: {CODER.examples['brief']}" in text


def test_no_work_is_answered_with_the_designs_shape(tmp_path: Path) -> None:
    r = rig(tmp_path, [calls(call("coder", "c1")), answer()])
    r.run("edit")
    text = r.session.state.result("c1").text  # type: ignore[union-attr]
    assert "a delegation is a design (work) or one task (title with brief)" in text
    assert "Do: send work=" in text
    assert f"example: {CODER.examples['work']}" in text


NESTED = replace(
    CODER,
    examples={
        "request": '{"task": "<what to change, in words>"}',
        "request.hunks": '[{"path": "calc.py", "old": "<exact text now in the file>"}]',
    },
)
"""A spec whose examples nest, as `coder`'s did in older versions flattened its arguments: the
rule
reads dotted paths, whatever tool declares them."""


@pytest.mark.parametrize(
    ("field", "shown"),
    [
        ("request.hunks.0.path", "request.hunks"),
        ("request.hunks", "request.hunks"),
        ("request", "request"),
        ("request.workspace", "request"),
        ("intent", None),
    ],
)
def test_the_longest_declared_path_the_field_lies_under_wins(
    field: str, shown: str | None
) -> None:
    pending = Pending.__new__(Pending)
    object.__setattr__(pending, "spec", NESTED)
    outcome = CallOutcome(pending, InvalidArgument(field, "why"))
    got = ShowExample().apply(outcome, None, None).answer  # type: ignore[arg-type]
    assert isinstance(got, InvalidArgument)
    assert got.example == (NESTED.examples[shown] if shown else None)


def test_an_example_is_shown_once() -> None:
    pending = Pending.__new__(Pending)
    object.__setattr__(pending, "spec", CODER)
    given = InvalidArgument("request", "why", example="already")
    outcome = CallOutcome(pending, given)
    assert ShowExample().apply(outcome, None, None).answer == given  # type: ignore[arg-type]
    assert replace(given, example=None).render().count("example") == 0


def test_a_retired_tool_is_refused_naming_what_replaced_it(tmp_path: Path) -> None:
    """D200 (the owner's REPL run): `find` searched symbol names, and every model read it as
    Unix `find`. It is retired; a call to it names `symbol` and `grep`."""
    r = rig(tmp_path, [calls(call("find", "f1", query="*.py")), answer()])
    r.run("files?")
    text = r.session.state.result("f1").text  # type: ignore[union-attr]
    assert text.startswith("UNKNOWN_TOOL{name: 'find', replaced_by: [symbol, grep]")
