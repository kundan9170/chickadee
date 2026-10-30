"""Line anchors on the worker's tools: `replace_lines` and
`insert_lines`, whose anchor arguments' schema is the policy's style, and whose answers show
their lines in it, resolved on the draft as it now reads.

The packet's view, which showed every read of a target in the style, retired with the packet
loop; these drive the task worker's dispatcher over a draft given the style.
The box's own port (`box/edits.WorktreeEdits`) takes no style yet, so a draft over it answers
no line anchor; until it does, the draft here is the edits port's double."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Literal

import pytest

from chickadee.edits.draft import CandidateDraft
from chickadee.edits.formats.base import Targets
from chickadee.edits.lines import HASH, NUMBER, AnchorStyle, decode, tag
from chickadee.roles import TASK_WORKER
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import DraftCall
from chickadee.tools.policy import RolePolicy
from chickadee.tools.rules import CallOutcome
from chickadee.tools.spec import ANCHOR_ARG

from .tools_helpers import EditWorker, refused, said

APP = "def build_widget(name):\n    x = 1\n    return x\n"
LINES = ("def build_widget(name):", "    x = 1", "    return x")
LINE_TOOLS = ("replace_lines", "insert_lines")


Style = Literal["number", "hash"]


def _policy(style: Style = "number", *, lines: bool = True) -> RolePolicy:
    extra = (*LINE_TOOLS,) if lines else ()
    return replace(TASK_WORKER, enabled=("edit", *extra), line_anchors=style)


class _DraftEdits:
    """An `EditsPort` over one draft of `app.py`, held across calls, in `style`."""

    def __init__(self, style: AnchorStyle | None) -> None:
        self.draft = CandidateDraft(
            Targets(writable=("app.py",)),
            lambda path: decode(APP) if path == "app.py" else None,
            lambda _: None,
            style=style,
        )

    @contextmanager
    def edit(self) -> Generator[DraftCall]:
        with self.draft.call() as call:
            yield call

    def text(self) -> str:
        return self.draft.text("app.py") or ""


class _LineTools:
    """The task worker's dispatcher under `policy`, its draft in the policy's style when it
    offers a line tool, as `RolePolicy.anchors_lines` says."""

    def __init__(self, policy: RolePolicy) -> None:
        style = policy.anchor_style if policy.anchors_lines(BUILTINS) else None
        self.edits = _DraftEdits(style)
        self.worker = EditWorker(self.edits, policy.enabled, line_anchors=policy.line_anchors)

    def call(self, name: str, **arguments: object) -> CallOutcome:
        return self.worker.call(name, **arguments)


@pytest.fixture
def number() -> _LineTools:
    return _LineTools(_policy("number"))


@pytest.fixture
def hashed() -> _LineTools:
    return _LineTools(_policy("hash"))


def _a(n: int) -> str:
    return HASH.anchor(n, LINES[n - 1])


# -- which style, and when --------------------------------------------------------------------


def test_line_anchors_are_on_only_when_a_line_tool_is_offered() -> None:
    assert not TASK_WORKER.anchors_lines(BUILTINS)
    assert not _policy("hash", lines=False).anchors_lines(BUILTINS)
    assert _policy("number").anchors_lines(BUILTINS)
    assert _policy("number").anchor_style is NUMBER
    assert _policy("hash").anchor_style is HASH


def test_the_anchor_arguments_take_the_styles_schema_and_keep_their_words() -> None:
    styles: tuple[tuple[Style, str], ...] = (("number", "integer"), ("hash", "string"))
    for style, kind in styles:
        specs = {s.name: s for s in _policy(style).surface(BUILTINS)}
        start = specs["replace_lines"].schema["properties"]["start"]  # type: ignore[index]
        assert start["type"] == kind and ANCHOR_ARG not in start  # type: ignore[index]
        assert "first line" in start["description"]  # type: ignore[index]
    tool = BUILTINS.get("edit")
    assert tool is not None and _policy("hash").resolved(tool.spec) is tool.spec


# -- a mismatched anchor fails validation, never reinterpreted ------------------------------


def test_a_bare_number_under_hash_and_a_tag_under_number_are_invalid_arguments(
    number: _LineTools, hashed: _LineTools
) -> None:
    for run, start in ((hashed, 2), (number, _a(2))):
        answer = run.call(
            "replace_lines", path="app.py", start=start, end=start, content="y = 2"
        )
        refused(answer, "INVALID_ARGUMENTS")
        assert run.edits.text() == APP


# -- replace and insert ------------------------------------------------------------------------


def test_replace_lines_by_number_answers_the_lines_in_the_style(number: _LineTools) -> None:
    answer = number.call("replace_lines", path="app.py", start=2, end=2, content="    x = 2")
    assert said(answer) == "ok: app.py\n2│    x = 2"


def test_replace_lines_by_hash_answers_tagged_rows(hashed: _LineTools) -> None:
    answer = hashed.call(
        "replace_lines", path="app.py", start=_a(2), end=_a(3), content="    return 2"
    )
    assert said(answer) == f"ok: app.py\n2:{tag('    return 2')}│    return 2"


def test_empty_content_deletes_the_lines(number: _LineTools) -> None:
    said(number.call("replace_lines", path="app.py", start=2, end=2, content=""))
    assert number.edits.text() == "def build_widget(name):\n    return x\n"


def test_a_range_ending_before_it_starts_is_refused(number: _LineTools) -> None:
    reason = refused(
        number.call("replace_lines", path="app.py", start=3, end=2, content="x"),
        "BAD_FORMAT",
    )
    assert "before its start" in reason


def test_insert_after_0_is_the_files_start_and_after_the_last_line_its_end(
    number: _LineTools,
) -> None:
    said(number.call("insert_lines", path="app.py", after=0, content="import os"))
    said(number.call("insert_lines", path="app.py", after=4, content="# end"))
    rows = number.edits.text().splitlines()
    assert rows[0] == "import os" and rows[-1] == "# end" and len(rows) == 5


def test_a_moved_line_is_stale_with_the_lines_there_now(hashed: _LineTools) -> None:
    old = _a(2)
    said(hashed.call("insert_lines", path="app.py", after="0", content="import os"))
    reason = refused(
        hashed.call("replace_lines", path="app.py", start=old, end=old, content="y"),
        "STALE_ANCHOR",
    )
    assert f"no longer reads as {old}" in reason
    assert f"3:{tag(LINES[1])}│    x = 1" in reason  # where the line went


def test_a_line_past_the_end_is_stale(number: _LineTools) -> None:
    reason = refused(
        number.call("replace_lines", path="app.py", start=9, end=9, content="y"),
        "STALE_ANCHOR",
    )
    assert "the file has 3 lines" in reason


def test_with_no_line_tool_an_edit_answers_plain() -> None:
    run = _LineTools(_policy("hash", lines=False))
    assert said(run.call("edit", path="app.py", old="x = 1", new="x = 2")) == (
        "ok: app.py\n    x = 2"
    )


def test_a_line_is_anchored_where_it_now_is_after_an_edit(hashed: _LineTools) -> None:
    said(hashed.call("insert_lines", path="app.py", after="0", content="import os"))
    anchor = HASH.anchor(2, LINES[0])
    said(
        hashed.call(
            "replace_lines", path="app.py", start=anchor, end=anchor, content="def f(name):"
        )
    )
    rows = hashed.edits.text().splitlines()
    assert rows[0] == "import os" and rows[1] == "def f(name):"
