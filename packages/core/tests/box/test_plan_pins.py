"""Pins of `plan(text)` and `delegate(title, brief)` before T-3.166 part 3, and how they moved
(each listed in its Log): the arguments each takes, and the answers Athena reads where the
work has no folder, which did not move."""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import APP, episode_rig
from refusals import bare
from repos import init_repo

from chickadee.context.results import InvalidArgument
from chickadee.tools.builtin import BUILTINS

PLAN = "# Plan: go returns 2\n\n1. **Return 2.** Change `go()`.\n\nFinal check: the suite.\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _schema(name: str) -> tuple[list[str], list[str]]:
    tool = BUILTINS.get(name)
    assert tool is not None
    schema = tool.spec.schema
    return list(schema["properties"]), list(schema["required"])  # type: ignore[call-overload]


def test_plan_and_delegate_take_no_text_with_a_folder_and_the_text_without() -> None:
    """Moved: `text` was required, now only where the work has no
    folder; `delegate` gains `task`, and its title and brief are for a work with no folder."""
    assert _schema("plan") == (["text"], [])
    assert _schema("delegate") == (["task", "title", "brief"], [])


def test_the_answers_athena_reads_around_the_plan(root: Path) -> None:
    rig = episode_rig(root, [], [])
    refused = rig.desk.delegate("Do it", "brief", call_id="a-d0")
    assert bare(refused) == InvalidArgument("plan", "no plan is approved yet")
    assert refused.do == "call plan first, then delegate its tasks"  # type: ignore[union-attr]
    approved = rig.desk.plan(PLAN, call_id="a-plan")
    assert approved == "the person approved the plan. Delegate its first task."
