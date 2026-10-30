"""The review's answers as the model is shown them.

`abandon` is `drop` now: the tool offers `accept`, `feedback`, `take_over`, `re_cut` and `drop`,
and a call that still sends `abandon` is told the new name, as a renamed argument is.
`fresh` is `true` or a list of paths. No prompt and no description of the review tool names
`abandon` as an answer any more; the run-level `abandon` (a whole run given up) is another
concept, and stays.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from chickadee.adapters.base import SHARED_TEMPLATES as SHARED
from chickadee.adapters.base import load_template
from chickadee.adapters.frontier import TEMPLATES as FRONTIER
from chickadee.hooks.builtins import review_owed
from chickadee.tools.builtin.review import ANSWERS, SPEC
from chickadee.tools.validate import Arguments, InvalidArguments, validate


def test_the_answers_are_the_six_and_abandon_is_not_one() -> None:
    """D240 §2.8 added `continue`, after a round held for a message."""
    assert ANSWERS == ("accept", "feedback", "continue", "take_over", "re_cut", "drop")


def test_abandon_is_answered_as_renamed() -> None:
    got = validate(SPEC, {"task": "T1", "answer": "abandon"})
    assert isinstance(got, InvalidArguments)
    [violation] = got.violations
    assert violation.field == "answer"
    assert "abandon is renamed: use drop" in violation.problem


@pytest.mark.parametrize("fresh", [True, False, ["app.py"], ["a.py", "b.py"]])
def test_fresh_is_true_or_a_list_of_paths(fresh: object) -> None:
    got = validate(SPEC, {"task": "T1", "answer": "feedback", "text": "x", "fresh": fresh})
    assert isinstance(got, Arguments), got


@pytest.mark.parametrize("fresh", ["app.py", [], [1]])
def test_fresh_is_refused_in_any_other_shape(fresh: object) -> None:
    got = validate(SPEC, {"task": "T1", "answer": "feedback", "text": "x", "fresh": fresh})
    assert isinstance(got, InvalidArguments)


_AS_AN_ANSWER = re.compile(r"`abandon`|\babandon it\b|\baccept or abandon\b|\babandoned\b")


@pytest.mark.parametrize(
    ("templates", "name"),
    [
        (FRONTIER, "athena/review.md"),
        (FRONTIER, "athena/outgrown.md"),
        (SHARED, "athena.md"),
    ],
)
def test_no_prompt_names_abandon_as_an_answer(templates: Path, name: str) -> None:
    text = load_template(templates, name)
    assert not _AS_AN_ANSWER.search(text), name


def test_the_review_tools_own_words_never_name_abandon_as_an_answer() -> None:
    answer = SPEC.schema["properties"]["answer"]  # type: ignore[index]
    assert "abandon" not in str(answer)  # type: ignore[arg-type]
    fresh = SPEC.schema["properties"]["fresh"]  # type: ignore[index]
    assert "true puts every file back" in str(fresh)  # type: ignore[arg-type]


@pytest.mark.parametrize("taken_over", [False, True])
def test_her_reminder_names_only_answers_the_tool_offers(taken_over: bool) -> None:
    """The review of T-3.189: the reminder still said `abandon`, which the tool refuses."""
    said = review_owed("T1", "x", taken_over=taken_over)
    assert "abandon" not in said
    assert (
        ("accept or drop" in said)
        if taken_over
        else ("accept, feedback, take_over, re_cut or drop" in said)
    )
