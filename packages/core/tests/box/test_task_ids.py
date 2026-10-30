"""A task is named by its plan, and each attempt keyed apart."""

from __future__ import annotations

import pytest

from chickadee.ids import parent_of, split_task_key, task_depth, task_key
from chickadee.schema.ids import TASK_ID_RE, TASK_KEY_RE


@pytest.mark.parametrize(
    ("task", "attempt", "key"), [("T3", 1, "T3"), ("T3", 2, "T3@2"), ("T3.1", 12, "T3.1@12")]
)
def test_a_key_round_trips(task: str, attempt: int, key: str) -> None:
    assert task_key(task, attempt) == key
    assert split_task_key(key) == (task, attempt)


def test_an_attempt_is_counted_from_one() -> None:
    with pytest.raises(ValueError, match="counted from 1"):
        task_key("T3", 0)


def test_every_id_written_by_an_older_version_is_a_key_and_its_own_first_attempt() -> None:
    for n in (1, 9, 10, 123456):
        old = f"T{n}"
        assert TASK_ID_RE.match(old) and TASK_KEY_RE.match(old)
        assert split_task_key(old) == (old, 1)


@pytest.mark.parametrize(
    "bad", ["T0", "T3.0", "T3@1", "T3@", "t3", "T3.1@0", "T1234567", "T3..1"]
)
def test_what_is_neither_an_id_nor_a_key(bad: str) -> None:
    assert not TASK_KEY_RE.match(bad)


def test_a_key_is_never_an_id() -> None:
    assert TASK_KEY_RE.match("T3@2") and not TASK_ID_RE.match("T3@2")


def test_a_piece_names_its_parent_and_its_depth() -> None:
    assert parent_of("T3") is None
    assert parent_of("T3.1") == "T3"
    assert parent_of("T3.1.2") == "T3.1"
    assert [task_depth(t) for t in ("T3", "T3.1", "T3.1.2")] == [0, 1, 2]
