"""One id module that formats and parses.

Every id that carries meaning parses back to the parts it was built from, and a string that is
not one parses to `None`, never to a guess. The packet loop's operation and candidate ids, and
their tests, retired with it.
"""

from __future__ import annotations

import re

import pytest

from chickadee import ids
from chickadee.errors import InfraError
from chickadee.schema.ids import RUN_ID_RE


def test_the_bootstrap_operation_is_run_scoped() -> None:
    assert ids.bootstrap_operation("2026-10-04-0a1b2c3d") == "bootstrap:2026-10-04-0a1b2c3d"


def test_a_task_commit_parses_back_to_its_task() -> None:
    """Resume finds an accepted task's commit by this id in its trailer."""
    assert ids.task_commit_id("T2", 3) == "task-commit:T2:3"
    assert ids.task_of_commit(ids.task_commit_id("T2")) == "T2"
    for other in ("task-commit:T2", "bootstrap:2026-10-04-0a1b2c3d", "step:T2:1:c1:ab12"):
        assert ids.task_of_commit(other) is None


def test_a_step_names_its_task_round_and_call_and_is_fresh_each_time() -> None:
    first, second = ids.step_id("T1", 2, "c:1"), ids.step_id("T1", 2, "c:1")
    assert first != second
    assert ids.step_prefix("T1", 2, "c:1") == "step:T1:2:c%3A1:"
    assert first.startswith("step:T1:2:c%3A1:") and second.startswith("step:T1:2:c%3A1:")
    assert ids.step_prefix(None, None, "c1") == "step:athena:0:c1:"


def test_the_minted_ids_keep_their_shapes() -> None:
    assert re.fullmatch(r"S-[0-9a-f]{8}", ids.new_session_id())
    assert re.fullmatch(r"scout-escape:[0-9a-f]{16}", ids.escape_id())
    assert RUN_ID_RE.match(ids.new_run_id())


# -- operation_component: an operation id as one name ----------------------


@pytest.mark.parametrize(
    ("op", "name"),
    [
        ("apply:p1:1:1", "apply%3Ap1%3A1%3A1"),
        ("bootstrap:2026-10-04-0a1b2c3d", "bootstrap%3A2026-10-04-0a1b2c3d"),
        ("verify:p1:A3:status", "verify%3Ap1%3AA3%3Astatus"),
        ("op_1", "op_1"),
        ("a.b", "a%2Eb"),
        ("a/b", "a%2Fb"),
        ("100%", "100%25"),
        ("é", "%C3%A9"),
    ],
)
def test_an_operation_component_percent_encodes_all_but_a_narrow_alphabet(
    op: str, name: str
) -> None:
    """The names the refs had before the move are unchanged: `apply%3Ap1%3A1%3A1`."""
    assert ids.operation_component(op) == name


def test_an_operation_component_is_legal_as_a_ref_and_a_file_name_everywhere() -> None:
    for op in ("apply:p1:1:1", "x<y>z", 'q"|?*\\', "..", ".lock", "con", " a. "):
        assert re.fullmatch(r"[A-Za-z0-9_%-]+", ids.operation_component(op))


def test_two_operations_never_share_a_component() -> None:
    """Injective: `a:b` and `a-b` would be one name under a replace-with-dash scheme."""
    ops = ["a:b", "a-b", "a%3Ab", "a_b", "a.b", "a%2Eb"]
    assert len({ids.operation_component(op) for op in ops}) == len(ops)


def test_an_empty_operation_id_has_no_component() -> None:
    with pytest.raises(InfraError):
        ids.operation_component("")
