"""`tests_for`: the affected suite's selection, moved from the verifier with its pins.
The test files within the depth's reverse edges of a change, and the changed test
files themselves, in path order."""

from __future__ import annotations

from chickadee.graph.tests_for import TESTS_FOR_DEPTH
from chickadee.graph.tests_for import tests_for as select
from chickadee.profiles.loader import load_profile

# -- the affected suite: the repository's tests that reach the change,
# through the graph's file edges, selected by no model.

PY = load_profile("python")

# src -> dst: `src` imports `dst`.
EDGES = (
    ("tests/test_app.py", "app.py", "import", 1),
    ("tests/test_cli.py", "cli.py", "import", 1),
    ("cli.py", "app.py", "import", 1),
    ("tests/test_far.py", "d.py", "import", 1),
    ("d.py", "c.py", "import", 1),
    ("c.py", "b.py", "import", 1),
    ("b.py", "app.py", "import", 1),
    ("tests/test_other.py", "other.py", "import", 1),
    ("app.py", None, "import", 1),
    ("app.py", "app.py", "import", 1),
)


def test_a_test_that_reaches_the_change_is_selected_with_what_it_reaches() -> None:
    got = select(("app.py",), EDGES, PY)
    assert got.tests == ("tests/test_app.py", "tests/test_cli.py")
    assert dict(got.reached) == {"tests/test_app.py": "app.py", "tests/test_cli.py": "app.py"}
    assert got.depth == TESTS_FOR_DEPTH


def test_a_test_beyond_the_depth_is_not() -> None:
    assert "tests/test_far.py" not in select(("app.py",), EDGES, PY).tests
    deeper = select(("app.py",), EDGES, PY, depth=4)
    assert "tests/test_far.py" in deeper.tests


def test_a_changed_test_file_is_its_own_selection() -> None:
    got = select(("tests/test_other.py",), EDGES, PY)
    assert got.tests == ("tests/test_other.py",)
    assert got.reached["tests/test_other.py"] == ""


def test_the_run_s_own_tests_are_excluded() -> None:
    got = select(("app.py",), EDGES, PY, exclude=("tests/test_cli.py",))
    assert got.tests == ("tests/test_app.py",)


def test_nothing_reaches_an_unconnected_change() -> None:
    assert select(("lonely.py",), EDGES, PY).tests == ()
