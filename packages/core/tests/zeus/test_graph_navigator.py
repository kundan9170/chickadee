"""`GraphNavigator`: each navigate answer is the graph's, rendered and never a file
body; a symbol the graph does not know is a typed failure, distinct from one with no callers;
no index is `INDEX_UNAVAILABLE`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from chickadee.context.results import IndexUnavailable, UnresolvedSearch
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.graph.store import db_path
from chickadee.profiles.loader import ProfileSet, load_profile
from chickadee.zeus.tools.graphnav import GraphNavigator, indexed_profiles, opened

CORE = (
    "def helper(x: int) -> int:\n"
    '    """Add one."""\n'
    "    secret_body_marker = x + 1\n"
    "    return secret_body_marker\n"
    "\n\n"
    "def lonely() -> None:\n"
    "    pass\n"
)
API = "from app.core import helper\n\n\ndef serve() -> int:\n    return helper(1)\n"
TEST = (
    "from app.core import helper\n\n\ndef test_helper() -> None:\n    assert helper(1) == 2\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for path, text in (
        ("app/core.py", CORE),
        ("app/api.py", API),
        ("tests/test_core.py", TEST),
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    return tmp_path


@pytest.fixture
def graph(repo: Path) -> Iterator[Graph]:
    Indexer(repo).full_index()
    graph = opened(repo)
    assert graph is not None
    yield graph
    graph.close()


@pytest.fixture
def nav(graph: Graph) -> GraphNavigator:
    return GraphNavigator(graph, [load_profile("python")])


def _text(answer: object) -> str:
    assert isinstance(answer, str), answer
    return answer


def _answers(nav: GraphNavigator) -> list[str]:
    return [
        _text(nav.atlas()),
        _text(nav.symbol("core.helper")),
        _text(nav.symbol("helper")),
        _text(nav.neighbors("helper", 2)),
        _text(nav.neighbors("app/core.py", 1)),
        _text(nav.impact(["helper"])),
    ]


def test_atlas_names_the_components_their_contracts_and_the_edges_between_them(
    nav: GraphNavigator,
) -> None:
    atlas = _text(nav.atlas())
    assert "app    2 files · public: " in atlas and "helper(x: int) -> int" in atlas
    assert "tests -> app" in atlas
    assert "parse failed: none" in atlas


def test_symbol_with_no_exact_name_lists_the_closest_by_name(nav: GraphNavigator) -> None:
    """What `find` answered, now `symbol`'s when no symbol is named exactly."""
    near = _text(nav.symbol("core.helper"))
    assert near.startswith("no symbol named 'core.helper'; by name,")
    assert "app/core.py:1    function app.core.helper" in near


def test_symbol_names_its_definition_callers_and_the_tests_that_touch_it(
    nav: GraphNavigator,
) -> None:
    text = _text(nav.symbol("helper"))
    assert "## SYMBOL app.core.helper    function · app/core.py:1-4" in text
    assert "doc: Add one." in text
    assert "callers: 2" in text and "app.api.serve    app/api.py:5" in text
    tests = text.split("  tests: ", 1)[1]
    assert tests.startswith("1") and "tests.test_core.test_helper" in tests
    assert "app.api.serve" not in tests


def test_neighbors_of_a_symbol_and_of_a_path(nav: GraphNavigator) -> None:
    near = _text(nav.neighbors("helper", 1))
    assert "shares its file with: lonely" in near
    assert "app.api.serve    calls it" in near
    of_file = _text(nav.neighbors("app/core.py", 1))
    assert "lonely() -> None" in of_file and "app.api.serve" in of_file


def test_impact_lists_the_dependents_with_their_confidence(nav: GraphNavigator) -> None:
    text = _text(nav.impact(["helper"]))
    assert "app/api.py\n    app.api.serve    " in text
    assert "tests/test_core.py\n    tests.test_core.test_helper" in text


def test_no_answer_holds_a_file_body(nav: GraphNavigator) -> None:
    for answer in _answers(nav):
        assert "secret_body_marker" not in answer
        assert "return helper(1)" not in answer


def test_an_unknown_symbol_is_a_typed_failure_and_no_callers_is_a_fact(
    nav: GraphNavigator,
) -> None:
    unknown = nav.symbol("nonesuch")
    assert isinstance(unknown, UnresolvedSearch) and unknown.query == "nonesuch"
    lonely = _text(nav.symbol("lonely"))
    assert "callers: none" in lonely and "UNRESOLVED" not in lonely
    # 33 §3 lists no UNRESOLVED for these two: the answer names the unknown instead.
    assert "not in the graph: 'nonesuch'" in _text(nav.neighbors("nonesuch", 1))
    assert "not in the graph: 'nonesuch'" in _text(nav.impact(["helper", "nonesuch"]))


def test_no_index_is_index_unavailable(repo: Path) -> None:
    assert opened(repo) is None
    assert not db_path(repo).exists(), "asking opened no database: `connect` would have"
    nav = GraphNavigator(None)
    for answer in (
        nav.atlas(),
        nav.symbol("x"),
        nav.neighbors("x", 1),
        nav.impact(["x"]),
    ):
        assert isinstance(answer, IndexUnavailable)


def test_the_profiles_are_those_the_index_holds(graph: Graph) -> None:
    profiles = indexed_profiles(graph, ProfileSet.load())
    assert [p.name for p in profiles] == ["python"]
    assert indexed_profiles(None, ProfileSet.load()) == ()
