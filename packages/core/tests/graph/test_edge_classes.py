"""D19 edge classes: resolved / fallback / unknown, and the property that matters."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.graph.edges import FALLBACK, RESOLVED, UNKNOWN, classify
from chickadee.graph.indexer import Indexer
from chickadee.graph.parse import Reference


class FakeLSP:
    """Stands in for the checker daemon `T-2.6` brings up. The `resolved` path has to be
    testable before that exists, or it ships untested."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    def definition(self, name: str, file: str, line: int) -> str | None:
        return self.answers.get(name)


REF = (Reference(name="helper", line=3, from_symbol="a.caller", kind="calls"),)


def test_lsp_present_gives_resolved() -> None:
    edges = classify(REF, "a.py", {}, FakeLSP({"helper": "helpers.helper"}))
    assert edges[0].edge_class == RESOLVED
    assert edges[0].dst == "helpers.helper"


def test_lsp_absent_gives_fallback() -> None:
    edges = classify(REF, "a.py", {"helper": "helpers.helper"}, None)
    assert edges[0].edge_class == FALLBACK
    assert edges[0].dst == "helpers.helper"


def test_unresolvable_gives_unknown() -> None:
    edges = classify(REF, "a.py", {}, None)
    assert edges[0].edge_class == UNKNOWN


def test_an_ambiguous_name_is_unknown_not_a_coin_flip() -> None:
    """The property: never a wrong edge in place of `unknown`.

    Two definitions of one name means the graph does not know which is meant. Picking one
    would be right half the time and confidently wrong the other half.
    """
    root_names: dict[str, str] = {}  # ambiguous names are absent from the index by design
    edges = classify(REF, "a.py", root_names, None)
    assert edges[0].edge_class == UNKNOWN


def test_ambiguous_definitions_are_excluded_from_the_fallback_index(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "one.py").write_text("def shared():\n    return 1\n", encoding="utf-8")
    (root / "two.py").write_text("def shared():\n    return 2\n", encoding="utf-8")
    (root / "use.py").write_text("def go():\n    return shared()\n", encoding="utf-8")

    ix = Indexer(root)
    ix.full_index()
    row = ix.conn.execute(
        "SELECT edge_class, dst FROM edges WHERE file = 'use.py' AND type = 'calls'"
    ).fetchone()
    assert row["edge_class"] == UNKNOWN, "an ambiguous name must not resolve to one of them"


def test_a_python_definition_is_not_a_candidate_for_a_javascript_call(tmp_path: Path) -> None:
    """A cross-language match is a wrong edge however it is labelled."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def helper(x):\n    return x\n", encoding="utf-8")
    (root / "b.js").write_text("function go(){ return helper(); }\n", encoding="utf-8")

    ix = Indexer(root)
    ix.full_index()
    row = ix.conn.execute("SELECT edge_class, dst FROM edges WHERE file = 'b.js'").fetchone()
    assert row["edge_class"] == UNKNOWN
    assert row["dst"] == "helper", "must not point at the Python definition"


@pytest.mark.parametrize("kind", ["calls", "inherits"])
def test_every_edge_carries_a_class(indexed, kind: str) -> None:
    rows = indexed.conn.execute(
        "SELECT edge_class FROM edges WHERE type = ?", (kind,)
    ).fetchall()
    assert rows, f"fixture should produce {kind} edges"
    assert all(r["edge_class"] in {RESOLVED, FALLBACK, UNKNOWN} for r in rows)
