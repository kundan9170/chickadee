"""GR-3: camelCase and snake_case both retrievable by any subword."""

from __future__ import annotations

import pytest

from chickadee.graph.fts import search, subwords


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("getUserName", ("get", "user", "name")),
        ("refresh_session", ("refresh", "session")),
        ("HTTPSConnection", ("https", "connection")),
        ("parseHTML5", ("parse", "html", "5")),
        ("already", ("already",)),
    ],
)
def test_subword_expansion(name: str, expected: tuple[str, ...]) -> None:
    assert subwords(name) == expected


@pytest.mark.parametrize("query", ["user", "name", "getUserName", "get"])
def test_a_camel_case_symbol_is_found_by_any_subword(indexed, query: str) -> None:
    assert any(s.name == "getUserName" for s in indexed.search(query))


@pytest.mark.parametrize("query", ["refresh", "session", "refresh_session"])
def test_a_snake_case_symbol_is_found_by_any_subword(indexed, query: str) -> None:
    assert any(s.name == "refresh_session" for s in indexed.search(query))


def test_both_conventions_answer_the_same_query(indexed) -> None:
    """The point of GR-3: you should not have to know which convention the author used."""
    found = {s.name for s in indexed.search("session")}
    assert "refresh_session" in found


def test_search_returns_nothing_for_an_absent_term(indexed) -> None:
    assert indexed.search("zzzznotasymbol") == ()


def test_fts_search_returns_sorted_bm25_scores(indexed) -> None:
    hits = search(indexed.conn, "session")
    assert hits
    scores = [score for _, score in hits]
    assert scores == sorted(scores)
    assert all(isinstance(sid, int) and isinstance(score, float) for sid, score in hits)
