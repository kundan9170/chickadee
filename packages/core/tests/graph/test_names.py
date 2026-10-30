"""`graph/names.py`: the one rule for a written name naming a qualified one."""

from __future__ import annotations

import pytest

from chickadee.graph.names import names_qualified


@pytest.mark.parametrize(
    ("qualified", "name", "named"),
    [
        ("app.Store.get", "app.Store.get", True),
        ("app.Store.get", "Store.get", True),
        ("app.Store.get", "get", True),
        ("app.Store.get", "ore.get", False),
        ("app.forget", "get", False),
        ("app.Store.get", "Store", False),
        ("app.Store.get", "", False),
    ],
)
def test_a_name_is_the_whole_qualified_name_or_a_suffix_at_a_dot(
    qualified: str, name: str, named: bool
) -> None:
    assert names_qualified(qualified, name) is named
