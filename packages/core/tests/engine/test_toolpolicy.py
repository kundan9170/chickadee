"""`[tools.<role>]` chooses tools and holds no numbers.

It used to hold `max_tool_calls`, `max_result_bytes` and `dedup`, filled from the cast and
validated, and nothing read any of them: `[tools.worker] max_tool_calls = 5` loaded and
changed nothing. A key that loads without effect is what `31-state.md` rule 4 forbids, so each
is now refused at load, naming where its number lives.

D48's promise stands, held where the number is: every tool-using role has a `max_tool_calls`
row in the limits register, which a config may change and cannot remove. That a
`[limits]` override resolves is `test_limits_resolve.py`'s.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from chickadee.config import EngineConfig
from chickadee.errors import ConfigError
from chickadee.roles import ROLES
from chickadee.state.layers import Layer
from chickadee.state.merge import UnknownKey
from chickadee.state.resolve import check_layer

TOOL_USING_ROLES = frozenset(r.name for r in ROLES if r.tools != "none")


def test_every_tool_using_role_has_a_cap_from_the_register() -> None:
    """D48 by D122: the numbers are recalibratable, that each tool-using role has one is not.
    A role with no surface has none."""
    caps = {r.name: r.max_tool_calls for r in ROLES}
    assert caps == {
        "zeus": 64,
        "athena": 100,
        "worker": 200,
        "locate": 40,
        "critic": 50,
        "reader": 30,
        "reviewer": 40,
        "researcher": 30,
        "residue": 0,
        "checkpoint": 0,
        "title": 0,
    }
    assert all(caps[name] > 0 for name in TOOL_USING_ROLES)


RETIRED: list[tuple[str, object, str]] = [
    ("max_tool_calls", 3, r'`\[limits\] "worker\.max_tool_calls"`'),
    ("max_result_bytes", 4096, r"bounded by the tool that produces it"),
    ("dedup", False, r"handled by the role's rules"),
]


@pytest.mark.parametrize(("key", "value", "where"), RETIRED, ids=[k for k, _, _ in RETIRED])
def test_a_number_in_a_tools_table_is_refused_naming_where_it_lives(
    raw: dict[str, Any], key: str, value: object, where: str
) -> None:
    raw["tools"] = {"worker": {key: value}}
    with pytest.raises(ConfigError, match=rf"`tools\.worker\.{key}` is no longer read") as err:
        EngineConfig.with_defaults(raw)
    assert err.match(where)
    assert err.match(r"\[tools\.worker\]` chooses tools only")


@pytest.mark.parametrize(("key", "value", "where"), RETIRED, ids=[k for k, _, _ in RETIRED])
def test_a_file_hears_the_same_refusal_before_the_merge(
    tmp_path: pathlib.Path, key: str, value: object, where: str
) -> None:
    """The layer check meets the key first, and says the same thing, not *unknown key*
    with a list that does not contain it."""
    layer = Layer("user", tmp_path / "config.toml", {"tools": {"locate": {key: value}}})
    with pytest.raises(UnknownKey, match=rf"`tools\.locate\.{key}` is no longer read") as err:
        check_layer(layer)
    assert "(layer: user)" in str(err.value)
    if key == "max_tool_calls":
        assert '`[limits] "locate.max_tool_calls"`' in str(err.value)


def test_an_empty_tools_table_loads(raw: dict[str, Any]) -> None:
    """It chooses nothing: the selection keys `T-3.114` added are all unset."""
    raw["tools"] = {"worker": {}}
    assert not EngineConfig.with_defaults(raw).tools["worker"].chosen()


def test_a_config_without_a_tools_table_loads_with_none(raw: dict[str, Any]) -> None:
    """Nothing is filled: the caps live in the register, not in this table."""
    assert EngineConfig.with_defaults(raw).tools == {}


def test_a_tools_row_for_a_name_that_is_not_a_role_is_rejected(raw: dict[str, Any]) -> None:
    raw["tools"] = {"summarizer": {}}
    with pytest.raises(ConfigError, match="not roles"):
        EngineConfig.with_defaults(raw)
