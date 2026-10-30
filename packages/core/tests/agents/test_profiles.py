"""D213 decision 1, T-3.188 Design §8: a profile is a role; a person's own profile, in their own
config, is a narrowing of one, and anything that would widen it is refused at load."""

from __future__ import annotations

from typing import Any

import pytest
from engines import raw_config

from chickadee.agents.profiles import config_profiles
from chickadee.config import EngineConfig
from chickadee.errors import ConfigError
from chickadee.roles import role


def _with(profile: dict[str, Any]) -> dict[str, Any]:
    raw = raw_config()
    raw["agents"] = {"profiles": {"quick": profile}}
    return raw


def test_the_built_ins_are_their_roles_as_the_config_resolves_them() -> None:
    config = EngineConfig.with_defaults(raw_config())
    got = config_profiles(config)
    assert set(got) == {"reader", "reviewer", "researcher"}
    reviewer = got["reviewer"]
    policy = role("reviewer").tool_defaults
    assert policy is not None and reviewer.tools == policy.enabled
    assert "shell" in reviewer.tools and "shell" not in got["reader"].tools
    assert (reviewer.limits.max_tool_calls, reviewer.limits.max_cost_cents) == (40, 50)
    researcher = got["researcher"]
    assert researcher.tools == ("web_search", "web_fetch")
    assert (researcher.limits.max_tool_calls, researcher.limits.seconds) == (30, 600)


def test_a_person_narrows_a_base() -> None:
    config = EngineConfig.with_defaults(
        _with({"base": "reviewer", "tools": ["read", "tests_for"], "max_tool_calls": 10})
    )
    quick = config_profiles(config)["quick"]
    assert (quick.role, quick.tools, quick.limits.max_tool_calls) == (
        "reviewer",
        ("read", "tests_for"),
        10,
    )
    assert quick.limits.seconds == config_profiles(config)["reviewer"].limits.seconds


@pytest.mark.parametrize(
    ("profile", "says"),
    [
        ({"base": "reader", "tools": ["read", "shell"]}, "not reader's tools"),
        ({"base": "reader", "max_tool_calls": 31}, "above reader's 30"),
        ({"base": "reader", "max_cost_cents": 51}, "above reader's 50"),
        ({"base": "reader", "max_cumulative_result_tokens": 0.9}, "above reader's 0.4"),
        ({"base": "reader", "rung": "nowhere"}, "is not declared"),
    ],
    ids=["tool", "calls", "cost", "results", "rung"],
)
def test_a_profile_that_would_widen_its_base_is_refused_at_load(
    profile: dict[str, Any], says: str
) -> None:
    with pytest.raises(ConfigError, match=says):
        EngineConfig.with_defaults(_with(profile))


def test_a_built_in_name_cannot_be_redefined() -> None:
    raw = raw_config()
    raw["agents"] = {"profiles": {"reader": {"base": "reader"}}}
    with pytest.raises(ConfigError, match="built-in profile"):
        EngineConfig.with_defaults(raw)


def test_another_rung_sizes_the_result_budget_on_its_own_window() -> None:
    raw = _with({"base": "reader", "rung": "low-param"})
    config = EngineConfig.with_defaults(raw)
    quick = config_profiles(config)["quick"]
    low = raw["models"][raw["rungs"]["low-param"]]
    window = low.get("input_limit") or low["context_window"]
    assert quick.rung == "low-param"
    assert quick.limits.max_result_tokens == int(0.4 * window)
