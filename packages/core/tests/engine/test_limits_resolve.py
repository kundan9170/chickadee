"""A fraction resolves against `input_limit` when set, else
`context_window`; a config override wins over the default; `all` expands to one row per
role, and `limits.<role>.<name>` overrides it for that role; a key naming no row, a fraction
outside (0, 1] and a non-positive count are each a `ConfigError` naming the key."""

from __future__ import annotations

import math
from typing import Any

import pytest

from chickadee.config import EngineConfig, RungEndpoint
from chickadee.errors import ConfigError
from chickadee.limits import HARNESS_OWNERS, LIMITS, ResolvedLimit, resolve, window
from chickadee.roles import DEFAULT_RUNGS, ROLES, assign


def _rung(**over: Any) -> RungEndpoint:
    fields: dict[str, Any] = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 131_072,
        "max_output_tokens": 16_384,
    }
    return RungEndpoint(**{**fields, **over})


def _rungs(by_name: dict[str, RungEndpoint] | None = None) -> dict[str, RungEndpoint]:
    """Each default rung's endpoint, `by_name` over the default one (keyed by rung)."""
    return {name: (by_name or {}).get(name, _rung()) for name in DEFAULT_RUNGS}


def _get(resolved: tuple[ResolvedLimit, ...], role: str, name: str) -> ResolvedLimit:
    return next(r for r in resolved if r.role == role and r.name == name)


def test_window_is_input_limit_when_set_else_context_window() -> None:
    assert window(_rung()) == 131_072
    assert window(_rung(input_limit=100_000)) == 100_000


def test_a_fraction_resolves_against_context_window() -> None:
    out = resolve(LIMITS, ROLES, _rungs(), {}, assignment=assign({}))
    got = _get(out, "worker", "worker.design_layer")
    assert got.tokens_or_value == math.floor(0.15 * 131_072)
    assert (got.rung, got.source) == ("low-param", "default")


def test_a_fraction_resolves_against_input_limit_when_set() -> None:
    out = resolve(
        LIMITS,
        ROLES,
        _rungs({"low-param": _rung(input_limit=100_000)}),
        {},
        assignment=assign({}),
    )
    assert _get(out, "worker", "worker.design_layer").tokens_or_value == 15_000


def test_a_non_fraction_is_its_value() -> None:
    out = resolve(LIMITS, ROLES, _rungs(), {}, assignment=assign({}))
    assert _get(out, "worker", "worker.max_tool_calls").tokens_or_value == 200
    assert _get(out, "zeus", "zeus.inline_bytes").tokens_or_value == 4096


def test_an_override_wins_and_says_so() -> None:
    out = resolve(
        LIMITS,
        ROLES,
        _rungs(),
        {"limits.worker.max_tool_calls": 12, "limits.worker.design_layer": 0.5},
        assignment=assign({}),
    )
    calls = _get(out, "worker", "worker.max_tool_calls")
    assert (calls.tokens_or_value, calls.source) == (12, "config")
    assert _get(out, "worker", "worker.design_layer").tokens_or_value == 65_536


def test_all_expands_to_one_row_per_role() -> None:
    out = resolve(LIMITS, ROLES, _rungs(), {}, assignment=assign({}))
    rows = [r for r in out if r.name == "all.compact_trigger"]
    assert [r.role for r in rows] == [r.name for r in ROLES]
    assert {r.rung for r in rows} == set(DEFAULT_RUNGS)


def test_a_per_role_key_beats_the_shared_one_for_that_role_only() -> None:
    out = resolve(
        LIMITS,
        ROLES,
        _rungs(),
        {"limits.all.compact_trigger": 0.7, "limits.athena.compact_trigger": 0.9},
        assignment=assign({}),
    )
    assert _get(out, "athena", "all.compact_trigger").tokens_or_value == math.floor(
        0.9 * 131_072
    )
    assert _get(out, "zeus", "all.compact_trigger").tokens_or_value == math.floor(0.7 * 131_072)


def test_resolution_follows_the_configs_role_assignment() -> None:
    out = resolve(
        LIMITS,
        ROLES,
        _rungs({"high-param": _rung(context_window=65_536)}),
        {},
        assignment=assign({"worker": "high-param"}),
    )
    got = _get(out, "worker", "worker.design_layer")
    assert (got.rung, got.tokens_or_value) == ("high-param", 9_830)


@pytest.mark.parametrize(
    "key",
    [
        "limits.worker.max_tool_call",  # a typo
        "limits.planner.max_tool_calls",  # a rung, not a role
        "limits.summarizer.compact_trigger",  # not in the cast
        "limits.worker.inline_bytes",  # a row, but not the worker's
    ],
)
def test_a_key_naming_no_row_is_refused_by_name(key: str) -> None:
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        resolve(LIMITS, ROLES, _rungs(), {key: 1}, assignment=assign({}))


@pytest.mark.parametrize("value", [0, 0.0, -0.1, 1.5, True, "0.3"])
def test_a_fraction_outside_zero_one_is_refused_naming_the_key(value: Any) -> None:
    with pytest.raises(ConfigError, match=r"limits\.worker\.design_layer"):
        resolve(
            LIMITS,
            ROLES,
            _rungs(),
            {"limits.worker.design_layer": value},
            assignment=assign({}),
        )


def test_a_whole_window_is_a_fraction() -> None:
    out = resolve(
        LIMITS,
        ROLES,
        _rungs(),
        {"limits.zeus.max_cumulative_result_tokens": 1},
        assignment=assign({}),
    )
    got = _get(out, "zeus", "zeus.max_cumulative_result_tokens")
    assert got.tokens_or_value == 131_072


@pytest.mark.parametrize("value", [0, -3, 2.5, False])
def test_a_non_positive_or_fractional_count_is_refused_naming_the_key(value: Any) -> None:
    with pytest.raises(ConfigError, match=r"limits\.locate\.max_tool_calls"):
        resolve(
            LIMITS,
            ROLES,
            _rungs(),
            {"limits.locate.max_tool_calls": value},
            assignment=assign({}),
        )


# --- through the config loader ------------------------------------------------------------


def _raw(**extra: Any) -> dict[str, Any]:
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 131_072,
        "max_output_tokens": 16_384,
    }
    return {
        "models": {f"{name}-model": dict(model) for name in DEFAULT_RUNGS},
        "rungs": {name: f"{name}-model" for name in DEFAULT_RUNGS},
        "spend": {"hard_cost_usd": 5.0},
        **extra,
    }


def test_the_config_resolves_once_at_load() -> None:
    cfg = EngineConfig.with_defaults(_raw())
    per_role = [lim for lim in LIMITS if lim.applies_to not in HARNESS_OWNERS]
    assert len(cfg.resolved_limits) == len(
        [r for lim in per_role for r in (ROLES if lim.applies_to == "all" else (0,))]
    )
    # A harness-wide row is one number, the config's or its default.
    assert cfg.harness_limit("lsp.servers_max") == 8  # since B2
    assert _get(cfg.resolved_limits, "locate", "locate.max_tool_calls").tokens_or_value == 40


def test_nested_and_dotted_tables_spell_the_same_key() -> None:
    nested = EngineConfig.with_defaults(_raw(limits={"worker": {"max_tool_calls": 9}}))
    dotted = EngineConfig.with_defaults(_raw(limits={"worker.max_tool_calls": 9}))
    for cfg in (nested, dotted):
        got = _get(cfg.resolved_limits, "worker", "worker.max_tool_calls")
        assert (got.tokens_or_value, got.source) == (9, "config")


def test_the_config_refuses_an_unknown_limit_key() -> None:
    with pytest.raises(ConfigError, match=r"limits\.worker\.max_tool_call\b"):
        EngineConfig.with_defaults(_raw(limits={"worker": {"max_tool_call": 9}}))


def test_the_config_resolves_against_its_role_assignment() -> None:
    raw = _raw(roles={"worker": "high-param"})
    raw["models"]["high-param-model"]["context_window"] = 65_536
    cfg = EngineConfig.with_defaults(raw)
    got = _get(cfg.resolved_limits, "worker", "worker.design_layer")
    assert (got.rung, got.tokens_or_value) == ("high-param", 9_830)


def test_a_fraction_resolves_exactly_as_written() -> None:
    """0.29 * 100,000 is 28,999.999… in binary floats; the bound is the number written."""
    out = resolve(
        LIMITS,
        ROLES,
        _rungs({"low-param": _rung(input_limit=100_000)}),
        {"limits.worker.design_layer": 0.29},
        assignment=assign({}),
    )
    assert _get(out, "worker", "worker.design_layer").tokens_or_value == 29_000


def test_a_shadowed_shared_value_is_still_checked() -> None:
    """Every role has its own key, so `limits.all.compact_trigger` is read by none of them;
    an out-of-range value is refused anyway."""
    overrides: dict[str, float | int] = {"limits.all.compact_trigger": 5.0}
    overrides |= {f"limits.{r.name}.compact_trigger": 0.7 for r in ROLES}
    with pytest.raises(ConfigError, match=r"limits\.all\.compact_trigger"):
        resolve(LIMITS, ROLES, _rungs(), overrides, assignment=assign({}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"limits.all.compact_target": 0.9},
        {"limits.worker.compact_trigger": 0.4},
        {"limits.athena.compact_target": 0.8},
    ],
)
def test_a_compaction_target_at_or_above_its_trigger_is_refused(
    overrides: dict[str, float | int],
) -> None:
    with pytest.raises(ConfigError, match=r"compact_target .* is not below compact_trigger"):
        resolve(LIMITS, ROLES, _rungs(), overrides, assignment=assign({}))


def test_a_key_spelled_both_ways_is_refused() -> None:
    raw = _raw(limits={"worker.max_tool_calls": 10, "worker": {"max_tool_calls": 9}})
    with pytest.raises(ConfigError, match=r"limits\.worker\.max_tool_calls is set twice"):
        EngineConfig.with_defaults(raw)
