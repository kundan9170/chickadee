"""`locate.snapshot_tokens` is a register row: the scout's map of the
repository has one budget, resolved like every other limit and counted in `locate`'s
never-evicted floor, so a scout rung that cannot hold it refuses at load."""

from __future__ import annotations

import re
from typing import Any

import pytest

from chickadee.config import EngineConfig, RungEndpoint
from chickadee.errors import ConfigError
from chickadee.limits import LIMITS, check_fit, limit, resolve
from chickadee.roles import DEFAULT_RUNGS, ROLES, assign, role
from chickadee.scout.snapshot import SNAPSHOT_TOKENS
from chickadee.scout.tools import LocateCaps

ROW = "locate.snapshot_tokens"
ESCAPED = re.escape(ROW)


def _rung(**over: Any) -> RungEndpoint:
    fields: dict[str, Any] = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 131_072,
        "max_output_tokens": 16_384,
    }
    return RungEndpoint(**{**fields, **over})


def _resolved(overrides: dict[str, float | int] | None = None) -> Any:
    return resolve(
        LIMITS,
        ROLES,
        {name: _rung() for name in DEFAULT_RUNGS},
        overrides or {},
        assignment=assign({}),
    )


def _value(resolved: Any) -> int:
    (row,) = [r for r in resolved if r.name == ROW]
    return int(row.tokens_or_value)


def test_the_row_is_in_the_register_as_set() -> None:
    row = limit(ROW)
    assert (row.value, row.unit, row.per, row.applies_to, row.measured_by) == (
        4096,
        "tokens",
        "call",
        "locate",
        "W8",
    )
    assert row.why


def test_the_scouts_default_is_the_registers_not_a_second_copy() -> None:
    assert limit(ROW).value == SNAPSHOT_TOKENS


def test_it_resolves_to_its_default_and_to_an_override() -> None:
    assert _value(_resolved()) == 4096
    assert _value(_resolved({f"limits.{ROW}": 2048})) == 2048


def test_the_scouts_caps_take_the_resolved_value() -> None:
    caps = LocateCaps.from_resolved(_resolved({f"limits.{ROW}": 2048}))
    assert caps.snapshot_tokens == 2048


def test_locates_never_evicted_floor_counts_the_snapshot() -> None:
    """A window that holds every other floor but not the snapshot refuses, naming it."""
    rung = _rung(context_window=24_000, max_output_tokens=16_000)
    resolved = _resolved()
    with pytest.raises(ConfigError, match=ESCAPED):
        check_fit(
            role("locate"),
            rung,
            resolved,
            template_tokens=4_000,
            rung_name=assign({})["locate"],
        )
    # The same rung fits once the snapshot is lowered.
    lowered = _resolved({f"limits.{ROW}": 512})
    check_fit(
        role("locate"), rung, lowered, template_tokens=4_000, rung_name=assign({})["locate"]
    )


def test_a_scout_rung_too_small_for_the_budget_refuses_at_load() -> None:
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 131_072,
        "max_output_tokens": 16_384,
    }
    models = {f"{name}-model": dict(model) for name in DEFAULT_RUNGS}
    raw: dict[str, Any] = {
        "models": models,
        "rungs": {name: f"{name}-model" for name in DEFAULT_RUNGS},
        "spend": {"hard_cost_usd": 5.0},
        "limits": {"locate": {"snapshot_tokens": 200_000}},
    }
    refused = rf"cannot fit one request on rung 'mid-param'.*{ESCAPED}"
    with pytest.raises(ConfigError, match=refused):
        EngineConfig.with_defaults(raw)
