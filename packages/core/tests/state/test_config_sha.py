"""What `config_sha` covers, and what provenance adds to it."""

from __future__ import annotations

from typing import Any

from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import Layer
from chickadee.state.resolve import resolve


def _complete() -> dict[str, Any]:
    """The smallest config that validates: a catalog of two models, the worker's rung on `w`
    and every other on `m` (the critic may not share the worker's model), and a finite
    cap."""
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 65_536,
        "max_output_tokens": 16_384,
    }
    return {
        "models": {"m": model, "w": dict(model)},
        "rungs": {name: "m" for name in DEFAULT_RUNGS} | {"low-param": "w"},
        "spend": {"hard_cost_usd": 5.0},
    }


def test_config_sha_is_about_the_merge_not_the_split() -> None:
    """ST-9. The same resolved config from different layer splits hashes the same —
    otherwise the sha would describe the filesystem rather than the configuration. Each
    layer sets only what its scope allows: the engine's in the engine config, the
    repository's facts in the project's."""
    whole = _complete()
    whole["profiles"] = "python"
    whole["spend"]["warning_fractions"] = [0.5, 0.9]
    one = resolve([Layer("defaults", None, whole)])
    two = resolve(
        [
            Layer("defaults", None, _complete()),
            Layer("user", None, {"spend": {"warning_fractions": [0.5, 0.9]}}),
            Layer("project", None, {"profiles": "python"}),
        ]
    )
    assert one.config == two.config
    assert one.config_sha == two.config_sha
    assert one.engine == two.engine


def test_config_sha_is_about_the_limit_not_its_spelling() -> None:
    """`35-limits.md` §4 spells one key two ways; the sha is of the key, so a person who
    rewrites `[limits.worker]` as a dotted key has not changed their configuration."""
    nested = _complete() | {"limits": {"worker": {"max_tool_calls": 60}}}
    dotted = _complete() | {"limits": {"worker.max_tool_calls": 60}}
    one = resolve([Layer("user", None, nested)])
    two = resolve([Layer("user", None, dotted)])
    assert one.config_sha == two.config_sha


def test_provenance_records_which_layer_moved() -> None:
    """A sha says two runs differed; provenance says which layer differed."""
    out = resolve(
        [
            Layer("defaults", None, _complete()),
            Layer("user", None, {"spend": {"hard_cost_usd": 7.5}}),
        ]
    )
    by_key = {p.key: p.layer for p in out.provenance}
    assert by_key["spend.hard_cost_usd"] == "user"
    assert by_key["models.m.model_id"] == "defaults"
    assert by_key["rungs.low-param"] == "defaults"
    assert out.engine.spend.hard_cost_usd == 7.5
