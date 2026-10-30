"""The harness's two markers are never read as configuration.

Every spawn carries `CHICKADEE=1` and `CHICKADEE_SESSION` (`process.supervisor.scrub_env`).
A `chickadee` started under the harness inherits both, and the environment layer reads
`CHICKADEE_*` as config: `CHICKADEE_SESSION` became an unknown key `session`, and the child's
config load refused. `31-state.md` §8 reserves the two names.
"""

from __future__ import annotations

from chickadee.process.supervisor import scrub_env
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.layers import (  # pyright: ignore[reportPrivateUsage]
    RESERVED_ENV,
    Layer,
    _from_env,
)
from chickadee.state.resolve import resolve


def test_the_markers_are_not_config() -> None:
    assert _from_env({"CHICKADEE": "1", "CHICKADEE_SESSION": "s-0113"}) == {}


def test_a_child_of_the_harness_loads_its_config() -> None:
    """The environment a spawn actually gets, not a hand-built one: whatever the scrub adds
    has to be something the loader can read back. Over a complete config, because a load
    validates into `EngineConfig` and an empty one refuses, naming its tables."""
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 65_536,
        "max_output_tokens": 16_384,
    }
    complete = {
        "models": {"m": model, "w": dict(model)},
        # The worker's rung on `w`: the critic may not share the worker's model.
        "rungs": {name: "m" for name in DEFAULT_RUNGS} | {"low-param": "w"},
        "spend": {"hard_cost_usd": 5.0},
    }
    layers = [
        Layer("defaults", None, complete),
        Layer("environment", None, _from_env(scrub_env())),
    ]
    resolve(layers)


def test_the_reservation_is_exactly_the_markers() -> None:
    """Any other `CHICKADEE_*` name is still configuration."""
    assert frozenset({"CHICKADEE", "CHICKADEE_SESSION"}) == RESERVED_ENV
    assert _from_env({"CHICKADEE_RUN__MAX_PACKETS": "3"}) == {"run": {"max_packets": 3}}
