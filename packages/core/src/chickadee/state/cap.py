"""The spend cap in force, as the person's engine config and the environment set it.

One reader for two surfaces: `chickadee init` reports the cap a repository will run under,
and `chickadee setup` the cap it seeded or found. Moved here from `init/orchestrate.py`
unchanged, so the two cannot disagree about which cap is in force.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from chickadee.config import is_cap
from chickadee.state.layers import Layer, LayerReadError
from chickadee.state.merge import is_table

ENGINE_LAYERS = ("user", "environment")
"""The layers that may set the cap: the person's file and the environment above it.
Not the repository's — a cap there refuses the load — so it is never reported."""


def engine_cap(
    engine_config: Path, layers: Sequence[Layer] | LayerReadError
) -> tuple[Decimal | None, str | None]:
    """The cap in force — the engine config's `[spend] hard_cost_usd`, the environment's
    over the file's — and the note a report carries when there is no engine config, or
    no cap in it. The cap is the person's, so init reads it rather than writing
    one; IN-11's finite seed is `chickadee setup`'s."""
    if isinstance(layers, LayerReadError):
        return None, f"the cap in force could not be read: {layers}"
    value: object = None
    for layer in layers:
        spend = layer.body.get("spend") if layer.name in ENGINE_LAYERS else None
        if is_table(spend) and "hard_cost_usd" in spend:
            value = spend["hard_cost_usd"]
    cap = usable_cap(value)
    if not engine_config.is_file():
        return cap, (
            f"no engine config at {engine_config}: models and the spend cap are set there, "
            "not in this repository (D127) — `chickadee setup` writes one"
        )
    if value is None:
        return None, (
            f"the engine config at {engine_config} sets no [spend] hard_cost_usd; a run "
            "refuses to load without a finite cap (D42)"
        )
    if cap is None:
        return None, (
            f"the cap in force, [spend] hard_cost_usd = {value!r}, is not one a run can "
            "use: it must be finite and positive, and a run refuses to load until it is (D42)"
        )
    return cap, None


def usable_cap(value: object) -> Decimal | None:
    """The cap a run would use: a finite, positive number. `inf` is valid TOML and no cap at
    all, and `0` refuses every load, so neither is reported as the cap in force."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not is_cap(value):
        return None
    return Decimal(str(value))
