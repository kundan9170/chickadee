"""The engine config `chickadee setup` seeds.

It **renders**; it writes nothing. The facade's `seed_engine_config` writes it, and only where
no file is (this package performs no disk writes).

What it holds:

- **A finite cap and warning thresholds** (`[spend]`): live, because a run refuses
  to load without a finite cap, and the seed is the one place one is written for a person.
- **Example `[models.<name>]` tables and a `[rungs]` table naming one per rung**,
  commented: the models are the person's to choose, and an example left live would be a model
  nobody picked. Filled in as written, they load, and every role fits one request on its rung.


Example lines are commented with `# `; prose with `#:`. Uncommenting the `# ` lines is filling
the examples in.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from chickadee.errors import HarnessBug
from chickadee.roles import DEFAULT_RUNGS

SEED_HARD_COST_USD = Decimal("5.00")
"""The cap `chickadee init` seeded in older versions moved it here: small enough that a first
session
cannot run up a bill, and the person's to raise."""

SEED_WARNING_FRACTIONS = (Decimal("0.5"), Decimal("0.8"))
"""Warnings at half and four fifths of the cap. Not 1.0: a warning at the cap is the cap."""


@dataclass(frozen=True)
class SeedReport:
    """What `seed_engine_config` did: where the engine config is, whether it wrote it, and the
    cap in force there (the environment's over the file's), with the note a report carries
    when there is no usable cap."""

    path: Path
    written: bool
    hard_cost_cap_usd: Decimal | None
    note: str | None = None


_RUNG_MODEL = {
    "high-param": "frontier",
    "mid-param": "frontier",
    "low-param": "local",
}
"""Which example model each default rung runs on. The frontier model takes the two
upper tiers; the local one the worker's, the calls that are many. `mid-param` is on the
frontier model so the seed, filled in, keeps the critic off the worker's model."""

_MODELS = """\
# [models.frontier]
#: A hosted model, reached with a key from the environment, never written here.
# base_url = "https://api.example.com/v1"
# model_id = "your-frontier-model"
# api_key_env = "FRONTIER_API_KEY"
# context_window = 200000
# max_output_tokens = 32000

# [models.local]
#: A model served on this machine through an OpenAI-compatible endpoint.
# base_url = "http://localhost:8000/v1"
# model_id = "your-local-coder"
# context_window = 131072
# max_output_tokens = 16384
"""


def render() -> str:
    """The seed's text. Pure: the same bytes every time, so a test can compare them."""
    if set(_RUNG_MODEL) != set(DEFAULT_RUNGS):
        # A default rung with no line here would seed a [rungs] table that refuses to load
        # when filled in. Caught at render, in every test, not by a person.
        raise HarnessBug(f"the seed's rungs {sorted(_RUNG_MODEL)} != {list(DEFAULT_RUNGS)}")
    rungs = "\n".join(f'# {rung} = "{_RUNG_MODEL[rung]}"' for rung in DEFAULT_RUNGS)
    warnings = ", ".join(str(f) for f in SEED_WARNING_FRACTIONS)
    return f"""\
#: Your engine config, written by `chickadee setup`. Yours to edit: setup never
#: writes over it. Models and the spend cap live here, never in a repository's file.

[spend]
#: Finite by construction. A run that reaches it stops with a resumable BUDGET_CAP
#: terminal rather than continuing quietly. Raise it here when you mean to spend more.
hard_cost_usd = {SEED_HARD_COST_USD}
warning_fractions = [{warnings}]

#: The models you can run. Uncomment one or both, set your own endpoints, and point
#: every rung at one. A rung exists because [rungs] names it: add a line to declare
#: one of your own, and move a role onto it under [roles]. A model needs a [prices.<name>]
#: row (price_in, price_out, USD per million tokens) before a call to it can be charged.
{_MODELS}
# [rungs]
{rungs}
"""
