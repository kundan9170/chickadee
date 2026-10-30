"""Plain stand-ins for the budget suite: a price table, caps, a driven clock."""

from __future__ import annotations

from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable, RungPrice
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema.adapter import Usage

# USD per million tokens, the unit a provider publishes. Chosen so the arithmetic
# is exact in decimal: a 1,000-token prompt at `local-coder` ($1/MTok) is $0.001, its
# completion ($2/MTok) $0.002, a cached read a tenth of the input rate.
CODER = RungPrice(price_in=1.0, price_out=2.0, price_cached_in=0.1, price_cache_write=1.25)
NO_CACHE_COLUMN = RungPrice(price_in=1.0, price_out=2.0, price_cached_in=None)
NO_WRITE_COLUMN = RungPrice(price_in=1.0, price_out=2.0, price_cached_in=0.1)
FRONTIER = RungPrice(price_in=10.0, price_out=50.0, price_cached_in=1.0)

# Keyed by model, the catalog's names — none of them a rung's name, so a charge made
# against a rung is an unpriced one here, as it is in a real config.
CODER_MODEL = "local-coder"
FRONTIER_MODEL = "frontier"
PRICES = PriceTable(
    {
        CODER_MODEL: CODER,
        "plain": NO_CACHE_COLUMN,
        "nowrite": NO_WRITE_COLUMN,
        FRONTIER_MODEL: FRONTIER,
    }
)


def usage(
    prompt: int = 1000,
    completion: int = 1000,
    cached: int | None = None,
    writes: int | None = None,
    *,
    estimated: bool = False,
) -> Usage:
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_prompt_tokens=cached,
        cache_write_tokens=writes,
        estimated=estimated,
    )


class DrivenClock:
    """A monotonic clock a test advances by hand — or turns backwards, to prove B-5."""

    def __init__(self, at: float = 100.0) -> None:
        self.now = at

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def ledger(
    cap: float | None = 0.01,
    fractions: tuple[float, ...] = (0.8,),
    packet_warning: float | None = None,
    clock: DrivenClock | None = None,
) -> CostLedger:
    caps = BudgetCaps(
        hard_cost_cap_usd=cap,
        warning_fractions=fractions,
        packet_warning_cost_usd=packet_warning,
    )
    return CostLedger(PRICES, caps, clock or DrivenClock(), rungs=DEFAULT_RUNGS)


CALL_USD = 0.003
"""One `local-coder` call of `usage()`: $0.001 in, $0.002 out."""
