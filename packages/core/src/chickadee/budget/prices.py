"""The price table and the cost formula.

    cost = ( (prompt - cached_read - cached_write) * price_in
           + cached_read * (price_cached_in ?? price_in) # absent column: no discount
           + cached_write * (price_cache_write ?? price_in)
           + completion * price_out ) / 1_000_000

Prices are **USD per million tokens** — the unit every provider publishes and the
one `config.PriceRow` and the formula both state; `cost()` divides by `PER_MTOK` once
and nothing else in the harness converts. Unknown cache status — the provider reported no cached
count
— prices as uncached: pessimistic, never flattering. A model the table does not price
raises `ConfigError`: charging against an unpriced model is an operator error, and silently
pricing it at zero would flatter every number we report.

The table is keyed by the catalog's model names, not by rung: a price is the provider's,
for a model, and a call is charged at the model that served it — which, after a `/rung`
switch, is not always the model its rung started the session on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from chickadee.config import PriceRow
from chickadee.errors import ConfigError
from chickadee.schema.adapter import Usage

__all__ = ["PER_MTOK", "CostBreakdown", "PriceTable", "RungPrice", "cost"]

PER_MTOK = 1_000_000.0
"""Tokens per price unit. The one place the unit is applied: a per-token reading of a
per-MTok table over-reports by 10⁶ and trips the cap on the first call, which is loud; a
per-MTok reading of per-token numbers under-reports by 10⁶ and the cap never fires, which
is the silent direction the cap exists to prevent. Hence one constant, one division, one test
that pins it."""


@dataclass(frozen=True)
class RungPrice:
    """One model's prices, USD per million tokens. `config.PriceRow` is the wire
    shape; this is the ledger's, with the cache-write column that the config row does
    not carry yet. The table is keyed by catalog model, not rung; the type
    keeps the name it had."""

    price_in: float
    price_out: float
    price_cached_in: float | None = None
    """None ⇒ the provider has no cached-input rate. Absent means *no discount is
    known*, so cached reads bill at `price_in` — the formula comment says the term is 0
    when the column is absent, which would make cached tokens free; the locked rule says
    pessimistic, never flattering, and the config row's own docstring reads it that way."""
    price_cache_write: float | None = None
    """None ⇒ writes are billed at `price_in`, and the handoff names the undercount
    when the provider reported writes."""
    cache_affinity: bool = False
    cache_ttl_s: int = 0
    """Per provider, never a hardcoded five minutes."""

    @staticmethod
    def from_config(row: PriceRow) -> RungPrice:
        # `price_cache_write` arrives on `config.PriceRow`; read it when
        # the row has it and bill writes at `price_in` (the undercount the handoff names)
        # when it does not, so the column flows through whichever branch merges second.
        write_rate: object = getattr(row, "price_cache_write", None)
        return RungPrice(
            price_in=row.price_in,
            price_out=row.price_out,
            price_cached_in=row.price_cached_in,
            price_cache_write=float(write_rate)
            if isinstance(write_rate, (int, float))
            else None,
            cache_affinity=row.cache_affinity,
            cache_ttl_s=row.cache_ttl_s,
        )


@dataclass(frozen=True)
class CostBreakdown:
    """The formula's four terms, so a report can show where a dollar went."""

    input_usd: float
    cached_read_usd: float
    cache_write_usd: float
    output_usd: float
    cached_read_tokens: int
    cache_write_tokens: int
    write_undercounted: bool
    """True when the provider reported cache writes and the model has no write price: the
    writes were billed at `price_in`, which Anthropic-style providers charge above.
    The handoff names this; a run does not."""

    @property
    def total(self) -> float:
        return self.input_usd + self.cached_read_usd + self.cache_write_usd + self.output_usd


def cost(usage: Usage, price: RungPrice) -> CostBreakdown:
    """The formula, prices per MTok. Cached reads and writes come off the
    plain-input term; each is billed at its own rate when the model has one and at
    `price_in` when it does not."""
    cached_read = usage.cached_prompt_tokens or 0
    cached_write = usage.cache_write_tokens or 0
    plain = max(0, usage.prompt_tokens - cached_read - cached_write)
    read_rate = price.price_cached_in if price.price_cached_in is not None else price.price_in
    write_rate = (
        price.price_cache_write if price.price_cache_write is not None else price.price_in
    )
    return CostBreakdown(
        input_usd=plain * price.price_in / PER_MTOK,
        cached_read_usd=cached_read * read_rate / PER_MTOK,
        cache_write_usd=cached_write * write_rate / PER_MTOK,
        output_usd=usage.completion_tokens * price.price_out / PER_MTOK,
        cached_read_tokens=cached_read,
        cache_write_tokens=cached_write,
        write_undercounted=cached_write > 0 and price.price_cache_write is None,
    )


@dataclass(frozen=True)
class PriceTable:
    """Keyed by model — the catalog's names — as closed config. A model with no row
    raises rather than pricing at zero: a rung is charged at the model it was running when
    the call was made, so callers pass the model that served it, never the rung."""

    rows: Mapping[str, RungPrice]
    """Model name → row."""

    @staticmethod
    def from_config(prices: Mapping[str, PriceRow]) -> PriceTable:
        return PriceTable({name: RungPrice.from_config(row) for name, row in prices.items()})

    def price_for(self, model: str) -> RungPrice:
        try:
            return self.rows[model]
        except KeyError:
            raise ConfigError(
                f"no price row for model {model!r} (priced: {sorted(self.rows)}); a charge "
                "against an unpriced model is an operator error, and pricing it at zero "
                "would flatter every number we report (08-budget.md §6)"
            ) from None

    def cost(self, usage: Usage, model: str) -> CostBreakdown:
        return cost(usage, self.price_for(model))
