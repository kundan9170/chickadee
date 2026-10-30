"""The cost formula: the cached-token split, the absent column, unknown-
cache pessimism, the D102 write term — and a golden against worked numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from budget_helpers import CODER, NO_CACHE_COLUMN, NO_WRITE_COLUMN, PRICES, usage

from chickadee.budget.prices import PER_MTOK, PriceTable, RungPrice, cost
from chickadee.config import PriceRow
from chickadee.errors import ConfigError


def test_golden_worked_example() -> None:
    """2,000 prompt tokens of which 500 cached reads and 200 cache writes, 1,000 completion,
    at coder's rates: 1,300 plain in at $1/MTok, 500 cached at $0.10/MTok, 200 writes at
    $1.25/MTok, 1,000 out at $2/MTok."""
    breakdown = cost(usage(2000, 1000, cached=500, writes=200), CODER)
    assert breakdown.input_usd == pytest.approx(0.0013)
    assert breakdown.cached_read_usd == pytest.approx(0.00005)
    assert breakdown.cache_write_usd == pytest.approx(0.00025)
    assert breakdown.output_usd == pytest.approx(0.002)
    assert breakdown.total == pytest.approx(0.0013 + 0.00005 + 0.00025 + 0.002)
    assert not breakdown.write_undercounted


def test_a_price_is_usd_per_million_tokens() -> None:
    """D119 pins the unit as a test, not a comment: `price_in = 3.0` is *three dollars per
    million tokens* — the number on a provider's price page — so 200k input tokens cost
    $0.60, not $600,000. A per-token reading of this table trips the cap on the first call
    (loud); a per-MTok reading of per-token numbers never trips it (silent). Both fail here."""
    assert PER_MTOK == 1_000_000
    published = RungPrice(price_in=3.0, price_out=15.0, price_cached_in=0.3)
    breakdown = cost(usage(200_000, 2_000), published)
    assert breakdown.input_usd == pytest.approx(0.60)
    assert breakdown.output_usd == pytest.approx(0.03)
    assert breakdown.total == pytest.approx(0.63)
    assert cost(usage(1_000_000, 0), published).total == pytest.approx(3.0), "one MTok, $3"


def test_cached_tokens_come_off_the_input_term() -> None:
    plain = cost(usage(1000, 0), CODER)
    half_cached = cost(usage(1000, 0, cached=500), CODER)
    assert plain.total == pytest.approx(0.001)
    assert half_cached.total == pytest.approx(0.0005 + 0.00005)
    assert half_cached.total < plain.total


def test_unknown_cache_status_prices_as_uncached() -> None:
    """`None` is not `0` — the provider said nothing, so no discount is assumed."""
    assert cost(usage(1000, 0, cached=None), CODER).total == cost(usage(1000, 0), CODER).total
    assert cost(usage(1000, 0, cached=0), CODER).total == cost(usage(1000, 0), CODER).total


def test_absent_cached_column_is_no_discount() -> None:
    """A provider with no cached-input rate charges cached tokens as input; billing them
    at zero would flatter the number."""
    with_cache = cost(usage(1000, 0, cached=600), NO_CACHE_COLUMN)
    assert with_cache.total == pytest.approx(0.001)
    assert with_cache.cached_read_tokens == 600


def test_cache_writes_bill_at_the_write_rate_or_at_input_and_say_so() -> None:
    priced = cost(usage(1000, 0, writes=400), CODER)
    assert priced.cache_write_usd == pytest.approx(0.0005)
    assert not priced.write_undercounted
    unpriced = cost(usage(1000, 0, writes=400), NO_WRITE_COLUMN)
    assert unpriced.cache_write_usd == pytest.approx(0.0004)
    assert unpriced.write_undercounted, "the handoff names the undercount (D102)"
    assert not cost(usage(1000, 0, writes=None), NO_WRITE_COLUMN).write_undercounted


def test_the_plain_term_never_goes_negative() -> None:
    """A provider that over-reports cached tokens cannot produce a refund."""
    breakdown = cost(usage(100, 0, cached=150), CODER)
    assert breakdown.input_usd == 0.0
    assert breakdown.total == pytest.approx(0.000015)


def test_an_unpriced_model_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="no price row for model 'zeus'"):
        PRICES.cost(usage(), "zeus")


def test_the_table_loads_from_engine_config_rows() -> None:
    """Keyed by the catalog's model names, as `EngineConfig.prices` is."""
    rows = {
        "local-coder": PriceRow(
            price_in=1.0, price_out=2.0, price_cached_in=0.1, cache_ttl_s=300
        ),
    }
    table = PriceTable.from_config(rows)
    price = table.price_for("local-coder")
    assert price == RungPrice(1.0, 2.0, 0.1, None, False, 300)
    assert price.price_cache_write is None, "no write column on the row: billed at price_in"


def test_a_config_row_with_a_write_column_is_read_when_it_has_one() -> None:
    """D102's `price_cache_write` lands on `config.PriceRow` with `T-3.17`; the ledger reads
    it the moment it is there and bills at `price_in` until then."""

    @dataclass(frozen=True)
    class RowWithWrites:
        price_in: float = 1.0
        price_out: float = 2.0
        price_cached_in: float | None = 0.1
        cache_affinity: bool = False
        cache_ttl_s: int = 0
        price_cache_write: float | None = 1.25

    price = RungPrice.from_config(cast("PriceRow", RowWithWrites()))
    assert price.price_cache_write == 1.25
    assert not cost(usage(1000, 0, writes=400), price).write_undercounted
