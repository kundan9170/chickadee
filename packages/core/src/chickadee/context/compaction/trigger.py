"""When to compact: the trigger, the rebuild signal, and
whether a cut pays for the cache it breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from chickadee.context.request import Ctx, fits
from chickadee.limits import limit

if TYPE_CHECKING:
    from chickadee.config import PriceRow
    from chickadee.schema.adapter import CacheMode, CacheObs

__all__ = [
    "COMPACT_TRIGGER_FRACTION",
    "NO_REBUILD",
    "RebuildSignal",
    "Trigger",
    "TriggerBasis",
    "TriggerKind",
    "payback",
    "should_compact",
    "unavoidable_rebuild",
]


COMPACT_TRIGGER_FRACTION: Final = float(limit("all.compact_trigger").value)
"""Owned by `35-limits.md` as `all.compact_trigger`, overridable per role. A configured default,
not a derived threshold, and the secondary trigger: the free rebuild is the one built first."""


class TriggerKind(StrEnum):
    MUST_FIT = "must_fit"
    """Step 0: the request does not go out otherwise. Not economic, and never declined."""
    FREE_REBUILD = "free_rebuild"
    """The suffix is rebuilt regardless, so the cut owes no premium."""
    WARM = "warm"
    """Over the size threshold with the premium still owed. `payback_requests` is its price —
    recorded, never enforced."""


class TriggerBasis(StrEnum):
    """What the trigger was decided on. A cut taken on size because no cache signal existed
    is a different event from one taken on an observed signal, and the log must tell them
    apart or every cache measurement is uninterpretable across deployments."""

    FIT = "fit"
    PROVIDER_OVERFLOW = "provider_overflow"
    """The provider rejected the request as too long: the local estimate that let it go out
    was wrong about the object, so the same estimate cannot decide the retry (§4 of
    `04-adapter.md`)."""
    CACHE_SIGNAL = "cache_signal"
    SIZE_ONLY_NO_CACHE_SIGNAL = "size_only_no_cache_signal"


@dataclass(frozen=True)
class RebuildSignal:
    """What says the cached suffix is rebuilt whatever we do."""

    idle_s: float | None = None
    """Seconds since the last response; None when unknown. Gap history, against the rung's
    TTL, is what says a prefix expired — never one reported miss (condition 2)."""
    model_changed: bool = False
    """The pending request goes to another model or configuration."""
    region_rewrite: bool = False
    """Something is already rewriting this region."""


NO_REBUILD: Final = RebuildSignal()


@dataclass(frozen=True)
class Trigger:
    kind: TriggerKind
    basis: TriggerBasis
    cache_mode: CacheMode
    premium: float | None = None
    """Zero for a free rebuild; not computed otherwise."""
    payback_requests: float | None = None
    """The N: how many requests must follow for the cut to pay. Filled in once the cut
    is planned, recorded, and never enforced — enforcing it needs the session's remaining
    length, the one quantity nothing here can supply."""


def unavoidable_rebuild(price_row: PriceRow, rebuild: RebuildSignal) -> bool:
    """The cache has expired by gap history, or the request changes model or configuration,
    or the region is being rewritten anyway.

    A cold response on the last request is not enough by itself: that request re-wrote the
    prefix, so the next one may well be warm. A TTL of zero is an
    undeclared TTL, never an instant expiry."""
    if rebuild.model_changed or rebuild.region_rewrite:
        return True
    ttl = price_row.cache_ttl_s
    return ttl > 0 and rebuild.idle_s is not None and rebuild.idle_s >= ttl


def should_compact(
    ctx: Ctx,
    price_row: PriceRow,
    cache_obs: CacheObs,
    prev_obs: CacheObs | None,
    pending_tokens: int,
    *,
    rebuild: RebuildSignal = NO_REBUILD,
    provider_overflow: bool = False,
    trigger_fraction: float = COMPACT_TRIGGER_FRACTION,
) -> Trigger | None:
    """§5.4, called once per dispatch attempt and never speculatively: issuing a probe to
    learn the cache's state manufactures the miss that would justify the cut. None is NO.

    `trigger_fraction` is the role's resolved `compact_trigger` when one is configured."""
    mode = cache_obs.mode
    if provider_overflow:
        return Trigger(TriggerKind.MUST_FIT, TriggerBasis.PROVIDER_OVERFLOW, mode)
    if not fits(ctx, pending_tokens):
        return Trigger(TriggerKind.MUST_FIT, TriggerBasis.FIT, mode)
    # Step 0b: `warm is None` is unknowable, not cold. With no signal on either side
    # there is no transition to observe, so the free-rebuild path is skipped entirely and
    # the weaker size trigger is declared rather than inferred later.
    signal = cache_obs.warm is not None and prev_obs is not None and prev_obs.warm is not None
    if signal and unavoidable_rebuild(price_row, rebuild):
        return Trigger(TriggerKind.FREE_REBUILD, TriggerBasis.CACHE_SIGNAL, mode, premium=0.0)
    if ctx.tokens / ctx.window < trigger_fraction:
        return None
    basis = TriggerBasis.CACHE_SIGNAL if signal else TriggerBasis.SIZE_ONLY_NO_CACHE_SIGNAL
    return Trigger(TriggerKind.WARM, basis, mode)


def payback(price_row: PriceRow, s: int, f: int, h: int = 0) -> float | None:
    """`N = ((w - d) * S + H) / (d * F)` — `S` tokens after the cut, `F` freed, `H`
    the residue call in input-token equivalents, `d` and `w` the cached-read and cache-write
    prices over the input price. Arithmetic on an assumed price row, not a measured saving.

    An unpriced cached read is priced at the full input price (B-4 prices unknown cache
    pessimistically) and an unpriced write at the input price too, which is what it bills.
    None when N has no finite value: nothing freed, or reads that cost nothing."""
    if f <= 0 or price_row.price_in <= 0:
        return None
    cached, write = price_row.price_cached_in, price_row.price_cache_write
    d = 1.0 if cached is None else cached / price_row.price_in
    w = 1.0 if write is None else write / price_row.price_in
    if d <= 0:
        return None
    return ((w - d) * s + h) / (d * f)
