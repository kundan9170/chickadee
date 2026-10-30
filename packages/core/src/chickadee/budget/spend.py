"""`Spend` — what one scope has spent, and the accumulator behind it.

Three scopes: the **session** owns the ledger and the cap — the budget belongs to the
session lineage, otherwise a delegated call resets the limit; **run** is one delegated coder
episode; **packet** is the current packet, reset on `start_packet`. Charged quantities are
tokens, cost and wall; retries are the machine's own counter and are not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

__all__ = ["SCOPES", "Scope", "Spend", "SpendAccumulator"]

Scope = Literal["session", "run", "packet"]
SCOPES: tuple[str, ...] = get_args(Scope)


@dataclass(frozen=True)
class Spend:
    """A read-only snapshot for the stream and the handoff."""

    scope: Scope
    tokens_in: int
    tokens_out: int
    cached_in: int
    cost_usd: float
    wall_ms: int
    cache_writes: int = 0
    """Cache-creation tokens. Beside the other fields so the handoff can name the
    undercount when a rung prices writes and the provider reported them."""
    estimated_calls: int = 0
    """Calls whose usage was client-estimated (`Usage.estimated`), so a cost total can say
    how much of it rests on reported tokens."""


class SpendAccumulator:
    """One scope's running totals. Mutable on purpose — it is the ledger's private state;
    `snapshot()` is what leaves it."""

    def __init__(self, scope: Scope) -> None:
        self.scope: Scope = scope
        self.tokens_in = 0
        self.tokens_out = 0
        self.cached_in = 0
        self.cache_writes = 0
        self.cost_usd = 0.0
        self.wall_ms = 0
        self.estimated_calls = 0

    def add_usage(
        self,
        *,
        tokens_in: int,
        tokens_out: int,
        cached_in: int,
        cache_writes: int,
        cost_usd: float,
        estimated: bool,
    ) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.cached_in += cached_in
        self.cache_writes += cache_writes
        self.cost_usd += cost_usd
        if estimated:
            self.estimated_calls += 1

    def add_wall(self, ms: int) -> None:
        # A clock anomaly clamps at zero rather than going negative.
        self.wall_ms += max(0, ms)

    def reset(self) -> None:
        self.__init__(self.scope)

    def snapshot(self) -> Spend:
        return Spend(
            scope=self.scope,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cached_in=self.cached_in,
            cost_usd=self.cost_usd,
            wall_ms=self.wall_ms,
            cache_writes=self.cache_writes,
            estimated_calls=self.estimated_calls,
        )
