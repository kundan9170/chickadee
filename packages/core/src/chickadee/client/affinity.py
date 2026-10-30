"""Provider cache affinity and cache mode.

~87% of a real agent's bill is prompt-cache traffic, so which replica serves a request is
a cost decision, not a routing detail. Hosted providers shard across replicas and expose a
key to pin them; a single local llama.cpp or vLLM has one prefix cache and needs nothing.

A row answers two independent questions, and conflating them is the bug D82 exists to avoid:

  WHERE does the request land? -> body_field / header (replica affinity)
  HOW is caching CAUSED there? -> cache_mode (explicit / automatic / none)

They are independent because a provider can cache automatically and report nothing back, and
because an `explicit` provider needs markers but no routing key at all.

Rows are config, not code: adding a provider is a TOML entry. That keeps `core` free of
provider-specific business logic — the alternative is a branch per vendor in the
request path, which is where provider assumptions leak in and stay.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from chickadee.schema.adapter import CacheMode


class AffinityRow(BaseModel):
    """How one provider kind wants the session key delivered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    body_field: str | None = None
    """e.g. Fireworks `prompt_cache_key`. The session key becomes this body field."""
    header: str | None = None
    """Providers that route on a header instead."""
    cache_mode: CacheMode = CacheMode.NONE
    """How caching is caused here. `NONE` is the default for the same reason an
    unknown provider injects nothing: assuming a cache that is not there makes CX-1 read a
    permanent miss as a cache-cold transition, and compact forever."""

    def inject(self, body: dict[str, Any], headers: dict[str, str], key: str | None) -> None:
        """Apply this row's rule. A row with neither target injects nothing, which is the
        correct behavior for a single local instance rather than a missing case."""
        if key is None:
            return
        if self.body_field:
            body[self.body_field] = key
        if self.header:
            headers[self.header] = key


# Shipped defaults. `local` is the important row: it exists so that "no injection" is a
# declared decision rather than an unhandled provider — and `cache_mode` extends that same
# idea, so "we do not know whether this caches" is a value rather than a silence.
DEFAULT_ROWS: tuple[AffinityRow, ...] = (
    # Local single instances: one prefix cache, nothing to route to, nothing to mark.
    AffinityRow(provider="local", cache_mode=CacheMode.AUTOMATIC),
    AffinityRow(provider="llama_cpp", cache_mode=CacheMode.AUTOMATIC),
    AffinityRow(provider="vllm", cache_mode=CacheMode.AUTOMATIC),
    # Hosted, prefix-matching: pin the replica, keep the bytes stable, mark nothing.
    AffinityRow(
        provider="fireworks", body_field="prompt_cache_key", cache_mode=CacheMode.AUTOMATIC
    ),
    AffinityRow(
        provider="together", body_field="prompt_cache_key", cache_mode=CacheMode.AUTOMATIC
    ),
    AffinityRow(
        provider="openai", body_field="prompt_cache_key", cache_mode=CacheMode.AUTOMATIC
    ),
    # Explicit: caching happens only where the assembler marked it. No routing key —
    # nothing to pin, because the marker travels with the request rather than the replica.
    AffinityRow(provider="anthropic", cache_mode=CacheMode.EXPLICIT),
    AffinityRow(provider="bedrock", cache_mode=CacheMode.EXPLICIT),
    AffinityRow(provider="openrouter", cache_mode=CacheMode.EXPLICIT),
)


class AffinityTable:
    def __init__(self, rows: tuple[AffinityRow, ...] = DEFAULT_ROWS) -> None:
        self._rows = {r.provider: r for r in rows}

    @staticmethod
    def from_toml(path: Path) -> AffinityTable:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        rows = tuple(AffinityRow.model_validate(r) for r in data.get("affinity", []))
        return AffinityTable(rows or DEFAULT_ROWS)

    def row(self, provider: str) -> AffinityRow:
        """An unknown provider injects nothing rather than guessing a field name.

        Guessing would be worse than doing nothing: an unrecognized body field is a 400 on
        some providers and silently ignored on others, and the second case looks like a
        working cache that never hits.
        """
        return self._rows.get(provider, AffinityRow(provider=provider))

    def providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._rows))
