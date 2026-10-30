"""The router's types: what it is asked about and what it answers.

`Tier`, `TIER_ORDER` and `FactHit` are `schema/authorization.py`'s — a tier and
the rows that set it are written down and read back, and a person's confirmation carries them
(`ActionApproval.hits`), so they are wire types, and a G0 contract cannot import from a package
built three phases later. They are re-exported here for reading; this package defines no second
one.

The router's run points, its claims and its `TierDecision` retired with the work tier: a
command's tier is the shell's, per call, and the work is approved at the
stops.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from chickadee.schema.authorization import TIER_ORDER, FactHit, Tier

__all__ = [
    "TIER_ORDER",
    "FactHit",
    "Level",
    "Preferences",
    "Tier",
    "higher",
    "highest",
]

Level = Literal["repo", "user"]


def higher(a: Tier, b: Tier) -> Tier:
    """`max()` over `TIER_ORDER`, stated once. `StrEnum` compares as text and "t0" < "t2" is
    true only by the accident of the spellings."""
    return a if TIER_ORDER.index(a) >= TIER_ORDER.index(b) else b


def highest(tiers: tuple[Tier, ...] | list[Tier], *, default: Tier = Tier.T0) -> Tier:
    out = default
    for t in tiers:
        out = higher(out, t)
    return out


@dataclass(frozen=True)
class Preferences:
    """The two levels, already merged by `prefs.py`, so nothing downstream can get the
    precedence wrong. `deny` is path globs; it resolves before the tier and is never a tier."""

    deny: tuple[str, ...] = ()
    command_overrides: Mapping[str, Tier] = field(default_factory=dict[str, Tier])
    """`[approval.commands]`: command class → floor. Lowerable classes only, same rule."""
    denied_commands: frozenset[str] = frozenset()
    """`[approval.commands]` classes set to `deny`: a command of one is refused before the
    tier, as a denied path is. `deny` itself is path globs."""
    source: Mapping[str, Level] = field(default_factory=dict[str, Level])
    """Which level set each rule, for the journal — keyed `deny:<glob>`, `command:<class>`."""
