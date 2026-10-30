"""The rung a box role runs on: its client, its endpoint, the catalog model it is
charged as, its name, and the session ledger's view of it.

Moved from `planner/client.AthenaRung` and named for both box roles, so the worker is
charged and gated before each model call exactly as Athena is: one shape, one home.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chickadee.budget.manager import Ledger
    from chickadee.client.request import ModelClient
    from chickadee.config import RungEndpoint

__all__ = ["BoxRung"]


@dataclass(frozen=True)
class BoxRung:
    """A box role's rung as it runs on it."""

    client: ModelClient
    endpoint: RungEndpoint
    model: str
    """The catalog model it is charged as."""
    name: str
    """The rung."""
    ledger: Ledger | None = None
    """The session ledger's view of it; its `may_start` gates each model call."""
