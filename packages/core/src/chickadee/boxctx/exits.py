"""The box's named exits.

When eviction is not enough, a box role leaves by a named exit. None is a silent truncation and
none retries in a loop: the exit says which role, how many tokens would not fit and against what
window. The worker's round ends `context_full`, and Athena's episode ends.

The packet worker's exit at pack time and the rescue reason Athena took it under retired with
the packet loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["BoxExit", "ExitKind"]

ExitKind = Literal[
    "worker_overflow_mid",
    "athena_exhausted",
    "critic_declined",
    "locate_truncated",
]


@dataclass(frozen=True)
class BoxExit:
    kind: ExitKind
    role: str
    tokens: int
    """Our count of what would not fit: the request after every eviction."""
    window: int
    detail: str
    """One line; never a digest."""

    def __post_init__(self) -> None:
        if "\n" in self.detail or "\r" in self.detail:
            raise ValueError("a BoxExit's detail is one line (36-box-context.md §9)")
