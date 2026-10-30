"""Contract 8 — Findings. The scout→planner seam.

Pointers are structural, never prose. A planner acting on "the auth logic is
probably in the middleware" cannot be checked against anything; a path and line range can.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

WIRE = ConfigDict(extra="forbid", frozen=True)


class Pointer(BaseModel):
    model_config = WIRE

    path: str
    line_start: int
    line_end: int
    symbol: str | None = None
    edge_class: str | None = None
    """`resolved` | `fallback` | `unknown`. A guess reported at the same confidence
    as a fact is how mislocalization becomes invisible."""

    @model_validator(mode="after")
    def _range_is_sane(self) -> Pointer:
        if self.line_start < 1:
            raise ValueError("line_start is 1-based")
        if self.line_end < self.line_start:
            raise ValueError(f"line_end {self.line_end} precedes line_start {self.line_start}")
        return self


class Unresolved(BaseModel):
    model_config = WIRE

    question: str
    candidates: tuple[str, ...] = ()
    """Candidate lists, never a chosen answer — resolution is the user's via Q1."""


class Findings(BaseModel):
    model_config = WIRE

    targets: tuple[Pointer, ...] = ()
    callers: tuple[Pointer, ...] = ()
    facts: tuple[str, ...] = ()
    unresolved: tuple[Unresolved, ...] = ()
    evidence: tuple[Pointer, ...] = ()
    truncated: bool = False
    """Set when a cap ended the loop. Exhaustion is typed evidence, never a silent
    stop — a truncated report that looked complete would be indistinguishable from a scout
    that simply found nothing."""
