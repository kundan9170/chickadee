"""The index's diagnostics, and how a read renders them.

Diagnostics come from the index, never a live server. A file nothing analyzed renders as
unavailable, never as silence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Protocol

from chickadee.context.spans import (
    Span,
    holds,
)

__all__ = [
    "DIAG_MAX_RENDERED",
    "DIAG_MIN_SEVERITY",
    "DiagnosticLike",
    "DiagnosticSetLike",
    "DiagnosticsIndex",
    "render_diagnostics",
    "unavailable",
]


DIAG_MAX_RENDERED: Final = 5
"""Rendered diagnostics per read. A file with forty errors does not need forty lines to
establish that it is broken — the count does that, and the first few say what kind."""


DIAG_MIN_SEVERITY: Final = "warning"
"""Errors and warnings render; anything below is counted, never listed."""


class DiagnosticLike(Protocol):
    """`Diagnostic`, structurally."""

    @property
    def line(self) -> int: ...
    @property
    def col(self) -> int | None: ...
    @property
    def severity(self) -> str: ...
    @property
    def code(self) -> str | None: ...
    @property
    def message(self) -> str: ...
    @property
    def source(self) -> str: ...


class DiagnosticSetLike(Protocol):
    """`DiagnosticSet`: the rows, and the coverage they were read at."""

    @property
    def in_span(self) -> Sequence[DiagnosticLike]: ...
    @property
    def out_of_span(self) -> Sequence[DiagnosticLike]: ...
    @property
    def coverage(self) -> str: ...
    @property
    def analyzed_sha(self) -> str | None: ...
    @property
    def stale_by(self) -> str | None:
        """The file whose change made the verdict stale, when one did."""
        ...


class DiagnosticsIndex(Protocol):
    """`GraphQuery.diagnostics_in`: one indexed SELECT, never a server."""

    def diagnostics_in(self, path: str, lines: range | None = None) -> DiagnosticSetLike: ...


_SEVERITY_RANK: Final[Mapping[str, int]] = {"error": 0, "warning": 1}
"""The severities that render, in order. Anything else is below `DIAG_MIN_SEVERITY`."""


_DIAG_MESSAGE_MAX: Final = 200


def render_diagnostics(
    found: DiagnosticSetLike, *, shown: Sequence[Span] | None, current_sha: str, name: str
) -> tuple[str, ...]:
    """The render rules. `shown` is the view the read returns (None for an outline: the
    whole file is then "here"). A file nobody analyzed never renders like a file with
    nothing wrong (invariant 20): `none` is reached only with coverage and no rows.

    Staleness is judged against the bytes this read returned, not only the index's copy: a
    verdict on other bytes is rendered with its age, never as current. A verdict on these very
    bytes that a dependency's change has outdated names that file."""
    if found.coverage == "unavailable":
        return (f"[diagnostics: unavailable — no analyzer verdict for {name}]",)
    rows = (*found.in_span, *found.out_of_span)
    # Its own bytes changed since the verdict: its age. A dependency's change: that file. Stale
    # by the index's copy alone, with no dependency named, while these very bytes are the ones
    # judged, is the index lagging the disk: the verdict is on what this read returned.
    own = found.analyzed_sha is not None and found.analyzed_sha != current_sha
    if own:
        judged = (found.analyzed_sha or "?")[:6]
        age = f"STALE — judged at {judged}, file is now {current_sha[:6]} · "
    elif found.stale_by is not None:
        age = f"STALE — {found.stale_by} changed since this was judged · "
    elif found.coverage == "stale" and found.analyzed_sha is None:
        age = f"STALE — judged at ?, file is now {current_sha[:6]} · "
    else:
        age = ""
    if not rows:
        return (f"[diagnostics: {age}none]",)
    listed = [d for d in rows if d.severity in _SEVERITY_RANK]
    here = [d for d in listed if shown is None or holds(shown, d.line)]
    elsewhere = [d for d in listed if shown is not None and not holds(shown, d.line)]
    counted = len(rows) - len(listed)
    where = (
        f"{len(here)} in this file"
        if shown is None
        else f"{len(here)} here, {len(elsewhere)} elsewhere in this file"
    )
    below = f" · {counted} below {DIAG_MIN_SEVERITY}, not listed" if counted else ""
    out = [f"[diagnostics: {age}{where}{below}]"]
    ordered = sorted(here, key=_diag_order) + sorted(elsewhere, key=_diag_order)
    out.extend(_diag_row(d) for d in ordered[:DIAG_MAX_RENDERED])
    if len(ordered) > DIAG_MAX_RENDERED:
        out.append(f"  [{len(ordered) - DIAG_MAX_RENDERED} more not listed]")
    return tuple(out)


def _diag_order(d: DiagnosticLike) -> tuple[int, int, int]:
    return _SEVERITY_RANK[d.severity], d.line, d.col or 0


def _diag_row(d: DiagnosticLike) -> str:
    where = f"{d.line}:{d.col}" if d.col is not None else str(d.line)
    code = f" {d.code}" if d.code else ""
    message = " ".join(d.message.split())
    if len(message) > _DIAG_MESSAGE_MAX:
        message = message[: _DIAG_MESSAGE_MAX - 1] + "…"
    return f"  {where} {d.severity}{code}: {message} ({d.source})"


def unavailable(why: str) -> tuple[str, ...]:
    return (f"[diagnostics: unavailable — {why}]",)
