"""The terms of a read: what was asked, and the bounds and aids it runs within.

Every reading module shares this vocabulary, so it sits below them all: `ReadAsk` and
`parse_ask` (a call's `lines=`, `at=` and `depth=`), the caps `read` owns, the rung's
vision, and the protocols of the read-only aids a reader is given (the symbol index and
the changed lines). `INLINE_BYTES` and `MAX_LINES` are rows of the limits register,
read from it so there is one place to change them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal, Protocol

from chickadee.context.files import FILE_BYTES
from chickadee.context.outline import (
    SymbolLike,
)
from chickadee.context.results import (
    InvalidArgument,
)
from chickadee.context.spans import (
    Span,
)
from chickadee.limits import CHARS_PER_TOKEN, limit

__all__ = [
    "CHANGED_AT",
    "DEPTHS",
    "GREP_IN",
    "IMAGE_MAX_EDGE",
    "INLINE_BYTES",
    "MAX_LINES",
    "NO_RAISE",
    "READ",
    "Caps",
    "ChangedLines",
    "Depth",
    "Held",
    "Raise",
    "ReadAsk",
    "ReadRules",
    "SymbolIndex",
    "Vision",
    "parse_ask",
]


READ: Final = "read"


GREP_IN: Final = "grep_in"

Depth = Literal["outline", "signatures", "full"]


DEPTHS: Final[Mapping[str, Depth]] = {
    "outline": "outline",
    "signatures": "signatures",
    "full": "full",
}


CHANGED_AT: Final = "@changed"
"""The one `at=` that is not an outline entry: the lines that differ from the run's base
 — often twenty lines of four hundred, and git computes it for free."""

# The caps are rows of the limits register. `read`'s line cap has no row of
# its own; `zeus.result_lines`, the line bound every other line-bounded read carries, stands in.

INLINE_BYTES: Final = int(limit("zeus.inline_bytes").value)


MAX_LINES: Final = int(limit("zeus.result_lines").value)


IMAGE_MAX_EDGE: Final = 1568
"""D110's starting value for `image_max_edge`: the ceiling the vendor applies anyway, so the
harness downscaling first makes the cost known rather than whatever the vendor did."""


@dataclass(frozen=True)
class Vision:
    """What the reading rung can see. The rung decides, never the role: any rung that
    holds `read` and can see, sees."""

    rung: str
    supports_images: bool = False
    supports_documents: bool = False
    """Whether the rung's model reads a PDF itself; otherwise it gets the text layer."""
    image_max_edge: int = IMAGE_MAX_EDGE


@dataclass(frozen=True)
class Caps:
    """The bounds `read` owns: an inline byte cap and a line cap on what one read
    newly sends (the read cap), the whole-file threshold, and `grep_in`'s result cap."""

    inline_bytes: int = INLINE_BYTES
    max_lines: int = MAX_LINES
    file_bytes: int = FILE_BYTES
    """The largest file `read` opens at all: it is held whole, before any cap applies."""
    result_lines: int = MAX_LINES
    whole_bytes: int | None = None
    """A plain `read(path)` of a file at or below this returns it whole, above
    it its outline. None is the read cap, Zeus's one limit for both."""

    @property
    def whole_file(self) -> int:
        """The preview's budget: the whole-file threshold, or the read cap without one."""
        return self.inline_bytes if self.whole_bytes is None else self.whole_bytes

    @property
    def any_whole(self) -> bool:
        """Whether a file of any format at or below the threshold is read whole; Zeus's one cap
        keeps that to code the index knows."""
        return self.whole_bytes is not None

    def lowered(self, max_bytes: int | None) -> Caps | InvalidArgument:
        """`max_bytes` may only LOWER the cap: if a caller could raise it, the cap would not
        be the tool's, and D78's rule would leak."""
        if max_bytes is None:
            return self
        if max_bytes < 1:
            return InvalidArgument(
                "max_bytes", "a cap is at least one byte", do="send it again without max_bytes"
            )
        return replace(
            self,
            inline_bytes=min(self.inline_bytes, max_bytes),
            whole_bytes=None if self.whole_bytes is None else min(self.whole_bytes, max_bytes),
        )

    def raised(self, inline_bytes: int, max_lines: int) -> Caps:
        """This read's cap raised for one file: by the reader's own rule,
        never by a caller's `max_bytes`."""
        return replace(
            self,
            inline_bytes=max(self.inline_bytes, inline_bytes),
            max_lines=max(self.max_lines, max_lines),
        )


@dataclass(frozen=True)
class Held:
    """How a re-read of lines already in the conversation is answered. A view
    older than `turns` model turns, or with more than `tokens` of conversation after it, is
    shown again; a recent one is pointed at, then shown `insist` times if asked right after."""

    turns: int
    tokens: int
    insist: int


@dataclass(frozen=True)
class Raise:
    """A file's read cap is raised for the task after `strikes` restarts past
    it, to at most `ceiling` bytes and `share` percent of the context still free. `strikes` 0
    never raises."""

    ceiling: int
    share: int
    strikes: int


NO_RAISE: Final = Raise(0, 0, 0)


@dataclass(frozen=True)
class ReadRules:
    """A role's read limits, from its rows of the limits register: what
    its reader is built with, so no such number is a constant in code."""

    caps: Caps
    held: Held | None = None
    """None: the role's reader keeps no context to answer from."""
    raise_: Raise = NO_RAISE

    @classmethod
    def of(cls, role: str, limits: Mapping[str, int]) -> ReadRules:
        """`role`'s rules from its resolved rows (`limits.role_limits`). A box role has every
        row; Zeus has only the held rows beside his caps, which stay as they are."""
        held = None
        if f"{role}.held_turns" in limits:
            held = Held(
                limits[f"{role}.held_turns"],
                limits[f"{role}.held_tokens"],
                limits[f"{role}.insist_showings"],
            )
        if f"{role}.read_bytes" not in limits:
            return cls(Caps(), held)
        caps = Caps(
            inline_bytes=limits[f"{role}.read_bytes"],
            max_lines=limits[f"{role}.read_lines"],
            whole_bytes=limits[f"{role}.whole_file_bytes"],
        )
        ceiling = int(limits[f"{role}.result_max"] * CHARS_PER_TOKEN)
        raise_ = Raise(ceiling, limits[f"{role}.raise_share"], limits[f"{role}.raise_strikes"])
        return cls(caps, held, raise_)


class SymbolIndex(Protocol):
    """`graph.Graph`, structurally: what an outline and `at=` need. `indexed_sha` is the
    sha256 of the bytes a file's symbols were parsed from, so a reader can tell when they no
    longer describe the file (`Parts.symbols`)."""

    def symbols_in(self, path: str) -> Sequence[SymbolLike]: ...
    def file_known(self, path: str) -> bool: ...
    def indexed_sha(self, path: str) -> str | None: ...


class ChangedLines(Protocol):
    """Where a file differs from the run's base, for `at="@changed"` (`context/changes.py`).
    None when there is no base to compare against, or git could not say. A span may run past
    the file's end; the reader clips it to the lines it holds."""

    def changed_lines(self, path: str) -> Sequence[Span] | None: ...


@dataclass(frozen=True)
class ReadAsk:
    region: Span | None
    at: str | None
    depth: Depth | None

    @property
    def named(self) -> bool:
        """Whether the call named a region — which is what `depth` defaults by."""
        return self.region is not None or self.at is not None


def parse_ask(
    lines: Span | str | None, at: str | None, depth: str | None
) -> ReadAsk | InvalidArgument:
    region: Span | None = None
    if isinstance(lines, str):
        try:
            region = Span.parse(lines)
        except ValueError as exc:
            return InvalidArgument(
                "lines", str(exc), do='send lines as "a-b", from 1, for example "10-40"'
            )
    else:
        region = lines
    if at is not None and not at.strip():
        return InvalidArgument(
            "at",
            "at= is empty",
            do="name an outline entry (a function or class name), or leave at= out",
        )
    if region is not None and at is not None:
        return InvalidArgument(
            "at",
            "lines= and at= both name a region",
            do="send one of them: lines= for a range, or at= for an outline entry",
        )
    chosen = DEPTHS.get(depth) if depth is not None else None
    if depth is not None and chosen is None:
        return InvalidArgument(
            "depth",
            f"{depth!r} is not outline, signatures or full",
            do='send depth as "outline", "signatures" or "full"',
        )
    return ReadAsk(region, at, chosen)
