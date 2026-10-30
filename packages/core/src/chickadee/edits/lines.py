"""Lines: one splitter, one newline policy, the `N│` line renderer, and the line anchor styles.


Every view that numbers a file's lines builds its rows here, and every reader of those rows
parses them here, so a prefix is spelled and measured once. The newline policy is what the
worker is shown and what an anchor is matched against: they must be the same text, or an
anchor copied from a view cannot match.

`ANCHOR_STYLES` are how a line tool's anchor is written: `number` (`12`) and `hash`
(`12:a3f`, the line's tag beside its number). A style renders every view of a target when a
line tool is offered, gives the anchor arguments their schema, and resolves an anchor to a line
or to a `Stale` naming the lines there now. `tag` is the one tag function (guard row E5).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from chickadee.context.spans import Span

__all__ = [
    "ANCHOR_STYLES",
    "BAR",
    "BAR_BYTES",
    "HASH",
    "NUMBER",
    "ROW_CHARS_MAX",
    "AnchorKind",
    "AnchorStyle",
    "Decoded",
    "Stale",
    "as_decoded",
    "clip",
    "decode",
    "elided",
    "encode",
    "hit",
    "number_lines",
    "parse_row",
    "prefix_bytes",
    "row",
    "split_lines",
    "tag",
]

BAR: Final = "│"
"""What separates a line's number from its text."""
BAR_BYTES: Final = len(BAR.encode())

_ROW: Final = re.compile(rf"^ *(\d+)(?::[0-9a-f]{{3}})?{BAR}(.*)$")


def split_lines(text: str, *, keepends: bool = False) -> tuple[str, ...]:
    """Lines as a reader numbers them: split on newline, a trailing newline ending the last
    line rather than starting an empty one, and a CR before the newline not part of it.

    With `keepends`, each line keeps its newline, CR included, so the lines join back to
    `text` exactly: the same lines, for a caller that cuts text at a line rather than reads it.

    Only `\\n` splits. `str.splitlines` also splits on a form feed, a vertical tab, a lone CR
    and `\\u2028`, so a line it numbers can be a line later than the file's."""
    if not text:
        return ()
    parts = text.split("\n")
    last = parts.pop()
    if keepends:
        return (*(p + "\n" for p in parts), *((last,) if last else ()))
    if last:
        parts.append(last)
    return tuple(p[:-1] if p.endswith("\r") else p for p in parts)


# -- the newline policy ------------------------------------------------------------------


@dataclass(frozen=True)
class Decoded:
    """A file's text as it is shown and matched, and how to write it back."""

    text: str
    crlf: bool
    """Every newline in the file was `\\r\\n`, and `text` shows each as `\\n`."""


def decode(text: str) -> Decoded:
    """A text whose every newline is `\\r\\n` is shown and matched as `\\n`, and `encode`
    writes it back as `\\r\\n`; any other text (LF, mixed, none) as it is. A uniform file maps
    line for line, so nothing can match across what the file separates; a mixed one is left
    exact rather than guessed at."""
    newlines = text.count("\n")
    if newlines and text.count("\r\n") == newlines:
        return Decoded(text.replace("\r\n", "\n"), crlf=True)
    return Decoded(text, crlf=False)


def as_decoded(text: str, crlf: bool) -> str:
    """Text written into a decoded file, in the form the file is held and matched in: a
    uniformly `\r\n` file is held as `\n`, so an anchor or a replacement written with the
    file's own `\r\n` is read as `\n` too, and `encode` writes it back once. Any other file is
    held as it is, and so is what is written into it."""
    return text.replace("\r\n", "\n") if crlf else text


def encode(text: str, decoded: Decoded) -> str:
    """`text`, made from `decoded`'s text, in the file's own newlines: each `\n` becomes
    `\r\n`, and nothing else changes, so a `\r` the file holds before a newline survives the
    round trip. Text written into it is in decoded form already (`match` decodes a hunk's)."""
    if not decoded.crlf:
        return text
    return text.replace("\n", "\r\n")


# -- the line renderer --------------------------------------------------------------------

ROW_CHARS_MAX: Final = 240
"""The characters a row shows of one line before it is clipped: a minified or generated line can
be thousands, and a row is read, not searched."""


def clip(text: str, limit: int = ROW_CHARS_MAX) -> str:
    """`text`, cut to `limit` characters and marked with how many more there were."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[+{len(text) - limit:,} chars]"


def row(n: int, width: int, text: str, *, tagged: bool = False) -> str:
    """One numbered line: `n`, right-aligned to `width`, its tag when `tagged` (the `hash`
    style), the bar, the text."""
    anchor = f"{n:>{width}}:{tag(text)}" if tagged else f"{n:>{width}}"
    return f"{anchor}{BAR}{text}"


def hit(path: str, anchor: str, text: str) -> str:
    """One search hit: `path:12:text`, or `path:12:a3f:text` under the `hash` style."""
    return f"{path}:{anchor}:{text}"


def prefix_bytes(last: int) -> int:
    """The bytes a row's prefix takes when `last` is the widest number shown."""
    return len(str(last)) + BAR_BYTES


def parse_row(text: str) -> tuple[int, str] | None:
    """A rendered row's line number and text, or None for anything else (a marker, prose)."""
    m = _ROW.match(text)
    return (int(m.group(1)), m.group(2)) if m is not None else None


def elided(start: int, end: int) -> str:
    """The marker for lines left out: an omission is shown as one, never reconstructed and
    never silent (invariant 16)."""
    return f"[elided: line {start}]" if start == end else f"[elided: lines {start}-{end}]"


def number_lines(
    lines: Sequence[str],
    spans: Sequence[Span],
    *,
    width: int | None = None,
    tagged: bool = False,
) -> list[str]:
    """Every line as `N│content`, `N` its line in the FILE, fixed width across the
    result, tagged under the `hash` style; a gap between spans is marked
    `[elided: lines a-b]`, never silence (invariant 16)."""
    if not spans:
        return []
    w = width if width is not None else len(str(spans[-1].end))
    rows: list[str] = []
    previous: Span | None = None
    for span in spans:
        if previous is not None and span.start > previous.end + 1:
            rows.append(elided(previous.end + 1, span.start - 1))
        rows.extend(
            row(n, w, lines[n - 1], tagged=tagged) for n in range(span.start, span.end + 1)
        )
        previous = span
    return rows


# -- line anchors ----------------------------------------------------------------

TAG_DIGITS: Final = 3
"""A tag's hex digits: a starting guess until E1 measures it (the owner's item 2). A
collision within a file is harmless: the anchor is the number and the tag together."""


def tag(line: str) -> str:
    """A line's tag: the first `TAG_DIGITS` hex digits of the SHA-256 of its text without its
    line ending. The one tag function (guard row E5)."""
    bare = line.removesuffix("\n").removesuffix("\r")
    return hashlib.sha256(bare.encode("utf-8")).hexdigest()[:TAG_DIGITS]


AnchorKind = Literal["line", "after"]
"""What an anchor argument names: a line, or the line an insertion follows (`0`, the file's
start)."""

_TAGGED: Final = re.compile(r"^([0-9]+):([0-9a-f]{3})$")
_AFTER: Final = re.compile(r"^(?:0|[0-9]+:[0-9a-f]{3})$")
_STALE_CONTEXT: Final = 2
"""The lines either side of a stale anchor's line that its refusal shows."""


@dataclass(frozen=True)
class Stale:
    """An anchor that names no line it can be applied to: why, with the lines there now."""

    reason: str


@dataclass(frozen=True)
class AnchorStyle:
    """How a line tool's anchor is written and read (D172 the anchor-style plug-in)."""

    name: Literal["number", "hash"]

    @property
    def tagged(self) -> bool:
        return self.name == "hash"

    def anchor(self, n: int, line: str) -> str:
        """Line `n`'s anchor: `12`, or `12:a3f`."""
        return f"{n}:{tag(line)}" if self.tagged else str(n)

    def example(self, n: int) -> str:
        """An anchor for an example line, which names no real line."""
        return f"{n}:a3f" if self.tagged else str(n)

    def rows(self, lines: Sequence[str], spans: Sequence[Span] | None = None) -> list[str]:
        """`lines` as the style shows them: every line, or `spans` of them, numbered."""
        whole = spans if spans is not None else ((Span(1, len(lines)),) if lines else ())
        return number_lines(lines, whole, tagged=self.tagged)

    def render(self, text: str) -> str:
        """A file's whole text as the style shows it: what `read` answers and what the packer
        packs for a target, so the budget counts what is shown."""
        return "\n".join(self.rows(split_lines(text)))

    def argument_schema(self, kind: AnchorKind) -> dict[str, object]:
        """An anchor argument's schema: a mismatched anchor fails it (`INVALID_ARGUMENTS`),
        never reinterpreted. Under `hash` an `after` is also `0`, as an integer or as the text
        a `TOOL` line carries it in."""
        if not self.tagged:
            return {"type": "integer", "minimum": 0 if kind == "after" else 1}
        if kind == "line":
            return {"type": "string", "pattern": _TAGGED.pattern}
        start: dict[str, object] = {"type": "integer", "enum": [0]}
        return {"oneOf": [start, {"type": "string", "pattern": _AFTER.pattern}]}

    def resolve(self, value: int | str, lines: Sequence[str], kind: AnchorKind) -> int | Stale:
        """The 1-based line `value` names in `lines` (`0`, the file's start, for an `after`),
        or a `Stale` with the lines there now. The schema has checked its shape."""
        if kind == "after" and value in (0, "0"):
            return 0
        if isinstance(value, int):
            n, want = value, None
        else:
            m = _TAGGED.match(value)
            if m is None:
                return Stale(f"{value!r} is not an anchor of this style")
            n, want = int(m.group(1)), m.group(2)
        if not 1 <= n <= len(lines):
            near = self._near(lines, n)
            return Stale(f"there is no line {n}: the file has {len(lines)} lines{near}")
        if want is not None and tag(lines[n - 1]) != want:
            near = self._near(lines, n)
            return Stale(f"line {n} no longer reads as {n}:{want}; read the lines again{near}")
        return n

    def _near(self, lines: Sequence[str], n: int) -> str:
        """`; the lines there now:` and the rows around line `n`, in this style."""
        first = max(1, min(n, len(lines)) - _STALE_CONTEXT)
        last = min(len(lines), n + _STALE_CONTEXT)
        if first > last:
            return "; the file is empty"
        shown = number_lines(lines, (Span(first, last),), tagged=self.tagged)
        return "; the lines there now:\n" + "\n".join(shown)


NUMBER: Final = AnchorStyle("number")
HASH: Final = AnchorStyle("hash")

ANCHOR_STYLES: Final[Mapping[str, AnchorStyle]] = {s.name: s for s in (NUMBER, HASH)}
"""The anchor styles by name, `line_anchors`' values."""
