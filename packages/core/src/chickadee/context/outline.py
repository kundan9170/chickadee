"""Skeletons, per format: the cheap default a read returns.

A read with no region returns the file's skeleton and a handle, never the file. The skeleton
is whatever cheap structure the format affords:

| Format | Preview | `at=` selects |
|---|---|---|
| code | the symbol outline, from the graph — nearly free | a symbol name |
| Markdown | the heading tree | a heading |
| JSON / TOML | the key skeleton, depth-limited, values elided to their types | — (deferred) |
| CSV / TSV | the header, the delimiter, the first rows, the row count | — |
| plain text, logs, anything unknown | head + tail, with the hole marked | — |
| minified, generated | the shape only | — |
| an image | the shape: media type, dimensions, the token price of looking | — |
| any other binary | the shape only; beyond it, `NOT_TEXT` | — |

A format with no selectors says so in its `ops` rather than accepting `at=` and guessing at
a span. Nothing here reads a model or spawns a process; the graph supplies code symbols.
"""

from __future__ import annotations

import io
import json
import math
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Final, Literal, Protocol, cast

from chickadee.client.request import IMAGE_MEDIA_TYPES, PDF_MEDIA_TYPE
from chickadee.context.results import InvalidArgument
from chickadee.context.spans import Span, runs
from chickadee.edits.lines import ROW_CHARS_MAX, clip, elided, split_lines
from chickadee.edits.lines import row as line_row
from chickadee.limits import tokens_for_chars
from chickadee.profiles.loader import ProfileSet, shipped

__all__ = [
    "Entry",
    "Format",
    "Outline",
    "Sniffed",
    "SymbolLike",
    "code_language",
    "code_outline",
    "csv_outline",
    "downscale",
    "fit_rows",
    "head_tail",
    "image_size",
    "image_tokens",
    "markdown_outline",
    "media_type_of",
    "resolve_at",
    "rows_size",
    "runs",
    "scaled_size",
    "signature_spans",
    "skeleton_outline",
    "sniff",
    "stored_sniff",
    "whole_file",
]

Format = Literal[
    "code", "markdown", "json", "toml", "yaml", "csv", "tsv", "text", "minified", "generated"
]


_TEXT_FORMATS: Final[Mapping[str, tuple[Format, str]]] = {
    ".md": ("markdown", "markdown"),
    ".markdown": ("markdown", "markdown"),
    ".json": ("json", "json"),
    ".toml": ("toml", "toml"),
    ".yml": ("yaml", "yaml"),
    ".yaml": ("yaml", "yaml"),
    ".csv": ("csv", "csv"),
    ".tsv": ("tsv", "tsv"),
}

_SNIFF_BYTES: Final = 8192
_MINIFIED_LINE: Final = 1000
_PREVIEW_LINE_MAX: Final = ROW_CHARS_MAX
"""A preview row longer than this is cut with its remainder counted — orientation, not
content. A view (`lines=`) is never cut this way; only its byte cap applies."""
_SKELETON_DEPTH: Final = 2
_CSV_ROWS: Final = 5
IMAGE_TOKEN_DIVISOR: Final = 750
"""The vendor's formula, `w * h / 750` (Anthropic's), on the DOWNSCALED size."""


# -- sniffing ---------------------------------------------------------------------------

_MAGIC: Final[tuple[tuple[bytes, str], ...]] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"\x00\x00\x01\x00", "image/x-icon"),
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),
    (b"\x1f\x8b", "application/gzip"),
    (b"BZh", "application/x-bzip2"),
    (b"\xfd7zXZ\x00", "application/x-xz"),
    (b"7z\xbc\xaf\x27\x1c", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "application/vnd.rar"),
    (b"\x7fELF", "application/x-executable"),
    (b"MZ", "application/x-msdownload"),
    (b"\xcf\xfa\xed\xfe", "application/x-mach-binary"),
    (b"\xfe\xed\xfa\xcf", "application/x-mach-binary"),
    (b"\xca\xfe\xba\xbe", "application/x-mach-binary"),
    (b"\x00asm", "application/wasm"),
    (b"wOFF", "font/woff"),
    (b"wOF2", "font/woff2"),
    (b"OTTO", "font/otf"),
    (b"\x00\x01\x00\x00", "font/ttf"),
    (b"ID3", "audio/mpeg"),
    (b"OggS", "audio/ogg"),
    (b"fLaC", "audio/flac"),
    (b"SQLite format 3\x00", "application/vnd.sqlite3"),
)
"""Consulted only for content that is already known not to be text, so a short printable
magic (`BM`, `MZ`) never turns a text file that happens to start with it into a binary."""


@dataclass(frozen=True)
class Sniffed:
    """What a file is, decided from its bytes and its name — never a parse guess."""

    category: Literal["text", "image", "document", "binary"]
    kind: str
    """`Shape.kind`: a language or format name, or a media type."""
    format: Format | None
    """None for an image, a document or a binary."""
    lines: tuple[str, ...] = ()
    """The text's lines, without their terminators; empty for anything not text."""
    size: tuple[int, int] | None = None
    """An image's `(width, height)`, from its header. An image whose header does not decode
    is sniffed as a binary: there are no pixels to promise."""


def media_type_of(data: bytes) -> str:
    """The media type the bytes' magic names, or `application/octet-stream`."""
    if data[:4] == b"RIFF" and len(data) >= 12:
        riff = {b"WEBP": "image/webp", b"WAVE": "audio/wav", b"AVI ": "video/x-msvideo"}
        return riff.get(data[8:12], "application/octet-stream")
    if data[257:262] == b"ustar":
        return "application/x-tar"
    for magic, media in _MAGIC:
        if data.startswith(magic):
            return media
    return "application/octet-stream"


def _decode(data: bytes) -> str | None:
    """The text, or None when the bytes are not text. A NUL in the first 8 KiB is binary; a
    stray invalid byte in a log is still a log, so a few replacements are tolerated."""
    if b"\x00" in data[:_SNIFF_BYTES]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
        if text.count("�") > max(1, len(data) // 100):
            return None
        return text


def code_language(path: str, languages: ProfileSet | None = None) -> str | None:
    """The language a preview labels `path` with, or None when it is not code: `languages`'
    answer, the run's set with the repository's overrides, else the shipped
    one. A language is read as code once it has a profile. Whether a code file has an OUTLINE
    is the graph's answer, not this: an unindexed `.py` is previewed as text and says
    `no index`. The extension is matched as written, as the index matches it: `X.PY` is not
    labelled code the index would call a gap."""
    found = (languages or shipped()).language_of(PurePosixPath(path).name)
    return found.name if found is not None else None


def _text_format(
    path: str, lines: Sequence[str], languages: ProfileSet | None
) -> tuple[Format, str]:
    name = PurePosixPath(path).name.lower()
    suffix = PurePosixPath(name).suffix
    language = code_language(path, languages)
    if name.endswith((".min.js", ".min.css")) or _looks_minified(lines):
        return "minified", language or "plain text"
    head = "\n".join(lines[:5]).lower()
    if "@generated" in head or "do not edit" in head:
        return "generated", language or _TEXT_FORMATS.get(suffix, ("text", "plain text"))[1]
    if language is not None:
        return "code", language
    return _TEXT_FORMATS.get(suffix, ("text", "plain text"))


def _looks_minified(lines: Sequence[str]) -> bool:
    if not lines:
        return False
    longest = max(len(line) for line in lines)
    mean = sum(len(line) for line in lines) / len(lines)
    return longest > _MINIFIED_LINE and (len(lines) <= 3 or mean > 300)


def sniff(path: str, data: bytes, languages: ProfileSet | None = None) -> Sniffed:
    """What `data` is, named `path`: `languages` says which names are code (`code_language`)."""
    if media_type_of(data) == PDF_MEDIA_TYPE:
        # A PDF by its magic, before the text test, since a PDF written without
        # binary streams decodes as text and is still never read as one. Whether it opens is
        # `Parts.of`'s to find out.
        return Sniffed("document", PDF_MEDIA_TYPE, None)
    text = _decode(data)
    if text is None:
        media = media_type_of(data)
        size = image_size(data) if media in IMAGE_MEDIA_TYPES else None
        if size is not None:
            return Sniffed("image", media, None, size=size)
        return Sniffed("binary", media, None)
    lines = split_lines(text)
    fmt, kind = _text_format(path, lines, languages)
    return Sniffed("text", kind, fmt, lines)


def stored_sniff(data: bytes) -> Sniffed:
    """Stored output is read as plain text whatever its first lines say: a build log that
    prints `@generated` is still a log."""
    sniffed = sniff("", data)
    if sniffed.category != "text":
        return sniffed
    return replace(sniffed, format="text", kind="plain text")


# -- images ----------------------------------------------------------------------


def image_tokens(width: int, height: int) -> int:
    return max(1, math.ceil(width * height / IMAGE_TOKEN_DIVISOR))


def image_est(data: bytes, known: int | None = None) -> int:
    """What one image costs a model, in tokens: the estimate its handle already carries
    when there is one, else its decoded size. The one rule the request's window estimate and a
    role's result budget both count by. Undecodable pixels are counted as bytes: an
    overestimate, the direction that cuts."""
    if known is not None:
        return known
    size = image_size(data)
    return image_tokens(*size) if size is not None else tokens_for_chars(len(data))


def scaled_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    """The size `downscale` sends: the longest edge at most `max_edge`, the aspect kept. It
    prices `depth="full"` before anything is decoded, so it may differ from the
    resampler's own rounding by a pixel — an estimate, as `est_tokens` says."""
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    scale = max_edge / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def image_size(data: bytes) -> tuple[int, int] | None:
    """`(width, height)` from the image's header, or None when it cannot be decoded."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def downscale(
    data: bytes, media_type: str, max_edge: int
) -> tuple[bytes, int, int, int, int] | None:
    """`(bytes, width, height, original_width, original_height)` with the longest edge at
    most `max_edge` — the original bytes when they already fit, so an image that needs no
    work is sent as it was. None when the pixels cannot be decoded: the shape is then all
    there is, and the caller says so rather than guessing."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.width, img.height
            if max(width, height) <= max_edge:
                return data, width, height, width, height
            small = img.copy()
            small.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            fmt = _PIL_FORMATS[media_type]
            if fmt == "JPEG" and small.mode not in ("RGB", "L"):
                small = small.convert("RGB")
            small.save(out, format=fmt)
            return out.getvalue(), small.width, small.height, width, height
    except (UnidentifiedImageError, OSError, ValueError, KeyError):
        return None


_PIL_FORMATS: Final[Mapping[str, str]] = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}


# -- rendering lines --------------------------------------------------------------------


def _preview_line(n: int, width: int, text: str) -> str:
    return line_row(n, width, clip(text, _PREVIEW_LINE_MAX))


def head_tail(lines: Sequence[str], head: int, tail: int, budget: int) -> Outline:
    """Head and tail, numbered, the hole between them marked — shrunk together until they
    fit `budget` bytes. For a structureless file this is orientation; `grep_in` is the
    access path. The lines it shows whole are a view (`Outline.shown`)."""
    total = len(lines)
    width = len(str(total))
    h, t = min(head, total), min(tail, max(0, total - head))
    while True:
        rows = [_preview_line(n, width, lines[n - 1]) for n in range(1, h + 1)]
        if h + t < total:
            rows.append(elided(h + 1, total - t))
        rows.extend(
            _preview_line(n, width, lines[n - 1]) for n in range(total - t + 1, total + 1)
        )
        if rows_size(rows) <= budget or h + t == 0:
            shown = (*range(1, h + 1), *range(total - t + 1, total + 1))
            return Outline(tuple(rows), shown=_whole(lines, shown))
        h, t = h // 2, t // 2


def whole_file(lines: Sequence[str], budget: int) -> Outline | None:
    """Every line, numbered as `head_tail` numbers them, when all of them fit `budget` bytes
    uncut; else None. A small code file is cheaper read whole than outlined and then opened
    (the owner, 2026-10-20): its preview is the file, and says so (`parts.WHOLE_FILE`)."""
    total = len(lines)
    if any(len(line) > _PREVIEW_LINE_MAX for line in lines):
        return None
    width = len(str(total))
    rows = [_preview_line(n, width, lines[n - 1]) for n in range(1, total + 1)]
    if rows_size(rows) > budget:
        return None
    return Outline(tuple(rows), shown=(Span(1, total),) if total else ())


def _whole(lines: Sequence[str], numbers: Iterable[int]) -> tuple[Span, ...]:
    """The numbered lines a preview shows uncut."""
    return runs(n for n in numbers if len(lines[n - 1]) <= _PREVIEW_LINE_MAX)


def fit_rows(rows: Sequence[str], budget: int, *, what: str = "entries") -> list[str]:
    """Rows up to `budget` bytes, then one line saying how many were not shown."""
    out: list[str] = []
    used = 0
    for i, row in enumerate(rows):
        cost = len(row.encode("utf-8")) + 1
        if used + cost > budget:
            out.append(f"[outline continues: {len(rows) - i:,} more {what} not shown]")
            return out
        out.append(row)
        used += cost
    return out


def rows_size(rows: Iterable[str]) -> int:
    """Rows as rendered: their bytes and a newline each."""
    return sum(len(r.encode("utf-8")) + 1 for r in rows)


# -- outlines ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One thing `at=` may name, and the span it resolves to."""

    name: str
    span: Span
    qualified: str | None = None


@dataclass(frozen=True)
class Outline:
    rows: tuple[str, ...]
    entries: tuple[Entry, ...] = ()
    """What `at=` may name, one per row. Empty when the format affords no selectors."""
    selectable: bool = False
    note: str | None = None
    shown: tuple[Span, ...] = ()
    """File lines the rows show whole, when the preview is itself a view: a structureless
    file's head and tail ARE `read(lines=1-20)` and the last twenty, already merged.
    A row cut at the preview width is not among them."""

    def __post_init__(self) -> None:
        if self.selectable and len(self.rows) != len(self.entries):
            raise ValueError("a selectable outline has one entry per row")

    def within(self, region: Span) -> Outline:
        """The entries that start inside `region`, with their rows: `depth="outline"` on a
        named region. Only a selectable outline has entries to narrow."""
        keep = [
            i for i, e in enumerate(self.entries) if region.start <= e.span.start <= region.end
        ]
        return Outline(
            tuple(self.rows[i] for i in keep),
            tuple(self.entries[i] for i in keep),
            selectable=self.selectable,
        )


class SymbolLike(Protocol):
    """`graph.queries.SymbolInfo`, structurally — the fields an outline renders."""

    @property
    def name(self) -> str: ...
    @property
    def qualified_name(self) -> str: ...
    @property
    def kind(self) -> str: ...
    @property
    def line_start(self) -> int: ...
    @property
    def line_end(self) -> int: ...
    @property
    def sig(self) -> str | None: ...
    @property
    def doc_first_line(self) -> str | None: ...


def _depths(symbols: Sequence[SymbolLike]) -> list[int]:
    """How deeply each symbol nests, by span containment: a method inside its class."""
    depths: list[int] = []
    open_spans: list[tuple[int, int]] = []
    for s in symbols:
        while open_spans and open_spans[-1][1] < s.line_start:
            open_spans.pop()
        depths.append(len(open_spans))
        open_spans.append((s.line_start, s.line_end))
    return depths


def _doc(doc: str | None) -> str:
    if not doc:
        return ""
    short = doc if len(doc) <= 60 else doc[:59] + "…"
    return f'  "{short}"'


def code_outline(symbols: Sequence[SymbolLike], total_lines: int) -> Outline:
    """The symbol outline, one row per symbol, nested by span. A symbol the index placed
    before line 1 has no span to name and is left out of both."""
    ordered = sorted(
        (s for s in symbols if s.line_start >= 1), key=lambda s: (s.line_start, -s.line_end)
    )
    width = len(str(max(total_lines, 1)))
    rows: list[str] = []
    for s, depth in zip(ordered, _depths(ordered), strict=True):
        label = s.sig or f"{s.kind} {s.name}"
        rows.append(f"{s.line_start:>{width}}  {'  ' * depth}{label}{_doc(s.doc_first_line)}")
    entries = tuple(
        Entry(s.name, Span(s.line_start, max(s.line_start, s.line_end)), s.qualified_name)
        for s in ordered
    )
    return Outline(tuple(rows), entries, selectable=True)


def signature_spans(
    symbols: Sequence[SymbolLike], region: Span, total_lines: int
) -> tuple[Span, ...]:
    """`depth="signatures"`: each symbol's first two lines inside `region` — the signature
    and the first line of its body or docstring. The caller renders the lines
    between them as elided, so every elided body is marked."""
    keep: set[int] = set()
    for s in symbols:
        if s.line_start < region.start or s.line_start > region.end:
            continue
        keep.add(s.line_start)
        if s.line_start + 1 <= min(s.line_end, region.end):
            keep.add(s.line_start + 1)
    return runs(sorted(n for n in keep if n <= total_lines))


def markdown_outline(lines: Sequence[str]) -> Outline:
    """The heading tree; a heading's span runs to the next heading at its level or above.
    Lines inside fenced code are not headings."""
    heads: list[tuple[int, int, str]] = []
    fence: str | None = None
    for n, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = None if fence == marker else (fence or marker)
            continue
        if fence is not None or not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        rest = stripped[level:]
        if 1 <= level <= 6 and (rest.startswith(" ") or rest == ""):
            heads.append((n, level, rest.strip().rstrip("#").strip()))
    width = len(str(max(len(lines), 1)))
    rows: list[str] = []
    entries: list[Entry] = []
    for i, (n, level, title) in enumerate(heads):
        end = len(lines)
        for m, other_level, _ in heads[i + 1 :]:
            if other_level <= level:
                end = m - 1
                break
        rows.append(f"{n:>{width}}  {'  ' * (level - 1)}{'#' * level} {title}")
        entries.append(Entry(title, Span(n, max(n, end))))
    return Outline(tuple(rows), tuple(entries), selectable=True)


def skeleton_outline(text: str, fmt: Literal["json", "toml"]) -> Outline | None:
    """Key paths with their types, values elided, depth-limited. None when the text does
    not parse — the caller previews it as text and says why. `at=` key paths are not built
    yet, and the outline says so rather than guessing at a span."""
    try:
        data: object = json.loads(text) if fmt == "json" else tomllib.loads(text)
    except (ValueError, tomllib.TOMLDecodeError):
        return None
    rows: list[str] = []
    _walk(data, "", 0, rows)
    return Outline(tuple(rows), note="key paths are not selectable yet; read(lines=…)")


def _type_of(value: object) -> str:
    if isinstance(value, dict):
        return f"object · {len(cast('dict[str, object]', value)):,} keys"
    if isinstance(value, list):
        return f"array · {len(cast('list[object]', value)):,} items"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _walk(value: object, path: str, depth: int, rows: list[str]) -> None:
    if path:
        rows.append(f"  {path}  {_type_of(value)}")
    if depth > _SKELETON_DEPTH:
        return
    if isinstance(value, dict):
        for key, child in cast("dict[str, object]", value).items():
            _walk(child, f"{path}.{key}" if path else key, depth + 1, rows)
    elif isinstance(value, list):
        items = cast("list[object]", value)
        # `type(...) in` rather than isinstance: parsed JSON and TOML hold exactly these two
        # containers, and the element stays typed as the `object` it is.
        if items and type(items[0]) in (dict, list):
            _walk(items[0], f"{path}[0]" if path else "[0]", depth + 1, rows)
    elif not path:
        rows.append(f"  (root)  {_type_of(value)}")


def csv_outline(lines: Sequence[str], delimiter: str) -> Outline:
    """The header and the first rows, numbered; the rest counted, not shown. Rows are
    counted by line, so a quoted newline inside a field is counted twice — said, not hidden."""
    shown = min(len(lines), 1 + _CSV_ROWS)
    width = len(str(max(len(lines), 1)))
    rows = [_preview_line(n, width, lines[n - 1]) for n in range(1, shown + 1)]
    if shown < len(lines):
        rows.append(elided(shown + 1, len(lines)))
    shown_delim = "tab" if delimiter == "\t" else f'"{delimiter}"'
    note = f"{max(len(lines) - 1, 0):,} data rows by line count · delimiter {shown_delim}"
    return Outline(tuple(rows), note=note, shown=_whole(lines, range(1, shown + 1)))


# -- at= --------------------------------------------------------------------------------


def resolve_at(outline: Outline, at: str) -> Span | InvalidArgument:
    """An outline entry by name: exact, then qualified-suffix, then case-insensitive. More
    than one candidate is refused with the candidates named — never a silent pick."""
    if not outline.selectable:
        return InvalidArgument(
            "at",
            "this format has no outline entries to name",
            do='read it with lines="a-b" instead',
        )
    exact = [e for e in outline.entries if e.name == at or e.qualified == at]
    if not exact:
        exact = [e for e in outline.entries if e.qualified and e.qualified.endswith("." + at)]
    if not exact:
        folded = at.casefold()
        exact = [e for e in outline.entries if e.name.casefold() == folded]
    if len(exact) == 1:
        return exact[0].span
    if not exact:
        return InvalidArgument(
            "at",
            f"no outline entry is named {at!r}",
            do="read the file without at= to see its outline, then name an entry it lists",
        )
    where = ", ".join(f"{e.qualified or e.name} at {e.span.start}" for e in exact[:5])
    return InvalidArgument(
        "at",
        f"{at!r} names {len(exact)} entries ({where})",
        do="send at= again with one of the qualified names above",
    )
