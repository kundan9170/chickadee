"""The handle contract: an address, not a copy (invariant 14).

Every read returns a **handle** — the address of the rest — and content only for a span the
caller named and paid for. A file handle is roughly `(path, content_sha)` and stores nothing,
because the bytes already exist on disk. Command output is the other resolution: it existed
nowhere else, so a blob holds it and its handle names the blob. One type, two resolutions.
Resolving a handle back to bytes is `resolver.py`; what a tool returns around
it is `results.py`.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import Field

from chickadee.context.spans import SpanWire
from chickadee.schema import sha256_hex

__all__ = [
    "HANDLE_ID_RE",
    "STORED_KINDS",
    "Handle",
    "HandleKind",
    "HandleTable",
    "Shape",
    "content_sha",
    "handle_id",
]


HandleKind = Literal["file", "diff", "log", "journal", "findings", "output", "attachment"]
STORED_KINDS: Final[frozenset[str]] = frozenset(
    {"diff", "log", "findings", "output", "attachment"}
)
"""The kinds whose bytes existed nowhere else, so a blob holds them. An `attachment` is
a file the person sent with a turn: its bytes came over the wire, not from a path."""

HANDLE_ID_RE: Final = re.compile(r"^h_[0-9a-f]{4,64}$")


def content_sha(data: bytes) -> str:
    """The freshness token: a per-file content hash — never mtime, never a generation."""
    return sha256_hex(data)


# -- shapes -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """Enough to decide whether to fetch, without fetching."""

    bytes: int
    lines: Annotated[int | None, Field(default=None)]
    """None for binary."""
    symbols: Annotated[int | None, Field(default=None)]
    """None when not code, or not indexed."""
    kind: str
    """A language name or a media type."""
    width: int | None = None
    """D110, images only: the ORIGINAL's dimensions."""
    height: int | None = None
    est_tokens: int | None = None
    """D110, images only: the vendor's formula on the DOWNSCALED size, so the price of
    `depth="full"` is known before it is paid. A PDF's too: its pages and text, as a
    model that reads PDFs takes it."""
    pages: Annotated[int | None, Field(default=None, exclude_if=lambda v: v is None)] = None
    """A PDF's page count; None for anything else."""

    @property
    def is_image(self) -> bool:
        return self.kind.startswith("image/")

    @property
    def is_pdf(self) -> bool:
        return self.kind == "application/pdf"


# -- the handle -------------------------------------------------------------------------


@dataclass(frozen=True)
class Handle:
    """Stable within the session; rehydratable.

    The first six fields are the spec's. The next three say where the bytes are, which the
    spec leaves to "resolves via": a file handle names its `path`; a stored handle names the
    run whose blob store holds it, or None for the session's own (a checkpoint's output
    lives in its run's store); a journal handle names its run and, when it is narrower than
    the whole journal, its seq range. `name` is a stored file's own name, when it had one.
    """

    id: str
    kind: HandleKind
    shape: Shape
    preview: str
    """Bounded; the smallest useful thing — for a file, its outline."""
    ops: Annotated[tuple[str, ...], Field(default=())]
    """What may be asked of it: read(lines=…), read(at=…), read(depth=…), grep_in."""
    content_sha: str
    """The freshness token — and, for anything stored, the blob's name."""
    path: str | None = None
    run_id: str | None = None
    seqs: SpanWire | None = None
    name: Annotated[str | None, Field(default=None, exclude_if=lambda v: v is None)] = None
    """An attachment's file name as the person gave it: what a header names it by, and
    what its text is outlined by when it is read again, so `parse.py` stays Python. Not part
    of the address: the same bytes under two names are one handle, named by the first. Left
    out when None, so every other handle's line keeps its bytes."""

    def __post_init__(self) -> None:
        if self.kind == "file" and self.path is None:
            raise ValueError("a file handle names its path")
        if self.kind == "journal" and self.run_id is None:
            raise ValueError("a journal handle names its run")

    @property
    def stored(self) -> bool:
        """Whether a blob holds the bytes. Output always; an image read from a file too — the
        blob holds the original, since the next run of whatever took a screenshot
        overwrites it."""
        return self.kind in STORED_KINDS or (self.kind == "file" and self.shape.is_image)

    @property
    def label(self) -> str:
        """How a header names it: the path for a file, an attachment's name, else the kind."""
        if self.path is not None:
            return self.path
        return self.name if self.name is not None else self.kind

    @property
    def address(self) -> str:
        """What it points at (`_address`): two handles with one address are one handle."""
        return _address(self.kind, self.path, self.run_id, self.content_sha)


def _address(kind: str, path: str | None, run_id: str | None, sha: str) -> str:
    """What a handle points at, as one string: a file is its path at a content hash;
    anything stored is its blob, in whichever store holds it."""
    return f"{kind}\0{path if path is not None else (run_id or '')}\0{sha}"


def handle_id(
    kind: HandleKind, *, path: str | None, run_id: str | None, sha: str, width: int = 4
) -> str:
    """The id an address derives: `h_` and the first `width` hex digits of its digest.

    Derived rather than counted, so the same file at the same content is the same handle
    every time it is read, and a component that mints outside a session (a checkpoint's
    output handle) arrives at the id the session would have given it."""
    digest = hashlib.sha256(_address(kind, path, run_id, sha).encode("utf-8")).hexdigest()
    return f"h_{digest[:width]}"


class HandleTable:
    """The session's handles, by id. A handle lives as long as its session (`HANDLE_TTL` is
    none): nothing here expires on a clock.

    An id is lengthened only when a different address already holds its prefix. A replayed
    session reproduces every id because each result line records the handle it minted.

    Minting takes a lock: family B reads may be dispatched in parallel, and two reads
    whose addresses share a short prefix must not both be handed it.
    """

    MIN_ID_HEX: Final = 4

    def __init__(self) -> None:
        self._by_id: dict[str, Handle] = {}
        self._by_address: dict[str, str] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Handle]:
        return iter(tuple(self._by_id.values()))

    def __contains__(self, handle_id: object) -> bool:
        return handle_id in self._by_id

    def get(self, handle_id: str) -> Handle | None:
        return self._by_id.get(handle_id)

    def find(
        self, kind: HandleKind, *, path: str | None, run_id: str | None, sha: str
    ) -> Handle | None:
        found = self._by_address.get(_address(kind, path, run_id, sha))
        return self._by_id[found] if found is not None else None

    def mint(
        self,
        kind: HandleKind,
        *,
        path: str | None,
        run_id: str | None,
        sha: str,
        make: Callable[[str], Handle],
    ) -> Handle:
        """The handle for this address: the one already here, or `make(id)`, reserved in
        the table before the lock is released."""
        key = _address(kind, path, run_id, sha)
        with self._lock:
            held = self._by_address.get(key)
            if held is not None:
                return self._by_id[held]
            width = self.MIN_ID_HEX
            candidate = handle_id(kind, path=path, run_id=run_id, sha=sha, width=width)
            while candidate in self._by_id:
                width += 2
                candidate = handle_id(kind, path=path, run_id=run_id, sha=sha, width=width)
            handle = make(candidate)
            if handle.id != candidate:
                raise ValueError(f"make() returned {handle.id}, not the reserved {candidate}")
            self._by_id[candidate] = handle
            self._by_address[key] = candidate
            return handle

    def add(self, handle: Handle) -> None:
        """Record a handle restored from the line that minted it. The same handle twice is
        one handle; a different one under a held id is corruption, never an overwrite."""
        key = handle.address
        with self._lock:
            held = self._by_id.get(handle.id)
            if held is not None and held.address != key:
                # The fold refuses such a line before it applies one (`SessionState.check`), so
                # this is a backstop: a caller that reached it has bypassed the check. Imported
                # here because `lines` types its handles with this module.
                from chickadee.context.lines import PayloadError

                raise PayloadError(f"handle {handle.id} already names a different address")
            self._by_id[handle.id] = handle
            self._by_address.setdefault(key, handle.id)
