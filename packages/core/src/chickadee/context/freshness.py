"""Freshness: a handle is stale when its bytes moved.

The token is a **per-file content hash** — never mtime, never a global generation. A commit to
`applier.py` must not invalidate a handle on `graph.py`, or every episode would invalidate the
whole context and the economy would collapse to re-reading.

Three things were once blurred into one:

- `is_stale()` — an internal predicate, computed on demand. Zero tokens.
- `dereference()` — the lazy check at a point of use: the current content, marked changed.
  Only what was asked for.
- `sweep()` — a notice when a view still expanded in context goes stale. One line, on the
  transition only.

The lazy check catches what the harness did not cause (an editor, a `git stash`); the sweep
catches what the lazy check structurally cannot — Zeus reasoning from view text already on
screen, with no call to check at.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from chickadee.context import standing
from chickadee.context.handle import Handle, content_sha
from chickadee.context.resolver import HandleResolver

if TYPE_CHECKING:
    from chickadee.context.session import Session

__all__ = ["CHANGED_NOTE", "Dereference", "dereference", "is_stale", "stale_check", "sweep"]

CHANGED_NOTE: Final = "changed since you last saw it"
"""What a stale dereference says beside the current content. Never an error: a stale
handle reports staleness and offers the current span, because the model usually wants what
the file says now."""


def is_stale(handle: Handle, resolver: HandleResolver) -> bool:
    """`sha256(current bytes) != handle.content_sha`. Computed, never stored: a stored
    `stale` flag can itself go stale.

    Only a file can be stale. Everything else is content-addressed — a blob named by its
    hash, an append-only journal — so its bytes cannot move. A file that is gone is stale:
    what the handle says it holds is no longer on disk."""
    if handle.kind != "file":
        return False
    data = resolver.disk_bytes(handle)
    return data is None or content_sha(data) != handle.content_sha


def stale_check(
    handles: Callable[[str], Handle | None], resolver: HandleResolver
) -> Callable[[str], bool]:
    """`is_stale` over handle ids, for `recovery.actual_recovery`. An id the session never
    minted cannot be rehydrated, so it answers as stale."""

    def check(handle_id: str) -> bool:
        handle = handles(handle_id)
        return handle is None or is_stale(handle, resolver)

    return check


@dataclass(frozen=True)
class Dereference:
    """The lazy check at a point of use."""

    stale: bool
    current: bytes | None
    """The file's bytes now; None when the file is gone. Unchecked against the deny list
    (`HandleResolver.disk_bytes`): what a model is shown comes through the Reader."""


def dereference(handle: Handle, resolver: HandleResolver) -> Dereference:
    data = resolver.disk_bytes(handle)
    return Dereference(
        stale=data is None or content_sha(data) != handle.content_sha, current=data
    )


def sweep(session: Session, resolver: HandleResolver) -> tuple[int, ...]:
    """EVERY TURN, before assembly: one `notice(kind=3)` per view still expanded in context
    that went stale and was not already noticed. Zero tokens unless something moved.

    Expanded views only: one already collapsed to a receipt cannot be reasoned from, so the
    lazy check is all it needs. The sweep compares hashes and does not care who moved the
    bytes — a run and an external editor are noticed alike. Returns the notice lines' seqs."""
    recorded: list[int] = []
    with session.lock:
        for view in session.state.expanded_views():
            if view.handle_id in session.state.noticed_stale:
                continue
            handle = session.state.handles.get(view.handle_id)
            if handle is None or not is_stale(handle, resolver):
                continue
            recorded.append(standing.notice(session, standing.view_stale(handle, view.spans)))
    return tuple(recorded)
