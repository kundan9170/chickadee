"""Recovery class: can a collapsed exchange be had back exactly?.

A tool declares a **ceiling**; staleness degrades it per result. Recovery is a property of
*this result at this moment*, never of the tool — writing "rehydrate exact" on a handle whose
file has moved is the bug the class exists to prevent, and step 4 below is what stops it.

There is no third class. An earlier draft carried `PINNED` for results that are a pure
function of a pinned snapshot; nothing qualifies, because the graph keeps no old generations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

__all__ = ["TOOL_REGISTRY", "Recovery", "actual_recovery", "ceiling"]


class Recovery(StrEnum):
    EXACT = "exact"
    """Re-reading the handle returns byte-identical content."""
    NONE = "none"
    """Re-calling returns a DIFFERENT answer, or nothing."""


TOOL_REGISTRY: Final[Mapping[str, Recovery]] = MappingProxyType(
    {
        # Handle-backed: the bytes are addressable and immutable at a content hash — an
        # image included, whose original the session's blob store holds.
        "read": Recovery.EXACT,
        "slice": Recovery.EXACT,
        "outline": Recovery.EXACT,
        "grep_in": Recovery.EXACT,
        # Append-only and seq-addressed, or blob-backed.
        "journal": Recovery.EXACT,
        "diff": Recovery.EXACT,
        "findings": Recovery.EXACT,
        # The graph is a live index: reindexing overwrites, so re-running returns today's
        # answer to yesterday's question. `neighbors` is a graph query too, though
        # the table names only four of the family.
        "atlas": Recovery.NONE,
        # `grep` and `glob` read the tree as it is now, which a later write changes.
        "grep": Recovery.NONE,
        "glob": Recovery.NONE,
        "symbol": Recovery.NONE,
        "neighbors": Recovery.NONE,
        "impact": Recovery.NONE,
        "recall": Recovery.NONE,
        # A bounded search whose context is discarded by design.
        "locate": Recovery.NONE,
        # A point-in-time observation of mutable state. `shell` keeps its output in a blob,
        # and the receipt still names that handle; what cannot be had back is the command's
        # answer NOW, which is what a re-call would be asked.
        "exec": Recovery.NONE,
        "shell": Recovery.NONE,
        "mcp": Recovery.NONE,
    }
)
"""The table. A tool it does not name — an MCP tool, the `tools` registry, `coder`,
`ask_user`, the record family — resolves to `NONE`: a ceiling nobody declared is not
exact."""


def ceiling(tool: str) -> Recovery:
    return TOOL_REGISTRY.get(tool, Recovery.NONE)


def actual_recovery(
    tool: str, handle: str | None, *, is_stale: Callable[[str], bool]
) -> Recovery:
    """§5.6.2, in its order: the ceiling, then the handle, then staleness.

    `is_stale` is asked about the handle id and computed on demand: a stored flag
    can itself go stale."""
    if ceiling(tool) is Recovery.NONE:
        return Recovery.NONE
    if handle is None:
        return Recovery.NONE
    if is_stale(handle):
        return Recovery.NONE
    return Recovery.EXACT
