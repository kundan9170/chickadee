"""A grammar's tags query run over a file: its definitions and references.

A tags query (`queries/tags.scm`, the format GitHub's code navigation and Aider's repository
map read) marks each definition's node `@definition.<kind>` and each reference's
`@reference.<kind>`, with the name in `@name`. One query per grammar stands in for a
hand-written walker for the languages that have none; `graph/parse` turns these tags into the
same `ParsedFile` its walkers return. Imports are not read: a reference resolves through the
name index alone, and `atlas` says so for a file read this way.

The query is the wheel's own, or one held in the harness's `profiles/queries/<grammar>.scm`,
which replaces it whole: Kotlin's and C#'s wheels ship none; C's, C++'s and Swift's mark no
references, which `tests_for` needs; and PHP's matches no unqualified call in its own grammar.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from tree_sitter import Node, Query, QueryCursor

from chickadee.graph.grammars import language, wheel_tags_query
from chickadee.profiles.loader import BUILTIN_DIR

__all__ = ["HELD_DIR", "Tag", "query_path", "tags"]

HELD_DIR: Final = BUILTIN_DIR / "queries"
"""The harness's own tags queries, by grammar name: each replaces its wheel's whole."""

Role = Literal["definition", "reference"]


@dataclass(frozen=True)
class Tag:
    """One definition or reference a tags query marked."""

    role: Role
    kind: str
    """The capture's last word: `function`, `class` or `method` for a definition; `call`,
    `class`, `type` or `implementation` for a reference."""
    name: str
    node: Node
    """The definition's or reference's whole node, as the query marked it."""
    name_node: Node


def query_path(grammar: str, held: Path = HELD_DIR) -> Path | None:
    """The tags query `grammar` is read by: the harness's own when it holds one, else its
    wheel's; None when there is neither."""
    own = held / f"{grammar}.scm"
    return own if own.is_file() else wheel_tags_query(grammar)


@functools.cache
def _query(grammar: str, path: Path) -> Query | None:
    """The query at `path`, compiled once per process. None for a grammar the harness lacks."""
    loaded = language(grammar)
    if loaded is None:
        return None
    return Query(loaded, path.read_text(encoding="utf-8"))


def tags(grammar: str, root: Node, source: bytes, held: Path = HELD_DIR) -> tuple[Tag, ...]:
    """Every definition and reference in the tree at `root`, in the text's order, each once.
    Empty when `grammar` has no tags query: its files are indexed with no symbols, which the
    index's coverage shows, never a crash."""
    path = query_path(grammar, held)
    query = _query(grammar, path) if path is not None else None
    if query is None:
        return ()
    found: dict[tuple[Role, int, int], Tag] = {}
    for _pattern, captures in QueryCursor(query).matches(root):
        tag = _tag(captures, source)
        if tag is not None:
            # Two patterns can mark one name (a method is also a function in some queries):
            # the first, in the query's order, is kept.
            key = (tag.role, tag.name_node.start_byte, tag.node.start_byte)
            found.setdefault(key, tag)
    return tuple(sorted(found.values(), key=lambda t: (t.node.start_byte, -t.node.end_byte)))


def _tag(captures: dict[str, list[Node]], source: bytes) -> Tag | None:
    names = captures.get("name")
    if not names:
        return None
    for capture, nodes in captures.items():
        role, _dot, kind = capture.partition(".")
        if role in ("definition", "reference") and kind and nodes:
            name_node = names[0]
            text = source[name_node.start_byte : name_node.end_byte].decode("utf-8", "replace")
            name = text.strip()
            if not name:
                return None
            return Tag(
                "definition" if role == "definition" else "reference",
                kind,
                name,
                nodes[0],
                name_node,
            )
    return None
