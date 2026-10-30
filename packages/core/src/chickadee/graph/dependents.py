"""Which file depends on which: the one home of the
file-to-file graph, read by `Graph.resolve` (import distance) and by the indexer (the files a
change makes stale).

A file depends on another when it imports it, or when one of its edges lands on a symbol the
other defines. Both are read from the index, never from the bytes. What the graph cannot see (a
dynamic import, a string-named call) is the one dependency missed, as `tests_for` says of the
same blind spot.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Final

from chickadee.graph.parse import module_name

__all__ = [
    "dependents",
    "file_graph",
    "import_graph",
    "python_relative_module",
    "relative_import_file",
]


def import_graph(conn: sqlite3.Connection, files: Collection[str]) -> dict[str, set[str]]:
    """Each of `files` to the files of `files` it imports: a relative path (`./util`) by its
    file, a module (`a.util`, `.util`) by every file whose module it is or is inside."""
    by_module: dict[str, list[str]] = {}
    for path in files:
        by_module.setdefault(module_name(path), []).append(path)
    graph: dict[str, set[str]] = {path: set() for path in files}
    for src, module in conn.execute("SELECT file, module FROM imports"):
        if src not in graph:
            continue
        rel = relative_import_file(src, module, files)
        if rel is not None:
            graph[src].add(rel)
            continue
        lookup = python_relative_module(src, module) or module
        # A module names a file when it is that file's module or lies inside it (`a.util.f`
        # names `a/util.py`): each dotted prefix of the name, looked up.
        parts = lookup.split(".")
        for end in range(len(parts), 0, -1):
            for path in by_module.get(".".join(parts[:end]), ()):
                if path != src:
                    graph[src].add(path)
    return graph


def file_graph(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Each indexed file to every indexed file it depends on: what it imports, and the files
    defining the symbols its edges land on."""
    files = {str(r[0]) for r in conn.execute("SELECT path FROM files")}
    graph = import_graph(conn, files)
    rows = conn.execute(
        "SELECT DISTINCT e.file, s.file FROM edges e "
        "JOIN symbols s ON s.qualified_name = e.dst WHERE e.file <> s.file"
    )
    for src, dst in rows:
        graph.setdefault(str(src), set()).add(str(dst))
    return graph


def dependents(conn: sqlite3.Connection, changed: Iterable[str]) -> dict[str, str]:
    """Every file that depends on one of `changed`, directly or through others, to the changed
    file it depends on most nearly: no depth limit, since a re-exported name breaks a file two
    hops away as surely as one. Ties go to the changed file first in path
    order, so the answer is the same on every run. The changed files are not in it."""
    origins = sorted(set(changed))
    if not origins:
        return {}
    depended_on: dict[str, set[str]] = {}
    for src, targets in file_graph(conn).items():
        for dst in targets:
            depended_on.setdefault(dst, set()).add(src)
    by: dict[str, str] = {origin: origin for origin in origins}
    queue: deque[str] = deque(origins)
    while queue:
        node = queue.popleft()
        for src in sorted(depended_on.get(node, ())):
            if src not in by:
                by[src] = by[node]
                queue.append(src)
    return {path: origin for path, origin in by.items() if path not in origins}


# -- resolving an import to a file ------------------------------------------------------------

# Longer suffixes first: `.ts` is a prefix of `.tsx`, `.js` of `.jsx`.
_IMPORT_EXTS: Final = (".tsx", ".ts", ".jsx", ".mjs", ".cjs", ".js")


def python_relative_module(src: str, module: str) -> str | None:
    """`from.foo import bar` is stored as `.foo.bar`; expand it against `src`'s package."""
    if not module.startswith(".") or module.startswith("./") or module.startswith("../"):
        return None
    dots = len(module) - len(module.lstrip("."))
    rest = module[dots:]
    parent = Path(src).parent.parts
    climb = dots - 1
    if climb >= len(parent):
        return None
    base = parent[: len(parent) - climb]
    if rest:
        return ".".join((*base, *rest.split("."))) if base else rest
    return ".".join(base) if base else None


def relative_import_file(src: str, module: str, files: Collection[str]) -> str | None:
    if not (module.startswith("./") or module.startswith("../")):
        return None
    spec = _import_file_spec(module)
    parent = Path(src).parent
    combined = (parent / spec).as_posix() if str(parent) != "." else spec
    stem = _posix_norm(combined)
    if stem == ".." or stem.startswith("../"):
        return None
    for candidate in _import_candidates(stem):
        if candidate in files:
            return candidate
    return None


def _import_file_spec(module: str) -> str:
    """The file portion of an extractor module (`./util.tsx.thing` → `./util.tsx`)."""
    spec = module.split("?", 1)[0].split("#", 1)[0]
    best = spec
    best_end = -1
    for ext in _IMPORT_EXTS:
        start = 0
        while True:
            idx = spec.find(ext, start)
            if idx < 0:
                break
            end = idx + len(ext)
            after = spec[end:]
            if (after == "" or after.startswith(".")) and end > best_end:
                best_end = end
                best = spec[:end]
            start = idx + 1
    return best


def _import_candidates(stem: str) -> tuple[str, ...]:
    if any(stem.endswith(ext) for ext in _IMPORT_EXTS):
        return (stem,)
    out = [stem]
    for ext in _IMPORT_EXTS:
        out.append(stem + ext)
        out.append(f"{stem}/index{ext}")
    return tuple(out)


def _posix_norm(path: str) -> str:
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            else:
                parts.append(part)
            continue
        parts.append(part)
    return "/".join(parts)
