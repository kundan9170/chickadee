"""`GraphQuery` implementation.

Every method is a pure read and safe to call concurrently — WAL gives readers a consistent
snapshot even mid-reindex. **Every answer that can be incomplete carries its own
incompleteness**: edges carry `edge_class`, resolution returns all candidates rather than
picking one, and nothing here returns a silently partial result.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tomllib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from chickadee.graph import diagnostics, fts, radius, store
from chickadee.graph.dependents import import_graph
from chickadee.graph.diagnostics import DiagnosticSet
from chickadee.graph.indexer import git_recency
from chickadee.paths import within
from chickadee.profiles.loader import ProfileSet

# Contract 8's `Pointer`, re-exported rather than redefined. This module had its own
# `(file, line, qualified_name)` — a lossy subset of the wire type, and the loss was the end
# line, which `where_defined` already held in the `SymbolInfo` it was reading from and threw
# away. One concept in two shapes, with the seam falling exactly on SC-1's validation gate,
# where a conversion written under one name for both is how `line` quietly becomes
# `line_start` and `line_end` gets filled in with a guess.
from chickadee.schema.findings import Pointer as Pointer


@dataclass(frozen=True)
class SymbolInfo:
    name: str
    qualified_name: str
    file: str
    kind: str
    line_start: int
    line_end: int
    sig: str | None
    doc_first_line: str | None
    exported: bool = False


# `Pointer` is contract 8's, imported from `schema/findings.py` — see the note by the import.


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: str
    edge_class: str
    file: str
    line: int
    stale: bool = False


@dataclass(frozen=True)
class ImportInfo:
    """One import, as a *query* answers it — so `file` is explicit.

    Named `ImportInfo` rather than `Import` because `graph/parse.py` already has an `Import`
    and the two are not the same shape: the parse-layer one is produced while walking a single
    file, so its file is implicit and it has no such field. Under one name, code moved between
    the two layers loses `.file` silently, and `verifier/gates_a.py` and `packer/types.py`
    already imported the two different types under the same spelling. The suffix is this
    module's own convention — `SymbolInfo`, `GlobalInfo` — and `Import` was the one type that
    did not follow it.
    """

    file: str
    module: str
    alias: str | None
    line: int


@dataclass(frozen=True)
class GlobalInfo:
    name: str
    qualified_name: str
    file: str
    line: int
    kind: str
    type_text: str | None


@dataclass(frozen=True)
class DependencyContract:
    source: str
    dependency: str
    symbols: tuple[str, ...]
    declared_version: str | None
    edge_class: str


_SYMBOL_COLS = (
    "name, qualified_name, file, kind, line_start, line_end, sig, doc_first_line, exported"
)


def _symbol(row: sqlite3.Row) -> SymbolInfo:
    return SymbolInfo(
        name=row["name"],
        qualified_name=row["qualified_name"],
        file=row["file"],
        kind=row["kind"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        sig=row["sig"],
        doc_first_line=row["doc_first_line"],
        exported=bool(row["exported"]),
    )


def _edge(row: sqlite3.Row) -> Edge:
    return Edge(
        src=row["src"],
        dst=row["dst"],
        type=row["type"],
        edge_class=row["edge_class"],
        file=row["file"],
        line=row["line"],
        stale=bool(row["stale"]),
    )


class Graph:
    """Read side of the code graph."""

    def __init__(
        self,
        repo: Path,
        conn: sqlite3.Connection | None = None,
        *,
        profiles: ProfileSet | None = None,
    ) -> None:
        self.repo = repo
        self.conn = conn or store.connect(repo)
        self._profiles = profiles
        """Which manifests pin a file's packages: its language's `deps.manifests`. The
        indexer's set when the composition passes it; else the repository's, loaded when first
        asked."""

    def close(self) -> None:
        self.conn.close()

    # -- resolution ---------------------------------------------------------

    def resolve(self, name: str, near: str | None = None) -> tuple[SymbolInfo, ...]:
        """May return more than one, and **never silently picks** (packer P-6).

        `near` is a file path. Candidates sort by shortest import-graph distance from that
        file, then path-lexicographic. Folder-prefix sort would prefer `src/parse.py` over
        the `parse` `src/app.py` actually imports.
        """
        rows = self.conn.execute(
            f"SELECT {_SYMBOL_COLS} FROM symbols WHERE name = ? OR qualified_name = ?",
            (name, name),
        ).fetchall()
        found = [_symbol(r) for r in rows]
        if near:
            dist = self._import_distance(near)
            unreachable = 10**9
            found.sort(key=lambda s: (dist.get(s.file, unreachable), s.file))
        else:
            found.sort(key=lambda s: s.file)
        return tuple(found)

    def _import_distance(self, start: str) -> dict[str, int]:
        files = {r[0] for r in self.conn.execute("SELECT path FROM files")}
        if start not in files:
            return {}
        graph = import_graph(self.conn, files)
        dist = {start: 0}
        queue: deque[str] = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in graph.get(node, ()):
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        return dist

    def where_defined(self, name: str) -> tuple[Pointer, ...]:
        return tuple(
            Pointer(
                path=s.file,
                line_start=s.line_start,
                line_end=s.line_end,
                symbol=s.qualified_name,
            )
            for s in self.resolve(name)
        )

    def symbols_in(self, path: str) -> tuple[SymbolInfo, ...]:
        rows = self.conn.execute(
            f"SELECT {_SYMBOL_COLS} FROM symbols WHERE file = ? ORDER BY line_start",
            (path,),
        ).fetchall()
        return tuple(_symbol(r) for r in rows)

    def sig_of(self, sym: str) -> str | None:
        return self._unique_column(sym, "sig")

    def doc_first_line(self, sym: str) -> str | None:
        return self._unique_column(sym, "doc_first_line")

    def _unique_column(self, sym: str, column: str) -> str | None:
        """Never silently pick. Unique qualified_name wins; an ambiguous bare name
        returns None rather than LIMIT 1."""
        if column not in {"sig", "doc_first_line"}:
            raise ValueError(f"unsupported symbol column {column!r}")
        rows = self.conn.execute(
            f"SELECT {column} AS val, qualified_name, name FROM symbols "
            "WHERE qualified_name = ? OR name = ? ORDER BY file, line_start",
            (sym, sym),
        ).fetchall()
        exact = [r for r in rows if r["qualified_name"] == sym]
        if len(exact) == 1:
            return exact[0]["val"]
        if len(exact) > 1:
            return None
        named = [r for r in rows if r["name"] == sym]
        if len(named) == 1:
            return named[0]["val"]
        return None

    # -- edges --------------------------------------------------------------

    def callees(self, sym: str, *, kind: str | None = "calls") -> tuple[Edge, ...]:
        return self._edges_from("src", sym, kind)

    def callers(self, sym: str, *, kind: str | None = "calls") -> tuple[Edge, ...]:
        return self._edges_from("dst", sym, kind)

    def _edges_from(self, end: str, sym: str, kind: str | None) -> tuple[Edge, ...]:
        sql = f"SELECT * FROM edges WHERE {end} = ?"
        params: list[str] = [sym]
        if kind is not None:
            sql += " AND type = ?"
            params.append(kind)
        sql += " ORDER BY file, line"
        rows = self.conn.execute(sql, params).fetchall()
        return tuple(_edge(r) for r in rows)

    def unresolved_from(self, path: str) -> tuple[Edge, ...]:
        """Edges leaving `path` that the graph could not resolve.

        Separate from `callees` because an unknown edge's `dst` is a bare name rather than
        a qualified one, so these are invisible to any query keyed on a resolved target.
        """
        rows = self.conn.execute(
            "SELECT * FROM edges WHERE file = ? AND edge_class = 'unknown' ORDER BY line",
            (path,),
        ).fetchall()
        return tuple(_edge(r) for r in rows)

    def imports(self, path: str) -> tuple[ImportInfo, ...]:
        rows = self.conn.execute(
            "SELECT file, module, alias, line FROM imports WHERE file = ? ORDER BY line",
            (path,),
        ).fetchall()
        return tuple(
            ImportInfo(file=r["file"], module=r["module"], alias=r["alias"], line=r["line"])
            for r in rows
        )

    def globals_in(self, path: str) -> tuple[GlobalInfo, ...]:
        """Module and class state visible from a target. **Never inferred values** — the
        graph reports a declaration and its type text, not what it evaluates to."""
        rows = self.conn.execute(
            "SELECT qualified_name, name, file, line, kind, type_text FROM globals "
            "WHERE file = ? ORDER BY line",
            (path,),
        ).fetchall()
        return tuple(
            GlobalInfo(
                name=r["name"],
                qualified_name=r["qualified_name"],
                file=r["file"],
                line=r["line"],
                kind=r["kind"],
                type_text=r["type_text"],
            )
            for r in rows
        )

    def dependency_contracts(self, paths: Sequence[str]) -> tuple[DependencyContract, ...]:
        """Per (file, module): the imported symbols and the weakest edge class among them.

        Weakest rather than best: a contract is only as trustworthy as its least-resolved
        member, and reporting `resolved` because one of four names matched would overstate
        exactly what D19's classes exist to keep honest.
        """
        order = {"resolved": 0, "fallback": 1, "unknown": 2}
        grouped: dict[tuple[str, str], list[str]] = {}
        for path in paths:
            for imp in self.imports(path):
                module, _, symbol = imp.module.rpartition(".")
                grouped.setdefault((path, module or imp.module), []).append(
                    symbol or imp.module
                )
        out: list[DependencyContract] = []
        manifests: _ManifestCache = {}
        pinned_by: dict[str, tuple[str, ...]] = {}
        for (source, dependency), symbols in sorted(grouped.items()):
            if source not in pinned_by:
                pinned_by[source] = self._manifests_of(source)
            classes = [self._class_of(s) for s in symbols]
            weakest = max(classes, key=lambda c: order[c]) if classes else "unknown"
            out.append(
                DependencyContract(
                    source=source,
                    dependency=dependency,
                    symbols=tuple(sorted(set(symbols))),
                    declared_version=_declared_version(
                        self.repo, source, pinned_by[source], dependency, manifests
                    ),
                    edge_class=weakest,
                )
            )
        return tuple(out)

    def _manifests_of(self, path: str) -> tuple[str, ...]:
        """The manifests that pin `path`'s packages: those of the profile the index recorded it
        under (which language a file is is the profiles' answer), or none."""
        row = self.conn.execute("SELECT lang FROM files WHERE path = ?", (path,)).fetchone()
        if row is None:
            return ()
        if self._profiles is None:
            self._profiles = ProfileSet.load(repo=self.repo)
        profile = self._profiles.by_name(str(row[0]))
        return profile.deps.manifests if profile is not None else ()

    def _class_of(self, name: str) -> str:
        row = self.conn.execute(
            "SELECT 1 FROM symbols WHERE name = ? OR qualified_name = ? LIMIT 1",
            (name, name),
        ).fetchone()
        return "fallback" if row else "unknown"

    def public_contracts(self, paths: Sequence[str]) -> tuple[SymbolInfo, ...]:
        """Declarations a file exports (profile conventions recorded at index time).

        Python: `__all__` when it is a literal list/tuple of strings; otherwise
        module-level names that do not start with `_`. Methods are never public.
        JavaScript/TypeScript: module-level ESM bindings the extractor marked.
        """
        out: list[SymbolInfo] = []
        for path in paths:
            out.extend(s for s in self.symbols_in(path) if s.exported)
        return tuple(out)

    # -- bulk ---------------------------------------------------------------

    def blast_radius(self, syms: Sequence[str], depth: int = 2) -> tuple[str, ...]:
        return radius.blast_radius(self.conn, syms, depth).files

    def reach(
        self, syms: Sequence[str], depth: int = 2
    ) -> tuple[tuple[str, str | None, str], ...]:
        """`blast_radius`'s symbols, each with its file and the `edge_class` of its best path
        (`RadiusResult.reached`): what `impact` reports, so an `unknown` step reads as
        *possibly affected*."""
        return radius.blast_radius(self.conn, syms, depth).reached

    def coverage(self) -> tuple[tuple[str, bool], ...]:
        """Every indexed file and whether it parsed: a file that did not is a hole, and an
        answer that leaves it out reads as one with nothing there (invariant 20)."""
        rows = self.conn.execute("SELECT path, parse_ok FROM files ORDER BY path")
        return tuple((r["path"], bool(r["parse_ok"])) for r in rows)

    def file_edges(self) -> tuple[tuple[str, str | None, str, int], ...]:
        """Edges between files, counted per class: `(from, to, edge_class, count)`, `to` None
        where the edge's target is no indexed symbol (an `unknown` edge's bare name)."""
        rows = self.conn.execute(
            "SELECT e.file AS src, s.file AS dst, e.edge_class AS cls, COUNT(*) AS n "
            "FROM edges e LEFT JOIN symbols s ON s.qualified_name = e.dst "
            "GROUP BY e.file, s.file, e.edge_class ORDER BY e.file, s.file, e.edge_class"
        )
        return tuple((r["src"], r["dst"], r["cls"], int(r["n"])) for r in rows)

    def search(self, q: str, limit: int = 50) -> tuple[SymbolInfo, ...]:
        """FTS5 subword hits, ranked by BM25 then git recency."""
        hits = fts.search(self.conn, q)
        recency = git_recency(self.repo)
        ranked: list[tuple[float, int, SymbolInfo]] = []
        for sid, score in hits:
            row = self.conn.execute(
                f"SELECT {_SYMBOL_COLS} FROM symbols WHERE id = ?", (sid,)
            ).fetchone()
            if row is None:
                continue
            info = _symbol(row)
            # BM25 score first (lower is better), recency as the tie-break. A
            # repository where nothing is in the last 200 commits ranks purely by relevance.
            ranked.append((score, recency.get(info.file, len(recency) + 1), info))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return tuple(info for _, _, info in ranked[:limit])

    # -- diagnostics --------------------------------------------------------

    def diagnostics_in(self, path: str, lines: range | None = None) -> DiagnosticSet:
        """What a language server said about `path`: one indexed
        SELECT, never a server, so a read that asks stays parallel-safe. The reader's
        `DiagnosticsIndex`."""
        return diagnostics.diagnostics_in(self.conn, path, lines)

    # -- existence ----------------------------------------------------------

    def symbol_exists(self, name: str) -> bool:
        """Exact-name existence for PRECOND. A `False` here is what turns a doomed dispatch
        into `REPLANNED` before a model is ever called."""
        row = self.conn.execute(
            "SELECT 1 FROM symbols WHERE name = ? OR qualified_name = ? LIMIT 1",
            (name, name),
        ).fetchone()
        return row is not None

    def file_known(self, path: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM files WHERE path = ? LIMIT 1", (path,))
        return row.fetchone() is not None

    def indexed_sha(self, path: str) -> str | None:
        """The sha256 of the bytes `path`'s rows were parsed from (`store.indexed_sha`): a
        reader compares it with the file's own, since a file changed outside the harness is
        not reindexed until something reindexes it."""
        return store.indexed_sha(self.conn, path)

    def indexed_paths(self) -> tuple[str, ...]:
        """Every source path the index holds, ordered. The packer's tool allowlist."""
        rows = self.conn.execute("SELECT path FROM files ORDER BY path")
        return tuple(r[0] for r in rows)

    def indexed_languages(self) -> tuple[tuple[str, int], ...]:
        """Every language the index holds a file of, by the profile's name it was recorded
        under (`FileLanguage.name`), and how many files, ordered by name."""
        rows = self.conn.execute("SELECT lang, COUNT(*) FROM files GROUP BY lang ORDER BY lang")
        return tuple((str(r[0]), int(r[1])) for r in rows)

    def generation(self) -> int:
        return store.generation(self.conn)


_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _package_key(dependency: str) -> str | None:
    """Top-level package for a graph import module. Relative imports have no pin."""
    if not dependency or dependency.startswith("."):
        return None
    if dependency.startswith("@"):
        parts = dependency.replace(".", "/").split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    return dependency.split("/")[0].split(".")[0]


_ManifestCache = dict[Path, dict[str, object] | None]


def _declared_version(
    repo: Path,
    source_file: str,
    manifests: Sequence[str],
    dependency: str,
    cache: _ManifestCache,
) -> str | None:
    """Nearest language-matching manifest pin, or None. Never invents a version.

    `manifests` are the source's language's (its profile's `deps.manifests`); of those, the
    ones a pin can be read from (`_PIN_READERS`) are looked for, so Python reads
    `pyproject.toml` and JS/TS `package.json`. A mixed tree
    that has both at the same directory must not hand the other ecosystem's pin to D37.
    The nearest matching file is the contract: if it exists and does not name the
    package, the answer is None — an ancestor pin is a different package's lockfile.
    """
    pkg = _package_key(dependency)
    if pkg is None:
        return None
    readable = [m for m in manifests if m in _PIN_READERS]
    if not readable:
        return None
    try:
        root = repo.resolve()
    except OSError:
        return None
    landed = within(repo, source_file)
    if landed is None or landed == root:
        # The root's own parent is outside it: the walk starts at a file's directory.
        return None
    here = landed.parent
    while True:
        for name in readable:
            manifest = here / name
            if manifest.is_file():
                return _PIN_READERS[name](manifest, pkg, cache)
        if here == root or here.parent == here:
            break
        here = here.parent
    return None


def _manifest_table(path: Path, cache: _ManifestCache) -> dict[str, object] | None:
    """Parse once per call. The cache is the caller's dict, never process-global."""
    if path not in cache:
        try:
            text = path.read_text(encoding="utf-8")
            loaded: object = json.loads(text) if path.suffix == ".json" else tomllib.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, tomllib.TOMLDecodeError):
            cache[path] = None
        else:
            cache[path] = _as_str_map(loaded)
    return cache[path]


def _as_str_map(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, object], value)


def _npm_pin(path: Path, package: str, cache: _ManifestCache) -> str | None:
    table = _manifest_table(path, cache)
    if table is None:
        return None
    for key in ("dependencies", "optionalDependencies", "peerDependencies", "devDependencies"):
        block = _as_str_map(table.get(key))
        if block is None or package not in block:
            continue
        pin = block[package]
        return pin if isinstance(pin, str) and pin else None
    return None


def _pyproject_pin(path: Path, package: str, cache: _ManifestCache) -> str | None:
    table = _manifest_table(path, cache)
    if table is None:
        return None
    project = _as_str_map(table.get("project"))
    if project is not None:
        raw = project.get("dependencies")
        if isinstance(raw, list):
            pin = _pep508_pin(cast(list[object], raw), package)
            if pin is not None:
                return pin
    tool = _as_str_map(table.get("tool"))
    if tool is None:
        return None
    poetry = _as_str_map(tool.get("poetry"))
    if poetry is None:
        return None
    deps = _as_str_map(poetry.get("dependencies"))
    if deps is None:
        return None
    return _poetry_pin(deps, package)


def _norm_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _pep508_pin(deps: list[object], package: str) -> str | None:
    want = _norm_dist(package)
    for item in deps:
        if not isinstance(item, str):
            continue
        match = _REQ.match(item)
        if match is None or _norm_dist(match.group(1)) != want:
            continue
        pin = match.group(2).strip()
        return pin or None
    return None


def _poetry_pin(deps: dict[str, object], package: str) -> str | None:
    want = _norm_dist(package)
    for name, spec in deps.items():
        if _norm_dist(name) != want:
            continue
        if isinstance(spec, str):
            return spec or None
        if isinstance(spec, dict):
            table = _as_str_map(cast(object, spec))
            if table is None:
                return None
            version = table.get("version")
            return version if isinstance(version, str) and version else None
    return None


_PIN_READERS: dict[str, Callable[[Path, str, _ManifestCache], str | None]] = {
    "pyproject.toml": _pyproject_pin,
    "package.json": _npm_pin,
}
"""How a pin is read from each manifest a profile names, by its file name. Which manifests a
language has is its profile's (`deps.manifests`); a manifest with no reader here pins
nothing."""
