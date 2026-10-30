"""The production `Navigator`: Zeus's `atlas`, `find`, `symbol`, `neighbors` and `impact`, and
the box's `tests_for`, over the code graph.

It renders; `graph.queries.Graph` queries. Every answer is pointers, names, signatures, doc
first lines and edges with their confidence, never a file body (invariant 13), and
it is never cut here: `NavigateTools` bounds it and stores the rest behind a handle.

**An absence is said, never left empty** (invariant 20). No index is `INDEX_UNAVAILABLE`. A
name the graph does not know is `UNRESOLVED` for `find` and `symbol`; for `neighbors` and
`impact`, whose failures 33 §3 does not list it among, the answer names it (`not in the
graph`). A known symbol with no callers says `callers: none`, which reads differently.

**The connection is the session worker's.** The engine opens it (`opened`) while it composes
the session, on the session's worker thread, and closes it with the session. Zeus runs every
call of a turn on that thread (a group is "not a thread pool", `ZeusLoop._group`), so the
connection never crosses threads.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Final

from chickadee.context.results import IndexUnavailable, ToolFailure, UnresolvedSearch
from chickadee.graph.parse import WALKED
from chickadee.graph.queries import Edge, Graph, SymbolInfo
from chickadee.graph.store import index_exists
from chickadee.graph.tests_for import TESTS_FOR_DEPTH, tests_for
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile

__all__ = ["GraphNavigator", "indexed_profiles", "opened"]

_NO_INDEX: Final = "this checkout has no index; `chickadee init` builds one"
_NEAR_LIMIT: Final = 50
"""Candidates `symbol` names when no symbol is named exactly: enough to pick from, and
within its inline bound."""
_ROOT: Final = "."
_TESTS_FOR_LIMIT: Final = (
    "run these first; dynamic imports, fixtures, data files and subprocesses are not seen"
)
"""What `tests_for` cannot see, said with every answer."""


def opened(repo: Path) -> Graph | None:
    """The checkout's graph, or None when it has no index (no `graph.db`, or one no index pass
    committed to). Asked before opening, because `connect` would create an empty database
    rather than say there is none."""
    return Graph(repo) if index_exists(repo) else None


def opened_until_closed(repo: Path, stack: ExitStack) -> Graph | None:
    """`opened`, closed when `stack` closes: one connection for the session's worker thread,
    where every call of a turn runs. The navigator's and `locate`'s graph, and every
    read's diagnostics."""
    graph = opened(repo)
    if graph is not None:
        stack.callback(graph.close)
    return graph


def indexed_profiles(graph: Graph | None, installed: ProfileSet) -> tuple[LanguageProfile, ...]:
    """The installed profiles the index read a file by: the languages it recorded, read off it
    rather than off a second walk or a second guess by extension."""
    if graph is None:
        return ()
    found = (installed.by_name(name) for name, _files in graph.indexed_languages())
    return tuple(p for p in found if p is not None)


class GraphNavigator:
    """`tools.context.Navigator` over `graph`, the checkout's, opened on the thread that will
    call it (`opened`), or None for a checkout with no index. `profiles` say what a test file
    is (`TestSpec.matches`), for `symbol`'s tests. The graph is its opener's to close."""

    def __init__(self, graph: Graph | None, profiles: Sequence[LanguageProfile] = ()) -> None:
        self.graph = graph
        self._profiles = tuple(profiles)

    # -- the five answers ------------------------------------------------------------------

    def atlas(self) -> str | ToolFailure:
        graph = self.graph
        if graph is None:
            return IndexUnavailable("", _NO_INDEX)
        coverage = graph.coverage()
        by_dir: dict[str, list[str]] = {}
        for path, _ in coverage:
            by_dir.setdefault(_component(path), []).append(path)
        lines = [f"## ATLAS    {len(coverage)} files · {len(by_dir)} components"]
        lines.append("components:")
        for name, paths in sorted(by_dir.items()):
            public = graph.public_contracts(paths)
            shown = ", ".join(_signed(s) for s in public) if public else "none"
            lines.append(f"  {name}    {len(paths)} files · public: {shown}")
        between: dict[tuple[str, str], dict[str, int]] = {}
        unresolved: dict[str, int] = {}
        for src, dst, cls, n in graph.file_edges():
            if dst is None:
                unresolved[_component(src)] = unresolved.get(_component(src), 0) + n
                continue
            a, b = _component(src), _component(dst)
            if a != b:
                counts = between.setdefault((a, b), {})
                counts[cls] = counts.get(cls, 0) + n
        lines.append("edges between components:")
        if not between:
            lines.append("  none")
        for (a, b), counts in sorted(between.items()):
            lines.append(f"  {a} -> {b}    {_counted(counts)}")
        failed = [path for path, ok in coverage if not ok]
        lines.append("not covered by the index:")
        lines.append(f"  parse failed: {', '.join(failed) if failed else 'none'}")
        gaps = ", ".join(f"{c} {n}" for c, n in sorted(unresolved.items()))
        lines.append(f"  edges with no indexed target: {gaps or 'none'}")
        lines.append("  files no profile parses are not indexed, and are not listed here")
        tagged = self._read_by_tags(graph)
        if tagged:
            lines.append(
                f"  read by a tags query, edges name-resolved and imports not read: {tagged}"
            )
        return "\n".join(lines)

    def _read_by_tags(self, graph: Graph) -> str:
        """The indexed languages no hand-written walker reads, with their
        file counts: a caller there is found by its name alone, never by what it imports."""
        profiles = {p.name: p for p in self._profiles}
        tagged = [
            f"{name} ({files} files)"
            for name, files in graph.indexed_languages()
            if name in profiles and any(g not in WALKED for g in profiles[name].grammars())
        ]
        return ", ".join(tagged)

    def symbol(self, name: str) -> str | ToolFailure:
        """The symbol `name` names exactly; with none, the closest by name (what `find`
        answered, the index's search over each name and its subwords, `graph/fts.py`)."""
        graph = self.graph
        if graph is None:
            return IndexUnavailable("", _NO_INDEX)
        found = graph.resolve(name)
        if not found:
            near = graph.search(name, limit=_NEAR_LIMIT)
            if not near:
                return _unresolved(graph, name)
            head = f"no symbol named {name!r}; by name, {len(near)} candidate(s):"
            return "\n".join([head, *(_pointer(s) for s in near)])
        lines: list[str] = []
        if len(found) > 1:
            lines.append(
                f"{name!r} is ambiguous: {len(found)} candidates, none picked (D19); "
                "qualify the name to choose one"
            )
        for s in found:
            callers = graph.callers(s.qualified_name, kind=None)
            callees = graph.callees(s.qualified_name, kind=None)
            tests = [e for e in callers if self._is_test(e.file)]
            lines.append(f"## SYMBOL {s.qualified_name}    {s.kind} · {_where(s)}")
            lines.append(f"  sig: {s.sig or 'none'}")
            lines.append(f"  doc: {s.doc_first_line or 'none'}")
            lines.extend(_edges("callers", callers, "src"))
            lines.extend(_edges("callees", callees, "dst"))
            lines.extend(_edges("tests", tests, "src"))
            lines.append(f"  body: read(path={s.file!r}, at={s.name!r})")
        return "\n".join(lines)

    def neighbors(self, target: str, depth: int) -> str | ToolFailure:
        graph = self.graph
        if graph is None:
            return IndexUnavailable("", _NO_INDEX)
        if graph.file_known(target):
            seeds = graph.symbols_in(target)
            head = f"## NEIGHBORS of {target}    file · {len(seeds)} symbol(s) · depth {depth}"
            lines = [head, "  in this file:"]
            lines.extend(f"    {_signed(s)}    {_where(s)}" for s in seeds)
            if not seeds:
                lines.append("    none")
            lines.extend(_rings(graph, [s.qualified_name for s in seeds], depth))
            return "\n".join(lines)
        found = graph.resolve(target)
        if not found:
            missing = f"not in the graph: {target!r} (no symbol or indexed path)"
            return f"## NEIGHBORS of {target}\n{missing}"
        lines = [f"## NEIGHBORS of {target}    {len(found)} candidate(s) · depth {depth}"]
        for s in found:
            lines.append(f"  {_signed(s)}    {_where(s)}")
            beside = [
                o for o in graph.symbols_in(s.file) if o.qualified_name != s.qualified_name
            ]
            shown = ", ".join(o.name for o in beside) if beside else "none"
            lines.append(f"  shares its file with: {shown}")
        lines.extend(_rings(graph, [s.qualified_name for s in found], depth))
        return "\n".join(lines)

    def impact(self, symbols: Sequence[str]) -> str | ToolFailure:
        graph = self.graph
        if graph is None:
            return IndexUnavailable("", _NO_INDEX)
        seeds: list[str] = []
        unknown: list[str] = []
        for name in symbols:
            found = graph.resolve(name)
            if found:
                seeds.extend(s.qualified_name for s in found)
            else:
                unknown.append(name)
        reached = [r for r in graph.reach(seeds) if r[0] not in seeds] if seeds else []
        files = {f for _, f, _ in reached}
        lines = [
            f"## IMPACT of {', '.join(symbols)}    {len(reached)} symbol(s) in "
            f"{len(files - {None})} file(s) · depth 2"
        ]
        if seeds and not reached:
            lines.append("  nothing in the graph calls or imports these")
        by_file: dict[str, list[tuple[str, str]]] = {}
        for sym, file, cls in reached:
            by_file.setdefault(file or "(no indexed file)", []).append((sym, cls))
        for file, entries in sorted(by_file.items()):
            lines.append(f"  {file}")
            lines.extend(f"    {sym}    {_confidence(cls)}" for sym, cls in entries)
        if unknown:
            lines.append(f"not in the graph: {', '.join(repr(n) for n in unknown)}")
        return "\n".join(lines)

    def tests_for(self, paths: Sequence[str]) -> str | ToolFailure:
        """The test files that depend on `paths` (`graph/tests_for`), each with the
        changed file it reaches, then the command that runs them, each file judged by its own
        language's profile; and every time, what the graph cannot see."""
        graph = self.graph
        if graph is None:
            return IndexUnavailable("", _NO_INDEX)
        edges = graph.file_edges()
        lines = [f"## TESTS FOR {', '.join(paths)}"]
        found = 0
        for profile in self._profiles:
            mine = [p for p in paths if profile.owns(p)]
            if not mine:
                continue
            selected = tests_for(mine, edges, profile)
            found += len(selected.tests)
            lines.extend(
                f"  {t}    {'changed itself' if not r else f'reaches {r}'}"
                for t, r in ((t, selected.reached[t]) for t in selected.tests)
            )
            if selected.tests and profile.tests.runner is not None:
                argv = [
                    a
                    for token in profile.tests.runner.argv
                    for a in (selected.tests if token == "{selector}" else (token,))
                ]
                lines.append(f"  run: {' '.join(argv)}")
        if found == 0:
            lines.append(f"  no test file within {TESTS_FOR_DEPTH} import hops")
        lines.append(_TESTS_FOR_LIMIT)
        return "\n".join(lines)

    def _is_test(self, path: str) -> bool:
        return any(p.tests.matches(path) for p in self._profiles)


# -- rendering -------------------------------------------------------------------------------


def _component(path: str) -> str:
    parent = str(PurePosixPath(path).parent)
    return _ROOT if parent in ("", ".") else parent


def _where(s: SymbolInfo) -> str:
    return f"{s.file}:{s.line_start}-{s.line_end}"


def _signed(s: SymbolInfo) -> str:
    return s.sig or s.name


def _pointer(s: SymbolInfo) -> str:
    return f"  {s.file}:{s.line_start}    {s.kind} {s.qualified_name}    {s.sig or ''}".rstrip()


def _counted(counts: dict[str, int]) -> str:
    return " · ".join(f"{cls} {n}" for cls, n in sorted(counts.items()))


def _confidence(cls: str) -> str:
    return {
        "resolved": "resolved",
        "fallback": "fallback (likely affected)",
    }.get(cls, "unknown (possibly affected)")


def _edges(label: str, edges: Sequence[Edge], end: str) -> list[str]:
    if not edges:
        return [f"  {label}: none"]
    out = [f"  {label}: {len(edges)}"]
    for e in edges:
        name = e.src if end == "src" else e.dst
        rel = "" if e.type == "calls" else f" {e.type}"
        stale = " stale" if e.stale else ""
        out.append(f"    {name}    {e.file}:{e.line}    [{e.edge_class}{rel}{stale}]")
    return out


def _rings(graph: Graph, seeds: Sequence[str], depth: int) -> list[str]:
    """Callers and callees out to `depth` hops, each named once at the hop that first reached
    it, with the edge's class: names and signatures, never bodies."""
    seen = set(seeds)
    frontier = list(seeds)
    out: list[str] = []
    for hop in range(1, depth + 1):
        found: list[tuple[str, str, str]] = []
        for sym in frontier:
            for e in graph.callers(sym, kind=None):
                if e.src not in seen:
                    seen.add(e.src)
                    found.append((e.src, "calls it", e.edge_class))
            for e in graph.callees(sym, kind=None):
                if e.dst not in seen:
                    seen.add(e.dst)
                    found.append((e.dst, "called", e.edge_class))
        out.append(f"  hop {hop}:")
        if not found:
            out.append("    none")
        for name, how, cls in found:
            sig = graph.sig_of(name)
            out.append(f"    {name}    {how}    [{cls}]{f'    {sig}' if sig else ''}")
        frontier = [name for name, _, _ in found]
        if not frontier:
            break
    return out


def _unresolved(graph: Graph, query: str) -> UnresolvedSearch:
    coverage = graph.coverage()
    return UnresolvedSearch(
        query,
        searched=tuple(path for path, ok in coverage if ok),
        uncovered=tuple(path for path, ok in coverage if not ok),
    )
