"""Graph-side inputs to the planner's `ContextManifest`, with explicit gaps.

`10-planner.md` owns `ContextManifest` itself; this module supplies the parts the graph can
answer for — dependency contracts, callers, imports, globals, public contracts — and,
crucially, **what it could not answer**.

PL-10's rule is that every dependency and global a target names is either a resolved
contract or an explicit `GraphGap`. The whole value is in the second half: a manifest that
simply omits what the graph could not resolve looks identical to one where everything
resolved, and Athena, elaborating the stub, has no way to know she is reasoning from a
partial world.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from chickadee.graph.queries import (
    DependencyContract,
    Edge,
    GlobalInfo,
    ImportInfo,
    SymbolInfo,
)


class ManifestGraph(Protocol):
    """What `build` asks of the graph: `graph.queries.Graph` answers it. A protocol so a
    caller holding the graph through a narrower view can pass it, and a test a fake."""

    def file_known(self, path: str) -> bool: ...
    def imports(self, path: str) -> tuple[ImportInfo, ...]: ...
    def globals_in(self, path: str) -> tuple[GlobalInfo, ...]: ...
    def symbols_in(self, path: str) -> tuple[SymbolInfo, ...]: ...
    def callers(self, sym: str, *, kind: str | None = "calls") -> tuple[Edge, ...]: ...
    def dependency_contracts(self, paths: Sequence[str]) -> tuple[DependencyContract, ...]: ...
    def unresolved_from(self, path: str) -> tuple[Edge, ...]: ...
    def public_contracts(self, paths: Sequence[str]) -> tuple[SymbolInfo, ...]: ...


@dataclass(frozen=True)
class GraphGap:
    """Something a target names that the graph could not resolve to a contract."""

    kind: str  # file | import | global | caller | symbol
    name: str
    file: str
    why: str


@dataclass(frozen=True)
class ManifestInputs:
    target_files: tuple[str, ...]
    dependency_contracts: tuple[DependencyContract, ...]
    callers: tuple[Edge, ...]
    imports: tuple[ImportInfo, ...]
    globals: tuple[GlobalInfo, ...]
    public_contracts: tuple[SymbolInfo, ...]
    graph_gaps: tuple[GraphGap, ...]

    @property
    def complete(self) -> bool:
        """No gaps. Named rather than left to the caller to compute, because `not gaps` is
        the check every consumer would otherwise write slightly differently."""
        return not self.graph_gaps


def build(graph: ManifestGraph, targets: Sequence[str]) -> ManifestInputs:
    """Collect every graph fact a packet's targets depend on, and name every hole."""
    known = [t for t in targets if graph.file_known(t)]
    seen = set(known)
    # `file`, not `symbol`: a consumer grouping gaps by kind has to tell "this path is not
    # indexed" apart from "this reference did not resolve", and PL-10 exists so Athena,
    # elaborating, can act on the specific hole rather than on a generic one.
    gaps: list[GraphGap] = [
        GraphGap("file", t, t, "file is not in the index") for t in targets if t not in seen
    ]

    imports: list[ImportInfo] = []
    globals_: list[GlobalInfo] = []
    callers: list[Edge] = []
    for path in known:
        imports.extend(graph.imports(path))
        globals_.extend(graph.globals_in(path))
        for sym in graph.symbols_in(path):
            callers.extend(graph.callers(sym.qualified_name))

    contracts = graph.dependency_contracts(known)
    for contract in contracts:
        if contract.edge_class == "unknown":
            gaps.append(
                GraphGap(
                    "import",
                    contract.dependency,
                    contract.source,
                    "imported module resolves to no indexed symbol",
                )
            )

    # Two directions, two different holes, and the outgoing one was missing entirely.
    #
    # *Outgoing*: a target references a name the graph cannot resolve. PL-10 requires every
    # dependency a target names to be a resolved contract or an explicit gap, and an
    # unresolved call is exactly such a dependency. This was silently absent because an
    # unknown edge's `dst` is a bare name, so it never appears in any target's `callers()`
    # — the test that claimed to cover it carried an `or complete` escape and passed.
    for path in known:
        for edge in graph.unresolved_from(path):
            gaps.append(
                GraphGap("symbol", edge.dst, edge.file, "reference could not be resolved")
            )

    # *Incoming*: something calls into this target and the graph cannot say what. Reachable
    # via GR-2, which downgrades an edge to unknown when its target disappears. Dropping it
    # would understate the blast radius of exactly the symbols least understood.
    for edge in callers:
        if edge.edge_class == "unknown":
            gaps.append(
                GraphGap("caller", edge.src, edge.file, "calling symbol could not be resolved")
            )

    for glob in globals_:
        if glob.type_text is None and glob.kind == "module_state":
            gaps.append(
                GraphGap("global", glob.qualified_name, glob.file, "no declared type text")
            )

    return ManifestInputs(
        target_files=tuple(known),
        dependency_contracts=contracts,
        callers=tuple(callers),
        imports=tuple(imports),
        globals=tuple(globals_),
        public_contracts=graph.public_contracts(known),
        graph_gaps=tuple(gaps),
    )
