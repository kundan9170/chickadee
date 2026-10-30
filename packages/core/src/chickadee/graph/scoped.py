"""Bounded model-facing graph views.

Models never receive the store. `ScopedGraph` wraps `Graph` and enforces allowed paths,
result count, traversal depth and call count per role.

**A denial is a typed answer, not an exception and not a silent empty result.** §5.1: an
out-of-scope request "is a D37 runtime-correction signal; it never silently broadens packet
scope". Returning `()` would be indistinguishable from "nothing there", which is how a
coder concludes a symbol does not exist when it was merely out of its packet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from chickadee.graph.queries import Edge, GlobalInfo, Graph, Pointer, SymbolInfo

Role = Literal["coder", "scout", "planner"]
"""Who is asking: the packet-scoped coder, or a repository-scoped caller. `rescue` and
`reader` named roles D96 retired — the second author is Athena, on `planner` — and no caller
passed them. The critic and Zeus have no value here yet; which caller is repository-
scoped is a property of its tool surface, and keying this on the surface is left to the task
that wires those callers."""

# §5.1: the coder is packet-scoped; every other rung is repository-scoped and bounded by
# its own tool session budget instead.
REPO_SCOPED: frozenset[str] = frozenset({"scout", "planner"})

CODER_METHODS = frozenset(
    {"resolve", "where_defined", "sig_of", "globals_in", "callers", "callees"}
)
"""Exactly what §5.1 grants the SLM coder. Anything else is denied by name rather than by
whether it happens to be implemented."""

MAX_DEPTH_CODER = 1
"""Depth-1 callers/callees only, per §5.1."""


@dataclass(frozen=True)
class Denied:
    """Why a request was refused, in terms the caller can correct."""

    method: str
    reason: str
    scope: tuple[str, ...] = ()


@dataclass(frozen=True)
class Answer:
    """A scoped result. `denied` is set instead of `items` when the request was refused."""

    items: tuple[object, ...] = ()
    denied: Denied | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.denied is None


@dataclass
class ScopedGraph:
    graph: Graph
    role: Role
    scope: tuple[str, ...] = ()
    """Allowed path prefixes. Empty means repository scope, which only the non-coder roles
    get."""
    max_results: int = 50
    max_calls: int = 8
    calls: int = 0

    def _budget(self, method: str) -> Denied | None:
        if self.calls >= self.max_calls:
            return Denied(method, f"tool call budget exhausted ({self.max_calls})")
        self.calls += 1
        return None

    def _in_scope(self, path: str) -> bool:
        """Packet-scoped roles fail **closed**.

        An empty scope used to mean "no restriction", so a coder constructed without one
        got the whole repository — the failure mode where forgetting to pass an argument
        silently removes the boundary. An empty scope now permits nothing, and repository
        access is something a role has, never something an omission produces.
        """
        if self.role in REPO_SCOPED:
            return True
        return any(path == s or path.startswith(s.rstrip("/") + "/") for s in self.scope)

    def _guard(self, method: str) -> Denied | None:
        if self.role == "coder" and method not in CODER_METHODS:
            return Denied(method, f"{method} is not available to the coder rung")
        return self._budget(method)

    def _answer(self, items: Sequence[object]) -> Answer:
        capped = tuple(items[: self.max_results])
        return Answer(items=capped, truncated=len(items) > self.max_results)

    # -- the granted surface ------------------------------------------------

    def resolve(self, name: str, near: str | None = None) -> Answer:
        denied = self._guard("resolve")
        if denied:
            return Answer(denied=denied)
        found = [s for s in self.graph.resolve(name, near) if self._in_scope(s.file)]
        if not found and self.graph.resolve(name, near):
            return Answer(
                denied=Denied("resolve", f"{name} is defined outside packet scope", self.scope)
            )
        return self._answer(found)

    def where_defined(self, name: str) -> Answer:
        denied = self._guard("where_defined")
        if denied:
            return Answer(denied=denied)
        all_found = self.graph.where_defined(name)
        found: list[Pointer] = [p for p in all_found if self._in_scope(p.path)]
        if all_found and not found:
            return Answer(
                denied=Denied(
                    "where_defined", f"{name} is defined outside packet scope", self.scope
                )
            )
        return self._answer(found)

    def sig_of(self, sym: str) -> Answer:
        """Scope-checked like every other method.

        This one read straight through `Graph.sig_of`, so a coder could recover the exact
        signature of any symbol in the repository — the packet boundary held for
        `globals_in` and not for the method most likely to be asked.
        """
        denied = self._guard("sig_of")
        if denied:
            return Answer(denied=denied)
        candidates = self.graph.resolve(sym)
        visible = [s for s in candidates if self._in_scope(s.file)]
        if candidates and not visible:
            return Answer(
                denied=Denied("sig_of", f"{sym} is defined outside packet scope", self.scope)
            )
        return self._answer([s.sig for s in visible if s.sig])

    def globals_in(self, path: str) -> Answer:
        denied = self._guard("globals_in")
        if denied:
            return Answer(denied=denied)
        if not self._in_scope(path):
            return Answer(
                denied=Denied("globals_in", f"{path} is outside packet scope", self.scope)
            )
        found: list[GlobalInfo] = list(self.graph.globals_in(path))
        return self._answer(found)

    def callers(self, sym: str, depth: int = 1) -> Answer:
        return self._edges("callers", sym, depth)

    def callees(self, sym: str, depth: int = 1) -> Answer:
        return self._edges("callees", sym, depth)

    def _edges(self, method: str, sym: str, depth: int) -> Answer:
        denied = self._guard(method)
        if denied:
            return Answer(denied=denied)
        # `Graph.callers`/`callees` return direct edges only, so any depth above 1 would be
        # a promise the implementation does not keep. Rejected for every role rather than
        # silently answered at depth 1; multi-hop traversal is `radius.blast_radius`.
        if depth > MAX_DEPTH_CODER:
            limit = f"depth-{MAX_DEPTH_CODER}"
            reason = (
                f"depth {depth} exceeds the coder's {limit} limit"
                if self.role == "coder"
                else f"{method} returns direct edges only; use blast_radius for depth {depth}"
            )
            return Answer(denied=Denied(method, reason))
        edges: tuple[Edge, ...] = (
            self.graph.callers(sym) if method == "callers" else self.graph.callees(sym)
        )
        return self._answer([e for e in edges if self._in_scope(e.file)])

    def symbols_in(self, path: str) -> Answer:
        denied = self._guard("symbols_in")
        if denied:
            return Answer(denied=denied)
        if not self._in_scope(path):
            return Answer(
                denied=Denied("symbols_in", f"{path} is outside packet scope", self.scope)
            )
        found: list[SymbolInfo] = list(self.graph.symbols_in(path))
        return self._answer(found)
