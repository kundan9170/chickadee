"""The findings report — pointers, not prose.

The scout's model ends its loop with a JSON draft. Nothing in it reaches Zeus unchecked: the
**validation gate** verifies every pointer — the file is there, the range is within
it, and a named symbol is at that span — and drops what fails, counting it. Deterministic
code checking model output, in the verifier's idiom. Facts get the same treatment where they
can: a fact must name a path, and the path must exist (or, for a fact saying it does not,
must not). What cannot be checked does not travel.

The report then answers Zeus as `locate`'s result. `unresolved` is a first-class field for
Zeus to settle; a report with no targets at all is `UNRESOLVED`, naming what was searched.
Nothing is trimmed here: over-inclusion is the packer's to
cut, budget-aware.

The frozen `Findings` (contract 8) has no field for the dropped count D132 names, nor for
how the call ended; `Report` carries it beside the contract rather than editing it.

Nothing here reads names out of the question: a name Zeus can write, Zeus looks up
itself, and a scout that cannot search says `UNAVAILABLE` rather than guessing candidates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from chickadee.adapters.base import parse_structured
from chickadee.edits.lines import split_lines
from chickadee.graph.names import names_qualified
from chickadee.limits import tokens_for_chars
from chickadee.paths import repo_relative, within
from chickadee.schema.adapter import ParseError
from chickadee.schema.findings import Findings, Pointer, Unresolved
from chickadee.secrets.denylist import Denylist, read_denied

__all__ = [
    "Checker",
    "Draft",
    "GraphCheck",
    "Outcome",
    "Report",
    "Unavailable",
    "gate",
    "parse_draft",
]

MAX_FACTS = 12
FACT_CHARS = 200
QUESTION_CHARS = 200
MAX_CANDIDATES = 8

Outcome = Literal["found", "unresolved", "cancelled", "unavailable"]
Unavailable = Literal["unreachable", "failed_mid_search", "no_tool_calling"]
"""`UNAVAILABLE{rung, reason}`'s `why`."""


# -- the graph, as the gate needs it -----------------------------------------------------


class Located(Protocol):
    """`graph.queries.SymbolInfo`, structurally."""

    @property
    def name(self) -> str: ...
    @property
    def qualified_name(self) -> str: ...
    @property
    def file(self) -> str: ...
    @property
    def line_start(self) -> int: ...
    @property
    def line_end(self) -> int: ...


class EdgeLike(Protocol):
    """`graph.queries.Edge`, structurally."""

    @property
    def file(self) -> str: ...
    @property
    def line(self) -> int: ...
    @property
    def edge_class(self) -> str: ...


class GraphCheck(Protocol):
    """What the gate asks the graph. `graph.queries.Graph` is this."""

    def file_known(self, path: str) -> bool: ...
    def symbols_in(self, path: str) -> Sequence[Located]: ...
    def resolve(self, name: str, near: str | None = None) -> Sequence[Located]: ...
    def callers(self, sym: str, *, kind: str | None = "calls") -> Sequence[EdgeLike]: ...


# -- the draft ---------------------------------------------------------------------------


class Draft(BaseModel):
    """What the model emits: `Findings`' shape, read leniently so one malformed pointer is
    one dropped pointer rather than a lost report. `truncated` is the harness's to set."""

    model_config = ConfigDict(extra="ignore")

    targets: list[object] = []
    callers: list[object] = []
    facts: list[object] = []
    unresolved: list[object] = []
    evidence: list[object] = []


def parse_draft(text: str) -> Draft | ParseError:
    return parse_structured(text, Draft)


def _pointer(raw: object) -> Pointer | None:
    if not isinstance(raw, dict):
        return None
    d = cast(dict[str, object], raw)
    line = d.get("line")
    start = d.get("line_start", line)
    end = d.get("line_end", start)
    try:
        # The model's `edge_class` is not taken: it is a claim about the graph, and the
        # gate asks the graph instead.
        return Pointer.model_validate(
            {
                "path": d.get("path"),
                "line_start": start,
                "line_end": end,
                "symbol": d.get("symbol"),
            }
        )
    except ValidationError:
        return None


# -- the gate ----------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATHISH = re.compile(
    r"(?<![\w/.\-])(?:[\w.\-]+/[\w.\-/]*|[\w\-]+\.[A-Za-z][A-Za-z0-9]{0,5})(?![\w/])"
)
_ABSENT = re.compile(
    r"\b(?:does not exist|doesn't exist|not exist|no such|missing|absent)\b", re.I
)


def _last(symbol: str) -> str:
    return re.split(r"[.:#]", symbol)[-1]


class Checker:
    """The gate's view of the workspace: files on disk, bounded to the root and to what the
    read rule allows, and the graph where it has the file. Line counts are read once per
    file per call.

    `denylist` is the session's (`zeus/tools/navigate.py` passes the executor's); the default
    is the built-in list, as `Search`'s is."""

    def __init__(
        self, root: Path, graph: GraphCheck | None, *, denylist: Denylist | None = None
    ) -> None:
        self.root = root.resolve()
        self.graph = graph
        self.denylist = denylist if denylist is not None else Denylist()
        self._lines: dict[str, list[str] | None] = {}

    def forget(self) -> None:
        """Drop the line cache: files change between `locate` calls, and a pointer is checked
        against the file as it is now."""
        self._lines.clear()

    def lines(self, path: str) -> list[str] | None:
        if path in self._lines:
            return self._lines[path]
        text: list[str] | None = None
        if repo_relative(path):
            full = within(self.root, path)
            # A pointer into a secret fails as one into a missing file does: the check reads
            # the file to verify the pointer, and the read rule decides what may be read.
            if full is not None and full.is_file() and not self._denied(path, full):
                try:
                    text = list(split_lines(full.read_bytes().decode("utf-8", "replace")))
                except OSError:
                    text = None
        self._lines[path] = text
        return text

    def _denied(self, path: str, real: Path) -> bool:
        return (
            read_denied(path, self.root / path, self.denylist, workspace=self.root, real=real)
            is not None
        )

    def symbols(self, name: str) -> list[Located]:
        """What the graph holds under `name`, which may be partly qualified
        (`HttpClient.request`): `resolve` matches a bare or a fully qualified name only."""
        if self.graph is None:
            return []
        found = list(self.graph.resolve(name))
        if not found and "." in name:
            tail = "." + name
            found = [
                s for s in self.graph.resolve(_last(name)) if s.qualified_name.endswith(tail)
            ]
        return found

    def exists(self, token: str) -> bool:
        path = token.rstrip("/")
        if not path or not repo_relative(path):
            return False
        full = within(self.root, path)
        return full is not None and full.exists()

    def pointer(
        self, p: Pointer, role: Literal["target", "caller", "evidence"]
    ) -> Pointer | None:
        """The pointer as it reaches Zeus, or None when it fails. A target's symbol must be
        *defined* at the span; a caller's names the callee, which must *appear* in it."""
        lines = self.lines(p.path)
        if lines is None or p.line_end > max(len(lines), 1):
            return None
        span = "\n".join(lines[p.line_start - 1 : p.line_end])
        if p.symbol is not None:
            if role == "caller":
                if _last(p.symbol) not in _WORD.findall(span):
                    return None
                return p.model_copy(update={"edge_class": self._edge(p)})
            if not self._defined(p, span):
                return None
        return p.model_copy(update={"edge_class": None})

    def _defined(self, p: Pointer, span: str) -> bool:
        assert p.symbol is not None
        want = p.symbol
        if self.graph is not None and self.graph.file_known(p.path):
            return any(
                (s.name == want or names_qualified(s.qualified_name, want))
                and s.line_start <= p.line_end
                and s.line_end >= p.line_start
                for s in self.graph.symbols_in(p.path)
            )
        # A file the graph does not index: the name must at least be written there.
        return _last(want) in _WORD.findall(span)

    def _edge(self, p: Pointer) -> str:
        """D19's class for this call site, from the graph's edges into the callee — `unknown`
        when the graph has no edge here, never the model's word for it."""
        if self.graph is None or p.symbol is None:
            return "unknown"
        names = {p.symbol, *(s.qualified_name for s in self.symbols(p.symbol))}
        for name in names:
            for e in self.graph.callers(name):
                if e.file == p.path and p.line_start <= e.line <= p.line_end:
                    return e.edge_class
        return "unknown"

    def fact(self, raw: object) -> str | None:
        """A fact survives only if it names a path the workspace can confirm: present, or —
        for a fact that says so — absent."""
        if not isinstance(raw, str):
            return None
        text = " ".join(raw.split())[:FACT_CHARS]
        tokens = _PATHISH.findall(text)
        if not tokens:
            return None
        absent = bool(_ABSENT.search(text))
        ok = (
            all(not self.exists(t) for t in tokens)
            if absent
            else all(self.exists(t) for t in tokens)
        )
        return text if ok else None

    def candidate(self, raw: object) -> str | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
        text = raw.strip()
        if self.exists(text):
            return text
        if self.symbols(text):
            return text
        return None


def gate(draft: Draft, checker: Checker, *, truncated: bool) -> tuple[Findings, int, int]:
    """SC-1. Returns the findings Zeus may see, and how many pointers and facts it dropped."""
    dropped = 0

    def keep(
        raw: list[object], role: Literal["target", "caller", "evidence"]
    ) -> tuple[Pointer, ...]:
        nonlocal dropped
        out: list[Pointer] = []
        for item in raw:
            p = _pointer(item)
            checked = checker.pointer(p, role) if p is not None else None
            if checked is None:
                dropped += 1
            elif checked not in out:
                out.append(checked)
        return tuple(out)

    targets = keep(draft.targets, "target")
    callers = keep(draft.callers, "caller")
    evidence = keep(draft.evidence, "evidence")
    # Facts past the cap are counted with the ones the gate refused: a count that left them
    # out would report fewer dropped than the model sent.
    facts = [f for f in (checker.fact(r) for r in draft.facts[:MAX_FACTS]) if f is not None]
    dropped_facts = len(draft.facts) - len(facts)
    unresolved: list[Unresolved] = []
    for raw in draft.unresolved:
        if not isinstance(raw, dict):
            continue
        d = cast(dict[str, object], raw)
        question = d.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        cands = d.get("candidates", [])
        kept = [
            c
            for c in (
                checker.candidate(x)
                for x in (cast(list[object], cands) if isinstance(cands, list) else [])
            )
            if c is not None
        ]
        unresolved.append(
            Unresolved(
                question=" ".join(question.split())[:QUESTION_CHARS],
                candidates=tuple(dict.fromkeys(kept))[:MAX_CANDIDATES],
            )
        )
    findings = Findings(
        targets=targets,
        callers=callers,
        facts=tuple(facts),
        unresolved=tuple(unresolved),
        evidence=evidence,
        truncated=truncated,
    )
    return findings, dropped, dropped_facts


# -- the report --------------------------------------------------------------------------


@dataclass(frozen=True)
class Report:
    """One `locate` call's answer: the checked `Findings`, and what the contract has no
    field for."""

    question: str
    findings: Findings
    outcome: Outcome
    model: str
    calls: int = 0
    escapes: int = 0
    tokens: int = 0
    """The scout's own model tokens — what it spent, against what it returns."""
    cost_usd: float = 0.0
    dropped_pointers: int = 0
    dropped_facts: int = 0
    searched: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple[str, ...])
    unavailable: Unavailable | None = None
    """Set with `outcome == "unavailable"`: why the scout could not search."""
    detail: str = ""
    """What `unavailable` observed: the client error, or the calls made before it."""

    @property
    def returned_tokens(self) -> int:
        return tokens_for_chars(len(self.render()))

    def render(self) -> str:
        """What Zeus reads as `locate`'s result: the shape, or the typed failure."""
        if self.outcome == "unavailable":
            # Not a findings report: nothing was searched, or what was is gone with the
            # rung, and an empty report would read as a search that found nothing.
            why = self.unavailable or "unreachable"
            return (
                f"UNAVAILABLE{{rung: {self.model!r}, reason: {why}, detail: {self.detail!r}}}"
                " — no search ran to completion; use symbol, find, grep_in or "
                'shell("rg") directly, or ask the user'
            )
        f = self.findings
        head = (
            f"## FINDINGS    scout: {self.model} · {self.calls} calls · {self.escapes} "
            f"escape(s) · {self.tokens:,} tokens"
        )
        lines = [head]
        lines.extend(f"note: {n}" for n in self.notes)
        if self.outcome == "cancelled":
            lines.append("CANCELLED — the session's stop ended the search; partial: true")
        if not f.targets:
            searched = ", ".join(self.searched) or "the snapshot"
            lines.append(
                f"UNRESOLVED{{question: {self.question[:QUESTION_CHARS]!r}, "
                f"searched: [{searched}]}}"
            )
        lines += _section("targets", [_row(p) for p in f.targets])
        lines += _section("callers", [_row(p, caller=True) for p in f.callers])
        lines += _section("facts", list(f.facts))
        lines += _section(
            "unresolved",
            [
                f"{u.question!r} → {len(u.candidates)} candidate(s)"
                + (f": {', '.join(u.candidates)}" if u.candidates else "")
                for u in f.unresolved
            ],
        )
        lines += _section("evidence", [_row(p) for p in f.evidence])
        if self.dropped_pointers or self.dropped_facts:
            lines.append(
                f"gate: dropped {self.dropped_pointers} pointer(s) and {self.dropped_facts} "
                f"fact(s) that failed validation or were past the {MAX_FACTS}-fact cap (SC-1)"
            )
        lines.append(f"truncated: {'true' if f.truncated else 'false'}")
        return "\n".join(lines)


def _section(title: str, rows: list[str]) -> list[str]:
    return [f"{title}:", *(f"  - {r}" for r in rows)] if rows else []


def _row(p: Pointer, *, caller: bool = False) -> str:
    span = f"{p.line_start}" if p.line_start == p.line_end else f"{p.line_start}-{p.line_end}"
    where = f"{p.path}:{span}"
    if p.symbol is None:
        return where
    if caller:
        return f"{where}   → {p.symbol}   ({p.edge_class or 'unknown'})"
    return f"{where}   {p.symbol}"
