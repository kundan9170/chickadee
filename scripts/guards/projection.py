"""The projection guard: the run journal is folded once, in
`journal/projection.py`, and every reader asks that fold by payload type.

Fourteen readers each folded the journal for themselves, and the same facts (an operation's
lifecycle, spend, whether the run ended, checkpoints, a packet's terminal) had up to seven
copies (#276 T3.1 to T3.4). `T-3.62` to `T-3.64` moved every one onto the projection. This
fails the next one, by three rules, over `packages/core/src/chickadee/**/*.py`:

- **P1, reading a line's kind**, outside `journal/projection.py`: an `x.kind` compared with an
  `EventKind` member, or tested for membership in a container of them (written in place, or
  bound to a module-level name); a `match` on `x.kind` with an `EventKind` case; or one
  line's `.kind` compared with another's, where a parameter of the function is annotated as a
  journal line (`JournalEvent`, `Append`). Readers dispatch on the payload's type, and narrowing
  a line the projection returned is never a violation.
- **P2, reading a journal file:** a call to `read_events` outside `journal/projection.py`,
  `journal/reader.py`, `journal/writer.py` and `journal/gc.py`. A finished run is read
  through `RunRecords.projection` (whose file read is `reader.read_run`), or `project` /
  `project_readable` over the lines.
- **P3, iterating a journal:** a call to a held journal's `events()` (a receiver named
  `journal`, as every `JournalPort` holder names it) anywhere but as the direct argument of
  `project` or `project_before`; and a `for` over `read_events(...)` outside the fold's own
  files (`projection.py`, `reader.py`, `writer.py`), which is the same act over the file.

Writing a line names its kind and is never a violation: no rule looks at a call's arguments,
and a bare `kind` is the kind a writer was handed (the journal's fsync policy, the scout's
`check_record`, `navigate`'s `_DURABLE`), never a line's, so P1 reads only `x.kind`.

Exemptions are keyed by file and qualified function name, each with its reason; one that no
longer exempts anything fails, as the records guard's do.

What it cannot see: a fold that dispatches on payload types over lines it obtained some other
way. That is the review checklist's first question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = "packages/core/src/chickadee/"

FOLD = CORE + "journal/projection.py"
"""The one fold: the only file that reads a line's kind."""

P2_HOMES: frozenset[str] = frozenset(
    {FOLD, CORE + "journal/reader.py", CORE + "journal/writer.py", CORE + "journal/gc.py"}
)
"""The journal package's files that read a journal's file."""

P3_HOMES: frozenset[str] = frozenset(
    {FOLD, CORE + "journal/reader.py", CORE + "journal/writer.py"}
)
"""The files whose loop over a journal's lines is the fold itself: the projection, the
reader that yields the lines, and the writer that folds them on append and on resume."""

PROJECTING = frozenset({"project", "project_before", "project_readable"})
"""The calls whose argument a journal's lines may be: each folds them once."""

LINE_TYPES = frozenset({"JournalEvent", "Append"})
"""Annotations that make a parameter a journal line, for P1's kind-against-kind form."""

EXEMPT: dict[tuple[str, str], str] = {
    (CORE + "journal/gc.py", "classify"): (
        "the blob walk reads every line's sha fields to keep the blobs they name: a reference "
        "scan, not a fold, and it must see every line whatever its kind"
    ),
}
"""D152 decision 7's, each with its reason. Its third, `navigate._DURABLE`, tests the bare
`kind` a writer is handed, which P1 does not read (the module docstring), so it has no entry:
an entry that exempts nothing fails."""


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST


def _is_event_kind(node: ast.expr) -> bool:
    """`EventKind.<MEMBER>`, however the module was imported (`schema.EventKind.X`)."""
    if not isinstance(node, ast.Attribute):
        return False
    owner = node.value
    return (isinstance(owner, ast.Name) and owner.id == "EventKind") or (
        isinstance(owner, ast.Attribute) and owner.attr == "EventKind"
    )


def _kind_container(node: ast.expr) -> bool:
    """A set, tuple or list written in place with an `EventKind` member in it, or one
    wrapped in a `set`, `frozenset` or `tuple` call."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"set", "frozenset", "tuple"} and len(node.args) == 1:
            return _kind_container(node.args[0])
        return False
    if isinstance(node, ast.Set | ast.Tuple | ast.List):
        return any(_is_event_kind(e) for e in node.elts)
    return False


def kind_containers(tree: ast.Module) -> frozenset[str]:
    """The module-level names bound to a container of `EventKind` members."""
    names: set[str] = set()
    for stmt in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(stmt, ast.Assign):
            targets, value = list(stmt.targets), stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        if value is not None and _kind_container(value):
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return frozenset(names)


def _annotation(node: ast.expr | None) -> str:
    if node is None:
        return ""
    text = node.value if isinstance(node, ast.Constant) else ast.unparse(node)
    return str(text).strip().rsplit(".", 1)[-1]


def _line_params(fn: ast.AST) -> frozenset[str]:
    if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
        return frozenset()
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    return frozenset(p.arg for p in params if _annotation(p.annotation) in LINE_TYPES)


def _kind_of(node: ast.expr, lines: frozenset[str]) -> bool:
    """`x.kind`, where `x` is a parameter annotated as a journal line."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "kind"
        and isinstance(node.value, ast.Name)
        and node.value.id in lines
    )


def _reads_kind(node: ast.Compare, containers: frozenset[str], lines: frozenset[str]) -> bool:
    operands = [node.left, *node.comparators]
    kinds = [o for o in operands if isinstance(o, ast.Attribute) and o.attr == "kind"]
    if not kinds:
        return False
    if any(_is_event_kind(o) or _kind_container(o) for o in operands):
        return True
    if any(isinstance(o, ast.Name) and o.id in containers for o in operands):
        return True
    return len(kinds) > 1 and any(_kind_of(o, lines) for o in kinds)


def _matches_kind(node: ast.Match) -> bool:
    subject = node.subject
    if not (isinstance(subject, ast.Attribute) and subject.attr == "kind"):
        return False
    return any(
        isinstance(p, ast.MatchValue) and _is_event_kind(p.value)
        for case in node.cases
        for p in ast.walk(case.pattern)
    )


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _held_events(node: ast.Call) -> bool:
    """`journal.events()`, or `<anything>.journal.events()`."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "events"):
        return False
    owner = func.value
    return (isinstance(owner, ast.Name) and owner.id == "journal") or (
        isinstance(owner, ast.Attribute) and owner.attr == "journal"
    )


def _projected(tree: ast.Module) -> set[int]:
    """The ids of every call that is the direct argument of a projecting call."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in PROJECTING:
            out.update(id(a) for a in node.args if isinstance(a, ast.Call))
    return out


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every P1, P2 and P3 site in `tree`, the module at repo-relative path `rel`."""
    containers = kind_containers(tree)
    projected = _projected(tree)
    params: dict[str, frozenset[str]] = {}
    out: list[_Site] = []
    for scope, node in functions(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            params[scope] = _line_params(node)
            continue
        lines = params.get(scope, frozenset())
        compared = isinstance(node, ast.Compare) and _reads_kind(node, containers, lines)
        matched = isinstance(node, ast.Match) and _matches_kind(node)
        if rel != FOLD and (compared or matched):
            out.append(_Site("P1", scope, node))
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "read_events" and rel not in P2_HOMES:
                out.append(_Site("P2", scope, node))
            if _held_events(node) and id(node) not in projected:
                out.append(_Site("P3", scope, node))
        if isinstance(node, ast.For | ast.AsyncFor) and rel not in P3_HOMES:
            it = node.iter
            if isinstance(it, ast.Call) and _call_name(it) == "read_events":
                out.append(_Site("P3", scope, node))
    return out


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one.

    Parses core's source, which is Python 3.12: run it with 3.12 or later, as CI's
    `python3` is."""
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for site in sites(ast.parse(source), rel):
            key = (rel, site.function)
            if key in EXEMPT:
                used.add(key)
                continue
            line = getattr(site.node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=f"{site.rule}: {text}"))
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="projection",
    rule=(
        "the run journal is folded once, in journal/projection.py: ask the projection by "
        "payload type, never read a line's kind (P1), a journal file (P2) or a journal's "
        "lines (P3) yourself (D152)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
