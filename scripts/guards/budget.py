"""The budget guard: one ledger with rungs built in, and one interface.

The session's ledger was a subclass that wrote its parent's private fields and was Zeus's rung
by default; there were three ledger interfaces, and a rung view the box could not use (#276
T5.2, T5.3). `T-3.74` made `budget/manager.py`'s `CostLedger` the one ledger and its `Ledger`
the one interface. This fails the next second copy, by two rules over
`packages/core/src/chickadee/**/*.py`:

- **B1, one ledger:** no class subclasses `CostLedger`. A subclass is how the last one wrote
  its parent's fields; what a ledger needs is the ledger's.
- **B2, one interface:** no class outside `budget/manager.py` defines `charge_usage`. A loop
  that spends takes `Ledger`, a rung's view; a second protocol or a second concrete ledger
  is the drift this theme removed.

- **B3, spend read once**, outside `budget/family.py` and
  `journal/projection.py`, the two readers of a record's cost:
  - a `.cost_usd` read off a value narrowed to `SpendLine` or `CoderDoneP` (by `isinstance`
    in the same function, or a parameter annotated as one);
  - a sum of `.cost_usd`: a `+=` whose value reads it, or a `sum(...)` over it. This is the
    form a copy takes when it folds a typed list it never narrowed, as `SessionState.spent()`
    did.

What it cannot see: a ledger that charges under another method name, or a cost summed some
other way. That is the review checklist's first question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = "packages/core/src/chickadee/"
HOME = CORE + "budget/manager.py"
"""The ledger and its interface: the one file B2 lets define `charge_usage`."""

LEDGER = "CostLedger"

SPEND_HOMES: frozenset[str] = frozenset(
    {CORE + "budget/family.py", CORE + "journal/projection.py"}
)
"""The two readers of a record's cost: the family's spend, and a run's."""

PRICED = frozenset({"SpendLine", "CoderDoneP"})
"""The records that carry a charge's cost."""

COST = "cost_usd"


def _base_name(node: ast.expr) -> str | None:
    """A base class as written: `CostLedger` or `manager.CostLedger`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return None


def _names(node: ast.expr) -> set[str]:
    """The class names a type expression or an `isinstance` second argument names."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            out.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            out.update(part.strip() for part in sub.value.replace("|", ",").split(","))
    return out


def _cost_of(node: ast.AST) -> ast.expr | None:
    """The receiver of a `.cost_usd` read, or None."""
    if isinstance(node, ast.Attribute) and node.attr == COST:
        return node.value
    return None


def _priced(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Expressions, as written, that `fn` narrows to a priced record or takes as one."""
    out: set[str] = set()
    a = fn.args
    for param in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
        if param.annotation is not None and _names(param.annotation) & PRICED:
            out.add(param.arg)
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
            and _names(node.args[1]) & PRICED
        ):
            out.add(ast.unparse(node.args[0]))
    return out


def _summed(node: ast.AST) -> bool:
    """A `+=` whose value reads `.cost_usd`, or a `sum(...)` over it."""
    if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
        return any(_cost_of(n) is not None for n in ast.walk(node.value))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sum":
        return any(_cost_of(n) is not None for a in node.args for n in ast.walk(a))
    return False


def _spend_sites(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    out: list[tuple[str, ast.AST]] = []
    seen: set[int] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        priced = _priced(fn)
        for node in ast.walk(fn):
            if id(node) in seen:
                continue
            receiver = _cost_of(node)
            if (receiver is not None and ast.unparse(receiver) in priced) or _summed(node):
                seen.add(id(node))
                out.append(("B3", node))
    return out


def sites(tree: ast.Module, rel: str) -> list[tuple[str, ast.AST]]:
    """Every B1, B2 and B3 site in `tree`, the module at repo-relative path `rel`."""
    out: list[tuple[str, ast.AST]] = []
    if rel not in SPEND_HOMES:
        out += _spend_sites(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(_base_name(b) == LEDGER for b in node.bases):
            out.append(("B1", node))
        if rel != HOME:
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name == "charge_usage"
                ):
                    out.append(("B2", item))
    return out


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site, one violation each.

    Parses core's source, which is Python 3.12: run it with 3.12 or later, as CI's
    `python3` is."""
    violations: list[Violation] = []
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for rule, node in sites(ast.parse(source), rel):
            line = getattr(node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=f"{rule}: {text}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="budget",
    rule=(
        "one ledger, one interface, spend read once: nothing subclasses CostLedger (B1); "
        "nothing outside budget/manager.py defines charge_usage, a loop takes a rung's Ledger "
        "view (B2); a record's cost is read and summed only in budget/family.py and "
        "journal/projection.py (B3) (D156, D152)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
