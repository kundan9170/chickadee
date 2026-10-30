"""The tokens guard: there is one token estimate, P-2's, in `limits.py`.

The estimate was 3.5 characters per token in two places, each with its own constant, and 4 in
two more, against the P-2 that `02-packer.md` §8 locks (#276 T7.3). `T-3.72` moved every site to
`limits.tokens_est` / `tokens_for_chars`. This fails the next copy, by two rules, over
`packages/core/src/chickadee/**/*.py`:

- **T1, a second constant:** an assignment to a name `CHARS_PER_TOKEN` outside `limits.py`.
- **T2, an inline estimate:** a division (`/` or `//`) whose left operand is a `len(...)` call
  and whose right operand is the literal `3.5` or `4`, outside `limits.py`.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. There are none at landing.

What it cannot see: an estimate written with another ratio, or through a name other than
`len`. That is the review checklist's first question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]

HOME = "packages/core/src/chickadee/limits.py"
"""The one home of P-2."""

RATIOS = frozenset({3.5, 4})
"""The two ratios the copies used: P-2's, and the 4 that contradicted it."""

EXEMPT: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class _Site:
    rule: str
    function: str
    node: ast.AST


def _assigns_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False
    return any(isinstance(t, ast.Name) and t.id == "CHARS_PER_TOKEN" for t in targets)


def _inline_estimate(node: ast.AST) -> bool:
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div | ast.FloorDiv)):
        return False
    left, right = node.left, node.right
    calls_len = (
        isinstance(left, ast.Call) and isinstance(left.func, ast.Name) and left.func.id == "len"
    )
    ratio = (
        isinstance(right, ast.Constant)
        and isinstance(right.value, int | float)
        and not isinstance(right.value, bool)
        and right.value in RATIOS
    )
    return calls_len and ratio


def sites(tree: ast.Module, rel: str) -> list[_Site]:
    """Every T1 and T2 site in `tree`, the module at repo-relative path `rel`."""
    if rel == HOME:
        return []
    out: list[_Site] = []
    for scope, node in functions(tree):
        if _assigns_constant(node):
            out.append(_Site("T1", scope, node))
        if _inline_estimate(node):
            out.append(_Site("T2", scope, node))
    return out


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one."""
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
    name="tokens",
    rule=(
        "one token estimate: use chickadee.limits.tokens_est / tokens_for_chars (P-2, 3.5 "
        "characters per token); never a second CHARS_PER_TOKEN (T1) or an inline "
        "len(...) / 3.5 or // 4 (T2) (D155)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
