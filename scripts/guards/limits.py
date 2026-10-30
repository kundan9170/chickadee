"""The limits guard (T-3.85's review): one resolved limit is read one way, in `limits.py`.

A role's resolved row was read by six scans of `EngineConfig.resolved_limits`, each with its
own answer to a missing row: a `KeyError`, a `HarnessBug` worded three ways, and in one place a
silent default. `T-3.85` moved every reader to `limits.resolved_value(resolved, role, name)`.
This fails the next copy, over `packages/core/src/**/*.py`:

- **L1, a row read in place:** the attribute `tokens_or_value` read outside `limits.py`.

Exemptions are keyed by file and qualified scope, each with its reason; one that no longer
exempts anything fails. There are none at landing.

What it cannot see: a reader that copies the rows into another structure through
`dataclasses.asdict` or `getattr`. That is the review checklist's first question.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation
from guards._scope import functions

REPO_ROOT = Path(__file__).resolve().parents[2]

HOME = "packages/core/src/chickadee/limits.py"
"""The one home of a resolved row's lookup."""

FIELD = "tokens_or_value"

EXEMPT: dict[tuple[str, str], str] = {}


def sites(tree: ast.Module, rel: str) -> list[tuple[str, ast.AST]]:
    """Every L1 site in `tree`, the module at repo-relative path `rel`, with its scope."""
    if rel == HOME:
        return []
    return [
        (scope, node)
        for scope, node in functions(tree)
        if isinstance(node, ast.Attribute) and node.attr == FIELD
    ]


def find(files: Sequence[Path]) -> list[Violation]:
    """Every site outside its exemptions, and every exemption that no longer exempts one."""
    violations: list[Violation] = []
    used: set[tuple[str, str]] = set()
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for scope, node in sites(ast.parse(source), rel):
            key = (rel, scope)
            if key in EXEMPT:
                used.add(key)
                continue
            line = getattr(node, "lineno", 0)
            text = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            violations.append(Violation(path=rel, line=line, text=f"L1: {text}"))
    for rel, fn in sorted(set(EXEMPT) - used):
        violations.append(Violation(path=rel, line=0, text=f"stale exemption: {fn}"))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="limits",
    rule=(
        "one lookup of a resolved limit: use chickadee.limits.resolved_value(resolved, role, "
        "name), never read ResolvedLimit.tokens_or_value in place (L1)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
