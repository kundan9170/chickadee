"""The instructions guard: every refusal a model reads ends with what to do next.

A failure type whose fields say why (`HANDLE_EXPIRED`, `SECRET_PATH`, `BOUND_REACHED`,...)
derives its instruction in its own class, and a test holds every type to rendering one. A
free-text refusal, `InvalidArgument(field, why)`, has nothing to derive it from, so each site
passes `do=`. This fails one built without it, over `packages/core/src/chickadee/**/*.py`:

- **I1, a bare refusal:** a call to `InvalidArgument` (by name or as an attribute) with no `do`
  keyword.

The baseline is the sites that predate D221, per file; they move area by area and the count may
only go down. A call that passes its arguments through `**` is counted too: the guard cannot see
inside it, and a refusal built that way should name its `do` where it is written.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation

REPO_ROOT = Path(__file__).resolve().parents[2]

REFUSAL = "InvalidArgument"
"""The free-text refusal (`context/results.py`), the one type with no instruction of its own."""


def _names_refusal(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == REFUSAL
    return isinstance(func, ast.Attribute) and func.attr == REFUSAL


def sites(tree: ast.Module) -> list[ast.Call]:
    """Every I1 site in `tree`: an `InvalidArgument(...)` with no `do=`."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _names_refusal(node.func)
        and not any(k.arg == "do" for k in node.keywords)
    ]


def find(files: Sequence[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in sites(ast.parse(source)):
            text = lines[node.lineno - 1].strip() if 0 < node.lineno <= len(lines) else ""
            violations.append(Violation(path=rel, line=node.lineno, text=f"I1: {text}"))
    return violations


BASELINE: dict[str, int] = {
    "packages/core/src/chickadee/skills/load.py": 8,
}

GUARD = Guard(
    name="instructions",
    rule=(
        "every refusal a model reads ends with what to do next: pass do= to InvalidArgument, "
        "the next action in words the model can act on (D221)"
    ),
    corpus=("packages/core/src/**/*.py",),
    find=find,
    baseline=BASELINE,
)
