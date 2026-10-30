"""The module-size guard: a module in `context/` holds at most 600 lines.

D157 split `context/`'s three largest files by job, because each had grown into several: a
1,714-line `slice.py` with nine jobs, a 1,492-line `compact.py` whose one production caller
used a sixth of it, a 627-line `handle.py` holding five concepts. A line budget is the proxy
that fails the next such file before it grows: a module that needs more than 600 lines is
doing more than one job, or one job that wants its own module.

A file of `n` lines yields `n - BUDGET` violations, one per line past the budget, so the ratchet
holds a file over budget exactly where it is: it may shrink, and its baseline then follows (the
runner's slack rule), but it never grows back. The runner prints the first few rows of a
failing file and counts the rest.

The corpus is `context/`, the package this theme split. Another package joins it at its own
theme, with the same budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from guards import Guard, Violation

REPO_ROOT = Path(__file__).resolve().parents[2]

BUDGET = 600
"""Lines per module. The one-job modules in `context/` ran 81 to 487 lines when D157 set it."""


def find(files: Sequence[Path]) -> list[Violation]:
    """One violation per line past `BUDGET`, in every file of the corpus."""
    violations: list[Violation] = []
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        n = len(path.read_text(encoding="utf-8").splitlines())
        violations.extend(
            Violation(path=rel, line=k, text=f"line {k} of {n}: past the {BUDGET}-line budget")
            for k in range(BUDGET + 1, n + 1)
        )
    return violations


BASELINE: dict[str, int] = {
    "packages/core/src/chickadee/context/session.py": 326,
    "packages/core/src/chickadee/context/store.py": 140,
}

GUARD = Guard(
    name="module_size",
    rule=(
        f"a module in context/ holds at most {BUDGET} lines: one past it is doing more than "
        "one job, so split it by job (D157)"
    ),
    corpus=("packages/core/src/chickadee/context/**/*.py",),
    find=find,
    baseline=BASELINE,
)
