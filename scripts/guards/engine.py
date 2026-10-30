"""The engine guard: the facade keeps only composition.

`engine.py` held the bootstrap, the family budget projection, the transport choice, the turn
frame's head read and the crash repair's wording beside its wiring, and ten feature tasks edited
it at once (#276 T9.1, T9.4). `T-3.52` and `T-3.53` moved them to their homes. This fails the
next one: any module-level class, function or assignment in `engine.py` whose name is not in
`ALLOWED`, wherever it sits at module level (inside a top-level `if` or `try` too).

It also holds D150 decision 5: `budget/__init__.py` never imports `budget/family.py`, which
imports `context.session`. D150 gave a cycle as the reason, through `context/session.py`'s
import of `budget.spend`; `T-3.75` removed that import, and the rule stands as decided.

What it cannot see is arithmetic written inside a facade method. That is a review question, and
the engine-composition theme owns the facade's own shape. A size ratchet was refused:
a line count says nothing about which lines belong.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from pathlib import Path

from guards import Guard, Violation

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE = "packages/core/src/chickadee/engine.py"
BUDGET_INIT = "packages/core/src/chickadee/budget/__init__.py"
FAMILY = "chickadee.budget.family"

ALLOWED: dict[str, str] = {
    # the facade
    "PacketEngine": "the facade itself (EN-1)",
    "init_repo": "`chickadee init` runs before there is a config to build a PacketEngine with",
    "seed_engine_config": "`chickadee setup` writes the engine config no PacketEngine can be "
    "built without (D127)",
    "replay_run": "`chickadee replay` folds a run's journal, which needs no config (T-5.4)",
    "gc_runs": "`chickadee gc` prunes the runs on disk, which needs no config (D111, T-5.4)",
    "inspect_runs": "`chickadee inspect` reads the runs and sessions on disk, read-only, which "
    "needs no config",
    # the values and handles it returns to hosts
    "RunOptions": "a value hosts pass to the facade",
    "RunResult": "a value the facade returns",
    "DEFAULT_RUN_OPTIONS": "the facade's default RunOptions",
    "SessionInfo": "a value the facade returns",
    "CoderHandle": "the handle `coder()` returns",
    "SessionHandle": "the handle the session calls return",
    "TurnHandle": "the handle `turn()` returns (D136)",
    "NonInteractive": "the default InteractionPort when a host gives none (CL-6)",
    "ScriptedInteraction": "the InteractionPort that answers from a script, for tests and a "
    "host that knows its answers (12-cli.md §4, §9; T-4.5), beside NonInteractive",
    "InitReport": "re-exported so a surface reaches init through the facade alone (EN-1)",
    # the composition's own private parts
    "_EngineRun": "the run port the composition hands Zeus's coder",
    "_EngineBox": "the box port the composition hands Zeus's act tools",
    "_PHASE": "the message a method not yet built raises",
    "__all__": "the facade's public names",
}
"""Every name `engine.py` may define at module level, with why it belongs to the facade. A new
one is added here and read by a person, never inferred: a new host-facing value is a facade
change, and a new helper has a home elsewhere. An entry whose name `engine.py` no longer defines
is itself a violation, so the list cannot outlive the code."""


def _top_level(statements: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """Every statement at module level, through `if`, `try` and `with` blocks, which do not
    make a name any less the module's."""
    for stmt in statements:
        if isinstance(stmt, ast.If):
            yield from _top_level(stmt.body)
            yield from _top_level(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                yield from _top_level(block)
            for handler in stmt.handlers:
                yield from _top_level(handler.body)
        elif isinstance(stmt, ast.With):
            yield from _top_level(stmt.body)
        else:
            yield stmt


def defined(tree: ast.Module) -> list[tuple[str, int]]:
    """Each module-level name a class, a function or an assignment defines, with its line.
    Imports are not definitions: a re-export is how the facade keeps a moved name."""
    out: list[tuple[str, int]] = []
    for stmt in _top_level(tree.body):
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.append((stmt.name, stmt.lineno))
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for node in ast.walk(target):
                    if isinstance(node, ast.Name):
                        out.append((node.id, stmt.lineno))
        elif isinstance(stmt, ast.AnnAssign | ast.AugAssign) and isinstance(
            stmt.target, ast.Name
        ):
            out.append((stmt.target.id, stmt.lineno))
    return out


def imports_family(tree: ast.Module) -> list[int]:
    """Lines of `budget/__init__.py` that import `budget/family.py`, however spelled."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == FAMILY or a.name.startswith(f"{FAMILY}.") for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0:
                hit = module == FAMILY or (
                    module == "chickadee.budget" and any(a.name == "family" for a in node.names)
                )
            else:
                hit = module == "family" or (
                    module == "" and any(a.name == "family" for a in node.names)
                )
            if hit:
                lines.append(node.lineno)
    return lines


def _at(lines: Sequence[str], line: int) -> str:
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def find(files: Sequence[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for path in files:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)
        if rel == ENGINE:
            names = defined(tree)
            for name, line in names:
                if name not in ALLOWED:
                    violations.append(Violation(path=rel, line=line, text=_at(lines, line)))
            present = {name for name, _ in names}
            for name in sorted(set(ALLOWED) - present):
                violations.append(
                    Violation(
                        path=rel, line=0, text=f"stale ALLOWED entry: {name} is not defined"
                    )
                )
        elif rel == BUDGET_INIT:
            for line in imports_family(tree):
                violations.append(Violation(path=rel, line=line, text=_at(lines, line)))
    return violations


BASELINE: dict[str, int] = {}

GUARD = Guard(
    name="engine",
    rule=(
        "engine.py defines only the facade, the values and handles it returns, and its "
        "wiring; domain code lives in its own module (D150), and budget/__init__.py never "
        "imports budget/family.py"
    ),
    corpus=(ENGINE, BUDGET_INIT),
    find=find,
    baseline=BASELINE,
)
