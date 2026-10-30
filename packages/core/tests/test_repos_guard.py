"""Tests make git repositories one way (`packages/testing/repos.py`): no test in any
package spawns git itself, and no suite keeps its own `git` helper. The guard that keeps the
second copy from coming back; there is no baseline, because every copy has
moved."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGES = Path(__file__).resolve().parents[2]
SUITES = tuple(sorted(PACKAGES.glob("*/tests")))
"""Every package's tests: `core`, `cli` and `eval` today, and any package added later."""
SPAWNS = frozenset({"run", "check_output", "check_call", "Popen", "call"})


def _git_spawns(tree: ast.AST) -> list[int]:
    """Lines where `subprocess.<spawn>(...)` is handed an argv that begins with "git"."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        func = node.func
        if not (
            isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
            and func.attr in SPAWNS
            and node.args
        ):
            continue
        argv = node.args[0]
        if isinstance(argv, ast.List | ast.Tuple) and argv.elts:
            first = argv.elts[0]
            if isinstance(first, ast.Constant) and first.value == "git":
                lines.append(node.lineno)
    return lines


def _git_helpers(tree: ast.Module) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in ("git", "_git")
    ]


def test_no_test_spawns_git_itself() -> None:
    found = [
        f"{path.relative_to(PACKAGES)}:{line}"
        for path in _test_modules()
        for line in _git_spawns(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not found, "run git through packages/testing/repos.py: " + ", ".join(found)


def test_no_suite_keeps_its_own_git_helper() -> None:
    found = [
        f"{path.relative_to(PACKAGES)}:{line}"
        for path in _test_modules()
        for line in _git_helpers(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not found, "import git from packages/testing/repos.py instead: " + ", ".join(found)


def _production_git(tree: ast.AST) -> list[int]:
    """Lines that import the harness's own `git` (`chickadee.process.git`) into a test. It runs
    through `ProcessSupervisor`, so arranging a test with it is counted as the harness's work
    (g2's spawn inventory). `run_git` stays importable: `test_harness_argv` tests it."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("chickadee.process")
        and any(alias.name == "git" for alias in node.names)
    ]


def test_no_test_arranges_a_repository_with_the_harness_git() -> None:
    found = [
        f"{path.relative_to(PACKAGES)}:{line}"
        for path in _test_modules()
        for line in _production_git(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not found, "use packages/testing/repos.py's git: " + ", ".join(found)


def test_the_guard_reads_every_package() -> None:
    assert {s.parent.name for s in SUITES} >= {"core", "cli", "eval"}


def _test_modules() -> list[Path]:
    return [p for suite in SUITES for p in sorted(suite.rglob("*.py"))]
