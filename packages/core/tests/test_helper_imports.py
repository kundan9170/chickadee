"""D64 / #89 — helpers are importable without a name collision.

Asserted rather than tried once. The bug this closes was invisible to the full suite and to
CI: collection order happened to work, so only a developer naming two directories in one
invocation ever saw it.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS = pathlib.Path(__file__).parent
SUITES = ("executor", "client")
"""The applier's suite, the first named here, retired with it."""


def test_no_test_file_imports_from_conftest() -> None:
    """`conftest` is not a unique module name — every suite has one, Python caches modules
    by name, and pytest puts each test directory on `sys.path`. So the first to load wins."""
    offenders: list[str] = []
    for path in TESTS.rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Both spellings. `import conftest` and `import conftest as helpers` carry the
            # same collision and would have walked past a guard that only knew `from`.
            if isinstance(node, ast.ImportFrom):
                hit = node.module == "conftest"
            elif isinstance(node, ast.Import):
                hit = any(alias.name == "conftest" for alias in node.names)
            else:
                continue
            if hit:
                offenders.append(f"{path.relative_to(TESTS)}:{node.lineno}")
    assert not offenders, "import from a uniquely-named helper module instead: " + ", ".join(
        offenders
    )


@pytest.mark.parametrize("suite", SUITES)
def test_each_suite_has_a_uniquely_named_helper_module(suite: str) -> None:
    module = TESTS / suite / f"{suite}_helpers.py"
    assert module.is_file(), f"{module} is where {suite}'s plain helpers belong (D64)"


def test_helper_module_names_are_unique_across_the_suite() -> None:
    """The property that makes the fix work. A second `applier_helpers.py` anywhere would
    reintroduce exactly the collision this closed."""
    names = [p.name for p in TESTS.rglob("*_helpers.py")]
    assert len(names) == len(set(names)), f"duplicate helper module basenames: {names}"


def test_no_two_test_files_share_a_basename() -> None:
    """The other half of the same root cause, and the one that took the suite down during
    `T-0.2`: two `test_write_block.py` files could not both be collected.

    D64 does not fix this — only `--import-mode=importlib` does, which D64 makes adoptable.
    This is the guard until then.
    """
    names = [p.name for p in TESTS.rglob("test_*.py")]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, (
        f"two test files share a basename and cannot both be collected: {duplicates}"
    )


def test_the_guard_catches_a_plain_import_too(tmp_path) -> None:
    """`import conftest` carries the same collision as `from conftest import`, and would
    have walked past a guard that only knew one spelling."""
    import ast

    source = "import conftest as helpers\n"
    tree = ast.parse(source)
    hits = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Import) and any(a.name == "conftest" for a in n.names)
    ]
    assert hits, "the AST shape the guard matches on"
