"""The packaging rule: eval reaches the harness only through its facade.

`cli/`, `mcp-server/` and `eval/` import **only** `chickadee.engine` and `chickadee.schema`.
Two reasons, and the second is the one specific to this package.

Components are private so the engine can enforce an ordering no host can skip — D57 records
`init` colliding with exactly this rule and moving into core rather than the rule bending.
And §5 of `19-eval`: the eval must not be helped by the thing it measures. An eval that
reached into `chickadee.profiles` for its test commands would let a profile edit move a
benchmark number with nothing in the diff to show it.

Asserted over the AST rather than by import-time failure, because a forbidden import that
only executes on one branch would pass a smoke test and fail in a run.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "chickadee_eval"

ALLOWED_HARNESS_MODULES = {"chickadee.engine", "chickadee.schema"}


def _harness_imports(tree: ast.AST) -> list[str]:
    """Every `chickadee.*` module this file imports, however it spells it."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] == "chickadee"]
        elif isinstance(node, ast.ImportFrom):
            # A relative import cannot reach another distribution, so only absolute ones
            # can break this rule.
            absolute = node.level == 0 and node.module
            if absolute and node.module and node.module.split(".")[0] == "chickadee":
                found.append(node.module)
    return found


def source_files() -> list[pathlib.Path]:
    return sorted(SRC.rglob("*.py"))


def test_the_package_has_sources_to_check() -> None:
    """A rule asserted over an empty set passes forever and means nothing."""
    assert len(source_files()) >= 4


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_module_imports_only_the_engine_facade(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for module in _harness_imports(tree):
        assert module in ALLOWED_HARNESS_MODULES, (
            f"{path.relative_to(SRC)} imports {module!r}. `eval` reaches the harness only "
            f"through {sorted(ALLOWED_HARNESS_MODULES)} (plan §3, EN-1) — a private module "
            "here would couple the measurement to internals the harness is free to change, "
            "and let a change move a benchmark number invisibly (19-eval §5)."
        )


def test_the_runner_does_not_import_an_arm() -> None:
    """Arms are injected, never imported. If the runner knew about a specific arm, the
    loop could not be tested without one, and adding an arm would edit the loop."""
    tree = ast.parse((SRC / "runner.py").read_text(encoding="utf-8"))
    imported = [
        m
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for m in [node.module]
    ]
    assert not any("arms" in m for m in imported), f"runner imports an arm: {imported}"


def test_the_engine_facade_is_importable_under_its_allowed_name() -> None:
    """The rule is only meaningful if the permitted surface actually exists."""
    import chickadee.engine
    import chickadee.schema

    assert hasattr(chickadee.engine, "PacketEngine")
    assert hasattr(chickadee.schema, "SandboxLevel")


def test_no_module_reaches_for_a_language_profile() -> None:
    """The specific coupling §5 warns about, named so a future edit trips on it by name
    rather than on the general rule."""
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        assert "chickadee.profiles" not in text, (
            f"{path.name} references chickadee.profiles; eval carries its own toolchain "
            "table so a profile edit cannot silently move a benchmark number"
        )
