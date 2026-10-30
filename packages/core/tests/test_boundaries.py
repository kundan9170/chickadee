"""Packaging boundary: `core` imports no CLI and no eval rig.

This is asserted statically rather than by importing `chickadee` and inspecting
`sys.modules`. A runtime check only sees the import graph that the test happened to
execute, so a violation inside a rarely-taken branch — a lazy import in an error path,
say — passes on every run until the day it doesn't. Parsing every source file sees all
of them.

A sibling rule runs in the other direction: surfaces import only `engine` and `schema`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

CORE_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# Distribution-level: nothing under core may reach for a surface package.
FORBIDDEN_ROOTS = {"chickadee_cli", "chickadee_eval"}


def _core_sources() -> list[pathlib.Path]:
    return sorted(CORE_SRC.rglob("*.py"))


def _imported_roots(tree: ast.Module) -> set[str]:
    """Top-level module name of every import, however it is spelled."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        # `level > 0` is a relative import and stays inside core by construction.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_core_sources_exist() -> None:
    """Guard the guard: an empty glob would make every assertion below vacuous."""
    assert _core_sources(), f"no Python sources found under {CORE_SRC}"


@pytest.mark.parametrize("path", _core_sources(), ids=lambda p: p.name)
def test_core_does_not_import_surfaces(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offending = _imported_roots(tree) & FORBIDDEN_ROOTS
    assert not offending, (
        f"{path.relative_to(CORE_SRC)} imports {sorted(offending)}; "
        "core is a library and the surfaces depend on it, never the reverse (plan §3)"
    )


def test_core_is_importable() -> None:
    import chickadee

    assert chickadee.__version__
