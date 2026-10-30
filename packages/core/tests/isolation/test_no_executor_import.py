"""Isolation and process must not import executor."""

from __future__ import annotations

import ast
from pathlib import Path

import chickadee.isolation as isolation
import chickadee.process as process


def _is_executor_module(name: str | None) -> bool:
    return name == "chickadee.executor" or bool(name and name.startswith("chickadee.executor."))


def _package_of(path: Path) -> Path:
    """The directory that is this file's package (parent of the module)."""
    return path.parent


def _resolve_from(path: Path, chickadee: Path, node: ast.ImportFrom) -> list[str]:
    """Absolute names an ImportFrom binds, including `from chickadee import executor`."""
    if node.level == 0:
        base = node.module or ""
    else:
        current = _package_of(path)
        for _ in range(node.level - 1):
            current = current.parent
        parts = [] if current == chickadee else list(current.relative_to(chickadee).parts)
        if node.module:
            parts.extend(node.module.split("."))
        base = "chickadee." + ".".join(parts) if parts else "chickadee"
    names = [f"{base}.{alias.name}" if base else alias.name for alias in node.names]
    return [base, *names] if base else names


def _imports_executor(pkg: object) -> list[str]:
    root = Path(pkg.__file__).parent  # type: ignore[arg-type]
    chickadee = root.parent
    hits: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = tuple(_resolve_from(path, chickadee, node))
            if any(_is_executor_module(name) for name in imported):
                hits.append(str(path.relative_to(root.parent)))
                break
    return hits


def test_isolation_does_not_import_executor() -> None:
    assert _imports_executor(isolation) == []


def test_process_does_not_import_executor() -> None:
    assert _imports_executor(process) == []
