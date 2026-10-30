"""Import audit: the graph is deterministic code and never calls a model (invariant 2)."""

from __future__ import annotations

import ast
from pathlib import Path

GRAPH = Path(__file__).resolve().parents[3] / "core" / "src" / "chickadee" / "graph"

FORBIDDEN = ("chickadee.client", "chickadee.adapters", "openai", "anthropic", "httpx")
SPAWN = "subprocess"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_graph_imports_nothing_from_client_or_adapters() -> None:
    offenders: list[str] = []
    for module in sorted(GRAPH.glob("*.py")):
        for imported in _imports(module):
            if any(imported == f or imported.startswith(f + ".") for f in FORBIDDEN):
                offenders.append(f"{module.name} imports {imported}")
    assert not offenders, "the graph must not reach a model surface: " + "; ".join(offenders)


def test_graph_does_not_import_subprocess() -> None:
    offenders = [
        module.name for module in sorted(GRAPH.glob("*.py")) if SPAWN in _imports(module)
    ]
    assert not offenders, "graph git goes through ProcessSupervisor: " + ", ".join(offenders)


def test_the_graph_package_has_the_modules_the_spec_names() -> None:
    """The module map is the checklist."""
    expected = {
        "__init__.py",
        "store.py",
        "parse.py",
        "edges.py",
        "fts.py",
        "queries.py",
        "scoped.py",
        "manifest.py",
        "radius.py",
        "indexer.py",
        "lsp.py",
        "ignore.py",
    }
    assert expected <= {p.name for p in GRAPH.glob("*.py")}
    assert (GRAPH / "schema.sql").is_file()
