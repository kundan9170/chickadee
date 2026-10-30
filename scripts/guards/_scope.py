"""Which function every node of a module sits in, by qualified name, and which strings are
docstrings.

The one home for each walk. A guard whose exemptions are keyed by file and function (the
records guard, D151 decision 7; the projection guard) names a site by the
function it is in, and two walkers would name one site two ways. A guard matching string
constants skips prose, and two readings of "docstring" would skip different strings (the globs,
effects and edits guards). Standard library only, as every guard is, and `_`-prefixed so the
runner does not take it for a guard.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

__all__ = ["MODULE", "docstrings", "functions"]

MODULE = "<module>"
"""The scope of a node outside every function and class."""


def functions(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Every node with the qualified name of the function it sits in; `<module>` outside
    any. A nested function is its own scope, named through its parents."""

    def walk(node: ast.AST, scope: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = child.name if scope == MODULE else f"{scope}.{child.name}"
                yield inner, child
                yield from walk(child, inner)
            else:
                yield scope, child
                yield from walk(child, scope)

    yield from walk(tree, MODULE)


def docstrings(tree: ast.Module) -> set[int]:
    """The ids of every docstring constant: a module's, class's or function's first statement,
    and an attribute docstring, the bare string after an assignment, as this repository writes
    them. Prose may show what code it describes spells; it is not that code."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = node.body
        for i, stmt in enumerate(body):
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
                and (i == 0 or isinstance(body[i - 1], ast.Assign | ast.AnnAssign))
            ):
                out.add(id(stmt.value))
    return out
