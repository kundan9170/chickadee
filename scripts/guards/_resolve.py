"""Which local names reach `module.attr`, and which calls go through them.

The one home for this resolution. The storage guard (`storage.py`) counts durable-write sites
with it, and invariant 1's check (`packages/core/tests/invariants/test_invariant_spawn_and_
writes.py`) finds spawns and writers with it; D149 decision 10 requires the two to resolve
aliases the same way, and two copies would drift apart. Standard library only, as
every guard is, and `_`-prefixed so the runner does not take it for a guard.

Three binding forms are resolved, each something a person does out of habit rather than
cunning: an aliased module import, an aliased `from` import, and an assignment to a local
name. Assignment is resolved to a fixed point, so `a = subprocess.Popen; b = a` is covered,
including the chained form `a = b = subprocess.Popen` and a module alias written as an
assignment (`fs = os`).

**What this cannot promise.** `getattr(subprocess, "Popen")`, an `exec`, or a name assembled at
runtime defeats any static resolver, and chasing those would be an arms race against an author
who has commit access and therefore does not need an alias to break an invariant. This exists
to catch **drift and accident**: someone aliases a module for brevity and a monopoly quietly
stops being one. The bar is "the ordinary ways of doing this by mistake now fail".
"""

from __future__ import annotations

import ast
from typing import NamedTuple

__all__ = ["WRITE_MODES", "Binding", "bindings", "call_sites", "is_write_open", "name_uses"]

WRITE_MODES = ("w", "a", "x", "+")


class Binding(NamedTuple):
    """Every local name through which one `module.attr` can be reached."""

    modules: frozenset[str]
    """Names the module is reachable under — `os`, or `fs` after `import os as fs`."""
    direct: frozenset[str]
    """Names bound straight to the attribute — `move` after `from os import replace as
    move`, or `spawn` after `spawn = subprocess.Popen`."""


def bindings(tree: ast.Module, module: str, attr: str) -> Binding:
    """Resolve the local names through which `module.attr` can be reached."""
    modules = {module}
    direct: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.asname or a.name for a in node.names if a.name == module)
        elif isinstance(node, ast.ImportFrom) and node.module == module:
            direct.update(a.asname or a.name for a in node.names if a.name == attr)

    # `a = b = x` is one Assign with two Name targets, not two Assigns. Restricting to
    # `len(targets) == 1` dropped the chained form; collecting every Name on the left covers
    # it. Tuple unpacking (`a, b =...`) is left alone — the value is then a sequence, not
    # `module.attr`, and treating both names as the attribute is a lie.
    assignments = [
        (target.id, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    grew = True
    while grew:
        grew = False
        for target, value in assignments:
            # `fs = os` / `sp = subprocess`: the alias must join `modules` so `fs.replace(...)`
            # and `sp.Popen(...)` match as attribute calls. Direct names
            # (`spawn = subprocess.Popen`) stay in `direct`.
            if isinstance(value, ast.Name) and value.id in modules and target not in modules:
                modules.add(target)
                grew = True
            if target in direct:
                continue
            reaches = (
                isinstance(value, ast.Attribute)
                and value.attr == attr
                and isinstance(value.value, ast.Name)
                and value.value.id in modules
            ) or (isinstance(value, ast.Name) and value.id in direct)
            if reaches:
                direct.add(target)
                grew = True
    return Binding(frozenset(modules), frozenset(direct))


def call_sites(tree: ast.Module, module: str, attr: str) -> list[ast.Call]:
    """Every call of `module.attr` under any name this module bound for it."""
    bound = bindings(tree, module, attr)
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == attr
            and isinstance(func.value, ast.Name)
            and func.value.id in bound.modules
        ) or (isinstance(func, ast.Name) and func.id in bound.direct):
            sites.append(node)
    return sites


def name_uses(tree: ast.Module, module: str, attr: str) -> list[ast.expr]:
    """Every read of the constant `module.attr` (`os.O_APPEND`), through a module alias or a
    `from` import. A name assigned from it is not followed: `flags = os.O_APPEND` is one
    read, and a later `flags` is the same one again, not another."""
    modules = bindings(tree, module, attr).modules
    imported = {
        a.asname or a.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for a in node.names
        if a.name == attr
    }
    uses: list[ast.expr] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == attr
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
        ) or (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in imported
        ):
            uses.append(node)
    return uses


def is_write_open(node: ast.Call) -> bool:
    """`open(p)` reads; `open(p, "w")` writes; so does `path.open("a")`.

    Only the mode is read: the `mode=` keyword, else the second positional of `open(...)`,
    or the first or second of a method `.open(...)`, since `Path.open` takes the mode first
    and `io.open` second. A positional counts as the mode only when it is made of mode
    letters alone. Scanning every argument looked equivalent and was not:
    `open(path, "r", -1, "ascii")` is a *read*, and `"ascii"` contains an `"a"`; so is
    `archive.open("data.txt")`.
    """
    method = isinstance(node.func, ast.Attribute)
    candidates = node.args[:2] if method else node.args[1:2]
    mode = next((arg for arg in candidates if _is_mode(arg)), None)
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and any(m in mode.value for m in WRITE_MODES)
    )


_MODE_LETTERS = frozenset("rwxabt+U")


def _is_mode(arg: ast.expr) -> bool:
    return (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and bool(arg.value)
        and set(arg.value) <= _MODE_LETTERS
    )
