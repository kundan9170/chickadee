"""Layering, the public error surface, and the identity of types exchanged at seams.

Three claims that are each stated in one place and checked in none, because each spans two
packages and every test directory under `packages/core/tests/` maps to exactly one.

The seam half is not hypothetical. D55 exists because contract 11 *had been defined twice
with different members* — `18-engine` now imports `GitHandoffChoice` rather than restating
it. A duplicated type passes both sides' unit tests and fails only where they meet, which
in this repository is Phase 3. Asserting identity now is what makes that failure impossible
rather than deferred.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import typing

import pytest

import chickadee
from chickadee import errors

CORE_SRC = pathlib.Path(chickadee.__file__).resolve().parent


def _module_files(package: str) -> list[pathlib.Path]:
    root = CORE_SRC / package
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _package_of(path: pathlib.Path) -> str:
    """The dotted package a file lives in, for resolving relative imports."""
    rel = path.relative_to(CORE_SRC)
    # The same for both file kinds: a module's package is the directory holding it, and
    # `foo/__init__.py` *is* `foo`, whose parent parts are likewise everything but the last.
    return ".".join(["chickadee", *rel.parts[:-1]])


def _imported_modules(tree: ast.Module, path: pathlib.Path) -> set[str]:
    """Every `chickadee.*` module this file imports, resolved to a canonical dotted name.

    Three forms have to be normalised or the check has a hole wide enough to drive a
    forbidden dependency through:

    - `import chickadee.executor` — already canonical.
    - `from chickadee import executor` — the node's `module` is only `chickadee`, so
      recording that alone never matches `chickadee.executor`. Each imported name is
      therefore also emitted as `module.name`. That over-approximates when the name is a
      class rather than a submodule, which is harmless: `chickadee.executor.Executor`
      matches the same prefix the module does, and nothing else in the table can collide.
    - `from..executor import Executor` — relative, and dropping it (as the first version
      did) means a sibling import is invisible. Resolved against the importing file's own
      package.

    Static, not runtime, for the reason `test_boundaries.py` gives: a runtime check sees
    only the import graph the run happened to execute.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            base = node.module or ""
        else:
            package = _package_of(path).split(".")
            trimmed = package[: len(package) - (node.level - 1)] or ["chickadee"]
            base = ".".join([*trimmed, node.module] if node.module else trimmed)
        if not base:
            continue
        found.add(base)
        found.update(f"{base}.{a.name}" for a in node.names)
    return {m for m in found if m.startswith("chickadee")}


# --- D70 layering -----------------------------------------------------------

FORBIDDEN_IMPORTS = {
    "isolation": ("chickadee.executor",),
    "process": ("chickadee.executor", "chickadee.isolation"),
    "graph": ("chickadee.client", "chickadee.adapters"),
}
"""Who may not import whom, and the decision that says so.

- `isolation` and `process` do not import `executor` (stated verbatim).
- `graph` calls no model; an import is the step before the call.

The `verifier` and `packer` rows retired with their packages.
"""


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS))
def test_a_package_does_not_import_what_it_sits_above(package: str) -> None:
    forbidden = FORBIDDEN_IMPORTS[package]
    offenders: list[str] = []
    for path in _module_files(package):
        imported = _imported_modules(ast.parse(path.read_text(encoding="utf-8")), path)
        for name in sorted(imported):
            if any(name == f or name.startswith(f + ".") for f in forbidden):
                offenders.append(f"{path.relative_to(CORE_SRC).as_posix()} -> {name}")
    assert not offenders, (
        f"`{package}` imports a package it sits above: {offenders}. See FORBIDDEN_IMPORTS "
        "for which decision states each edge."
    )


def test_the_layering_table_names_real_packages() -> None:
    """A rule about a package that was renamed away is a rule that stopped applying without
    anyone noticing."""
    missing = [p for p in FORBIDDEN_IMPORTS if not (CORE_SRC / p).is_dir()]
    assert not missing, f"FORBIDDEN_IMPORTS names packages that do not exist: {missing}"


# --- EN-3: the public error surface is not shadowed -------------------------

PUBLIC_ERRORS = {
    name
    for name, obj in inspect.getmembers(errors, inspect.isclass)
    if issubclass(obj, errors.ChickadeeError)
}

KNOWN_SHADOWS: dict[str, str] = {}
"""Classes elsewhere in `core` that reuse a public error's name without being catchable as
one. Enumerated so a **new** shadow fails this test.

**Empty since 2026-09-12, and the two that were here are worth recording.** Both were
`ConfigError`: `config.py` defined a `ValueError` and `client/errors.py` a
`ClientError`, so three unrelated classes shared the name `18-engine` §5 specifies as *one*
public class, and a host writing the documented `except ConfigError` caught a third of them.

They were left in place because the facade mapping that would translate them does not exist
yet, and each module belongs to another task. That reasoning held right up until check 12
showed the same name colliding three ways in one codebase — at which point the hazard was no
longer "a thing T-3.4 will meet", it was a public contract the code could not honour. The
fix: `errors.ConfigError` gained `ValueError` in its bases (pydantic collects only
`ValueError` out of a validator, which is why `config.py` had its own), and the transport one
became `CredentialError`, which is what both its raise sites actually are.
"""


def _class_defs(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]


def _dotted(path: pathlib.Path) -> str:
    rel = path.relative_to(CORE_SRC).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(["chickadee", *parts])


def test_the_public_hierarchy_is_not_shadowed_by_an_uncatchable_twin() -> None:
    """A host sees these classes and nothing else. A same-named class that is not a
    `ChickadeeError` defeats that by looking like one at the call site."""
    shadows: dict[str, str] = {}
    for path in sorted(p for p in CORE_SRC.rglob("*.py") if "__pycache__" not in p.parts):
        module = _dotted(path)
        if module == "chickadee.errors":
            continue
        for name in _class_defs(path):
            if name not in PUBLIC_ERRORS:
                continue
            obj = getattr(importlib.import_module(module), name, None)
            if isinstance(obj, type) and not issubclass(obj, errors.ChickadeeError):
                shadows[f"{module}.{name}"] = "not a ChickadeeError"

    unexpected = sorted(set(shadows) - set(KNOWN_SHADOWS))
    assert not unexpected, (
        f"New shadow of a public error name: {unexpected}. A host writing `except "
        "chickadee.errors.X` will not catch it. Either derive from `ChickadeeError`, rename "
        "it, or add it to KNOWN_SHADOWS with the Issue that tracks the mapping."
    )


def test_the_recorded_shadows_are_still_real() -> None:
    """If one gets fixed, this file should stop claiming it is a hazard."""
    stale: list[str] = []
    for dotted in KNOWN_SHADOWS:
        module, _, name = dotted.rpartition(".")
        obj = getattr(importlib.import_module(module), name, None)
        if obj is None or issubclass(obj, errors.ChickadeeError):
            stale.append(dotted)
    assert not stale, (
        f"KNOWN_SHADOWS lists entries that are no longer shadows: {stale}. Remove them — a "
        "stale exception list reads as a defect that still exists."
    )


# --- seam identity: one type, imported, never restated ----------------

SEAMS = [
    # IsolationSpec and GitHandoffChoice: workspace produces, handoff consumes.
    ("chickadee.handoff", "describe", "spec", "chickadee.schema.isolation"),
    ("chickadee.handoff", "finalize", "choice", "chickadee.schema.isolation"),
    # SandboxLevel: isolation reports it, the executor records it on `ExecResult.sandbox`,
    # `schema/` owns it. It WAS defined twice with identical members, which `==` hid and
    # `is` did not — see `schema/execution.py`. This row is what stops it coming back.
    (
        "chickadee.isolation.base",
        "SandboxBackend.level",
        "return",
        "chickadee.schema.execution",
    ),
]
"""(module, callable, parameter or `"return"`, module that owns the type).

Each row is a value one component hands another, checked from **both** ends where both
exist, so a divergence between producer and consumer fails here rather than where they meet.
The Applier-to-Verifier rows retired with both.

The assertion is *identity* — the annotation resolves to the same class object the owning
module exports, not to a structurally similar local copy. That is the failure D55 records:
contract 11 defined twice with different members, found only because someone read both
definitions. The lesson: the contract lives in one frozen place and both sides import it.
"""


def _resolve(module: str, dotted_callable: str) -> object:
    obj: object = importlib.import_module(module)
    for part in dotted_callable.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize(
    ("consumer", "func", "param", "owner"), SEAMS, ids=[f"{c}:{f}" for c, f, _, _ in SEAMS]
)
def test_a_seam_type_is_the_owning_module_s_own_class(
    consumer: str, func: str, param: str, owner: str
) -> None:
    target = _resolve(consumer, func)
    hints = typing.get_type_hints(target)
    key = "return" if param == "return" else param
    assert key in hints, f"{consumer}:{func} has no annotated `{param}`"

    annotated = hints[key]
    origin = typing.get_origin(annotated) or annotated
    owning = importlib.import_module(owner)
    exported = getattr(owning, getattr(origin, "__name__", ""), None)

    assert exported is not None, (
        f"{owner} does not export `{getattr(origin, '__name__', origin)}`, which "
        f"{consumer}:{func} annotates `{param}` with. The seam names a type its owner does "
        "not publish."
    )
    assert exported is origin, (
        f"{consumer}:{func}'s `{param}` is a different class object from {owner}'s. Two "
        "definitions of one contract pass both sides' unit tests and fail where they meet "
        "(D55). Import the type; do not restate it."
    )


# --- the import resolver itself ---------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import chickadee.executor", "chickadee.executor"),
        # Both evasions found in review on PR #137: the first recorded only `chickadee`,
        # and dropped relative imports entirely.
        ("from chickadee import executor", "chickadee.executor"),
        ("from ..executor import Executor", "chickadee.executor"),
        ("from chickadee.executor import Executor", "chickadee.executor"),
    ],
    ids=["absolute-module", "from-package", "relative", "from-module"],
)
def test_the_import_resolver_canonicalises_every_form(source: str, expected: str) -> None:
    inside = CORE_SRC / "box" / "report.py"
    assert expected in _imported_modules(ast.parse(source), inside)


def test_the_import_resolver_ignores_the_standard_library() -> None:
    found = _imported_modules(
        ast.parse("import ast\nfrom pathlib import Path"), CORE_SRC / "x.py"
    )
    assert found == set()
