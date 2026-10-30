"""Surfaces import only `chickadee.engine` and `chickadee.schema`.

The other half of `tests/test_boundaries.py`. That one stops core reaching into a surface;
this one stops a surface reaching past the facade into core's private modules.

EN-1 is why the facade is worth anything: components are private so the engine can enforce
an ordering no host can skip — route before plan, lease before mutate, gates after every
patch. A CLI holding a `Verifier` directly could call it in an order the invariants forbid,
and nothing would stop it. That guarantee is only as good as this check.

Static, for the same reason the sibling test is: a runtime scan of `sys.modules` sees only
the paths a test happened to execute, so a lazy import inside an error branch stays
invisible until the error happens in front of a user.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGES = pathlib.Path(__file__).resolve().parents[3]
SURFACES = ("cli", "eval", "mcp-server")

# The whole public surface of core, as §6 defines it.
PUBLIC = {"chickadee.engine", "chickadee.errors", "chickadee.schema"}


def _surface_sources() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for name in SURFACES:
        src = PACKAGES / name / "src"
        if src.is_dir():
            out.extend(sorted(src.rglob("*.py")))
    return out


def _chickadee_imports(tree: ast.Module) -> set[str]:
    """Every `chickadee` module a file reaches, whichever way it spells it.

    `from chickadee import journal` is the case worth spelling out: the module being
    imported *from* is the public root, so recording only that would pass, while the name
    actually bound is a private submodule. The names have to be resolved against the
    module they come from, or the check waves through the shortest route around itself.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.split(".")[0] == "chickadee")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "chickadee"
        ):
            found.add(node.module)
            # `from chickadee import x` binds `chickadee.x`; `from chickadee.schema import
            # Packet` binds a name, not a module. Only the former can smuggle a submodule.
            if node.module == "chickadee":
                found.update(f"chickadee.{a.name}" for a in node.names)
    return found


def _is_public(module: str) -> bool:
    # `chickadee.schema.packet` is fine — the package is the surface, and its submodules
    # are re-exported from it. `chickadee.journal.writer` is not.
    return (
        any(module == p or module.startswith(p + ".") for p in PUBLIC) or module == "chickadee"
    )


@pytest.mark.parametrize(
    "path", _surface_sources() or [None], ids=lambda p: p.name if p else "none-yet"
)
def test_surfaces_import_only_the_facade(path: pathlib.Path | None) -> None:
    if path is None:
        pytest.skip("no surface sources yet; the CLI and eval rig arrive in later phases")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    private = sorted(m for m in _chickadee_imports(tree) if not _is_public(m))
    assert not private, (
        f"{path.name} imports {private}; surfaces may use only {sorted(PUBLIC)} "
        "(EN-1). Reaching a private module makes it API by accident, and lets a host "
        "call components in an order the invariants forbid."
    )


@pytest.mark.parametrize(
    "source, private",
    [
        ("from chickadee import journal", "chickadee.journal"),
        ("from chickadee.journal import Journal", "chickadee.journal"),
        ("import chickadee.journal.writer", "chickadee.journal.writer"),
        ("from chickadee import config, engine", "chickadee.config"),
    ],
)
def test_the_check_catches_every_spelling(source: str, private: str) -> None:
    """Guard the guard, positively: each of these reaches a private module by a different
    route, and the shortest one — importing a submodule by name off the public root —
    slipped through the first version of this test."""
    found = _chickadee_imports(ast.parse(source))
    assert private in found
    assert not _is_public(private)


@pytest.mark.parametrize(
    "source",
    [
        "from chickadee.schema import JournalEvent",
        "import chickadee.engine",
        "import chickadee",
    ],
)
def test_the_check_does_not_object_to_the_facade(source: str) -> None:
    assert all(_is_public(m) for m in _chickadee_imports(ast.parse(source)))


def test_the_facade_modules_exist() -> None:
    """Guard the guard: if `engine` or `errors` vanished, every check above would pass by
    finding nothing to object to."""
    import chickadee.engine
    import chickadee.errors
    import chickadee.schema

    assert chickadee.engine.PacketEngine
    assert chickadee.errors.HarnessBug
    assert chickadee.schema.JournalEvent
