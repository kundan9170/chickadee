"""`chickadee.*` is imported only by `host.py`: every other module of the
client speaks the wire and nothing else, so a frontend that attaches to a running server
loads no engine. Checked twice: what the source says it imports, and what importing it loads."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "chickadee_wire"
HOST = "host.py"

WIRE_CONTRACT = {
    "chickadee_serve.frames",
    "chickadee_serve.routes",
    "chickadee_serve.bind",
    "chickadee_serve.refusals",
}
"""The server's modules a client may import: the contract, the bind types it addresses, and
the reasons a start is refused."""


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and not node.level:
            found.add(node.module)
    return found


def _modules() -> list[Path]:
    return sorted(p for p in SRC.glob("*.py") if p.name != HOST)


def test_only_host_imports_the_engine() -> None:
    assert any(m.startswith("chickadee.") for m in _imports(SRC / HOST))
    trespass = {
        p.name: sorted(
            m
            for m in _imports(p)
            if m == "chickadee"
            or m.startswith("chickadee.")
            or (m.startswith("chickadee_serve") and m not in WIRE_CONTRACT)
        )
        for p in _modules()
    }
    assert {name: mods for name, mods in trespass.items() if mods} == {}


@pytest.mark.parametrize("module", [p.stem for p in _modules()])
def test_importing_a_client_module_loads_no_engine(module: str) -> None:
    probe = textwrap.dedent(
        f"""
        import sys
        import chickadee_wire.{module}
        print(",".join(sorted(m for m in sys.modules if m.split(".")[0] == "chickadee")))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"chickadee_wire.{module} pulled in engine modules: {out}"
