"""The REPL speaks the wire and nothing else: no module of the CLI
imports `chickadee.*` except the commands that need no server, which reach the engine on
purpose (`init`, `setup`, `doctor`, `replay`, `gc`, `trust`, `hooks`, `inspect`), and the
TUI's preferences file, read and written through the storage module. Hosting an
engine goes through `chickadee_wire.host`, imported inside the command that hosts. Checked
twice: what the source says it imports, and what importing it loads."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "chickadee_cli"

ENGINE_SIDE = {
    "init.py",
    "setup.py",
    "doctor.py",
    "replay.py",
    "gc.py",
    "trust.py",
    "hooks.py",
    "inspector.py",
    "tui_prefs.py",
}
"""Commands that run without a server, against the engine directly (a
journal folded or its old blobs removed, `09-journal.md` §3-§4; a repository's executable
config accepted, `31-state.md` §6; the hooks it binds listed, D227; the runs and sessions read
for `chickadee inspect`, D224; the TUI's `tui.toml`, the person's full-screen choice written
through the facade's `write_user_file`, which the screen itself never touches)."""

SERVER_SIDE = {
    "chickadee_serve.frames",
    "chickadee_serve.routes",
    "chickadee_serve.bind",
    "chickadee_serve.refusals",
    "chickadee_serve.__main__",  # `chickadee serve`, imported inside that command only
}
"""The server's modules the CLI may name: the contract, the bind types, and its entry."""


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None and not node.level:
            found.add(node.module)
    return found


def _modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if p.name not in ENGINE_SIDE)


def _dotted(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_only_the_serverless_commands_import_the_engine() -> None:
    trespass = {
        _dotted(p): sorted(
            m
            for m in _imports(p)
            if m == "chickadee"
            or m.startswith("chickadee.")
            or (m.startswith("chickadee_serve") and m not in SERVER_SIDE)
        )
        for p in _modules()
    }
    assert {name: mods for name, mods in trespass.items() if mods} == {}


@pytest.mark.parametrize("module", [_dotted(p) for p in _modules()])
def test_importing_a_repl_module_loads_no_engine(module: str) -> None:
    probe = textwrap.dedent(
        f"""
        import sys
        import {module}
        print(",".join(sorted(m for m in sys.modules if m.split(".")[0] == "chickadee")))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"{module} pulled in engine modules: {out}"
