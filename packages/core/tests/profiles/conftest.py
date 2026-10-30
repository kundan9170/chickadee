"""A builtin-profile directory under tmp, so tests never depend on shipped file contents."""

from __future__ import annotations

import pathlib

import pytest

PYTHON_TOML = """
name = "python"
extensions = [".py", ".pyi"]
tree_sitter_grammar = "python"

[syntax]
mode = "inproc"

[tests]
globs = ["tests/**", "test_*.py"]
runner = "pytest -q {selector}"
timeout_s = 120

[build]
compile_cmd = ""
import_smoke = "python -c 'import {module}'"

[deps]
manifests = ["pyproject.toml"]
install_cmd = "uv add {name}"

[conventions]
file = "conventions/python.md"
"""


@pytest.fixture
def builtin(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "profiles"
    (d / "conventions").mkdir(parents=True)
    (d / "python.toml").write_text(PYTHON_TOML)
    (d / "conventions" / "python.md").write_text("# python conventions\n")
    return d


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r
