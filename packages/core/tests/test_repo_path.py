"""`paths.repo_path`: a path as a model wrote it, spelled as the repository names it.
Lexical only; everything it does not normalise comes back exactly as written."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.paths import repo_path

ROOT = Path("/abs/path/to/repo")


@pytest.mark.parametrize(
    ("written", "named"),
    [
        ("./calc.py", "calc.py"),
        ("./services/billing/api/v2/invoice.py", "services/billing/api/v2/invoice.py"),
        ("src/./app/user.py", "src/app/user.py"),
        ("src//app/user.py", "src/app/user.py"),
        ("/abs/path/to/repo/src/app/user.py", "src/app/user.py"),
        ("calc.py", "calc.py"),
        ("src/app/", "src/app"),
    ],
)
def test_what_is_normalised(written: str, named: str) -> None:
    assert repo_path(written, ROOT) == named


@pytest.mark.parametrize(
    "written",
    [
        "v1.2/config.prod.yaml",
        "My Docs/Ünïcode Name.PY",
        ".github/workflows/ci.yml",
        "../outside.py",
        "src/../calc.py",
        "/etc/passwd",
        "/abs/path/to/repository-other/x.py",
        "src\\app\\user.py",
        "./",
        ".",
        "/abs/path/to/repo",
    ],
)
def test_what_is_left_as_written(written: str) -> None:
    assert repo_path(written, ROOT) == written


def test_without_a_root_an_absolute_path_is_left_alone() -> None:
    assert repo_path("/abs/path/to/repo/calc.py") == "/abs/path/to/repo/calc.py"
    assert repo_path("./calc.py") == "calc.py"


def test_a_root_reached_through_a_link_is_the_same_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    assert repo_path(f"{real}/calc.py", link) == "calc.py"
    assert repo_path(f"{link}/calc.py", link) == "calc.py"
