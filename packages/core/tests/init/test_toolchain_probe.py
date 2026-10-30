"""`init`'s toolchain probe finds a profile's commands as the project's do
(`init/toolchain.version_of` over `profiles/tools.find_tool`).

What these pin: `python` is found as `python3` where only that is installed, and says so; a
tool in the repository's own `.venv` is found though `PATH` lacks it; Chickadee's own
environment is never the project's; nothing found is None.
"""

from __future__ import annotations

import os
import shutil
import stat
import sysconfig
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.init.toolchain import version_of

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX executables")


def _tool(where: Path, name: str, says: str) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    made = where / name
    made.write_text(f'#!/bin/sh\necho "{says}"\n', encoding="utf-8")
    made.chmod(made.stat().st_mode | stat.S_IXUSR)
    return made


def _path(*dirs: Path) -> str:
    """`dirs`, then a directory holding git alone: the lookup asks git whether a `.venv` is
    tracked, and git's own directory may hold a `python` this machine has."""
    git = shutil.which("git")
    assert git is not None
    alone = dirs[0].parent / "git-alone" if dirs else Path("/nonexistent")
    alone.mkdir(parents=True, exist_ok=True)
    if not (alone / "git").exists():
        (alone / "git").symlink_to(git)
    return os.pathsep.join((*map(str, dirs), str(alone)))


def test_python_is_found_as_python3_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_ = tmp_path / "bin"
    _tool(bin_, "python3", "Python 3.12.3")
    monkeypatch.setenv("PATH", _path(bin_))
    assert version_of("python") == "Python 3.12.3 (as python3)"
    _tool(bin_, "python", "Python 3.13.0")
    assert version_of("python") == "Python 3.13.0", "`python` itself, where there is one"


def test_the_repositorys_own_venv_is_looked_in_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    _tool(repo / ".venv" / "bin", "pytest", "pytest 8.3.0")
    _tool(tmp_path / "bin", "pytest", "pytest 7.0.0")
    monkeypatch.setenv("PATH", _path(tmp_path / "empty"))
    assert version_of("pytest", repo) == "pytest 8.3.0", "a uv project's pytest is in its .venv"
    assert version_of("pytest") is None, "not on PATH, and no repository to look in"
    monkeypatch.setenv("PATH", _path(tmp_path / "bin"))
    assert version_of("pytest", repo) == "pytest 8.3.0", "the repository's own, over PATH's"


def test_chickadees_own_environment_is_never_the_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = _tool(tmp_path / "own" / "bin", "pytest", "pytest 9.9.9")
    monkeypatch.setattr(sysconfig, "get_path", lambda _name: str(own.parent))
    monkeypatch.setenv("PATH", _path(tmp_path / "empty"))
    assert version_of("pytest", tmp_path) is None
