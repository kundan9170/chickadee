"""Where a profile's executable is found (`profiles/tools.find_tool`), and the settings
a language server is answered with (`LspSpec.configuration`, `graph/conversation.client_reply`).

What these pin: the repository's `.venv` first, then `PATH`, then, for the harness's own
servers only, Chickadee's own environment; the path is absolute; `python` is `python3` where
only that is installed, place by place. Python's profile names the
shipped server and asks pyright's `standard` checks, answered by dotted section, `null` for a
section it does not configure.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sysconfig
from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.graph.conversation import client_reply
from chickadee.profiles.loader import load_profile
from chickadee.profiles.tools import find_tool


def _with_git(first: Path) -> str:
    """`first`, then git's directory: the lookup asks git whether a `.venv` is tracked."""
    found = shutil.which("git")
    assert found is not None
    return os.pathsep.join((str(first), str(Path(found).parent)))


def _tool(where: Path, name: str) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    made = where / name
    made.write_text("#!/bin/sh\n", encoding="utf-8")
    made.chmod(made.stat().st_mode | stat.S_IXUSR)
    return made


@pytest.mark.skipif(os.name == "nt", reason="POSIX executables")
def test_the_repositorys_venv_comes_before_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    on_path = _tool(tmp_path / "path", "srv")
    pinned = _tool(tmp_path / "repo" / ".venv" / "bin", "srv")
    monkeypatch.setenv("PATH", _with_git(on_path.parent))

    assert find_tool("srv", tmp_path / "repo") == pinned
    assert find_tool("srv") == on_path
    assert find_tool("srv", tmp_path / "elsewhere") == on_path


@pytest.mark.skipif(os.name == "nt", reason="POSIX executables")
def test_chickadees_own_environment_only_for_its_own_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installed as a tool, Chickadee's environment is never on `PATH`: its shipped servers are
    found there, and only when asked for one (`own`)."""
    own = _tool(tmp_path / "own" / "bin", "srv")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(sysconfig, "get_path", lambda _name: str(own.parent))

    assert find_tool("srv", tmp_path) is None
    assert find_tool("srv", tmp_path, own=True) == own
    assert find_tool("srv", own=True) is not None and find_tool("srv", own=True).is_absolute()  # type: ignore[union-attr]


@pytest.mark.skipif(os.name == "nt", reason="POSIX executables")
def test_python_is_python3_where_only_that_is_installed_place_by_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    three = _tool(tmp_path / "path", "python3")
    git = shutil.which("git")
    assert git is not None
    alone = tmp_path / "git-alone"  # git's own directory may hold a `python` of this machine's
    alone.mkdir()
    (alone / "git").symlink_to(git)
    monkeypatch.setenv("PATH", os.pathsep.join((str(three.parent), str(alone))))
    assert find_tool("python") == three
    pinned = _tool(tmp_path / "repo" / ".venv" / "bin", "python")
    assert find_tool("python", tmp_path / "repo") == pinned, "the repository's, before PATH's"
    assert find_tool("python3", tmp_path / "repo") == three, "an alternate is only python's"
    assert find_tool("pytest") is None


def test_pythons_server_is_the_shipped_one_asked_for_pyrights_standard_checks() -> None:
    lsp = load_profile("python").lsp
    assert lsp.server is not None and lsp.server.argv[0] == "basedpyright-langserver"
    assert find_tool("basedpyright-langserver", own=True) is not None, "a core dependency"
    assert lsp.configuration("basedpyright") == {"analysis": {"typeCheckingMode": "standard"}}
    assert lsp.configuration("basedpyright.analysis.typeCheckingMode") == "standard"
    assert lsp.configuration("python") is None
    assert lsp.configuration("basedpyright.nothing.here") is None


def test_workspace_configuration_is_answered_item_by_item() -> None:
    lsp = load_profile("python").lsp
    framed = client_reply(
        7,
        {"items": [{"section": "python"}, {"section": "basedpyright"}, {}]},
        lsp.configuration,
    )
    body = json.loads(framed.split(b"\r\n\r\n", 1)[1])
    assert body == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": [None, {"analysis": {"typeCheckingMode": "standard"}}, None],
    }
    # With nothing configured, every item is the server's own default.
    bare = json.loads(client_reply(8, {"items": [{"section": "x"}]}).split(b"\r\n\r\n", 1)[1])
    assert bare["result"] == [None]


@pytest.mark.skipif(os.name == "nt", reason="POSIX executables")
def test_a_venv_the_repository_brought_is_never_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clone can commit `.venv/bin/ruff`, and a server starts as soon as the repository is
    opened: only an environment git tracks nothing of, one made here, is looked in."""
    on_path = _tool(tmp_path / "path", "srv")
    monkeypatch.setenv("PATH", _with_git(on_path.parent))
    root = init_repo(tmp_path / "cloned")
    shipped = _tool(root / ".venv" / "bin", "srv")
    git(root, "add", "--force", ".venv")
    git(root, "commit", "-q", "-m", "a venv in the repository")
    assert find_tool("srv", root) == on_path, "the tracked one is passed over"
    git(root, "rm", "-q", "-r", "--cached", ".venv")
    git(root, "commit", "-q", "-m", "untracked again")
    assert find_tool("srv", root) == shipped, "made here: used first"
