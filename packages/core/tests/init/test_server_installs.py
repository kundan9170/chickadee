"""Each language's servers on this machine, and their install (`init/servers.py`).

What these pin: every server a profile names is listed once, with the languages that share
it; a missing one with an installer is installable only when the tool the installer runs is
here; an install runs with the network's environment, into Chickadee's servers directory, and
counts as done only when the server is then found there.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest
from sandboxes import Scripted

from chickadee.init.servers import ServerNeed, install, server_needs
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.tools import find_tool, servers_dir


@pytest.fixture(autouse=True)
def _own_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_every_server_is_listed_once_with_the_languages_it_serves() -> None:
    needs = {n.server: n for n in server_needs(ProfileSet.load())}
    assert needs["clangd"].languages == ("c", "cpp")
    assert needs["typescript-language-server"].languages == ("javascript", "typescript")
    assert {"basedpyright-langserver", "ruff", "gopls", "rust-analyzer"} <= set(needs)
    gopls = needs["gopls"].install
    assert gopls is not None and f"GOBIN={servers_dir()}/bin" in gopls
    assert needs["jdtls"].install is None, "no one portable command: installed by the person"


def test_a_missing_installer_tool_is_named_and_nothing_is_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    [go] = [n for n in server_needs(ProfileSet.load()) if n.server == "gopls"]
    if go.found is not None:
        pytest.skip("gopls is installed in this environment")
    assert go.needs == "go" and not go.installable
    assert install(go).said == "installing gopls needs go, not found"


@pytest.mark.skipif(os.name == "nt", reason="POSIX executables")
def test_an_install_counts_only_when_the_server_is_then_found() -> None:
    need = ServerNeed(("go",), "fake-ls", None, ("installer", "fake-ls"), None)

    class Lands(Scripted):
        def run(self, argv: tuple[str, ...], **kwargs: Any) -> Any:
            made = servers_dir() / "bin" / "fake-ls"
            made.write_text("#!/bin/sh\n", encoding="utf-8")
            made.chmod(made.stat().st_mode | stat.S_IXUSR)
            self.env = dict(kwargs["env"])
            self.cwd = kwargs["cwd"]
            return super().run(argv, **kwargs)

    nothing = install(need, Scripted())
    assert not nothing.ok and "is not found after it" in nothing.said

    lands = Lands()
    done = install(need, lands)
    assert done.ok and done.said == f"installed at {servers_dir() / 'bin' / 'fake-ls'}"
    assert find_tool("fake-ls", own=True) == servers_dir() / "bin" / "fake-ls"
    assert lands.cwd == servers_dir()


def test_a_failed_install_says_why() -> None:
    need = ServerNeed(("go",), "fake-ls", None, ("installer",), None)
    failed = install(need, Scripted(b"no such module", exit_code=1))
    assert not failed.ok and failed.said == "exit 1: no such module"
