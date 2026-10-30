"""`load_host_config`: a host's repository, its layers resolved and its trust decided, the one
way every host builds an engine from."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from engines import config_toml
from repos import init_repo

from chickadee.engine import PacketEngine
from chickadee.state.load import load_host_config
from chickadee.state.merge import MergeError
from chickadee.state.roots import NotARepository


@pytest.fixture
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The engine config under a temporary `$XDG_CONFIG_HOME`, no `CHICKADEE_*` set."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in [n for n in os.environ if n.startswith("CHICKADEE_")]:
        monkeypatch.delenv(name)
    home = tmp_path / "xdg" / "chickadee"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config_toml())
    return home


def test_a_repository_loads_from_any_directory_inside_it(tmp_path: Path, xdg: Path) -> None:
    repo = init_repo(tmp_path / "repo", {"src/app.py": "x = 1\n"})
    host = load_host_config(repo / "src")
    assert host.repo == repo
    assert host.roots.repo_local == (repo / ".git" / "chickadee").resolve()
    assert host.engine.spend.hard_cost_usd == 5.0
    assert host.trust.engine is host.engine, "the trust is taken for this config"
    assert host.trust.accepted, "a repository that declares nothing executable asks nothing"


def test_the_engine_takes_what_it_loads(tmp_path: Path, xdg: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    host = load_host_config(repo)
    PacketEngine(host.repo, host.engine, trust=host.trust).close()


def test_outside_a_repository_is_said(tmp_path: Path, xdg: Path) -> None:
    with pytest.raises(NotARepository):
        load_host_config(tmp_path)


def test_a_bad_key_is_the_loaders_refusal_unchanged(tmp_path: Path, xdg: Path) -> None:
    repo = init_repo(tmp_path / "repo")
    (repo / ".chickadee").mkdir()
    (repo / ".chickadee" / "config.toml").write_text("[spend]\nhard_cost_usd = 1.0\n")
    with pytest.raises(MergeError):
        load_host_config(repo)
