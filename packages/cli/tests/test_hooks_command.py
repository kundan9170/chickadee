"""`chickadee hooks`: the points, the bounds, and each
configured entry over the defaults with what binding it would do now, from the config alone.

What these pin: with nothing configured, the default `owed_review` is bound; a repository's
script is inert until the person accepts it, and says how; a repository's entry under a
default's `(event, name)` switches it off; and a hook script appears in `chickadee trust`, since
it is part of the executable subset whose digest the person accepts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from engines import config_toml
from repos import init_repo

from chickadee.hooks import show
from chickadee.isolation.types import SandboxLevel
from chickadee_cli import hooks, trust

_SCRIPT = '[[hooks]]\nevent = "after_tool"\nname = "lint"\ncommand = ["./lint.sh"]\n'
_OFF = '[[hooks]]\nevent = "before_stop"\nname = "owed_review"\nenabled = false\n'


@pytest.fixture
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in [n for n in os.environ if n.startswith("CHICKADEE_")]:
        monkeypatch.delenv(name)
    home = tmp_path / "xdg" / "chickadee"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config_toml())
    return home


def _repo(tmp_path: Path, config: str = "") -> Path:
    repo = init_repo(tmp_path / "repo", {"app.py": "x = 1\n", "lint.sh": "exit 0\n"})
    (repo / ".chickadee").mkdir(exist_ok=True)
    if config:
        (repo / ".chickadee" / "config.toml").write_text(config)
    return repo


def _shown(repo: Path) -> tuple[int, str]:
    said: list[str] = []
    return hooks.run(repo, say=said.append), "\n".join(said)


def test_the_points_and_the_default_are_shown(tmp_path: Path, xdg: Path) -> None:
    code, out = _shown(_repo(tmp_path))
    assert code == 0
    assert "before_tool  level 1" in out and "(Claude Code: PreToolUse)" in out
    assert "round_ended  level 2" in out
    assert "before_stop owed_review: use owed_review role athena [user] bound" in out


def _sealed(monkeypatch: pytest.MonkeyPatch, level: SandboxLevel) -> None:
    monkeypatch.setattr(show, "seals", lambda backend: level is not SandboxLevel.NONE)


def test_a_repositorys_script_is_inert_until_accepted(
    tmp_path: Path, xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sealed(monkeypatch, SandboxLevel.NATIVE)
    code, out = _shown(_repo(tmp_path, _SCRIPT))
    assert code == 0
    assert "after_tool lint: command ['./lint.sh'] [project] inert" in out
    assert "`chickadee trust`" in out


def test_a_persons_script_where_nothing_seals_is_refused(
    tmp_path: Path, xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sealed(monkeypatch, SandboxLevel.NONE)
    config = xdg / "config.toml"
    config.write_text(config.read_text() + "\n" + _SCRIPT)
    _, out = _shown(_repo(tmp_path))
    row = "after_tool lint: command ['./lint.sh'] [user] refused: hook 'lint' is a script"
    assert row in out
    assert "no sandbox backend" in out


def test_a_repositorys_unaccepted_script_is_inert_even_where_nothing_seals(
    tmp_path: Path, xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It never runs, so it never stops the clone from running for want of a sandbox."""
    _sealed(monkeypatch, SandboxLevel.NONE)
    _, out = _shown(_repo(tmp_path, _SCRIPT))
    assert "after_tool lint: command ['./lint.sh'] [project] inert" in out


def test_a_repository_switches_a_default_off(tmp_path: Path, xdg: Path) -> None:
    _, out = _shown(_repo(tmp_path, _OFF))
    assert "before_stop owed_review: use owed_review role athena" in out
    assert out.rstrip().endswith("off")


def test_a_hook_script_is_part_of_what_trust_shows(tmp_path: Path, xdg: Path) -> None:
    said: list[str] = []
    trust.run(_repo(tmp_path, _SCRIPT), ask=None, say=said.append)
    assert "lint.sh" in "\n".join(said)


def test_outside_a_repository_it_says_so(tmp_path: Path, xdg: Path) -> None:
    code, out = _shown(tmp_path)
    assert code == 4 and "not a repository" in out
