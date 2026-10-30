"""`chickadee trust`: the repository's executable
config and its `[[approval.grants]]`, each shown itemised and verbatim with its digest, and both
recorded on the person's yes, through the digests the engine checks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from engines import config_toml
from repos import init_repo

from chickadee.engine import repository_trust
from chickadee.state.load import load_host_config
from chickadee_cli import trust

_ROW = '[[commands]]\nprefix = ["deploy.sh"]\nclass = "push"\n'
_GRANT = '[[approval.grants]]\neffect = "command"\nclass = "push"\nprefix = ["git", "push"]\n'


@pytest.fixture
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for name in [n for n in os.environ if n.startswith("CHICKADEE_")]:
        monkeypatch.delenv(name)
    home = tmp_path / "xdg" / "chickadee"
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config_toml())
    return home


def _repo(tmp_path: Path, *, config: str = "", approval: str = "") -> Path:
    repo = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    (repo / ".chickadee").mkdir(exist_ok=True)
    if config:
        (repo / ".chickadee" / "config.toml").write_text(config)
    if approval:
        (repo / ".chickadee" / "approval.toml").write_text(approval)
    return repo


def _run(repo: Path, answer: str | None) -> tuple[int, str, list[str]]:
    said: list[str] = []
    asked: list[str] = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        assert answer is not None
        return answer

    code = trust.run(repo, ask=None if answer is None else ask, say=said.append)
    return code, "\n".join(said), asked


def test_both_parts_are_shown_verbatim_with_their_digests(tmp_path: Path, xdg: Path) -> None:
    repo = _repo(tmp_path, config=_ROW, approval=_GRANT)
    shown = repository_trust(repo)
    code, out, _ = _run(repo, "n")
    assert code == 0
    assert f"digest  {shown.executable.digest}  NOT accepted: inert" in out
    assert 'commands = [{"class": "push", "prefix": ["deploy.sh"]}]' in out
    assert f"digest  {shown.grants.digest}  NOT accepted: inert" in out
    assert '[0] {"class": "push", "effect": "command", "prefix": ["git", "push"]}' in out


def test_a_yes_accepts_both_and_the_engine_reads_them_accepted(
    tmp_path: Path, xdg: Path
) -> None:
    repo = _repo(tmp_path, config=_ROW, approval=_GRANT)
    assert not load_host_config(repo).trust.accepted
    code, out, asked = _run(repo, "y")
    assert code == 0
    assert asked == [f"accept {trust.EXECUTABLE} and {trust.APPROVAL} for this clone? [y/N] "]
    assert "accepted, recorded in" in out
    assert load_host_config(repo).trust.accepted, "the digest shown is the engine's"
    after = repository_trust(repo)
    assert after.grants.trusted and after.pending == ()


def test_anything_but_yes_accepts_nothing(tmp_path: Path, xdg: Path) -> None:
    repo = _repo(tmp_path, config=_ROW)
    code, out, _ = _run(repo, "")
    assert code == 0 and "nothing accepted" in out
    assert not load_host_config(repo).trust.accepted


def test_not_a_terminal_shows_and_accepts_nothing(tmp_path: Path, xdg: Path) -> None:
    repo = _repo(tmp_path, approval=_GRANT)
    code, out, _ = _run(repo, None)
    assert code == 6 and "nothing was accepted" in out
    assert repository_trust(repo).pending


def test_nothing_declared_asks_nothing(tmp_path: Path, xdg: Path) -> None:
    repo = _repo(tmp_path)
    code, out, asked = _run(repo, None)
    assert code == 0 and asked == []
    assert out.count("none declared") == 2


def test_a_changed_row_is_asked_again(tmp_path: Path, xdg: Path) -> None:
    """The postcondition: accepting a row never accepts its replacement."""
    repo = _repo(tmp_path, config=_ROW)
    _run(repo, "y")
    (repo / ".chickadee" / "config.toml").write_text(_ROW.replace("push", "destructive"))
    assert not load_host_config(repo).trust.accepted
    assert repository_trust(repo).pending


def test_outside_a_repository_is_a_precondition(tmp_path: Path, xdg: Path) -> None:
    code, out, _ = _run(tmp_path, None)
    assert code == 4 and "not a repository" in out


def test_an_unreadable_approval_file_is_a_usage_error(tmp_path: Path, xdg: Path) -> None:
    repo = _repo(tmp_path, approval="[[approval.grants]\n")
    code, out, _ = _run(repo, None)
    assert code == 1 and "malformed TOML" in out


def test_the_prompt_names_only_what_is_pending(tmp_path: Path, xdg: Path) -> None:
    repo = _repo(tmp_path, config=_ROW)
    _run(repo, "y")
    (repo / ".chickadee" / "approval.toml").write_text(_GRANT)
    _, _, asked = _run(repo, "y")
    assert asked == [f"accept {trust.APPROVAL} for this clone? [y/N] "]


def test_a_repositorys_allow_is_shown_and_inert_until_accepted(
    tmp_path: Path, xdg: Path
) -> None:
    """A committed `allow` lowers a class for every clone, so it waits for this clone's yes,
    under the approval file's digest."""
    from chickadee.approval.prefs import load_preferences
    from chickadee.schema.authorization import Tier

    repo = _repo(tmp_path, approval='[approval.commands]\nlockfile_install = "allow"\n')
    roots = load_host_config(repo).roots
    assert "lockfile_install" not in load_preferences(roots).command_overrides
    code, out, _ = _run(repo, "n")
    assert code == 0 and '[approval.commands] lockfile_install = "allow"' in out
    _run(repo, "y")
    assert load_preferences(roots).command_overrides["lockfile_install"] is Tier.T0
