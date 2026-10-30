"""Fresh repo, re-run, --force, non-git, and the exclude line."""

from __future__ import annotations

import pathlib

import pytest
from repos import git

from chickadee.init import InitError, detect, init, publish, require_initialised
from chickadee_cli.init import render


def test_fresh_repo_publishes_a_complete_generation(repo: pathlib.Path) -> None:
    report = init(repo)
    state = publish.read_state(repo)
    assert state is not None and state["complete"] is True
    assert state["generation"] == report.generation
    assert (publish.chickadee_dir(repo) / "generations" / report.generation).is_dir()
    assert "python" in [e.name for e in report.profiles]


def test_rerun_preserves_config_and_rebuilds_the_index(repo: pathlib.Path) -> None:
    """The index is derived and always reconstructible; the config is not."""
    first = init(repo)
    config = publish.chickadee_dir(repo) / publish.CONFIG_NAME
    config.write_text(config.read_text() + "\n# a human wrote this\n")

    second = init(repo)
    assert second.config_preserved
    assert "# a human wrote this" in config.read_text()
    assert second.generation != first.generation
    assert publish.read_state(repo)["generation"] == second.generation  # type: ignore[index]


def test_force_rewrites_config_after_reporting_it(repo: pathlib.Path) -> None:
    init(repo)
    config = publish.chickadee_dir(repo) / publish.CONFIG_NAME
    config.write_text("# handmade\n")
    report = init(repo, force=True)
    assert not report.config_preserved
    assert "# handmade" not in config.read_text()
    assert any("--force rewrote" in w for w in report.warnings)


def test_a_folder_that_is_not_a_repository_is_made_one_secrets_left_out(
    tmp_path: pathlib.Path,
) -> None:
    """No longer a refusal naming `git init`. The folder's files are one first
    commit; what the deny list denies, and what no project keeps, stay on disk, untracked and
    excluded, so a later `git add --all` cannot take them in either."""
    folder = tmp_path / "plain"
    (folder / "src" / "__pycache__").mkdir(parents=True)
    (folder / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (folder / "src" / "__pycache__" / "app.pyc").write_text("c", encoding="utf-8")
    (folder / ".env").write_text("TOKEN=1\n", encoding="utf-8")
    said: list[str] = []
    report = init(folder, say=said.append)
    assert git(folder, "log", "--format=%s").strip() == "chickadee: start"
    assert git(folder, "ls-files").split() == ["src/app.py"]
    assert git(folder, "status", "--short").strip() == ""
    assert (folder / ".env").read_text(encoding="utf-8") == "TOKEN=1\n", "left on disk"
    assert "git repository, first commit 'chickadee: start'" in report.created
    assert any(".env" in w for w in report.warnings)
    assert any("making one" in line for line in said)


def test_the_home_directory_and_above_are_never_made_a_repository(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home" / "me"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    for folder in (home, home.parent):
        with pytest.raises(InitError, match="will not make one there"):
            init(folder)
        assert not (folder / ".git").exists()


def test_a_folder_that_does_not_exist_still_says_so(tmp_path: pathlib.Path) -> None:
    with pytest.raises(detect.NotAGitRepository):
        init(tmp_path / "nowhere")


def test_exclude_line_added_once_and_never_duplicated(repo: pathlib.Path) -> None:
    """Never the user's tracked.gitignore, and never a second copy."""
    init(repo)
    exclude = detect.exclude_path(repo)
    assert publish.GIT_EXCLUDE_LINE in exclude.read_text()
    init(repo)
    init(repo)
    lines = [ln.strip() for ln in exclude.read_text().splitlines()]
    assert lines.count(publish.GIT_EXCLUDE_LINE) == 1
    assert not (repo / ".gitignore").exists()


def test_the_users_gitignore_is_never_touched(repo: pathlib.Path) -> None:

    (repo / ".gitignore").write_text("*.log\n")
    git(repo, "add", ".gitignore")
    init(repo)
    assert (repo / ".gitignore").read_text() == "*.log\n"


def test_nothing_outside_chickadee_is_written(repo: pathlib.Path) -> None:
    """Non-goal: no repo mutation outside.chickadee/ and the resolved exclude line."""
    before = {p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()}
    init(repo)
    after = {p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file()}
    new = {p for p in after - before if not p.startswith((".chickadee/", ".git/"))}
    assert new == set()


def test_require_initialised_guards_before_and_passes_after(repo: pathlib.Path) -> None:
    with pytest.raises(InitError, match="chickadee init"):
        require_initialised(repo)
    init(repo)
    require_initialised(repo)


def test_the_report_says_what_the_index_holds(repo: pathlib.Path) -> None:
    """§3 step 5: the index's size and what it could not read are surfaced, not buried."""
    report = init(repo)
    assert report.index.files_indexed == 1
    assert "  index       1 files," in render(report)


def test_an_engine_command_makes_the_repository_and_leaves_the_index_to_the_engine(
    tmp_path: pathlib.Path,
) -> None:
    """Before an engine starts, a folder that is not a repository is made one, the
    person told; the index is not built here, but by the engine beside them
    (`PacketEngine.index_in_background`), so the screen opens at once. A repository is left
    alone, with nothing said."""
    from chickadee.engine import in_repository, is_initialised
    from chickadee_cli.init import prepare

    folder = tmp_path / "plain"
    folder.mkdir()
    (folder / "app.py").write_text("x = 1\n", encoding="utf-8")
    said: list[str] = []
    assert prepare(folder, say=said.append) is None
    assert in_repository(folder) and not is_initialised(folder)
    assert any("making one" in line for line in said)
    again: list[str] = []
    assert prepare(folder, say=again.append) is None and again == []


def test_the_home_directory_is_refused_before_any_engine_starts(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chickadee_cli.init import prepare

    monkeypatch.setenv("HOME", str(tmp_path))
    failed = prepare(tmp_path, say=lambda _line: None)
    assert failed is not None and failed[0] == 4
    assert "will not make one there" in failed[1]


def test_many_secrets_and_no_git_identity_still_make_the_first_commit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More secret paths than an argv may hold go to git on stdin; an author git has none of
    is Chickadee's, each part only where it is missing."""
    # The harness's git keeps `HOME` and drops the rest: a home with no `.gitconfig`.
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    folder = tmp_path / "plain"
    keys = folder / "fixtures"
    keys.mkdir(parents=True)
    for n in range(3000):
        (keys / f"key-{n:04d}-{'x' * 60}.pem").write_text("k\n", encoding="utf-8")
    (folder / "app.py").write_text("x = 1\n", encoding="utf-8")
    init(folder)
    assert git(folder, "ls-files").split() == ["app.py"]
    assert git(folder, "log", "--format=%an <%ae>").strip() == "Chickadee <chickadee@localhost>"
