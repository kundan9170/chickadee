"""A Zeus command's write in the session's worktree, through the engine (D199 amendment 1): it
is committed on its own, so the next `coder` call cannot fold it
into a task and no task undo can take it back silently; a `git switch` moves the session's
`workspace` line; `/undo <operation>` reaches it; and the handoff reports it as not gated.
Nothing is declared: every command that is not a pure read is recorded.

The real box and workspace, the model scripted and no sandbox: the person allowed an
unsandboxed shell and approves every confirmation, so the step is what is tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, Approving, build, make_ignoring_repo, zeus_delegates
from repos import git
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

from chickadee.context.project import JournalRecords
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError

OP = f"{SID}:s1"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Ignoring what running its checks leaves behind, as real repositories do
    (`test_undo.py`)."""
    return make_ignoring_repo(tmp_path / "repo")


def turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
) -> PacketEngine:
    raw["sandbox"] = {"allow_unsandboxed_shell": {"zeus": True}}
    engine, _ = build(monkeypatch, repo, raw, script, interaction=Approving())
    engines.append(engine)
    engine.turn(SID, "go").wait(timeout=WAIT_S)
    return engine


def text(engine: PacketEngine, cid: str) -> str:
    line = state(engine, SID).result(cid)
    assert line is not None, f"call {cid} has no result"
    return line.text


def refs(engine: PacketEngine) -> list[WorkspaceRef]:
    return [w.ref for w in state(engine, SID)._workspaces]  # pyright: ignore[reportPrivateUsage]


def worktree(engine: PacketEngine) -> Path:
    path = refs(engine)[-1].worktree_path
    assert path is not None
    return Path(path)


def runs(engine: PacketEngine) -> list[str]:
    return [e.run_id for e in state(engine, SID).episodes()]


def task_of(engine: PacketEngine, run_id: str) -> str:
    """The one task run `run_id` committed."""
    where = engine._where(SID)  # pyright: ignore[reportPrivateUsage]
    _, cached = engine._caches_for(where)  # pyright: ignore[reportPrivateUsage]
    projection = JournalRecords(where, cached).projection(run_id)
    (task,) = {
        op.task_id
        for op in projection.operations.values()
        if op.task_id and op.intended is not None and op.intended.effect_kind == "task_commit"
    }
    return task


def shell(command: str) -> Any:
    return calls(call("shell", "s1", command=command))


def test_a_git_switch_in_the_sessions_worktree_appends_a_workspace_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        *zeus_delegates(),
        shell("git switch -q -c feature/x"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    first, latest = refs(engine)
    assert latest.branch == "feature/x"
    assert latest.worktree_path == first.worktree_path
    assert latest.handoff == first.handoff
    assert git(worktree(engine), "branch", "--show-current") == "feature/x"


def test_the_next_coder_call_does_not_fold_a_shell_write_into_its_commit(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The shell's edit is its own commit, under its operation's trailer, so the second run's
    task commit holds only the task's change. Not source (`app.py` would be put back): a
    changelog."""
    script = [
        *zeus_delegates(),
        shell("printf '# formatted\\n' >> CHANGELOG.md"),
        *zeus_delegates("return 2", "return 3", cid="c2"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    tree = worktree(engine)
    assert git(tree, "status", "--porcelain") == ""
    subjects = git(tree, "log", "--format=%s", "-3").splitlines()
    assert subjects[1] == f"chickadee: shell {OP}"
    assert git(tree, "show", "--name-only", "--format=", "HEAD").split() == ["app.py"]
    assert "+    return 3" in git(tree, "show", "--format=", "HEAD", "--", "app.py")
    assert (tree / "CHANGELOG.md").read_text() == "# formatted\n"


def test_a_task_undo_below_a_shell_commit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        *zeus_delegates(),
        shell("echo data > gen.txt"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    tree = worktree(engine)
    tip = git(tree, "rev-parse", "HEAD")
    (run,) = runs(engine)

    with pytest.raises(SessionError, match="later"):
        engine.undo_task(SID, run, task_of(engine, run))

    assert git(tree, "rev-parse", "HEAD") == tip
    assert (tree / "gen.txt").read_text() == "data\n"


def test_undo_reaches_a_shell_write(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        *zeus_delegates(),
        shell("echo data > gen.txt"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    done = engine.undo_operation(SID, OP)

    assert done.restored == ("gen.txt",)
    assert not (worktree(engine) / "gen.txt").exists()


def test_the_handoff_lists_a_shell_write_as_not_gated(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        *zeus_delegates(),
        shell("echo data > gen.txt"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    summary = engine.handoff(SID)

    (line,) = [u for u in summary.unverified if OP in u]
    assert "not gated" in line and "gen.txt" in line


def test_undo_restores_only_what_the_command_wrote(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Nothing is declared, so the undo reaches what the command changed, and nothing the
    person or a task wrote beside it."""
    script = [
        *zeus_delegates(),
        shell("echo data > gen.txt"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)
    tree = worktree(engine)

    done = engine.undo_operation(SID, OP)

    assert done.restored == ("gen.txt",)
    assert not (tree / "gen.txt").exists()
    assert "return 2" in (tree / "app.py").read_text()
    assert git(tree, "status", "--porcelain") == ""


def test_a_shell_write_into_the_work_folder_is_put_back_and_said(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D204 decision 10 (was: kept and committed, pinned before T-3.166 part 2): the work
    folder is written with `doc`, so a command's write there is put back, as source's is."""
    script = [
        *zeus_delegates(),
        shell(
            "mkdir -p docs/chickadee/w/design && printf '# a\\n' > docs/chickadee/w/design/a.md"
        ),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    tree = worktree(engine)
    assert not (tree / "docs/chickadee/w/design/a.md").exists()
    assert "the work folder is written with doc" in text(engine, "s1")
    assert git(tree, "status", "--porcelain") == ""


@pytest.mark.writes_worktree("docs", "docs/chickadee", "docs/chickadee/.DS_Store")
def test_a_command_may_delete_a_finder_file_in_the_work_folder(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """T-3.204, the owner's run of 2026-10-24: `doc` refused on `docs/chickadee/.DS_Store`, and
    Zeus's `find -delete` of it was put back, leaving no exit. `doc` never writes one, so the
    delete stays, and is committed as the command's."""
    (repo / "docs/chickadee").mkdir(parents=True)
    (repo / "docs/chickadee/.DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "a Finder file tracked")
    script = [*zeus_delegates(), shell("find docs -name .DS_Store -delete"), answer("done")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    tree = worktree(engine)
    assert not (tree / "docs/chickadee/.DS_Store").exists()
    assert "put back" not in text(engine, "s1")
    assert git(tree, "status", "--porcelain") == ""


def test_a_finder_file_beside_a_commands_write_is_not_committed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A file the operating system writes while a command runs reads as the
    command's; it is no one's, so it stays on disk and out of the harness's commit."""
    script = [
        *zeus_delegates(),
        shell("echo data > gen.txt && touch .DS_Store"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    tree = worktree(engine)
    assert git(tree, "ls-files", "gen.txt").strip() == "gen.txt"
    assert git(tree, "ls-files", ".DS_Store").strip() == ""
    assert (tree / ".DS_Store").exists()
