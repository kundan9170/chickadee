"""`open_works`: the
repository's open work, read from every session's log, the newest line about a work deciding
what it is; nothing written."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from repos import git, init_repo

import chickadee.context.store as store
from chickadee.context.session import Session, SessionState
from chickadee.context.work_line import WorkLine
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.open_works import open_works, work_key
from chickadee.work import MARKER

DIR = "docs/chickadee"


@pytest.fixture(autouse=True)
def ticking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each line a second after the last, so which is newest is never a tie."""
    seconds = itertools.count()

    def now() -> str:
        n = next(seconds)
        return f"2026-10-24T10:{n // 60:02d}:{n % 60:02d}Z"

    monkeypatch.setattr(store, "_now", now)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})


@pytest.fixture
def sessions(tmp_path: Path) -> Iterator[dict[str, Session]]:
    made: dict[str, Session] = {}
    yield made
    for session in made.values():
        session.close()


def _session(sessions: dict[str, Session], root: Path, sid: str) -> Session:
    session = Session.create(root / "sessions", sid)
    sessions[sid] = session
    return session


def _work(
    name: str = "2026-10-24-rats",
    state: Literal["open", "landed", "kept", "dropped"] = "open",
) -> WorkLine:
    return WorkLine(name=name, dir=DIR, state=state)


def _readme(root: Path, name: str, title: str) -> None:
    folder = root / DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text(f"{MARKER}\n# {title}\n", encoding="utf-8")


def _ref(
    path: Path,
    branch: str,
    base: str,
    handoff: Literal["pending", "apply", "squash", "keep", "discard"] = "pending",
) -> WorkspaceRef:
    return WorkspaceRef(
        mode="worktree",
        lease_id="lease",
        source_ref="main",
        source_head=base,
        branch=branch,
        worktree_path=str(path),
        handoff=handoff,
    )


def _all(sessions: dict[str, Session]) -> list[tuple[str, SessionState]]:
    return [(sid, s.state) for sid, s in sessions.items()]


@pytest.mark.writes_worktree(
    "docs",
    "docs/chickadee",
    "docs/chickadee/2026-10-24-rats",
    "docs/chickadee/2026-10-24-rats/README.md",
)
def test_an_open_in_place_work_is_listed_with_its_title(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    _readme(repo, "2026-10-24-rats", "power-ups and rats")
    _session(sessions, tmp_path, "s1").append(_work())
    (found,) = open_works(repo, _all(sessions))
    assert found.key == work_key(_work())
    assert found.title == "power-ups and rats"
    assert found.mode == "in_place"
    assert found.session_id == "s1"


def test_a_work_continued_by_a_later_session_is_that_sessions(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    _session(sessions, tmp_path, "s1").append(_work())
    _session(sessions, tmp_path, "s2").append(_work())
    (found,) = open_works(repo, _all(sessions))
    assert found.session_id == "s2"


def test_a_work_landed_by_a_later_session_is_closed_whatever_the_first_says(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    first = _session(sessions, tmp_path, "s1")
    first.append(_work())
    later = _session(sessions, tmp_path, "s2")
    later.append(_work())
    later.append(_work(state="landed"))
    assert open_works(repo, _all(sessions)) == ()


@pytest.mark.parametrize("state", ["kept", "open"])
def test_leaving_is_keep_so_a_kept_work_is_open(
    tmp_path: Path,
    repo: Path,
    sessions: dict[str, Session],
    state: Literal["open", "kept"],
) -> None:
    _session(sessions, tmp_path, "s1").append(_work(state=state))
    assert len(open_works(repo, _all(sessions))) == 1


@pytest.mark.parametrize("state", ["landed", "dropped"])
def test_a_landed_or_dropped_work_is_closed(
    tmp_path: Path,
    repo: Path,
    sessions: dict[str, Session],
    state: Literal["landed", "dropped"],
) -> None:
    _session(sessions, tmp_path, "s1").append(_work(state=state))
    assert open_works(repo, _all(sessions)) == ()


@pytest.mark.writes_worktree(
    "docs",
    "docs/chickadee",
    "docs/chickadee/2026-10-24-rats",
    "docs/chickadee/2026-10-24-rats/README.md",
    "docs/chickadee/2026-10-24-rats/report.md",
)
def test_an_in_place_work_whose_report_is_committed_has_landed(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    _readme(repo, "2026-10-24-rats", "power-ups and rats")
    (repo / DIR / "2026-10-24-rats" / "report.md").write_text(f"{MARKER}\n# Report\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the person committed the work")
    _session(sessions, tmp_path, "s1").append(_work())
    assert open_works(repo, _all(sessions)) == ()


def test_a_work_held_by_a_live_session_is_not_free(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    _session(sessions, tmp_path, "s1").append(_work())
    assert open_works(repo, _all(sessions), live=lambda sid: sid == "s1") == ()


def test_a_worktree_still_written_is_open_until_handed_off(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "branch", "chickadee/s1")
    session = _session(sessions, tmp_path, "s1")
    session.record_workspace(_ref(tmp_path / "wt", "chickadee/s1", base))
    (found,) = open_works(repo, _all(sessions))
    assert found.mode == "worktree"
    assert found.workspace is not None and found.workspace.branch == "chickadee/s1"
    session.record_workspace(_ref(tmp_path / "wt", "chickadee/s1", base, handoff="apply"))
    assert open_works(repo, _all(sessions)) == ()


@pytest.mark.writes_worktree("app.py")
def test_a_worktree_merged_by_the_person_has_landed(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "checkout", "-q", "-b", "chickadee/s1")
    (repo / "app.py").write_text("x = 2\n")
    git(repo, "commit", "-qam", "the coder's task")
    git(repo, "checkout", "-q", "main")
    session = _session(sessions, tmp_path, "s1")
    session.record_workspace(_ref(tmp_path / "wt", "chickadee/s1", base, handoff="keep"))
    assert len(open_works(repo, _all(sessions))) == 1
    git(repo, "merge", "-q", "--ff-only", "chickadee/s1")
    assert open_works(repo, _all(sessions)) == ()


def test_a_worktree_holding_an_open_work_is_one_entry_titled_from_the_worktree(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "branch", "chickadee/s1")
    _readme(tmp_path / "wt", "2026-10-24-rats", "power-ups and rats")
    session = _session(sessions, tmp_path, "s1")
    session.record_workspace(_ref(tmp_path / "wt", "chickadee/s1", base))
    session.append(_work())
    (found,) = open_works(repo, _all(sessions))
    assert found.key == work_key(_work())
    assert found.mode == "worktree"
    assert found.title == "power-ups and rats"


def test_the_newest_comes_first(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    _session(sessions, tmp_path, "s1").append(_work("2026-10-24-a"))
    _session(sessions, tmp_path, "s2").append(_work("2026-10-24-b"))
    found = open_works(repo, _all(sessions))
    assert [w.work.name for w in found if w.work] == ["2026-10-24-b", "2026-10-24-a"]


def test_a_kept_work_stays_open_when_the_same_session_opens_another(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    """Review of 2026-10-25: only a session's newest `work` line was read, so a work it kept
    vanished once it opened another."""
    session = _session(sessions, tmp_path, "s1")
    session.append(_work("2026-10-24-a"))
    session.append(_work("2026-10-24-a", state="kept"))
    session.append(_work("2026-10-24-b"))
    found = open_works(repo, _all(sessions))
    assert sorted(w.work.name for w in found if w.work) == ["2026-10-24-a", "2026-10-24-b"]


def _in_place(base: str) -> WorkspaceRef:
    return WorkspaceRef(
        mode="in_place",
        lease_id="lease",
        source_ref="main",
        source_head=base,
        branch=None,
        worktree_path=None,
        handoff="pending",
    )


def test_a_work_opened_after_leaving_a_worktree_is_in_place_and_the_worktree_its_own(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    """Review of 2026-10-25: a design opened in the checkout after `worktree leave` was tied
    to the worktree the session still held, so continuing it would have looked there."""
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "branch", "chickadee/s1")
    session = _session(sessions, tmp_path, "s1")
    session.record_workspace(_ref(tmp_path / "wt", "chickadee/s1", base))
    session.record_workspace(_in_place(base))
    session.append(_work())
    found = {w.key: w for w in open_works(repo, _all(sessions))}
    assert found[work_key(_work())].mode == "in_place"
    assert found[f"worktree:{tmp_path / 'wt'}"].mode == "worktree"


def test_a_work_opened_in_a_worktree_stays_there_after_leaving(
    tmp_path: Path, repo: Path, sessions: dict[str, Session]
) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "branch", "chickadee/s1")
    session = _session(sessions, tmp_path, "s1")
    session.record_workspace(_ref(tmp_path / "wt", "chickadee/s1", base))
    session.append(_work())
    session.record_workspace(_in_place(base))
    session.append(_work(state="kept"))
    (found,) = open_works(repo, _all(sessions))
    assert found.key == work_key(_work()) and found.mode == "worktree"
