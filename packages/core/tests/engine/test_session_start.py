"""A session starts where the work is.

`T-3.206`: the work runs in the person's checkout unless someone asked for a worktree; a dirty
checkout is never a reason to refuse; nothing commits on the person's branch as a session
starts. Which workspace a `coder` call runs in has one rule (`delegation.chosen_workspace`),
pinned here branch by branch, then through the engine.

`T-3.207`: a new session continues the repository's open work (`open_works`,
`create_session(adopt=…)`): D230's run of 2026-10-24, end to end.

The real box and workspace, the model scripted and no sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from coder_helpers import (
    DESIGN_README,
    SID,
    athena_accepts,
    athena_plans_folder,
    build,
    make_ignoring_repo,
    solved,
    work_name,
    worker_edit,
    worktree_count,
)
from engines import answer, call, calls
from repos import git
from zeus_turn_helpers import WAIT_S, state

from chickadee.context.session import Session
from chickadee.context.view import session_mode
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.delegation import chosen_workspace
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.work import open_work

ONE_TASK = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}


def _ref(
    mode: Literal["worktree", "in_place"] = "worktree",
    handoff: Literal["pending", "apply", "squash", "keep", "discard"] = "pending",
) -> WorkspaceRef:
    return WorkspaceRef(
        mode=mode,
        lease_id="lease",
        source_ref="main",
        source_head="0" * 40,
        branch="chickadee/s" if mode == "worktree" else None,
        worktree_path="/tmp/wt" if mode == "worktree" else None,
        handoff=handoff,
    )


# -- the rule ------------------------------------------------------------------------------


@pytest.mark.parametrize("asked", ["worktree", "in_place"])
def test_zeus_asking_for_a_workspace_wins(asked: Literal["worktree", "in_place"]) -> None:
    for folder in (None, True, False):
        assert chosen_workspace(asked, folder_in_worktree=folder, default="worktree") == asked


def test_a_design_runs_where_its_folder_is() -> None:
    assert chosen_workspace(None, folder_in_worktree=True, default="in_place") == "worktree"
    assert chosen_workspace(None, folder_in_worktree=False, default="worktree") == "in_place"


@pytest.mark.parametrize("default", ["worktree", "in_place"])
def test_otherwise_where_the_session_is(default: Literal["worktree", "in_place"]) -> None:
    assert chosen_workspace(None, folder_in_worktree=None, default=default) == default


# -- where the session is -----------------------------------------------


@pytest.fixture
def log(tmp_path: Path) -> Session:
    return Session.create(tmp_path / "sessions", "S-mode")


def test_a_session_already_in_a_worktree_keeps_working_there(log: Session) -> None:
    log.record_workspace(_ref())
    assert session_mode(log.state, "in_place") == "worktree"


def test_a_worktree_handed_off_is_not_kept_writing(log: Session) -> None:
    log.record_workspace(_ref())
    log.record_workspace(_ref(handoff="apply"))
    assert session_mode(log.state, "in_place") == "in_place"


def test_a_session_moved_back_to_the_checkout_works_there(log: Session) -> None:
    """Was: `[workspace] mode` decided once the session left its worktree."""
    log.record_workspace(_ref())
    log.record_workspace(_ref(mode="in_place"))
    assert session_mode(log.state, "worktree") == "in_place"


def test_a_flag_after_the_session_was_put_somewhere_wins(log: Session) -> None:
    """The owner's case: `--in-place` on a session that continued a worktree."""
    log.record_workspace(_ref())
    log.record_user("fix the typo", "in_place")
    assert session_mode(log.state, "worktree") == "in_place"


def test_a_move_after_the_flag_wins(log: Session) -> None:
    """`--in-place`, then "try it in a worktree": the later word decides."""
    log.record_user("fix the typo", "in_place")
    log.record_workspace(_ref())
    assert session_mode(log.state, "in_place") == "worktree"


def test_the_flag_outlasts_its_turn(log: Session) -> None:
    """Sent with the first turn only, it holds for the turns after it."""
    log.record_user("hi", "in_place")
    log.record_user("now fix the typo")
    assert session_mode(log.state, "worktree") == "in_place"


@pytest.mark.parametrize("default", ["worktree", "in_place"])
def test_otherwise_the_configured_mode(
    log: Session, default: Literal["worktree", "in_place"]
) -> None:
    assert session_mode(log.state, default) == default
    log.record_user("no flag")
    assert session_mode(log.state, default) == default


# -- through the engine --------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_ignoring_repo(tmp_path / "repo")


def _turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
) -> PacketEngine:
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
    return engine


def _result(engine: PacketEngine, cid: str) -> str:
    line = state(engine, SID).result(cid)
    assert line is not None, f"call {cid} has no result"
    return line.text


def test_a_coder_call_with_no_workspace_runs_in_the_checkout_and_commits_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """In place by default. Was: a worktree, its task committed there."""
    head = git(repo, "rev-parse", "HEAD")
    script = [calls(call("coder", "c1", **ONE_TASK)), *solved("return 1", "return 2")]
    engine = _turn(monkeypatch, repo, raw, engines, [*script, answer("done")])

    assert "COMPLETED" in _result(engine, "c1")
    assert "return 2" in (repo / "app.py").read_text()
    assert git(repo, "rev-parse", "HEAD") == head, "in place, nothing is committed (D39)"
    assert worktree_count(repo) == 1, "no worktree was made"


@pytest.mark.writes_worktree("notes.txt")
def test_a_dirty_checkout_is_not_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The checkout is ground truth. Was: the session's first worktree was
    refused until the person committed, stashed or chose in place."""
    (repo / "notes.txt").write_text("the person's own, uncommitted\n", encoding="utf-8")
    script = [calls(call("coder", "c1", **ONE_TASK)), *solved("return 1", "return 2")]
    engine = _turn(monkeypatch, repo, raw, engines, [*script, answer("done")])

    assert "COMPLETED" in _result(engine, "c1")
    assert (repo / "notes.txt").read_text() == "the person's own, uncommitted\n"
    assert "return 2" in (repo / "app.py").read_text()


def test_the_configured_worktree_mode_still_makes_one(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    raw["workspace"] = {"mode": "worktree"}
    script = [calls(call("coder", "c1", **ONE_TASK)), *solved("return 1", "return 2")]
    engine = _turn(monkeypatch, repo, raw, engines, [*script, answer("done")])

    assert "COMPLETED" in _result(engine, "c1")
    assert worktree_count(repo) == 2
    assert "return 1" in (repo / "app.py").read_text(), "the checkout is untouched"


def test_a_session_start_commits_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Nothing commits on the person's branch as a session starts. Was: the
    work folder's index was healed by a commit."""
    engine, _ = build(monkeypatch, repo, raw, [])
    engines.append(engine)
    head = git(repo, "rev-parse", "HEAD")
    engine.create_session("second")
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "status", "--porcelain") == ""


# -- a new session continues the open work ----------------------------------------

TITLE = "Power-ups and rats"
GOAL = "`go()` returns 2"


def _first_session_writes_a_design_and_leaves(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    readme = DESIGN_README.format(title=TITLE, goal=GOAL)
    doc = call("doc", "d1", op="write", file="README.md", content=readme, title=TITLE)
    engine, _ = build(monkeypatch, repo, raw, [calls(doc), answer("written; not built yet")])
    engines.append(engine)
    engine.turn(SID, "design power-ups and rats").wait(timeout=WAIT_S)
    engine.close()  # left without a handoff: leaving is keep


@pytest.mark.writes_worktree("notes.txt", "docs/chickadee/.DS_Store")
def test_a_second_session_continues_the_first_ones_work_with_no_question_and_no_commit(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D230's run of 2026-10-24: a second session, not `--continue`, after a first that was
    never handed off, with a dirty checkout and a Finder file in the work folder. Was: asked to
    commit first, then refused on `.DS_Store`. Now it continues the design and builds it."""
    _first_session_writes_a_design_and_leaves(monkeypatch, repo, raw, engines)
    (repo / "notes.txt").write_text("the person's own, uncommitted\n", encoding="utf-8")
    (repo / "docs/chickadee/.DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    head = git(repo, "rev-parse", "HEAD")

    script = [
        calls(call("coder", "c1", work=work_name(TITLE))),
        *athena_plans_folder(TITLE, GOAL),
        *worker_edit("return 1", "return 2"),
        *athena_accepts(),
        answer("built"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    (work,) = engine.open_works()
    assert (work.title, work.mode, work.session_id) == (TITLE, "in_place", SID)
    info = engine.create_session("second", adopt=work.key)
    assert info.continuing == work
    assert git(repo, "rev-parse", "HEAD") == head, "nothing commits as a session starts"
    engine.open_session("second")
    opened = open_work(state(engine, "second"))
    assert opened is not None and opened.name == work_name(TITLE)

    engine.turn("second", "build it").wait(timeout=WAIT_S)
    line = state(engine, "second").result("c1")
    assert line is not None and "COMPLETED" in line.text
    assert "return 2" in (repo / "app.py").read_text()
    assert (repo / "notes.txt").read_text() == "the person's own, uncommitted\n"
    assert engine.open_works() == (), "held by a live session, it is not free to continue"
    engine.close_session("second")
    (still,) = engine.open_works()
    assert still.session_id == "second", "the work is the second session's now"


def test_a_new_session_adopts_nothing_unless_asked(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    _first_session_writes_a_design_and_leaves(monkeypatch, repo, raw, engines)
    engine, _ = build(monkeypatch, repo, raw, [])
    engines.append(engine)
    info = engine.create_session("fresh")
    assert info.continuing is None
    engine.open_session("fresh")
    assert open_work(state(engine, "fresh")) is None
    assert len(engine.open_works()) == 1, "the work stays open for a later session"


def test_adopting_work_that_is_not_open_is_refused_before_anything_is_written(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, [])
    engines.append(engine)
    with pytest.raises(SessionError, match="not open work"):
        engine.create_session("nope", adopt="work:docs/chickadee/2026-01-01-gone")
    assert "nope" not in {s.session_id for s in engine.sessions()}
