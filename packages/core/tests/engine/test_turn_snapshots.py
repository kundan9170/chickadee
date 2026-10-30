"""A turn that changes code is snapshotted before its first write and when it ends: one pair per
workspace it writes in, none for a turn that only talks. The
size bound skips large untracked files and names them (decision 8); retention is the one place
a snapshot ref is deleted (decision 7)."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo, zeus_delegates
from repos import git, init_repo
from zeus_turn_helpers import WAIT_S, Answering, answer, state

import chickadee.undo as undo_module
import chickadee.workspace as workspace_module
from chickadee.config import UndoPolicy
from chickadee.context.rewind import CodeSnapshotLine, taken
from chickadee.context.session import Session
from chickadee.engine import PacketEngine
from chickadee.errors import InfraError, SessionError
from chickadee.schema.interaction import ApprovalStop, StopDecision
from chickadee.workspace import (
    DAY_S,
    IN_PLACE_KEY,
    OPS_REF_PREFIX,
    SNAPSHOT_REF_PREFIX,
    TURNS_REF_PREFIX,
    SnapshotSite,
    plan_undo,
    snapshot_operation,
    snapshot_turn,
    sweep_snapshots,
    turn_ref,
)
from chickadee.workspace import turn_refs as alive_refs
from chickadee.zeus.snapshots import TurnSnapshots

SRC = Path(__file__).resolve().parents[2] / "src" / "chickadee"
BIG = 4096


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def live(engine: PacketEngine) -> Session:
    """The session's writer, open on `engine`."""
    conversation = engine._conversations.get(SID)  # pyright: ignore[reportPrivateUsage]
    opened = conversation.opened() if conversation is not None else None
    assert opened is not None, f"session {SID} is not built"
    return opened.session


def snapshots(engine: PacketEngine) -> tuple[CodeSnapshotLine, ...]:
    return taken(live(engine))


def tree_paths(repo: Path, tree: str) -> set[str]:
    return set(git(repo, "ls-tree", "-r", "--name-only", tree).splitlines())


def show(repo: Path, tree: str, path: str) -> str:
    return git(repo, "show", f"{tree}:{path}")


def run_turns(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    script: list[Any],
    *said: str,
) -> PacketEngine:
    engine, _ = build(monkeypatch, repo, raw, script)
    for text in said:
        engine.turn(SID, text).wait(timeout=WAIT_S)
    return engine


# -- the turn ------------------------------------------------------------------------------


def test_a_turn_that_writes_has_one_pair_and_a_talking_turn_none(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    script = [
        *zeus_delegates(workspace="in_place"),
        answer("changed"),
        answer("just talking"),
    ]
    engine = run_turns(monkeypatch, repo, raw, script, "change go()", "how are you?")
    try:
        lines = snapshots(engine)
        (wrote, talked) = [t.seq for t in state(engine, SID).turns() if t.role == "user"]
        assert [(s.turn, s.phase, s.key) for s in lines] == [
            (wrote, "before", IN_PLACE_KEY),
            (wrote, "after", IN_PLACE_KEY),
        ]
        assert all(s.turn != talked for s in lines), "a turn that only talks costs nothing"
        before, after = lines
        assert show(repo, before.tree, "app.py").endswith("return 1")
        assert show(repo, after.tree, "app.py").endswith("return 2")
        for s in lines:
            assert s.ref == turn_ref(SID, wrote, IN_PLACE_KEY, s.phase)
            assert git(repo, "rev-parse", f"{s.ref}^{{tree}}") == s.tree
    finally:
        engine.close()


def test_two_worktree_runs_in_one_turn_share_the_sessions_pair(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """Both runs write in the session's one worktree, so the turn has one
    `before` and one `after` of it (D179 keys a pair by workspace; it was one per run while
    each run had a worktree of its own)."""
    script = [
        *zeus_delegates(),
        *zeus_delegates("def go()", "def go_now()", cid="c2"),
        answer("both done"),
    ]
    engine = run_turns(monkeypatch, repo, raw, script, "two changes")
    try:
        lines = snapshots(engine)
        assert len(state(engine, SID).episodes()) == 2
        assert sorted((s.key, s.phase) for s in lines) == [(SID, "after"), (SID, "before")]
        assert {s.mode for s in lines} == {"worktree"}
        assert len({s.turn for s in lines}) == 1
    finally:
        engine.close()


class EditsWhileAsked(Answering):
    """The person approves the work at its stop, and edits a file while the turn waits on
    them there: before the harness wrote anything."""

    def __init__(self, repo: Path) -> None:
        super().__init__("yes")
        self.repo = repo

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        (self.repo / "mine.txt").write_text("the person's\n", encoding="utf-8")
        return super().confirm_stop(stop)


@pytest.mark.writes_worktree("mine.txt")
def test_an_edit_the_person_made_during_the_turn_is_inside_its_before(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """Lazy: the `before` is taken at the first `coder` dispatch, not when the turn began, so
    what the person did meanwhile is in it, and an undo keeps it."""
    script = [
        *zeus_delegates(workspace="in_place"),
        answer("changed"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    try:
        engine.interaction = EditsWhileAsked(repo)
        engine.turn(SID, "change go()").wait(timeout=WAIT_S)
        before = next(s for s in snapshots(engine) if s.phase == "before")
        assert show(repo, before.tree, "mine.txt") == "the person's"
    finally:
        engine.close()


def test_in_place_the_first_coder_call_stages_the_checkout_once(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The turn's `before` is the tree the D39 pre-run snapshot has just
    written, not a second `git add -A` of the checkout."""
    staged: list[str] = []
    real = workspace_module._write_tree  # pyright: ignore[reportPrivateUsage]

    def counted(source: Path, what: str, **kw: Any) -> tuple[str, tuple[str, ...]]:
        staged.append(what)
        return real(source, what, **kw)

    monkeypatch.setattr(workspace_module, "_write_tree", counted)
    script = [
        *zeus_delegates(workspace="in_place"),
        answer("changed"),
    ]
    engine = run_turns(monkeypatch, repo, raw, script, "change go()")
    try:
        before = next(s for s in snapshots(engine) if s.phase == "before")
        (run,) = [e.run_id for e in state(engine, SID).episodes()]
        pre_run = git(repo, "rev-parse", f"{SNAPSHOT_REF_PREFIX}/{run}^{{tree}}")
        assert before.tree == pre_run
        assert staged.count("snapshot") == 1, "the pre-run snapshot staged it"
        assert [w for w in staged if w.startswith("turn")] == [
            f"turn {before.turn}'s after snapshot"
        ]
    finally:
        engine.close()


# -- the size bound (decision 8) ----------------------------------------------------------


@pytest.mark.writes_worktree("big.bin", "small.txt", "app.py")
def test_a_large_untracked_file_is_skipped_and_named_a_large_tracked_one_kept(
    repo: Path,
) -> None:
    (repo / "big.bin").write_bytes(b"x" * BIG)
    (repo / "small.txt").write_text("small\n", encoding="utf-8")
    (repo / "app.py").write_text("def go() -> int:\n" + "    # pad\n" * BIG, encoding="utf-8")
    site = SnapshotSite(repo, "in_place", IN_PLACE_KEY)
    taken = snapshot_turn(
        site, session_id=SID, turn=7, phase="after", untracked_max_bytes=BIG - 1
    )
    assert taken.skipped == ("big.bin",)
    paths = tree_paths(repo, taken.tree)
    assert "big.bin" not in paths
    assert {"small.txt", "app.py"} <= paths, "a tracked file is kept whatever its size"
    assert git(repo, "status", "--porcelain", "--", "big.bin").startswith("??"), (
        "the person's own index is untouched"
    )


def test_a_snapshot_is_taken_once_per_phase_of_a_workspace_in_a_turn(repo: Path) -> None:
    site = SnapshotSite(repo, "in_place", IN_PLACE_KEY)
    snapshot_turn(site, session_id=SID, turn=3, phase="before", untracked_max_bytes=BIG)
    with pytest.raises(Exception, match="already has a before snapshot"):
        snapshot_turn(site, session_id=SID, turn=3, phase="before", untracked_max_bytes=BIG)


# -- retention (decision 7) ---------------------------------------------------------------


def turn_refs(repo: Path) -> set[str]:
    out = git(repo, "for-each-ref", "--format=%(refname)", f"{TURNS_REF_PREFIX}/")
    return {ln for ln in out.splitlines() if ln}


def take_turns(repo: Path, turns: range) -> SnapshotSite:
    site = SnapshotSite(repo, "in_place", IN_PLACE_KEY)
    for turn in turns:
        for phase in ("before", "after"):
            snapshot_turn(site, session_id=SID, turn=turn, phase=phase, untracked_max_bytes=BIG)
    return site


def test_retention_keeps_the_newest_turns_whatever_their_age(repo: Path) -> None:
    take_turns(repo, range(1, 6))
    later = 10 * DAY_S
    deleted = sweep_snapshots(repo, SID, keep_days=7, keep_turns=2, now=time.time() + later)
    kept = turn_refs(repo)
    assert {r.split("/")[4] for r in kept} == {"4", "5"}, "the newest two, though old"
    assert {r.split("/")[4] for r in deleted} == {"1", "2", "3"}
    assert len(deleted) == 6, "both phases of each removed turn"


def test_retention_removes_nothing_younger_than_keep_days(repo: Path) -> None:
    take_turns(repo, range(1, 6))
    assert sweep_snapshots(repo, SID, keep_days=7, keep_turns=1, now=time.time()) == ()
    assert len(turn_refs(repo)) == 10


def test_retention_leaves_another_sessions_turns_and_the_pre_run_snapshots(repo: Path) -> None:
    take_turns(repo, range(1, 3))
    other = SnapshotSite(repo, "in_place", IN_PLACE_KEY)
    snapshot_turn(other, session_id="S-other", turn=1, phase="before", untracked_max_bytes=BIG)
    workspace_module.snapshot(repo, "R-pre")
    sweep_snapshots(repo, SID, keep_days=0, keep_turns=0, now=time.time() + DAY_S)
    left = turn_refs(repo)
    assert left == {turn_ref("S-other", 1, IN_PLACE_KEY, "before")}
    assert git(repo, "for-each-ref", "--format=%(refname)", f"{SNAPSHOT_REF_PREFIX}/") == (
        f"{SNAPSHOT_REF_PREFIX}/R-pre"
    )


def test_operation_refs_follow_the_same_rule_per_scope(repo: Path) -> None:
    for op in ("op-a", "op-b", "op-c"):
        snapshot_operation(repo, op)
    sweep_snapshots(repo, SID, keep_days=7, keep_turns=1, now=time.time() + 10 * DAY_S)
    out = git(repo, "for-each-ref", "--format=%(refname)", f"{OPS_REF_PREFIX}/")
    assert len([ln for ln in out.splitlines() if ln]) == 1, "the newest of the scope is kept"


def test_a_snapshot_taken_runs_the_sweep(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    swept: list[str] = []
    real = sweep_snapshots

    def watched(root: Path, session_id: str, **kw: Any) -> tuple[str, ...]:
        swept.append(session_id)
        return real(root, session_id, **kw)

    monkeypatch.setattr("chickadee.zeus.snapshots.sweep_snapshots", watched)
    script = [
        *zeus_delegates(workspace="in_place"),
        answer("changed"),
    ]
    engine = run_turns(monkeypatch, repo, raw, script, "change go()")
    engine.close()
    assert swept == [SID, SID], "once for the before, once for the after"


# -- the one deleter ----------------------------------------------------------------------


def _deletes_a_ref(node: ast.Call) -> bool:
    words = [a.value for a in node.args if isinstance(a, ast.Constant)]
    return "update-ref" in words and "-d" in words


def test_no_code_outside_the_retention_sweep_deletes_a_ref() -> None:
    """The sweep is the one place a snapshot ref is deleted. Every
    `update-ref -d` in the package is inside it."""
    where: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and _deletes_a_ref(node):
                    where.append(f"{path.relative_to(SRC)}:{fn.name}")
    assert sorted(set(where)) == ["workspace.py:sweep_snapshots"]


# -- a turn that ends badly ----------------------------------------------------------------


def broke(_request: object) -> Any:
    raise RuntimeError("the model broke")


def test_a_turn_that_wrote_and_then_failed_still_gets_its_after(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    script = [
        *zeus_delegates(workspace="in_place"),
        broke,
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    try:
        with pytest.raises(SessionError, match="the model broke"):
            engine.turn(SID, "change go()").wait(timeout=WAIT_S)
        assert [s.phase for s in snapshots(engine)] == ["before", "after"]
    finally:
        engine.close()


def test_an_after_that_cannot_be_taken_does_not_hide_why_the_turn_failed(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The bracket `given_back_on_failure` uses: the turn's own error is the one raised, and
    the snapshot's failure is a note on it."""
    real = snapshot_turn

    def no_after(site: SnapshotSite, **kw: Any) -> Any:
        if kw["phase"] == "after":
            raise InfraError("git could not write the after tree")
        return real(site, **kw)

    monkeypatch.setattr("chickadee.zeus.snapshots.snapshot_turn", no_after)
    script = [
        *zeus_delegates(workspace="in_place"),
        broke,
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    try:
        with pytest.raises(SessionError, match="the model broke") as err:
            engine.turn(SID, "change go()").wait(timeout=WAIT_S)
        cause = err.value.__cause__
        assert cause is not None
        assert any("could not write the after tree" in n for n in cause.__notes__)
    finally:
        engine.close()


# -- a wake turn ------------------------------------------------------------


@pytest.mark.writes_worktree("*")
def test_a_wake_turn_is_its_persons_turn_next_part_and_undo_takes_both_back(
    tmp_path: Path,
) -> None:
    """The person asked for a fix; Zeus made it; a job's exit woke Zeus, who fixed its test.
    The wake's work is the turn's next part: `/undo` of the turn takes back both writes, and
    what the person edited by hand between the two stays the person's."""
    root = init_repo(tmp_path / "repo", {"login.py": "bug\n", "README.md": "old\n"})
    session = Session.create(tmp_path / "sessions", "S-wake")
    turn = session.record_user("fix the login bug")
    snaps = TurnSnapshots(session, UndoPolicy(), root)
    site = SnapshotSite(root, "in_place", IN_PLACE_KEY)

    snaps.before(site)
    (root / "login.py").write_text("fixed\n")
    snaps.ended()
    (root / "README.md").write_text("mine\n")  # the person, by hand, while the tests ran
    snaps.woken()
    snaps.before(site)
    (root / "test_login.py").write_text("def test_login(): ...\n")
    snaps.ended()

    shots = taken(session)
    assert sorted((s.part, s.phase) for s in shots) == [
        (0, "after"),
        (0, "before"),
        (1, "after"),
        (1, "before"),
    ]
    groups, missing = undo_module._pairs(  # pyright: ignore[reportPrivateUsage]
        session.state, shots, alive_refs(root, session.session_id), turn.seq
    )
    assert missing is None
    plan = plan_undo(
        root, [p.trees() for p in groups[IN_PLACE_KEY]], untracked_max_bytes=1 << 20
    )
    assert set(plan.restore) == {"login.py", "test_login.py"}, "both parts are the harness's"
    assert "README.md" in plan.person and "README.md" not in plan.restore
    session.close()


@pytest.mark.writes_worktree("*")
def test_the_persons_next_turn_starts_at_part_zero_again(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    session = Session.create(tmp_path / "sessions", "S-wake")
    session.record_user("one")
    snaps = TurnSnapshots(session, UndoPolicy(), root)
    snaps.woken()
    second = session.record_user("two")
    site = SnapshotSite(root, "in_place", IN_PLACE_KEY)
    snaps.before(site)
    (root / "a.py").write_text("a = 2\n")
    snaps.ended()
    assert {(s.turn, s.part) for s in taken(session)} == {(second.seq, 0)}
    session.close()


def _held_turn(tmp_path: Path) -> tuple[Session, TurnSnapshots, SnapshotSite, Path]:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    session = Session.create(tmp_path / "sessions", "S-held")
    session.record_user("go")
    return (
        session,
        TurnSnapshots(session, UndoPolicy(), root),
        SnapshotSite(root, "in_place", IN_PLACE_KEY),
        root,
    )


def _after(session: Session, root: Path) -> str:
    [line] = [s for s in taken(session) if s.phase == "after"]
    return show(root, line.tree, "a.py")


def test_two_background_runs_in_one_turn_are_both_inside_its_after(tmp_path: Path) -> None:
    """A run that ends while its turn goes on leaves the `after` to the turn's end,
    so a second run the turn starts in the same workspace is inside it too."""
    session, snaps, site, root = _held_turn(tmp_path)
    snaps.before(site)
    snaps.hold(site)
    (root / "a.py").write_text("a = 2\n")
    snaps.run_ended(site)  # the first run, while the turn goes on
    snaps.before(site)
    snaps.hold(site)
    snaps.ended()  # the turn ends while the second run works
    (root / "a.py").write_text("a = 3\n")
    snaps.run_ended(site)
    assert _after(session, root) == "a = 3"
    session.close()


def test_a_run_that_outlives_its_turn_takes_the_after_at_its_end(tmp_path: Path) -> None:
    session, snaps, site, root = _held_turn(tmp_path)
    snaps.before(site)
    snaps.hold(site)
    snaps.ended()
    assert [s.phase for s in taken(session)] == ["before"], "the run still writes there"
    (root / "a.py").write_text("a = 2\n")
    snaps.run_ended(site)
    assert _after(session, root) == "a = 2"
    session.close()


def test_a_run_abandoned_before_it_started_leaves_the_after_to_the_turn(tmp_path: Path) -> None:
    """A call whose answer is never asked for ends its hold (`_EngineBox._dispatched`), so the
    turn's pair is whole."""
    session, snaps, site, _root = _held_turn(tmp_path)
    snaps.before(site)
    snaps.hold(site)
    snaps.run_ended(site)  # abandoned, within the turn
    snaps.ended()
    assert [s.phase for s in taken(session)] == ["before", "after"]
    session.close()
