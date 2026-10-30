"""Zeus's view follows the session's workspace.

The box writes the session's worktree; before this, Zeus kept reading the
checkout, so a solved edit looked like a failed one. After the first `coder` call, `read`, the
graph, `locate` and `shell` see the worktree; the `coder` result that moved the view says so,
and a frame records the workspace once per change. A path in the main checkout still runs, and
its result says whose files those are. A resumed session reads the worktree its line names.

Engine-level, through the real box and workspace with the model scripted, as `coder_helpers`
runs them; the sandbox half (a write in the worktree is still refused) needs bubblewrap.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo, one_task, solved, zeus_delegates
from repos import git
from zeus_turn_helpers import WAIT_S, Step, answer, call, calls, state

import chickadee.engine as engine_module
from chickadee.context.view import Rooted, SessionView
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.engine import PacketEngine
from chickadee.errors import InfraError
from chickadee.isolation.base import has
from chickadee.isolation.linux import BINARY, Bubblewrap
from chickadee.isolation.macos import BINARY as SEATBELT
from chickadee.isolation.macos import Seatbelt
from chickadee.secrets.denylist import resolve as resolve_denylist
from chickadee.zeus.tools import graphnav
from chickadee.zeus.tools.graphnav import opened

CHECKOUT = "this is your checkout, not the session's worktree"
MOVED = "view moved to the session's worktree"

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")
seatbelt_only = pytest.mark.skipif(not has(SEATBELT), reason="sandbox-exec unavailable")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: Sequence[Step],
    *,
    turns: int = 1,
) -> PacketEngine:
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    for n in range(turns):
        engine.turn(SID, f"turn {n}").wait(timeout=WAIT_S)
    return engine


def text(engine: PacketEngine, cid: str) -> str:
    line = state(engine, SID).result(cid)
    assert line is not None, f"call {cid} has no result"
    return line.text


def owned(engine: PacketEngine) -> WorkspaceRef:
    refs = [w.ref for w in state(engine, SID)._workspaces]  # pyright: ignore[reportPrivateUsage]
    assert refs, "the session recorded no workspace"
    return refs[-1]


def worktree(engine: PacketEngine) -> Path:
    ref = owned(engine)
    assert ref.worktree_path is not None
    return Path(ref.worktree_path)


# -- before and after the first coder call -----------------------------------------------------


@pytest.mark.writes_worktree("notes.txt")
def test_before_any_coder_call_reads_see_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    (repo / "notes.txt").write_text("the person's own notes\n")
    script = [calls(call("read", "r1", path="notes.txt", depth="full")), answer("seen")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert "the person's own notes" in text(engine, "r1")
    assert CHECKOUT not in text(engine, "r1"), "the checkout is the view: nothing to label"


def test_a_read_after_a_solved_coder_call_in_the_same_turn_sees_the_edit(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The owner's G3 session: the box wrote the worktree, Zeus read the checkout, and a solved
    edit looked like a failed one."""
    script = [
        *zeus_delegates(),
        calls(call("read", "r1", path="app.py", depth="full")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert "return 2" in text(engine, "r1")
    assert "return 1" not in text(engine, "r1")
    assert "return 1" in (repo / "app.py").read_text(), "the checkout is unchanged"


def test_the_shell_runs_in_the_worktree_after_the_move(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    script = [
        *zeus_delegates(),
        calls(call("shell", "s1", command="pwd && cat app.py")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    out = text(engine, "s1")
    assert str(worktree(engine).resolve()) in out
    assert "return 2" in out
    assert CHECKOUT not in out


# -- telling Zeus ------------------------------------------------------------------------------


def test_the_coder_call_that_creates_the_worktree_says_the_view_moved(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A new trigger in the table: the move is told in the `coder` result, the first thing
    Zeus reads after it, inside the same turn. A reattach moves nothing and says nothing."""
    start = git(repo, "rev-parse", "HEAD")
    script = [
        *zeus_delegates(),
        *zeus_delegates("return 2", "return 3", cid="c2"),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    first = text(engine, "c1")
    branch = owned(engine).branch
    assert branch is not None
    assert MOVED in first and branch in first and start[:8] in first
    assert "/handoff" in first
    assert MOVED not in text(engine, "c2")


def test_a_frame_is_appended_once_per_workspace_change(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`frame_due` gains one rule: the view's branch, which names its workspace, differs from
    the last frame's. Not when `head` moves: every `coder` call moves it, and a frame per call
    would be noise."""
    script = [
        *zeus_delegates(),
        answer("first"),
        *zeus_delegates("return 2", "return 3", cid="c2"),
        answer("second"),
        answer("third"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script, turns=3)

    frames = state(engine, SID).frames()
    assert len(frames) == 2, [f.text for f in frames]
    ref = owned(engine)
    assert frames[0].branch == git(repo, "branch", "--show-current"), "began in the checkout"
    assert frames[1].branch == ref.branch
    assert ref.branch is not None and ref.branch in frames[1].text


# -- the main checkout after the move ----------------------------------------------------------


def test_a_read_of_the_main_checkout_after_the_move_runs_and_is_labelled(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The owner's call: label, do not refuse."""
    script = [
        *zeus_delegates(),
        calls(call("read", "r1", path=str(repo / "app.py"), depth="full")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    got = text(engine, "r1")
    assert got.startswith(CHECKOUT), got
    branch = owned(engine).branch
    assert branch is not None and branch in got.splitlines()[0]
    assert "return 1" in got, "the checkout, as it is"


def test_a_shell_command_in_the_main_checkout_after_the_move_runs_and_is_labelled(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    script = [
        *zeus_delegates(),
        calls(call("shell", "s1", command="cat app.py", cwd=str(repo))),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    got = text(engine, "s1")
    assert got.startswith(CHECKOUT), got
    assert "return 1" in got


# -- resume ------------------------------------------------------------------------------------


def test_a_resumed_session_reads_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The `workspace` line is the record; a resumed session builds its view over it."""
    first = turn(
        monkeypatch,
        repo,
        raw,
        engines,
        [*zeus_delegates(), answer("done")],
    )
    first.close()

    script = [calls(call("read", "r1", path="app.py", depth="full")), answer("seen")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    assert "return 2" in text(engine, "r1")
    assert CHECKOUT not in text(engine, "r1")


def test_a_host_run_on_the_session_moves_the_view_before_the_next_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The view follows the session's `workspace` line, whoever wrote it: a host's own `coder`
    run creates the worktree outside any Zeus call, and Zeus's next turn reads it."""
    script = [
        *solved("return 1", "return 2"),
        calls(call("read", "r1", path="app.py", depth="full")),
        answer("seen"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.open_session(SID)
    handle = engine.coder(one_task(), session_id=SID)
    assert handle.result().outcome == "COMPLETED"

    engine.turn(SID, "look").wait(timeout=WAIT_S)

    assert "return 2" in text(engine, "r1")
    assert state(engine, SID).frames()[-1].branch == owned(engine).branch


def test_a_relative_path_climbing_into_the_checkout_runs_there_and_is_labelled(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The worktree is `<checkout>/.git/chickadee-worktrees/<session>`, so three levels up is
    the checkout: a path that climbs there is the checkout's, labelled, as an absolute one
    is."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    script = [
        *zeus_delegates(),
        calls(call("read", "r1", path="../../../app.py", depth="full")),
        calls(call("shell", "s1", command="cat app.py", cwd="../../..")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    for cid in ("r1", "s1"):
        got = text(engine, cid)
        assert got.startswith(CHECKOUT), (cid, got)
        assert "return 1" in got, cid


def test_the_worktree_shell_keeps_the_checkouts_masks(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A command in the worktree can still name a file in the checkout by its absolute path,
    so the worktree shell's masks are walked over the checkout as well as the worktree."""
    walked: list[list[Path]] = []
    real = resolve_denylist

    def recording(denylist: Any, roots: Any, **kwargs: Any) -> tuple[Path, ...]:
        walked.append([Path(r).resolve() for r in roots])
        return real(denylist, roots, **kwargs)

    monkeypatch.setattr(engine_module, "_resolve_denylist", recording)
    script = [*zeus_delegates(), answer("done")]
    engine = turn(monkeypatch, repo, raw, engines, script)

    into = worktree(engine).resolve()
    (roots,) = [w for w in walked if into in w]
    assert repo.resolve() in roots


def test_a_view_that_cannot_open_the_worktree_still_returns_the_runs_report(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A part that will not open over the worktree (its graph, here) neither loses the run's
    report nor leaves Zeus reading the checkout as if it were the worktree: it is told."""
    real = opened

    def refusing(root: Path) -> Any:
        if "chickadee-worktrees" in str(root):
            raise InfraError("the worktree's graph is locked")
        return real(root)

    monkeypatch.setattr(graphnav, "opened", refusing)
    script = [
        *zeus_delegates(),
        calls(call("read", "r1", path="app.py", depth="full")),
        answer("done"),
    ]
    engine = turn(monkeypatch, repo, raw, engines, script)

    report = text(engine, "c1")
    assert "T1 'Return 2': committed" in report, "the run's report is all there"
    assert "could not move to the session's worktree" in report
    assert "the worktree's graph is locked" in report
    assert "return 1" in text(engine, "r1"), "the view stayed in the checkout, and said so"


# -- the view on its own -----------------------------------------------------------------------


def _ref(path: Path, branch: str = "chickadee/x-11111111") -> WorkspaceRef:
    return WorkspaceRef(
        mode="worktree",
        lease_id="l",
        source_ref="main",
        source_head="0" * 40,
        branch=branch,
        worktree_path=str(path),
        handoff="pending",
    )


def _view(tmp_path: Path) -> tuple[SessionView, list[Path]]:
    built: list[Path] = []

    def build(root: Path) -> Rooted:
        built.append(root)
        return Rooted(root, Any, Any, Any, Any, Any, Any)  # pyright: ignore[reportArgumentType]

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    return SessionView(checkout, build), built


def test_a_worktree_whose_directory_is_gone_leaves_the_view_where_it_is(tmp_path: Path) -> None:
    """`still_written`: the view stays until the box adds the directory back, rather than
    going to the checkout and telling Zeus it moved again when the worktree returns."""
    view, built = _view(tmp_path)
    assert view.follow(_ref(tmp_path / "never")) is None
    assert view.where is None, "a worktree that is not there is never entered"
    tree = tmp_path / "tree"
    tree.mkdir()
    ref = _ref(tree)
    assert view.follow(ref) == ref
    tree.rmdir()
    assert view.follow(ref) is None
    assert view.where == ref and view.root == tree
    assert built == [tmp_path / "checkout", tree]


def test_each_root_is_built_once_however_often_the_view_returns_to_it(tmp_path: Path) -> None:
    view, built = _view(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    ref = _ref(tree)
    assert view.follow(ref) == ref
    assert view.follow(ref) is None, "a reattach moves nothing"
    assert view.follow(None) is None and view.where is None
    assert view.follow(ref) == ref
    assert built == [tmp_path / "checkout", tree]


# -- the sandbox -------------------------------------------------------------------------------


@pytest.fixture
def outside_tmp() -> Iterator[Path]:
    """A workspace outside `/tmp`, which the wrap replaces with a private tmpfs
    (`zeus/test_no_write.py`)."""
    base = Path("/var/tmp") if Path("/var/tmp").is_dir() else Path(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="session-view-", dir=base))
    try:
        yield make_repo(root / "repo")
    finally:
        shutil.rmtree(root, ignore_errors=True)


@bwrap_only
def test_a_write_in_the_worktree_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
    outside_tmp: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """An undeclared Zeus command stays sealed and read-only: it now reads the worktree,
    and writes nothing there."""
    script = [
        *zeus_delegates(),
        calls(call("shell", "s1", command="echo x > app.py")),
        answer("done"),
    ]
    engine, _ = build(monkeypatch, outside_tmp, raw, script)
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Bubblewrap())
    engines.append(engine)
    engine.turn(SID, "write").wait(timeout=WAIT_S)

    assert "return 2" in (worktree(engine) / "app.py").read_text()


@seatbelt_only
def test_an_edit_lands_under_seatbelt(
    monkeypatch: pytest.MonkeyPatch,
    outside_tmp: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """#540, the macOS twin of the test above: a sandboxed `git status` opens /dev/null, so
    without the profile's rule every edit failed on a Mac. The edit lands, and an undeclared
    write is still refused."""
    script = [
        *zeus_delegates(),
        calls(call("shell", "s1", command="echo x > app.py")),
        answer("done"),
    ]
    engine, _ = build(monkeypatch, outside_tmp, raw, script)
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Seatbelt())
    engines.append(engine)
    engine.turn(SID, "write").wait(timeout=WAIT_S)

    assert "return 2" in (worktree(engine) / "app.py").read_text()
