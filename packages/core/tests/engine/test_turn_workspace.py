"""A turn says where its work runs: the person's `--worktree` or `--in-place`, carried on their
turn line, is
that turn's default in place of `[workspace] mode`; Zeus's own `workspace` still decides its
call; and `worktree` counts as the person naming one for `worktree enter`.

The real box and workspace, the model scripted and no sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, Approving, build, make_ignoring_repo, solved
from engines import answer, call, calls
from zeus_turn_helpers import WAIT_S, result, state

from chickadee.context.view import held_worktree
from chickadee.engine import PacketEngine
from chickadee.work import works

ONE_TASK = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}
DESIGN = "# Retry\n\n## Goal\nRetry the call.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_ignoring_repo(tmp_path / "repo")


def _turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
    *,
    words: str = "make go return 2",
    workspace: Any = None,
    person: Any = None,
) -> tuple[PacketEngine, Any]:
    person = person or Approving()
    engine, _ = build(monkeypatch, repo, raw, script, interaction=person)
    engines.append(engine)
    engine.turn(SID, words, workspace=workspace).wait(timeout=WAIT_S)
    return engine, person


def _coder(**extra: object) -> list[Any]:
    return [calls(call("coder", "c1", **ONE_TASK, **extra)), *solved("return 1", "return 2")]


def test_the_flag_is_written_on_the_persons_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _turn(monkeypatch, repo, raw, engines, [answer("ok")], workspace="worktree")

    (turn,) = [t for t in state(engine, SID).turns() if t.role == "user"]
    assert turn.workspace == "worktree"


def test_a_worktree_flag_runs_a_call_with_no_workspace_in_a_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """In place is the default; the person's flag moves it for the turn."""
    engine, _ = _turn(
        monkeypatch, repo, raw, engines, [*_coder(), answer("done")], workspace="worktree"
    )

    assert held_worktree(state(engine, SID)) is not None
    assert "return 1" in (repo / "app.py").read_text(), "the checkout was not written"


def test_without_a_flag_the_default_stays_in_place(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _turn(monkeypatch, repo, raw, engines, [*_coder(), answer("done")])

    assert held_worktree(state(engine, SID)) is None
    assert "return 2" in (repo / "app.py").read_text()


def test_zeus_s_own_workspace_still_decides_its_call(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _turn(
        monkeypatch,
        repo,
        raw,
        engines,
        [*_coder(workspace="in_place"), answer("done")],
        workspace="worktree",
    )

    assert held_worktree(state(engine, SID)) is None
    assert "return 2" in (repo / "app.py").read_text()


def test_a_worktree_flag_names_one_for_enter(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The words say nothing of a worktree, but the flag does: `enter` asks nothing."""
    script = [calls(call("worktree", "w1", action="enter")), answer("in it")]
    engine, person = _turn(
        monkeypatch, repo, raw, engines, script, words="try it separately", workspace="worktree"
    )

    assert result(engine, "w1", SID).startswith("entered the worktree")
    assert person.shown == [], "nothing was asked"


def test_a_worktree_flag_opens_a_designs_folder_in_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Final review of 2026-10-25: the `doc` port read `[workspace] mode` alone, so under
    `run --yes` a design was written in the checkout, and its run then went there too."""
    raw["sandbox"] = {"require_sandbox": False}
    write = call("doc", "d1", op="write", file="README.md", content=DESIGN, title="Retry")
    engine, _ = _turn(
        monkeypatch, repo, raw, engines, [calls(write), answer("ok")], workspace="worktree"
    )

    assert not result(engine, "d1", SID).startswith("DENIED")
    held = held_worktree(state(engine, SID))
    assert held is not None and held.worktree_path is not None
    (work,) = works(state(engine, SID))
    assert (Path(held.worktree_path) / work.dir / work.name / "design" / "README.md").is_file()
    assert not (repo / work.dir).exists(), "the checkout was not written"


# -- D230 Amendment 4: the flag moves the session --------------------------------------------


def _two_turns(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
    first: tuple[str, Any],
    second: tuple[str, Any],
) -> PacketEngine:
    engine, _ = build(monkeypatch, repo, raw, script, interaction=Approving())
    engines.append(engine)
    for words, flag in (first, second):
        engine.turn(SID, words, workspace=flag).wait(timeout=WAIT_S)
    return engine


def test_in_place_on_a_session_in_its_worktree_works_in_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The owner's case (2026-10-25): `--in-place` on a session that holds a worktree ran in
    that worktree, the flag being only a default the held worktree outranked. Now the turn
    opens by leaving it, the worktree kept, and the work lands in the checkout."""
    script = [
        calls(call("worktree", "w1", action="enter")),
        answer("in it"),
        *_coder(),
        answer("done"),
    ]
    engine = _two_turns(
        monkeypatch,
        repo,
        raw,
        engines,
        script,
        ("try it in a worktree", None),
        ("make go return 2", "in_place"),
    )

    assert "return 2" in (repo / "app.py").read_text(), "the work is in the checkout"
    held = held_worktree(state(engine, SID))
    assert held is not None and Path(str(held.worktree_path)).is_dir(), "the worktree is kept"
    told = [t.text for t in state(engine, SID).turns() if t.role == "harness"]
    assert any("--in-place" in t and "back in the user's checkout" in t for t in told)


def test_in_place_given_once_holds_for_the_turns_after_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The REPL sends the flag with its first turn only; `[workspace] mode` says worktree."""
    raw["workspace"] = {"mode": "worktree"}
    script = [answer("hello"), *_coder(), answer("done")]
    engine = _two_turns(
        monkeypatch,
        repo,
        raw,
        engines,
        script,
        ("hi", "in_place"),
        ("make go return 2", None),
    )

    assert "return 2" in (repo / "app.py").read_text()
    assert held_worktree(state(engine, SID)) is None, "no worktree was made"


def test_words_after_the_flag_move_the_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`--in-place`, then "try it in a worktree": the later word decides, and the flag typed
    earlier never pulls the work back."""
    script = [
        answer("hello"),
        calls(call("worktree", "w1", action="enter")),
        *_coder(),
        answer("done"),
    ]
    engine = _two_turns(
        monkeypatch,
        repo,
        raw,
        engines,
        script,
        ("hi", "in_place"),
        ("try it in a worktree, make go return 2", None),
    )

    assert "return 1" in (repo / "app.py").read_text(), "the checkout was not written"
    held = held_worktree(state(engine, SID))
    assert held is not None
    assert "return 2" in (Path(str(held.worktree_path)) / "app.py").read_text()
