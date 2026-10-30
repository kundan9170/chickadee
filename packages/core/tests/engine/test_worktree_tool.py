"""Zeus's `worktree` tool, through the engine: the person asks, Zeus moves the work, and the
harness does the git.

The round trip the owner asked for: "try it in a worktree" → the coder works there → "bring it
back" → the work is in the checkout, one confirmation asked. And each move's edges: entering a
worktree the person's words did not name asks first; leaving keeps the worktree, and entering
again returns to it; a discard keeps a snapshot; nothing to land is said so.

The real box and workspace, the model scripted and no sandbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    Approving,
    build,
    make_ignoring_repo,
    solved,
    worktree_count,
)
from engines import answer, call, calls
from repos import git
from zeus_turn_helpers import WAIT_S, result, state

from chickadee.context.view import latest_workspace
from chickadee.engine import PacketEngine, ScriptedInteraction
from chickadee.zeus.worktree import DISCARDED_REF_PREFIX, named_worktree

ONE_TASK = {"title": "Return 2", "brief": "## Goal\n`go()` returns 2.\n"}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_ignoring_repo(tmp_path / "repo")


def _turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
    words: str,
    interaction: Any,
) -> PacketEngine:
    engine, _ = build(monkeypatch, repo, raw, script, interaction=interaction)
    engines.append(engine)
    engine.turn(SID, words).wait(timeout=WAIT_S)
    return engine


def worktree(action: str, cid: str, **more: object) -> Any:
    return calls(call("worktree", cid, action=action, **more))


def test_the_word_worktree_is_matched_plainly() -> None:
    assert named_worktree("try it in a Worktree, please")
    assert not named_worktree("try it separately")


@pytest.mark.parametrize(
    "words",
    [
        "don't use a worktree",
        "no worktree, just edit it here",
        "do it without a worktree",
        "never make a new worktree for this",
        "please do not open a worktree",
        "a worktree isn't needed",
    ],
)
def test_words_that_refuse_a_worktree_do_not_name_one(words: str) -> None:
    """Review of 2026-10-25: a plain match read "don't use a worktree" as asking for one, so
    Zeus could enter it without asking."""
    assert not named_worktree(words)


@pytest.mark.parametrize(
    "words",
    [
        "no tests yet, but try it in a worktree",
        "don't touch main. use a worktree",
        "use the worktree, not the checkout",
    ],
)
def test_a_worktree_asked_for_beside_a_negation_elsewhere_is_named(words: str) -> None:
    assert named_worktree(words)


def test_try_it_in_a_worktree_then_bring_it_back(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The owner's round trip. Entering asks nothing, since the words named a worktree; the
    coder works there, the checkout untouched; landing asks once, and the work is then in the
    checkout."""
    person = Approving()
    script = [
        worktree("enter", "w1"),
        calls(call("coder", "c1", **ONE_TASK)),
        *solved("return 1", "return 2"),
        worktree("land", "w2"),
        answer("done"),
    ]
    engine = _turn(
        monkeypatch, repo, raw, engines, script, "try it in a worktree, then bring it", person
    )

    assert result(engine, "w1", SID).startswith("entered the worktree")
    assert "COMPLETED" in result(engine, "c1", SID)
    assert result(engine, "w2", SID).startswith("landed")
    assert [a.display.split(",")[0] for a in person.shown] == [
        a.display.split(",")[0] for a in person.shown if a.display.startswith("land ")
    ], "only the landing was asked"
    assert len(person.shown) == 1
    assert "return 2" in (repo / "app.py").read_text(), "the work is in the checkout"


def test_a_worktree_the_words_did_not_name_is_asked_first_and_a_no_keeps_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    person = ScriptedInteraction(actions=["deny"])
    script = [worktree("enter", "w1"), answer("staying here")]
    engine = _turn(monkeypatch, repo, raw, engines, script, "try it separately", person)

    assert result(engine, "w1", SID).startswith("not entered")
    assert len(person.approvals) == 1 and person.approvals[0].kind == "git"
    assert worktree_count(repo) == 1


def test_leaving_keeps_the_worktree_and_entering_again_returns_to_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    person = Approving()
    script = [
        worktree("enter", "w1"),
        worktree("leave", "w2"),
        worktree("enter", "w3"),
        answer("back in it"),
    ]
    engine = _turn(monkeypatch, repo, raw, engines, script, "use a worktree", person)

    first = result(engine, "w1", SID)
    assert result(engine, "w2", SID).startswith("back in the user's checkout")
    again = result(engine, "w3", SID)
    assert first.split(" at ")[0] == again.split(" at ")[0], "the same worktree"
    assert worktree_count(repo) == 2, "one worktree, kept across the leave"
    here = latest_workspace(state(engine, SID))
    assert here is not None and here.mode == "worktree"


def test_a_coder_call_into_a_left_worktree_moves_the_session_back_into_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Review of 2026-10-25: a `coder` call asking for the worktree the session had left
    reattached it but recorded no line, so the session's latest line still named the checkout:
    Zeus's view stayed there, and the next in-place call wrote in the checkout."""
    script = [
        worktree("enter", "w1"),
        worktree("leave", "w2"),
        calls(call("coder", "c1", workspace="worktree", **ONE_TASK)),
        *solved("return 1", "return 2"),
        answer("done"),
    ]
    engine = _turn(monkeypatch, repo, raw, engines, script, "use a worktree", Approving())

    assert "COMPLETED" in result(engine, "c1", SID)
    here = latest_workspace(state(engine, SID))
    assert here is not None and here.mode == "worktree"
    assert "return 1" in (repo / "app.py").read_text(), "the checkout never had the work"
    assert worktree_count(repo) == 2, "the same worktree, not a second"


def test_a_left_worktree_is_open_work_for_a_later_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Leaving keeps the worktree held, so `open_works` still lists
    it once the session is closed."""
    script = [worktree("enter", "w1"), worktree("leave", "w2"), answer("left")]
    engine = _turn(monkeypatch, repo, raw, engines, script, "use a worktree", Approving())
    engine.close_session(SID)

    (work,) = engine.open_works()
    assert work.mode == "worktree" and work.session_id == SID


def test_a_discard_asks_first_and_keeps_a_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    person = Approving()
    script = [
        worktree("enter", "w1"),
        calls(call("coder", "c1", **ONE_TASK)),
        *solved("return 1", "return 2"),
        worktree("discard", "w2"),
        answer("dropped"),
    ]
    engine = _turn(monkeypatch, repo, raw, engines, script, "try it in a worktree", person)

    assert result(engine, "w2", SID).startswith("discarded")
    assert len(person.shown) == 1 and person.shown[0].display.startswith("discard ")
    assert "return 1" in (repo / "app.py").read_text(), "the checkout never had the work"
    kept = git(repo, "for-each-ref", "--format=%(refname)", DISCARDED_REF_PREFIX).split()
    assert len(kept) == 1, "its last state is kept"
    assert worktree_count(repo) == 1


def test_nothing_to_land_is_said(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    person = Approving()
    script = [worktree("land", "w1"), answer("nothing")]
    engine = _turn(monkeypatch, repo, raw, engines, script, "bring it back", person)

    assert result(engine, "w1", SID).startswith("DENIED")
    assert "no worktree" in result(engine, "w1", SID)
    assert person.shown == []


@pytest.mark.writes_worktree("notes.txt")
def test_entering_from_a_dirty_checkout_carries_it_into_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Was: refused, telling the person to commit or stash (part 1). `T-3.212` moves it: the
    worktree starts with the checkout's changes (`tests/engine/test_seeded_worktree.py`)."""
    (repo / "notes.txt").write_text("the person's own, uncommitted\n", encoding="utf-8")
    script = [worktree("enter", "w1"), answer("in it")]
    engine = _turn(monkeypatch, repo, raw, engines, script, "use a worktree", Approving())

    assert result(engine, "w1", SID).startswith("entered the worktree")
    assert worktree_count(repo) == 2


@pytest.mark.writes_worktree("app.py")
def test_a_merge_that_stops_on_conflicts_names_them_and_the_way_forward(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Zeus merges with git through his shell, as any git line; when git
    stops on conflicts, the answer names the files and the two exits, the coder or an abort,
    since source is the coder's to resolve."""
    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "app.py").write_text((repo / "app.py").read_text().replace("return 1", "return 3"))
    git(repo, "commit", "-qam", "feature: return 3")
    git(repo, "checkout", "-q", "main")
    (repo / "app.py").write_text((repo / "app.py").read_text().replace("return 1", "return 2"))
    git(repo, "commit", "-qam", "main: return 2")
    raw["sandbox"] = {"allow_unsandboxed_shell": {"zeus": True}}
    script = [calls(call("shell", "s1", command="git merge feature")), answer("conflicts")]
    engine = _turn(monkeypatch, repo, raw, engines, script, "merge feature", Approving())

    said = result(engine, "s1", SID)
    assert "git stopped on conflicts in app.py" in said
    assert "coder" in said and "--abort" in said
    assert "<<<<<<<" in (repo / "app.py").read_text(), "git's markers are kept, not put back"


@pytest.mark.writes_worktree("app.py")
def test_a_landing_the_handoff_would_refuse_is_said_before_the_person_is_asked(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The person changed the file the coder changed: `bring` would write over it, so the
    refusal is the answer, and no one is asked to approve a landing that then fails."""

    def the_person_edits_then_zeus_lands(_request: object) -> Any:
        (repo / "app.py").write_text("def go() -> int:\n    return 9\n")
        return worktree("land", "w2")

    person = Approving()
    script = [
        worktree("enter", "w1"),
        calls(call("coder", "c1", **ONE_TASK)),
        *solved("return 1", "return 2"),
        the_person_edits_then_zeus_lands,
        answer("refused"),
    ]
    engine = _turn(monkeypatch, repo, raw, engines, script, "try it in a worktree", person)

    said = result(engine, "w2", SID)
    assert said.startswith("DENIED") and "app.py" in said
    assert person.shown == []
    assert "return 9" in (repo / "app.py").read_text(), "the person's edit is untouched"


def test_a_worktree_that_cannot_be_described_is_refused_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Final review of 2026-10-25: with its branch gone the review offers nothing and refuses
    nothing, and `land` looked its words up unguarded."""

    def the_branch_is_deleted_then_zeus_lands(_request: object) -> Any:
        for branch in git(
            repo, "for-each-ref", "--format=%(refname)", "refs/heads/chickadee/"
        ).split():
            git(repo, "update-ref", "-d", branch)
        return worktree("land", "w2")

    person = Approving()
    script = [worktree("enter", "w1"), the_branch_is_deleted_then_zeus_lands, answer("gone")]
    engine = _turn(monkeypatch, repo, raw, engines, script, "try it in a worktree", person)

    said = result(engine, "w2", SID)
    assert said.startswith("DENIED") and "cannot be described" in said
    assert person.shown == []
