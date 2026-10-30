"""The E1 Chickadee arm: the complete system, driven the way a person drives it."""

from __future__ import annotations

from pathlib import Path

import pytest
from engines import answer
from scripted_arm import delegates_and_doubles, doubled, make_problem, opener, workspace

from chickadee_eval.arms.chickadee import (
    DIFF_NAME,
    JOURNALS_NAME,
    TESTS_NAME,
    ChickadeeArm,
    first_turn,
)
from chickadee_eval.problems import polyglot
from chickadee_eval.runner import score_hidden, seal


def test_delegated_work_is_applied_before_it_is_scored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Zeus delegates; the worker edits in the leased worktree, so the problem's directory
    holds nothing to test until the arm hands the session off."""
    problem = make_problem(tmp_path / "problems")
    arm = ChickadeeArm(
        open_engine=opener(monkeypatch, delegates_and_doubles()),
        oracles={problem.id: problem},
        score=doubled,
    )
    ws = workspace(tmp_path)

    result = arm.attempt(problem.prompt_view(), ws)

    assert "return x * 2" in (ws / "lib.py").read_text(encoding="utf-8")
    assert result.solved_claim and result.attempts == 1
    assert result.prompts == (first_turn(problem.prompt_view()),)
    stats = arm.stats[0]
    assert stats.error == "" and stats.not_applied == []
    assert (stats.runs, stats.tasks, stats.rounds) == (1, 1, 1)
    assert stats.reviews == {"accept": 1}
    assert stats.route != "none"
    assert stats.spent_usd > 0 and result.cost_usd == stats.spent_usd


def test_a_failing_turn_is_retried_with_the_redacted_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Retry parity is a person's turn: the next turn says what failed, and
    the hidden test's lines are not in it. A turn with no delegation has nothing to
    apply, and says so rather than failing the attempt."""
    problem = make_problem(tmp_path / "problems")
    arm = ChickadeeArm(
        open_engine=opener(monkeypatch, [answer("I cannot."), answer("Still no.")]),
        oracles={problem.id: problem},
        score=doubled,
        max_turns=2,
    )

    result = arm.attempt(problem.prompt_view(), workspace(tmp_path))

    assert not result.solved_claim and result.attempts == 2
    retry = result.prompts[1]
    assert "NotImplementedError" in retry and "See the testing errors above" in retry
    assert polyglot.leaks(retry, problem) == []
    stats = arm.stats[0]
    assert stats.route == "none" and stats.runs == 0
    assert len(stats.not_applied) == 2


def test_a_failed_problem_keeps_what_shows_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The workspace is deleted when the problem ends, so what a reader needs to tell a wrong
    answer from an unfinished one is kept: each turn's test output, redacted as the retry
    turn is (the archive never holds the oracle), the diff, and the journals."""
    problem = make_problem(tmp_path / "problems")
    keep = tmp_path / "kept"
    arm = ChickadeeArm(
        open_engine=opener(monkeypatch, [answer("I cannot."), answer("Still no.")]),
        oracles={problem.id: problem},
        score=doubled,
        max_turns=2,
        keep=keep,
    )

    arm.attempt(problem.prompt_view(), workspace(tmp_path))

    kept = keep / problem.id
    tests = (kept / TESTS_NAME).read_text(encoding="utf-8")
    assert "== turn 1: fail ==" in tests and "== turn 2: fail ==" in tests
    assert "NotImplementedError" in tests
    assert polyglot.leaks(tests, problem) == []
    assert (kept / DIFF_NAME).read_text(encoding="utf-8") == ""
    assert list((kept / JOURNALS_NAME / "sessions").glob("*/session.jsonl"))
    assert not list(kept.rglob("LOCK"))


def test_an_unregistered_problem_is_refused(tmp_path: Path) -> None:
    known = make_problem(tmp_path, "known")
    stranger = make_problem(tmp_path, "stranger")
    arm = ChickadeeArm(
        open_engine=lambda _ws: pytest.fail("no engine opens for an unknown problem"),
        oracles={known.id: known},
        score=doubled,
    )
    with pytest.raises(KeyError, match="no oracle registered"):
        arm.attempt(stranger.prompt_view(), tmp_path / "ws")


def test_the_hidden_tests_are_on_disk_only_while_they_run(tmp_path: Path) -> None:
    """The arm's model can read files, so the scorer takes the tests away again, and a file the
    arm put where they go is put back as it was."""
    problem = make_problem(tmp_path / "problems")
    ws = workspace(tmp_path)
    sealed = seal(require_sandbox=False, probe=False)

    passed, output = score_hidden(problem, ws, sealed)
    assert not passed and output
    assert not (ws / problem.test_path).exists()

    (ws / problem.test_path).write_text("planted\n", encoding="utf-8")
    (ws / "lib.py").write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    passed, _ = score_hidden(problem, ws, sealed)
    assert passed
    assert (ws / problem.test_path).read_text(encoding="utf-8") == "planted\n"
