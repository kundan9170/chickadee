"""Caps: every manifest arm declares a finite one, and crossing it is scored, not raised.

§5.9 — an arm without a cap cannot be preregistered, because the cap is part of the frozen
protocol. And §7 — a cap terminal is a recorded outcome under the preregistered missing
rule, so the run stops with rows on disk rather than throwing away what it already paid
for.

The other property here is D14's, which is easy to state and easy to get backwards:
`infra_skip` leaves the denominator. A problem whose toolchain was absent was never
attempted, and counting it as unsolved understates the solve rate for a reason that has
nothing to do with the harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chickadee_eval import runner
from chickadee_eval.manifest import Arm, Language
from chickadee_eval.problems import polyglot

STUB = "def f(x):\n    raise NotImplementedError\n"
TEST = (
    "from lib import f\n\n\n"
    "def test_f_doubles_its_argument():\n"
    '    assert f(2) == 4, "doubling two has to give four"\n'
)
SOLUTION = "def f(x):\n    return x * 2\n"


def make_problem(root: Path, pid: str) -> None:
    d = root / pid
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("Double it.\n", encoding="utf-8")
    (d / "lib.py").write_text(STUB, encoding="utf-8")
    (d / "lib_test.py").write_text(TEST, encoding="utf-8")


def arm(cap: float) -> Arm:
    return Arm(name="fake", harness="fake", harness_version="0", hard_cost_usd=cap)


class SpendingArm:
    """Solves nothing, bills a fixed amount per problem."""

    name = "spender"

    def __init__(self, per_problem: float) -> None:
        self.per_problem = per_problem
        self.seen: list[str] = []

    def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
        self.seen.append(problem.id)
        return runner.Attempt(solved_claim=False, cost_usd=self.per_problem)


class SolvingArm:
    name = "solver"

    def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
        (workspace / problem.stub_path).write_text(SOLUTION, encoding="utf-8")
        return runner.Attempt(solved_claim=True, cost_usd=0.01)


def test_an_arm_without_a_finite_cap_cannot_be_declared() -> None:
    """The cap is part of the protocol, so it is rejected at construction."""
    with pytest.raises(ValidationError):
        Arm(name="x", harness="h", harness_version="0", hard_cost_usd=0.0)
    with pytest.raises(ValidationError):
        Arm(name="x", harness="h", harness_version="0", hard_cost_usd=-1.0)


def test_crossing_the_cap_stops_the_run_and_is_recorded(tmp_path: Path) -> None:
    for i in range(5):
        make_problem(tmp_path, f"p{i}")
    problems = runner.load_all(tmp_path, [(f"p{i}", Language.PYTHON) for i in range(5)])
    sealed = runner.seal(require_sandbox=False, probe=False)

    summary = runner.run_arm(
        problems, SpendingArm(per_problem=0.40), arm(cap=1.00), sealed=sealed
    )

    assert summary.terminal is runner.RunTerminal.CAP
    # Stopped after the row that crossed, not before it: the spend happened and the record
    # of it is what the preregistered missing rule scores.
    assert len(summary.records) == 3
    assert summary.spent_usd == pytest.approx(1.20)


def test_a_run_inside_its_cap_completes(tmp_path: Path) -> None:
    for i in range(3):
        make_problem(tmp_path, f"p{i}")
    problems = runner.load_all(tmp_path, [(f"p{i}", Language.PYTHON) for i in range(3)])
    sealed = runner.seal(require_sandbox=False, probe=False)

    summary = runner.run_arm(
        problems, SpendingArm(per_problem=0.10), arm(cap=100.0), sealed=sealed
    )
    assert summary.terminal is runner.RunTerminal.COMPLETE
    assert len(summary.records) == 3


def test_cap_terminal_leaves_rows_on_disk(tmp_path: Path) -> None:
    """Partial results are archived. A run that threw away what it paid for would
    make the cap more expensive than the overspend."""
    for i in range(4):
        make_problem(tmp_path, f"p{i}")
    problems = runner.load_all(tmp_path, [(f"p{i}", Language.PYTHON) for i in range(4)])
    out = tmp_path / "results.jsonl"

    runner.run_arm(
        problems,
        SpendingArm(per_problem=1.0),
        arm(cap=1.5),
        sealed=runner.seal(require_sandbox=False, probe=False),
        results_path=out,
    )
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_infra_skip_leaves_the_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D14 / §7. The direction matters: counting an unattemptable problem as unsolved
    understates the solve rate for an infrastructure reason."""
    make_problem(tmp_path, "p0")
    problems = runner.load_all(tmp_path, [("p0", Language.PYTHON)])
    monkeypatch.setattr(polyglot, "toolchain_available", lambda _lang: False)

    summary = runner.run_arm(
        problems,
        SolvingArm(),
        arm(cap=10.0),
        sealed=runner.seal(require_sandbox=False, probe=False),
    )
    assert summary.records[0].outcome == polyglot.Outcome.INFRA_SKIP
    assert summary.denominator == 0
    assert summary.infra_skipped == 1
    assert summary.solved == 0


def test_solved_is_decided_by_the_tests_not_the_arms_claim(tmp_path: Path) -> None:
    """An arm's belief is recorded and never trusted. `run_tests` decides."""
    make_problem(tmp_path, "p0")
    problems = runner.load_all(tmp_path, [("p0", Language.PYTHON)])

    class Liar:
        name = "liar"

        def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
            return runner.Attempt(solved_claim=True, cost_usd=0.0)

    summary = runner.run_arm(
        problems,
        Liar(),
        arm(cap=10.0),
        sealed=runner.seal(require_sandbox=False, probe=False),
    )
    row = summary.records[0]
    assert row.solved_claim is True
    assert row.solved is False, "the stub still raises; the tests are the oracle"
    assert summary.denominator == 1


def test_a_real_solution_passes_the_real_tests(tmp_path: Path) -> None:
    """End to end on the local toolchain: the loop actually runs pytest and reads it."""
    make_problem(tmp_path, "p0")
    problems = runner.load_all(tmp_path, [("p0", Language.PYTHON)])

    summary = runner.run_arm(
        problems,
        SolvingArm(),
        arm(cap=10.0),
        sealed=runner.seal(require_sandbox=False, probe=False),
    )
    assert summary.records[0].outcome == polyglot.Outcome.PASS
    assert summary.solved == 1
    assert summary.denominator == 1


# --- a leak is discarded, not archived ---------------------------------------
# Greptile P1 on #56: the row was written and flushed *before* the leak check raised, so
# the "results discarded" left the invalid row on disk — and, because the row carried the
# matched lines verbatim, it wrote the oracle into an archive people share.


class LeakingArm:
    name = "leaker"

    def __init__(self, body: str) -> None:
        self.body = body

    def attempt(self, problem: polyglot.PromptView, workspace: Path) -> runner.Attempt:
        return runner.Attempt(solved_claim=False, cost_usd=0.0, prompts=(self.body,))


def test_a_leak_raises_and_writes_nothing(tmp_path: Path) -> None:
    make_problem(tmp_path, "p0")
    problems = runner.load_all(tmp_path, [("p0", Language.PYTHON)])
    out = tmp_path / "results.jsonl"
    body = problems[0].test_source()

    with pytest.raises(runner.TestBodyLeak, match="discarded"):
        runner.run_arm(
            problems,
            LeakingArm(body),
            arm(cap=10.0),
            sealed=runner.seal(require_sandbox=False, probe=False),
            results_path=out,
        )

    # §7: results discarded. Not "written then disowned".
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


def test_a_leak_never_puts_the_oracle_in_a_results_row(tmp_path: Path) -> None:
    """Even a row that *is* kept must not carry the matched text: `results.jsonl` is
    archived and shared, and an archive containing the answer key is an oracle leak with a
    longer half-life than the prompt that caused it."""
    make_problem(tmp_path, "p0")
    problems = runner.load_all(tmp_path, [("p0", Language.PYTHON)])
    body = problems[0].test_source()

    record = runner.run_problem(
        problems[0],
        LeakingArm(body),
        runner.seal(require_sandbox=False, probe=False),
    )
    assert record.leaked is True
    assert record.leaked_line_numbers, "the leak must still be identifiable"
    assert all(isinstance(n, int) for n in record.leaked_line_numbers)

    import json
    from dataclasses import asdict

    serialized = json.dumps(asdict(record))
    for line in body.splitlines():
        stripped = line.strip()
        if len(stripped) >= polyglot.MIN_MARKER:
            assert stripped not in serialized, f"oracle text reached the record: {stripped!r}"


def test_a_clean_run_still_writes_its_rows(tmp_path: Path) -> None:
    """The guard must not have made the happy path stop recording."""
    make_problem(tmp_path, "p0")
    problems = runner.load_all(tmp_path, [("p0", Language.PYTHON)])
    out = tmp_path / "results.jsonl"

    runner.run_arm(
        problems,
        SolvingArm(),
        arm(cap=10.0),
        sealed=runner.seal(require_sandbox=False, probe=False),
        results_path=out,
    )
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 1
