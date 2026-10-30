"""D14 accounting and the $/solved-task maths.

Two rules that are trivial to state and get backwards, so both directions are asserted here
rather than the convenient one:

- `infra_skip` **leaves** the denominator. A problem whose toolchain was absent was never
  attempted; scoring it unsolved understates the solve rate for an infrastructure reason.
- `REPLANNED` / `UNSATISFIABLE` **stay in** it and count unsolved. The harness attempted the
  problem and declined; excluding those would overstate the rate.

And the arithmetic case that has no right answer: cost per solved task when nothing was
solved. Not zero, not infinity — undefined, printed `—`. A zero there reads as "free", which
is precisely the direction an arm benefits from.
"""

from __future__ import annotations

from chickadee_eval.manifest import Language
from chickadee_eval.metrics import Ratio, aggregate, outcome_terminal, pair
from chickadee_eval.problems.polyglot import Outcome
from chickadee_eval.runner import ProblemRecord


def record(
    pid: str,
    *,
    arm: str = "a",
    outcome: Outcome = Outcome.PASS,
    solved: bool | None = None,
    claim: bool | None = None,
    cost: float = 0.0,
    language: Language = Language.PYTHON,
    detail: str = "",
    planted: bool = False,
    verified: bool = True,
    deps: str = "",
) -> ProblemRecord:
    resolved = (outcome is Outcome.PASS) if solved is None else solved
    return ProblemRecord(
        problem_id=pid,
        language=language,
        arm=arm,
        outcome=outcome,
        solved=resolved,
        solved_claim=resolved if claim is None else claim,
        exit_code=0 if resolved else 1,
        attempts=1,
        cost_usd=cost,
        wall_ms=10,
        sandbox="native",
        network_verified=verified,
        detail=detail,
        oracle_planted=planted,
        deps_digest=deps,
    )


# --- D14, both directions -----------------------------------------------------


def test_infra_skip_leaves_the_denominator() -> None:
    """The direction that matters: counting an unattemptable problem as unsolved
    understates the solve rate for a reason unrelated to the harness."""
    m = aggregate([record("p1"), record("p2", outcome=Outcome.INFRA_SKIP)])["a"]
    assert m.denominator == 1
    assert m.infra_skipped == 1
    assert m.solved == 1
    assert m.solve_rate.value == 1.0, "the skipped problem must not dilute the rate"


def test_a_run_of_only_skips_has_no_solve_rate() -> None:
    """Zero of zero is not 0%. It is a claim about a benchmark that never ran."""
    m = aggregate([record("p1", outcome=Outcome.INFRA_SKIP)])["a"]
    assert m.denominator == 0
    assert m.solve_rate.value is None
    assert m.solve_rate.render(percent=True) == "—"


def test_replanned_and_unsatisfiable_count_unsolved_and_stay_in_the_denominator() -> None:
    """The other direction. The harness attempted these and declined, so excluding them
    would overstate the rate — the mirror image of the infra_skip mistake."""
    rows = [
        record("p1"),
        record("p2", outcome=Outcome.FAIL, detail="terminal=replanned after 3 attempts"),
        record("p3", outcome=Outcome.FAIL, detail="terminal=unsatisfiable"),
    ]
    m = aggregate(rows)["a"]
    assert m.denominator == 3, "declining to solve is not the same as not attempting"
    assert m.solved == 1
    assert m.solve_rate.value == 1 / 3
    assert m.terminal_counts["replanned"] == 1
    assert m.terminal_counts["unsatisfiable"] == 1


def test_terminal_counts_are_zero_rather_than_absent_at_phase_1() -> None:
    """Nothing emits terminals yet. A dict missing the keys would read as "no such thing",
    while zeros read as "none occurred" — and only the second is checkable later."""
    m = aggregate([record("p1")])["a"]
    assert set(m.terminal_counts) == {"replanned", "unsatisfiable", "crash"}
    assert sum(m.terminal_counts.values()) == 0


def test_outcome_terminal_reads_nothing_from_an_ordinary_detail() -> None:
    assert outcome_terminal(record("p", detail="exit=1 wall_ms=12")) is None
    assert outcome_terminal(record("p", detail="TERMINAL=CRASH")) == "crash"


# --- $/solved-task ------------------------------------------------------------


def test_cost_per_solved_task_is_undefined_when_nothing_solved() -> None:
    """Not zero, not infinity. A zero reads as "free", which is the direction that
    flatters the arm."""
    rows = [
        record("p1", outcome=Outcome.FAIL, cost=0.5),
        record("p2", outcome=Outcome.FAIL, cost=0.5),
    ]
    m = aggregate(rows)["a"]
    assert m.total_cost_usd == 1.0
    assert m.cost_per_solved_task.value is None
    assert m.cost_per_solved_task.render() == "—"
    assert m.cost_per_task.value == 0.5, "cost per *task* is still defined"


def test_cost_per_solved_task_divides_total_spend_including_failures() -> None:
    """§4 says the cost side includes spend on failed tasks. Dividing only successful spend
    by successes would price an arm as if its failures were free."""
    rows = [record("p1", cost=1.0), record("p2", outcome=Outcome.FAIL, cost=3.0)]
    m = aggregate(rows)["a"]
    assert m.total_cost_usd == 4.0
    assert m.cost_per_solved_task.value == 4.0


def test_cost_includes_rows_outside_the_denominator() -> None:
    """An infra_skip costs nothing today. A rule that summed only attempted rows would drop
    real spend the day it does."""
    rows = [record("p1", cost=1.0), record("p2", outcome=Outcome.INFRA_SKIP, cost=0.25)]
    assert aggregate(rows)["a"].total_cost_usd == 1.25


def test_ratio_keeps_both_parts() -> None:
    """ "0 of 0" and "0 of 83" are different facts; a float would collapse them."""
    empty, measured = Ratio(0, 0), Ratio(0, 83)
    assert empty.value is None and not empty.defined
    assert measured.value == 0.0 and measured.defined
    assert empty.render(percent=True) == "—"
    assert measured.render(percent=True) == "0.0%"


# --- honesty counters ---------------------------------------------------------


def test_the_arms_claim_is_recorded_against_the_oracle() -> None:
    """§4 asks for checkpoint-pass-but-acceptance-fail. Checkpoints do not exist yet; this
    is the same question one layer out and it is measurable now."""
    rows = [
        record("p1", outcome=Outcome.FAIL, claim=True),
        record("p2", outcome=Outcome.PASS, claim=False),
        record("p3"),
    ]
    m = aggregate(rows)["a"]
    assert m.claimed_but_unsolved == 1
    assert m.solved_but_unclaimed == 1


def test_integrity_counters_surface_from_the_rows() -> None:
    rows = [
        record("p1", planted=True, verified=False),
        record("p2", detail="unpriced: cost_usd=0 is not a measurement", deps="abc"),
    ]
    m = aggregate(rows)["a"]
    assert m.oracle_plant_attempts == 1
    assert m.unsealed_rows == 1
    assert m.unpriced_rows == 1
    assert m.deps_digests == {"abc"}


def test_per_language_rates_are_separate() -> None:
    """A headline averaging 34 Python and 49 JavaScript problems would hide a harness that
    works in one language and not the other."""
    rows = [
        record("p1", language=Language.PYTHON),
        record("p2", language=Language.PYTHON, outcome=Outcome.FAIL),
        record("j1", language=Language.JAVASCRIPT),
    ]
    by_lang = aggregate(rows)["a"].by_language
    assert by_lang["python"].value == 0.5
    assert by_lang["javascript"].value == 1.0


def test_a_language_with_no_rows_has_no_rate() -> None:
    by_lang = aggregate([record("p1", language=Language.PYTHON)])["a"].by_language
    assert by_lang["javascript"].value is None, "0% would claim JavaScript was measured"


# --- pairing ------------------------------------------------------------------


def test_pairing_uses_only_problems_both_arms_attempted() -> None:
    """The gap is largest exactly when one arm skipped more, which is when it matters."""
    rows = [
        record("p1", arm="chickadee"),
        record("p2", arm="chickadee"),
        record("p3", arm="chickadee", outcome=Outcome.INFRA_SKIP),
        record("p1", arm="baseline", outcome=Outcome.FAIL),
        record("p2", arm="baseline"),
        record("p3", arm="baseline"),
    ]
    comparison = pair(aggregate(rows), "chickadee", "baseline")
    assert comparison.n == 2, "p3 is not paired: one arm never attempted it"
    assert comparison.solve_delta_pp == 50.0
    assert comparison.discordant == ["p1"]


def test_pairing_an_unknown_arm_raises() -> None:
    import pytest

    with pytest.raises(KeyError, match="not among the arms"):
        pair(aggregate([record("p1")]), "ghost", "a")


def test_an_empty_pairing_has_no_delta() -> None:
    rows = [record("p1", arm="x"), record("p2", arm="y")]
    comparison = pair(aggregate(rows), "x", "y")
    assert comparison.n == 0
    assert comparison.solve_delta_pp is None, "a delta over no pairs is not zero"
