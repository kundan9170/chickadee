"""A golden report from a fixture result set.

The golden is the point: `render` is deterministic, so the whole document is compared byte
for byte and any change to what a reader sees has to be looked at deliberately. A report is
the artifact people read *instead of* the archive, so a silent change to it is a silent
change to what the project claims.

Beyond the golden, three properties that a diff of prose would not reliably catch:

- the seal's verdict appears **above** the numbers, in all three of its states;
- the perturbed set is reported separately and never folded into a headline;
- no undefined quantity is ever printed as `0`.
"""

from __future__ import annotations

import csv
import io
import re

from chickadee.schema import SandboxLevel
from chickadee_eval import report
from chickadee_eval.manifest import Language
from chickadee_eval.metrics import aggregate, pair
from chickadee_eval.problems.polyglot import Outcome
from chickadee_eval.runner import ProblemRecord, Seal

GOLDEN = """\
# E1 run report

> **Seal: verified.** Applied `native`; connect() failed inside the sealed environment.

> **Diagnostic panel coverage: 40%.** The rest of §4's panel has no source in this run and \
is listed, with reasons, below. Unavailable is not zero.

## Arms

| metric | `naive` |
|---|---|
| problems | 4 |
| denominator (D14) | 3 |
| infra_skip (excluded) | 1 |
| solved | 2 |
| solve rate | 66.7% |
| total cost USD | 0.6000 |
| cost/task | 0.200 |
| cost/solved task | 0.300 |
| claimed solved, oracle disagreed | 1 |
| oracle-plant attempts | 0 |
| rows without verified seal | 0 |
| rows with unpriced cost | 0 |
| wall ms | 40 |

"""


def record(
    pid: str,
    *,
    outcome: Outcome = Outcome.PASS,
    arm: str = "naive",
    cost: float = 0.2,
    claim: bool | None = None,
    language: Language = Language.PYTHON,
    verified: bool = True,
) -> ProblemRecord:
    solved = outcome is Outcome.PASS
    return ProblemRecord(
        problem_id=pid,
        language=language,
        arm=arm,
        outcome=outcome,
        solved=solved,
        solved_claim=solved if claim is None else claim,
        exit_code=0 if solved else 1,
        attempts=1,
        cost_usd=cost,
        wall_ms=10,
        sandbox="native",
        network_verified=verified,
    )


def sealed(verified: bool = True) -> Seal:
    return Seal(
        available=SandboxLevel.NATIVE,
        applied=SandboxLevel.NATIVE,
        arm_isolation="test fixture",
        network_verified=verified,
        verification=(
            "connect() failed inside the sealed environment"
            if verified
            else "probe exited 7, inconclusive"
        ),
        env={},
    )


def fixture_rows() -> list[ProblemRecord]:
    return [
        record("p1"),
        record("p2"),
        record("p3", outcome=Outcome.FAIL, claim=True),
        record("p4", outcome=Outcome.INFRA_SKIP, cost=0.0),
    ]


# --- the golden ---------------------------------------------------------------


def test_the_report_opens_with_the_golden_header_and_arm_table() -> None:
    """Byte for byte down to the end of the arm table. The panels follow and are covered by
    `test_diagnostics.py`; pinning them here too would make this golden fail for reasons
    that belong to that file."""
    text = report.render(aggregate(fixture_rows()), sealed=sealed())
    head = text.split("## Perturbed set", 1)[0]
    assert head == GOLDEN, f"report header changed:\n{head!r}"


def test_rendering_is_deterministic() -> None:
    """A golden over a non-deterministic renderer is a flake, not a test."""
    metrics = aggregate(fixture_rows())
    assert report.render(metrics, sealed=sealed()) == report.render(metrics, sealed=sealed())


# --- the seal is stated above the numbers -------------------------------------


def test_a_verified_seal_is_reported_before_any_solve_rate() -> None:
    text = report.render(aggregate(fixture_rows()), sealed=sealed())
    assert text.index("Seal: verified") < text.index("solve rate")


def test_an_unverified_seal_says_the_numbers_are_diagnostic_only() -> None:
    """Contamination was measured at 63% of one frontier model's resolutions and
    sealing cost it 14.1pp. A reader must not have to assume which kind of number this is."""
    text = report.render(aggregate(fixture_rows()), sealed=sealed(verified=False))
    assert "Seal: NOT verified" in text
    assert "diagnostic only" in text
    assert text.index("NOT verified") < text.index("solve rate")


def test_a_missing_seal_is_not_silence() -> None:
    """Rendering without a `Seal` must not read the same as rendering with a good one."""
    text = report.render(aggregate(fixture_rows()))
    assert "Seal: not recorded" in text
    assert "unsealed until shown otherwise" in text


# --- perturbed set: separate, never averaged ----------------------


def test_the_perturbed_table_is_its_own_section_with_its_reading() -> None:
    original = aggregate(fixture_rows())
    perturbed = aggregate(
        [record("p1-perturbed", outcome=Outcome.FAIL), record("p2-perturbed")]
    )
    text = report.render(original, perturbed=perturbed, original=original)

    assert "Perturbed set (D23)" in text
    assert "Never averaged into the headline" in text
    assert "memorisation is doing the work" in text, "the table alone has a comfortable reading"

    # Signed either way. This fixture's perturbed set does *worse* (66.7% -> 50.0%), which
    # is the direction §5.3 cares about, so asserting a "+" would have been asserting the
    # comfortable case.
    section = text.split("Perturbed set", 1)[1]
    assert re.search(r"\|\s*[+-]\d+\.\d\s*\|", section), section


def test_the_headline_solve_rate_ignores_the_perturbed_set() -> None:
    """The property EV-5 actually protects: the arm table must be identical whether or not
    perturbed results were passed in."""
    original = aggregate(fixture_rows())
    perturbed = aggregate(
        [record("q1", outcome=Outcome.FAIL), record("q2", outcome=Outcome.FAIL)]
    )
    without = report.render(original).split("## Perturbed set", 1)[0]
    with_pert = report.render(original, perturbed=perturbed, original=original).split(
        "## Perturbed set", 1
    )[0]
    assert without == with_pert


def test_no_perturbed_results_is_not_evidence_of_no_memorisation() -> None:
    text = report.render(aggregate(fixture_rows()))
    assert "Contamination control did not run" in text
    assert "not evidence that memorisation is absent" in text


# --- undefined quantities print as em dashes ----------------------------------


def test_nothing_solved_prints_an_em_dash_for_cost_per_solved_task() -> None:
    """Not `0.000`. A zero there reads as "free"."""
    rows = [record("p1", outcome=Outcome.FAIL, cost=1.0)]
    text = report.render(aggregate(rows), sealed=sealed())
    line = next(ln for ln in text.splitlines() if "cost/solved task" in ln)
    assert "—" in line and "0.000" not in line


def test_a_run_of_only_skips_prints_an_em_dash_for_the_solve_rate() -> None:
    rows = [record("p1", outcome=Outcome.INFRA_SKIP)]
    text = report.render(aggregate(rows), sealed=sealed())
    line = next(ln for ln in text.splitlines() if "solve rate" in ln)
    assert "—" in line and "0.0%" not in line


def test_an_empty_run_renders_without_claiming_anything() -> None:
    text = report.render({})
    assert "_no arms_" in text


# --- paired section -----------------------------------------------------------


def test_the_paired_section_refuses_to_look_like_a_gate() -> None:
    """The decision rule needs the preregistered interval, which does not exist until
    Phase 6. A point estimate rendered without that caveat would read as a pass."""
    rows = [
        *fixture_rows(),
        record("p1", arm="baseline", outcome=Outcome.FAIL),
        record("p2", arm="baseline"),
        record("p3", arm="baseline"),
    ]
    metrics = aggregate(rows)
    text = report.render(metrics, comparison=pair(metrics, "naive", "baseline"))
    assert "Paired comparison" in text
    assert "Point estimates only" in text
    assert "nothing here passes a gate" in text


def test_no_paired_section_when_no_comparison_was_asked_for() -> None:
    assert "Paired comparison" not in report.render(aggregate(fixture_rows()))


# --- CSV ----------------------------------------------------------------------


def test_csv_has_one_row_per_problem_and_a_stable_header() -> None:
    rows = list(csv.DictReader(io.StringIO(report.csv_rows(fixture_rows()))))
    assert len(rows) == 4
    assert tuple(rows[0]) == report.CSV_COLUMNS


def test_csv_states_the_denominator_rather_than_leaving_it_to_be_recomputed() -> None:
    """D14 is the rule most likely to be reimplemented wrongly downstream. A spreadsheet
    re-deriving it from `outcome` is one `!=` away from counting infra_skip as unsolved."""
    rows = {
        r["problem_id"]: r for r in csv.DictReader(io.StringIO(report.csv_rows(fixture_rows())))
    }
    assert rows["p1"]["counts_in_denominator"] == "1"
    assert rows["p4"]["counts_in_denominator"] == "0"
    assert rows["p4"]["outcome"] == "infra_skip"


def test_csv_carries_a_leak_count_and_never_the_leaked_text() -> None:
    """A CSV is the most-shared artifact of a run. An archive holding the oracle outlives
    the prompt that leaked it."""
    leaked = record("p9")
    leaked = ProblemRecord(**{**leaked.__dict__, "leaked_line_numbers": [4, 7]})
    text = report.csv_rows([leaked])
    assert "leaked_lines" in text
    assert ",2," in text
    assert "[4, 7]" not in text
