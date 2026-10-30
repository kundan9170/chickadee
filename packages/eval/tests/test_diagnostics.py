"""The §4 panel renders, and its gaps are gaps rather than zeros.

`test_accounting.py` checks the arithmetic. This file checks the one property that is
specific to Phase 1 and easy to lose: **five of the six panel sources do not exist yet**,
and the panel must say so rather than printing a plausible zero.

The failure being guarded against is not a crash. It is a panel that renders
*"model 0%, packer 0%, spec 0%"* and reads as "nothing failed for those reasons", or a gate
vector of zeros that reads as "every gate passed" — when §4 asks that vector to show *"which
gate bites, which silently isn't running"*. A panel that cannot distinguish "skipped" from
"passed" has become the thing it exists to detect.

So the assertions here are mostly about *absence being visible*: that unavailable sections
name their source, that no unavailable section emits a number, and that the attribution
check refuses to say "holds" when there is nothing to check.
"""

from __future__ import annotations

import re

from chickadee_eval import diagnostics
from chickadee_eval.diagnostics import Availability
from chickadee_eval.manifest import Language
from chickadee_eval.metrics import ArmMetrics, aggregate
from chickadee_eval.problems.polyglot import Outcome
from chickadee_eval.runner import ProblemRecord

HARNESS_FED = (
    diagnostics.attribution_split,
    diagnostics.taxonomy,
    diagnostics.gate_vector,
    diagnostics.reject_reasons,
    diagnostics.edge_class_health,
    diagnostics.cache_and_tokens,
)


def record(
    pid: str,
    *,
    outcome: Outcome = Outcome.PASS,
    arm: str = "a",
    cost: float = 0.0,
    language: Language = Language.PYTHON,
    claim: bool | None = None,
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
        wall_ms=5,
        sandbox="native",
        network_verified=True,
    )


def fixture_arm() -> object:
    rows = [
        record("p1"),
        record("p2", outcome=Outcome.FAIL),
        record("p3", outcome=Outcome.TIMEOUT),
        record("p4", outcome=Outcome.INFRA_SKIP),
        record("j1", language=Language.JAVASCRIPT),
    ]
    return aggregate(rows)["a"]


# --- the panel renders --------------------------------------------------------


def test_the_panel_renders_from_a_fixture_result_set() -> None:
    panel = diagnostics.panel(fixture_arm())  # type: ignore[arg-type]
    text = panel.render()
    assert "Diagnostic panel" in text
    assert len(panel.sections) == len(diagnostics.PANEL_SECTIONS)
    for section in panel.sections:
        assert section.name.lower() in text.lower()


def test_measured_sections_come_before_the_missing_ones() -> None:
    """A reader should reach the real numbers before the list of what is absent — and reach
    that list before concluding the panel is complete."""
    names = [s.name for s in diagnostics.panel(fixture_arm()).sections]  # type: ignore[arg-type]
    first_unavailable = names.index("failure-attribution split")
    assert names.index("D14 accounting") < first_unavailable


# --- empty harness columns render honestly ------------------------------------


def test_every_harness_fed_section_is_unavailable_not_empty() -> None:
    """`UNAVAILABLE` and `EMPTY` are different claims. "We looked and found none" is a
    finding; "we never looked" is not, and only the second is true here."""
    arm = fixture_arm()
    for build in HARNESS_FED:
        section = build(arm)  # type: ignore[arg-type]
        assert section.availability is Availability.UNAVAILABLE, section.name
        assert not section.rows, f"{section.name} emitted rows it has no source for"


def test_an_unavailable_section_names_the_source_that_would_fill_it() -> None:
    """Otherwise the panel says only that something is missing, and a reader cannot tell
    whether it is missing because it was never built or because it broke."""
    arm = fixture_arm()
    expected = {
        "failure-attribution split": "classifier verdicts",
        "taxonomy distribution": "`classify` events",
        "per-gate outcome vector": "`gate_result` events",
        "apply-reject reasons": "patch_applied",
        "edge-class health": "edge_classes",
        "token and cache accounting": "adapter usage",
    }
    for build in HARNESS_FED:
        section = build(arm)  # type: ignore[arg-type]
        assert expected[section.name] in section.reason, section.reason


def test_no_unavailable_section_renders_a_number() -> None:
    """The property that matters. A zero in these tables is a false statement, so the
    rendered text must contain no digits that could be read as a count.

    Checked over the rendered output rather than the fields, because the rendering is what
    a person sees — and a section could carry an empty `rows` while its prose still said
    "0 failures attributed".
    """
    arm = fixture_arm()
    for build in HARNESS_FED:
        text = build(arm).render()  # type: ignore[arg-type]
        body = text.split("\n", 1)[1]
        stripped = re.sub(r"T-3\.5|D18|D19|D41|K-4|Phase 1", "", body)
        assert not re.search(r"\b\d+(\.\d+)?%?\b", stripped), (
            f"{build.__name__} rendered a number where it has no data:\n{text}"
        )


def test_unavailable_sections_share_one_reason_for_being_absent() -> None:
    """Five sections absent for five unrelated reasons would look like decay. They are
    absent for one reason, and saying so once makes that legible."""
    arm = fixture_arm()
    engine_fed = [b for b in HARNESS_FED if b is not diagnostics.cache_and_tokens]
    for build in engine_fed:
        assert diagnostics.PHASE_1_NO_ENGINE in build(arm).reason  # type: ignore[arg-type]


def test_coverage_reports_how_much_of_the_panel_is_real() -> None:
    """Printed at the top of the panel so "incomplete diagnostic" is the first impression
    rather than a discovery."""
    panel = diagnostics.panel(fixture_arm())  # type: ignore[arg-type]
    assert panel.coverage.value is not None
    assert 0 < panel.coverage.value < 0.6, "at Phase 1 most of §4's panel has no source"
    assert len(panel.measured) + len(panel.unavailable) == len(panel.sections)
    assert "Coverage" in panel.render()


# --- the sections that do have data -------------------------------------------


def test_the_accounting_section_shows_the_denominator_and_the_exclusion() -> None:
    section = diagnostics.accounting(fixture_arm())  # type: ignore[arg-type]
    rows = dict(section.rows)
    assert section.measured
    assert rows["rows"] == "5"
    assert rows["denominator"] == "4"
    assert rows["infra_skip (excluded)"] == "1"
    assert rows["solve rate"] == "50.0%"


def test_the_outcome_distribution_keeps_its_zeros() -> None:
    """A distribution that dropped `timeout` because it did not occur would read
    identically to one where timeouts were never measured."""
    rows = dict(diagnostics.outcome_distribution(fixture_arm()).rows)  # type: ignore[arg-type]
    assert set(rows) >= {o.value for o in Outcome}
    assert rows["pass"] == "2"
    assert rows["infra_skip"] == "1"


def test_a_run_with_no_rows_is_empty_not_unavailable() -> None:
    """The distinction the `EMPTY` state exists for: the source ran and produced nothing."""
    section = diagnostics.outcome_distribution(ArmMetrics("empty", []))
    assert section.availability is Availability.EMPTY
    assert "no rows" in section.reason


def test_the_integrity_section_surfaces_reasons_to_distrust_the_number() -> None:
    """A run with unsealed rows and a plant attempt has a solve rate, and the solve rate is
    not the interesting part."""
    rows = dict(diagnostics.integrity(fixture_arm()).rows)  # type: ignore[arg-type]
    assert "oracle-plant attempts" in rows
    assert "rows without a verified seal" in rows


# --- attribution split sums to 100% of failures -------------------------------


def test_the_attribution_check_is_not_established_without_a_classifier() -> None:
    """The heart of it. A boolean check would return True — "the split is consistent" — for
    a split that does not exist, and that passing assertion would sit in a suite reading as
    evidence. Three-valued, so it cannot."""
    check = diagnostics.attribution_covers_all_failures(fixture_arm())  # type: ignore[arg-type]
    assert not check.established
    assert not check.holds
    assert diagnostics.PHASE_1_NO_ENGINE in check.reason
    assert check.failures == 2, "it still counts the failures it could not attribute"


def test_the_attribution_check_holds_when_every_failure_has_a_verdict() -> None:
    arm = fixture_arm()
    verdicts = {"p2": "model", "p3": "spec"}
    check = diagnostics.attribution_covers_all_failures(arm, verdicts)  # type: ignore[arg-type]
    assert check.established and check.holds
    assert check.attributed == check.failures == 2


def test_the_attribution_check_fails_when_a_failure_is_unattributed() -> None:
    """Established and false, which is a different state from unestablished — a reader can
    act on it."""
    arm = fixture_arm()
    check = diagnostics.attribution_covers_all_failures(arm, {"p2": "model"})  # type: ignore[arg-type]
    assert check.established and not check.holds
    assert "p3" in check.reason


def test_infra_skips_are_not_failures_needing_attribution() -> None:
    """D14 again, one layer up: a problem never attempted has no failure cause, and
    demanding a verdict for it would make the split impossible to satisfy."""
    arm = fixture_arm()
    check = diagnostics.attribution_covers_all_failures(arm, {"p2": "model", "p3": "spec"})  # type: ignore[arg-type]
    assert check.holds, "p4 was an infra_skip and must not be counted as a failure"
