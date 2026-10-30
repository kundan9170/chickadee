"""Two fixture runs → a correct before/after table.

§10.6 requires this on every schema, render or gate PR, so the bar is that a reviewer can
read the output and act without opening either archive. Three properties carry that:

- **regressions are visible even under a net gain** — six fixed and four broken nets to +2,
  and the four are the finding;
- **problems present on one side only are not deltas** — the D14 distinction applied to a
  comparison, so a run that dropped twenty problems cannot read as an improvement;
- **incomparable archives say so** — two runs under different dependency trees are two
  experiments, not a before and an after.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict

import pytest

from chickadee_eval import differ
from chickadee_eval.manifest import Language
from chickadee_eval.metrics import aggregate
from chickadee_eval.problems.polyglot import Outcome
from chickadee_eval.runner import ProblemRecord


def record(
    pid: str,
    *,
    outcome: Outcome = Outcome.PASS,
    arm: str = "a",
    cost: float = 0.1,
    wall: int = 10,
    deps: str = "",
    sandbox: str = "native",
) -> ProblemRecord:
    solved = outcome is Outcome.PASS
    return ProblemRecord(
        problem_id=pid,
        language=Language.PYTHON,
        arm=arm,
        outcome=outcome,
        solved=solved,
        solved_claim=solved,
        exit_code=0 if solved else 1,
        attempts=1,
        cost_usd=cost,
        wall_ms=wall,
        sandbox=sandbox,
        network_verified=True,
        deps_digest=deps,
    )


def write_archive(root: pathlib.Path, rows: list[ProblemRecord]) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.jsonl").write_text(
        "".join(json.dumps(asdict(r), sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return root


# --- the before/after table ---------------------------------------------------


def test_two_fixture_runs_produce_a_correct_table() -> None:
    before = [
        record("p1", outcome=Outcome.FAIL),
        record("p2"),
        record("p3"),
        record("p4", outcome=Outcome.FAIL),
    ]
    after = [
        record("p1"),  # fixed
        record("p2", outcome=Outcome.FAIL),  # regressed
        record("p3"),  # unchanged
        record("p4", outcome=Outcome.TIMEOUT),  # still failing, different outcome
    ]
    diff = differ.compare(before, after)

    assert [d.problem_id for d in diff.fixed] == ["p1"]
    assert [d.problem_id for d in diff.regressions] == ["p2"]
    assert diff.solve_delta == 0

    text = diff.render()
    assert "REGRESSED" in text
    assert "`p3`" not in text, "an unchanged problem does not belong in the change table"
    assert "`p4`" in text, "an outcome change with the same solved flag is still a change"


def test_regressions_are_listed_first() -> None:
    """A net gain with regressions underneath is two findings, and the ordering is what
    stops the second being missed."""
    before = [
        record("p1"),
        record("p2", outcome=Outcome.FAIL),
        record("p3", outcome=Outcome.FAIL),
    ]
    after = [record("p1", outcome=Outcome.FAIL), record("p2"), record("p3")]
    # Within the table, not the whole document: the summary line above it already says
    # "fixed: N", and comparing against that would pass regardless of the row order.
    table = differ.compare(before, after).render().split("| problem |", 1)[1]
    assert table.index("REGRESSED") < table.index("fixed")


def test_a_net_gain_does_not_hide_a_regression() -> None:
    before = [record(f"p{i}", outcome=Outcome.FAIL) for i in range(6)] + [record("keeper")]
    after = [record(f"p{i}") for i in range(6)] + [record("keeper", outcome=Outcome.FAIL)]
    diff = differ.compare(before, after)
    assert diff.solve_delta == 5
    assert len(diff.regressions) == 1, "the regression survives the net gain"


def test_nothing_changed_says_so() -> None:
    rows = [record("p1"), record("p2", outcome=Outcome.FAIL)]
    assert "No problem changed outcome" in differ.compare(rows, list(rows)).render()


# --- problems on one side only ------------------------------------------------


def test_a_problem_only_in_the_after_set_is_not_an_improvement() -> None:
    diff = differ.compare([record("p1")], [record("p1"), record("p2")])
    assert diff.added == ["p2"]
    assert len(diff.deltas) == 1
    assert diff.solve_delta == 0, "an added problem is not a fix"
    assert "not an improvement" in diff.render()


def test_a_problem_only_in_the_before_set_is_not_a_regression() -> None:
    """A run that dropped twenty problems must not read as twenty regressions — nor as an
    improvement from a smaller denominator."""
    diff = differ.compare([record("p1"), record("p2")], [record("p1")])
    assert diff.removed == ["p2"]
    assert not diff.regressions
    assert "not a regression" in diff.render()


# --- cost ---------------------------------------------------------------------


def test_cost_deltas_are_summed_over_paired_problems() -> None:
    diff = differ.compare([record("p1", cost=1.0)], [record("p1", cost=1.5)])
    assert diff.cost_delta_usd == pytest.approx(0.5)
    assert diff.cost_ratio.value == pytest.approx(1.5)


def test_a_zero_cost_before_leaves_the_ratio_undefined() -> None:
    """Dividing by nothing is not an infinite increase; it is a ratio that does not exist."""
    diff = differ.compare([record("p1", cost=0.0)], [record("p1", cost=1.0)])
    assert diff.cost_ratio.value is None
    assert "—" in diff.render()


# --- comparability ------------------------------------------------------------


def test_different_dependency_trees_are_reported_as_incomparable() -> None:
    """Two runs scored under different jest versions are two experiments. The table still
    renders — a reviewer may want to look — but it is labelled."""
    diff = differ.compare([record("p1", deps="aaa")], [record("p1", deps="bbb")])
    assert not diff.comparable
    assert any(i.field == "dependency tree" for i in diff.incomparable)
    assert "not a before/after pair" in diff.render()


def test_different_sandboxes_are_reported_as_incomparable() -> None:
    """Sealing cost one frontier model 14.1pp. Comparing a sealed run against an
    unsealed one measures the seal, not the change."""
    diff = differ.compare([record("p1", sandbox="native")], [record("p1", sandbox="none")])
    assert any(i.field == "sandbox" for i in diff.incomparable)


def test_matching_runs_are_comparable() -> None:
    diff = differ.compare([record("p1", deps="aaa")], [record("p1", deps="aaa")])
    assert diff.comparable
    assert "not a before/after pair" not in diff.render()


# --- arms ---------------------------------------------------------------------


def test_several_arms_without_a_selection_raises() -> None:
    """Averaging arms into a single delta is the mistake a differ exists to prevent."""
    rows = [record("p1", arm="x"), record("p1", arm="y")]
    with pytest.raises(differ.ArchiveError, match="several arms"):
        differ.compare(rows, rows)


def test_an_arm_can_be_selected() -> None:
    before = [record("p1", arm="x"), record("p1", arm="y", outcome=Outcome.FAIL)]
    after = [record("p1", arm="x", outcome=Outcome.FAIL), record("p1", arm="y")]
    assert differ.compare(before, after, arm="x").regressions
    assert differ.compare(before, after, arm="y").fixed


# --- reading archives ---------------------------------------------------------


def test_archives_round_trip_through_disk(tmp_path: pathlib.Path) -> None:
    a = write_archive(tmp_path / "before", [record("p1", outcome=Outcome.FAIL)])
    b = write_archive(tmp_path / "after", [record("p1")])
    diff = differ.compare_archives(a, b)
    assert [d.problem_id for d in diff.fixed] == ["p1"]


def test_an_archive_from_a_later_version_still_compares(tmp_path: pathlib.Path) -> None:
    """Unknown keys are dropped rather than raising. §10.6 demands this diff on schema PRs,
    which is exactly when the two sides have different fields — refusing would make the
    tool useless in its own required case."""
    root = tmp_path / "after"
    root.mkdir()
    blob = asdict(record("p1"))
    blob["a_field_from_the_future"] = 42
    (root / "results.jsonl").write_text(json.dumps(blob) + "\n", encoding="utf-8")

    diff = differ.compare(
        load := [record("p1", outcome=Outcome.FAIL)], differ.load_records(root)
    )
    assert load and [d.problem_id for d in diff.fixed] == ["p1"]


def test_a_missing_archive_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(differ.ArchiveError, match=r"no results\.jsonl"):
        differ.load_records(tmp_path / "nowhere")


def test_an_empty_archive_raises(tmp_path: pathlib.Path) -> None:
    """Zero rows is not a run to compare against, and pairing against it would report every
    problem as `removed` — technically true, useless as a diff."""
    root = tmp_path / "empty"
    root.mkdir()
    (root / "results.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(differ.ArchiveError, match="no rows"):
        differ.load_records(root)


def test_a_corrupt_line_names_its_line_number(tmp_path: pathlib.Path) -> None:
    """A valid row first, so the failure really is on line 2 — the first draft put `{}`
    there, which raised on line 1 and made the assertion pass for the wrong reason."""
    root = tmp_path / "bad"
    root.mkdir()
    good = json.dumps(asdict(record("p1")))
    (root / "results.jsonl").write_text(f"{good}\nnot json\n", encoding="utf-8")
    with pytest.raises(differ.ArchiveError, match=r"results\.jsonl:2 is not JSON"):
        differ.load_records(root)


def test_a_json_line_that_is_not_a_result_row_is_named_too(tmp_path: pathlib.Path) -> None:
    """`{}` parses as JSON and then fails to build a record. That was a bare `TypeError`
    naming `ProblemRecord.__init__` — a message about this code rather than about the
    file the reader has to fix."""
    root = tmp_path / "shape"
    root.mkdir()
    (root / "results.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(differ.ArchiveError, match=r"results\.jsonl:1 is not a result row"):
        differ.load_records(root)


# --- the one-line summary -----------------------------------------------------


def test_the_summary_line_reports_both_rates() -> None:
    before = aggregate([record("p1"), record("p2", outcome=Outcome.FAIL)])["a"]
    after = aggregate([record("p1"), record("p2")])["a"]
    line = differ.summarize(before, after)
    assert "50.0%" in line and "100.0%" in line and "+50.0 pp" in line


# --- review findings on #68 ---------------------------------------------------


def test_a_json_value_that_is_not_an_object_is_named(tmp_path: pathlib.Path) -> None:
    """`null`, `[]` and `3` are valid JSON and none of them is a row.

    Previously `blob.items()` raised a bare `AttributeError` — outside the handled type, so
    `compare_archives` lost the archive path and line number and reported that `None` has no
    attributes, which tells a reader nothing about the file they have to fix.
    """
    for payload in ("null", "[]", "3", '"a string"'):
        root = tmp_path / f"case-{payload.strip('"')}"
        root.mkdir()
        (root / "results.jsonl").write_text(payload + "\n", encoding="utf-8")
        with pytest.raises(differ.ArchiveError, match="not a result object"):
            differ.load_records(root)


def test_differing_network_verification_is_incomparable() -> None:
    """`sandbox` records which backend was applied; `network_verified` records whether a
    socket was proven to fail. Two runs can agree on the first and differ on the second —
    `Seal` keeps them apart for exactly this reason — and §5.2 puts the seal's effect at
    14.1pp, so comparing across it measures the seal rather than the change.
    """
    before = [record("p1", sandbox="native")]
    after = [
        ProblemRecord(**{**record("p1", sandbox="native").__dict__, "network_verified": False})
    ]
    diff = differ.compare(before, after)
    assert not diff.comparable
    assert any(i.field == "network verification" for i in diff.incomparable)
    assert "not a before/after pair" in diff.render()


def test_matching_verification_stays_comparable() -> None:
    """The check must not fire on every diff — a guard that always trips gets deleted."""
    diff = differ.compare([record("p1")], [record("p1", outcome=Outcome.FAIL)])
    assert diff.comparable


def test_the_summary_refuses_to_invent_a_delta_when_a_rate_is_undefined() -> None:
    """The sharpest of the three: `(value or 0)` turned an undefined rate into `0.0` and
    printed a numeric pp change for a comparison that was never measured — the exact
    substitution `Ratio` exists to prevent, in the module that argues against it."""
    before = aggregate([record("p1", outcome=Outcome.INFRA_SKIP)])["a"]
    after = aggregate([record("p1")])["a"]
    assert before.solve_rate.value is None

    line = differ.summarize(before, after)
    assert "(—)" in line, line
    assert "+100.0 pp" not in line and "0.0 pp" not in line


def test_the_summary_still_reports_a_real_delta() -> None:
    before = aggregate([record("p1", outcome=Outcome.FAIL), record("p2")])["a"]
    after = aggregate([record("p1"), record("p2")])["a"]
    assert "+50.0 pp" in differ.summarize(before, after)
