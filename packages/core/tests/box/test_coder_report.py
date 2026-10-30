"""The report v2: handles, never the diff; `why` iff
the outcome is not `COMPLETED`; handles iff a run was journaled; the request is exactly one kind
of work; the goldens hold."""

from __future__ import annotations

import pytest
from box_helpers import BoxRig, design, worker_call
from box_helpers import athena_call as acall
from engines import answer, calls
from pydantic import ValidationError

from chickadee.box.export import GOLDENS
from chickadee.box.report import CoderReport
from chickadee.box.request import ApprovedDesign, CoderRequest, OneTask
from chickadee.journal.projection import RunSpend
from chickadee.journal.writer import Journal
from chickadee.schema import sha256_hex
from chickadee.state.roots import roots as resolve_roots

MARK = "41_235"
"""What the edit writes, and nothing else in the episode says."""


def _one_edit(rig: BoxRig) -> None:
    rig.athena_script = [
        calls(acall("plan", "a0", text="1. **Return 2.** `go()` returns 2.")),
        calls(acall("delegate", "a1", title="Return 2", brief="## Task\n`go()` returns 2.")),
        calls(acall("review", "a2", task="T1", answer="accept")),
        answer("Done: go() returns 2."),
    ]
    rig.worker_script = [
        calls(worker_call("edit", "w1", path="app.py", old="return 1", new=f"return {MARK}")),
        answer("T1: go() returns 2."),
    ]


def test_the_report_names_the_diff_and_never_carries_it(rig: BoxRig) -> None:
    _one_edit(rig)
    report = rig.box().open(design()).report()
    assert isinstance(report, CoderReport) and report.handles is not None
    # Only the diff holds the changed line: the goal, the briefs and the replies do not.
    assert MARK not in report.model_dump_json()
    diff = report.handles.diff
    assert diff is not None and diff.kind == "diff" and diff.run_id == report.run_id
    ws = rig.workspaces[0]
    journal = Journal.open_for_resume(resolve_roots(ws.root), report.run_id or "")
    try:
        assert f"+    return {MARK}".encode() in journal.get_blob(diff.content_sha)
    finally:
        journal.close()


def test_why_is_set_iff_not_completed() -> None:
    with pytest.raises(ValidationError):
        CoderReport(outcome="BLOCKED", cost=RunSpend())
    with pytest.raises(ValidationError):
        CoderReport(outcome="COMPLETED", why="it ended", cost=RunSpend())
    CoderReport(outcome="PARTIAL", why="T2 is open", cost=RunSpend())


def test_handles_are_named_exactly_when_a_run_was_journaled() -> None:
    before = CoderReport.before_run("ABORTED", "the person cancelled at stop 1")
    assert before.run_id is None and before.handles is None
    with pytest.raises(ValidationError):
        CoderReport(outcome="ABORTED", why="w", run_id="2026-10-15-0a1b2c3d", cost=RunSpend())


def test_a_request_is_exactly_one_kind_of_work() -> None:
    text, brief = "# A design\n", "## Task\nDo it.\n"
    approved = ApprovedDesign(text=text, sha=sha256_hex(text), template_sha="t")
    task = OneTask(title="Do it", brief=brief, sha=sha256_hex(brief), template_sha="t")
    with pytest.raises(ValidationError, match="exactly one"):
        CoderRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        CoderRequest(design=approved, task=task)
    assert CoderRequest(design=approved).work_sha == sha256_hex(text)
    assert CoderRequest(task=task).work_sha == sha256_hex(brief)


def test_the_goal_is_the_tasks_title_or_the_designs_first_line() -> None:
    text = "# Make go return 2\n\n## What changes\n"
    approved = ApprovedDesign(text=text, sha=sha256_hex(text), template_sha="t")
    assert CoderRequest(design=approved).goal == "Make go return 2"
    task = OneTask(title="Return 2", brief="b", sha=sha256_hex("b"), template_sha="t")
    assert CoderRequest(task=task).goal == "Return 2"


@pytest.mark.parametrize("name", sorted(GOLDENS.contracts))
def test_golden_matches_regeneration(name: str) -> None:
    golden = GOLDENS.path(name)
    assert golden.exists(), f"missing {golden}; run python -m chickadee.box.export"
    assert golden.read_text(encoding="utf-8") == GOLDENS.render(name), (
        f"{name} drifted from its golden: regenerate with `python -m chickadee.box.export` "
        "and review the diff as a contract change (D54)"
    )
