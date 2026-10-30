"""The repository WRITE lease is taken at `open` and released on every exit: a report, a blocked
episode, an abandoned run, and a raise in the run."""

from __future__ import annotations

import pytest
from box_helpers import BoxRig, design, lease_is_free

from chickadee.box.report import CoderReport
from chickadee.errors import HarnessBug
from chickadee.schema.adapter import Completion
from chickadee.workspace import Workspace


def _unreachable(request: object) -> Completion:
    raise ConnectionError("the model is unreachable")


def test_open_holds_the_lease_and_a_report_releases_it(rig: BoxRig) -> None:
    run = rig.box().open(design())
    assert not lease_is_free(rig.source)
    run.report()
    assert lease_is_free(rig.source)


def test_a_blocked_episode_releases_it(rig: BoxRig) -> None:
    rig.athena_script = [_unreachable]
    report = rig.box().open(design()).report()
    assert isinstance(report, CoderReport) and report.outcome == "BLOCKED"
    assert lease_is_free(rig.source)


def test_an_abandoned_run_releases_it(rig: BoxRig) -> None:
    run = rig.box().open(design())
    run.abandon()
    assert lease_is_free(rig.source)
    with pytest.raises(HarnessBug):
        run.report()


def test_a_raise_in_the_run_releases_it(rig: BoxRig) -> None:
    def crash(ws: Workspace) -> None:
        raise RuntimeError("the index could not be opened")

    rig.parts_hook = crash
    run = rig.box().open(design())
    # The error is the report, not raised into the call that asked.
    report = run.report()
    assert isinstance(report, CoderReport) and report.outcome == "BLOCKED"
    assert report.why is not None and "the index could not be opened" in report.why
    assert lease_is_free(rig.source)


def test_abandoning_a_reported_run_releases_nothing_twice(rig: BoxRig) -> None:
    run = rig.box().open(design())
    run.report()
    run.abandon()
    assert lease_is_free(rig.source)
