"""The budget is the session's: an episode starts its own run scope and cannot
reset what the session has spent."""

from __future__ import annotations

from box_helpers import BoxRig, design

from chickadee.box.report import CoderReport


def test_a_second_episode_cannot_reset_the_session_budget(rig: BoxRig) -> None:
    box = rig.box()
    first = box.open(design()).report()
    # It plans nothing, so it is `PARTIAL`; the spend is what is checked.
    assert isinstance(first, CoderReport) and first.outcome == "PARTIAL"
    rig.ledger.charge_wall(500)
    second = box.open(design()).report()
    assert isinstance(second, CoderReport)
    assert rig.ledger.spend("session").wall_ms >= 500
    assert rig.ledger.spend("run").wall_ms < 500  # the second episode's own scope
