"""Charge across scopes with packet reset; 80% and 100% crossings exactly once each; the
boundary at the exact cap; a crossed cap is a resumable BUDGET_CAP terminal."""

from __future__ import annotations

import pytest
from budget_helpers import CALL_USD, CODER_MODEL, FRONTIER_MODEL, DrivenClock, ledger, usage

from chickadee.budget.manager import BUDGET_CAP, BudgetCrossing
from chickadee.budget.spend import Spend
from chickadee.errors import HarnessBug


def test_a_charge_lands_in_every_scope_and_a_packet_reset_clears_only_the_packet() -> None:
    led = ledger(cap=1.0)
    led.start_run("r1")
    led.start_packet("p1")
    assert led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL) == pytest.approx(
        CALL_USD
    )
    for scope in ("session", "run", "packet"):
        assert led.spend(scope).cost_usd == pytest.approx(CALL_USD), scope
        assert led.spend(scope).tokens_in == 1000 and led.spend(scope).tokens_out == 1000
    led.start_packet("p2")
    assert led.spend("packet").cost_usd == 0.0
    assert led.spend("run").cost_usd == pytest.approx(CALL_USD)
    assert led.spend("session").cost_usd == pytest.approx(CALL_USD)


def test_a_new_episode_resets_the_run_scope_and_never_the_session() -> None:
    """The budget belongs to the session lineage; a delegated call does not reset the
    limit."""
    led = ledger(cap=1.0)
    led.start_run("r1")
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    led.start_run("r2")
    assert led.spend("run").cost_usd == 0.0
    assert led.spend("session").cost_usd == pytest.approx(CALL_USD)


def test_a_call_outside_any_packet_charges_run_and_session_only() -> None:
    led = ledger(cap=1.0)
    led.start_run("r1")
    led.for_rung("low-param").charge_usage("", usage(), FRONTIER_MODEL)
    assert led.spend("packet").cost_usd == 0.0
    assert led.spend("run").cost_usd > 0 and led.spend("session").cost_usd > 0


def test_failed_calls_and_estimated_usage_are_charged_and_counted() -> None:
    led = ledger(cap=1.0)
    led.for_rung("low-param").charge_usage("p1", usage(estimated=True), CODER_MODEL)
    spend = led.spend("session")
    assert spend.cost_usd == pytest.approx(CALL_USD)
    assert spend.estimated_calls == 1


def test_warning_and_hard_cap_each_fire_exactly_once() -> None:
    """Cap $0.01, one call $0.003: the 80% warning fires on the third call ($0.009), the
    cap on the fourth ($0.012), and neither ever again."""
    led = ledger(cap=0.01, fractions=(0.8,))
    led.start_packet("p1")
    seen: list[BudgetCrossing] = []
    for _ in range(6):
        led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
        seen.extend(led.crossings())
    kinds = [(c.kind, c.fraction, c.terminal) for c in seen]
    assert kinds == [("warning", 0.8, None), ("hard_cap", None, BUDGET_CAP)]
    assert seen[0].spent_usd == pytest.approx(0.009) and seen[0].cap_usd == 0.01
    assert seen[1].spent_usd == pytest.approx(0.012)
    assert all(c.scope == "session" for c in seen)
    assert led.crossings() == (), "drained means drained"


def test_the_boundary_is_the_exact_cap() -> None:
    """Reaching the cap exactly is crossing it: $0.012 of cap on four $0.003 calls."""
    led = ledger(cap=0.012, fractions=(0.5,))
    for _ in range(3):
        led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    assert [c.kind for c in led.crossings()] == ["warning"]
    assert led.for_rung("low-param").may_start() and led.terminal() is None
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    (hit,) = led.crossings()
    assert hit.kind == "hard_cap" and hit.spent_usd == pytest.approx(0.012)
    assert not led.for_rung("low-param").may_start() and led.terminal() == BUDGET_CAP
    assert led.exhausted


def test_a_warning_never_ends_the_run_and_multiple_fractions_fire_in_order() -> None:
    led = ledger(cap=0.03, fractions=(0.2, 0.5, 0.8))
    fired: list[float | None] = []
    for _ in range(11):
        led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
        for c in led.crossings():
            fired.append(c.fraction)
            if c.kind == "warning":
                assert c.terminal is None and led.for_rung("low-param").may_start()
    assert fired == [0.2, 0.5, 0.8, None]


def test_one_operation_can_overshoot_but_nothing_starts_after() -> None:
    """Stop at the first crossing, overshoot bounded by one operation."""
    led = ledger(cap=0.004)
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    assert led.for_rung("low-param").may_start()
    led.for_rung("low-param").charge_usage(
        "p1", usage(), CODER_MODEL
    )  # the crossing call itself is charged in full
    assert led.spend("session").cost_usd == pytest.approx(0.006)
    assert not led.for_rung("low-param").may_start()


def test_the_terminal_is_budget_cap_never_a_silent_stop() -> None:
    led = ledger(cap=0.001)
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    # One charge crossed the warning and the cap at once: both are recorded, in order.
    warning, hit = led.crossings()
    assert warning.kind == "warning" and warning.terminal is None
    assert hit.terminal == "BUDGET_CAP" and led.terminal() == "BUDGET_CAP"
    payload = hit.payload()
    assert payload.kind == "hard_cap" and payload.terminal is True and payload.scope == "run"


def test_the_packet_warning_is_diagnostic_and_per_packet() -> None:
    led = ledger(cap=1.0, packet_warning=0.005)
    led.start_packet("p1")
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    (warn,) = led.crossings()
    assert warn.scope == "packet" and warn.kind == "warning" and warn.terminal is None
    assert warn.payload().scope == "packet"
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    assert led.crossings() == (), "once per packet"
    led.start_packet("p2")
    led.for_rung("low-param").charge_usage("p2", usage(), CODER_MODEL)
    led.for_rung("low-param").charge_usage("p2", usage(), CODER_MODEL)
    assert [c.scope for c in led.crossings()] == ["packet"], "re-armed by the reset"
    assert led.for_rung("low-param").may_start(), (
        "a packet warning never substitutes for the run cap"
    )


def test_no_cap_means_no_crossings_and_nothing_stops() -> None:
    led = ledger(cap=None)
    for _ in range(50):
        led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)
    assert (
        led.crossings() == ()
        and led.for_rung("low-param").may_start()
        and led.terminal() is None
    )


def test_wall_charging_clamps_at_zero_when_the_clock_goes_backwards() -> None:
    clock = DrivenClock(at=100.0)
    led = ledger(cap=1.0, clock=clock)
    led.mark()
    clock.advance(1.5)
    assert led.charge_wall_since() == 1500
    led.mark()
    clock.advance(-5.0)
    assert led.charge_wall_since() == 0
    assert led.spend("session").wall_ms == 1500
    led.charge_wall(-300)
    assert led.spend("session").wall_ms == 1500
    with pytest.raises(HarnessBug, match="mark"):
        ledger(cap=1.0).charge_wall_since()


def test_resume_from_journaled_spend_continues_from_what_was_spent() -> None:
    """The cap is resumable. A run topped up by config continues from its spend, and
    crossings already journaled are not re-emitted."""
    led = ledger(cap=0.02, fractions=(0.5, 0.8))
    led.resume_from(Spend("session", 4000, 4000, 0, 0.012, 900))
    assert led.spend("session").cost_usd == pytest.approx(0.012)
    assert led.crossings() == (), "the 50% warning was journaled by the run that crossed it"
    led.for_rung("low-param").charge_usage(
        "p1", usage(), CODER_MODEL
    )  # 0.015: no new fraction yet
    assert led.crossings() == ()
    led.for_rung("low-param").charge_usage("p1", usage(), CODER_MODEL)  # 0.018: past 80%
    assert [c.fraction for c in led.crossings()] == [0.8]
    exhausted = ledger(cap=0.01)
    exhausted.resume_from(Spend("session", 0, 0, 0, 0.011, 0))
    assert (
        not exhausted.for_rung("low-param").may_start() and exhausted.terminal() == BUDGET_CAP
    )


def test_unknown_scope_is_a_harness_bug() -> None:
    with pytest.raises(HarnessBug, match="unknown spend scope"):
        ledger().spend("galaxy")
