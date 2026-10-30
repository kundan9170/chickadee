"""Pins of what the session's ledger does, before the budget theme makes it one ledger with
rungs built in.

Landed on the code before it, where `engine._SessionLedger` was the `zeus` rung's own ledger
and `_RungLedger` its view for every other rung, and passing unchanged after it: `T-3.74`
changed only this file's setup (how a ledger is built, how a rung's view is taken), never an
expected value.

Every call costs $1.10: 100 prompt tokens and 10 completion tokens at $10,000 per million.
"""

from __future__ import annotations

import errno
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from journal_pins import scratch_roots

from chickadee.budget.manager import BudgetCaps, CostLedger, RungLedger
from chickadee.budget.prices import PriceTable
from chickadee.budget.spend import Spend
from chickadee.config import PriceRow
from chickadee.context.lines import SpendLine
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.schema.adapter import Usage

USAGE = Usage(prompt_tokens=100, completion_tokens=10)
CALL = 1.10
NOTHING = Spend("session", 0, 0, 0, 0.0, 0)
PINNED_RUNGS = ("zeus", "planner", "coder", "scout", "critic")
"""The rungs these pins were written against, declared to this ledger as a config declares
its own: the ledger reads no name of its own, so its answers do not move."""
PRICES = PriceTable.from_config({"m": PriceRow(price_in=10_000.0, price_out=10_000.0)})


def caps(total: float | None, fractions: tuple[float, ...] = (0.8,)) -> BudgetCaps:
    return BudgetCaps(hard_cost_cap_usd=total, warning_fractions=fractions)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    with Session.create(sessions_root(scratch_roots(tmp_path)), "S-pins") as s:
        yield s


# -- setup: how a ledger and a rung's view are had (changes with the ledger's shape) --------


def ledger_for(
    session: Session, total: float | None, fractions: tuple[float, ...] = (0.8,)
) -> CostLedger:
    """The session's ledger, recording into `session`."""
    return CostLedger(
        PRICES, caps(total, fractions), record=session.record_spend, rungs=PINNED_RUNGS
    )


def view(ledger: CostLedger, rung: str) -> RungLedger:
    """`rung`'s view: what a loop on that rung holds."""
    return ledger.for_rung(rung)


def restore(
    ledger: CostLedger,
    total: float | None,
    spent: Spend = NOTHING,
    rung_caps: Mapping[str, float] | None = None,
    fractions: tuple[float, ...] = (0.8,),
) -> None:
    ledger.restore(caps(total, fractions), spent, rung_caps or {}, {})


def lines(session: Session) -> list[SpendLine]:
    return [r for _e, r in session.state.lines() if isinstance(r, SpendLine)]


# -- the record of a charge ----------------------------------------------------------


def test_a_charge_with_no_packet_is_one_spend_line_naming_its_rung(session: Session) -> None:
    ledger = ledger_for(session, 100.0)
    view(ledger, "zeus").charge_usage("", USAGE, "m")
    view(ledger, "scout").charge_usage("", USAGE, "m")
    written = lines(session)
    assert [(ln.rung, ln.model, ln.cost_usd) for ln in written] == [
        ("zeus", "m", pytest.approx(CALL)),
        ("scout", "m", pytest.approx(CALL)),
    ]


@pytest.mark.parametrize("scope", ["T1", ""])
def test_a_journaled_charge_is_never_a_spend_line(session: Session, scope: str) -> None:
    """A box role's calls are `model_call` in its run's journal; a line here too would be
    counted twice. Athena's outside any task (scope "") included: keyed by
    the empty scope, they were once recorded here as well."""
    ledger = ledger_for(session, 100.0)
    view(ledger, "coder").charge_usage(scope, USAGE, "m", journaled=True)
    assert lines(session) == []
    assert ledger.spend("session").cost_usd == pytest.approx(CALL)


def test_a_record_that_cannot_be_written_leaves_the_charge_counted(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(*_: Any, **__: Any) -> Any:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(session, "record_spend", full)
    ledger = ledger_for(session, 100.0)
    with pytest.raises(OSError, match="No space"):
        view(ledger, "zeus").charge_usage("", USAGE, "m")
    assert ledger.spend("session").cost_usd == pytest.approx(CALL)


# -- caps and refusals ----------------------------------------------------------


def test_the_total_at_its_cap_stops_every_rung_and_says_so(session: Session) -> None:
    ledger = ledger_for(session, 2.0)
    for _ in range(2):
        view(ledger, "zeus").charge_usage("", USAGE, "m")
    total = "the session family's spend cap of $2.00 is reached ($2.20 spent)"
    for rung in ("zeus", "scout", "coder", "planner", "critic"):
        assert not view(ledger, rung).may_start(), rung
        assert ledger.stopped_by(rung) == total, rung


def test_a_rung_at_its_cap_stops_only_that_rung(session: Session) -> None:
    ledger = ledger_for(session, 100.0)
    restore(ledger, 100.0, rung_caps={"scout": 1.0})
    view(ledger, "scout").charge_usage("", USAGE, "m")
    assert not view(ledger, "scout").may_start()
    assert ledger.stopped_by("scout") == (
        "the scout rung's spend cap of $1.00 is reached ($1.10 spent on it); "
        "the family's total still has room"
    )
    for rung in ("zeus", "coder", "planner", "critic"):
        assert view(ledger, rung).may_start(), rung
        assert ledger.stopped_by(rung) is None, rung


def test_the_totals_refusal_is_named_before_a_rungs(session: Session) -> None:
    ledger = ledger_for(session, 2.0)
    restore(ledger, 2.0, rung_caps={"zeus": 1.0})
    for _ in range(2):
        view(ledger, "zeus").charge_usage("", USAGE, "m")
    assert ledger.stopped_by("zeus") == (
        "the session family's spend cap of $2.00 is reached ($2.20 spent)"
    )


def test_a_lifted_total_stops_nothing_and_warns_of_nothing(session: Session) -> None:
    ledger = ledger_for(session, 2.0)
    restore(ledger, None)
    for _ in range(5):
        view(ledger, "zeus").charge_usage("", USAGE, "m")
    assert view(ledger, "zeus").may_start()
    assert ledger.stopped_by("zeus") is None
    assert ledger.crossings() == ()
    assert ledger.spend("session").cost_usd == pytest.approx(5 * CALL)


# -- crossings ------------------------------------------------------------------


def _drained(ledger: CostLedger) -> list[tuple[str, float | None]]:
    return [(c.kind, c.fraction) for c in ledger.crossings()]


def test_each_crossing_queues_once(session: Session) -> None:
    """Cap $10.00, fractions 0.5 and 0.8: the fifth call ($5.50) crosses half, the eighth
    ($8.80) four-fifths, the tenth ($11.00) the cap; none fires again."""
    ledger = ledger_for(session, 10.0, (0.5, 0.8))
    seen: list[tuple[str, float | None]] = []
    for _ in range(12):
        view(ledger, "zeus").charge_usage("p", USAGE, "m")
        seen += _drained(ledger)
    assert seen == [("warning", 0.5), ("warning", 0.8), ("hard_cap", None)]


def test_a_restore_with_the_same_caps_emits_nothing_again(session: Session) -> None:
    ledger = ledger_for(session, 10.0, (0.5,))
    for _ in range(5):
        view(ledger, "zeus").charge_usage("p", USAGE, "m")
    assert _drained(ledger) == [("warning", 0.5)]
    restore(ledger, 10.0, ledger.spend("session"), fractions=(0.5,))
    view(ledger, "zeus").charge_usage("p", USAGE, "m")
    assert _drained(ledger) == []


def test_a_moved_cap_re_arms_its_crossings(session: Session) -> None:
    """A raise from $10 to $20 with $5.50 spent: the new cap's half is $10, not yet reached,
    so it fires again on the call that reaches it, and the old hard cap's stop is lifted."""
    ledger = ledger_for(session, 10.0, (0.5,))
    for _ in range(10):
        view(ledger, "zeus").charge_usage("p", USAGE, "m")
    assert _drained(ledger) == [("warning", 0.5), ("hard_cap", None)]
    assert not view(ledger, "zeus").may_start()
    restore(ledger, 20.0, Spend("session", 0, 0, 0, 5.5, 0), fractions=(0.5,))
    assert view(ledger, "zeus").may_start()
    seen: list[tuple[str, float | None]] = []
    for _ in range(5):
        view(ledger, "zeus").charge_usage("p", USAGE, "m")
        seen += _drained(ledger)
    assert seen == [("warning", 0.5)]


# -- the lock -------------------------------------------------------------------------------


def test_two_threads_charging_one_ledger_lose_nothing(session: Session) -> None:
    """The box charges from its own thread while Zeus's view charges from the loop's: the
    totals, and the record of every charge, are exact."""
    ledger = ledger_for(session, 1_000_000.0)
    each = 150
    start = threading.Barrier(2)

    def charge(rung: str) -> None:
        start.wait()
        for _ in range(each):
            view(ledger, rung).charge_usage("", USAGE, "m")

    threads = [threading.Thread(target=charge, args=(r,)) for r in ("zeus", "scout")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    spent = ledger.spend("session")
    assert spent.cost_usd == pytest.approx(2 * each * CALL)
    assert spent.tokens_in == 2 * each * 100
    written = lines(session)
    assert sum(ln.rung == "zeus" for ln in written) == each
    assert sum(ln.rung == "scout" for ln in written) == each
