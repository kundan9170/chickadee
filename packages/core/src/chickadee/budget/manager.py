"""`CostLedger` — ledgers per scope, the crossings queue, packet reset.

One place answers two questions: what has been spent, and may the next priced operation
start? **Charging happens after every model call and separately priced effect.** There
is no `authorize()` — a pre-flight estimate was a predictor standing in front of a
measurement — so the cap means *stop at the first crossing*, with overshoot bounded by one
operation. Actual usage is charged including failed calls.

The cap belongs to the **session** lineage: a delegated episode does not reset the
limit. Crossings are recorded **once** each into a queue the driver drains after each
charge; a warning re-derived on every charge would fire on every charge after the first.
Crossing the hard cap yields a resumable `BUDGET_CAP` terminal, never a silent stop.

**One ledger, rungs built in.** Each rung may carry its own cap beneath the total,
and the ledger holds each rung's cap and spend beside its own. It has **no default
rung**: a loop holds one rung's view (`for_rung`, the `Ledger` interface), and every charge and
every "may I start" names the rung. Nothing subclasses the ledger (`scripts/guards/budget.py`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from chickadee.budget.prices import PriceTable
from chickadee.budget.spend import SCOPES, Scope, Spend, SpendAccumulator
from chickadee.config import SpendPolicy, is_cap
from chickadee.errors import ConfigError, HarnessBug
from chickadee.roles import ROLES
from chickadee.schema.adapter import Usage
from chickadee.schema.journal import BudgetP, RunCapsP

__all__ = [
    "BUDGET_CAP",
    "BudgetCaps",
    "BudgetCrossing",
    "Clock",
    "CostLedger",
    "Ledger",
    "MonotonicClock",
    "RungLedger",
    "SpendRecorder",
]

BUDGET_CAP = "BUDGET_CAP"
"""The terminal a hard-cap crossing yields. A terminal is a result, never an
exception, and it is never disguised as a generic `ABORTED`."""


class Clock(Protocol):
    """Monotonic, and injected so tests can drive it."""

    def monotonic(self) -> float: ...


class MonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True)
class BudgetCaps:
    """EngineConfig — no dollar limit is hardcoded anywhere.

    `hard_cost_cap_usd` is `None` only for an interactive run that chose no cap; the cap makes
    it required in `SpendPolicy`, so `from_config` never produces `None`.
    """

    hard_cost_cap_usd: float | None
    warning_fractions: tuple[float, ...] = (0.8,)
    packet_warning_cost_usd: float | None = None

    def __post_init__(self) -> None:
        cap = self.hard_cost_cap_usd
        # `> 0` alone passes `inf`, and an infinite cap is no cap.
        if cap is not None and not is_cap(cap):
            raise ConfigError(f"hard_cost_cap_usd must be positive and finite, got {cap}")
        prev = 0.0
        for f in self.warning_fractions:
            if not 0.0 < f < 1.0 or f <= prev:
                raise ConfigError(
                    f"warning fractions must strictly increase inside (0, 1), got "
                    f"{self.warning_fractions}"
                )
            prev = f
        if self.packet_warning_cost_usd is not None and not self.packet_warning_cost_usd > 0:
            raise ConfigError("packet_warning_cost_usd must be positive when set")

    @staticmethod
    def from_config(spend: SpendPolicy) -> BudgetCaps:
        return BudgetCaps(
            hard_cost_cap_usd=spend.hard_cost_usd,
            warning_fractions=spend.warning_fractions,
            packet_warning_cost_usd=spend.packet_warning_cost_usd,
        )


@dataclass(frozen=True)
class BudgetCrossing:
    """The `budget{...}` payload as a value, drained by the driver after each charge."""

    scope: Scope
    kind: Literal["warning", "hard_cap"]
    spent_usd: float
    cap_usd: float
    fraction: float | None
    """The `warning_fractions` entry that fired; `None` for `hard_cap`."""
    terminal: Literal["BUDGET_CAP"] | None
    """Set only by `hard_cap`; a warning never ends a run."""

    def payload(self) -> BudgetP:
        """The frozen `budget` event. `BudgetP.scope` predates sessions and knows `run` and
        `packet` only, so the session's crossing is journaled under the run it happened
        in — the amount is the lineage's, and the run is where it was noticed."""
        return BudgetP(
            scope="packet" if self.scope == "packet" else "run",
            kind=self.kind,
            spent_usd=self.spent_usd,
            cap_usd=self.cap_usd,
            terminal=self.terminal is not None,
        )


ROLE_NAMES: Final = tuple(r.name for r in ROLES)
"""The cast's roles: the only names a spend line may give for whose call it was."""


class SpendRecorder(Protocol):
    """Writes a charge no run journals, Zeus's or the scout's, to the session log: the
    session's `record_spend`, which the engine passes in. A box role's charge is its run's
    `model_call`, which the projection reads, so it is never recorded here: counted once.
    Injected, so the budget package imports nothing from `context`."""

    def __call__(
        self, model: str, usage: Usage, cost_usd: float, rung: str, role: str | None = None
    ) -> object: ...


class Ledger(Protocol):
    """One rung's view of the session's ledger: the one interface every loop that spends
    holds. Zeus holds the view of the rung the `zeus` role is assigned, the scout the
    `locate` role's, and the box's driver cuts one per call, by the rung of the role making it
    (`EngineConfig.assignment`)."""

    @property
    def rung(self) -> str: ...

    def may_start(self) -> bool:
        """The gate: the total and this rung both have room. Nothing is estimated."""
        ...

    def stopped_by(self) -> str | None:
        """The cap that refuses this rung's next call, named, or None when it may start."""
        ...

    def charge_usage(
        self, scope_packet: str, usage: Usage, model: str, *, journaled: bool = False
    ) -> float:
        """Charge a call this rung made, after it happened, failed calls included.
        `journaled`: the run's journal records the call (`model_call`), so the session log
        must not (`CostLedger.charge`)."""
        ...


class CostLedger:
    """Ledgers per scope, one crossings queue, packet reset, and each rung's cap and spend.

    **One lock covers the ledger's state** (`_lock`): every public method that reads or writes
    it takes it, so a charge, its rung's spend and its record are one step, and so is a
    restore. The box charges from its own thread while Zeus's view charges from the loop's."""

    def __init__(
        self,
        prices: PriceTable,
        caps: BudgetCaps,
        clock: Clock | None = None,
        *,
        rungs: Collection[str],
        record: SpendRecorder | None = None,
    ) -> None:
        """`rungs` is the config's declared rungs (`EngineConfig.rung_names`), passed once: a
        view is handed out for those alone."""
        self.prices = prices
        self._rungs = frozenset(rungs)
        self.clock: Clock = clock or MonotonicClock()
        self._lock = threading.RLock()
        self._caps = caps
        self._record = record
        self._scopes: dict[str, SpendAccumulator] = {
            "session": SpendAccumulator("session"),
            "run": SpendAccumulator("run"),
            "packet": SpendAccumulator("packet"),
        }
        self._queue: list[BudgetCrossing] = []
        self._fired_fractions: set[float] = set()
        self._hard_cap_fired = False
        self._packet_warned = False
        self._packet_id: str | None = None
        self._run_id: str | None = None
        self._last_mark: float | None = None
        self._rung_caps: dict[str, float] = {}
        self._rung_spent: dict[str, float] = {}
        self._lifted_rungs: tuple[str, ...] = ()
        self.configured_total = caps.hard_cost_cap_usd
        """The config's total cap, which `run_start` records whatever the family's
        line has since made of it; `run_caps` says what is in force."""

    @property
    def caps(self) -> BudgetCaps:
        """The total's cap and warning fractions in force; `restore` replaces them."""
        with self._lock:
            return self._caps

    # -- rungs --------------------------------------------------------------

    def for_rung(self, rung: str, *, role: str | None = None) -> RungLedger:
        """`rung`'s view: what a loop on that rung holds. The rung is named where the view is
        handed out, never inferred from a model's name, which two rungs may share; so is the
        role whose calls it charges (`zeus`, `locate`, `residue`), recorded on each spend
        line, since two roles may share a rung. None names no role: a view no spend
        line is read for (the tests', a box role's whose calls its run journals)."""
        if rung not in self._rungs:
            raise HarnessBug(
                f"no rung {rung!r} to hand a ledger to; the declared rungs are "
                f"{sorted(self._rungs)}"
            )
        if role is not None and role not in ROLE_NAMES:
            # Its spend lines are written for good: a misspelt role would never be read.
            raise HarnessBug(
                f"no role {role!r} to name on a spend line; the cast: {ROLE_NAMES}"
            )
        return RungLedger(self, rung, role)

    def stopped_by(self, rung: str) -> str | None:
        """The cap that refuses a call on `rung`, named, or `None` when the call may start.
        Reaching a cap exactly is crossing it, as for the total. The total first: reaching it
        stops every rung, and saying the rung's would hide that. One rule decides both whether
        a call starts and what the refusal says, so they cannot disagree."""
        with self._lock:
            spent = self._scopes["session"].cost_usd
            if self._hard_cap_fired:
                total = self._caps.hard_cost_cap_usd
                at = "" if total is None else f" of ${total:.2f}"
                return f"the session family's spend cap{at} is reached (${spent:.2f} spent)"
            cap = self._rung_caps.get(rung)
            on = self._rung_spent.get(rung, 0.0)
            if cap is not None and on >= cap:
                return (
                    f"the {rung} rung's spend cap of ${cap:.2f} is reached (${on:.2f} spent "
                    "on it); the family's total still has room"
                )
            return None

    def charge(
        self,
        rung: str,
        scope_packet: str,
        usage: Usage,
        model: str,
        *,
        journaled: bool = False,
        role: str | None = None,
    ) -> float:
        """Charge a model call's actual usage to the packet, the run and the session, and to
        `rung`'s spend; record it on the session unless the run's journal records it.
        Returns the cost for the `model_call` payload. Failed calls are charged too.

        `scope_packet` names the packet the call belonged to; a call outside any packet (the
        planner, Zeus, the scout) passes `""` and charges the run and session only. `model` is
        the catalog name that served the call — after a `/rung` switch, the new one,
        never the rung, which may have run on two models in one session. A model with no price
        row raises `ConfigError` before anything is charged. A record that cannot be
        written raises after the charge: the spend is counted either way.

        **The rule a second charger keeps.** A `journaled` charge, every call a box
        role makes, is not recorded here: it is its run's `model_call`, which the projection
        reads, and recording it too would count it twice. Athena's calls outside any task are
        such charges too: keyed by the empty scope, they were once recorded on the
        session as well, and a run's spend read twice in the family's total."""
        with self._lock:
            cost = self._charge_scopes(scope_packet, usage, model)
            self._rung_spent[rung] = self._rung_spent.get(rung, 0.0) + cost
            if not journaled and self._record is not None:
                self._record(model, usage, cost, rung, role)
            return cost

    def _charge_scopes(self, scope_packet: str, usage: Usage, model: str) -> float:
        breakdown = self.prices.cost(usage, model)
        cached = usage.cached_prompt_tokens or 0
        writes = usage.cache_write_tokens or 0
        targets = ["session", "run"] + (["packet"] if scope_packet else [])
        if scope_packet and scope_packet != self._packet_id:
            self.start_packet(scope_packet)
        for name in targets:
            self._scopes[name].add_usage(
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                cached_in=cached,
                cache_writes=writes,
                cost_usd=breakdown.total,
                estimated=usage.estimated,
            )
        self._check_crossings()
        return breakdown.total

    # -- scopes ------------------------------------------------------------------------

    def start_packet(self, packet_id: str) -> None:
        """Resets the packet scope. The packet warning is per packet, so it re-arms."""
        with self._lock:
            self._scopes["packet"].reset()
            self._packet_id = packet_id
            self._packet_warned = False

    def start_run(self, run_id: str) -> None:
        """Resets the run scope — one delegated coder episode. The session scope and
        the cap are untouched: a new episode does not reset the limit."""
        with self._lock:
            self._scopes["run"].reset()
            self._scopes["packet"].reset()
            self._run_id = run_id
            self._packet_id = None
            self._packet_warned = False

    def spend(self, scope: str) -> Spend:
        """Read-only, for the stream and the handoff."""
        if scope not in self._scopes:
            raise HarnessBug(f"unknown spend scope {scope!r}; scopes are {SCOPES}")
        with self._lock:
            return self._scopes[scope].snapshot()

    # -- wall time ---------------------------------------------------------------------

    def charge_wall(self, ms: int) -> None:
        """After every `exec`. Negative (a clock anomaly) clamps at 0."""
        with self._lock:
            for acc in self._scopes.values():
                acc.add_wall(ms)

    def mark(self) -> float:
        """Take the clock now, for `charge_wall_since`."""
        with self._lock:
            self._last_mark = self.clock.monotonic()
            return self._last_mark

    def charge_wall_since(self, mark: float | None = None) -> int:
        """Charge the wall time since `mark` (or the last `mark()`), clamped at 0 if the
        clock went backwards. Returns the milliseconds charged."""
        with self._lock:
            start = self._last_mark if mark is None else mark
            if start is None:
                raise HarnessBug("charge_wall_since needs a mark; call mark() first")
            ms = max(0, int((self.clock.monotonic() - start) * 1000))
            self.charge_wall(ms)
            return ms

    # -- crossings -----------------------------------------------------------------------

    def crossings(self) -> tuple[BudgetCrossing, ...]:
        """Drain the queue. Each crossing is returned exactly once."""
        with self._lock:
            out = tuple(self._queue)
            self._queue.clear()
            return out

    @property
    def exhausted(self) -> bool:
        """The hard cap has been crossed: no further priced operation starts."""
        with self._lock:
            return self._hard_cap_fired

    def _check_crossings(self) -> None:
        cap = self._caps.hard_cost_cap_usd
        spent = self._scopes["session"].cost_usd
        if cap is not None:
            for fraction in self._caps.warning_fractions:
                if fraction in self._fired_fractions or spent < fraction * cap:
                    continue
                self._fired_fractions.add(fraction)
                self._queue.append(
                    BudgetCrossing("session", "warning", spent, cap, fraction, None)
                )
            # The boundary: reaching the cap exactly is crossing it. A cap of $5.00 means
            # $5.00 may be spent and not a cent more starts.
            if not self._hard_cap_fired and spent >= cap:
                self._hard_cap_fired = True
                self._queue.append(
                    BudgetCrossing("session", "hard_cap", spent, cap, None, BUDGET_CAP)
                )
        packet_cap = self._caps.packet_warning_cost_usd
        if packet_cap is not None and self._packet_id is not None and not self._packet_warned:
            packet_spent = self._scopes["packet"].cost_usd
            if packet_spent >= packet_cap:
                self._packet_warned = True
                self._queue.append(
                    BudgetCrossing("packet", "warning", packet_spent, packet_cap, None, None)
                )

    # -- the resumable terminal, and the records -----------------------------------------

    def terminal(self) -> str | None:
        """`BUDGET_CAP` once the cap is crossed, else `None`. A result, never an exception:
        the driver writes the resumable handoff and ends the run with it."""
        with self._lock:
            return BUDGET_CAP if self._hard_cap_fired else None

    def resume_from(self, session: Spend, run: Spend | None = None) -> None:
        """Rebuild the ledger from journaled spend on resume (the cap is resumable — a
        run that crossed it and was topped up by config continues from what was spent,
        not from zero). Crossings already journaled are not re-emitted: the fractions at or
        below the resumed spend are marked fired without queueing."""
        with self._lock:
            self._scopes["session"] = _restore(session, "session")
            if run is not None:
                self._scopes["run"] = _restore(run, "run")
            cap = self._caps.hard_cost_cap_usd
            if cap is not None:
                for fraction in self._caps.warning_fractions:
                    if session.cost_usd >= fraction * cap:
                        self._fired_fractions.add(fraction)
                self._hard_cap_fired = session.cost_usd >= cap

    def restore(
        self,
        caps: BudgetCaps,
        spent: Spend,
        rung_caps: Mapping[str, float] | None = None,
        rung_spent: Mapping[str, float] | None = None,
        lifted_rungs: tuple[str, ...] = (),
    ) -> None:
        """The family's cap and spend, as the records say before a turn. A cap
        that moved starts its crossings over: a raise may un-stop the session, and its
        warnings belong to the new cap, not the old one's fractions. A lifted cap (`None`)
        stops nothing and warns of nothing; `resume_from` still restores the spend.

        Each rung's cap in force and its spend, likewise, replace what was held. A rung
        absent from `rung_caps` has no cap of its own."""
        with self._lock:
            if caps != self._caps:
                self._caps = caps
                self._fired_fractions.clear()
                self._hard_cap_fired = False
            self.resume_from(spent)
            self._rung_caps = dict(rung_caps or {})
            self._rung_spent = dict(rung_spent or {})
            self._lifted_rungs = lifted_rungs

    def run_caps(self) -> tuple[float, RunCapsP]:
        """What a run's `run_start` records: the config's total, and
        every cap in force, a lifted total or rung said as lifted. A run starts under
        a lifted total; the record says so."""
        with self._lock:
            configured = self.configured_total
            if configured is None:
                raise ConfigError("a run starts with a finite hard cost cap configured (D42)")
            total = self._caps.hard_cost_cap_usd
            return configured, RunCapsP(
                total=total,
                total_lifted=total is None,
                rungs=dict(self._rung_caps),
                lifted_rungs=self._lifted_rungs,
            )


@dataclass(frozen=True)
class RungLedger:
    """One rung's view of a `CostLedger`: what `for_rung` hands a loop. Its charges are
    the ledger's, named for the rung; its `may_start` asks both the total and the rung's own
    cap."""

    ledger: CostLedger
    rung: str
    role: str | None = None
    """The role whose calls charge through this view, named on its spend lines (`for_rung`)."""

    def may_start(self) -> bool:
        return self.ledger.stopped_by(self.rung) is None

    def stopped_by(self) -> str | None:
        return self.ledger.stopped_by(self.rung)

    def charge_usage(
        self, scope_packet: str, usage: Usage, model: str, *, journaled: bool = False
    ) -> float:
        return self.ledger.charge(
            self.rung, scope_packet, usage, model, journaled=journaled, role=self.role
        )


def _restore(spend: Spend, scope: Scope) -> SpendAccumulator:
    acc = SpendAccumulator(scope)
    acc.tokens_in = spend.tokens_in
    acc.tokens_out = spend.tokens_out
    acc.cached_in = spend.cached_in
    acc.cache_writes = spend.cache_writes
    acc.cost_usd = spend.cost_usd
    acc.wall_ms = spend.wall_ms
    acc.estimated_calls = spend.estimated_calls
    return acc
