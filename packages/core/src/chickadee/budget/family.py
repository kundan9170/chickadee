"""The session family's spend and caps, projected from the records.

A session family is every session sharing a fork root. Its spend is what each member
wrote: its own `spend` lines and its own episodes' journals. Its caps are the latest `cap` line
any member wrote, for the total and for each rung, else the config's. Computed
per call, stored nowhere, and best effort: a member whose log will not read contributes
nothing rather than refusing the session.

Moved out of the facade unchanged. `engine.py` calls `family_budget` for `budget()` and
the header, and `ledger_inputs` before every Zeus turn, and re-exports `SessionBudget` and
`RungBudget` where hosts import them.

**Imported by its full path, never re-exported by `budget/__init__.py`** (held by the engine
guard, `scripts/guards/engine.py`). This module imports `context.session`.
A re-export would make every user of the ledger (the orchestrator, Zeus, the scout) import the
session's machinery too, and it was an import cycle while `context.session` imported
`budget.spend`, which no longer happens. The rule stands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final, Literal

from chickadee.budget.manager import BudgetCaps
from chickadee.budget.spend import Spend
from chickadee.config import SpendPolicy
from chickadee.context.lines import CapLine, EpisodeLine, PayloadError, SpendLine
from chickadee.context.replay import ReplayCache
from chickadee.context.session import ForkLine, SessionState
from chickadee.context.store import (
    SessionStoreError,
    list_sessions,
    sessions_root,
)
from chickadee.journal.projection import RunSpend
from chickadee.journal.runs import RunProjections
from chickadee.roles import current_rung
from chickadee.state.roots import Roots

__all__ = ["RungBudget", "SessionBudget", "family_budget", "ledger_inputs", "session_spend"]


@dataclass(frozen=True)
class SessionBudget:
    """A session family's spend against its cap — what `/budget` shows. An
    engine value, computed per call and stored nowhere; not a schema contract
    (`BudgetStatus` stays retired)."""

    spent_usd: float
    """The family's spend: every member's own charges and episode journals."""
    cap_usd: float | None
    """The cap in force; `None` when the person lifted it for this family."""
    cap_source: Literal["config", "session"]
    cap_seq: int | None
    """The `cap` line's seq when `cap_source` is `"session"`."""
    rungs: Mapping[str, RungBudget] = field(default_factory=dict[str, "RungBudget"], hash=False)
    """Every rung with a cap, a lifted cap, or a charge, by name — the cast's rungs
    first, then any rung the records name that the cast has since retired, whose spend still
    counts. Out of the hash, as a dict cannot be hashed; equality still compares it, so
    equal budgets hash equal.

    The shares need not add up to `spent_usd`: a run journal's `budget` floor raises the
    total and names no rung."""
    roles: Mapping[str, float] = field(default_factory=dict[str, float], hash=False)
    """The family's spend by the role that made each charge. `zeus` is the
    session log's own charges, Zeus's conversation and the scout he calls (`spend` lines name a
    rung, not a role); `athena` and `worker` are their runs' `model_call` lines. Like `rungs`,
    the shares need not add up to `spent_usd`: a `budget` floor names no call."""
    warning_fractions: tuple[float, ...] = ()
    """The fractions of the cap the ledger warns at
    (`spend.warning_fractions`), strictly increasing inside (0, 1)."""


@dataclass(frozen=True)
class RungBudget:
    """One rung's share of the family's spend, and the rung's own cap beneath the
    total. An engine value, like `SessionBudget`."""

    spent_usd: float
    """What the family spent on this rung: its `spend` lines and its episodes' records."""
    cap_usd: float | None
    """The rung's cap in force; `None` when it has none, or the person lifted it."""
    cap_source: Literal["config", "session", "none"]
    cap_seq: int | None
    """The `cap` line's seq when `cap_source` is `"session"`."""


def _family_root(session_id: str, state: SessionState, folds: ReplayCache) -> str:
    """The session at the start of `state`'s fork chain: the owner of the first line it
    folds, since a fork reads its ancestry first. A session never forked is its own
    root, as is one with no lines yet.

    A fork taken at seq 0 reads none of its parent's lines, so its first line is its own
    `fork` line, and a later fork of it reads that line first too: the chain goes on through
    the parent the line names. Taken as its own root, it would be a family of one, and its
    spend would never reach its parent's cap."""
    seen = {session_id}
    while True:
        first = next(state.lines(), None)
        if first is None:
            return session_id
        event, record = first
        if not isinstance(record, ForkLine):
            return event.session_id
        # A first line that is a `fork` line — this session's, or an ancestor's read through
        # — is a fork taken at seq 0: its owner's family is its parent's.
        parent = record.parent_id
        if parent in seen:
            return session_id  # a cycle no store writes; stop rather than loop
        seen.add(parent)
        try:
            _log, state = folds.get(parent)
        except (SessionStoreError, PayloadError, OSError, ValueError):
            return parent  # a family read is best effort
        session_id = parent


def _family_members(
    session_id: str, state: SessionState, where: Roots, folds: ReplayCache
) -> list[tuple[str, SessionState]]:
    """Every session sharing `session_id`'s root, each with its fold: this one live,
    the others read without their locks. A member whose log will not read is left out, since
    a family read is best effort and never a refusal."""
    root = _family_root(session_id, state, folds)
    members: list[tuple[str, SessionState]] = [(session_id, state)]
    for member in list_sessions(sessions_root(where)):
        if member == session_id:
            continue
        try:
            _log, other = folds.get(member)
        except (SessionStoreError, PayloadError, OSError, ValueError):
            continue
        if _family_root(member, other, folds) == root:
            members.append((member, other))
    return members


def _family_caps(
    members: Sequence[tuple[str, SessionState]],
) -> dict[str | None, tuple[str, CapLine]]:
    """The family's `cap` line in force for the total (key `None`) and for each
    rung, with the member that wrote it, from one pass over the family.

    The latest line any member **wrote**. Within one log that is its last line by seq,
    whatever the clock said: a clock stepped back between two `set_cap` calls must not let
    the earlier one win. Across members, the time each was written orders them, the
    session's id breaking a tie in the same second. A line for one rung never overrides
    another's, nor the total's. A key is absent when no member has set it, and the config's
    cap holds."""
    latest: dict[str | None, tuple[tuple[str, str], CapLine]] = {}
    for member, folded in members:
        own: dict[str | None, tuple[str, CapLine]] = {}
        for event, record in folded.lines():
            if event.session_id == member and isinstance(record, CapLine):
                # A line on a former rung caps its successor.
                rung = current_rung(record.rung) if record.rung is not None else None
                own[rung] = (event.at, record)
        for rung, (at, record) in own.items():
            key = (at, member)
            held = latest.get(rung)
            if held is None or key > held[0]:
                latest[rung] = (key, record)
    return {rung: (key[1], record) for rung, (key, record) in latest.items()}


def _rung_caps(
    configured: Mapping[str, float],
    lines: Mapping[str | None, tuple[str, CapLine]],
    declared: Sequence[str],
) -> dict[str, tuple[float | None, tuple[str, CapLine] | None]]:
    """Each declared rung's cap — the family's latest line for it, else the engine
    config's — with the line it came from, for every rung that has either. A line on a rung
    the config no longer declares caps nothing that runs."""
    out: dict[str, tuple[float | None, tuple[str, CapLine] | None]] = {}
    for rung in declared:
        line = lines.get(rung)
        if line is not None:
            out[rung] = (line[1].hard_cost_usd, line)
        elif rung in configured:
            out[rung] = (configured[rung], None)
    return out


def _in_force(caps: Mapping[str, tuple[float | None, object]]) -> dict[str, float]:
    """The rung caps that stop anything: a lifted one (`None`) does not."""
    return {rung: cap for rung, (cap, _line) in caps.items() if cap is not None}


def _caps_under(base: BudgetCaps, line: tuple[str, CapLine] | None) -> BudgetCaps:
    """The ledger's caps for a family: the config's, with its hard cap replaced by the
    family's line when there is one (`None` lifts it)."""
    return base if line is None else replace(base, hard_cost_cap_usd=line[1].hard_cost_usd)


ZEUS_ROLE: Final = "zeus"
"""The role the session log's own charges count under (`SessionBudget.roles`)."""


@dataclass(frozen=True)
class _Spending:
    """`_spending`'s answer: the family's `Spend`, and its cost by rung and by role."""

    spend: Spend
    by_rung: dict[str, float]
    by_role: dict[str, float]


def _spending(members: Sequence[tuple[str, SessionState]], runs: RunProjections) -> _Spending:
    """The projection over the family: what every member has spent, from the records
    alone, computed per call and stored nowhere, with the cost **by rung** beside it.

    Each member counts only what it **wrote**: its own `spend` lines, and its own episodes'
    journals through their projection (`RunSpend`) — the `coder_done` costs,
    never below a `budget` crossing's amount. A line a fork reads through from its parent is
    the parent's, and counting it again for every child would bill the parent once per
    descendant. A `spend` line names its rung, and an episode's cost is by its `coder_done`'s
    `rung`, else `coder`; a `budget` floor names no call, so it raises the total and no rung.
    A former rung's name counts on its successor (`roles.current_rung`).

    Best effort, never a refusal: a member whose log will not read, or a
    journal damaged part-way, contributes what it proves.

    **What the `Spend` covers.** `cost_usd` is the whole family's: the cap is checked
    against it. The token counts add the journals' `coder_done` counts to Zeus's. The cache
    and estimate counts are Zeus's alone, since `coder_done` does not carry them, and a
    `budget` floor raises the cost with no tokens behind it."""
    tokens_in = tokens_out = cached_in = cache_writes = estimated = 0
    cost = floor = 0.0
    by_rung: dict[str, float] = {}
    by_role: dict[str, float] = {}
    for member, folded in members:
        for event, record in folded.lines():
            if event.session_id != member:
                continue  # read through from an ancestor: that member counts it
            if isinstance(record, SpendLine):
                tokens_in += record.tokens_in
                tokens_out += record.tokens_out
                cached_in += record.cached_in
                cache_writes += record.cache_writes
                estimated += int(record.estimated)
                cost += record.cost_usd
                rung = current_rung(record.rung)
                by_rung[rung] = by_rung.get(rung, 0.0) + record.cost_usd
                by_role[ZEUS_ROLE] = by_role.get(ZEUS_ROLE, 0.0) + record.cost_usd
            elif isinstance(record, EpisodeLine):
                run = _run_spend(runs, record.run_id)
                tokens_in += run.tokens_in
                tokens_out += run.tokens_out
                cost += run.cost_usd
                floor = max(floor, run.floor_usd)
                for named, amount in run.by_rung.items():
                    rung = current_rung(named)
                    by_rung[rung] = by_rung.get(rung, 0.0) + amount
                for role, amount in run.by_role.items():
                    by_role[role] = by_role.get(role, 0.0) + amount
    spend = Spend(
        "session",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cached_in=cached_in,
        cost_usd=max(cost, floor),
        wall_ms=0,
        cache_writes=cache_writes,
        estimated_calls=estimated,
    )
    return _Spending(spend, by_rung, by_role)


def session_spend(session_id: str, state: SessionState, runs: RunProjections) -> Spend:
    """One session's own spend, as `_spending` counts each member's: what the handoff reports
    for the session (08 session scope), where `budget()` reports the
    family's."""
    return _spending([(session_id, state)], runs).spend


def _run_spend(runs: RunProjections, run_id: str) -> RunSpend:
    """One run's spend, as its journal's projection reads it: tokens, the
    box's cost, cost by rung (`model_call.rung`, else its role) and the `budget` floor. A
    journal written by an older version reads as nothing spent. Read up to any damage
    (`readable`), so a torn or corrupt journal still counts what it held; a run with no
    journal yet spent nothing. The engine's `RunProjections` folds each journal once and
    reads only what it gained since."""
    try:
        return runs.readable(run_id).spend
    except FileNotFoundError:
        return RunSpend()


def family_budget(
    session_id: str,
    state: SessionState,
    where: Roots,
    folds: ReplayCache,
    runs: RunProjections,
    policy: SpendPolicy,
    declared: Sequence[str],
) -> tuple[SessionBudget, dict[str | None, str]]:
    """`budget()`'s value, and the member whose log holds each `cap` line in force, keyed by
    rung (`None` for the total): a line set from a fork caps its parent too, and the header
    names where it came from. `state` is `session_id`'s fold, read by the caller. `folds` and
    `runs` are the engine's one cache of each; `declared` the config's
    declared rungs (`EngineConfig.rung_names`)."""
    members = _family_members(session_id, state, where, folds)
    spending = _spending(members, runs)
    by_rung = spending.by_rung
    lines: dict[str | None, str] = {}
    rungs: dict[str, RungBudget] = {}
    cap_lines = _family_caps(members)
    caps = _rung_caps(policy.rungs, cap_lines, declared)
    # The declared rungs, then any a record names that the config no longer declares (or a
    # former rung with no successor): its spend is in the total, so the breakdown names it.
    undeclared = sorted(r for r in by_rung if r not in declared)
    for rung in (*declared, *undeclared):
        if rung not in caps and rung not in by_rung:
            continue
        spent_on = by_rung.get(rung, 0.0)
        cap, from_line = caps.get(rung, (None, None))
        if from_line is not None:
            lines[rung] = from_line[0]
            rungs[rung] = RungBudget(spent_on, cap, "session", from_line[1].seq)
        elif cap is not None:
            rungs[rung] = RungBudget(spent_on, cap, "config", None)
        else:
            rungs[rung] = RungBudget(spent_on, None, "none", None)
    spent = spending.spend.cost_usd
    roles = dict(spending.by_role)
    warns = tuple(float(f) for f in policy.warning_fractions)
    line = cap_lines.get(None)
    if line is None:
        budget = SessionBudget(spent, policy.hard_cost_usd, "config", None, rungs, roles, warns)
        return budget, lines
    member, cap_line = line
    lines[None] = member
    budget = SessionBudget(
        spent, cap_line.hard_cost_usd, "session", cap_line.seq, rungs, roles, warns
    )
    return budget, lines


def ledger_inputs(
    session_id: str,
    state: SessionState,
    where: Roots,
    folds: ReplayCache,
    runs: RunProjections,
    base: BudgetCaps,
    configured: Mapping[str, float],
    declared: Sequence[str],
) -> tuple[BudgetCaps, Spend, dict[str, float], dict[str, float], tuple[str, ...]]:
    """What the session's ledger restores before every turn: the family's caps (the total's
    line over `base`), its spend, each rung's cap in force, each rung's spend, and the rungs
    whose cap the person lifted. `configured` is the config's per-rung caps,
    `declared` its declared rungs."""
    members = _family_members(session_id, state, where, folds)
    lines = _family_caps(members)
    caps = _caps_under(base, lines.get(None))
    spending = _spending(members, runs)
    by_cap = _rung_caps(configured, lines, declared)
    lifted = tuple(rung for rung, (cap, _line) in by_cap.items() if cap is None)
    return caps, spending.spend, _in_force(by_cap), spending.by_rung, lifted
