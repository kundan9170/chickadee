"""How a test reads a session's spend and caps, now that the session fold keeps no copies of
its own.

At the tests' root, beside `journal_pins.py`, so every suite imports one copy.

- **The lines** are the fold's own: `state.lines()`, by type. `SessionState.charges()` and
  `caps()` were these lines, kept twice.
- **The spend** is the family's one reader, `budget.family.ledger_inputs`, what the engine
  restores its ledger from before every turn. `SessionState.spent()` and
  `family.family_spend`, which only tests called, were copies of it.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.budget.family import ledger_inputs
from chickadee.budget.manager import BudgetCaps
from chickadee.budget.spend import Spend
from chickadee.context.replay import ReplayCache
from chickadee.context.session import SessionState
from chickadee.context.store import SESSIONS_DIR
from chickadee.journal.runs import RunProjections
from chickadee.roles import DEFAULT_RUNGS
from chickadee.state.roots import Roots


def lines_of[T](state: SessionState, model: type[T]) -> tuple[T, ...]:
    """Every line of `model` the session reads, its lineage's up to each fork point included,
    in order."""
    return tuple(r for _event, r in state.lines() if isinstance(r, model))


def roots_for(sessions: Path) -> Roots:
    """Roots whose sessions directory is `sessions`, so the family is read where the test's
    sessions are."""
    if sessions.name != SESSIONS_DIR:
        raise ValueError(f"{sessions} is not a {SESSIONS_DIR!r} directory")
    base = sessions.parent
    return Roots(
        project=base, repo_local=base, checkout_local=base / "checkout", user=base / "user"
    )


def family_spent(sessions: Path, session_id: str, state: SessionState) -> Spend:
    """What `session_id`'s family has spent: every member's own `spend` lines and its own
    episodes' journals, as the engine's ledger is restored from them."""
    where = roots_for(sessions)
    caps = BudgetCaps(hard_cost_cap_usd=None)
    return ledger_inputs(
        session_id,
        state,
        where,
        ReplayCache(sessions),
        RunProjections(where),
        caps,
        {},
        DEFAULT_RUNGS,
    )[1]
