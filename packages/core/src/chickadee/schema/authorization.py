"""Contract 10 — ActionAuthorization.

What plan approval actually authorizes, and what invalidates it. Authorization is never
carried forward across a material change: the concrete command, target, network, credential
set and effect surface are all part of what was approved, so changing any of them means the
approval was for a different action.

**The row records the tier the approval router computed.** It used to carry a
`RiskClass` of its own, which meant the same decision had two classification schemes with no
relation between them — so the record could contradict the decision in either direction and
nothing would notice. An authorization that does not attest the decision actually made is
evidence for a claim nobody checked.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

WIRE = ConfigDict(extra="forbid", frozen=True)


class Tier(StrEnum):
    """How much the user is consulted before a change happens.

    Ordered — `TIER_ORDER` is the order, and `max()` over it is what makes "never
    de-escalate" (invariant 17) structurally impossible to violate rather than a policy
    someone must remember to enforce.

    It lives here rather than in `chickadee/approval/` because a G0 contract carries it and
    `schema/` cannot import from a package built three phases later. That is also the right
    home on the merits: a tier is written down and read back — by the journal, the handoff,
    the visibility surface — and what crosses that boundary is a wire type.
    """

    T0 = "t0"
    """Do it, report after."""
    T1 = "t1"
    """Approve the design, once, before spend."""
    T2 = "t2"
    """Approve, then confirm each effect. Keyed on irreversibility (invariant 18), which is
    why no preference file may lower it."""


TIER_ORDER: tuple[Tier, ...] = (Tier.T0, Tier.T1, Tier.T2)
"""The escalation order. `StrEnum` compares as text, and "t0" < "t1" < "t2" is true only by
the accident of these spellings — so the order is stated rather than relied upon."""


class FactHit(BaseModel):
    """One fact row that fired, with the evidence that fired it.

    Here rather than in `chickadee/approval/` for `Tier`'s reason: a person's
    dispatch-time confirmation carries the rows that raised the tier (`ActionApproval.hits`,
    contract 6), and `schema/` cannot import from the package that computes them."""

    model_config = WIRE

    row: int
    """1-8; the fact table is the enumeration."""
    floor: Tier
    lowerable: bool
    evidence: str
    """The path, symbol, glob, command class or declared effect — never a score. Prose for
    the journal and the prompt; nothing is recovered from it by scanning (the two fields
    below are the typed subjects)."""
    path: str | None = None
    """The path this hit is about, when it is about one — the touched path for rows 2, 4
    and 7, the file the symbol is defined in for rows 5 and 8. A path preference
    (`always_ask` / `never_ask`) is matched against this by glob, never against `evidence`
    by substring: `never_ask = ["src/app"]` must not reach `src/app_config/db.py`, and the
    journal's `preference_applied` must name a rule that names the file."""
    command_class: str | None = None
    """The command table's class when the argv's row is this hit's evidence (row 1 on the
    shell path), so `[approval.commands]` reaches the hit by the name a person wrote."""


class RiskClass(StrEnum):
    """**Retired; nothing constructs one.**

    It classified actions by significance, which is the question replaced by
    *is this cheap to undo*. Two members show why it could not simply be re-keyed:
    `scope_expanding` named something completely reversible inside a leased worktree, and
    `destructive` collapsed the two verbs the router separates — editing a migration (T1,
    lowerable) and running one (T2, hard).

    Kept defined and exported rather than deleted: removing an exported symbol is itself a
    break, disuse costs nothing, and a reader who meets the name in an old journal line still
    finds what it meant. Same treatment as `PacketOrigin.rescue`.
    """

    ROUTINE = "routine"
    DESTRUCTIVE = "destructive"
    NETWORKED = "networked"
    CREDENTIAL_BEARING = "credential_bearing"
    SCOPE_EXPANDING = "scope_expanding"


class Confirmation(BaseModel):
    model_config = WIRE

    at: str
    source: str
    scope_sha: str
    """The hash of exactly what was confirmed. A confirmation that did not name its subject
    could be replayed against a different action."""


class ActionAuthorization(BaseModel):
    model_config = WIRE

    operation_id: str
    approved_hash: str
    effect_surface: tuple[str, ...] = ()
    network: tuple[str, ...] = ()
    credentials: tuple[str, ...] = ()
    tier: Tier
    """The tier `approval.compute()` returned at `RunPoint.DISPATCH`. Written straight
    through, with no translation step — a translation is exactly where the record and the
    decision would drift apart again."""
    granted_at: str
    confirmations: tuple[Confirmation, ...] = ()

    @model_validator(mode="after")
    def _confirmations_match_tier(self) -> ActionAuthorization:
        """One rule per tier, in the tier's own terms.

        The three checks below are kept verbatim — count, distinct sources, and
        each confirmation naming this action. What changed is only *when* each fires: the
        old row demanded nothing of `routine` and two of everything else, so T1's "approve the
        design before spend" had no way to be recorded and `scope_expanding` — reversible by
        discarding a branch — forced a prompt no preference could reach.
        """
        if self.tier is Tier.T0:
            return self

        required = 2 if self.tier is Tier.T2 else 1
        if len(self.confirmations) < required:
            what = (
                "a second confirmation at dispatch, not only the plan approval"
                if self.tier is Tier.T2
                else "the design approval it was granted under"
            )
            raise ValueError(f"tier={self.tier} requires {required} confirmation(s): {what}")

        # Counting was the whole check, so two copies of one plan approval satisfied it —
        # and `Confirmation.scope_sha` says in its own docstring what that allows. Distinct
        # sources, and each one naming this action.
        if self.tier is Tier.T2 and len({c.source for c in self.confirmations}) < 2:
            raise ValueError(
                f"tier={self.tier} requires confirmations from two distinct sources; "
                f"got {sorted(c.source for c in self.confirmations)} (D40)"
            )
        stray = sorted({c.scope_sha for c in self.confirmations} - {self.approved_hash})
        if stray:
            raise ValueError(
                f"confirmation scope_sha {stray} does not name this action "
                f"({self.approved_hash}); a confirmation for something else is not consent"
            )
        return self

    def is_valid_for(self, recomputed_hash: str) -> bool:
        """Whether this authorization still covers the action about to be dispatched.

        The hash check alone let a stale or unrelated confirmation ride along: the
        validator proved two existed, and nothing proved they were about *this* action.
        Re-checked here rather than trusted from construction time, because that is the
        moment the effect happens and the model is frozen but its meaning is not.
        """
        if recomputed_hash != self.approved_hash:
            return False
        if self.tier is Tier.T0:
            return True
        return all(c.scope_sha == recomputed_hash for c in self.confirmations)
