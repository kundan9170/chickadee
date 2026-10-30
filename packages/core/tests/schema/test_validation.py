"""One rejecting fixture per rule in the §2.2 / §5 tables.

Each case is a rule the docs state in prose; a test that only exercised the happy path
would let any of them quietly stop holding.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from pydantic import ValidationError

from chickadee.schema import (
    TIER_ORDER,
    ActionApproval,
    ActionAuthorization,
    Answer,
    AnswerSource,
    ApprovalDecision,
    Confirmation,
    FactHit,
    GitHandoffChoice,
    InteractionPort,
    Tier,
)


def _authorization(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "operation_id": "op-1",
        "approved_hash": "f" * 64,
        "tier": Tier.T2,
        "granted_at": "2026-08-18T09:00:00",
        "confirmations": (
            Confirmation(at="2026-08-18T09:00:00", source="plan", scope_sha="f" * 64),
            Confirmation(at="2026-08-18T09:00:05", source="dispatch", scope_sha="f" * 64),
        ),
    }
    return {**base, **over}


def test_non_routine_confirmations_must_name_this_action() -> None:
    """A confirmation that did not name its subject can be replayed against a
    different action. `Confirmation.scope_sha` documents that risk; nothing enforced it,
    so the count was the whole check."""
    with pytest.raises(ValidationError, match="does not name this action"):
        ActionAuthorization.model_validate(
            _authorization(
                confirmations=(
                    Confirmation(at="2026-08-18T09:00:00", source="plan", scope_sha="f" * 64),
                    Confirmation(
                        at="2026-08-18T09:00:05", source="dispatch", scope_sha="a" * 64
                    ),
                )
            )
        )


def test_two_copies_of_one_approval_are_not_two_confirmations() -> None:
    """The second confirmation exists to be *at dispatch*. Duplicating the plan
    approval satisfies a count and authorizes nothing."""
    dup = Confirmation(at="2026-08-18T09:00:00", source="plan", scope_sha="f" * 64)
    with pytest.raises(ValidationError, match="two distinct sources"):
        ActionAuthorization.model_validate(_authorization(confirmations=(dup, dup)))


def test_t0_actions_need_no_confirmation() -> None:
    auth = ActionAuthorization.model_validate(_authorization(tier=Tier.T0, confirmations=()))
    assert auth.is_valid_for("f" * 64) is True


def test_t1_records_the_design_approval_it_was_granted_under() -> None:
    """The old row demanded nothing of `routine` and two of everything else, so T1 —
    *approve the design, once, before spend* — had no shape it could be recorded in. An
    approval that happened and was not written down is not evidence that it happened."""
    with pytest.raises(ValidationError, match="tier=t1 requires 1 confirmation"):
        ActionAuthorization.model_validate(_authorization(tier=Tier.T1, confirmations=()))

    one = Confirmation(at="2026-09-12T09:00:00", source="plan", scope_sha="f" * 64)
    auth = ActionAuthorization.model_validate(
        _authorization(tier=Tier.T1, confirmations=(one,))
    )
    assert auth.is_valid_for("f" * 64) is True


def test_t1_does_not_demand_a_second_source() -> None:
    """Two distinct sources is T2's rule, because T2 is the tier that confirms each effect.
    Requiring it of T1 would be the over-asking invariant 18 exists to prevent."""
    two_plans = (
        Confirmation(at="2026-09-12T09:00:00", source="plan", scope_sha="f" * 64),
        Confirmation(at="2026-09-12T09:00:01", source="plan", scope_sha="f" * 64),
    )
    auth = ActionAuthorization.model_validate(
        _authorization(tier=Tier.T1, confirmations=two_plans)
    )
    assert auth.is_valid_for("f" * 64) is True


def test_tier_order_is_stated_not_inferred() -> None:
    """`StrEnum` compares as text, so the escalation order must not depend on the spellings
    happening to sort. `max()` over `TIER_ORDER` is what makes invariant 17's "never
    de-escalate" structurally impossible to violate."""
    assert TIER_ORDER == (Tier.T0, Tier.T1, Tier.T2)
    assert max((Tier.T1, Tier.T0, Tier.T2), key=TIER_ORDER.index) is Tier.T2


def test_is_valid_for_rechecks_the_binding_at_dispatch() -> None:
    """Construction-time validity is not dispatch-time validity: the effect happens here."""
    auth = ActionAuthorization.model_validate(_authorization())
    assert auth.is_valid_for("f" * 64) is True
    assert auth.is_valid_for("b" * 64) is False


def test_handoff_choices_are_the_four_and_bring() -> None:
    """`apply` and `squash` are separate git operations, not two spellings. D230
    Amendment 2 appended `bring`, the run's changes written into the checkout uncommitted,
    which no other member does.

    Pinned as a set because the failure this guards is a member quietly reappearing:
    after `contracts-v1` the enum is additive-only, so a stray `apply_squash` could never
    be removed again and would sit beside `apply` as a third way to say the same thing.
    """
    assert {c.value for c in GitHandoffChoice} == {
        "apply",
        "squash",
        "keep",
        "discard",
        "bring",
    }


def test_squash_is_not_reachable_by_asking_for_apply() -> None:
    """The two must stay distinct values. Aliasing them would put the choice back inside
    the engine, where D39's packet-level rollback becomes unreachable for the user."""
    assert GitHandoffChoice.APPLY is not GitHandoffChoice.SQUASH
    assert GitHandoffChoice("apply") is GitHandoffChoice.APPLY
    assert GitHandoffChoice("squash") is GitHandoffChoice.SQUASH


def test_no_handoff_choice_implies_a_push() -> None:
    """Nothing silently pushes, under any choice. A member named for a remote
    operation would be the first sign that changed."""
    assert not any("push" in c.value for c in GitHandoffChoice)


def _answer(**kw: Any) -> Answer:
    return Answer(question_id="q1", choice="skip", source=AnswerSource.HUMAN, **kw)


def test_answer_note_roundtrips_verbatim() -> None:
    """The note is the person's words, whitespace included."""
    text = "  that's the prod database — use the staging URL\n"
    a = _answer(note=text)
    assert Answer.model_validate_json(a.model_dump_json()).note == text


def test_answer_without_note_is_none() -> None:
    a = _answer()
    assert Answer.model_validate_json(a.model_dump_json()).note is None


def test_answer_empty_note_is_preserved_not_none() -> None:
    """A person who typed nothing on purpose is not a person who did not type."""
    a = _answer(note="")
    revived = Answer.model_validate_json(a.model_dump_json())
    assert revived.note == ""
    assert revived.note is not None
    assert revived != _answer()


# -- contract 6 at G4: the dispatch-time confirmation -------------


def _approval(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        action_id="a-1",
        operation_id="exec:a1:1:1",
        kind="command",
        display="make gen  (in .)",
        spec_sha="c" * 64,
        network=False,
        idempotent=None,
        reversible=False,
        tier=Tier.T2,
    )
    base.update(over)
    return base


def test_an_action_approval_validates_as_declared_and_carries_the_routers_rows() -> None:
    hit = FactHit(row=1, floor=Tier.T2, lowerable=False, evidence="destructive: rm -rf")
    approval = ActionApproval(**_approval(hits=(hit,), credentials=("GITHUB_TOKEN",)))
    assert approval.task_id is None and approval.effect_surface == ()
    assert approval.asker == "zeus", "Zeus's own unless a box role asks inside its task"
    assert approval.idempotent is None, "unknown is not False (invariant 20)"
    assert ActionApproval.model_validate_json(approval.model_dump_json()) == approval


@pytest.mark.parametrize(
    "bad",
    [{"kind": "shell"}, {"tier": "t9"}, {"surprise": 1}],
    ids=["kind", "tier", "extra"],
)
def test_an_action_approval_refuses_what_section_4_does_not_declare(
    bad: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        ActionApproval(**_approval(**bad))


def test_an_approval_decision_has_three_verdicts_and_four_sources() -> None:
    d = ApprovalDecision(action_id="a-1", verdict="abort", source="explicit_authorization")
    assert (d.remember, d.note) == (None, None)
    with pytest.raises(ValidationError):
        ApprovalDecision(action_id="a-1", verdict="maybe", source="human")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ApprovalDecision(action_id="a-1", verdict="deny", source="guess")  # type: ignore[arg-type]


def test_fact_hit_is_the_routers_own_and_a_wire_type() -> None:
    """Moved into `schema/`, not copied: the router's name is this one."""
    from chickadee.approval import FactHit as Routers

    assert Routers is FactHit
    with pytest.raises(ValidationError):
        FactHit(row=1, floor=Tier.T2, lowerable=False, evidence="x", score=0.9)  # type: ignore[call-arg]


def test_a_port_written_without_cancel_still_satisfies_the_protocol() -> None:
    """`cancel` is a keyword with a `None` default, added after the freeze, and the
    runtime check a host's port meets is by name, so one written without it still passes.
    That is all this proves: the engine calls `ask(q, cancel=...)`, so such a port fails at
    its first question, and one written before T-4.11 lacks `confirm_action` too. A host
    port is updated for both; the static check (pyright) says so."""

    class Older:
        def ask(self, q: object) -> object: ...
        def confirm_action(
            self, approval: object, *, cancel: threading.Event | None = None
        ) -> object: ...
        def confirm_stop(self, stop: object, *, cancel: object = None) -> object: ...

    assert isinstance(Older(), InteractionPort)
