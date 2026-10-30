"""Preregistration: what is fixed before the first result is seen.

The whole point is the ordering. Every field here is a decision that becomes unfalsifiable
once results are visible — how many repeats, when to stop, which confidence-interval
procedure, what to do about a crashed pair. Chosen afterwards, each is a knob that can be
turned until the answer is the one you wanted, and no reader can tell that happened.

So this is a schema with a timestamp, and `freeze()` is a one-way door. G6's claim is
"≤5 percentage-point solve loss and ≤50% cost under a *preregistered* paired comparison";
without the preregistration the same numbers are a description of one run.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

WIRE = ConfigDict(extra="forbid", frozen=True)


class CIMethod(StrEnum):
    BOOTSTRAP_PAIRED = "bootstrap_paired"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MCNEMAR_EXACT = "mcnemar_exact"


class MissingPolicy(StrEnum):
    """What a crashed, timed-out or absent result counts as.

    Never "drop it": dropping a pair silently moves the denominator, and the direction it
    moves is the direction that flatters whichever arm failed.
    """

    UNSOLVED = "unsolved"
    """Counts against the arm. The default for a crash inside the arm's own work."""
    DROP_PAIR = "drop_pair"
    """Both arms' results for that task are discarded together, keeping the pairing
    intact. Legal only for infrastructure faults outside either arm."""


class CommonPolicy(BaseModel):
    """Everything both arms must share for a comparison to mean anything.

    A difference in any of these is a difference in the experiment, not in the harnesses.
    Named explicitly rather than assumed equal, because "we ran them the same way" is the
    claim most easily believed and least often true.
    """

    model_config = WIRE

    task_statement_sha: str
    repo_snapshot_sha: str
    visible_tests_sha: str
    acceptance_oracle_sha: str
    tools_available: tuple[str, ...]
    tool_permissions: tuple[str, ...]
    test_commands: tuple[str, ...]
    per_command_timeout_s: int = Field(gt=0)
    per_task_timeout_s: int = Field(gt=0)
    human_intervention: str
    """The policy, not a boolean: "none" and "operator may retry once" are both answers,
    and which one was in force decides whether the result is autonomous."""
    handoff_policy: str


class Preregistration(BaseModel):
    """Frozen before the first result."""

    model_config = WIRE

    experiment: str
    frontier_model: str
    frontier_revision: str
    slm_model: str
    slm_revision: str
    arm_versions: dict[str, str]

    task_order_seed: int
    paired_seeds: dict[str, int]
    """Task id → seed, shared by both arms. Pairing is the entire basis of the inference:
    unpaired, a difference between arms is confounded with a difference between tasks."""
    repeats: int = Field(ge=1)
    stopping_rule: str
    attempt_policy: str

    accuracy_margin_pp: float = Field(gt=0)
    cost_ratio_max: float = Field(gt=0)
    ci_method: CIMethod
    ci_level: float = Field(gt=0.5, lt=1.0)
    cost_formula: str
    slm_shadow_price: str
    cache_treatment: str
    missing_policy: MissingPolicy
    hard_cost_usd_per_arm: dict[str, float]

    common: CommonPolicy
    named_asymmetries: tuple[str, ...] = ()
    """Any unavoidable difference between arms, stated up front. A run with a *material*
    asymmetry is diagnostic rather than headline-eligible — declaring it does not make it
    disappear, it makes it visible."""

    frozen_at: str | None = None
    """Set by `freeze()`. Absent means results must not be recorded yet."""

    @model_validator(mode="after")
    def _check(self) -> Preregistration:
        if not self.arm_versions:
            raise ValueError("preregistration names no arms")
        missing_cap = sorted(set(self.arm_versions) - set(self.hard_cost_usd_per_arm))
        if missing_cap:
            raise ValueError(
                f"arm(s) {missing_cap} have no hard cost cap; every arm is capped and the "
                "cap is part of the frozen protocol (EV-12)"
            )
        bad = sorted(a for a, c in self.hard_cost_usd_per_arm.items() if c <= 0)
        if bad:
            raise ValueError(f"arm(s) {bad} have a non-positive cap; unlimited is not a cap")
        return self

    @property
    def is_frozen(self) -> bool:
        return self.frozen_at is not None

    def freeze(self, at: str) -> Preregistration:
        """Stamp the preregistration. Refuses a second time.

        Re-freezing would let the protocol be edited between a peek at the results and the
        record of what was promised, which is the one thing preregistration exists to
        prevent.
        """
        if self.is_frozen:
            raise ValueError(
                f"already frozen at {self.frozen_at}; a preregistration is written once, "
                "and re-freezing after seeing results is what it exists to prevent"
            )
        return self.model_copy(update={"frozen_at": at})

    def pairing_is_complete(self, task_ids: tuple[str, ...]) -> list[str]:
        """Task ids with no paired seed. Empty means the run may proceed."""
        return sorted(set(task_ids) - set(self.paired_seeds))


def validate_arms_differ_in_one_variable(
    a: dict[str, object], b: dict[str, object]
) -> list[str]:
    """Fields differing between two ablation configs.

    A diagnostic arm pair must differ in **exactly one** variable, asserted before either
    run starts. Two differences make the readout uninterpretable, and the cost of finding
    that out afterwards is the whole run.
    """
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
