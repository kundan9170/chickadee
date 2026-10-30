"""Benchmark, subset, arm and config definitions.

Three manifests, and the rule that shapes all of them: **counts are recorded, not
assumed**. A manifest that says "~30 problems" is a plan; a manifest that says which
30, by id, is evidence. The difference matters at G3 and G6, where the question is whether
a number moved — and a set that quietly changed size between runs answers that wrong.

Tune on dev, gate on confirmation. `CONFIRMATION` is never read by tuning work,
which is why it is a distinct kind here rather than a flag on one list: a boolean is easy
to pass the wrong way round, and the cost of doing so is a gate that measured what it was
fitted to.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

WIRE = ConfigDict(extra="forbid", frozen=True)


class SubsetKind(StrEnum):
    DEV = "dev"
    """Iteration and tuning. Freely used."""
    CONFIRMATION = "confirmation"
    """Held out. Gates only, never touched by tuning."""
    PERTURBED = "perturbed"
    """Contamination control: seeded renames and spec tweaks, reported separately."""


class Language(StrEnum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class Problem(BaseModel):
    """One benchmark problem, named by the id its loader resolves."""

    model_config = WIRE

    id: str
    language: Language


class Subset(BaseModel):
    """A committed problem set. Its size is whatever `problems` actually holds."""

    model_config = WIRE

    name: str
    kind: SubsetKind
    problems: tuple[Problem, ...]
    seed: int | None = None
    """Required for `perturbed`: D23's generator must be reproducible, and a perturbation
    nobody can regenerate is an anecdote."""

    @property
    def count(self) -> int:
        return len(self.problems)

    def by_language(self) -> dict[Language, int]:
        out: dict[Language, int] = {lang: 0 for lang in Language}
        for p in self.problems:
            out[p.language] += 1
        return out

    @model_validator(mode="after")
    def _check(self) -> Subset:
        ids = [p.id for p in self.problems]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"{self.name}: duplicate problem ids {dupes}")
        if self.kind is SubsetKind.PERTURBED and self.seed is None:
            raise ValueError(f"{self.name}: a perturbed subset needs a seed (D23)")
        if self.kind is not SubsetKind.PERTURBED and self.seed is not None:
            raise ValueError(f"{self.name}: only a perturbed subset carries a seed")
        # E1 is python + javascript only. The other four Polyglot languages
        # have no profile, so a problem in one could never be attempted, and counting it
        # in a denominator would understate the solve rate for a reason unrelated to the
        # harness.
        return self


class Arm(BaseModel):
    """One side of a comparison. Everything that changes behaviour is named here."""

    model_config = WIRE

    name: str
    harness: str
    harness_version: str
    frontier_model: str | None = None
    slm_model: str | None = None
    flags: tuple[str, ...] = ()
    hard_cost_usd: float = Field(gt=0)
    """Finite, always. An unbounded arm cannot be compared on cost and cannot be
    preregistered, since the cap is part of the frozen protocol."""


class Manifest(BaseModel):
    """A committed experiment definition (E1, E2 or E3)."""

    model_config = WIRE

    experiment: str
    subsets: tuple[Subset, ...]
    arms: tuple[Arm, ...]

    @model_validator(mode="after")
    def _check(self) -> Manifest:
        names = [s.name for s in self.subsets]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.experiment}: subset names must be unique, got {names}")
        arm_names = [a.name for a in self.arms]
        if len(set(arm_names)) != len(arm_names):
            raise ValueError(f"{self.experiment}: arm names must be unique, got {arm_names}")
        # Tuning and gating sets must not share problems, or the gate measures what the
        # tuning fitted. Checked mechanically because the overlap is invisible by
        # eye once either set is more than a screenful.
        dev = {p.id for s in self.subsets if s.kind is SubsetKind.DEV for p in s.problems}
        conf = {
            p.id for s in self.subsets if s.kind is SubsetKind.CONFIRMATION for p in s.problems
        }
        leaked = sorted(dev & conf)
        if leaked:
            raise ValueError(
                f"{self.experiment}: {leaked} appear in both the dev and confirmation "
                "sets; the confirmation set is never touched by tuning (EV-11)"
            )
        return self

    def subset(self, kind: SubsetKind) -> Subset | None:
        return next((s for s in self.subsets if s.kind is kind), None)
