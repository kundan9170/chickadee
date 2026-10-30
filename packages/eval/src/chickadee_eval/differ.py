"""A/B two archived result sets → per-problem deltas + summary.

§10.6 requires this on *every* schema, render or gate PR, which sets the bar: the output has
to make a reviewer able to say "this change moved four problems and cost 3% more" without
opening either archive.

**Per-problem before anything is averaged.** A summary that reported only "+2 solved" hides
the case that matters — six problems fixed and four broken nets to +2, and the four are the
finding. `Delta.regressions` is what a reviewer reads first, which is why it is a separate
field and not something to derive from a rate.

**Problems present in only one side are not deltas.** They are reported as `added` and
`removed`, because a problem that was not attempted did not "fail" — the same D14 distinction
the denominator makes, applied to a comparison. Silently intersecting would let a run that
dropped twenty problems look like an improvement.

**Comparability is checked, not assumed.** Two archives can differ in the manifest they ran,
the arm they ran, or the dependency tree that scored them, and a diff across those is not a
before/after — it is two experiments. `compare` records what differed and the report prints
it above the table.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from chickadee_eval.metrics import ArmMetrics, Ratio
from chickadee_eval.runner import ProblemRecord


class ArchiveError(ValueError):
    """An archive that cannot be read as a result set."""


def load_records(path: pathlib.Path) -> list[ProblemRecord]:
    """Read `results.jsonl`. Accepts the file or the directory holding it.

    Unknown keys are dropped rather than raising: an archive written by a later version is
    still comparable on the fields both versions share, and refusing would make the differ
    useless exactly when a schema change is what is being reviewed — which §10.6 says is
    when it must run.
    """
    target = path / "results.jsonl" if path.is_dir() else path
    if not target.is_file():
        raise ArchiveError(f"no results.jsonl at {target}")
    known = set(ProblemRecord.__dataclass_fields__)
    out: list[ProblemRecord] = []
    for n, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            blob = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArchiveError(f"{target}:{n} is not JSON: {exc}") from exc
        # `null`, `[]` and `3` are all valid JSON and none of them is a row. Checked before
        # `.items()` rather than catching `AttributeError` after, so the message names the
        # file and says what was found instead of reporting that `None` has no attributes.
        if not isinstance(blob, dict):
            raise ArchiveError(
                f"{target}:{n} is valid JSON but not a result object: {type(blob).__name__}"
            )
        try:
            out.append(ProblemRecord(**{k: v for k, v in blob.items() if k in known}))
        except TypeError as exc:
            # A row missing required fields is a malformed archive, not a programming
            # error in the caller. Raised as `ArchiveError` with the line number so the
            # message points at the file rather than at this function's signature.
            raise ArchiveError(f"{target}:{n} is not a result row: {exc}") from exc
    if not out:
        raise ArchiveError(f"{target} holds no rows")
    return out


@dataclass(frozen=True)
class ProblemDelta:
    """One problem, before and after."""

    problem_id: str
    before: ProblemRecord
    after: ProblemRecord

    @property
    def solve_changed(self) -> bool:
        return self.before.solved != self.after.solved

    @property
    def regressed(self) -> bool:
        return self.before.solved and not self.after.solved

    @property
    def fixed(self) -> bool:
        return not self.before.solved and self.after.solved

    @property
    def cost_delta_usd(self) -> float:
        return self.after.cost_usd - self.before.cost_usd

    @property
    def wall_delta_ms(self) -> int:
        return self.after.wall_ms - self.before.wall_ms

    @property
    def outcome_changed(self) -> bool:
        """Includes changes that leave `solved` alone — `fail` becoming `timeout` is a real
        change and an averaged solve rate cannot see it."""
        return self.before.outcome != self.after.outcome

    def symbol(self) -> str:
        if self.fixed:
            return "fixed"
        if self.regressed:
            return "REGRESSED"
        if self.outcome_changed:
            return "outcome"
        return "same"


@dataclass(frozen=True)
class Incomparability:
    """A reason two archives are not a before/after pair."""

    field: str
    before: str
    after: str

    def render(self) -> str:
        return f"{self.field}: `{self.before}` → `{self.after}`"


@dataclass(frozen=True)
class Diff:
    """The result of comparing two archives for one arm."""

    arm: str
    deltas: Sequence[ProblemDelta]
    added: Sequence[str] = field(default_factory=list)
    """In the after set only. Not an improvement — a problem that was not there before."""
    removed: Sequence[str] = field(default_factory=list)
    """In the before set only. Not a regression, and not something to quietly drop."""
    incomparable: Sequence[Incomparability] = field(default_factory=list)

    @property
    def fixed(self) -> list[ProblemDelta]:
        return [d for d in self.deltas if d.fixed]

    @property
    def regressions(self) -> list[ProblemDelta]:
        """Read this first. A net gain with regressions underneath is two findings."""
        return [d for d in self.deltas if d.regressed]

    @property
    def solve_delta(self) -> int:
        return len(self.fixed) - len(self.regressions)

    @property
    def cost_delta_usd(self) -> float:
        return sum(d.cost_delta_usd for d in self.deltas)

    @property
    def cost_ratio(self) -> Ratio:
        return Ratio(
            sum(d.after.cost_usd for d in self.deltas),
            sum(d.before.cost_usd for d in self.deltas),
        )

    @property
    def comparable(self) -> bool:
        return not self.incomparable

    def render(self) -> str:
        lines = [f"## A/B differ — `{self.arm}`\n"]

        if self.incomparable:
            lines.append(
                "> **These archives are not a before/after pair.** The table below is two "
                "experiments side by side, not the effect of a change:\n>\n"
                + "\n".join(f"> - {i.render()}" for i in self.incomparable)
                + "\n"
            )

        lines.append(
            f"- paired problems: **{len(self.deltas)}**\n"
            f"- fixed: **{len(self.fixed)}** · regressed: **{len(self.regressions)}** · "
            f"net solved: **{self.solve_delta:+d}**\n"
            f"- cost: **{self.cost_delta_usd:+.4f}** USD "
            f"(ratio after/before {self.cost_ratio.render()})\n"
        )
        if self.added:
            lines.append(f"- only in *after* (not an improvement): {sorted(self.added)}\n")
        if self.removed:
            lines.append(f"- only in *before* (not a regression): {sorted(self.removed)}\n")

        changed = [d for d in self.deltas if d.solve_changed or d.outcome_changed]
        if not changed:
            lines.append("\n_No problem changed outcome._\n")
            return "".join(lines)

        # Regressions first: the ordering is the point of the table.
        changed.sort(key=lambda d: (not d.regressed, not d.fixed, d.problem_id))
        rows = "\n".join(
            f"| `{d.problem_id}` | {d.symbol()} | {d.before.outcome} → {d.after.outcome} "
            f"| {d.cost_delta_usd:+.4f} |"
            for d in changed
        )
        lines.append(
            "\n| problem | change | outcome | Δcost USD |\n|---|---|---|---|\n" + rows + "\n"
        )
        return "".join(lines)


def _incomparabilities(
    before: Sequence[ProblemRecord], after: Sequence[ProblemRecord]
) -> list[Incomparability]:
    out: list[Incomparability] = []
    for name, values in (
        ("dependency tree", ({r.deps_digest for r in before}, {r.deps_digest for r in after})),
        ("sandbox", ({r.sandbox for r in before}, {r.sandbox for r in after})),
        # `sandbox` records which backend was *applied*; `network_verified` records whether
        # a socket was actually proven to fail. Two runs can agree on the first and differ on
        # the second — `Seal` keeps them apart for exactly this reason — and comparing a
        # proven-sealed run against an unproven one measures the seal, which §5.2 puts at
        # 14.1pp. Checking only `sandbox` left that case reading as the effect of the change.
        (
            "network verification",
            (
                {"verified" if r.network_verified else "unverified" for r in before},
                {"verified" if r.network_verified else "unverified" for r in after},
            ),
        ),
    ):
        lhs, rhs = values
        lhs.discard("")
        rhs.discard("")
        if lhs and rhs and lhs != rhs:
            out.append(Incomparability(name, ", ".join(sorted(lhs)), ", ".join(sorted(rhs))))
    return out


def compare(
    before: Iterable[ProblemRecord], after: Iterable[ProblemRecord], *, arm: str | None = None
) -> Diff:
    """Pair two result sets by `problem_id`.

    `arm` selects one when the sets hold several; with several arms and no selection this
    raises, because averaging arms together is the mistake a differ exists to prevent.
    """
    lhs, rhs = list(before), list(after)
    arms = {r.arm for r in lhs} | {r.arm for r in rhs}
    if arm is None:
        if len(arms) > 1:
            raise ArchiveError(
                f"these archives hold several arms {sorted(arms)}; pass arm= to pick one. "
                "Comparing them together would average arms into a single delta."
            )
        arm = next(iter(arms), "")
    lhs = [r for r in lhs if r.arm == arm]
    rhs = [r for r in rhs if r.arm == arm]

    left = {r.problem_id: r for r in lhs}
    right = {r.problem_id: r for r in rhs}
    shared = sorted(set(left) & set(right))
    return Diff(
        arm=arm,
        deltas=[ProblemDelta(p, left[p], right[p]) for p in shared],
        added=sorted(set(right) - set(left)),
        removed=sorted(set(left) - set(right)),
        incomparable=_incomparabilities(lhs, rhs),
    )


def compare_archives(
    before: pathlib.Path, after: pathlib.Path, *, arm: str | None = None
) -> Diff:
    """The entry point: two archive directories in, one before/after table out."""
    return compare(load_records(before), load_records(after), arm=arm)


def summarize(before: ArmMetrics, after: ArmMetrics) -> str:
    """A one-line headline for a PR comment. The table is the evidence; this is the lede.

    The delta is `—` when either side has no solve rate. The first version wrote
    `(after.solve_rate.value or 0) - (before.solve_rate.value or 0)`, which turns an
    undefined rate into `0.0` and prints a numeric percentage-point change for a comparison
    that was never measured — the exact substitution `Ratio` exists to prevent, in the module
    whose whole argument is that undefined is not zero. Caught in review on #68.
    """
    lhs, rhs = before.solve_rate.value, after.solve_rate.value
    delta = f"{(rhs - lhs) * 100:+.1f} pp" if (lhs is not None and rhs is not None) else "—"
    return (
        f"{after.arm}: solve {before.solve_rate.render(percent=True)} → "
        f"{after.solve_rate.render(percent=True)} ({delta}), "
        f"cost/task {before.cost_per_task.render()} → {after.cost_per_task.render()}"
    )
