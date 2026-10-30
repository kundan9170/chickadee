"""Aggregation over per-problem records.

Turning 83 rows into one number is where a benchmark most easily flatters itself, so every
quantity here is either derived from data that exists or reported as **absent**. Three
places that matters:

**D14's denominator.** `infra_skip` leaves it. A problem whose toolchain was missing was
never attempted, and scoring it unsolved understates the solve rate for a reason that has
nothing to do with the harness. `REPLANNED` and `UNSATISFIABLE` go the other way — they are
the harness declining, which is a real failure to solve, so they count unsolved and stay in
the denominator. Both directions are easy to get backwards and both are asserted.

**Cost per solved task with nothing solved.** `spend / 0` is not infinity and not zero; it
is a quantity that does not exist. It is `None`, and the report prints `—`. A zero there
reads as "free", which is the exact inversion the arm would want.

**Rates over an empty denominator.** Same rule. A solve rate of `0.0` from zero attempted
problems is a claim about a benchmark that never ran.

`Ratio` exists so those two cases cannot be collapsed by accident: it carries the numerator
and denominator, and refuses to produce a value when the denominator is zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from chickadee_eval.manifest import Language
from chickadee_eval.problems.polyglot import Outcome
from chickadee_eval.runner import ProblemRecord

UNSOLVED_TERMINALS: tuple[str, ...] = ("replanned", "unsatisfiable", "crash")
"""Harness terminals that count as **unsolved** rather than excluded.

The distinction from `infra_skip` is the whole of D14: a missing toolchain means the problem
could never have been attempted, while a harness that replanned until it gave up attempted
it and failed. Counting the first unsolved understates the solve rate; excluding the second
overstates it.

Recorded here as strings because at Phase 1 nothing emits them — the engine that would is
`T-3.5`. `outcome_terminal()` reads `ProblemRecord.detail`, and `terminal_counts` reports
zero honestly rather than implying the harness never replanned.
"""


@dataclass(frozen=True)
class Ratio:
    """A quotient that knows when it has no value.

    Both parts are kept rather than a float, because "0 of 0" and "0 of 83" are different
    facts and a report that printed `0.0%` for both would be wrong about one of them.
    """

    numerator: float
    denominator: float

    @property
    def value(self) -> float | None:
        """`None` when the denominator is zero — never `0.0`, never `inf`."""
        return self.numerator / self.denominator if self.denominator else None

    @property
    def defined(self) -> bool:
        return self.denominator != 0

    def render(self, *, percent: bool = False, places: int = 3) -> str:
        """`—` when undefined. The em dash is the report's one way of saying "no value"."""
        v = self.value
        if v is None:
            return "—"
        return f"{v * 100:.1f}%" if percent else f"{v:.{places}f}"


@dataclass(frozen=True)
class ArmMetrics:
    """Everything §4 asks per arm that Phase 1 data can actually support.

    Fields the spec lists and this cannot fill — tokens by model and purpose, cache-hit
    rate, escalation and rescue rates, human-intervention rate — are **absent rather than
    zero**, and `diagnostics` renders them as unavailable with the reason. `ProblemRecord`
    carries no token counts because nothing on the Phase 1 path produces them.
    """

    arm: str
    records: Sequence[ProblemRecord]

    # --- D14 accounting ------------------------------------------------------
    @property
    def attempted(self) -> list[ProblemRecord]:
        """Rows that count. `infra_skip` is not among them."""
        return [r for r in self.records if r.counts_in_denominator]

    @property
    def denominator(self) -> int:
        return len(self.attempted)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.attempted if r.solved)

    @property
    def infra_skipped(self) -> int:
        return len(self.records) - self.denominator

    @property
    def solve_rate(self) -> Ratio:
        return Ratio(self.solved, self.denominator)

    # --- cost ------------------------------------------------------
    @property
    def total_cost_usd(self) -> float:
        """Including spend on failed tasks (the E2 rule says so explicitly).

        Every row, not just attempted ones: an `infra_skip` costs nothing today, but a rule
        that summed only the denominator would quietly drop real spend the day it does.
        """
        return sum(r.cost_usd for r in self.records)

    @property
    def cost_per_task(self) -> Ratio:
        return Ratio(self.total_cost_usd, self.denominator)

    @property
    def cost_per_solved_task(self) -> Ratio:
        """Undefined when nothing was solved — not zero, and not infinity."""
        return Ratio(self.total_cost_usd, self.solved)

    @property
    def unpriced_rows(self) -> int:
        """Rows whose arm said its cost was not a measurement.

        `e1_naive` and `api_baseline` both label an unpriced zero in `detail`. A cost
        column summed over rows that are mostly unpriced is a number with a footnote, and
        the footnote belongs next to it — `diagnostics.integrity` prints this count beside
        the cost figures for that reason.
        """
        return sum(1 for r in self.records if "unpriced" in r.detail)

    # --- honesty counters ----------------------------------------------------
    @property
    def claimed_but_unsolved(self) -> int:
        """The arm said solved; the oracle disagreed.

        §4 asks for a *checkpoint*-pass-but-acceptance-fail rate. Checkpoints do not exist
        at Phase 1, and this is the same question one layer out: how often the thing under
        test believes it succeeded and the independent oracle says otherwise. Named for what
        it measures rather than for the metric it stands in for.
        """
        return sum(1 for r in self.attempted if r.solved_claim and not r.solved)

    @property
    def solved_but_unclaimed(self) -> int:
        """The oracle passed while the arm thought it had failed. Rarer and worth seeing:
        it usually means the arm's own success signal is broken."""
        return sum(1 for r in self.attempted if r.solved and not r.solved_claim)

    @property
    def oracle_plant_attempts(self) -> int:
        """Rows where the arm wrote a file where the hidden tests go.

        Always overwritten, so it cannot succeed — but attempting it is a finding, and a
        panel that did not surface it would leave the attempt in a JSONL nobody reads.
        """
        return sum(1 for r in self.records if r.oracle_planted)

    @property
    def unsealed_rows(self) -> int:
        """Rows scored without a proven-sealed environment."""
        return sum(1 for r in self.records if not r.network_verified)

    # --- distributions -------------------------------------------------------
    @property
    def outcome_counts(self) -> dict[str, int]:
        """Every `Outcome`, including the zeros.

        Zeros are kept on purpose: a distribution that omitted `timeout` because it did not
        occur reads identically to one where timeouts were never measured.
        """
        counts = {o.value: 0 for o in Outcome}
        for r in self.records:
            counts[r.outcome] = counts.get(r.outcome, 0) + 1
        return counts

    @property
    def terminal_counts(self) -> dict[str, int]:
        """D14's unsolved terminals. All zero at Phase 1 — see `UNSOLVED_TERMINALS`."""
        counts: dict[str, int] = dict.fromkeys(UNSOLVED_TERMINALS, 0)
        for r in self.attempted:
            terminal = outcome_terminal(r)
            if terminal:
                counts[terminal] += 1
        return counts

    @property
    def by_language(self) -> dict[str, Ratio]:
        """Solve rate per language, so a two-language claim can be checked per language.

        A headline that averaged 34 Python and 49 JavaScript problems into one number would
        hide a harness that works in one of them and not the other.
        """
        out: dict[str, Ratio] = {}
        for language in Language:
            rows = [r for r in self.attempted if r.language == language]
            out[language.value] = Ratio(sum(1 for r in rows if r.solved), len(rows))
        return out

    @property
    def deps_digests(self) -> set[str]:
        """Which pinned dependency trees scored these rows.

        More than one in a single arm means the run changed underneath itself, which makes
        the aggregate a mixture rather than a measurement.
        """
        return {r.deps_digest for r in self.records if r.deps_digest}

    @property
    def total_wall_ms(self) -> int:
        return sum(r.wall_ms for r in self.records)

    @property
    def total_attempts(self) -> int:
        return sum(r.attempts for r in self.attempted)


def outcome_terminal(record: ProblemRecord) -> str | None:
    """The D14 terminal a row reports, if any.

    Read out of `detail` because `ProblemRecord` has no terminal field: at Phase 1 nothing
    produces one, and inventing a schema field for a value no producer emits would put an
    always-empty column in every archive. When `T-3.5` lands an engine that emits terminals,
    this is the one function that has to change.
    """
    detail = record.detail.lower()
    return next((t for t in UNSOLVED_TERMINALS if t in detail), None)


def aggregate(records: Iterable[ProblemRecord]) -> dict[str, ArmMetrics]:
    """Group rows by arm. One `ArmMetrics` per arm, in first-seen order."""
    grouped: dict[str, list[ProblemRecord]] = {}
    for record in records:
        grouped.setdefault(record.arm, []).append(record)
    return {arm: ArmMetrics(arm, rows) for arm, rows in grouped.items()}


@dataclass(frozen=True)
class PairedComparison:
    """Two arms over the problems **both** attempted.

    Paired on `problem_id`, and only where both arms have a row in the denominator. An
    unpaired comparison across differently-sized sets is not the quantity the decision rule
    is about, and the difference is largest exactly when one arm skipped more problems —
    which is when it matters most.
    """

    treatment: str
    control: str
    pairs: Sequence[tuple[ProblemRecord, ProblemRecord]]

    @property
    def n(self) -> int:
        return len(self.pairs)

    @property
    def solve_delta_pp(self) -> float | None:
        """`solve_treatment` minus `solve_control`, in percentage points. `None` when
        unpaired."""
        if not self.pairs:
            return None
        t = sum(1 for a, _ in self.pairs if a.solved)
        c = sum(1 for _, b in self.pairs if b.solved)
        return (t - c) / len(self.pairs) * 100

    @property
    def cost_ratio(self) -> Ratio:
        """`cost_treatment / cost_control`, over the paired problems only."""
        return Ratio(
            sum(a.cost_usd for a, _ in self.pairs),
            sum(b.cost_usd for _, b in self.pairs),
        )

    @property
    def discordant(self) -> list[str]:
        """Problems where exactly one arm solved it. The rows a paired test is made of, and
        the ones worth reading when a delta looks surprising."""
        return [a.problem_id for a, b in self.pairs if a.solved != b.solved]


def pair(metrics: Mapping[str, ArmMetrics], treatment: str, control: str) -> PairedComparison:
    """Build the paired set for two arms. Missing arms raise rather than pairing zero rows."""
    for name in (treatment, control):
        if name not in metrics:
            raise KeyError(f"{name!r} is not among the arms {sorted(metrics)}")
    left = {r.problem_id: r for r in metrics[treatment].attempted}
    right = {r.problem_id: r for r in metrics[control].attempted}
    shared = sorted(set(left) & set(right))
    return PairedComparison(treatment, control, [(left[p], right[p]) for p in shared])
