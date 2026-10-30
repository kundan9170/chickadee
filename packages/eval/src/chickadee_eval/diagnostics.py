"""The §4 diagnostic panel — a first-class output, not a byproduct.

§4 names six sections and, for each, the event stream that feeds it. At Phase 1 **five of
those streams do not exist**: there is no classifier emitting verdicts, no `classify` or
`gate_result` events, no `patch_applied{reason}`, no `IndexReport.edge_classes`. The engine
that produces them arrives at `T-3.5`.

That is the whole design problem here, and it has a wrong answer that looks tidy: render
the sections anyway, with zeros. An attribution table reading *"model 0%, packer 0%, spec
0%"* is not an empty table — it is a claim that nothing failed for any of those reasons,
which is the opposite of the truth. The same zero, in the gate vector, says every gate
passed. §4 asks that vector to show *"which gate bites, which silently isn't running"*, and
a panel that cannot tell those apart has become the thing it was built to detect.

So every section is a `Section` carrying its own availability and, when absent, **why**.
That mirrors `runner.Seal`, which reports three states rather than two for the same reason:
proven, disproven, or not established. `render()` prints the reason where the table would
have been, so a reader sees the shape of what is missing instead of a plausible zero.

The two sections Phase 1 data *does* support are marked available and carry real numbers.
Nothing is padded to make the panel look complete.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from chickadee_eval.metrics import ArmMetrics, Ratio
from chickadee_eval.problems.polyglot import Outcome

PHASE_1_NO_ENGINE = (
    "no data: the PacketEngine that emits this does not exist yet (T-3.5). Not zero — "
    "unmeasured."
)
"""One reason string, used verbatim by every harness-fed section.

Deliberately identical across sections so a reader sees at a glance that they are absent
for one shared reason rather than five unrelated ones — and so a grep finds every place
that becomes real when the engine lands.
"""


class Availability(StrEnum):
    """Why a panel section has the content it has."""

    MEASURED = "measured"
    """Real data from this run."""
    UNAVAILABLE = "unavailable"
    """No source exists. The section prints its reason, never a zero."""
    EMPTY = "empty"
    """The source exists and reported nothing. Genuinely zero, and says so — distinct from
    `UNAVAILABLE`, because "we looked and found none" is a finding and "we never looked" is
    not."""


@dataclass(frozen=True)
class Section:
    """One row of the panel table."""

    name: str
    availability: Availability
    reason: str = ""
    rows: Sequence[tuple[str, str]] = field(default_factory=list)
    """`(label, value)` pairs, already rendered. The panel is a report artifact, so
    formatting decisions belong here rather than in the reader."""

    @property
    def measured(self) -> bool:
        return self.availability is Availability.MEASURED

    def render(self) -> str:
        head = f"### {self.name}\n"
        if self.availability is not Availability.MEASURED:
            return f"{head}\n_{self.availability}: {self.reason}_\n"
        if not self.rows:
            return f"{head}\n_measured, no rows_\n"
        width = max(len(label) for label, _ in self.rows)
        body = "\n".join(f"| `{label:<{width}}` | {value} |" for label, value in self.rows)
        return f"{head}\n| item | value |\n|---|---|\n{body}\n"


# --- sections with no source at Phase 1 ---------------------------------------
# Each names the event stream §4 assigns it, so the section says what would fill it rather
# than only that something is missing.


def attribution_split(_arm: ArmMetrics) -> Section:
    """*is this the model's fault, the packer's, or the spec's? — opposite fixes.*

    Sourced from classifier verdicts. Rendering a zeroed split would assert that no
    failure had any of those causes, and the three fixes it distinguishes are the reason the
    section exists — getting it backwards costs more than not having it.
    """
    return Section(
        "failure-attribution split",
        Availability.UNAVAILABLE,
        f"source: classifier verdicts (K-4). {PHASE_1_NO_ENGINE}",
    )


def taxonomy(_arm: ArmMetrics) -> Section:
    """*which failure mode dominates.* Sourced from `classify` events."""
    return Section(
        "taxonomy distribution",
        Availability.UNAVAILABLE,
        f"source: `classify` events. {PHASE_1_NO_ENGINE}",
    )


def gate_vector(_arm: ArmMetrics) -> Section:
    """*which gate bites, which silently isn't running* — skips included.

    The section most damaged by a zero. "Gate skipped" and "gate passed" are the two states
    it exists to separate, and an empty table says the second.
    """
    return Section(
        "per-gate outcome vector",
        Availability.UNAVAILABLE,
        f"source: `gate_result` events, including skips. {PHASE_1_NO_ENGINE}",
    )


def reject_reasons(_arm: ArmMetrics) -> Section:
    """Apply-reject reasons — test-edit attempts (D20 gaming proxy), oversized rewrites,
    empty blocks. Sourced from `patch_applied{reason}`."""
    return Section(
        "apply-reject reasons",
        Availability.UNAVAILABLE,
        f"source: `patch_applied{{reason}}`. {PHASE_1_NO_ENGINE}",
    )


def edge_class_health(_arm: ArmMetrics) -> Section:
    """*is the graph resolving, falling back, or blind*. From
    `IndexReport.edge_classes`."""
    return Section(
        "edge-class health",
        Availability.UNAVAILABLE,
        f"source: `IndexReport.edge_classes` (D19). {PHASE_1_NO_ENGINE}",
    )


def cache_and_tokens(_arm: ArmMetrics) -> Section:
    """The per-task token accounting and the cache-hit column.

    Not in the panel table, but §4 asks for it per task and `metrics` cannot supply it:
    `ProblemRecord` carries no token counts, because nothing on the Phase 1 path produces
    them. Listed rather than omitted — a panel that silently dropped the column would let a
    cost comparison look complete while its cache accounting was missing entirely.
    """
    return Section(
        "token and cache accounting",
        Availability.UNAVAILABLE,
        "source: adapter usage per call (D18, D41). `ProblemRecord` carries no token "
        "counts; the arms report cost only. Not zero — unmeasured.",
    )


# --- sections Phase 1 data actually supports ----------------------------------


def outcome_distribution(arm: ArmMetrics) -> Section:
    """What the oracle concluded, per outcome, zeros kept.

    Not the taxonomy — that classifies *why* a failure happened and needs the classifier.
    This is the coarser fact the runner does record, named for what it is so the two are not
    confused when the real taxonomy arrives.
    """
    counts = arm.outcome_counts
    rows = [(name, str(count)) for name, count in counts.items()]
    total = sum(counts.values())
    rows.append(("total", str(total)))
    return Section(
        "outcome distribution",
        Availability.MEASURED if total else Availability.EMPTY,
        "" if total else "the run produced no rows",
        rows if total else [],
    )


def accounting(arm: ArmMetrics) -> Section:
    """D14's accounting, stated so the denominator can be checked by eye."""
    return Section(
        "D14 accounting",
        Availability.MEASURED,
        rows=[
            ("rows", str(len(arm.records))),
            ("denominator", str(arm.denominator)),
            ("infra_skip (excluded)", str(arm.infra_skipped)),
            ("solved", str(arm.solved)),
            ("solve rate", arm.solve_rate.render(percent=True)),
            ("cost/task", arm.cost_per_task.render()),
            ("cost/solved task", arm.cost_per_solved_task.render()),
        ],
    )


def integrity(arm: ArmMetrics) -> Section:
    """Signals that a number should not be trusted at face value.

    Every row here is a reason to discount the arm's own result, which is why they are in
    the panel rather than in a log: a run with 40 unsealed rows and one oracle-plant attempt
    has a solve rate, and the solve rate is not the interesting part.
    """
    return Section(
        "result integrity",
        Availability.MEASURED,
        rows=[
            ("claimed solved, oracle disagreed", str(arm.claimed_but_unsolved)),
            ("solved, arm did not claim it", str(arm.solved_but_unclaimed)),
            ("oracle-plant attempts", str(arm.oracle_plant_attempts)),
            ("rows without a verified seal", str(arm.unsealed_rows)),
            ("rows with unpriced cost", str(arm.unpriced_rows)),
            ("distinct dependency trees", str(len(arm.deps_digests))),
        ],
    )


def per_language(arm: ArmMetrics) -> Section:
    """Solve rate by language, so a two-language claim can be checked per language."""
    return Section(
        "solve rate by language",
        Availability.MEASURED,
        rows=[(name, ratio.render(percent=True)) for name, ratio in arm.by_language.items()],
    )


PANEL_SECTIONS = (
    accounting,
    outcome_distribution,
    per_language,
    integrity,
    attribution_split,
    taxonomy,
    gate_vector,
    reject_reasons,
    edge_class_health,
    cache_and_tokens,
)
"""Measured sections first, then the ones waiting on the engine. Order is deliberate: a
reader should reach the real numbers before the list of what is missing, and should reach
the list of what is missing before deciding the panel is complete."""


@dataclass(frozen=True)
class Panel:
    """One arm's §4 panel."""

    arm: str
    sections: Sequence[Section]

    @property
    def measured(self) -> list[Section]:
        return [s for s in self.sections if s.measured]

    @property
    def unavailable(self) -> list[Section]:
        return [s for s in self.sections if s.availability is Availability.UNAVAILABLE]

    @property
    def coverage(self) -> Ratio:
        """How much of the panel this run could actually fill.

        Printed at the top of the rendered panel. A number a reader can act on: at Phase 1
        it is well under half, and a report that did not say so would look like a complete
        diagnostic with quiet gaps.
        """
        return Ratio(len(self.measured), len(self.sections))

    def render(self) -> str:
        head = (
            f"## Diagnostic panel — `{self.arm}`\n\n"
            f"**Coverage: {self.coverage.render(percent=True)}** of §4's panel "
            f"({len(self.measured)} of {len(self.sections)} sections measured; "
            f"{len(self.unavailable)} unavailable). Unavailable sections print their source "
            "and are **not** zeros.\n"
        )
        return head + "\n" + "\n".join(s.render() for s in self.sections)


def panel(arm: ArmMetrics) -> Panel:
    return Panel(arm.arm, [build(arm) for build in PANEL_SECTIONS])


def panels(metrics: Mapping[str, ArmMetrics]) -> list[Panel]:
    return [panel(m) for m in metrics.values()]


# --- the property T-1.4 is asked to guarantee ---------------------------------


@dataclass(frozen=True)
class AttributionCheck:
    """Whether the attribution split accounts for every failure.

    Three-valued on purpose, and this is the point of the type. With no classifier there are
    no verdicts, and a boolean check would return `True` — "the split is consistent" — for a
    split that does not exist. That is a passing assertion about nothing, which is worse than
    a failing one: it would sit in a test suite reading as evidence.
    """

    established: bool
    holds: bool
    reason: str
    attributed: int = 0
    failures: int = 0


def attribution_covers_all_failures(
    arm: ArmMetrics, verdicts: Mapping[str, str] | None = None
) -> AttributionCheck:
    """Check the "split sums to 100% of failures".

    `verdicts` maps `problem_id` to an attribution category. `None` means no classifier ran
    — the Phase 1 case — and the answer is *not established*, never *holds*.
    """
    failures = [r for r in arm.attempted if not r.solved and r.outcome != Outcome.INFRA_SKIP]
    if verdicts is None:
        return AttributionCheck(
            established=False,
            holds=False,
            reason=f"no classifier verdicts. {PHASE_1_NO_ENGINE}",
            failures=len(failures),
        )
    attributed = sum(1 for r in failures if r.problem_id in verdicts)
    if attributed == len(failures):
        return AttributionCheck(
            True, True, "every failure carries a verdict", attributed, len(failures)
        )
    missing = [r.problem_id for r in failures if r.problem_id not in verdicts]
    return AttributionCheck(
        established=True,
        holds=False,
        reason=(
            f"{len(missing)} of {len(failures)} failures have no verdict, so the split "
            f"cannot sum to 100%: {missing[:5]}"
        ),
        attributed=attributed,
        failures=len(failures),
    )
