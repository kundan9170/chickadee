"""`report.md` + CSV rows: side-by-side arms, perturbed delta table, diagnostic panel.

The report is what someone reads instead of the archive, which makes every omission here a
claim. Three rules follow, and they are the same rule three times:

**Say what the run could not measure, at the top.** The header carries panel coverage and
the seal's own verdict before any solve rate. A reader who stops after the first table
should already know whether the numbers were produced under a proven seal — not discover it
in a footnote under a panel they did not scroll to.

**Perturbed deltas are reported separately and never averaged in.** The
perturbed set is a *different set of problems*; folding it into the headline would dilute
both numbers into one that describes neither. the reading — that a perturbed delta much
smaller than the original-set delta means memorisation is doing the work — is printed next
to the table, because a delta table without its interpretation invites the comfortable
reading.

**`—` means no value.** Never `0`, never `0.0%`, never a blank cell. `metrics.Ratio` decides
this; the report only prints what it is given.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence

from chickadee_eval import diagnostics
from chickadee_eval.metrics import ArmMetrics, PairedComparison
from chickadee_eval.runner import ProblemRecord, Seal

CSV_COLUMNS = (
    "problem_id",
    "language",
    "arm",
    "outcome",
    "solved",
    "solved_claim",
    "counts_in_denominator",
    "attempts",
    "cost_usd",
    "wall_ms",
    "sandbox",
    "network_verified",
    "oracle_planted",
    "deps_digest",
    "leaked_lines",
    "detail",
)
"""One row per problem, flat, stable order.

`counts_in_denominator` is written explicitly rather than left for a reader to re-derive
from `outcome`: D14 is the rule most likely to be reimplemented wrongly downstream, and a
spreadsheet that recomputed it would be one `!=` away from counting `infra_skip` as unsolved.

`leaked_lines` is a **count**, never the text — the same rule `ProblemRecord` follows. A CSV
is the most-shared artifact of a run, and an archive carrying the oracle outlives the prompt
that leaked it.
"""


def csv_rows(records: Sequence[ProblemRecord]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in records:
        writer.writerow(
            {
                "problem_id": r.problem_id,
                "language": r.language,
                "arm": r.arm,
                "outcome": r.outcome,
                "solved": int(r.solved),
                "solved_claim": int(r.solved_claim),
                "counts_in_denominator": int(r.counts_in_denominator),
                "attempts": r.attempts,
                "cost_usd": f"{r.cost_usd:.6f}",
                "wall_ms": r.wall_ms,
                "sandbox": r.sandbox,
                "network_verified": int(r.network_verified),
                "oracle_planted": int(r.oracle_planted),
                "deps_digest": r.deps_digest,
                "leaked_lines": len(r.leaked_line_numbers),
                "detail": r.detail,
            }
        )
    return out.getvalue()


def seal_banner(sealed: Seal | None) -> str:
    """The seal's verdict, above the numbers.

    §5.2 is blunt about why: contamination was measured at 63% of one frontier model's
    SWE-bench Pro resolutions and sealing cost it 14.1pp. A solve rate produced without a
    proven seal is a different quantity from one produced with it, and the report should not
    let a reader assume which they are looking at.
    """
    if sealed is None:
        return (
            "> **Seal: not recorded.** This report was rendered without a `Seal`, so nothing "
            "here states whether benchmark code ran isolated. Treat every number as "
            "unsealed until shown otherwise.\n"
        )
    if sealed.network_verified:
        return f"> **Seal: verified.** Applied `{sealed.applied}`; {sealed.verification}.\n"
    return (
        f"> **Seal: NOT verified.** Applied `{sealed.applied}` (available "
        f"`{sealed.available}`) — {sealed.verification}. An unsealed number is not a "
        "number (§5.2, EX-2); these results are diagnostic only.\n"
    )


def arm_table(metrics: Mapping[str, ArmMetrics]) -> str:
    """Side-by-side arms. One column per arm, so a reader compares rows not reports."""
    if not metrics:
        return "_no arms_\n"
    names = list(metrics)
    header = "| metric | " + " | ".join(f"`{n}`" for n in names) + " |"
    divider = "|---" * (len(names) + 1) + "|"

    def row(label: str, fn: object) -> str:
        cells = " | ".join(str(fn(metrics[n])) for n in names)  # type: ignore[operator]
        return f"| {label} | {cells} |"

    rows = [
        row("problems", lambda m: len(m.records)),
        row("denominator (D14)", lambda m: m.denominator),
        row("infra_skip (excluded)", lambda m: m.infra_skipped),
        row("solved", lambda m: m.solved),
        row("solve rate", lambda m: m.solve_rate.render(percent=True)),
        row("total cost USD", lambda m: f"{m.total_cost_usd:.4f}"),
        row("cost/task", lambda m: m.cost_per_task.render()),
        row("cost/solved task", lambda m: m.cost_per_solved_task.render()),
        row("claimed solved, oracle disagreed", lambda m: m.claimed_but_unsolved),
        row("oracle-plant attempts", lambda m: m.oracle_plant_attempts),
        row("rows without verified seal", lambda m: m.unsealed_rows),
        row("rows with unpriced cost", lambda m: m.unpriced_rows),
        row("wall ms", lambda m: m.total_wall_ms),
    ]
    return "\n".join([header, divider, *rows]) + "\n"


def perturbed_table(
    original: Mapping[str, ArmMetrics], perturbed: Mapping[str, ArmMetrics]
) -> str:
    """D23's contamination control, **reported separately and never averaged in**.

    The interpretation is printed with the table because the table alone has a comfortable
    reading. §5.3: if the perturbed delta is much smaller than the original-set delta,
    memorisation is doing the work and the headline claim must be restated.
    """
    if not perturbed:
        return (
            "## Perturbed set (D23)\n\n_No perturbed results in this run. Contamination "
            "control did not run — this is not evidence that memorisation is absent._\n"
        )

    lines = [
        "## Perturbed set (D23)\n",
        "Reported separately. **Never averaged into the headline** (EV-5): these are "
        "different problems, and folding them in would produce a number describing "
        "neither set.\n",
        "\n| arm | original solve | perturbed solve | delta pp |",
        "\n|---|---|---|---|\n",
    ]
    for name, pert in perturbed.items():
        orig = original.get(name)
        o = orig.solve_rate.value if orig else None
        p = pert.solve_rate.value
        delta = f"{(p - o) * 100:+.1f}" if (o is not None and p is not None) else "—"
        lines.append(
            f"| `{name}` | {orig.solve_rate.render(percent=True) if orig else '—'} "
            f"| {pert.solve_rate.render(percent=True)} | {delta} |\n"
        )
    lines.append(
        "\n_Reading (§5.3): a perturbed delta much smaller than the original-set delta "
        "means memorisation is doing the work, and the headline claim must be restated._\n"
    )
    return "".join(lines)


def paired_section(comparison: PairedComparison | None) -> str:
    """The paired quantities, over problems both arms attempted."""
    if comparison is None:
        return ""
    delta = comparison.solve_delta_pp
    return (
        f"## Paired comparison — `{comparison.treatment}` vs `{comparison.control}`\n\n"
        f"- paired problems: **{comparison.n}** (both arms attempted)\n"
        f"- Δsolve: **{f'{delta:+.1f} pp' if delta is not None else '—'}**\n"
        f"- cost ratio: **{comparison.cost_ratio.render()}**\n"
        f"- discordant problems: {comparison.discordant or 'none'}\n\n"
        "_Point estimates only. §4's decision rule requires the preregistered paired "
        "interval, which `inference.py` supplies at Phase 6; nothing here passes a gate._\n"
    )


def render(
    metrics: Mapping[str, ArmMetrics],
    *,
    sealed: Seal | None = None,
    perturbed: Mapping[str, ArmMetrics] | None = None,
    original: Mapping[str, ArmMetrics] | None = None,
    comparison: PairedComparison | None = None,
    title: str = "E1 run report",
) -> str:
    """The whole report. Deterministic — same inputs, same bytes, so it can be a golden."""
    panels = diagnostics.panels(metrics)
    # `— ` rather than `0%` when there is no panel to measure, for the same reason every
    # other quantity here does it: a coverage of 0% claims the panel was checked and found
    # empty. Reachable only with no arms today, since `PANEL_SECTIONS` is never empty — but
    # this line previously read `(p.coverage.value or 0)`, which is the substitution the
    # whole module argues against, sitting in the sentence that announces the policy.
    defined = [p.coverage.value for p in panels if p.coverage.value is not None]
    coverage = f"{min(defined) * 100:.0f}%" if defined else "—"

    parts = [
        f"# {title}\n\n",
        seal_banner(sealed),
        "\n",
        f"> **Diagnostic panel coverage: {coverage}.** The rest of §4's panel "
        "has no source in this run and is listed, with reasons, below. Unavailable is not "
        "zero.\n\n",
        "## Arms\n\n",
        arm_table(metrics),
        "\n",
        perturbed_table(original or metrics, perturbed or {}),
        "\n",
        paired_section(comparison),
        "\n",
    ]
    parts.extend(p.render() + "\n" for p in panels)
    return "".join(parts)
