"""What one researcher round scored, and what a run of them adds up to.

**Citation accuracy** is read from the round's own check (`AgentReport.checked`, the gate of
D232 decision 7): its web citations, quoted or flagged, and its claims, cited or not. **Source
quality** is the share of those web citations on the question's preferred hosts. **Factual
accuracy** is the judge's: each fact of the key stated, contradicted or missing. Every rate is
said with its counts, since a rate of three is not a rate of thirty, and a rate over nothing is
said as none, never as zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

from chickadee.engine import AgentReport, CitationVerdict, WebCitation
from chickadee_eval.research.judge import FactVerdict, Judged
from chickadee_eval.research.questions import ResearchQuestion

__all__ = [
    "Citations",
    "ResearchRound",
    "Sources",
    "Summary",
    "compare",
    "score_round",
    "summarize",
]


@dataclass(frozen=True)
class Citations:
    web: int = 0
    """Web citations checked."""
    quoted: int = 0
    not_fetched: int = 0
    not_on_page: int = 0
    claims: int = 0
    cited: int = 0
    """Claims that cite anything, a file or a page."""


@dataclass(frozen=True)
class Sources:
    web: int = 0
    preferred: int = 0
    hosts: tuple[str, ...] = ()
    """Every host a web citation names, once each, in the order cited."""


@dataclass(frozen=True)
class ResearchRound:
    """One round's record: a line of the run's `results.jsonl`."""

    question: str
    repeat: int
    status: str
    """The report's: `complete`, `partial`, `failed`, `cancelled`, or `refused` when the
    round never ran (the spawn was refused)."""
    why: str = ""
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    citations: Citations = field(default_factory=Citations)
    sources: Sources = field(default_factory=Sources)
    facts: tuple[FactVerdict, ...] = ()
    judge_tokens: int = 0
    judge_cost_usd: float = 0.0

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(raw: dict[str, Any]) -> ResearchRound:
        return ResearchRound(
            **{
                **raw,
                "citations": Citations(**raw["citations"]),
                "sources": Sources(
                    **{**raw["sources"], "hosts": tuple(raw["sources"]["hosts"])}
                ),
                "facts": tuple(FactVerdict(**f) for f in raw["facts"]),
            }
        )

    def stated(self) -> int:
        return sum(1 for f in self.facts if f.verdict == "stated")


def score_round(
    question: ResearchQuestion,
    repeat: int,
    report: AgentReport,
    seconds: float,
    judged: Judged | None,
) -> ResearchRound:
    """`report` scored: its citations as its own check found them, its sources against the
    question's preferred hosts, its facts as `judged` found them (none for a round with no
    report to judge)."""
    checked = report.checked
    web: list[WebCitation] = []
    verdicts: list[CitationVerdict] = []
    if checked is not None:
        for c in checked.checked:
            if isinstance(c.citation, WebCitation):
                web.append(c.citation)
                verdicts.append(c.verdict)
    citations = Citations(
        web=len(web),
        quoted=verdicts.count(CitationVerdict.QUOTED),
        not_fetched=verdicts.count(CitationVerdict.NOT_FETCHED),
        not_on_page=verdicts.count(CitationVerdict.NOT_ON_PAGE),
        claims=checked.claims if checked is not None else 0,
        cited=checked.cited_claims if checked is not None else 0,
    )
    hosts = [(urlsplit(c.url).hostname or "").lower() for c in web]
    preferred = sum(1 for h in hosts if _preferred(h, question.prefer))
    return ResearchRound(
        question=question.id,
        repeat=repeat,
        status=report.status,
        why=report.why,
        tool_calls=report.tool_calls,
        tokens=report.tokens,
        cost_usd=report.cost_usd,
        seconds=round(seconds, 1),
        citations=citations,
        sources=Sources(len(web), preferred, tuple(dict.fromkeys(h for h in hosts if h))),
        facts=judged.verdicts if judged is not None else (),
        judge_tokens=judged.tokens if judged is not None else 0,
        judge_cost_usd=judged.cost_usd if judged is not None else 0.0,
    )


def _preferred(host: str, prefer: Sequence[str]) -> bool:
    return any(host == p or host.endswith(f".{p}") for p in prefer)


@dataclass(frozen=True)
class Rate:
    hits: int
    of: int

    def render(self) -> str:
        if not self.of:
            return "none"
        return f"{100 * self.hits / self.of:.0f}% ({self.hits}/{self.of})"

    @property
    def value(self) -> float | None:
        return self.hits / self.of if self.of else None


@dataclass(frozen=True)
class QuestionRow:
    question: str
    attempts: int
    facts: Rate
    spread: tuple[int, int]
    """The fewest and the most facts one repeat stated: how far the repeats disagree."""
    quoted: Rate
    preferred: Rate
    cost_usd: float


@dataclass(frozen=True)
class Summary:
    attempts: int
    statuses: dict[str, int]
    facts: Rate
    """Facts stated, of the facts judged."""
    contradicted: Rate
    unjudged: int
    quoted: Rate
    """Web citations quoted from the page, of those checked: citation accuracy."""
    cited: Rate
    """Claims citing anything, of the claims."""
    preferred: Rate
    """Web citations on a preferred host, of those checked: source quality."""
    cost_usd: float
    judge_cost_usd: float
    seconds: float
    tool_calls: int
    rows: tuple[QuestionRow, ...]

    def render(self, title: str) -> str:
        statuses = ", ".join(f"{k} {v}" for k, v in sorted(self.statuses.items()))
        lines = [
            f"# {title}",
            "",
            f"{self.attempts} rounds: {statuses}.",
            "",
            "| score | value |",
            "|---|---|",
            f"| facts stated (judge) | {self.facts.render()} |",
            f"| facts contradicted (judge) | {self.contradicted.render()} |",
            f"| facts unjudged | {self.unjudged} |",
            f"| citation accuracy: quotes on the page | {self.quoted.render()} |",
            f"| claims cited | {self.cited.render()} |",
            f"| source quality: on a preferred host | {self.preferred.render()} |",
            f"| researcher cost | {self.cost_usd:.4f} USD |",
            f"| judge cost | {self.judge_cost_usd:.4f} USD |",
            f"| time | {self.seconds:.0f} s |",
            f"| tool calls | {self.tool_calls} |",
            "",
            "| question | rounds | facts stated | repeats stated | quoted | preferred | cost |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            low, high = r.spread
            lines.append(
                f"| {r.question} | {r.attempts} | {r.facts.render()} | {low}-{high} | "
                f"{r.quoted.render()} | {r.preferred.render()} | {r.cost_usd:.4f} |"
            )
        lines += [
            "",
            "Facts are the judge's reading of each report against a person's answer key; "
            "quotes and sources are the harness's own check. Neither says a report is true.",
        ]
        return "\n".join(lines) + "\n"


def summarize(attempts: Sequence[ResearchRound]) -> Summary:
    statuses: dict[str, int] = {}
    for a in attempts:
        statuses[a.status] = statuses.get(a.status, 0) + 1
    judged = [f for a in attempts for f in a.facts if f.verdict != "unjudged"]
    rows: list[QuestionRow] = []
    for question in dict.fromkeys(a.question for a in attempts):
        mine = [a for a in attempts if a.question == question]
        mine_judged = [f for a in mine for f in a.facts if f.verdict != "unjudged"]
        stated = [a.stated() for a in mine]
        rows.append(
            QuestionRow(
                question,
                len(mine),
                Rate(sum(stated), len(mine_judged)),
                (min(stated), max(stated)),
                Rate(sum(a.citations.quoted for a in mine), sum(a.citations.web for a in mine)),
                Rate(sum(a.sources.preferred for a in mine), sum(a.sources.web for a in mine)),
                round(sum(a.cost_usd for a in mine), 6),
            )
        )
    return Summary(
        attempts=len(attempts),
        statuses=statuses,
        facts=Rate(sum(1 for f in judged if f.verdict == "stated"), len(judged)),
        contradicted=Rate(sum(1 for f in judged if f.verdict == "contradicted"), len(judged)),
        unjudged=sum(1 for a in attempts for f in a.facts if f.verdict == "unjudged"),
        quoted=Rate(
            sum(a.citations.quoted for a in attempts), sum(a.citations.web for a in attempts)
        ),
        cited=Rate(
            sum(a.citations.cited for a in attempts), sum(a.citations.claims for a in attempts)
        ),
        preferred=Rate(
            sum(a.sources.preferred for a in attempts), sum(a.sources.web for a in attempts)
        ),
        cost_usd=round(sum(a.cost_usd for a in attempts), 6),
        judge_cost_usd=round(sum(a.judge_cost_usd for a in attempts), 6),
        seconds=sum(a.seconds for a in attempts),
        tool_calls=sum(a.tool_calls for a in attempts),
        rows=tuple(rows),
    )


def compare(before: Summary, after: Summary) -> str:
    """Two runs' scores side by side, and each question whose facts stated moved."""
    rows = [
        ("facts stated (judge)", before.facts, after.facts),
        ("facts contradicted (judge)", before.contradicted, after.contradicted),
        ("citation accuracy: quotes on the page", before.quoted, after.quoted),
        ("claims cited", before.cited, after.cited),
        ("source quality: on a preferred host", before.preferred, after.preferred),
    ]
    lines = ["| score | before | after |", "|---|---|---|"]
    lines += [f"| {name} | {b.render()} | {a.render()} |" for name, b, a in rows]
    lines += [
        f"| researcher cost | {before.cost_usd:.4f} USD | {after.cost_usd:.4f} USD |",
        f"| tool calls | {before.tool_calls} | {after.tool_calls} |",
        f"| time | {before.seconds:.0f} s | {after.seconds:.0f} s |",
    ]
    earlier = {r.question: r for r in before.rows}
    moved = [
        (r, earlier[r.question])
        for r in after.rows
        if r.question in earlier and r.facts.value != earlier[r.question].facts.value
    ]
    if moved:
        lines += ["", "| question | facts stated before | after |", "|---|---|---|"]
        lines += [
            f"| {r.question} | {b.facts.render()} | {r.facts.render()} |" for r, b in moved
        ]
    only = sorted({r.question for r in before.rows} ^ {r.question for r in after.rows})
    if only:
        lines += ["", f"Asked in one run only, so not compared: {', '.join(only)}."]
    return "\n".join(lines) + "\n"
