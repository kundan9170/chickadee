"""An agent's report: one shape, its evidence checked by code (Amendment 1
decision 3).

**The shape is Markdown**, as every hand-off is since: a model writes a document, not a
JSON object, and a heading is something it gets right. Three sections always, and the skill's
`chickadee-report` sections after them:

```
## Summary
What was found, in a few sentences.
## Findings
- One claim per bullet, citing where it is: `src/app/auth.py:40-52`.
## Not done
What was asked and not done, or "nothing".
```

A report missing a section is asked for again, once by default (`<profile>.report_reasks`), by
the shared loop's re-ask (`tools/loop.py`). One that still does not parse ends `failed`, and the
harness never writes one in its place.

**Evidence is checked, never believed.** Each finding's citations go through the citation gate
(`context/citations.py`): the file and lines must exist, and where the agent read those
lines, they must still say what it saw; a web finding's quote must be on the page the round
fetched. What fails is flagged on the finding, and the report's
header says how many were checked and on what basis. Nothing here judges whether a claim is
true: that is the reader's, and the header says so.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from chickadee.context.citations import (
    CitationGateResult,
    LineSource,
    SeenSource,
    WebSource,
    claims,
    gate,
)
from chickadee.schema.adapter import ParseError
from chickadee.schema.journal import AgentStatus

__all__ = [
    "BASE_SECTIONS",
    "AgentReport",
    "ReportDraft",
    "ReportFinding",
    "check",
    "contract",
    "parse",
]

BASE_SECTIONS: Final = ("Summary", "Findings", "Not done")

_HEADING: Final = re.compile(r"^##\s+(?P<name>.+?)\s*#*\s*$")
_BULLET: Final = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")


@dataclass(frozen=True)
class ReportDraft:
    """A report as the model wrote it, its sections found."""

    sections: Mapping[str, str]
    """Each section's text by its heading, as the contract names it."""

    @property
    def summary(self) -> str:
        return self.sections["Summary"]

    @property
    def findings(self) -> str:
        return self.sections["Findings"]

    @property
    def not_done(self) -> str:
        return self.sections["Not done"]


@dataclass(frozen=True)
class ReportFinding:
    claim: str
    cited: int
    """How many locations the claim cites."""
    flagged: tuple[str, ...] = ()
    """Each cited location the gate did not pass, as it renders it."""


@dataclass(frozen=True)
class AgentReport:
    agent: str
    """`A1`, `A2`,... in the order the session spawned them."""
    round: int
    """0 for the spawn; `n` for the `n`th follow-up."""
    skill: str | None
    profile: str
    status: AgentStatus
    summary: str = ""
    findings: tuple[ReportFinding, ...] = ()
    sections: Mapping[str, str] = field(default_factory=dict[str, str])
    """The skill's own sections, in its order."""
    not_done: str = ""
    checked: CitationGateResult | None = None
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    transcript: str | None = None
    """The handle of the round's messages, for Zeus to read when it asks to."""
    why: str = ""
    """For `failed` and `cancelled`, and a `partial` the agent did not explain: what ended
    it."""

    def render(self) -> str:
        """What Zeus reads: the facts the harness observed, then the agent's words, apart (the
        same split `skills/frame.py` keeps between a skill's facts and its body)."""
        skill = f" · skill {self.skill}" if self.skill else ""
        facts = (
            f"[agent {self.agent} · round {self.round}{skill} · {self.profile} · "
            f"{self.status} · {self.tool_calls} tool calls · {self.tokens:,} tokens · "
            f"${self.cost_usd:.4f}"
            + (f" · transcript {self.transcript}" if self.transcript else "")
            + "]"
        )
        lines = [facts]
        if self.why:
            lines.append(f"[{self.why}]")
        if self.checked is not None:
            lines.append(self.checked.render())
        if self.status == "failed" or (self.status == "cancelled" and not self.summary):
            return "\n".join(lines)
        checked = (
            "the locations and quotes" if self.checked and self.checked.web else "the locations"
        )
        lines.append(
            f"An agent's report: its claims are its own, and only {checked} above were checked."
        )
        lines += ["## Summary", self.summary, "## Findings"]
        lines += [_finding(f) for f in self.findings] or ["(none)"]
        for name, text in self.sections.items():
            lines += [f"## {name}", text]
        lines += ["## Not done", self.not_done or "nothing"]
        return "\n".join(lines)


def _finding(finding: ReportFinding) -> str:
    flags = "".join(f"\n  FLAGGED: {f}" for f in finding.flagged)
    return f"- {finding.claim}{flags}"


def contract(extra: Sequence[str]) -> tuple[str, ...]:
    """The sections a report must carry: the three, then the skill's own."""
    return (*BASE_SECTIONS, *(s for s in extra if s not in BASE_SECTIONS))


def parse(text: str, required: Sequence[str]) -> ReportDraft | ParseError:
    """`text`'s sections, or the error a re-ask names the missing ones with. A heading is
    matched by name, case and trailing punctuation aside; text before the first heading is
    not part of any section."""
    found: dict[str, list[str]] = {}
    current: list[str] | None = None
    wanted = {_key(r): r for r in required}
    for line in text.replace("\r\n", "\n").split("\n"):
        heading = _HEADING.match(line)
        if heading is not None:
            name = wanted.get(_key(heading.group("name")))
            current = found.setdefault(name, []) if name is not None else None
            continue
        if current is not None:
            current.append(line)
    missing = [r for r in required if r not in found]
    if missing:
        named = ", ".join(f"`## {m}`" for m in missing)
        return ParseError(
            kind="bad_format",
            hint=(
                f"your report is missing {named}. Send the whole report again with every "
                f"section, in this order: {', '.join(f'## {r}' for r in required)}"
            ),
        )
    return ReportDraft({r: "\n".join(found[r]).strip() for r in required})


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def check(
    draft: ReportDraft,
    *,
    read: LineSource,
    seen: SeenSource | None,
    fetched: WebSource | None = None,
) -> tuple[tuple[ReportFinding, ...], CitationGateResult]:
    """Each finding, with what the citation gate flagged on it, and the gate's whole result.
    A finding is a bullet of `## Findings`; a findings section saying there are none has
    none. `fetched` is the round's pages, for a finding citing the web."""
    bullets = _bullets(draft.findings)
    result = gate("\n".join(f"- {b}" for b in bullets), read=read, seen=seen, fetched=fetched)
    flagged: dict[int, list[str]] = {}
    for c in result.rejected:
        flagged.setdefault(c.citation.claim, []).append(c.render())
    cited: dict[int, int] = {}
    for c in result.checked:
        cited[c.citation.claim] = cited.get(c.citation.claim, 0) + 1
    # `gate` numbers its claims by `claims()` over the bullets, one bullet one or more
    # sentences; a finding is a bullet, so its sentences' numbers fold back to it.
    owner = _owners(bullets)
    findings: list[ReportFinding] = []
    for i, bullet in enumerate(bullets):
        mine = [n for n, b in owner.items() if b == i]
        findings.append(
            ReportFinding(
                bullet,
                sum(cited.get(n, 0) for n in mine),
                tuple(f for n in mine for f in flagged.get(n, ())),
            )
        )
    return tuple(findings), result


def _bullets(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for line in text.split("\n"):
        if _BULLET.match(line):
            out.append(_BULLET.sub("", line).strip())
        elif line.strip() and out:
            out[-1] = f"{out[-1]} {line.strip()}"
    return tuple(b for b in out if b)


def _owners(bullets: Sequence[str]) -> dict[int, int]:
    """Which bullet each of `claims()`'s units came from."""
    owner: dict[int, int] = {}
    n = 0
    for i, bullet in enumerate(bullets):
        for _ in claims(f"- {bullet}"):
            owner[n] = i
            n += 1
    return owner
