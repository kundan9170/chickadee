"""The judge: each fact of a question's answer key, found stated, contradicted or missing in a
report. A model of the eval's choosing (`--judge`, a catalog model of the
person's engine config), never the model the researcher ran on, so a model is never asked to
grade its own answers. Its prompt is `templates/judge.md`; it is asked through the engine
facade (`PacketEngine.ask_model`), outside the run's session, so its spend is the eval's.

The judge says only what a report states, never whether a fact is true: the answer key is a
person's. An answer that is not the one JSON object asked for, or that skips a fact, is
`unjudged` for each fact it does not answer, never a guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

from chickadee.engine import HostConfig, PacketEngine
from chickadee_eval.research.questions import ResearchQuestion

__all__ = [
    "TEMPLATE",
    "FactVerdict",
    "Judge",
    "Judged",
    "ModelJudge",
    "Verdict",
    "check_judge",
    "prompt",
    "read_verdicts",
]

Verdict = Literal["stated", "contradicted", "missing", "unjudged"]
_GIVEN: Final = frozenset({"stated", "contradicted", "missing"})
TEMPLATE: Final = Path(__file__).resolve().parent / "templates" / "judge.md"
_OBJECT: Final = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class FactVerdict:
    fact: str
    verdict: Verdict
    why: str = ""


@dataclass(frozen=True)
class Judged:
    verdicts: tuple[FactVerdict, ...]
    tokens: int = 0
    cost_usd: float = 0.0


class Judge(Protocol):
    @property
    def model(self) -> str: ...

    def judge(self, question: ResearchQuestion, report: str) -> Judged: ...


def prompt(question: ResearchQuestion, report: str) -> str:
    facts = "\n".join(f"{n}. {fact}" for n, fact in enumerate(question.facts, 1))
    text = TEMPLATE.read_text(encoding="utf-8")
    return (
        text.replace("{{question}}", question.ask.strip())
        .replace("{{facts}}", facts)
        .replace("{{report}}", report.strip())
    )


class ModelJudge:
    """One call a report, on the catalog model `name`, through `engine`."""

    def __init__(self, engine: PacketEngine, name: str) -> None:
        self._engine = engine
        self._name = name

    @property
    def model(self) -> str:
        return self._name

    def judge(self, question: ResearchQuestion, report: str) -> Judged:
        asked = self._engine.ask_model(self._name, prompt(question, report))
        return Judged(read_verdicts(question, asked.text), asked.tokens, asked.cost_usd)


def read_verdicts(question: ResearchQuestion, answer: str) -> tuple[FactVerdict, ...]:
    """The judge's answer, one verdict a fact in the key's order: `unjudged` for a fact the
    answer does not judge in the shape asked."""
    given: dict[int, tuple[Verdict, str]] = {}
    found = _OBJECT.search(answer)
    try:
        data = cast("dict[str, Any]", json.loads(found.group(0))) if found else {}
        for item in cast("list[dict[str, Any]]", data.get("facts", [])):
            verdict = str(item.get("verdict", ""))
            if verdict in _GIVEN:
                given[int(item["n"])] = (cast("Verdict", verdict), str(item.get("why", "")))
    except (ValueError, TypeError, KeyError, AttributeError):
        given = {}
    return tuple(
        FactVerdict(fact, *given.get(n, ("unjudged", "the judge did not answer for it")))
        for n, fact in enumerate(question.facts, 1)
    )


def check_judge(host: HostConfig, name: str) -> None:
    """`--judge`'s model, checked before a run spends anything: a catalog model of the
    person's engine config, with a price row, and never the model the researcher's rung runs,
    which would grade its own answers. `ValueError` says which of those it is not."""
    config = host.engine
    if name not in config.models:
        raise ValueError(
            f"no model {name!r} in the engine config's catalog; it has "
            f"{', '.join(sorted(config.models))}"
        )
    if name == config.rungs[config.rung_name_for("researcher")]:
        raise ValueError(
            f"{name!r} is the model the researcher runs on, and a judge never grades its own "
            "model's answers: name another"
        )
    if name not in config.prices:
        raise ValueError(f"the judge {name!r} has no price row, so its spend cannot be said")
