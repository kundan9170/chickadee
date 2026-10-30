"""The question set (`manifests/research.toml`): what is asked, and what a right answer holds.

A question carries its **answer key**, the facts a right answer states, each a short claim a
judge can find or not find in a report; and the **domains** a good answer cites, the project's
own pages first. Both are a person's, written once and reviewed, never a model's: a key a model
wrote measures that model's agreement with itself.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["QUESTIONS", "QuestionSet", "ResearchQuestion", "load_questions"]

WIRE = ConfigDict(extra="forbid", frozen=True)

QUESTIONS = Path(__file__).resolve().parents[3] / "manifests" / "research.toml"


class ResearchQuestion(BaseModel):
    model_config = WIRE

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    ask: str = Field(min_length=1)
    """The task as Zeus would hand it to a researcher."""
    facts: tuple[str, ...] = Field(min_length=1)
    """What a right answer states, one checkable claim each."""
    prefer: tuple[str, ...] = ()
    """Hosts a good answer cites (`docs.python.org`); a subdomain counts as its host."""
    note: str = ""
    """Why the question is in the set, and what makes it hard: for the person reviewing it."""


class QuestionSet(BaseModel):
    model_config = WIRE

    version: int = Field(ge=1)
    """Raised whenever a question or a fact changes, so two runs on different keys are told
    apart: it is part of each archive's identity."""
    questions: tuple[ResearchQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> QuestionSet:
        seen: set[str] = set()
        for q in self.questions:
            if q.id in seen:
                raise ValueError(f"question id {q.id!r} appears twice")
            seen.add(q.id)
        return self

    def chosen(self, only: str | None, limit: int | None) -> tuple[ResearchQuestion, ...]:
        """`--questions` by id, then `--limit`, in the set's order."""
        picked = self.questions
        if only:
            wanted = {w.strip() for w in only.split(",") if w.strip()}
            unknown = sorted(wanted - {q.id for q in picked})
            if unknown:
                raise ValueError(f"no question is named {', '.join(unknown)}")
            picked = tuple(q for q in picked if q.id in wanted)
        return picked[:limit] if limit else picked


def load_questions(path: Path = QUESTIONS) -> QuestionSet:
    return QuestionSet(**tomllib.loads(path.read_text(encoding="utf-8")))
