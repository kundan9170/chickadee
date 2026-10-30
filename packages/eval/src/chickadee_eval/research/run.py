"""One research run: every question, `repeats` times, each a researcher round of its own in
one session, scored and written as it ends.

The rounds run through the session's own agents (`PacketEngine.spawn_agent`), so a run measures
the wiring Zeus spawns, its limits and its web as the person configured them, and records each
round on the session's log as any round is. A row is written the moment its round is scored, so
a run stopped part way keeps every round it finished. A round the harness refused to start is a
row too, `refused`, with why: a set of twenty that quietly became nineteen answers the wrong
question.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chickadee.engine import InvalidArgument, PacketEngine
from chickadee_eval import archive
from chickadee_eval.arms.chickadee import commit_all
from chickadee_eval.research.judge import Judge
from chickadee_eval.research.questions import ResearchQuestion
from chickadee_eval.research.score import ResearchRound, score_round

__all__ = ["PROFILE", "REPORTS_NAME", "ResearchRun"]

PROFILE = "researcher"
REPORTS_NAME = "reports"
"""Each round's report as the researcher wrote it and the harness checked it, under its
question's id and repeat: what the judge read, kept for the person to read too."""


@dataclass
class ResearchRun:
    open_engine: Callable[[Path], PacketEngine]
    """The engine for the run's workspace: `open_host_engine` in a real run."""
    judge: Callable[[PacketEngine], Judge] | None
    """The judge, given the run's engine to ask its model through. None scores no facts:
    `--no-judge`, for a run that measures only the harness's checks."""
    repeats: int = 3
    say: Callable[[str], None] = lambda line: None
    clock: Callable[[], float] = time.monotonic
    attempts: list[ResearchRound] = field(default_factory=list[ResearchRound])

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")

    def run(
        self, questions: Sequence[ResearchQuestion], workspace: Path, out: Path
    ) -> list[ResearchRound]:
        """Every round, scored, each a line of the archive's results written as it ends."""
        reports = out / REPORTS_NAME
        reports.mkdir(parents=True, exist_ok=True)
        commit_all(workspace, "the research run's empty workspace")
        engine = self.open_engine(workspace)
        try:
            engine.init_repo()
            sid = engine.create_session(pinned=True).session_id
            judge = self.judge(engine) if self.judge is not None else None
            with (out / archive.RESULTS_NAME).open("a", encoding="utf-8") as results:
                for question in questions:
                    for n in range(1, self.repeats + 1):
                        done = self._round(engine, judge, sid, question, n, reports)
                        self.attempts.append(done)
                        results.write(json.dumps(done.as_json(), sort_keys=True) + "\n")
                        results.flush()
                        self.say(_line(done, len(question.facts)))
        finally:
            engine.close()
        return self.attempts

    def _round(
        self,
        engine: PacketEngine,
        judge: Judge | None,
        sid: str,
        question: ResearchQuestion,
        n: int,
        reports: Path,
    ) -> ResearchRound:
        started = self.clock()
        got = engine.spawn_agent(sid, PROFILE, question.ask, call_id=f"eval-{question.id}-{n}")
        seconds = self.clock() - started
        if isinstance(got, InvalidArgument):
            return ResearchRound(question.id, n, "refused", why=got.render(), seconds=seconds)
        text = got.render()
        (reports / f"{question.id}-{n}.md").write_text(text, encoding="utf-8")
        answered = got.status in ("complete", "partial")
        judged = judge.judge(question, text) if judge is not None and answered else None
        return score_round(question, n, got, seconds, judged)


def _line(done: ResearchRound, facts: int) -> str:
    stated = f"{done.stated()}/{facts} facts" if done.facts else "not judged"
    c = done.citations
    return (
        f"  {done.status:9} {done.question}#{done.repeat}  {stated}  "
        f"quotes {c.quoted}/{c.web}  ${done.cost_usd:.4f}  {done.seconds:.0f}s"
    )
