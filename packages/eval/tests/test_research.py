"""The researcher's evaluation: the question set, the judge, the scores and a
run. The run is driven with an engine double whose rounds are canned reports, so these tests
measure the scoring, never a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from engines import raw_config

from chickadee.agents.report import AgentReport
from chickadee.config import EngineConfig
from chickadee.context.citations import Checked, CitationGateResult, Verdict, WebCitation
from chickadee.context.results import InvalidArgument
from chickadee.engine import HostConfig, ModelAnswer, PacketEngine
from chickadee_eval import archive, cli
from chickadee_eval.research.judge import (
    FactVerdict,
    Judged,
    ModelJudge,
    check_judge,
    prompt,
    read_verdicts,
)
from chickadee_eval.research.questions import QuestionSet, ResearchQuestion, load_questions
from chickadee_eval.research.run import REPORTS_NAME, ResearchRun
from chickadee_eval.research.score import ResearchRound, compare, score_round, summarize

LIMIT = ResearchQuestion(
    id="limit",
    ask="What is the limit?",
    facts=("The limit is 40.", "It was raised in 2.0."),
    prefer=("example.org",),
)
STATED = FactVerdict("The limit is 40.", "stated")
MISSED = FactVerdict("The limit is 40.", "missing")


def web(url: str, verdict: Verdict) -> Checked:
    return Checked(
        WebCitation(url, "the limit is 40", f'{url} "the limit is 40"'), verdict, "page"
    )


def researched(*checked: Checked, status: str = "complete", claims: int = 2) -> AgentReport:
    return AgentReport(
        agent="A1",
        round=0,
        skill=None,
        profile="researcher",
        status=cast("Any", status),
        summary="The limit is 40.",
        checked=CitationGateResult(checked, claims=claims, cited_claims=len(checked)),
        tool_calls=4,
        tokens=1200,
        cost_usd=0.01,
    )


# -- the question set -------------------------------------------------------------------------


def test_a_set_is_read_from_toml_and_chosen_by_id_then_limit(tmp_path: Path) -> None:
    path = tmp_path / "q.toml"
    path.write_text(
        "version = 1\n"
        + "".join(f'[[questions]]\nid = "{q}"\nask = "a"\nfacts = ["f"]\n' for q in "abc"),
        encoding="utf-8",
    )
    chosen = load_questions(path)
    assert [q.id for q in chosen.chosen(None, 2)] == ["a", "b"]
    assert [q.id for q in chosen.chosen("c, a", None)] == ["a", "c"]
    with pytest.raises(ValueError, match="no question is named z"):
        chosen.chosen("a,z", None)


def test_a_set_refuses_a_repeated_id_and_a_question_with_no_facts() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        QuestionSet(version=1, questions=(LIMIT, LIMIT))
    with pytest.raises(ValueError):
        ResearchQuestion(id="x", ask="a", facts=())


# -- the judge --------------------------------------------------------------------------------


def test_the_prompt_numbers_the_facts_and_holds_the_report() -> None:
    text = prompt(LIMIT, "REPORT-BODY")
    assert "1. The limit is 40.\n2. It was raised in 2.0." in text
    assert "What is the limit?" in text and "REPORT-BODY" in text
    assert "{{" not in text


def test_verdicts_are_read_in_the_keys_order_and_a_skipped_fact_is_unjudged() -> None:
    said = 'Sure. {"facts": [{"n": 1, "verdict": "stated", "why": "says 40"}]} done'
    got = read_verdicts(LIMIT, said)
    assert [(f.fact, f.verdict) for f in got] == [
        ("The limit is 40.", "stated"),
        ("It was raised in 2.0.", "unjudged"),
    ]
    assert got[0].why == "says 40"


def test_an_answer_that_is_not_the_shape_asked_judges_nothing() -> None:
    for said in (
        "no json here",
        '{"facts": "all good"}',
        '{"facts": [{"n": 1, "verdict": "yes"}]}',
    ):
        assert {f.verdict for f in read_verdicts(LIMIT, said)} == {"unjudged"}


@dataclass(frozen=True)
class ResearchHost:
    """What `check_judge` reads of a `HostConfig`: its engine config."""

    engine: EngineConfig


def research_host(**prices: float) -> HostConfig:
    raw = raw_config()
    raw["prices"] = {m: {"price_in": p, "price_out": p} for m, p in prices.items()}
    return cast("HostConfig", ResearchHost(EngineConfig.with_defaults(raw)))


def test_the_judge_is_refused_for_its_own_model_an_unknown_one_and_an_unpriced_one() -> None:
    host = research_host()
    researcher = host.engine.rungs[host.engine.rung_name_for("researcher")]
    other = next(m for m in host.engine.models if m != researcher)
    with pytest.raises(ValueError, match="no model 'nope'"):
        check_judge(host, "nope")
    with pytest.raises(ValueError, match="never grades its own"):
        check_judge(research_host(**{researcher: 1.0}), researcher)
    with pytest.raises(ValueError, match="no price row"):
        check_judge(host, other)
    check_judge(research_host(**{other: 1.0}), other)


@dataclass
class AskingEngine:
    """The facade's `ask_model`, scripted: one canned answer, every prompt kept."""

    text: str
    asked: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def ask_model(self, model: str, prompt: str) -> ModelAnswer:
        self.asked.append((model, prompt))
        return ModelAnswer(self.text, tokens=110, cost_usd=0.25)


def test_the_judge_asks_its_model_once_and_keeps_what_it_cost() -> None:
    engine = AskingEngine('{"facts": [{"n": 2, "verdict": "missing"}]}')
    judge = ModelJudge(cast("PacketEngine", engine), "judge-model")
    judged = judge.judge(LIMIT, "REPORT")
    assert [f.verdict for f in judged.verdicts] == ["unjudged", "missing"]
    assert (judged.tokens, judged.cost_usd) == (110, 0.25)
    ((model, said),) = engine.asked
    assert model == "judge-model" and "REPORT" in said


# -- the scores -------------------------------------------------------------------------------


def test_a_round_is_scored_from_its_own_check_and_the_preferred_hosts() -> None:
    report = researched(
        web("https://docs.example.org/limits", Verdict.QUOTED),
        web("https://blog.other.net/post", Verdict.NOT_ON_PAGE),
        web("https://example.org/faq", Verdict.NOT_FETCHED),
    )
    judged = Judged((FactVerdict("The limit is 40.", "stated"),), tokens=50, cost_usd=0.002)
    done = score_round(LIMIT, 1, report, 12.34, judged)
    assert done.citations.web == 3
    assert (done.citations.quoted, done.citations.not_on_page, done.citations.not_fetched) == (
        1,
        1,
        1,
    )
    assert (done.sources.preferred, done.sources.hosts) == (
        2,
        ("docs.example.org", "blog.other.net", "example.org"),
    )
    assert (done.seconds, done.stated(), done.judge_cost_usd) == (12.3, 1, 0.002)
    assert ResearchRound.from_json(json.loads(json.dumps(done.as_json()))) == done


def test_a_host_that_only_ends_like_a_preferred_one_is_not_preferred() -> None:
    done = score_round(
        LIMIT, 1, researched(web("https://notexample.org/x", Verdict.QUOTED)), 1, None
    )
    assert done.sources.preferred == 0


def test_a_rate_over_nothing_is_said_as_none_never_zero() -> None:
    summary = summarize([ResearchRound("limit", 1, "failed", why="the model failed")])
    rendered = summary.render("R")
    assert "| facts stated (judge) | none |" in rendered
    assert "| citation accuracy: quotes on the page | none |" in rendered
    assert "1 rounds: failed 1." in rendered


def test_a_summary_counts_every_round_and_each_questions_spread() -> None:
    stated = FactVerdict("a", "stated")
    missing = FactVerdict("b", "missing")
    rounds = [
        ResearchRound("limit", 1, "complete", facts=(stated, stated)),
        ResearchRound("limit", 2, "complete", facts=(stated, missing)),
        ResearchRound("other", 1, "refused", why="no such profile"),
    ]
    summary = summarize(rounds)
    assert (summary.facts.hits, summary.facts.of) == (3, 4)
    assert summary.statuses == {"complete": 2, "refused": 1}
    row = next(r for r in summary.rows if r.question == "limit")
    assert (row.attempts, row.spread) == (2, (1, 2))


def test_compare_names_each_question_that_moved_and_each_asked_once() -> None:
    stated, missing = FactVerdict("a", "stated"), FactVerdict("a", "missing")
    before = summarize(
        [
            ResearchRound("limit", 1, "complete", facts=(missing,)),
            ResearchRound("old", 1, "complete"),
        ]
    )
    after = summarize(
        [
            ResearchRound("limit", 1, "complete", facts=(stated,)),
            ResearchRound("new", 1, "complete"),
        ]
    )
    text = compare(before, after)
    assert "| limit | 0% (0/1) | 100% (1/1) |" in text
    assert "Asked in one run only, so not compared: new, old." in text


# -- a run ------------------------------------------------------------------------------------


@dataclass
class ResearchSession:
    session_id: str


@dataclass
class ResearchEngine:
    """The engine a run drives, its rounds canned: one report or refusal per call, in order."""

    rounds: list[AgentReport | InvalidArgument]
    calls: list[tuple[str, str, str, str]] = field(
        default_factory=list[tuple[str, str, str, str]]
    )
    closed: bool = False

    def init_repo(self) -> None:
        """A run initializes the workspace it was given; there is none here."""

    def create_session(self, *, pinned: bool) -> ResearchSession:
        assert pinned
        return ResearchSession("S-research")

    def spawn_agent(
        self, session_id: str, profile: str, task: str, *, call_id: str
    ) -> AgentReport | InvalidArgument:
        self.calls.append((session_id, profile, task, call_id))
        return self.rounds.pop(0)

    def close(self) -> None:
        self.closed = True


class KeyJudge:
    """States the first fact of the key, misses the rest."""

    model = "judge-model"

    def __init__(self) -> None:
        self.read: list[str] = []

    def judge(self, question: ResearchQuestion, report: str) -> Judged:
        self.read.append(report)
        first, *rest = question.facts
        verdicts = (FactVerdict(first, "stated"), *(FactVerdict(f, "missing") for f in rest))
        return Judged(verdicts, tokens=10, cost_usd=0.001)


def test_a_run_scores_each_round_as_it_ends_and_keeps_each_report(tmp_path: Path) -> None:
    engine = ResearchEngine(
        [
            researched(web("https://docs.example.org/limits", Verdict.QUOTED)),
            researched(status="failed"),
            InvalidArgument("spawn_agent", "the researcher has no price row"),
        ]
    )
    judge = KeyJudge()
    out = archive.create("2026-10-25", {"label": "t"}, root=tmp_path / "home")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    said: list[str] = []
    run = ResearchRun(
        open_engine=lambda _path: cast("PacketEngine", engine),
        judge=lambda _engine: judge,
        repeats=3,
        say=said.append,
    )
    done = run.run([LIMIT], workspace, out)
    assert [a.status for a in done] == ["complete", "failed", "refused"]
    assert [c[3] for c in engine.calls] == ["eval-limit-1", "eval-limit-2", "eval-limit-3"]
    assert {c[:3] for c in engine.calls} == {("S-research", "researcher", "What is the limit?")}
    assert engine.closed
    # Only an answered round is judged; a failed one has nothing to judge.
    assert len(judge.read) == 1 and done[0].stated() == 1 and not done[1].facts
    assert "no price row" in done[2].why
    lines = (out / archive.RESULTS_NAME).read_text(encoding="utf-8").splitlines()
    assert [ResearchRound.from_json(json.loads(line)) for line in lines] == done
    reports = sorted(p.name for p in (out / REPORTS_NAME).iterdir())
    assert reports == ["limit-1.md", "limit-2.md"]
    assert len(said) == 3


def test_a_run_needs_a_repeat() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        ResearchRun(open_engine=lambda _path: cast("PacketEngine", None), judge=None, repeats=0)


# -- the CLI -----------------------------------------------------------------------------------


def research_archive(home: Path, label: str, rounds: list[ResearchRound]) -> Path:
    out = archive.create("2026-10-25", {"kind": cli.RESEARCH_KIND, "label": label}, root=home)
    (out / archive.RESULTS_NAME).write_text(
        "".join(json.dumps(a.as_json()) + "\n" for a in rounds), encoding="utf-8"
    )
    return out


def test_list_and_diff_read_a_research_archive_as_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    research_archive(
        tmp_path, "before", [ResearchRound("limit", 1, "complete", facts=(MISSED,))]
    )
    research_archive(
        tmp_path, "after", [ResearchRound("limit", 1, "complete", facts=(STATED,))]
    )
    assert cli.main(["--results-root", str(tmp_path), "list"]) == 0
    assert "research" in capsys.readouterr().out
    assert cli.main(["--results-root", str(tmp_path), "diff", "before", "after"]) == 0
    assert "| limit | 0% (0/1) | 100% (1/1) |" in capsys.readouterr().out


def test_diff_refuses_a_research_archive_beside_another_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    research_archive(tmp_path, "before", [ResearchRound("limit", 1, "complete")])
    archive.create("2026-10-25", {"label": "after", "subset": "e1-smoke"}, root=tmp_path)
    assert cli.main(["--results-root", str(tmp_path), "diff", "before", "after"]) == 1
    assert "diff two of a kind" in capsys.readouterr().err


def test_a_question_set_that_cannot_be_read_names_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "none.toml"
    argv = ["--results-root", str(tmp_path), "research", "--questions-file", str(missing)]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert str(missing) in err and "--questions-file" in err
