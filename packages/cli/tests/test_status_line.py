"""The REPL's status line, from recorded frames and `/budget` alone:
Zeus's model, the run's stage, spend against the cap, and his context, amber past 80% and
`compacting…` while a cut runs. No engine is read."""

from __future__ import annotations

from chickadee_cli.render.status import StatusLine
from chickadee_cli.render.style import Style
from chickadee_serve.frames import ErrorBody, ErrorCode, JsonValue, ProgressFrame, RecordFrame
from chickadee_wire.errors import WireError
from chickadee_wire.models import JobView, SessionBudget

PLAIN = Style(tty=False)
AMBER = "\x1b[33m"


def _budget(spent: float, cap: float | None) -> SessionBudget:
    return SessionBudget.model_validate(
        {"cap_seq": None, "cap_source": "config", "cap_usd": cap, "spent_usd": spent}
    )


def _record(
    kind: str,
    payload: dict[str, JsonValue],
    *,
    run: str | None = None,
    task: str | None = None,
    seq: int = 1,
) -> RecordFrame:
    return RecordFrame(
        scope=run or "S", seq=seq, at="t", kind=kind, payload=payload, run_id=run, task_id=task
    )


def _context(used: int, window: int, *, compacting: bool = False) -> ProgressFrame:
    return ProgressFrame(
        scope="S",
        kind="context",
        payload={"used": used, "window": window, "model": "qwen3", "compacting": compacting},
    )


def test_model_stage_spend_and_context() -> None:
    """Zeus's model comes from his context frame: his rung may be named anything (`high-param`
    here), so a spend record never says which model is his."""
    status = StatusLine(PLAIN, lambda: _budget(0.72, 5.0))
    assert status.line() == ""
    spend = {
        "cache_writes": 0,
        "cached_in": 0,
        "cost_usd": 0.01,
        "estimated": False,
        "model": "qwen3",
        "rung": "high-param",
        "tokens_in": 10,
        "tokens_out": 2,
    }
    status.record(_record("spend", spend))
    status.record(
        _record(
            "episode",
            {"run_id": "R-1", "path": "planned"},
        )
    )
    status.record(
        _record(
            "task_delegated",
            {
                "brief_sha": "b",
                "by": "athena",
                "start_tree": "t",
                "task_id": "T1",
                "title": "add retry",
            },
            run="R-1",
            task="T1",
            seq=2,
        )
    )
    status.progress(_context(53_000, 128_000))
    line = status.line()
    assert line.startswith("qwen3 · WORK 0/1 · $0.72 / $5.00 · context 41% (75k left)"), line


def test_context_is_amber_past_80_and_says_compacting_during_a_cut() -> None:
    status = StatusLine(Style(tty=True), lambda: _budget(0.0, None))
    status.progress(_context(110_000, 128_000))
    assert AMBER in status.line()
    status.progress(_context(110_000, 128_000, compacting=True))
    assert "compacting…" in status.line()
    status.progress(_context(40_000, 128_000))
    assert "compacting…" not in status.line() and AMBER not in status.line()


def test_a_server_that_cannot_say_leaves_the_last_spend() -> None:
    answers: list[SessionBudget | WireError] = [_budget(1.0, 2.0)]

    def budget() -> SessionBudget:
        answer = answers[0]
        if isinstance(answer, WireError):
            raise answer
        return answer

    status = StatusLine(PLAIN, budget)
    status.refresh(now=True)
    answers[0] = WireError(503, ErrorBody(code=ErrorCode.NOT_FOUND, message="gone"))
    status.refresh(now=True)
    assert "$1.00 / $2.00" in status.line()


def _job(job_id: str, *, running: bool) -> JobView:
    """A job as `/jobs` lists it, with only what the status line reads."""
    return JobView.model_construct(job_id=job_id, running=running)


def test_running_jobs_are_counted_from_the_route_after_a_turn_and_on_an_exit() -> None:
    """T-3.161's handoff: no frame says a job started, so the count is asked of
    `/jobs` as the REPL opens, after each turn and on each job's exit turn, and held between."""
    listed = [_job("j1", running=True), _job("j2", running=True), _job("j3", running=False)]
    asked: list[int] = []

    def jobs() -> list[JobView]:
        asked.append(1)
        return list(listed)

    status = StatusLine(PLAIN, lambda: _budget(0.0, None), jobs)
    status.recount()
    assert status.line() == "2 jobs"
    listed[0] = _job("j1", running=False)
    status.progress(ProgressFrame(scope="S", kind="thinking", payload={"seq": 1}))
    assert status.line().endswith("2 jobs"), "held between turns"
    status.progress(ProgressFrame(scope="S", kind="answer", payload={"seq": 2}))
    assert status.line().endswith("1 job")
    listed[1] = _job("j2", running=False)
    status.record(_record("turn", {"role": "harness", "text": "job j2 exited 0", "trigger": 8}))
    assert "job" not in status.line(), "none running is left out"
    assert len(asked) == 3


def test_a_server_that_cannot_list_jobs_leaves_the_last_count() -> None:
    answers: list[list[JobView] | WireError] = [[_job("j1", running=True)]]

    def jobs() -> list[JobView]:
        answer = answers[0]
        if isinstance(answer, WireError):
            raise answer
        return answer

    status = StatusLine(PLAIN, lambda: _budget(0.0, None), jobs)
    status.recount()
    answers[0] = WireError(503, ErrorBody(code=ErrorCode.NOT_FOUND, message="gone"))
    status.recount()
    assert status.line().endswith("1 job")
