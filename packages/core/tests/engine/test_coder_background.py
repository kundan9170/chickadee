"""`coder` in the background (as amended 2026-10-27): in an attended engine the call answers
once the run starts, the run goes on on its
own thread, and its end is a trigger-4 notice, carrying the report and its handle, that wakes
Zeus. The run's workspace gets its `after` when the run ends, under the turn that started it.
A `coder` call while the run works is answered at once, nothing asked or opened."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    Approving,
    CoderModel,
    athena_accepts,
    athena_plans_folder,
    build,
    make_repo,
    work_name,
    worker_edit,
    zeus_designs,
)
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

import chickadee.delegation as delegation_module
from chickadee.context.rewind import taken
from chickadee.context.session import SessionState
from chickadee.context.standing import Notice, NoticeTrigger
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError

TITLE = "Return 2"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def raw(raw: dict[str, Any]) -> dict[str, Any]:
    raw["workspace"] = {"mode": "worktree"}
    return raw


def _ended(engine: PacketEngine) -> Notice:
    """The run's end as Zeus is told it, once its watcher wrote it."""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        for t in state(engine, SID).turns():
            n = t.notice
            if n is not None and n.trigger is NoticeTrigger.RUN_ENDED and n.handle:
                return n
        time.sleep(0.05)
    raise AssertionError("the run's end was never told")


def _woken(engine: PacketEngine, said: str) -> SessionState:
    """The session once Zeus answered `said` in the turn the run's end woke."""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        now = state(engine, SID)
        if any(t.role == "assistant" and t.text == said for t in now.turns()):
            return now
        time.sleep(0.05)
    raise AssertionError(f"Zeus never answered {said!r} in a wake turn")


def _attended(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    zeus: list[Any],
    team: list[Any],
) -> PacketEngine:
    """An attended engine, as the server makes one, Zeus and the team each on a script of its
    own: they run at once, so one script would be raced for."""
    engine, _ = build(monkeypatch, repo, raw, zeus, interaction=Approving())
    monkeypatch.setattr(delegation_module, "client_for", lambda _endpoint: CoderModel(team))
    engine.attended = True
    return engine


def test_the_call_answers_at_once_and_the_run_s_end_wakes_zeus(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    team = [*athena_plans_folder(), *worker_edit("return 1", "return 2"), *athena_accepts()]
    engine = _attended(
        monkeypatch,
        repo,
        raw,
        [*zeus_designs(TITLE), answer("The team is on it."), answer("Done: go returns 2.")],
        team,
    )
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        said = state(engine, SID).result("c1")
        assert said is not None and said.text.startswith("RUNNING · run ")
        assert "mode FRESH" in said.text

        told = _ended(engine)
        assert told.text.startswith("COMPLETED · run "), "the report, as Zeus reads it"
        assert (
            told.carries is not None and f"{told.carries.kind}: {told.carries.id}" in told.text
        )
        now = _woken(engine, "Done: go returns 2.")
        assert now.handles.get(told.carries.id) == told.carries, "its handle is readable"

        # The run's changes are the run's: its workspace's pair, both under the turn that
        # started it, the `after` taken at the run's end.
        session = engine._conversations.get(SID).opened().session  # type: ignore[union-attr] # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]
        shots = [(s.turn, s.part, s.phase) for s in taken(session)]
        turn = shots[0][0]
        assert shots == [(turn, 0, "before"), (turn, 0, "after")]
    finally:
        engine.close()


def test_coder_while_the_team_works_is_answered_at_once(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    go = threading.Event()

    def waiting(request: object) -> Any:
        assert go.wait(WAIT_S), "the test never let the team go on"
        return athena_plans_folder()[0]

    team = [waiting, *athena_plans_folder()[1:], *worker_edit("return 1", "return 2")]
    zeus = [
        *zeus_designs(TITLE),
        calls(call("coder", "c2", work=work_name(TITLE))),
        answer("Waiting for the team."),
    ]
    engine = _attended(monkeypatch, repo, raw, zeus, [*team, *athena_accepts()])
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        again = state(engine, SID).result("c2")
        assert again is not None and again.text.startswith("INTERRUPTED{tool: coder")
        assert "is working; one run at a time" in again.text
        # The run's state beside Zeus's: working, never read as paused.
        started = state(engine, SID).result("c1")
        assert started is not None
        live = engine.liveness(SID)
        assert live.team_run == started.text.split()[3]
        assert live.paused_run is None
        # Refused before anything opened: no run, so no `episode` line naming one that never
        # exists. A screen following episodes left the working run for it on every refusal.
        assert (again.run_id, again.outcome) == (None, None)
        assert [e.run_id for e in state(engine, SID).episodes()] == [started.text.split()[3]], (
            "one episode: the run that works"
        )
    finally:
        go.set()
        engine.close()


def test_pause_suspends_the_run_and_coder_continues_it(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """`pause` is D215's suspend: no `run_end`, nothing lost, said in a notice that
    wakes nobody, since whoever paused knows."""

    def slow(request: object) -> Any:
        time.sleep(1.0)
        return athena_plans_folder()[0]

    team = [slow, *athena_plans_folder()[1:], *worker_edit("return 1", "return 2")]
    zeus = [*zeus_designs(TITLE), calls(call("pause", "p1")), answer("Paused it.")]
    engine = _attended(monkeypatch, repo, raw, zeus, [*team, *athena_accepts()])
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        paused = state(engine, SID).result("p1")
        assert paused is not None and " paused: nothing it did is lost" in paused.text
        run_id = paused.text.split()[1]
        told: list[Notice] = []
        deadline = time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            told = [
                t.notice
                for t in state(engine, SID).turns()
                if t.notice is not None and t.notice.trigger is NoticeTrigger.RUN_ENDED
            ]
            if told:
                break
            time.sleep(0.05)
        assert told and told[0].text.startswith(f"PAUSED · run {run_id}")
        assert told[0].handle is None, "its end is still to be told"
        from chickadee.journal.runs import RunProjections
        from chickadee.state.roots import roots as resolve_roots

        assert RunProjections(resolve_roots(repo)).readable(run_id).end is None, "suspended"
        live = engine.liveness(SID)
        assert (live.team_run, live.paused_run) == (None, run_id)
    finally:
        engine.close()


def test_pause_with_nothing_running_says_so(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine = _attended(
        monkeypatch, repo, raw, [calls(call("pause", "p1")), answer("Nothing ran.")], []
    )
    try:
        engine.turn(SID, "pause it").wait(timeout=WAIT_S)
        said = state(engine, SID).result("p1")
        assert said is not None and "nothing is running" in said.text
    finally:
        engine.close()


def test_zeus_s_shell_reads_but_may_not_write_while_the_team_works(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The team writes the workspace while its run is live; a command of Zeus's that
    may write is refused, naming `pause`, and a pure read still runs."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}

    def slow(request: object) -> Any:
        time.sleep(1.5)
        return athena_plans_folder()[0]

    team = [slow, *athena_plans_folder()[1:], *worker_edit("return 1", "return 2")]
    zeus = [
        *zeus_designs(TITLE),
        calls(call("shell", "s1", command="touch beside.txt")),
        calls(call("shell", "s2", command="ls")),
        calls(call("job", "j1", action="list")),
        answer("Looked around."),
    ]
    engine = _attended(monkeypatch, repo, raw, zeus, [*team, *athena_accepts()])
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        wrote = state(engine, SID).result("s1")
        assert wrote is not None and wrote.text.startswith("WRITE_DENIED")
        assert "pause the team first" in wrote.text
        read = state(engine, SID).result("s2")
        assert read is not None and not read.text.startswith("WRITE_DENIED")
        assert not (repo / "beside.txt").exists()
        listed = state(engine, SID).result("j1")
        assert listed is not None and " · run · working in the background" in listed.text
    finally:
        engine.close()


def test_another_session_s_run_is_never_this_one_s(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """One engine serves several sessions, one run at a time: another
    session's run is not this one's to pause, drop or list, or to name in a refusal of its
    shell, and its `coder` is refused saying whose the run is."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    go = threading.Event()

    def waiting(request: object) -> Any:
        assert go.wait(WAIT_S), "the test never let the team go on"
        return athena_plans_folder()[0]

    team = [waiting, *athena_plans_folder()[1:], *worker_edit("return 1", "return 2")]
    zeus = [
        *zeus_designs(TITLE),
        answer("The team is on it."),
        calls(call("pause", "p9")),
        calls(call("job", "j9", action="list")),
        calls(call("shell", "s9", command="touch beside.txt")),
        calls(call("coder", "c9", title="Other", brief="## Goal\nAnything.\n")),
        answer("Looked around."),
    ]
    engine = _attended(monkeypatch, repo, raw, zeus, [*team, *athena_accepts()])
    other = "S-other"
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        started = state(engine, SID).result("c1")
        assert started is not None and started.text.startswith("RUNNING · run ")
        run_id = started.text.split()[3]

        engine.create_session(other)
        engine.turn(other, "look around").wait(timeout=WAIT_S)
        paused = state(engine, other).result("p9")
        assert paused is not None and "nothing is running in this session" in paused.text
        listed = state(engine, other).result("j9")
        assert listed is not None and run_id not in listed.text
        wrote = state(engine, other).result("s9")
        # Refused by the repository's WRITE lease the run holds, never as this session's team.
        assert wrote is not None and "pause the team first" not in wrote.text
        refused = state(engine, other).result("c9")
        assert refused is not None and refused.text.startswith("INTERRUPTED{tool: coder")
        assert f"run {run_id} of another session is working" in refused.text

        with pytest.raises(SessionError, match="nothing is running in this session"):
            engine.pause_run(other)
        with pytest.raises(SessionError, match="no run of this session"):
            engine.drop_run(other)
        assert engine._episodes.busy(SID) == f"run {run_id}", "still working"  # pyright: ignore[reportPrivateUsage]
    finally:
        go.set()
        engine.close()


def test_job_refuses_the_team_s_run_naming_drop(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """The run sits beside the jobs in `job list` and is no job: `kill` on it is
    refused, naming `pause` and `/drop`, never answered as an unknown job, and `tail` shows its
    task board."""
    go = threading.Event()

    def waiting(request: object) -> Any:
        assert go.wait(WAIT_S), "the test never let the team go on"
        return athena_plans_folder()[0]

    def on_run(action: str, cid: str) -> Any:
        def step(request: object) -> Any:
            started = state(engine, SID).result("c1")
            assert started is not None
            return calls(call("job", cid, action=action, id=started.text.split()[3]))

        return step

    team = [waiting, *athena_plans_folder()[1:], *worker_edit("return 1", "return 2")]
    zeus = [*zeus_designs(TITLE), on_run("kill", "k1"), on_run("tail", "t1"), answer("Seen.")]
    engine = _attended(monkeypatch, repo, raw, zeus, [*team, *athena_accepts()])
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        killed = state(engine, SID).result("k1")
        assert killed is not None and killed.text.startswith("INVALID_ARGUMENTS")
        assert "the team's run, not a job" in killed.text and "/drop" in killed.text
        tailed = state(engine, SID).result("t1")
        assert tailed is not None and tailed.text.startswith("run ")
        assert " · working in the background · 0 task(s) delegated" in tailed.text
        assert engine._episodes.busy(SID) is not None, "still working"  # pyright: ignore[reportPrivateUsage]
    finally:
        go.set()
        engine.close()


def test_a_resumed_run_s_end_wakes_zeus(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """D240 §2.10, §2.11: `/resume` continues the paused run in the background, and its end is
    told and wakes Zeus as a `coder` run's does, not left for his next turn."""

    def slow(request: object) -> Any:
        time.sleep(1.0)
        return athena_plans_folder()[0]

    team = [slow, *athena_plans_folder(), *worker_edit("return 1", "return 2")]
    zeus = [
        *zeus_designs(TITLE),
        calls(call("pause", "p1")),
        answer("Paused it."),
        answer("Resumed and done."),
    ]
    engine = _attended(monkeypatch, repo, raw, zeus, [*team, *athena_accepts()])
    try:
        engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
        paused = state(engine, SID).result("p1")
        assert paused is not None and " paused: " in paused.text
        run_id = paused.text.split()[1]
        resumed = engine.resume(SID)
        assert resumed.in_flight is not None and resumed.in_flight.run_id == run_id
        told = _ended(engine)
        assert told.handle == run_id
        _woken(engine, "Resumed and done.")
    finally:
        engine.close()
