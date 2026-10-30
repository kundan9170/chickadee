"""The TUI's feed: the wire folded into what each pane shows.

What these pin: the recorded run reads as its board, Athena's tail, the worker's stream and
facts, and the code pane; the board is `fold_run`'s, one record at a time; the spend is the
engine's and unknown until it says; an undo asks `/hidden` again and hides what it names;
words sent are pending until their turn lands; progress moves only `busy`, and dropping every
progress frame changes nothing else; a record seen twice changes nothing; one that cannot be
read is said in its pane; and a facts block never says more than ran.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from tui_frames import (
    RUN_ID,
    SID,
    answered,
    assistant,
    episode,
    progress,
    run_frames,
    run_record,
    session,
    spend,
    stop_asked,
    user,
)

from chickadee_cli.render.board import fold_run
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import (
    AthenaKind,
    Pane,
    PlanState,
    SpendLevel,
    Stage,
    StaleRoute,
    WorkerKind,
    ZeusKind,
)
from chickadee_serve.frames import Hidden, JsonValue, RecordFrame
from chickadee_wire.models import (
    Liveness,
    ModelWiring,
    RungBudget,
    SessionBudget,
    TaskFactsP,
    WorkProgress,
)


def _fed(frames: list[RecordFrame]) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in frames:
        feed.apply(frame)
    return feed


# -- the recorded run ------------------------------------------------------------------------


def test_the_recorded_run_reads_as_its_board_and_its_tail() -> None:
    feed = _fed(run_frames())
    athena = feed.state.athena
    assert athena.run_id == RUN_ID
    assert (athena.plan_version, athena.plan_state) == (1, PlanState.APPROVED)
    assert athena.board is not None
    assert [(t.task_id, t.state, t.rounds) for t in athena.board.tasks.values()] == [
        ("T1", "committed", 2),
        ("T2", "dropped", 1),
    ]
    assert athena.board.end is not None and athena.board.end.outcome == "PARTIAL"
    assert athena.cost_usd == pytest.approx(7 * 0.0123)
    reviews = [e.record for e in athena.tail if e.kind is AthenaKind.REVIEW]
    assert [getattr(r, "answer", None) for r in reviews] == ["feedback", "accept", "abandon"]


def test_the_run_the_team_works_on_is_shown_from_its_first_record() -> None:
    """A run Zeus's `coder` resumed goes on under its first id, and that call's own
    `episode` names a run that never starts. Liveness names the run the team works on: the
    screen shows it, from its first record, which the reader follows it from again."""
    frames = run_frames()
    feed = _fed(frames[:4])  # the run, read up to its pause
    feed.apply(episode(9, "2026-10-29-0000beef"))  # the resuming call's: no run behind it
    assert feed.state.athena.run_id == "2026-10-29-0000beef"
    feed.snapshot(Liveness(kind="working", last_seq=9, team_run=RUN_ID))
    assert feed.state.athena.run_id == RUN_ID
    for frame in frames:  # followed again from its first record
        feed.apply(frame)
    assert feed.state.athena.board == fold_run(RUN_ID, frames)
    board = feed.state.athena.board
    feed.snapshot(Liveness(kind="idle", last_seq=10, team_run=RUN_ID))
    assert feed.state.athena.board is board, "the run shown: nothing starts again"
    feed.snapshot(Liveness(kind="idle", last_seq=11))
    assert feed.state.athena.run_id == RUN_ID, "no team run: the screen keeps what it shows"


def _in_place(frames: list[RecordFrame]) -> list[RecordFrame]:
    """The recorded run as it is written in place: each accepted task's commit is its
    tree recorded as known good instead."""
    out: list[RecordFrame] = []
    for frame in frames:
        if (
            frame.kind == "effect_intended"
            and frame.payload.get("effect_kind") == "task_commit"
        ):
            frame = frame.model_copy(
                update={"payload": {**frame.payload, "effect_kind": "task_known_good"}}
            )
        out.append(frame)
    return out


def test_a_task_accepted_in_place_is_kept_and_counts_as_done() -> None:
    """In place nothing is committed, and the header's `WORK n/m` counted only a
    commit, so it read 0 for a run that had done its tasks. A known-good record settles the
    task as `kept`, never `committed`."""
    feed = Feed(SID)
    feed.apply(episode(1))
    counts: list[int] = []
    for frame in _in_place(run_frames()):
        feed.apply(frame)
        counts.append(feed.state.header.work_done)
    board = feed.state.athena.board
    assert board is not None
    assert [(t.task_id, t.state, t.committed) for t in board.tasks.values()] == [
        ("T1", "kept", ()),
        ("T2", "dropped", ()),
    ]
    assert counts[-1] == 2 and sorted(set(counts)) == [0, 1, 2], "it moves as each settles"
    worktree = _fed(run_frames()).state.athena.board
    assert worktree is not None and worktree.tasks["T1"].state == "committed", "a commit still"


EARLIER = "2026-10-09-00000001"


def _work(*tasks: tuple[str, str, str | None]) -> WorkProgress:
    """The engine's fold of the work (`/session/{sid}/work`): each task, its state, its run."""
    part = {"version": 1, "approved": "t", "current": "t"}
    return WorkProgress.model_validate(
        {
            "key": {"dir": "docs", "name": "w"}, "design": part, "plan": part,
            "plan_run": EARLIER, "plan_sha": "p", "plan_version": 1, "run": None,
            "runs": [EARLIER, RUN_ID],
            "tasks": [
                {"id": i, "title": f"task {i}", "state": st, "run": run} for i, st, run in tasks
            ],
        }
    )  # fmt: skip


def test_a_continued_work_shows_the_tasks_its_earlier_runs_did_as_done() -> None:
    """A run that continues a work inherits its plan, and its journal names none of
    the tasks the earlier runs settled; the work's fold, which the engine answers, does. Each
    is drawn done, said as an earlier run's, and counted; a task still to do waits."""
    from chickadee_cli.tui.feed import Work
    from chickadee_cli.tui.panes.athena import athena_body, waiting
    from chickadee_cli.tui.theme import UNICODE, palette

    feed = Feed(SID)
    changed = feed.apply(episode(1))
    assert StaleRoute.WORK in changed.refresh, "read when the run is shown"
    plan = {
        "asked": False, "by": "athena", "sha": "p", "version": 1, "inherited_from": EARLIER,
        "tasks": [{"id": "T1", "title": "task T1"}, {"id": "T2", "title": "task T2"},
                  {"id": "T3", "title": "task T3"}],
    }  # fmt: skip
    assert StaleRoute.WORK in feed.apply(run_record("plan_written", plan, 1)).refresh
    feed.snapshot(
        Work(
            _work(
                ("T1", "committed", EARLIER),
                ("T2", "unchecked", EARLIER),
                ("T3", "lost", EARLIER),
            )
        )
    )
    athena = feed.state.athena
    assert [(t.task_id, t.state) for t in athena.carried] == [
        ("T1", "committed"),
        ("T2", "kept"),
    ]
    assert [p.id for p in waiting(feed.state)] == ["T3"], "a lost commit is to do again"
    header = feed.state.header
    assert (header.work_done, header.work_delegated, header.work_planned) == (2, 2, 3)
    lines = athena_body(feed.state, UNICODE, palette("truecolor")).plain.splitlines()
    assert "✔ T1 task T1" in lines and "  committed in an earlier run" in lines
    assert "  kept in an earlier run" in lines and "○ T3 task T3" in lines
    feed.snapshot(Work(None))
    assert athena.carried == (), "no open work: nothing carried"


def test_a_task_this_run_settles_is_its_own_never_carried() -> None:
    from chickadee_cli.tui.feed import Work

    feed = _fed(run_frames())
    feed.snapshot(Work(_work(("T1", "committed", RUN_ID), ("T9", "committed", RUN_ID))))
    assert feed.state.athena.carried == (), "this run's: its board says so"
    feed.snapshot(Work(_work(("T1", "committed", EARLIER))))
    assert feed.state.athena.carried == (), "on this run's board: its board says so"


def test_the_board_s_words_for_a_carried_task_are_each_state_the_work_calls_done() -> None:
    """One reading of "done": the work's fold's `DONE`, each in the board's words."""
    from chickadee.work_progress import DONE
    from chickadee_cli.render.board import STATES
    from chickadee_cli.tui.feed import CARRIED_WORDS

    assert set(CARRIED_WORDS) == DONE
    assert set(CARRIED_WORDS.values()) <= STATES


def test_the_header_s_tree_is_read_again_when_a_record_says_it_may_have_moved() -> None:
    """The header's uncommitted paths were read when the screen opened and never
    again. Each record that can move the tree asks for them again; one that cannot does not."""
    stale = StaleRoute.HEADER
    feed = Feed(SID)
    assert stale in feed.apply(assistant("done", 1)).refresh, "his turn ended"
    called = session(
        "turn",
        {"role": "assistant", "text": "", "calls": [{"id": "c1", "name": "read", "args": {}}]},
        2,
    )
    assert stale not in feed.apply(called).refresh, "a turn that goes on: not yet"
    moved: dict[int, bool] = {}
    whole = Feed(SID)
    whole.apply(episode(1))
    for frame in run_frames():
        moved[frame.seq] = stale in whole.apply(frame).refresh
    kinds = {f.seq: f.kind for f in run_frames()}
    asked = {kinds[seq] for seq, yes in moved.items() if yes}
    assert {"task_reviewed", "run_end"} <= asked
    assert "effect_completed" in asked, "a step that changed files, and a task's commit"
    assert "model_call" not in asked and "tool_call" not in asked


def test_a_route_that_answers_again_is_no_longer_said_lost() -> None:
    """A failed read stayed red beside where the session runs for ever."""
    from chickadee_cli.tui.feed import Recovered
    from chickadee_cli.tui.header import lost_line

    feed = Feed(SID)
    feed.lost("/budget", "GET /session/S-tui/budget: no answer came within 30 s", stream=False)
    assert "lost /budget" in lost_line(feed.state)
    feed.snapshot(Recovered("/hidden"))
    assert "lost /budget" in lost_line(feed.state), "another route's answer changes nothing"
    feed.snapshot(Recovered("/budget"))
    assert lost_line(feed.state) == ""


def test_the_board_is_fold_run_s_one_record_at_a_time() -> None:
    frames = run_frames()
    board = _fed(frames).state.athena.board
    assert board == fold_run(RUN_ID, frames)


def test_the_worker_shows_each_task_s_calls_steps_and_commands_in_order() -> None:
    worker = _fed(run_frames()).state.worker
    assert worker.current == "T2"
    t1 = worker.tasks["T1"]
    assert (t1.title, t1.round, t1.delegated_at) == (
        "Raise the retry limit to 5",
        2,
        "2026-10-10T10:00:25Z",
    )
    assert t1.cost_usd == pytest.approx(6 * 0.0004)
    assert [(e.kind, e.step) for e in t1.entries] == [
        (WorkerKind.STEP, 1),
        (WorkerKind.CALL, None),
        (WorkerKind.EXEC, None),
        (WorkerKind.STEP, 2),
        (WorkerKind.CALL, None),
        (WorkerKind.STEP, 3),
        (WorkerKind.CALL, None),
        (WorkerKind.EXEC, None),
        (WorkerKind.STEP, 4),
        (WorkerKind.CALL, None),
    ]
    assert [e.step for e in worker.tasks["T2"].entries if e.step] == [5]
    assert worker.end is not None and worker.end.outcome == "PARTIAL"


def test_the_facts_are_the_latest_round_s_and_the_code_pane_follows_them() -> None:
    frames = run_frames()
    round_1 = _fed([f for f in frames if f.seq <= 19]).state
    facts = round_1.worker.tasks["T1"].facts_line
    assert facts is not None
    assert (facts.files, facts.added, facts.removed, facts.tests) == (1, 1, 1, "failed")
    assert facts.failing == ("tests/test_client.py::test_retries",)
    assert facts.syntax_ok is None, "no file's syntax was checked: not 'ok'"
    code = round_1.task_diff
    assert [(f.path, f.added, f.removed) for f in code.files] == [("src/client.py", 1, 1)]
    assert code.failing is not None and code.failing.command == "pytest -q"
    assert code.steps == (1, 2)
    assert (code.task_id, code.lines) == ("T1", (1, 1))

    round_2 = _fed([f for f in frames if f.seq <= 35]).state
    facts = round_2.worker.tasks["T1"].facts_line
    assert facts is not None and (facts.files, facts.tests, facts.failing) == (2, "passed", ())
    assert round_2.task_diff.failing is None


def test_before_a_report_the_code_pane_names_the_files_the_steps_touched() -> None:
    state = _fed([f for f in run_frames() if f.seq <= 12]).state
    assert [(f.path, f.added) for f in state.task_diff.files] == [("src/client.py", None)]


# -- verification is never overstated --------------------------------------------------------


def test_a_test_run_that_collected_nothing_never_reads_as_a_pass() -> None:
    feed = _fed([f for f in run_frames() if f.seq <= 6])
    facts = TaskFactsP.model_validate(
        {
            "start_tree": "a", "end_tree": "b",
            "files": [{"path": "x.py", "added": 1, "removed": 0, "kind": "edited"}],
            "commands": [{
                "command": "pytest -k nothing", "operation_id": "op", "tail": "no tests ran",
                "timed_out": False, "exit_code": 5, "test_outcome": "not_collected",
            }],
            "syntax": [{"path": "x.py", "ok": False, "detail": "line 3"}],
        }
    )  # fmt: skip
    payload = {"task_id": "T1", "round": 1, "stopped": "replied", "facts": facts.model_dump()}
    feed.apply(run_record("task_reported", payload, 100, task_id="T1"))
    line = feed.state.worker.tasks["T1"].facts_line
    assert line is not None
    assert (line.tests, line.failing, line.syntax_ok) == ("not_collected", (), False)


# -- what the wire does not carry yet stays unknown ------------------------------------------


def test_the_placeholders_stay_unknown_whatever_the_records_say() -> None:
    # W1, W2, W3 in the: each is shown as not known until the wire carries it,
    # never filled in from what is near it.
    feed = _fed([])
    feed.snapshot(
        ModelWiring(catalog=["big", "small"], rungs={"high-param": "big"}, switches=[])
    )
    seen: set[StaleRoute] = set()
    stages: set[Stage] = set()
    for frame in run_frames():
        seen |= feed.apply(frame).refresh
        stages.add(feed.state.header.stage)
        assert feed.state.header.work_planned is None, "the plan's size is not on the wire"
    assert Stage.CHECK not in stages, "no record says the last task is done"
    assert StaleRoute.RUNG not in seen, "no record says a model was switched"
    call: dict[str, JsonValue] = {
        "role": "athena", "rung": "high-param", "cost_usd": 0.01, "request": "r",
        "stop_reason": "end_turn", "tokens_in": 1, "tokens_out": 1,
    }  # fmt: skip
    feed.apply(run_record("model_call", call, 100))
    athena = feed.state.athena
    assert (athena.rung, athena.model) == ("high-param", None), "the wiring is not who served"
    assert feed.state.worker.tasks["T1"].model is None


# -- spend -----------------------------------------------------------------------------------


def test_spend_is_unknown_until_the_engine_says_and_then_is_its_answer() -> None:
    feed = _fed(run_frames())
    spend_ = feed.state.header.spend
    assert (spend_.total_usd, spend_.cap_usd, spend_.lifted) == (None, None, False)
    feed.snapshot(SessionBudget(spent_usd=0.31, cap_usd=5.0, cap_source="config", cap_seq=None))
    assert (spend_.total_usd, spend_.cap_usd) == (0.31, 5.0)
    feed.snapshot(SessionBudget(spent_usd=0.31, cap_usd=None, cap_source="session", cap_seq=4))
    assert spend_.cap_usd is None and spend_.lifted, "D144: a lifted cap, not an unknown one"


def test_zeus_s_spend_and_served_model_come_from_his_spend_records() -> None:
    """His pane's spend is every session charge; the model that served him is his own line's,
    one naming his role (wire 4.9)."""
    feed = Feed(SID)
    changed = feed.apply(spend(0.2, 1)) | feed.apply(spend(0.11, 2, model="other"))
    assert feed.state.zeus.cost_usd == pytest.approx(0.31)
    assert feed.state.role_model("zeus") is None, "lines naming no role: not his to name"
    assert StaleRoute.BUDGET in changed.refresh


def test_a_budget_record_raises_the_level_and_never_lowers_it() -> None:
    feed = _fed([])
    budget = {"cap_usd": 5.0, "scope": "run", "spent_usd": 4.0, "terminal": False}
    feed.apply(run_record("budget", {**budget, "kind": "warning"}, 1))
    assert feed.state.header.spend.level is SpendLevel.WARNING
    feed.apply(run_record("budget", {**budget, "kind": "hard_cap", "terminal": True}, 2))
    feed.apply(run_record("budget", {**budget, "kind": "warning"}, 3))
    assert feed.state.header.spend.level is SpendLevel.CAP


def test_the_engine_s_budget_answer_sets_the_level_so_a_raised_cap_clears_the_red() -> None:
    """Records only raise the level; `/budget`'s answer is the level now. A cap raised past the
    spend, or lifted, takes the header out of red and its banner away."""
    feed = _fed([])
    hard = {"cap_usd": 1.0, "scope": "run", "spent_usd": 1.0, "kind": "hard_cap"}
    feed.apply(run_record("budget", {**hard, "terminal": True}, 1))
    spend_ = feed.state.header.spend
    at_cap = SessionBudget(
        spent_usd=1.0, cap_usd=1.0, cap_source="config", cap_seq=None, warning_fractions=[0.8]
    )
    feed.snapshot(at_cap)
    assert spend_.level is SpendLevel.CAP and spend_.capped is not None, "still at its cap"
    feed.snapshot(at_cap.model_copy(update={"cap_usd": 1.2, "cap_source": "session"}))
    assert spend_.level is SpendLevel.WARNING and spend_.capped is None, "raised: amber"
    feed.snapshot(at_cap.model_copy(update={"cap_usd": None, "cap_source": "session"}))
    assert spend_.level is SpendLevel.OK and spend_.capped is None, "lifted"
    over_rung = at_cap.model_copy(
        update={
            "cap_usd": 5.0,
            "rungs": {
                "frontier": RungBudget(
                    spent_usd=0.5, cap_usd=0.5, cap_source="session", cap_seq=3
                )
            },
        }
    )
    feed.snapshot(over_rung)
    assert spend_.level is SpendLevel.CAP, "a rung at its own cap stops its calls"


# -- Zeus's transcript -----------------------------------------------------------------------


def test_an_undo_asks_hidden_again_and_hides_only_what_it_names() -> None:
    feed = Feed(SID)
    feed.apply(user("one", 1))
    feed.apply(assistant("done", 2))
    changed = feed.apply(session("undo", {"to_turn": 1, "code": True, "conversation": True}, 3))
    assert StaleRoute.HIDDEN in changed.refresh
    feed.snapshot(Hidden(seqs=(1, 2)))
    zeus = feed.state.zeus
    assert [e.seq for e in zeus.visible()] == [3]
    assert [e.seq for e in zeus.entries] == [1, 2, 3], "hidden, never deleted"


def test_words_sent_are_pending_until_their_turn_lands() -> None:
    feed = Feed(SID)
    feed.sent("also update the CHANGELOG")
    assert feed.state.zeus.pending == ["also update the CHANGELOG"]
    feed.apply(user("also update the CHANGELOG", 1))
    zeus = feed.state.zeus
    assert zeus.pending == []
    assert [e.kind for e in zeus.entries] == [ZeusKind.YOU]


def test_a_stop_s_answer_is_a_marker_that_names_the_stop() -> None:
    feed = Feed(SID)
    feed.apply(stop_asked("stop-1", "design", 1))
    assert [o.id for o in feed.state.stops.open] == ["stop-1"]
    changed = feed.apply(answered("stop-1", 2))
    assert feed.state.stops.open == ()
    marker = feed.state.zeus.entries[-1]
    assert (marker.kind, marker.stop) == (ZeusKind.MARKER, "design")
    assert {Pane.STOPS, Pane.ZEUS} <= changed.panes and StaleRoute.LIVENESS in changed.refresh


# -- progress is a hint ----------------------------------------------------------------------


def test_progress_moves_only_busy_and_the_next_record_clears_it() -> None:
    feed = Feed(SID)
    assert not feed.apply(progress("alive")), "the keep-alive changes nothing"
    feed.apply(progress("thinking"))
    assert feed.state.zeus.busy == "thinking"
    feed.apply(assistant("here", 1))
    assert feed.state.zeus.busy is None


def test_an_error_frame_is_said_even_with_no_record() -> None:
    feed = Feed(SID)
    feed.apply(progress("error", text="model refused"))
    entry = feed.state.zeus.entries[-1]
    assert (entry.seq, entry.kind, entry.text) == (None, ZeusKind.ERROR, "model refused")


def test_dropping_every_progress_frame_changes_nothing_but_busy() -> None:
    records = [user("hi", 1), assistant("hello", 2), episode(3), *run_frames()]
    hints = [progress("thinking"), progress("tool_call", tool="read")]
    with_hints = Feed(SID)
    for frame in records:
        for hint in hints:
            with_hints.apply(hint)
        with_hints.apply(frame)
    without = Feed(SID)
    for frame in records:
        without.apply(frame)
    assert replace(with_hints.state.zeus, busy=None) == without.state.zeus
    assert with_hints.state.athena == without.state.athena
    assert with_hints.state.worker == without.state.worker
    assert with_hints.state.header == without.state.header


# -- robustness ------------------------------------------------------------------------------


def test_a_record_seen_twice_changes_nothing() -> None:
    feed = _fed(run_frames())
    again = feed.apply(run_frames()[30])
    assert not again
    assert len(feed.state.worker.tasks["T1"].entries) == 10


def test_a_record_this_client_cannot_read_is_said_in_its_pane() -> None:
    feed = _fed([])
    feed.apply(session("turn", {"role": "user"}, 2))  # no text
    feed.apply(run_record("tool_call", {"role": "worker"}, 1, task_id="T1"))
    feed.apply(run_record("model_call", {"cost_usd": "lots"}, 2))
    zeus_last = feed.state.zeus.entries[-1]
    assert (zeus_last.kind, zeus_last.text) == (ZeusKind.UNREADABLE, "turn")
    worker_last = feed.state.worker.tasks["T1"].entries[-1]
    assert (worker_last.kind, worker_last.text) == (WorkerKind.UNREADABLE, "tool_call")
    athena_last = feed.state.athena.tail[-1]
    assert (athena_last.kind, athena_last.text) == (AthenaKind.UNREADABLE, "model_call")


def test_an_unreadable_question_does_not_break_the_open_ones() -> None:
    feed = Feed(SID)
    feed.apply(stop_asked("stop-1", "design", 1))
    feed.apply(session("question_asked", {"stop": {"id": "broken"}}, 2))
    assert [o.id for o in feed.state.stops.open] == ["stop-1"]
    assert feed.state.zeus.entries[-1].kind is ZeusKind.UNREADABLE


def test_an_older_run_s_records_do_not_reach_the_screen() -> None:
    feed = _fed([])
    feed.apply(episode(2, run_id="2026-10-11-later"))
    assert not feed.apply(run_frames()[5])
    assert feed.state.worker.tasks == {}
