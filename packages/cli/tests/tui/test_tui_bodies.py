"""What the panes say, built from the recorded run without a
terminal: each line from the records its pane is fed, and nothing the wire does not carry.

What these pin: Athena's board names every task's state in a word and her cards show her calls
and verdicts, not her model calls; the worker's cards are its calls, steps and commands in
order (`test_tui_cards.py` pins a card's parts), its time read from the records, and its
facts say what ran and whether a commit landed; the diff names the task's files and counts,
says the line diff is not on the wire, and shows the failing test while it fails; a refused
call and an unreadable record are said.
"""

from __future__ import annotations

from tui_frames import SID, episode, run_frames, run_record

from chickadee_cli.render.diff import NO_LINES
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import TuiState
from chickadee_cli.tui.panes.athena import TAIL_SHOWN, athena_body, athena_cards
from chickadee_cli.tui.panes.cards import CardText
from chickadee_cli.tui.panes.diff import diff_body, file_tabs
from chickadee_cli.tui.panes.worker import elapsed, facts_lines, worker_body, worker_cards
from chickadee_cli.tui.theme import UNICODE, palette
from chickadee_serve.frames import JsonValue

COLOURS = palette("truecolor")
MINUS = UNICODE.minus


def _state(upto: int, *extra: tuple[str, dict[str, JsonValue], str | None]) -> TuiState:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    for seq, (kind, payload, task) in enumerate(extra, start=1000):
        feed.apply(run_record(kind, payload, seq, **({"task_id": task} if task else {})))
    return feed.state


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _said(card: CardText) -> tuple[str, str, str, str]:
    return (
        card.title.plain,
        card.tick.plain,
        card.body.plain,
        card.foot.plain if card.foot else "",
    )


# -- Athena ----------------------------------------------------------------------------------


def test_athena_s_board_names_each_task_s_state_in_a_word() -> None:
    lines = _lines(athena_body(_state(56), UNICODE, COLOURS).plain)
    assert lines[0] == "PLAN v1 · approved"
    assert lines[1] == "✔ T1 Raise the retry limit to 5"
    assert lines[2] == "  committed · 2 rd · $0.0024"
    assert lines[3] == "✖ T2 Lower the timeout"
    assert lines[4] == "  dropped · 1 rd · $0.0004 · limit"


def test_the_running_task_is_on_the_band_and_says_it_runs() -> None:
    body = athena_body(_state(30), UNICODE, COLOURS)
    assert "  running · 1 rd" in body.plain
    assert any("#0f2f2d" in str(span.style) for span in body.spans), "the mockup's run band"


def test_her_cards_are_her_calls_and_verdicts_never_her_model_calls() -> None:
    state = _state(56)
    assert "───" not in athena_body(state, UNICODE, COLOURS).plain, "the board, then the cards"
    cards = [_said(c) for c in athena_cards(state, COLOURS)]
    assert len(cards) == TAIL_SHOWN
    assert cards == [
        ("review", "✔ accept", "T1 Raise the retry limit to 5", ""),
        ("delegate", "", "Lower the timeout", "T2 stopped: limit"),
        ("review", "✖ abandon", "T2 Lower the timeout", ""),
    ], "a verdict once, as `task_reviewed` says it: her `review` call says it again, later"
    assert not any("model" in part for card in cards for part in card)


def test_no_plan_yet_is_said() -> None:
    assert athena_body(_state(2), UNICODE, COLOURS).plain.startswith("PLAN not written yet")


# -- the worker ------------------------------------------------------------------------------


def test_the_worker_s_cards_are_its_calls_steps_and_commands_in_order() -> None:
    state = _state(19)
    head = worker_body(state, COLOURS).plain
    assert head == "T1 · round 1  0:13", "the records' own times: 10:00:25 to 10:00:38"
    assert [_said(c) for c in worker_cards(state, COLOURS)] == [
        ("step 1", "✔", "✎ src/client.py", ""),
        (
            "step 2",
            "✖ exit 1",
            "$ pytest -q",
            "FAILED tests/test_client.py::test_retries - assert 3 == 5",
        ),
    ], "an edit its files, never its text; a command `$` and the command"


def test_the_time_is_read_from_the_records_never_the_clock() -> None:
    task = _state(56).worker.tasks["T1"]
    assert (task.delegated_at, task.last_at) == ("2026-10-10T10:00:25Z", "2026-10-10T10:01:00Z")
    assert elapsed(task) == "0:35"


def test_the_facts_say_what_ran_and_whether_a_commit_landed() -> None:
    round_1 = _lines(facts_lines(_state(19), UNICODE, COLOURS).plain)
    assert round_1 == [
        f"1 file(s) +1 {MINUS}1  after round 1",
        "tests: 1 failing",
        "syntax: not checked",
        "no commit yet",
    ]
    assert _lines(facts_lines(_state(38), UNICODE, COLOURS).plain)[1:] == [
        "tests: passed",
        "syntax: not checked",
        "no commit yet",
    ], "accepted, its commit not landed yet"
    assert _lines(facts_lines(_state(41), UNICODE, COLOURS).plain)[-1] == "committed"


def test_a_round_under_way_is_shown_before_it_reports_and_the_facts_say_whose_they_are() -> (
    None
):
    # Round 1 reported at seq 19 with a failing test; round 2 is under way at seq 30.
    head = worker_body(_state(30), COLOURS).plain
    assert head.startswith("T1 · round 2"), "the envelope's round, before its report"
    # Step 3 landed at seq 26; the call that made it lands at 27.
    last = worker_cards(_state(26), COLOURS)[-1]
    assert _said(last)[:3] == ("step 3", "running…", "✎ tests/test_client.py") and last.live
    facts = _lines(facts_lines(_state(30), UNICODE, COLOURS).plain)
    assert facts[0].endswith("after round 1") and facts[1] == "tests: 1 failing"


def test_before_its_first_report_a_task_has_no_facts_yet() -> None:
    assert facts_lines(_state(12), UNICODE, COLOURS).plain == "facts once the round reports"


def test_a_refused_call_and_an_unreadable_record_are_said() -> None:
    refused: dict[str, JsonValue] = {
        "role": "worker", "tool": "shell", "args_sha": "a", "outcome": "served",
        "result_tokens": 0, "denied_reason": "network is off",
    }  # fmt: skip
    state = _state(
        6,
        ("tool_call", refused, "T1"),
        ("exec", {"label": "x"}, "T1"),
    )
    refusal, unreadable = (_said(c) for c in worker_cards(state, COLOURS)[-2:])
    assert refusal == ("shell", "✖ refused", "", "network is off")
    assert unreadable == (
        "unreadable record",
        "✖",
        "exec: a record this client cannot read",
        "",
    )


# -- the diff --------------------------------------------------------------------------------


def test_the_diff_names_the_task_s_files_and_says_the_lines_are_not_on_the_wire() -> None:
    # The files are the CODE pane's tabs; the body is the chosen file's.
    view = _state(19).task_diff
    assert file_tabs(view, UNICODE, COLOURS, None).plain == f" client.py  +1 {MINUS}1 "
    lines = _lines(diff_body(view, UNICODE, COLOURS).plain)
    assert lines[0] == NO_LINES
    assert "✖ tests/test_client.py::test_retries" in lines, "the failing test while it fails"
    assert "«1 · «2" in lines
    later = diff_body(_state(35).task_diff, UNICODE, COLOURS).plain
    assert "test_retries" not in later, "round 2 passed"


def test_before_any_report_the_diff_names_the_paths_the_steps_touched() -> None:
    tabs = file_tabs(_state(12).task_diff, UNICODE, COLOURS, None).plain
    assert tabs == " client.py  ", "no counts before the round's facts measure them"
