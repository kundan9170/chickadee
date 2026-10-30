"""How frames read: goldens for the stream at each verbosity, the
question block, and a pipe against a terminal: no colour and a hard wrap at 80 in a pipe."""

from __future__ import annotations

from dataclasses import replace

import pytest
from recordings import run_frames

from chickadee.zeus.loop import Event
from chickadee_cli.interaction import parse_answer
from chickadee_cli.render.board import BoardTask, board_lines, fold_run
from chickadee_cli.render.diff import NO_LINES, diff_lines
from chickadee_cli.render.question import question_lines
from chickadee_cli.render.stream import TURN_ERROR, Verbosity, frame_lines
from chickadee_cli.render.style import PIPE_WIDTH, Style
from chickadee_serve.frames import JsonValue, ProgressFrame, RecordFrame
from chickadee_serve.progress import describe
from chickadee_wire.models import ActionApproval, CommandBinding, Option, Question
from chickadee_wire.questions import OpenQuestion

PIPE = Style(tty=False)
TTY = Style(tty=True)


def _record(kind: str, payload: dict[str, JsonValue], seq: int = 1) -> RecordFrame:
    return RecordFrame(scope="S", seq=seq, at="t", kind=kind, payload=payload)


ASSISTANT = _record(
    "turn",
    {
        "role": "assistant",
        "text": "Looking.",
        "calls": [{"id": "c1", "name": "read_file", "arguments": {}}],
    },
)
TOOL = _record("tool", {"call_id": "c1", "ran": True, "text": "first line\nsecond"})
SPEND = _record(
    "spend",
    {
        "cost_usd": 0.0123, "cache_writes": 0, "cached_in": 0, "estimated": False,
        "model": "m", "rung": "zeus", "tokens_in": 100, "tokens_out": 10,
    },
)  # fmt: skip


def test_each_verbosity_shows_what_it_says() -> None:
    assert frame_lines(ASSISTANT, Verbosity.QUIET, PIPE) == ["Looking."]
    assert frame_lines(ASSISTANT, Verbosity.NORMAL, PIPE) == [
        "Looking.",
        "  → read_file",
    ]
    assert frame_lines(TOOL, Verbosity.QUIET, PIPE) == []
    # #561: the header, then the first lines of the body.
    assert frame_lines(TOOL, Verbosity.NORMAL, PIPE) == ["  ← first line", "     second"]
    assert frame_lines(SPEND, Verbosity.NORMAL, PIPE) == []
    assert frame_lines(SPEND, Verbosity.VERBOSE, PIPE) == [
        "  spend $0.0123 on zeus (m): 100 in, 10 out"
    ]


def test_the_person_s_words_and_opaque_records() -> None:
    user = _record("turn", {"role": "user", "text": "fix it"})
    assert frame_lines(user, Verbosity.QUIET, PIPE) == ["> fix it"]
    opaque = _record("scout", {"anything": 1})
    assert frame_lines(opaque, Verbosity.NORMAL, PIPE) == []
    assert frame_lines(opaque, Verbosity.VERBOSE, PIPE) == ["  [scout]"]


def test_a_payload_the_client_cannot_read_is_said_not_raised() -> None:
    broken = _record("spend", {"cost_usd": "lots"})
    assert frame_lines(broken, Verbosity.QUIET, PIPE) == [
        "[spend: a payload this client cannot read]"
    ]


def test_progress_is_for_a_terminal_at_v_only() -> None:
    thinking = ProgressFrame(scope="S", kind="tool_call", payload={"tool": "grep"})
    alive = ProgressFrame(scope="S", kind="alive")
    assert frame_lines(thinking, Verbosity.VERBOSE, PIPE) == []
    assert frame_lines(thinking, Verbosity.NORMAL, TTY) == []
    assert frame_lines(thinking, Verbosity.VERBOSE, TTY) == ["\x1b[2m  … tool_call grep\x1b[0m"]
    assert frame_lines(alive, Verbosity.VERBOSE, TTY) == []


def test_an_error_says_why_at_every_verbosity_and_to_a_pipe() -> None:
    """The frame carries the turn's reason (`progress.describe`), and it is said."""
    error = ProgressFrame(scope="S", kind="error", payload={"seq": 3, "text": "no price row"})
    bare = ProgressFrame(scope="S", kind="error", payload={"seq": 3})
    for verbosity in Verbosity:
        assert frame_lines(error, verbosity, Style(tty=False)) == [
            f"{TURN_ERROR}: no price row"
        ]
    assert frame_lines(bare, Verbosity.QUIET, Style(tty=False)) == [TURN_ERROR]
    event = Event("error", 3, "no price row")
    assert describe("S", event).payload == {"seq": 3, "text": "no price row"}


def test_a_pipe_is_plain_and_wrapped_at_80_and_a_terminal_is_not() -> None:
    long = _record("turn", {"role": "assistant", "text": "word " * 40})
    piped = frame_lines(long, Verbosity.QUIET, PIPE)
    assert len(piped) > 1 and all(len(line) <= PIPE_WIDTH for line in piped)
    assert all("\x1b[" not in line for line in piped)
    assert len(frame_lines(long, Verbosity.QUIET, TTY)) == 1


QUESTION = OpenQuestion(
    id="q1",
    seq=3,
    question=Question(
        id="q1",
        kind="clarify_goal",
        text="Which branch?",
        options=[Option(key="main", label="the default"), Option(key="dev", label="dev")],
        default="main",
        evidence=["two branches look live"],
    ),
    approval=None,
)


def test_the_question_block() -> None:
    assert question_lines(QUESTION, PIPE) == [
        "? Which branch?",
        "    two branches look live",
        "    [1] main — the default  (default)",
        "    [2] dev",
        "    answer by key or number; Enter for the default",
    ]


def test_an_answer_names_its_option_by_key_number_or_default() -> None:
    assert parse_answer(QUESTION, "") == ("main", None)
    assert parse_answer(QUESTION, "2") == ("dev", None)
    assert parse_answer(QUESTION, "DEV because it is newer") == ("dev", "because it is newer")
    assert parse_answer(QUESTION, "3") is None
    assert parse_answer(QUESTION, "staging") is None


def _approval(
    grant: CommandBinding | None,
    *,
    exact: CommandBinding | None = None,
    offer_repo: bool = True,
    any_arguments: bool = False,
) -> OpenQuestion:
    return OpenQuestion(
        id="a1",
        seq=4,
        question=None,
        approval=ActionApproval(
            action_id="a1",
            operation_id="op1",
            kind="command",
            display="git push origin main",
            spec_sha="0" * 64,
            network=True,
            idempotent=False,
            reversible=False,
            tier="t2",
            grant=grant,
            exact=exact,
            offer_repo=offer_repo,
            repo_any_arguments=any_arguments,
        ),
    )


def test_an_approval_offers_always_only_with_a_grant_and_says_what_it_covers() -> None:
    """D106 §4-§5, D165 §3: the person is shown what *always* would grant."""
    push = CommandBinding(
        cls="push", prefix=["git", "push"], destination="origin", network=True
    )
    granted = question_lines(_approval(push), PIPE)
    assert "    always covers push `git push` to origin network on" in granted
    assert "    within its allowed options" in granted
    assert [line for line in granted if line.lstrip().startswith("[")] == [
        "    [1] approve — once",
        "    [2] session — always for this session",
        "    [3] repo — always for this repo",
        "    [4] deny  (default)",
        "    [5] abort",
    ]
    once = question_lines(_approval(None), PIPE)
    assert [line for line in once if line.lstrip().startswith("[")] == [
        "    [1] approve",
        "    [2] deny  (default)",
        "    [3] abort",
    ]
    assert not any("always" in line for line in once)


# -- a run's records: the lines, the board and the diff ------------------------------------


def _run_frames() -> list[RecordFrame]:
    """The recorded task run (`recordings.RUN_FIXTURE`), as the server frames it:
    scope and `run_id` are the run's."""
    return run_frames()


def _shown(verbosity: Verbosity) -> list[str]:
    return [line for f in _run_frames() for line in frame_lines(f, verbosity, PIPE)]


def test_a_run_reads_as_task_grouped_lines() -> None:
    shown = _shown(Verbosity.NORMAL)
    run_id = _run_frames()[0].scope
    assert shown == [
        f"run {run_id}: Raise the retry limit and lower the timeout · worktree",
        "· cap $5.00",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · tool_use → plan",
        "plan v1 by athena · approved at stop 2",
        '    athena → plan {"text": "# Plan …"}',
        "        ← approved at stop 2",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · tool_use → delegate",
        "> T1 delegated by athena: Raise the retry limit to 5",
        "    worker (T1) · task · 1,600 in / 90 out · $0.0004 · tool_use → edit",
        "    step succeeded: src/client.py",
        '    worker (T1) → edit {"new": "max_retries = 5", "old": "max_retries = 3",',
        '    "path": "src/client.py"}',
        "        ← ok: src/client.py syntax ok: src/client.py",
        "    worker (T1) · task · 1,600 in / 90 out · $0.0004 · tool_use → shell",
        "    ran run the tests: exit 1",
        '    worker (T1) → shell {"command": "pytest -q"}',
        "        ← exit 1 FAILED tests/test_client.py::test_retries - assert 3 == 5 1",
        "        failed",
        "    worker (T1) · task · 1,600 in / 90 out · $0.0004 · end_turn",
        "        “T1: max_retries is 5; tests/test_client.py still expects 3.”",
        "  T1 round 1 replied · 1 file(s) · 1 command(s) · tests failed",
        '    athena → delegate {"title": "Raise the retry limit to 5"}',
        "        ← T1 replied",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · tool_use → review",
        "  T1 reviewed: feedback",
        "    worker (T1) · task · 1,600 in / 90 out · $0.0004 · tool_use → edit",
        "    step succeeded: tests/test_client.py",
        '    worker (T1) → edit {"new": "== 5", "old": "== 3", "path":',
        '    "tests/test_client.py"}',
        "        ← ok: tests/test_client.py syntax ok: tests/test_client.py",
        "    worker (T1) · task · 1,600 in / 90 out · $0.0004 · tool_use → shell",
        "    ran run the tests: exit 0",
        '    worker (T1) → shell {"command": "pytest -q"}',
        "        ← exit 0 1 passed",
        "    worker (T1) · task · 1,600 in / 90 out · $0.0004 · end_turn",
        "        “T1: the test expects 5 now; 1 passed.”",
        "  T1 round 2 replied · 2 file(s) · 1 command(s) · tests passed",
        '    athena → review {"answer": "feedback"}',
        "        ← T1 replied",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · tool_use → review",
        "  T1 reviewed: accept",
        '    athena → review {"answer": "accept"}',
        "        ← T1 committed",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · tool_use → delegate",
        "> T2 delegated by athena: Lower the timeout",
        "    worker (T2) · task · 1,600 in / 90 out · $0.0004 · tool_use → edit",
        "    step succeeded: src/config.py",
        '    worker (T2) → edit {"new": "timeout = 10", "old": "timeout = 30", "path":',
        '    "src/config.py"}',
        "        ← ok: src/config.py syntax ok: src/config.py",
        "  T2 round 1 limit · 1 file(s)",
        '    athena → delegate {"title": "Lower the timeout"}',
        "        ← T2 stopped: limit",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · tool_use → review",
        "  T2 reviewed: abandon",
        '    athena → review {"answer": "abandon"}',
        "        ← T2 abandoned",
        "    athena · episode · 2,400 in / 180 out · $0.0123 · end_turn",
        "        “T1 is committed; T2 was abandoned: which timeout to lower is not",
        "        settled.”",
        f"run {run_id} ended PARTIAL · 1/2 task(s) committed · $0.0889 · 41.0s",
        "· T2 abandoned: which timeout to lower is not settled",
    ]


def test_a_run_at_q_is_silent_and_at_v_names_the_rest() -> None:
    assert _shown(Verbosity.QUIET) == []
    verbose = _shown(Verbosity.VERBOSE)
    assert "    step succeeded: nothing changed" in verbose, "a step that changed nothing, at v"
    assert "    [effect_intended]" in verbose, "a kind with no line of its own, by name"
    assert len(verbose) > len(_shown(Verbosity.NORMAL))


def test_the_board_is_each_task_and_the_end() -> None:
    frames = _run_frames()
    view = fold_run(frames[0].scope, frames)
    assert board_lines(view, PIPE) == [
        f"run {frames[0].scope}: Raise the retry limit and lower the timeout",
        "  T1   committed   2 round(s)  $0.0024  Raise the retry limit to 5",
        # The fixture's T2 was answered `abandon`, in older versions: it reads as a drop.
        "  T2   dropped     1 round(s)  $0.0004  Lower the timeout",
        "       last round: limit",
        "  ended PARTIAL · 1/2 task(s) committed · $0.0889",
    ]


def test_the_fold_holds_every_field_of_each_task_and_the_end() -> None:
    # Pinned before the fold was shared one record at a time: what the board does
    # not print (the files changed, the exact cost, the whole end) is held here.
    frames = _run_frames()
    view = fold_run(frames[0].scope, frames)
    assert view.run_id == "2026-10-10-0123abcd"
    assert view.goal == "Raise the retry limit and lower the timeout"
    tasks = list(view.tasks.values())
    assert [t.cost_usd for t in tasks] == pytest.approx([0.0024, 0.0004])
    assert [replace(t, cost_usd=0.0) for t in tasks] == [
        BoardTask(
            "T1",
            "Raise the retry limit to 5",
            "committed",
            2,
            0.0,
            None,
            ("src/client.py", "tests/test_client.py"),
            ("src/client.py", "tests/test_client.py"),
        ),
        BoardTask(
            "T2", "Lower the timeout", "dropped", 1, 0.0, "limit", ("src/config.py",), ()
        ),
    ]
    assert view.end is not None
    assert (view.end.outcome, view.end.committed, view.end.tasks) == ("PARTIAL", 1, 2)
    assert view.end.why == "T2 abandoned: which timeout to lower is not settled"


def test_an_unfinished_run_shows_how_its_last_round_ended() -> None:
    frames = [f for f in _run_frames() if f.seq <= 50]  # T2 reported, not yet reviewed
    view = fold_run(frames[0].scope, frames)
    lines = board_lines(view, PIPE)
    assert "  T2   running     1 round(s)  $0.0004  Lower the timeout" in lines
    assert "       last round: limit" in lines
    assert lines[-1] == "  in flight: no run_end yet"


def test_the_diff_names_committed_files_and_says_lines_are_not_on_the_wire() -> None:
    frames = _run_frames()
    assert diff_lines(fold_run(frames[0].scope, frames), PIPE) == [
        f"run {frames[0].scope}: Raise the retry limit and lower the timeout",
        "  T1  committed",
        "      src/client.py",
        "      tests/test_client.py",
        "  T2  changed, not committed (dropped)",
        "      src/config.py",
        f"  {NO_LINES}",
    ]
    unreported = [f for f in frames if f.seq <= 18]  # T1's first round, before its report
    assert "no file changed" in diff_lines(fold_run(frames[0].scope, unreported), PIPE)[-1]


# -- a coder result says why it did not complete ---------------------------------------


def _tool(text: str, outcome: dict[str, JsonValue] | None = None) -> RecordFrame:
    payload: dict[str, JsonValue] = {"call_id": "c1", "text": text, "ran": True}
    if outcome is not None:
        payload["outcome"] = outcome
    return RecordFrame(
        scope="S-1", seq=7, at="t", kind="tool", payload=payload, run_id="2026-10-13-be6bf9eb"
    )


BLOCKED = _tool(
    "BLOCKED{tier_above_approved} · run not started · $0.0000\n"
    "why: the plan needs t2, above the approved t0: rows [2]",
    {
        "terminal": "BLOCKED{tier_above_approved}",
        "why": ["the plan needs t2, above the approved t0: rows [2]"],
    },
)


def test_a_blocked_coder_result_prints_its_reason() -> None:
    assert frame_lines(BLOCKED, Verbosity.NORMAL, PIPE) == [
        "  ← BLOCKED{tier_above_approved} · run not started · $0.0000",
        "     the plan needs t2, above the approved t0: rows [2]",
    ]


def test_an_escalated_coder_result_prints_the_packet_that_failed_capped() -> None:
    why: list[JsonValue] = [
        "packet e1: escalated: infra: git status failed",
        "failed: infra in packet e1",
        "packet e2: escalated",
        "packet e3: escalated",
        "packet e4: escalated",
    ]
    frame = _tool(
        "ESCALATED · run 2026-10-13-c555647a · $0.0000", {"terminal": "ESCALATED", "why": why}
    )
    lines = frame_lines(frame, Verbosity.NORMAL, PIPE)
    assert lines[1] == f"     {why[0]}"
    assert len(lines) == 1 + 4 + 1
    assert lines[-1] == "     … 1 more; /status shows the board"


def test_verbose_shows_every_reason_line() -> None:
    why: list[JsonValue] = [f"packet e{n}: escalated" for n in range(1, 7)]
    frame = _tool("ESCALATED · run r · $0.0000", {"terminal": "ESCALATED", "why": why})
    lines = frame_lines(frame, Verbosity.VERBOSE, PIPE)
    assert lines[1:] == [f"     {w}" for w in why], "no cap and no /status hint at -v"


def test_a_completed_coder_result_stays_one_line_and_another_tool_shows_its_body() -> None:
    done = _tool("COMPLETED · run r · $0.01\nthe report", {"terminal": "COMPLETED", "why": []})
    assert frame_lines(done, Verbosity.NORMAL, PIPE) == ["  ← COMPLETED · run r · $0.01"]
    other = _tool("def get(key):\nreturn key")
    assert frame_lines(other, Verbosity.NORMAL, PIPE) == [
        "  ← def get(key):",
        "     return key",
    ]


def test_a_result_shows_its_header_then_its_first_body_lines_and_v_shows_all() -> None:
    """#561: a shell result's header alone said nothing about what the command printed."""
    text = "exit 0 · 24 ms\nREADME.md\ncalc.py\nsetup.py\ntests\n\n"
    shown = frame_lines(_tool(text), Verbosity.NORMAL, PIPE)
    assert shown == [
        "  ← exit 0 · 24 ms",
        "     README.md",
        "     calc.py",
        "     setup.py",
        "     … 1 more lines",
    ]
    whole = frame_lines(_tool(text), Verbosity.VERBOSE, PIPE)
    assert whole[1:] == ["     README.md", "     calc.py", "     setup.py", "     tests"]


def test_a_zeus_call_shows_its_arguments_but_its_findings() -> None:
    """#561: a bare `→ shell` said nothing; the arguments as sent, less the findings. No
    purpose line follows the call."""
    turn = _record(
        "turn",
        {
            "role": "assistant",
            "text": "",
            "calls": [
                {
                    "id": "c1",
                    "name": "shell",
                    "arguments": {"command": "ls -la", "findings": {}},
                },
                {"id": "c2", "name": "atlas", "arguments": {}},
            ],
        },
    )
    assert frame_lines(turn, Verbosity.NORMAL, PIPE) == [
        '  → shell {"command": "ls -la"}',
        "  → atlas",
    ]


def test_the_reason_shows_on_a_terminal_too() -> None:
    shown = "\n".join(frame_lines(BLOCKED, Verbosity.NORMAL, TTY))
    assert "the plan needs t2, above the approved t0: rows [2]" in shown


def test_an_approval_says_which_always_it_offers_and_how_far_each_reaches() -> None:
    """T-3.158 E.1 and E.3: *this session* is the exact command; the repository answer says
    when it covers any arguments, and is left out where the prompt does not offer it."""
    publish = CommandBinding(cls="push", prefix=["npm", "publish"], destination="index")
    exact = CommandBinding(cls="push", prefix=["npm", "publish"], destination="index")
    asked = _approval(publish, exact=exact, offer_repo=False, any_arguments=True)
    lines = question_lines(asked, PIPE)
    assert [line for line in lines if line.lstrip().startswith("[")] == [
        "    [1] approve — once",
        "    [2] session — always for this session, this exact command",
        "    [3] deny  (default)",
        "    [4] abort",
    ]
    assert not any("always covers" in line for line in lines)
    pull = CommandBinding(cls="external_fetch", prefix=["docker", "pull"])
    lines = question_lines(_approval(pull, exact=pull, any_arguments=True), PIPE)
    assert "    always covers external_fetch `docker pull`" in lines
    assert "    with any arguments, without asking" in lines


def test_a_drop_is_shown_as_dropped_as_an_abandon_was() -> None:
    """D214 renamed `abandon` to `drop`: a run answered either way reads the same."""
    frames = [
        f.model_copy(update={"payload": {**f.payload, "answer": "drop"}})
        if f.kind == "task_reviewed" and f.payload.get("answer") == "abandon"
        else f
        for f in _run_frames()
    ]
    view = fold_run(frames[0].scope, frames)
    assert "  T2   dropped     1 round(s)  $0.0004  Lower the timeout" in board_lines(
        view, PIPE
    )


def test_every_review_answer_is_a_state_the_board_draws() -> None:
    """A review answer the board does not map is shown as its own word, which no surface draws
    (`TASK_GLYPH` is indexed by state): `continue` is running again."""
    from typing import get_args

    from chickadee.schema.journal import ReviewAnswer
    from chickadee_cli.render.board import (
        _REVIEWED as REVIEWED,  # pyright: ignore[reportPrivateUsage]
    )
    from chickadee_cli.render.board import STATES

    assert {a: REVIEWED.get(a, a) for a in get_args(ReviewAnswer)}["continue"] == "running"
    assert {REVIEWED.get(a, a) for a in get_args(ReviewAnswer)} <= STATES
