"""Zeus's tool calls and the approvals they needed, as cards (`panes/cards.py`).

What these pin: a command, or a call an approval names, has a card of its own and the calls
between share one; a command is drawn whole, never cut; the tick says how an approval ended
and never who; each result is `✔` or `✖` and its first line in the card's foot, with the REPL's
body lines expanded; an approval of another role's has a card of its own; and in the pane no
`answered … by human` line is drawn.
"""

from __future__ import annotations

import pytest
from overlay_fakes import Wire, hosted
from tui_frames import (
    RUN_ID,
    SID,
    answered,
    approval_asked,
    episode,
    run_frames,
    run_record,
    session,
    user,
)

from chickadee_cli.render.stream import Verbosity
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import Approval, AthenaEntry, AthenaKind, StaleRoute, drawn_in_card
from chickadee_cli.tui.panes.athena import athena_cards
from chickadee_cli.tui.panes.cards import (
    Card,
    CardText,
    approval_card,
    box_cards,
    failed,
    hook_tick,
    tick,
    turn_cards,
)
from chickadee_cli.tui.panes.worker import worker_cards, worker_line
from chickadee_cli.tui.panes.zeus import ToolMode, entry_text
from chickadee_cli.tui.theme import ASCII, UNICODE, palette
from chickadee_serve.frames import JsonValue
from chickadee_wire.models import (
    AssistantTurnP,
    EffectCompletedP,
    HookP,
    PathDiff,
    PlanTaskP,
    RunDiff,
    ToolCallP,
    ToolRecord,
)

pytestmark = pytest.mark.anyio

COLOURS = palette("truecolor")
LONG = (
    "uv run --frozen pytest packages/core/tests/net/test_fetch.py -k 'retry and backoff' "
    "-x --maxfail=1 -q --no-header -p no:cacheprovider 2>&1 | tail -40"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _call(cid: str, name: str, **arguments: JsonValue) -> dict[str, JsonValue]:
    return {"id": cid, "name": name, "arguments": dict(arguments)}


def _session() -> Feed:
    """Two reads, a command Zeus was asked to approve, and their results."""
    feed = Feed(SID)
    feed.apply(user("make the retry back off", 1))
    calls: list[JsonValue] = [
        _call("c1", "read", target="net/fetch.py"),
        _call("c2", "grep", pattern="retry|503", path="core"),
        _call("c3", "shell", command=LONG),
    ]
    feed.apply(session("turn", {"role": "assistant", "text": "Looking.", "calls": calls}, 2))
    feed.apply(session("tool", {"call_id": "c1", "ran": True, "text": "212 lines\nclass F"}, 3))
    feed.apply(session("tool", {"call_id": "c2", "ran": True, "text": "14 matches"}, 4))
    feed.apply(
        approval_asked(
            "act-7", 5, asker="zeus", task_id=None, operation_id=f"{SID}:c3", command=LONG
        )
    )
    return feed


def _ran(feed: Feed, seq: int = 7) -> None:
    text = "exit 1 · 412 ms · sandbox none\n     FAILED test_retry_on_503\n1 attempt"
    feed.apply(session("tool", {"call_id": "c3", "ran": True, "text": text}, seq))


def _turn(feed: Feed) -> AssistantTurnP:
    record = feed.state.zeus.entries[1].record
    assert isinstance(record, AssistantTurnP)
    return record


def _cards(feed: Feed, *, expanded: bool = False) -> list[tuple[str, str, str, str]]:
    cards = turn_cards(
        _turn(feed), feed.state.zeus, COLOURS, expanded=expanded, verbosity=Verbosity.NORMAL
    )
    return [
        (c.title.plain, c.tick.plain, c.body.plain, c.foot.plain if c.foot else "")
        for c in cards
    ]


def test_a_command_has_a_card_of_its_own_and_the_calls_between_share_one() -> None:
    feed = _session()
    (reads, command) = _cards(feed)
    assert reads[0] == "2 calls" and reads[1] == ""
    assert reads[2].splitlines() == [
        "read  net/fetch.py",
        "grep  path core · pattern retry|503",
    ]
    assert reads[3].splitlines() == ["✔ 212 lines", "✔ 14 matches"]
    assert command[0] == "$ shell" and command[2] == LONG, "a command whole, never cut"


def test_the_tick_waits_then_says_how_it_ended_never_who() -> None:
    feed = _session()
    assert _cards(feed)[1][1] == "waiting for you"
    assert _cards(feed)[1][3] == "running…"
    feed.apply(answered("act-7", 6))
    _ran(feed)
    _, tick_said, _, foot = _cards(feed)[1]
    assert tick_said == "✔ approved once" and "human" not in tick_said
    assert foot == "✖ exit 1 · 412 ms · sandbox none"


def test_expanded_the_foot_adds_the_result_s_lines_without_the_stream_s_indent() -> None:
    feed = _session()
    feed.apply(answered("act-7", 6))
    _ran(feed)
    foot = _cards(feed, expanded=True)[1][3].splitlines()
    assert foot[0].startswith("✖ exit 1") and foot[1] == "FAILED test_retry_on_503"


@pytest.mark.parametrize(
    ("choice", "scope", "source", "said"),
    [
        ("approve", "once", "human", "✔ approved once"),
        ("approve", "session", "human", "✔ approved for this session"),
        ("approve", "repo", "human", "✔ approved for this repo"),
        ("approve", "once", "default", "✔ approved by default"),
        ("approve", "session", "grant", "✔ allowed by a grant"),
        ("approve", "never_ask", "never_ask", "✔ never asked"),
        ("deny", "once", "human", "✖ denied"),
        ("deny", "once", "default", "✖ denied · no answer"),
        ("abort", "once", "human", "✖ stopped"),
        ("deny", "once", "withdrawn", "withdrawn"),
        (None, "once", None, "waiting for you"),
    ],
)
def test_each_answer_s_tick(
    choice: str | None, scope: str, source: str | None, said: str
) -> None:
    approval = Approval(asker="zeus", command="ls", choice=choice, scope=scope, source=source)
    assert tick(approval, COLOURS).plain == said
    assert tick(None, COLOURS).plain == ""


def test_a_result_failed_when_it_did_not_run_or_exited_other_than_0() -> None:
    def result(text: str, *, ran: bool = True) -> ToolRecord:
        return ToolRecord(call_id="c", ran=ran, text=text)

    assert not failed(result("exit 0 · 3 ms"))
    assert failed(result("exit 2 · 3 ms"))
    assert not failed(result("212 lines")), "a read's header is no exit"
    assert failed(result("denied", ran=False))


def test_a_denied_command_says_so_in_its_tick_and_its_foot() -> None:
    feed = _session()
    feed.apply(answered("act-7", 6, choice="deny"))
    feed.apply(session("tool", {"call_id": "c3", "ran": False, "text": "denied"}, 7))
    _, tick_said, _, foot = _cards(feed)[1]
    assert tick_said == "✖ denied" and foot == "✖ not run: denied"


def test_another_role_s_approval_is_a_card_of_its_own() -> None:
    feed = Feed(SID)
    feed.apply(approval_asked("act-9", 1, command="uv add numpy"))
    feed.apply(answered("act-9", 2))
    approval = feed.state.zeus.approvals["act-9"]
    assert approval.call_id is None, "the worker's asks about no call of Zeus's"
    card = approval_card(approval, COLOURS)
    assert (card.title.plain, card.tick.plain, card.body.plain) == (
        "$ worker · T2",
        "✔ approved once",
        "uv add numpy",
    )


async def test_in_the_pane_the_command_is_whole_and_no_answer_line_is_drawn() -> None:
    feed = _session()
    feed.apply(answered("act-7", 6))
    _ran(feed)
    feed.apply(approval_asked("act-9", 8, command="uv add numpy"))
    feed.apply(answered("act-9", 9))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire())
    async with app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        cards = list(app.zeus.query(Card))
        titles = [str(card.border_title) for card in cards]
        assert len(cards) == 3, "two under his turn, one for the worker's approval"
        assert "approved once" in titles[1] and "approved once" in titles[2]
        shell = cards[1]
        width = shell.body.content_region.width
        drawn = "".join(str(shell.body.render()).split())
        assert drawn == "".join(LONG.split()), "the command whole, wrapped, never cut"
        assert width < len(LONG)
        said = " ".join(str(view.render()) for view in app.zeus.query(".entry"))
        assert "answered" not in said and "by human" not in said


# -- the box roles' cards ----------------------------------------------------------


def _run(upto: int) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


def _said(card: CardText) -> tuple[str, str, str, str]:
    return (
        card.title.plain,
        card.tick.plain,
        card.body.plain,
        card.foot.plain if card.foot else "",
    )


def test_a_step_s_counts_are_the_server_s_and_its_paths_alone_until_read() -> None:
    feed = _run(19)
    first = worker_cards(feed.state, COLOURS)[0]
    assert first.body.plain == "✎ src/client.py", "the paths it changed, no counts yet"
    step = next(
        e.record
        for e in feed.state.worker.tasks["T1"].entries
        if isinstance(e.record, EffectCompletedP)
    )
    changed = PathDiff(path="src/client.py", kind="edited", added=1, removed=1)
    diff = RunDiff(
        run_id=RUN_ID, task_id="T1", operation_id=step.operation_id, base="b" * 40,
        tip="t" * 40, files=[changed], text="",
    )  # fmt: skip
    feed.snapshot(diff)
    assert worker_cards(feed.state, COLOURS)[0].body.plain == "✎ src/client.py  +1 -1"
    assert "T1" not in feed.state.diffs, "a step's diff is never the task's"


def test_her_own_edit_and_command_are_cards_as_the_worker_s() -> None:
    feed = _run(6)
    step: dict[str, JsonValue] = {
        "operation_id": "step:athena:9", "outcome": "succeeded",
        "observed_surface": ["PLAN.md"],
    }  # fmt: skip
    ran: dict[str, JsonValue] = {
        "operation_id": "step:athena:10", "policy": "worktree", "label": "git status",
        "argv_sha": "a", "cwd": ".", "exit_code": 2, "network": False, "sandbox": "landlock",
        "stderr_sha": "e", "stdout_sha": "o", "timed_out": False, "violations": [],
        "wall_ms": 3,
    }  # fmt: skip

    def call(tool: str, args: str, result: str) -> dict[str, JsonValue]:
        return {
            "role": "athena", "tool": tool, "args_sha": "a", "outcome": "served",
            "result_tokens": 1, "args_preview": args, "result_preview": result,
        }  # fmt: skip

    feed.apply(run_record("effect_completed", step, 900))
    feed.apply(run_record("tool_call", call("edit", '{"path": "PLAN.md"}', "ok: PLAN.md"), 901))
    feed.apply(run_record("exec", ran, 902))
    feed.apply(
        run_record(
            "tool_call", call("shell", '{"command": "git status"}', "exit 2\nnot a repo"), 903
        )
    )
    edit, shell = (_said(c) for c in athena_cards(feed.state, COLOURS)[-2:])
    assert edit[1:3] == ("✔", "✎ PLAN.md")
    assert shell[1:] == ("✖ exit 2", "$ git status", "not a repo")


def _hook(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"event": "before_stop", "name": "progress_watchdog", "decision": "continue",
            "wall_ms": 12, "role": "worker", **fields}  # fmt: skip


def test_a_hook_that_fires_for_the_worker_is_a_card_where_it_fired() -> None:
    """In the task's stream, between the cards before and after it; the set an
    episode bound at its start is no firing, and is not drawn."""
    feed = _run(19)
    before = len(worker_cards(feed.state, COLOURS))
    feed.apply(run_record("hook", _hook(decision="bound", event="bound"), 900, task_id="T1"))
    assert len(worker_cards(feed.state, COLOURS)) == before, "the bound set draws nothing"
    said = "you have run no test since your last edit; run one, then report"
    feed.apply(run_record("hook", _hook(message=said), 901, task_id="T1"))
    card = worker_cards(feed.state, COLOURS)[-1]
    assert _said(card) == (
        "↪ hook",
        "continue",
        f"progress_watchdog at before_stop · 12 ms\n{said}",
        "",
    )
    assert card.marks == frozenset({901})
    assert worker_line(feed.state.worker.tasks["T1"].entries[-1], UNICODE, COLOURS).plain == (
        "↪ progress_watchdog continue"
    )
    assert worker_line(feed.state.worker.tasks["T1"].entries[-1], ASCII, COLOURS).plain == (
        "~> progress_watchdog continue"
    ), "a terminal without the glyph"


def test_a_hook_with_no_task_is_hers() -> None:
    feed = _run(19)
    feed.apply(run_record("hook", _hook(name="owed_review", role="athena"), 902))
    title, decided, where, _ = _said(athena_cards(feed.state, COLOURS)[-1])
    assert (title, decided, where) == (
        "↪ hook",
        "continue",
        "owed_review at before_stop · 12 ms",
    )


@pytest.mark.parametrize(
    ("fields", "word", "tone"),
    [
        ({"decision": "deny"}, "✖ deny", "bad"),
        ({"decision": "halt"}, "✖ halt", "bad"),
        ({"decision": "error", "error": "boom"}, "✖ error", "bad"),
        ({"decision": "note", "timed_out": True}, "✖ timed out", "bad"),
        ({"decision": "retry_task"}, "retry task", "zeus"),
        ({"decision": "ask"}, "ask", "zeus"),
        ({"decision": "note"}, "note", "dim"),
        ({"decision": "warn"}, "warn", "dim"),
        ({"decision": "proceed"}, "proceed", "dim"),
    ],
)
def test_what_a_hook_decided_is_said_in_its_tone(
    fields: dict[str, JsonValue], word: str, tone: str
) -> None:
    said = hook_tick(HookP.model_validate(_hook(**fields)), COLOURS)
    assert said.plain == word
    assert {span.style for span in said.spans} == {COLOURS[tone]}


def test_a_failed_hook_s_message_is_red_and_a_reached_bound_is_said() -> None:
    from chickadee_cli.tui.panes.cards import hook_card

    failed_ = hook_card(
        HookP.model_validate(_hook(decision="error", error="boom", message="boom")), 7, COLOURS
    )
    assert failed_.body.plain.splitlines()[1] == "boom"
    assert any(s.style == COLOURS["bad"] for s in failed_.body.spans)
    bounded = hook_card(HookP.model_validate(_hook(bounded=True)), 8, COLOURS)
    assert bounded.body.plain.splitlines()[-1] == (
        "its bound was reached: it decides nothing more"
    )


def test_a_plan_card_lists_the_plan_s_tasks() -> None:
    tasks = [PlanTaskP(id="T1", title="Raise the limit"), PlanTaskP(id="T2", title="Lower it")]
    plan = ToolCallP(
        role="athena", tool="plan", args_sha="a", outcome="served", result_tokens=1,
        result_preview="approved at stop 2",
    )  # fmt: skip
    entry = AthenaEntry(4, AthenaKind.CALL, plan)
    (card,) = box_cards([entry], step_diffs={}, colours=COLOURS, plan=tasks)
    assert _said(card) == ("plan", "✔ 2 tasks", "T1 Raise the limit\nT2 Lower it", "")


async def test_the_worker_s_cards_grow_in_place() -> None:
    feed = _run(12)
    # As a replay: no diffs served, so the cards show the paths alone.
    feed.state.not_served |= {StaleRoute.TASK_DIFF, StaleRoute.STEP_DIFFS}
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire())
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        worker = app.base.query_one("#worker")
        before = list(worker.query(Card))
        for frame in run_frames():
            if 12 < frame.seq <= 19:
                app.take(frame)
        await pilot.pause()
        after = list(worker.query(Card))
        assert len(after) > len(before) and after[: len(before)] == before, "never rebuilt"
        assert all(c.border_title for c in after)


# -- his call's question (the owner's call, 2026-10-26) --------------------------------------


QUESTION = "Which branch? I'll use main if you don't say."
DEFAULT = "no answer: proceed on the default the question states"


def _asking() -> Feed:
    feed = Feed(SID)
    feed.apply(user("fix it", 10))
    calls: list[JsonValue] = [_call("q1", "ask_user", question=QUESTION)]
    feed.apply(session("turn", {"role": "assistant", "text": "", "calls": calls}, 11))
    question: dict[str, JsonValue] = {
        "id": "q1", "kind": "clarify_goal", "text": QUESTION, "default": DEFAULT,
    }  # fmt: skip
    feed.apply(session("question_asked", {"question": question}, 12))
    return feed


def _question_card(feed: Feed) -> tuple[str, str, str, str]:
    (card,) = _cards(feed)
    return card


def test_his_question_is_in_its_card_waiting_and_never_printed() -> None:
    feed = _asking()
    assert _question_card(feed) == (
        "question",
        "waiting for you",
        QUESTION,
        "Enter for the default",
    )
    (asked,) = feed.state.stops.open
    assert drawn_in_card(asked, feed.state.zeus), "the REPL's renderer prints it no lines"


def test_an_answer_in_words_is_its_card_s_foot_and_no_line() -> None:
    feed = _asking()
    answered: dict[str, JsonValue] = {
        "id": "q1", "choice": "use release-2", "source": "human", "elapsed_ms": 9,
    }  # fmt: skip
    feed.apply(session("question_answered", answered, 13))
    assert _question_card(feed) == ("question", "✔ answered", QUESTION, "use release-2")
    marker = feed.state.zeus.entries[-1]
    assert (
        entry_text(marker, ToolMode.EXPANDED, Verbosity.NORMAL, COLOURS, feed.state.zeus)
        is None
    )


def test_enter_is_no_answer_proceeding_on_the_default() -> None:
    feed = _asking()
    answered: dict[str, JsonValue] = {
        "id": "q1", "choice": DEFAULT, "source": "human", "elapsed_ms": 9,
    }  # fmt: skip
    feed.apply(session("question_answered", answered, 13))
    assert _question_card(feed)[1:] == ("no answer", QUESTION, "proceeding on the default")


def test_a_one_call_card_never_repeats_its_name() -> None:
    feed = Feed(SID)
    feed.apply(user("read it", 1))
    calls: list[JsonValue] = [_call("c1", "read", target="net/fetch.py")]
    feed.apply(session("turn", {"role": "assistant", "text": "", "calls": calls}, 2))
    assert _cards(feed)[0][:3] == ("read", "", "net/fetch.py")
