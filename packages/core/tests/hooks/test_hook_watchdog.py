"""The progress watchdog's ladder: the meters over tagged calls, the rungs they give, and
the words. Pure: a round is a `WorkerProgress`, as the worker's record tells it.

What these pin: one ordinary mistake never fires it; a loop of nothing is nudged at 4 calls,
told to report at 6 and halted 3 calls later; wandering fires in responses, a continuation read
is not counted, and heavy batching fires at three times the threshold in calls; a change ends
a stall short of its report; a new problem is nudged again, the same problem or a third stall
goes straight to the report; the replay of a round gives each rung once, at its response; and
the report's reader is told what was measured when the round was handed up.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from chickadee.errors import ConfigError
from chickadee.hooks.bind import bind
from chickadee.hooks.builtins import bind_builtin
from chickadee.hooks.payloads import (
    AfterBatch,
    HookEvent,
    ProgressCall,
    ProgressTag,
    Regression,
    RoundEnded,
    TestRunSeen,
    TestsSeen,
    WorkerProgress,
)
from chickadee.hooks.points import HookPoint
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Halt, Note, Proceed
from chickadee.hooks.watchdog import (
    HEAD,
    WatchdogParams,
    progress_report,
    progress_watchdog,
    replay,
    unchecked,
    wandering_threshold,
)
from chickadee.schema.journal import RoundEnd, TaskFactsP

LADDER: dict[str, object] = {
    "looping": 4,
    "wandering_base": 12,
    "wandering_min": 12,
    "wandering_max": 30,
}
"""The thresholds these tests' numbers are written for (D231's): the ladder's working is
tested at fixed values; the defaults are pinned once, below."""

PARAMS = WatchdogParams.model_validate(LADDER)


def test_the_defaults_are_generous_on_a_medium_codebase() -> None:
    """Looping nudges at 5 and reports at 10; wandering is `20 + 2n`, 20 to 40."""
    defaults = WatchdogParams()
    assert (defaults.looping, defaults.report_looping) == (5, 2.0)
    assert wandering_threshold(defaults, 0, 200) == 20
    assert wandering_threshold(defaults, 10, 200) == 40
    assert wandering_threshold(defaults, 10, 100) == 20  # a fifth of the calls left
    loop = [_call(i, "nothing", "k") for i in range(1, 15)]
    rungs = replay(WorkerProgress(calls=tuple(loop), calls_left=200), defaults).due
    assert [(d.response, d.rung) for d in rungs] == [
        (5, "nudge"),
        (10, "report"),
        (13, "backstop"),
    ]


def _call(
    response: int,
    tag: ProgressTag,
    key: str = "k",
    *,
    tool: str = "note",
    file: str | None = None,
) -> ProgressCall:
    return ProgressCall(
        response=response, tool=tool, tag=tag, key=key, about=f"{tool} {key}", file=file
    )


def _one_each(
    tags: Sequence[ProgressTag], key: str = "k", start: int = 1
) -> list[ProgressCall]:
    """One call a response, each with `key`."""
    return [_call(start + i, t, key) for i, t in enumerate(tags)]


def _nothing(n: int) -> list[ProgressTag]:
    return ["nothing"] * n


def _rungs(
    calls: Sequence[ProgressCall], base: WorkerProgress | None = None
) -> list[tuple[int, str, str]]:
    progress = (base or WorkerProgress()).model_copy(update={"calls": tuple(calls)})
    return [(d.response, d.rung, d.meter) for d in replay(progress, PARAMS).due]


def test_one_refusal_and_its_correction_never_fires() -> None:
    assert _rungs(_one_each(["nothing", "changed", "nothing", "learned", "changed"])) == []


def test_a_loop_of_nothing_is_nudged_then_told_to_report_then_halted() -> None:
    # The refused `note` sent 25 times: nudge at 4, report at 8, backstop 3 calls later.
    assert _rungs(_one_each(_nothing(25))) == [
        (4, "nudge", "looping"),
        (8, "report", "looping"),
        (11, "backstop", "looping"),
    ]


def test_new_results_lower_the_loop_meter_without_resetting_it() -> None:
    # A command whose output happens to vary cannot hide a loop: learned lowers by one.
    tags: list[ProgressTag] = ["nothing", "nothing", "learned", "nothing", "nothing", "nothing"]
    assert _rungs(_one_each(tags)) == [(6, "nudge", "looping")]


def test_wandering_counts_responses_and_fires_at_its_threshold() -> None:
    reads = [_call(i, "learned", f"r{i}", tool="read", file=f"f{i}.py") for i in range(1, 31)]
    assert _rungs(reads) == [
        (12, "nudge", "wandering"),
        (18, "report", "wandering"),
        (21, "backstop", "wandering"),
    ]


def test_a_continuation_read_is_one_reading_but_paging_on_still_ends() -> None:
    # Reading one long file in capped pieces, each read going on where the last stopped, is
    # not a response each; its calls still count, so 36 of them (3 x W) with nothing written
    # is wandering.
    reads = [_call(i, "learned", f"r{i}", tool="read", file="big.py") for i in range(1, 41)]
    assert _rungs(reads[:35]) == []
    assert _rungs(reads)[0] == (36, "nudge", "wandering")


def test_batching_fires_wandering_at_three_times_its_threshold_in_calls() -> None:
    one_response = [_call(1, "learned", f"g{i}", tool="grep") for i in range(36)]
    assert _rungs(one_response) == [(1, "nudge", "wandering")]
    assert _rungs(one_response[:35]) == []


def test_the_wandering_threshold_grows_with_the_brief_and_shrinks_with_the_calls_left() -> None:
    assert wandering_threshold(PARAMS, 0, 200) == 12
    assert wandering_threshold(PARAMS, 3, 200) == 18
    assert wandering_threshold(PARAMS, 9, 200) == 30
    assert wandering_threshold(PARAMS, 12, None) == 30
    assert wandering_threshold(PARAMS, 9, 100) == 20
    # A lower cap, or a late round with few calls left, lowers it: the ladder fits inside.
    assert wandering_threshold(PARAMS, 3, 40) == 8


def test_a_late_round_with_few_calls_left_reaches_its_report_before_the_cap() -> None:
    # 20 calls left: W = 4 responses, report at 6, backstop 3 calls later: 9 calls.
    reads = [_call(i, "learned", f"r{i}", tool="grep") for i in range(1, 21)]
    rungs = _rungs(reads, WorkerProgress(calls_left=20))
    assert rungs == [
        (4, "nudge", "wandering"),
        (6, "report", "wandering"),
        (9, "backstop", "wandering"),
    ]


def test_a_change_after_the_nudge_ends_the_stall_short_of_its_report() -> None:
    calls = [
        *_one_each(_nothing(4)),
        _call(5, "changed"),
        *_one_each(_nothing(2), "x", 6),
    ]
    assert _rungs(calls) == [(4, "nudge", "looping")]


def test_a_new_problem_is_nudged_again_and_the_same_one_goes_to_report() -> None:
    first = _one_each(_nothing(4), "a")
    second = _one_each(_nothing(4), "b", 6)
    again = _one_each(_nothing(4), "a", 11)
    calls = [*first, _call(5, "changed"), *second, _call(10, "changed"), *again]
    assert _rungs(calls) == [
        (4, "nudge", "looping"),
        (9, "nudge", "looping"),
        (14, "report", "looping"),
    ]


def test_the_third_stall_of_a_round_goes_straight_to_report() -> None:
    calls = [
        *_one_each(_nothing(4), "a"),
        _call(5, "changed"),
        *_one_each(_nothing(4), "b", 6),
        _call(10, "changed"),
        *_one_each(_nothing(4), "c", 11),
    ]
    assert [r for _, r, _ in _rungs(calls)] == ["nudge", "nudge", "report"]


def test_the_backstop_counts_calls_after_the_report_not_responses() -> None:
    before = _one_each(_nothing(8))
    batch = [_call(9, "learned", f"g{i}", tool="grep") for i in range(3)]
    assert _rungs([*before, *batch])[-1] == (9, "backstop", "looping")


def _firings(
    calls: Sequence[ProgressCall], base: WorkerProgress | None = None
) -> list[tuple[int, object]]:
    """The watchdog asked after each response, as the loop asks it, with the round so far."""
    hook = progress_watchdog(PARAMS)
    said: list[tuple[int, object]] = []
    for end in sorted({c.response for c in calls}):
        upto = tuple(c for c in calls if c.response <= end)
        progress = (base or WorkerProgress()).model_copy(update={"calls": upto})
        event = HookEvent(
            HookPoint.AFTER_BATCH, "worker", AfterBatch(tools=("note",), progress=progress)
        )
        verdict = hook(event)
        if not isinstance(verdict, Proceed):
            said.append((end, verdict))
    return said


def test_each_rung_is_given_once_at_its_response() -> None:
    said = _firings(_one_each(_nothing(14)))
    assert [(r, type(v).__name__) for r, v in said] == [(4, "Note"), (8, "Note"), (11, "Halt")]


def test_the_nudge_quotes_what_failed_and_the_files_it_names_not_opened() -> None:
    failure = 'File "src/segtree.py", line 41, in query\nIndexError: list index out of range'
    said = _firings(
        _one_each(_nothing(4)),
        WorkerProgress(
            last_failure=failure,
            failure_call=2,
            failure_files=("src/segtree.py",),
            opened=("tests/test_segtree.py",),
        ),
    )
    [(_, nudge)] = said
    assert isinstance(nudge, Note)
    assert nudge.text.startswith("[progress watchdog] Your last 4 calls")
    assert "note k (4 times)" in nudge.text
    assert "IndexError: list index out of range" in nudge.text
    assert "It names src/segtree.py, which you have not opened." in nudge.text


def test_the_report_asks_for_finish_and_the_backstop_for_a_plain_reply() -> None:
    _, report, backstop = (v for _, v in _firings(_one_each(_nothing(11))))
    assert isinstance(report, Note) and "call `finish`" in report.text
    assert "4. what you would try next" in report.text
    assert isinstance(backstop, Halt) and "Reply now, in plain text" in backstop.reason


def test_the_wandering_nudge_names_the_briefs_files_and_which_were_opened() -> None:
    # Two files named: W = 12 + 2·2 = 16 responses.
    reads = [_call(i, "learned", f"r{i}", tool="read", file=f"f{i}.py") for i in range(1, 17)]
    [(at, nudge)] = _firings(
        reads, WorkerProgress(brief_files=("f1.py", "src/core.py"), opened=("f1.py",))
    )
    assert at == 16
    assert isinstance(nudge, Note)
    assert "f1.py (opened), src/core.py (not opened)" in nudge.text


def _ended(
    progress: WorkerProgress | None, stopped: RoundEnd = "replied", report: str | None = "done"
) -> HookEvent:
    facts = TaskFactsP(start_tree="a", end_tree="b")
    ended = RoundEnded(
        task="T1", round=1, stopped=stopped, report=report, facts=facts, progress=progress
    )
    return HookEvent(HookPoint.ROUND_ENDED, "athena", ended)


def _stop(progress: WorkerProgress, stopped: RoundEnd = "replied") -> str:
    verdict = progress_report(PARAMS)(_ended(progress, stopped))
    assert isinstance(verdict, Note)
    return verdict.text


def test_every_report_gains_the_summary_line() -> None:
    text = _stop(WorkerProgress(calls=tuple(_one_each(["learned", "changed", "nothing"]))))
    assert text == (
        "Progress (harness): 1 change(s) to the files this round; the longest stretch "
        "without one, 1 call(s); 0 nudge(s) given."
    )


def test_a_handed_up_round_carries_what_the_harness_measured() -> None:
    progress = WorkerProgress(calls=tuple(_one_each(_nothing(9))), last_failure="boom")
    text = _stop(progress)
    assert "Measured by the harness (progress watchdog):" in text
    assert "the worker was asked to stop and report (looping)" in text
    assert "calls since the last change to the files: 9" in text
    assert "boom" not in text  # it came before the stall: no `failure_call` places it in it
    placed = progress.model_copy(update={"failure_call": 3})
    assert "boom" in _stop(placed)


def test_a_round_that_ended_with_no_reply_is_measured_too() -> None:
    # The task's time ran out mid-loop: no stop, no reply, but its reader is told.
    progress = WorkerProgress(calls=tuple(_one_each(_nothing(9))))
    text = _stop(progress, stopped="limit")
    assert "Measured by the harness (progress watchdog):" in text


def test_a_failure_before_the_last_change_is_never_quoted() -> None:
    # A test failed (call 1), the worker fixed it (call 2), then looped on something else.
    calls = [
        _call(1, "learned", "t"),
        _call(2, "changed", "e"),
        *_one_each(_nothing(4), "x", 3),
    ]
    old = WorkerProgress(last_failure="AssertionError: old", failure_call=1)
    [(_, nudge)] = _firings(calls, old)
    assert isinstance(nudge, Note) and "AssertionError: old" not in nudge.text
    new = WorkerProgress(last_failure="KeyError: new", failure_call=5)
    [(_, nudge)] = _firings(calls, new)
    assert isinstance(nudge, Note) and "KeyError: new" in nudge.text


def test_a_python_block_counts_by_the_calls_it_made() -> None:
    # Each block read ten files: 4 blocks are 40 calls, past wandering's 36.
    blocks = [
        ProgressCall(
            response=i, tool="python", tag="learned", key=f"b{i}", about="b", weight=10
        )
        for i in range(1, 5)
    ]
    assert _rungs(blocks, WorkerProgress(calls_left=200)) == [(4, "nudge", "wandering")]


def test_the_hooks_say_nothing_without_progress() -> None:
    batch = HookEvent(HookPoint.AFTER_BATCH, "athena", AfterBatch(tools=("read",)))
    assert isinstance(progress_watchdog(PARAMS)(batch), Proceed)
    assert isinstance(progress_report(PARAMS)(_ended(None)), Proceed)


def test_the_builtins_bind_at_their_points_with_their_parameters() -> None:
    watchdog = HookSpec.model_validate(
        {
            "event": "after_batch",
            "name": "progress",
            "use": "progress_watchdog",
            "role": "worker",
            "with": {"looping": 5},
        }
    )
    assert bind_builtin(watchdog).spec.use == "progress_watchdog"
    report = HookSpec.model_validate(
        {"event": "after_batch", "name": "progress", "use": "progress_report"}
    )
    with pytest.raises(ConfigError, match="fires at round_ended"):
        bind_builtin(report)
    wrong = HookSpec.model_validate(
        {"event": "after_batch", "name": "p", "use": "progress_watchdog", "with": {"L": 4}}
    )
    with pytest.raises(ConfigError, match="`with`"):
        bind_builtin(wrong)


def _entry(event: str, use: str, role: str, with_: dict[str, object]) -> HookSpec:
    return HookSpec.model_validate(
        {"event": event, "name": "progress", "use": use, "role": role, "with": with_}
    )


def test_the_report_and_the_watchdog_must_share_their_thresholds() -> None:
    # The watchdog is the worker's; the report is read at Athena's desk.
    watchdog = _entry("after_batch", "progress_watchdog", "worker", {"looping": 5})
    same = _entry("round_ended", "progress_report", "athena", {"looping": 5})
    assert len(bind((watchdog, same), accepted=False, scripts=None).hooks) == 3
    other = _entry("round_ended", "progress_report", "athena", {})
    with pytest.raises(ConfigError, match="give both the same `with`"):
        bind((watchdog, other), accepted=False, scripts=None)


def test_a_worker_nudged_for_a_short_loop_gets_wanderings_room_to_investigate() -> None:
    # Nudged after 4 calls that did nothing, it reads new files to find the cause and writes
    # nothing: it is handed up when a worker that only read would be (18 responses), not 6
    # responses after the nudge.
    loop = _one_each(_nothing(4))
    reads = [
        _call(4 + i, "learned", f"r{i}", tool="read", file=f"f{i}.py") for i in range(1, 17)
    ]
    rungs = _rungs([*loop, *reads], WorkerProgress(calls_left=200))
    assert rungs[:2] == [(4, "nudge", "looping"), (18, "report", "looping")]


def test_every_text_the_worker_reads_begins_with_the_watchdogs_mark() -> None:
    """The progress record knows the watchdog's words on an answer by this mark
    (`box/progress._text`): a template that dropped it would be read as the tool's output."""
    for name in (
        "progress_nudge_looping",
        "progress_nudge_wandering",
        "progress_report",
        "progress_backstop",
    ):
        assert _template(name).startswith(HEAD), name


def _template(name: str) -> str:
    from chickadee.adapters.base import SHARED_TEMPLATES, load_template

    return load_template(SHARED_TEMPLATES / "hooks", f"{name}.md")


# -- stage 2 ---------------------------------------------------------


def _edit(response: int, file: str = "app.py") -> ProgressCall:
    return ProgressCall(
        response=response,
        tool="edit",
        tag="changed",
        key=f"e{response}",
        about=f"edit {file}",
        changed=(file,),
    )


def _test_run(response: int, **seen: object) -> ProgressCall:
    run = TestRunSeen.model_validate({"outcome": "failed", "how": "summary", **seen})
    return ProgressCall(
        response=response,
        tool="shell",
        tag="learned",
        key=f"t{response}",
        about="pytest",
        run=run,
    )


def _batch(progress: WorkerProgress) -> HookEvent:
    return HookEvent(
        HookPoint.AFTER_BATCH, "worker", AfterBatch(tools=("shell",), progress=progress)
    )


def test_unchecked_counts_from_the_first_change_after_the_last_test_run() -> None:
    calls = [
        _edit(1),
        _test_run(2),
        _call(3, "learned", "r"),
        _edit(4, "a.py"),
        _edit(5, "b.py"),
    ]
    assert unchecked(calls) == unchecked(calls[3:])
    found = unchecked(calls)
    assert (found.responses, found.calls, found.files) == (2, 2, ("a.py", "b.py"))
    assert unchecked([_edit(1), _test_run(2)]).responses == 0


def test_changes_without_a_test_run_are_nudged_then_reported() -> None:
    """25 responses of changes with a runner there and no run is a nudge, 38
    the report; a change neither ends the stall nor feeds the other meters."""
    calls = tuple(_edit(i) for i in range(1, 40))
    due = replay(WorkerProgress(calls=calls, runnable=True), WatchdogParams()).due
    assert [(d.response, d.rung, d.meter) for d in due] == [
        (25, "nudge", "unchecked"),
        (38, "report", "unchecked"),
    ]
    assert "25 of your responses have changed files without a test run" in _nudge_text(calls)


def _nudge_text(calls: Sequence[ProgressCall]) -> str:
    progress = WorkerProgress(calls=tuple(calls[:25]), runnable=True)
    said = progress_watchdog(WatchdogParams())(_batch(progress))
    assert isinstance(said, Note)
    return said.text


def test_no_runner_or_no_shell_means_no_unchecked_meter() -> None:
    calls = tuple(_edit(i) for i in range(1, 40))
    assert replay(WorkerProgress(calls=calls, runnable=False), WatchdogParams()).due == ()


def test_a_test_run_ends_an_unchecked_stall_short_of_its_report() -> None:
    calls = [
        *(_edit(i) for i in range(1, 26)),
        _test_run(26),
        *(_edit(i) for i in range(27, 40)),
    ]
    due = replay(WorkerProgress(calls=tuple(calls), runnable=True), WatchdogParams()).due
    assert [(d.response, d.rung) for d in due] == [(25, "nudge")]


def test_a_confirmed_regression_rides_on_the_run_that_confirmed_it() -> None:
    regressed = Regression(
        test="test_route", passed_at="call 9 of round 1", changed=("api.py",)
    )
    run = _test_run(3).model_copy(update={"regressed": (regressed,)})
    progress = WorkerProgress(calls=(_edit(1), _test_run(2), run))
    said = progress_watchdog(PARAMS)(_batch(progress))
    assert isinstance(said, Note)
    assert said.text == (
        f"{HEAD} test_route passed at call 9 of round 1 and still fails after your edits "
        "since: api.py. Look at what those edits changed for it before you go on."
    )


def test_the_report_s_reader_is_told_the_task_s_tests() -> None:
    tests = TestsSeen(
        fixed=("t::a", "t::b"),
        broken=("t::c",),
        failing=("t::d",),
        already_failing=("t::e",),
        edited=("t::b",),
    )
    progress = WorkerProgress(calls=(_edit(1), _test_run(2)), tests=tests)
    said = progress_report(PARAMS)(_ended(progress))
    assert isinstance(said, Note)
    first, second = said.text.splitlines()[:2]
    assert first.endswith("0 nudge(s) given; tests fixed 2, broken 1.")
    assert second == (
        "Tests (harness, read from what the runs printed): fixed: t::a, t::b (its file changed "
        "by the task); broken: t::c; last seen failing: t::d; failing before the task touched "
        "them, not counted against it: t::e."
    )
