"""`Hooks.fire`: what it runs, what it journals, how a failure
is handled, and what a bound counts.

What these pin: no hook bound is no line and no answer; a hook that does not match is not run; a
silent `Proceed` writes no line and every other verdict one, before it is acted on; a hook that
raises or answers what its point does not take has failed, `on_error = "allow"` going on with a
warning and `"deny"` failing closed as the point's blocking verdict; `used` counts from the
journal, never a bounded line.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from chickadee.hooks.fire import BoundHook, HookError, Hooks, HookSet
from chickadee.hooks.payloads import BeforeStop, BeforeTool, HookEvent
from chickadee.hooks.points import HookPoint
from chickadee.hooks.sinks import RunHookSink
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Continue, Deny, HookVerdict, Note, Proceed, RetryTask, Warn
from chickadee.journal.projection import Placed
from chickadee.journal.writer import Journal
from chickadee.schema.journal import HookP
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-10-24-0a1b2c3d"


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    return Journal.create(resolve_roots(tmp_path), RUN, {})


def _hook(name: str, verdict: HookVerdict | Exception, **kw: object) -> BoundHook:
    fields: dict[str, object] = {"event": "before_tool", "name": name, "use": name}
    fields.update(kw)
    spec = HookSpec.model_validate(fields)

    def run(event: HookEvent) -> HookVerdict:
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    return BoundHook(spec, run, digest=name)


def _tool_event() -> HookEvent:
    payload = BeforeTool(tool="read", tool_sent="read", input={"path": "a.py"})
    return HookEvent(HookPoint.BEFORE_TOOL, "worker", payload, run_id=RUN, call_id="c1")


def _lines(journal: Journal) -> list[HookP]:
    return [p.payload for p in journal.projection().lines(HookP)]


def test_no_hook_bound_is_no_answer_and_no_line(journal: Journal) -> None:
    hooks = Hooks(HookSet(), RunHookSink(journal))
    assert not hooks.has(HookPoint.BEFORE_TOOL)
    got = hooks.fire(_tool_event())
    assert got.decided is None and got.notes == () and _lines(journal) == []
    assert Hooks.none().fire(_tool_event()).decided is None


def test_a_pass_writes_no_line_and_a_verdict_writes_one(journal: Journal) -> None:
    bound = HookSet((_hook("quiet", Proceed()), _hook("says", Note("look at b.py"))))
    got = Hooks(bound, RunHookSink(journal)).fire(_tool_event())
    # A built-in's words are the harness's own: never labelled (a script's are).
    assert got.notes == ("look at b.py",)
    (line,) = _lines(journal)
    assert (line.event, line.name, line.decision, line.role, line.call_id) == (
        "before_tool",
        "says",
        "note",
        "worker",
        "c1",
    )
    assert line.text_sha is not None and journal.get_blob(line.text_sha) == b"look at b.py"
    assert line.input_sha is not None


def test_a_hook_that_does_not_match_is_not_run(journal: Journal) -> None:
    ran: list[str] = []

    def run(event: HookEvent) -> HookVerdict:
        ran.append(event.point)
        return Deny("no")

    spec = HookSpec(event=HookPoint.BEFORE_TOOL, name="athena-only", use="x", role="athena")
    got = Hooks(HookSet((BoundHook(spec, run, "d"),)), RunHookSink(journal)).fire(_tool_event())
    assert ran == [] and got.decided is None


def test_a_verdict_the_point_does_not_take_is_a_failure(journal: Journal) -> None:
    bound = HookSet((_hook("wrong", RetryTask("again")),))
    got = Hooks(bound, RunHookSink(journal)).fire(_tool_event())
    assert got.decided is None
    assert got.warns and "does not take" in got.warns[0]
    (line,) = _lines(journal)
    assert line.decision == "error" and line.error is not None and "retry" in line.error


def test_on_error_allow_goes_on_and_deny_fails_closed(journal: Journal) -> None:
    allowed = Hooks(HookSet((_hook("flaky", HookError("exit 3")),)), RunHookSink(journal))
    assert allowed.fire(_tool_event()).decided is None
    closed = Hooks(
        HookSet((_hook("guard", HookError("timed out", timed_out=True), on_error="deny"),)),
        RunHookSink(journal),
    )
    decided = closed.fire(_tool_event()).decided
    assert isinstance(decided, Deny) and "hook guard failed: timed out" in decided.reason
    assert [line.timed_out for line in _lines(journal)] == [False, True]


def test_a_builtin_that_raises_is_the_hooks_failure_never_the_runs(journal: Journal) -> None:
    hooks = Hooks(HookSet((_hook("buggy", ValueError("bad")),)), RunHookSink(journal))
    got = hooks.fire(_tool_event())
    assert got.decided is None and "ValueError: bad" in got.warns[0]


def test_used_counts_from_the_journal_and_never_a_bounded_line(journal: Journal) -> None:
    sink = RunHookSink(journal)
    spec = HookSpec(event=HookPoint.BEFORE_STOP, name="again", use="x")
    hooks = Hooks(HookSet((BoundHook(spec, lambda e: Continue("not yet"), "d"),)), sink)
    stop = HookEvent(
        HookPoint.BEFORE_STOP, "worker", BeforeStop(reply="done"), task_id="T1", round=1
    )
    assert isinstance(hooks.fire(stop).decided, Continue)
    # Past its bound: one line, marked, and the verdict is not taken.
    assert hooks.fire(stop, bounded=frozenset({Continue})).decided is None
    assert [line.bounded for line in _lines(journal)] == [False, True]
    # A fresh `Hooks` over the same journal: what a resumed run reads.
    again = Hooks(hooks.bound, RunHookSink(journal))
    used = again.used(HookPoint.BEFORE_STOP, Continue, role="worker", task_id="T1", round=1)
    assert used == 1
    assert again.used(HookPoint.BEFORE_STOP, Continue, role="worker", round=2) == 0
    assert again.used(HookPoint.BEFORE_STOP, Continue, role="athena") == 0
    assert again.used(HookPoint.BEFORE_STOP, Warn, role="worker") == 0


def test_the_sets_digest_moves_with_its_hooks() -> None:
    one = HookSet((_hook("a", Proceed()),))
    two = HookSet((_hook("a", Proceed()), _hook("b", Proceed())))
    assert one.digest != two.digest and one.digest == HookSet((_hook("a", Proceed()),)).digest


def _stop(round_: int = 1) -> HookEvent:
    return HookEvent(
        HookPoint.BEFORE_STOP, "worker", BeforeStop(reply="done"), task_id="T1", round=round_
    )


def _at_stop(name: str, verdict: HookVerdict | Exception, **kw: object) -> BoundHook:
    return _hook(name, verdict, event="before_stop", **kw)


def test_a_failure_that_fails_closed_is_held_and_counted_as_its_verdict(
    journal: Journal,
) -> None:
    """Review finding: `on_error = "deny"` at a stop is a `Continue`; past the bound it is not
    taken, and below it is counted, as a hook's own `Continue` is."""
    hooks = Hooks(
        HookSet((_at_stop("guard", HookError("exit 1"), on_error="deny"),)),
        RunHookSink(journal),
    )
    assert isinstance(hooks.fire(_stop()).decided, Continue)
    assert hooks.fire(_stop(), bounded=frozenset({Continue})).decided is None
    assert [(line.decision, line.bounded, line.error) for line in _lines(journal)] == [
        ("continue", False, "exit 1"),
        ("continue", True, "exit 1"),
    ]
    assert hooks.used(HookPoint.BEFORE_STOP, Continue, role="worker", task_id="T1") == 1


def test_one_firing_is_counted_once_however_many_hooks_answered(journal: Journal) -> None:
    """Review finding: two hooks' `Continue`s are one `Continue` taken: one use of the bound."""
    hooks = Hooks(
        HookSet((_at_stop("a", Continue("one")), _at_stop("b", Continue("two")))),
        RunHookSink(journal),
    )
    said = hooks.fire(_stop())
    assert isinstance(said.decided, Continue) and len(_lines(journal)) == 2
    assert hooks.used(HookPoint.BEFORE_STOP, Continue, role="worker", task_id="T1") == 1


def test_the_steering_bound_holds_only_the_hooks_that_steer(journal: Journal) -> None:
    """Review finding: the harness's own reminder (`owed_review`) is neither held by
    `max_steering` nor counted against it; a person's hook is both."""
    own = replace(_at_stop("owed", Continue("review T4 first")), steers=False)
    theirs = _at_stop("mine", Continue("run the suite"))
    hooks = Hooks(HookSet((own, theirs)), RunHookSink(journal))
    said = hooks.fire(_stop(), steering=frozenset({Continue}))
    assert isinstance(said.decided, Continue) and said.decided.text == "review T4 first"
    assert [(line.name, line.bounded) for line in _lines(journal)] == [
        ("owed", False),
        ("mine", True),
    ]
    assert hooks.used(HookPoint.BEFORE_STOP, Continue, role="worker", steering=True) == 0


def test_bounds_are_counted_from_the_lines_read_once(journal: Journal) -> None:
    """Review finding: a count reads the sink once and then the lines `Hooks` keeps as it
    writes, never the whole journal again for each decision."""
    sink = RunHookSink(journal)
    read: list[int] = []
    real = sink.lines

    def counted() -> tuple[Placed[HookP], ...]:
        read.append(1)
        return real()

    sink.lines = counted  # type: ignore[method-assign]
    hooks = Hooks(HookSet((_at_stop("again", Continue("not yet")),)), sink)
    for round_ in (1, 2, 3):
        hooks.fire(_stop(round_))
        hooks.used(HookPoint.BEFORE_STOP, Continue, role="worker", task_id="T1")
    assert read == [1]
    assert hooks.used(HookPoint.BEFORE_STOP, Continue, role="worker", task_id="T1") == 3


def test_firings_from_two_threads_are_one_at_a_time(journal: Journal) -> None:
    """Review finding: a job's waiter and the turn share a session's `Hooks`; one firing's
    hooks never interleave with another's."""
    inside = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def slow(event: HookEvent) -> HookVerdict:
        order.append("slow in")
        inside.set()
        assert release.wait(5)
        order.append("slow out")
        return Note("slow")

    def quick(event: HookEvent) -> HookVerdict:
        order.append("quick")
        return Note("quick")

    hooks = Hooks(
        HookSet(
            (
                replace(_hook("slow", Note("x")), run=slow),
                replace(_at_stop("quick", Note("y")), run=quick),
            )
        ),
        RunHookSink(journal),
    )
    first = threading.Thread(target=hooks.fire, args=(_tool_event(),))
    first.start()
    assert inside.wait(5)
    second = threading.Thread(target=hooks.fire, args=(_stop(),))
    second.start()
    second.join(0.2)
    release.set()
    first.join(5)
    second.join(5)
    assert order == ["slow in", "slow out", "quick"]
