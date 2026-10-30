"""Each shared rule, result rule and wrapper against its parameters. An observer's exception
never failing a call is `test_dispatch.py`'s."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace

import pytest
from refusals import stated

from chickadee.context.results import ToolResult
from chickadee.errors import InfraError
from chickadee.journal.effects import EffectKind
from chickadee.schema.authorization import Tier
from chickadee.schema.journal import EventKind, P
from chickadee.tools.catalogue import (
    Account,
    AskWhen,
    BoundText,
    CallCap,
    Dedup,
    NoProgress,
    Offered,
    PerResponse,
    Redact,
    Repetition,
    ResultFits,
    ResultTokenCap,
    Untrusted,
    Valid,
    answer_text,
    capped,
)
from chickadee.tools.context import ToolContext
from chickadee.tools.rules import (
    Answered,
    AskPerson,
    BoundReached,
    CallOutcome,
    Noted,
    Pass,
    Pending,
    Pipeline,
    Refuse,
    Stop,
)
from chickadee.tools.tool import Answer, CallTimedOut, Invocation, TextAnswer, Tool
from chickadee.tools.wrappers import EffectLifecycle, RetryIdempotent, Timeout

from .tools_helpers import DictFiles, call, dispatcher, echo_tool, make_handle, make_spec


def _caps(*admit: object, result: tuple[object, ...] = (), **limits: int) -> Pipeline:
    return Pipeline(
        admit=(*admit, Valid()),  # type: ignore[arg-type]
        result=result,  # type: ignore[arg-type]
        counts_at=("turn",),
    )


def _decided(outcome: CallOutcome) -> object:
    return outcome.pending.decided


# -- admit ---------------------------------------------------------------------------------


def test_call_cap_refuses_from_the_cap_plus_first_without_counting() -> None:
    d = dispatcher([echo_tool()], pipeline=_caps(CallCap("turn", "cap")), limits={"cap": 2})
    outcomes = [d.call(call(cid=str(i))) for i in range(3)]
    assert [o.answer is not None for o in outcomes] == [True, True, False]
    refusal = _decided(outcomes[2])
    assert isinstance(refusal, Refuse) and not refusal.counts
    assert refusal.failure == BoundReached("tool_calls", 2, 2, "turn")
    assert d.state.calls("turn") == 2


def test_a_call_cap_of_zero_never_refuses() -> None:
    d = dispatcher([echo_tool()], pipeline=_caps(CallCap("turn", "cap")), limits={"cap": 0})
    assert all(d.call(call(cid=str(i))).answer is not None for i in range(5))


def test_a_call_cap_spares_its_exempt_families() -> None:
    # Athena's task tools: past her cap a read is refused, and a task
    # tool still runs, so she can always delegate, review or answer.
    read = echo_tool("look")
    plan = echo_tool("plan")
    plan = Tool(replace(plan.spec, family="task"), plan.needs, plan.run)
    cap = CallCap("turn", "cap", exempt=frozenset({"task"}))
    d = dispatcher([read, plan], pipeline=_caps(cap), limits={"cap": 1})
    assert d.call(call(name="look", cid="1")).answer is not None
    assert isinstance(_decided(d.call(call(name="look", cid="2"))), Refuse)
    assert d.call(call(name="plan", cid="3")).answer is not None


def test_a_call_cap_can_stop_the_loop_instead() -> None:
    d = dispatcher(
        [echo_tool()], pipeline=_caps(CallCap("turn", "cap", on_cap="stop")), limits={"cap": 1}
    )
    d.call(call(cid="1"))
    second = d.call(call(cid="2"))
    assert isinstance(_decided(second), Stop) and d.stopped is not None


@pytest.mark.parametrize("stage", ["admit", "dispatch"])
def test_result_token_cap_refuses_once_the_allowance_is_spent(stage: str) -> None:
    cap = ResultTokenCap("turn", "tokens", stage=stage)  # type: ignore[arg-type]
    pipeline = (
        _caps(cap, result=(Account(("turn",)),))
        if stage == "admit"
        else Pipeline(
            admit=(Valid(),), dispatch=(cap,), result=(Account(("turn",)),), counts_at=("turn",)
        )
    )
    d = dispatcher([echo_tool()], pipeline=pipeline, limits={"tokens": 1})
    assert d.call(call(cid="1")).answer is not None
    refused = d.call(call(cid="2"))
    decided = _decided(refused)
    assert isinstance(decided, Refuse) and decided.failure.resource == "result_tokens"  # type: ignore[attr-defined]


def test_per_response_refuses_a_second_exclusive_and_capped_past_the_fanout() -> None:
    d = dispatcher(
        [echo_tool("coder", parallel="exclusive"), echo_tool("locate", parallel="capped")],
        pipeline=_caps(PerResponse("fanout")),
        limits={"fanout": 2},
    )
    pending = d.admit(
        [call("coder", cid="1"), call("coder", cid="2")]
        + [call("locate", cid=str(i)) for i in range(3, 6)]
    )
    refused = [p.call.id for p in pending if isinstance(p.decided, Refuse)]
    assert refused == ["2", "5"]
    assert d.state.calls("turn") == 3, "a per-response refusal does not count"
    assert not any(isinstance(p.decided, Refuse) for p in d.admit([call("coder", cid="6")]))


def test_repetition_notes_at_n_and_refuses_at_2n_and_resets_on_another_call() -> None:
    d = dispatcher(
        [echo_tool(), echo_tool("other")],
        pipeline=_caps(Repetition("turn", "repeat")),
        limits={"repeat": 2},
    )
    verdicts = [d.call(call(cid=str(i))).pending.verdicts for i in range(4)]
    assert not any(isinstance(v, Noted) for v in verdicts[0])
    assert any(isinstance(v, Noted) for v in verdicts[1])
    assert isinstance(verdicts[3][-1], Refuse)
    d.call(call("other", cid="x"))
    assert not any(isinstance(v, Noted) for v in d.call(call(cid="y")).pending.verdicts)


def test_repetition_tells_two_unoffered_names_apart() -> None:
    """Two calls to names this scope does not offer resolve to no tool; what the model sent
    is what makes them the same call or not. Keyed by the missing tool alone, five different
    unknown names read as one call repeated, and the fourth stopped the worker."""
    d = dispatcher(
        [echo_tool()], pipeline=_caps(Repetition("turn", "repeat")), limits={"repeat": 2}
    )
    verdicts = [
        d.call(call(name, cid=name)).pending.verdicts
        for name in ("shell", "mcp", "write", "exec", "apply")
    ]
    for vs in verdicts:
        assert not any(isinstance(v, Noted) for v in vs)
        bound = (v for v in vs if isinstance(v, Refuse))
        assert not any(isinstance(v.failure, BoundReached) for v in bound)
    same = [d.call(call("shell", cid=f"s{i}")).pending.verdicts for i in range(2)]
    assert any(isinstance(v, Noted) for v in same[1])


def test_dedup_answers_a_repeated_call_with_a_marker_and_counts_it() -> None:
    runs: list[str] = []
    d = dispatcher([echo_tool(runs=runs)], pipeline=_caps(Offered(), Dedup("turn")))
    d.call(call(cid="1"))
    again = d.call(call(cid="2"))
    assert runs == ["1"]
    assert answer_text(again) == "[dedup: the same echo call as call #1; nothing re-run]"
    assert d.state.calls("turn") == 2


def test_confirm_when_asks_only_above_t0() -> None:
    spec = make_spec()
    pending = Pending(Invocation("c", "c", 0), "echo", {"text": "x"}, "echo", spec)
    asks = AskWhen(lambda s: True, lambda p: Tier.T1)
    quiet = AskWhen(lambda s: True, lambda p: Tier.T0)
    from chickadee.tools.rules import ScopeState

    state = ScopeState({})
    assert asks.check(pending, state) == AskPerson("run echo?", Tier.T1)
    assert quiet.check(pending, state) == Pass()


# -- result ---------------------------------------------------------------------------------


def test_account_adds_what_a_counted_call_took_in() -> None:
    d = dispatcher([echo_tool()], pipeline=_caps(result=(Account(("turn",), lambda o: 7),)))
    d.call(call())
    d.call(call(text=""))  # invalid: still counted, still accounted
    assert d.state.tokens("turn") == 14


def test_account_served_only_skips_a_refusal() -> None:
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(result=(Account(("turn",), lambda o: 7, served_only=True),)),
    )
    d.call(call(text=""))
    assert d.state.tokens("turn") == 0


def test_result_fits_withholds_a_result_past_the_allowance_and_stops() -> None:
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(
            result=(
                ResultFits("turn", "tokens", lambda o: 10),
                Account(("turn",), lambda o: 10),
            )
        ),
        limits={"tokens": 15},
    )
    first = d.call(call(cid="1"))
    second = d.call(call(cid="2"))
    assert first.answer is not None and second.answer is None
    assert second.stop is not None and d.stopped is not None
    assert d.state.tokens("turn") == 10
    assert answer_text(second).startswith("BOUND_REACHED{resource: result_tokens, used: 20")


def test_result_fits_keeps_what_it_withheld_for_the_role_s_record() -> None:
    """The scout's `scout_tool` line records what a withheld result measured."""
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(result=(ResultFits("turn", "tokens", lambda o: 10),)),
        limits={"tokens": 5},
    )
    withheld = d.call(call(text="kept"))
    assert withheld.answer is None and withheld.stop is not None
    assert isinstance(withheld.withheld, TextAnswer)
    assert withheld.withheld.text == "echo: kept"


def test_result_fits_judges_a_refusal_s_text_too() -> None:
    """The scout's loop compared every counted answer with what remains, a refusal's
    included: an unknown name past the allowance ends the search as a result would."""
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(Offered(), result=(ResultFits("turn", "tokens", lambda o: 10),)),
        limits={"tokens": 5},
    )
    outcome = d.call(call(name="nope"))
    assert isinstance(outcome.pending.decided, Refuse)
    assert outcome.stop is not None and d.stopped is not None


def test_offered_names_the_nearest_or_lists_the_offered_in_the_role_s_form() -> None:
    zeus = dispatcher([echo_tool()], pipeline=_caps(Offered()))
    assert stated(answer_text(zeus.call(call(name="ecco")))) == (
        "UNKNOWN_TOOL{name: 'ecco', nearest: 'echo'}"
    )
    scout = dispatcher([echo_tool()], pipeline=_caps(Offered(show="offered")))
    assert stated(answer_text(scout.call(call(name="ecco")))) == (
        "UNKNOWN_TOOL{name: 'ecco', offered: [echo]}"
    )


def test_no_progress_reads_its_streak() -> None:
    rule = NoProgress("turn", "n")
    d = dispatcher(
        [echo_tool()], pipeline=_caps(Dedup("turn"), result=(rule,)), limits={"n": 9}
    )
    for cid in ("1", "2", "3"):
        d.call(call(cid=cid))
    assert rule.streak(d.state) == 2
    d.call(call(text="other", cid="4"))
    assert rule.streak(d.state) == 0


def test_no_progress_stops_after_n_repeated_answers() -> None:
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(Dedup("turn"), result=(NoProgress("turn", "n"),)),
        limits={"n": 2},
    )
    d.call(call(cid="1"))
    assert d.call(call(cid="2")).stop is None
    assert d.call(call(cid="3")).stop is not None


def test_bound_cuts_text_and_keeps_the_rest_behind_a_handle() -> None:
    files = DictFiles({})
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(result=(BoundText("bytes", "lines"),)),
        limits={"bytes": 8, "lines": 1},
        ctx=ToolContext(role="test", handles=files),
    )
    outcome = d.call(call(text="a long answer\nsecond line"))
    assert isinstance(outcome.answer, ToolResult)
    assert outcome.answer.truncation is not None
    assert files.stored, "the whole answer is stored behind the handle"


def test_bound_leaves_a_result_its_producer_bounded() -> None:
    result = ToolResult(make_handle("file", "a.py"), content="x" * 100)
    outcome = CallOutcome(Pending(Invocation("c", "c", 0), "read", {}, "read", None), result)
    from chickadee.tools.rules import ScopeState

    out = BoundText("b", "l").apply(
        outcome, ScopeState({"b": 1, "l": 1}), ToolContext(role="t")
    )
    assert out.answer is result


def test_a_sent_image_counts_against_a_result_budget_as_the_request_counts_it() -> None:
    """A result's tokens are its text and the image it sends, by the one image rule
    (`context.outline.image_est`), not its one-line header alone."""
    import io

    from PIL import Image

    from chickadee.client.request import ImagePart
    from chickadee.context.outline import image_est
    from chickadee.tools.catalogue import result_tokens

    png = io.BytesIO()
    Image.new("RGB", (800, 600), (1, 2, 3)).save(png, format="PNG")
    pending = Pending(Invocation("c", "c", 0), "read", {}, "read", None)
    text_only = ToolResult(make_handle("file", "shot.png"), header="shot.png")
    shown = ToolResult(
        text_only.handle,
        header="shot.png",
        image=ImagePart("image/png", "s", png.getvalue()),
    )
    plain = result_tokens(CallOutcome(pending, text_only))
    with_image = result_tokens(CallOutcome(pending, shown))
    known = text_only.handle.shape.est_tokens
    assert with_image - plain == image_est(png.getvalue(), known)
    assert with_image - plain > 100, "an 800x600 image is not a one-line header"


def test_capped_by_lines_and_by_bytes() -> None:
    handle = make_handle("findings", "x")
    assert capped("a\nb\n", handle, limit=5, unit="lines") == ("a\nb\n", None)
    text, cut = capped("a\nb\nc\n", handle, limit=2, unit="lines")
    assert text == "a\nb\n" and cut is not None and cut.reason == "result_cap"
    text, cut = capped("aaaa\nbbbb\n", handle, limit=6, unit="bytes")
    assert text == "aaaa\n" and cut is not None and cut.reason == "byte_cap"


def test_redact_removes_a_secret_from_what_the_model_sees() -> None:
    secret = "ghp_" + "a" * 36

    def leak(c: Invocation, a: object, ctx: ToolContext) -> Answer:
        return TextAnswer(f"token={secret}", "bytes")

    from chickadee.tools.tool import Tool

    d = dispatcher(
        [Tool(make_spec(), frozenset(), leak)],  # type: ignore[arg-type]
        pipeline=_caps(result=(Redact(),)),
    )
    shown = answer_text(d.call(call()))
    assert secret not in shown


def test_untrusted_frames_the_answer_through_the_wrapper_it_is_given() -> None:
    d = dispatcher(
        [echo_tool()],
        pipeline=_caps(
            result=(Untrusted(lambda text, source: f"<{source}>{text}</{source}>"),)
        ),
    )
    assert answer_text(d.call(call())) == "<echo>echo: hi</echo>"


# -- around ---------------------------------------------------------------------------------


@dataclass
class Log:
    lines: list[tuple[EventKind, P]] = field(default_factory=list[tuple[EventKind, P]])

    def record(self, kind: EventKind, payload: P) -> int:
        self.lines.append((kind, payload))
        return len(self.lines)


def test_effect_records_the_lifecycle_of_a_dispatcher_effect() -> None:
    log = Log()
    writes = echo_tool("write", mutates="environment", effects="dispatcher", idempotent=False)
    d = dispatcher(
        [writes],
        pipeline=Pipeline(
            admit=(Valid(),),
            around=(EffectLifecycle(EffectKind.WORKER_STEP),),
            counts_at=("turn",),
        ),
        ctx=ToolContext(role="test", effect_log=log),
    )
    d.call(call("write"))
    assert [k for k, _ in log.lines] == [
        EventKind.EFFECT_INTENDED,
        EventKind.EFFECT_STARTED,
        EventKind.EFFECT_COMPLETED,
    ]


def test_effect_passes_through_a_tool_that_changes_nothing() -> None:
    log = Log()
    d = dispatcher(
        [echo_tool()],
        pipeline=Pipeline(
            admit=(Valid(),),
            around=(EffectLifecycle(EffectKind.WORKER_STEP),),
            counts_at=("turn",),
        ),
        ctx=ToolContext(role="test", effect_log=log),
    )
    d.call(call())
    assert log.lines == []


def test_a_mutating_spec_without_an_effect_lifecycle_cannot_exist() -> None:
    with pytest.raises(ValueError, match="no effect lifecycle"):
        make_spec(mutates="environment", effects="none")


def test_timeout_sets_the_cancel_event_and_answers_timed_out() -> None:
    cancel = threading.Event()

    def waits(c: Invocation, a: object, ctx: ToolContext) -> Answer:
        assert ctx.cancel is not None
        ctx.cancel.wait(5)
        return TextAnswer("finished anyway", "bytes")

    from chickadee.tools.tool import Tool

    d = dispatcher(
        [Tool(make_spec(), frozenset(), waits)],  # type: ignore[arg-type]
        pipeline=Pipeline(admit=(Valid(),), around=(Timeout(0.05),), counts_at=("turn",)),
        ctx=ToolContext(role="test", cancel=cancel),
    )
    outcome = d.call(call())
    assert isinstance(outcome.answer, CallTimedOut) and cancel.is_set()


def test_a_quick_run_beats_its_timeout() -> None:
    cancel = threading.Event()
    d = dispatcher(
        [echo_tool()],
        pipeline=Pipeline(admit=(Valid(),), around=(Timeout(5),), counts_at=("turn",)),
        ctx=ToolContext(role="test", cancel=cancel),
    )
    assert isinstance(d.call(call()).answer, TextAnswer) and not cancel.is_set()


@pytest.mark.parametrize(("idempotent", "runs"), [(True, 3), (False, 1)])
def test_retry_reruns_only_an_idempotent_spec(idempotent: bool, runs: int) -> None:
    tries: list[int] = []

    def flaky(c: Invocation, a: object, ctx: ToolContext) -> Answer:
        tries.append(1)
        if len(tries) < 3:
            raise InfraError("the index is busy")
        return TextAnswer("ok", "bytes")

    from chickadee.tools.tool import Tool

    d = dispatcher(
        [Tool(make_spec(idempotent=idempotent), frozenset(), flaky)],  # type: ignore[arg-type]
        pipeline=Pipeline(admit=(Valid(),), around=(RetryIdempotent(2),), counts_at=("turn",)),
    )
    if idempotent:
        assert isinstance(d.call(call()).answer, TextAnswer)
    else:
        with pytest.raises(InfraError):
            d.call(call())
    assert len(tries) == runs


def test_verdicts_that_answer_are_the_ones_that_decide() -> None:
    pending = Pending(Invocation("c", "c", 0), "echo", {}, "echo", None)
    pending = pending.with_verdict(Noted("n")).with_verdict(Answered("x"))
    assert isinstance(pending.decided, Answered) and pending.notes == ("n",)


def test_repetition_tells_the_model_what_the_role_says() -> None:
    """Zeus's note names the generation its identity carries."""
    rule = Repetition(
        "turn",
        "repeat",
        identity=lambda p: (p.tool, 7),
        note=lambda count, identity: f"repetition{{count: {count}, since_generation: 7}}",
    )
    d = dispatcher([echo_tool()], pipeline=_caps(rule), limits={"repeat": 2})
    d.call(call(cid="1"))
    assert d.call(call(cid="2")).pending.notes == ("repetition{count: 2, since_generation: 7}",)


# -- Clip ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("per", "total", "used", "kept"),
    [
        (5, 100, 0, 5),
        (50, 20, 12, 8),
        (0, 20, 12, 8),
        (5, 0, 0, 5),
        (0, 0, 0, None),
    ],
    ids=["per-result", "what-is-left", "no-per-bound", "no-total-bound", "unbounded"],
)
def test_clip_cuts_to_the_smaller_bound_and_zero_is_no_bound(
    per: int, total: int, used: int, kept: int | None
) -> None:
    """The smaller of the per-result bound and what the scope has left; a bound of 0 is no
    bound, as `ResultTokenCap` reads it, never a cut to nothing."""
    from chickadee.tools.catalogue import Clip, truncate
    from chickadee.tools.rules import CallOutcome, Pending, ScopeState
    from chickadee.tools.tool import Invocation, TextAnswer

    text = "x" * 1000
    state = ScopeState({"per": per, "total": total})
    state.took("attempt", used)
    pending = Pending(Invocation("c", "c", 0), "read", {}, "read", None)
    outcome = CallOutcome(pending, TextAnswer(text, "bytes"))
    clipped = Clip("attempt", "per", "total").apply(outcome, state, ToolContext(role="t"))
    expected = text if kept is None else truncate(text, kept)
    assert isinstance(clipped.answer, TextAnswer) and clipped.answer.text == expected
