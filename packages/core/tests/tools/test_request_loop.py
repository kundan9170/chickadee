"""`tools/loop.py`: one tool-using request to its answer, for any role.

The scout's behaviour on this loop is pinned byte for byte in `tests/scout/test_loop_pins.py`.
These tests hold the loop's own rules with a role that knows nothing else: the re-ask bound is
the role's, a turn cut at `max_tokens` is never dispatched, every call of a turn is answered, a
call that ends tool use makes the next turn the answer, calls after the end are answered once,
and cancel and the spend gate are read before every call.
"""

from __future__ import annotations

import threading

from chickadee.client.errors import RequestCancelled, TransportExhausted
from chickadee.client.request import ChatRequest, Message
from chickadee.schema import RetryParams
from chickadee.schema.adapter import Completion, ParseError, StopReason, ToolCall, Usage
from chickadee.tools.loop import CUT, AskResult, Messages, ask, kept

from .tools_helpers import ScriptedRole


def _usage() -> Usage:
    return Usage(prompt_tokens=10, completion_tokens=2)


def _calls(*names: str) -> Completion:
    return Completion(
        text="",
        usage=_usage(),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=tuple(
            ToolCall(id=f"c{i}", name=n, arguments={}) for i, n in enumerate(names)
        ),
    )


def _text(text: str, stop: StopReason = StopReason.END_TURN) -> Completion:
    return Completion(text=text, usage=_usage(), latency_ms=1, stop_reason=stop)


class _Client:
    def __init__(self, *replies: Completion | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self) -> None:
        pass


def _conversation() -> Messages:
    return Messages(
        [Message(role="user", content="q")], RetryParams(temperature=0.0, max_tokens=100)
    )


def test_an_answer_after_tool_calls_is_returned_with_its_counts() -> None:
    client = _Client(_calls("a", "b"), _text("ok:done"))
    role = ScriptedRole()
    asked = ask(client, _conversation(), role, reasks=1, charge=lambda _u: 0.5)
    assert (asked.answer, asked.end) == ("done", "answered")
    assert role.ran == ["a", "b"]
    assert (asked.model_calls, asked.tokens, asked.cost_usd) == (2, 24, 1.0)
    second = client.requests[1].messages
    assert [m.role for m in second] == ["user", "assistant", "tool", "tool"]
    assert [m.tool_call_id for m in second[2:]] == ["c0", "c1"]


def test_the_re_ask_bound_is_the_role_s() -> None:
    """Athena's bound is two: the third failure ends the request with no answer."""
    client = _Client(_text("x"), _text("y"), _text("z"))
    asked = ask(client, _conversation(), ScriptedRole(), reasks=2)
    assert (asked.answer, asked.end, asked.model_calls) == (None, "unparsed", 3)
    assert asked.error is not None and asked.error.kind == "bad_format"
    last = client.requests[2].messages
    assert (last[-2].content, last[-1].content) == ("y", "again: start with ok:")


def test_a_turn_cut_at_max_tokens_is_re_asked_and_never_dispatched() -> None:
    role = ScriptedRole()
    cut = _calls("a").model_copy(update={"stop_reason": StopReason.MAX_TOKENS})
    client = _Client(cut, _text("ok:fine"))
    asked = ask(client, _conversation(), role, reasks=1)
    assert role.ran == [] and asked.answer == "fine"
    assert client.requests[1].messages[-2].content == CUT
    assert client.requests[1].messages[-1].content == "again: shorter"


def test_a_call_that_ends_tool_use_makes_later_calls_answered_once_never_run() -> None:
    role = ScriptedRole()
    client = _Client(_calls("stop"), _calls("again"), _calls("and_again"))
    asked = ask(client, _conversation(), role, reasks=1)
    assert role.ran == ["stop"]
    assert asked.tools_ended and asked.end == "kept_calling"
    assert client.requests[2].messages[-1].content == "ENDED"


def test_cancel_between_a_turn_s_calls_dispatches_no_more() -> None:
    cancel = threading.Event()
    role = ScriptedRole(on_call=cancel)
    asked = ask(_Client(_calls("a", "b")), _conversation(), role, reasks=1, cancel=cancel)
    assert role.ran == ["a"] and asked.end == "cancelled"


def test_a_cancelled_request_and_a_set_token_start_nothing_more() -> None:
    assert ask(_Client(RequestCancelled(0)), _conversation(), ScriptedRole(), reasks=1).end == (
        "cancelled"
    )
    cancel = threading.Event()
    cancel.set()
    client = _Client()
    assert (
        ask(client, _conversation(), ScriptedRole(), reasks=1, cancel=cancel).end == "cancelled"
    )
    assert client.requests == []


def test_the_spend_gate_is_read_before_every_call() -> None:
    gate = iter([True, False])
    client = _Client(_calls("a"))
    asked = ask(client, _conversation(), ScriptedRole(), reasks=1, may_start=lambda: next(gate))
    assert asked.end == "spend_cap" and asked.model_calls == 1


def test_a_client_failure_is_returned_typed() -> None:
    asked = ask(
        _Client(TransportExhausted("down", 3)), _conversation(), ScriptedRole(), reasks=1
    )
    assert asked.end == "failed" and isinstance(asked.failure, TransportExhausted)
    assert asked.model_calls == 0


def test_the_request_carries_the_cancel_token() -> None:
    cancel = threading.Event()
    client = _Client(_text("ok:1"))
    ask(client, _conversation(), ScriptedRole(), reasks=0, cancel=cancel)
    assert client.requests[0].cancel is cancel


# -- #560: the refusal each call is told ---------------------------------------------


def _told(*replies: Completion) -> list[tuple[str, str | None]]:
    seen: list[tuple[str, str | None]] = []

    def on_call(completion: Completion, cost: float, refusal: ParseError | None) -> None:
        del cost
        seen.append((completion.text, None if refusal is None else refusal.hint))

    ask(_Client(*replies), _conversation(), ScriptedRole(), reasks=2, on_call=on_call)
    return seen


def test_a_refused_answer_is_told_with_its_error_and_a_taken_one_without() -> None:
    assert _told(_text("x"), _text("ok:done")) == [("x", "start with ok:"), ("ok:done", None)]


def test_the_refusal_that_ends_the_request_is_told_too() -> None:
    assert [r for _, r in _told(_text("x"), _text("y"), _text("z"))] == ["start with ok:"] * 3


def test_a_cut_turn_is_told_as_cut_and_a_tool_turn_has_none() -> None:
    told = _told(_calls("a"), _text("ok:", StopReason.MAX_TOKENS), _text("ok:fine"))
    assert [r for _, r in told] == [None, "shorter", None]


def test_kept_is_the_answer_as_its_conversation_keeps_it() -> None:
    """The one form of a final answer a continued conversation adds: the model's
    words as its reply; nothing for a request that did not answer, whatever it said."""
    answered = AskResult[str](answer="done", end="answered", text="done")
    assert kept(answered) == Message(role="assistant", content="done")
    assert kept(AskResult[str](answer=None, end="cancelled", text="")) is None
    assert kept(AskResult[str](answer=None, end="unparsed", text="")) is None
    # An empty reply that answered is kept as it was, never dropped.
    assert kept(AskResult[str](answer="", end="answered", text="")) == Message(
        role="assistant", content=""
    )
