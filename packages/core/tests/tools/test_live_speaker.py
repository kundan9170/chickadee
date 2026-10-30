"""A box role's replies shown as they stream: `tools.loop.ask` gives each model
call a fresh live view, labelled with the role and the task it holds, flushed however the call
ends; without a speaker, nothing is sent with the request."""

from __future__ import annotations

from chickadee.client.errors import TransportExhausted
from chickadee.client.live import LiveEvent, Speaker, Text, TextUpdate
from chickadee.client.request import ChatRequest, Message
from chickadee.schema import RetryParams
from chickadee.schema.adapter import Completion, ParseError, StopReason, ToolCall, Usage
from chickadee.tools.loop import Messages, Reply, ask


class _Streaming:
    """Streams `said` through the request's `live`, then answers or fails."""

    def __init__(self, *replies: tuple[str, Completion | Exception]) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        said, reply = self.replies.pop(0)
        if request.live is not None:
            for word in said.split(" "):
                request.live(Text(word + " "))
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self) -> None:
        pass


class _Plain:
    def call(self, call: ToolCall, cancel: object) -> Reply:
        return Reply(f"ran {call.name}")

    def after_end(self) -> str:
        return "ENDED"

    def parse(self, text: str) -> str | ParseError:
        return text

    def reask(self, error: ParseError) -> str:
        return error.hint

    def cut(self) -> ParseError:
        return ParseError(kind="truncated", hint="shorter")


def _usage() -> Usage:
    return Usage(prompt_tokens=10, completion_tokens=2)


def _calls(name: str) -> Completion:
    return Completion(
        text="",
        usage=_usage(),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=(ToolCall(id="c0", name=name, arguments={}),),
    )


def _text(text: str) -> Completion:
    return Completion(text=text, usage=_usage(), latency_ms=1, stop_reason=StopReason.END_TURN)


def _conversation() -> Messages:
    return Messages(
        [Message(role="user", content="go")], RetryParams(temperature=0.0, max_tokens=100)
    )


def test_each_call_is_shown_labelled_with_its_role_and_task() -> None:
    shown: list[LiveEvent] = []
    client = _Streaming(("reading it", _calls("read")), ("all done", _text("all done")))
    ask(
        client,
        _conversation(),
        _Plain(),
        reasks=0,
        speaker=Speaker(shown.append, "worker", lambda: "T1"),
    )
    assert {(e.role, e.task) for e in shown} == {("worker", "T1")}
    first, second = shown[0].reply, shown[-1].reply
    assert first != second, "each model call is a reply of its own"
    text = {
        reply: "".join(
            e.update.piece
            for e in shown
            if e.reply == reply and isinstance(e.update, TextUpdate)
        )
        for reply in (first, second)
    }
    assert text == {first: "reading it ", second: "all done "}


def test_a_failed_call_still_flushes_what_streamed() -> None:
    shown: list[LiveEvent] = []
    client = _Streaming(("half a", TransportExhausted("gone", 1)))
    asked = ask(
        client, _conversation(), _Plain(), reasks=0, speaker=Speaker(shown.append, "athena")
    )
    assert asked.end == "failed"
    assert (
        "".join(e.update.piece for e in shown if isinstance(e.update, TextUpdate)) == "half a "
    )


def test_without_a_speaker_no_live_is_sent() -> None:
    client = _Streaming(("quiet", _text("quiet")))
    ask(client, _conversation(), _Plain(), reasks=0)
    assert client.requests[0].live is None
