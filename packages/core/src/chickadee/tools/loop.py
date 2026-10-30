"""One tool-using request, to its answer: the loop the scout and Athena share.

A role that asks a model a question, lets it call tools, and takes a structured answer at the
end runs this loop. It used to live in the scout alone; Athena's three request kinds need the
same one, and a second copy is what one concept, one home forbids.

What the loop decides, and nothing else:

- **a turn is read from `stop_reason` and `tool_calls`, never from text**. Output cut at
  `max_tokens` is truncated, never a terminal parse, and its calls are not dispatched;
- **every call of a turn is answered**, in order, in one batch — a provider refuses a turn whose
  calls are not all answered — and cancellation is checked between them;
- **a call that ends the role's tool use** (a cap, no progress) makes the next turn the answer.
  A call after the end is answered with the role's words for that, once, and never dispatched;
- **a final answer that does not parse is re-asked** with the parse error's hint, up to the
  role's bound; then the loop ends with no answer. A re-ask also makes the next turn the answer;
- **before every model call** the cancel token and the spend gate are read: a crossed cap starts
  no call;
- **a hold** (a message for the team waits) is read where the cancel token is:
  before a model call the request ends `held`; between a turn's calls, the rest are answered
  `HELD_NOT_RUN`, so the conversation stays well formed and goes on from there;
- **a call that failed after it was sent is charged** what it is known to have cost, through
  the one charge of a failed call (`client/charge.py`), so the worker's and
  Athena's timed-out attempts count toward the cap as Zeus's do;
- **the role's hooks are asked twice a turn**, through `LoopHooks`, which this loop
  knows only as two questions: before a model call, whether to make it; after a turn's calls,
  what the last answer carries and whether tool use ends.

What a call does, what the answer is, how the conversation is held and compacted, and what each
ending means to the caller are the role's: the `RequestTranscript` and `Role` it passes in.

**The final answer is never added to the conversation here.** A role whose conversation goes on
after its answer (the worker's next round, Athena's next request, an agent's follow-up) adds
`kept(asked)` itself, when it chooses: the worker journals it after its round's report line, so
a round with a reply is never open on resume (`box/resume.owed_reply`). `kept` is the one form
of it, so the three cannot drift; `tests/box/test_kept_answer_pins.py` and
`tests/agents/test_follow_up.py` hold each role to it.

**Concurrency, stated where the state is.** One `ask` runs on its caller's thread, one model
call at a time; a turn's calls are dispatched in order, never on threads (the graph's SQLite
connection belongs to the thread that opened it). Nothing here outlives the call.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final, Literal, Protocol

from chickadee.client.charge import charge_unanswered
from chickadee.client.errors import ClientError, RequestCancelled
from chickadee.client.live import Speaker, speaking
from chickadee.client.request import (
    ChatRequest,
    Message,
    ModelClient,
    Part,
    TextPart,
    ToolSchema,
)
from chickadee.context.request import request_tokens
from chickadee.context.result_data import ResultData
from chickadee.schema import RetryParams
from chickadee.schema.adapter import Completion, ParseError, StopReason, ToolCall, Usage

__all__ = [
    "AskResult",
    "BatchSaid",
    "End",
    "Halted",
    "LoopHooks",
    "Messages",
    "Reply",
    "RequestTranscript",
    "Role",
    "ask",
    "kept",
]

_log = logging.getLogger(__name__)

CUT: str = "[cut at max_tokens]"
"""What a cut turn with no text is kept as, so the re-ask follows something the model said."""


@dataclass(frozen=True)
class Reply:
    """One tool call's answer, as the model reads it."""

    content: str | tuple[Part, ...]
    ends: bool = False
    """This call ended the role's tool use (a cap, no progress): the next turn is the answer."""
    final: str | None = None
    """This call is the role's answer (the worker's `finish`): the request ends on
    it, as on a reply with no tool call, read through the role's `parse`."""
    data: ResultData | None = None
    """What the answer was made from, for a role whose calls are Python
    blocks; never sent to a model."""
    refused: bool = False
    """The call was refused or failed (`catalogue.refused`): a Python block that made it has
    failed, and the reply's later blocks do not run."""


class RequestTranscript(Protocol):
    """What is sent: the whole next request, and the messages the loop adds to it. A role that
    compacts its context does it in `request()`, before the call it is for."""

    def request(self) -> ChatRequest: ...

    def add(self, *messages: Message) -> None: ...


@dataclass(frozen=True)
class Halted:
    """A hook ended the request or the role's tool use: which, and why."""

    by: str
    reason: str


@dataclass(frozen=True)
class BatchSaid:
    """What the role's hooks said after a turn's calls: a note the last answer carries, and
    the halt that ends tool use, if one does."""

    note: str | None = None
    halt: Halted | None = None


class LoopHooks(Protocol):
    """The role's hooks, as this loop asks them: `hooks/loop.py` answers for them."""

    def before_request(self, request: ChatRequest) -> Halted | None:
        """The halt that means no model call is made now, or None."""
        ...

    def after_batch(self, calls: tuple[ToolCall, ...]) -> BatchSaid: ...


class Role[A](Protocol):
    """How one role's request is answered: its calls, its after-the-end words, and its final
    answer's parse and re-ask."""

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply: ...

    def after_end(self) -> str:
        """What a call made after the role's tool use ended is answered."""
        ...

    def parse(self, text: str) -> A | ParseError: ...

    def reask(self, error: ParseError) -> str:
        """The user message that follows a final answer that did not parse."""
        ...

    def cut(self) -> ParseError:
        """The error a turn cut at `max_tokens` is re-asked with."""
        ...


End = Literal[
    "answered", "unparsed", "kept_calling", "cancelled", "spend_cap", "failed", "halted", "held"
]
"""How the loop ended:

- `answered`: the final answer parsed;
- `unparsed`: the re-asks ran out (`error` is the last parse error);
- `kept_calling`: the model called tools again after its after-the-end answer;
- `cancelled`: the stop, before a call, during one, or between a turn's calls;
- `spend_cap`: the spend gate refused the next call;
- `failed`: the client failed (`failure`);
- `halted`: a hook ended the request before a model call (`halted` says why);
- `held`: a hold was set: no further model call was made."""


@dataclass
class AskResult[A]:
    """What one `ask` came to. The counts are the role's to report."""

    answer: A | None = None
    end: End = "answered"
    model_calls: int = 0
    tokens: int = 0
    """Prompt and completion tokens, every call that answered."""
    cost_usd: float = 0.0
    """What every call was charged, a failed call's estimate included."""
    tools_ended: bool = False
    """A call ended the role's tool use (`Reply.ends`), or a hook did after a turn's calls."""
    halted: Halted | None = None
    """The hook that ended the request or the role's tool use, and why."""
    error: ParseError | None = None
    failure: ClientError | None = None
    text: str = ""
    """The answer's words as the model gave them: the answering turn's text, or what a tool
    handed over (`Reply.final`, the worker's `finish`). Set only when it `answered`."""

    def after(self, earlier: AskResult[A]) -> AskResult[A]:
        """This ask as the continuation of `earlier`, one request's two asks as one: what a
        hook's `Continue` at a role's stop makes of them. Its ending is this one's; the
        counts are both's, never a record's spend (the journal's, `projection.py`)."""
        return replace(
            self,
            model_calls=earlier.model_calls + self.model_calls,
            tokens=earlier.tokens + self.tokens,
            cost_usd=earlier.cost_usd + self.cost_usd,
        )


@dataclass
class Messages:
    """The plain conversation: a list, sent whole every time, with fixed params and tools.
    The tools stay on every request, the last one included: a history holding tool calls is
    refused by some providers without them, and dropping them moves the cached prefix."""

    messages: list[Message]
    params: RetryParams
    tools: tuple[ToolSchema, ...] = ()
    session_key: str | None = None

    def request(self) -> ChatRequest:
        return ChatRequest(
            messages=tuple(self.messages),
            params=self.params,
            tools=self.tools,
            session_key=self.session_key,
        )

    def add(self, *messages: Message) -> None:
        self.messages.extend(messages)


_AFTER_FINAL: Final = "not run: the call before it ended the task"

HELD_NOT_RUN: Final = "not run: held for a message from the team lead"
"""A call of a turn a hold stopped before it ran: every call is answered."""


@dataclass
class _State:
    reasks: int = 0
    wrap_up: bool = False
    """The next turn is the answer: tool use ended, or the answer was re-asked."""
    answered_after_end: bool = False
    done: bool = False


def ask[A](
    client: ModelClient,
    conversation: RequestTranscript,
    role: Role[A],
    *,
    reasks: int,
    charge: Callable[[Usage], float] | None = None,
    charge_failed: Callable[[Usage], float] | None = None,
    may_start: Callable[[], bool] | None = None,
    cancel: threading.Event | None = None,
    on_call: Callable[[Completion, float, ParseError | None], None] | None = None,
    speaker: Speaker | None = None,
    force_tool: bool = False,
    hooks: LoopHooks | None = None,
    hold: threading.Event | None = None,
) -> AskResult[A]:
    """Run one request to its answer, or to the first ending that stops it.

    `reasks` is how many times a final answer that did not parse is asked again. `charge`
    prices each answered call and returns what it charged (the ledger's); `charge_failed`
    prices a call that failed after it was sent (`client/charge.py`), which no `model_call`
    line records, so a role's ledger writes its own `spend` line for it; `may_start` is the
    spend gate, read before every call. `on_call` is told each answered call, what
    it was charged, and the error its answer was refused on, if it was, before anything
    the turn asks for is done, so a role can line it. `speaker` is the role's live voice:
    each model call's reply is shown as it streams, never journaled. `force_tool` asks
    that each turn be a tool call, except a wrap-up turn, which must be the answer.
    `hooks` are the role's, asked before each model call and after each turn's calls.
    `hold` holds the request at its next boundary: read before each model call
    and between a turn's calls, never cutting a call in flight here."""
    out: AskResult[A] = AskResult()
    state = _State()
    while True:
        if cancel is not None and cancel.is_set():
            out.end = "cancelled"
            return out
        if hold is not None and hold.is_set():
            out.end = "held"
            return out
        if may_start is not None and not may_start():
            out.end = "spend_cap"
            return out
        with speaking(speaker) as live:
            request = replace(
                conversation.request(),
                cancel=cancel,
                live=live,
                force_tool=force_tool and not state.wrap_up,
            )
            if hooks is not None and (halt := hooks.before_request(request)) is not None:
                out.end, out.halted = "halted", halt
                return out
            try:
                completion = client.send(request)
            except ClientError as exc:
                if charge_failed is not None:
                    out.cost_usd += _charge_failed(charge_failed, request, exc)
                if isinstance(exc, RequestCancelled):
                    out.end = "cancelled"
                else:
                    out.end, out.failure = "failed", exc
                return out
        out.model_calls += 1
        out.tokens += completion.usage.prompt_tokens + completion.usage.completion_tokens
        cost = charge(completion.usage) if charge is not None else 0.0
        out.cost_usd += cost
        said = _read(completion, role)
        if on_call is not None:
            on_call(completion, cost, said if isinstance(said, ParseError) else None)
        _turn(completion, said, conversation, role, state, out, reasks, cancel, hooks, hold)
        if state.done:
            return out


def _after_batch[A](
    hooks: LoopHooks,
    calls: tuple[ToolCall, ...],
    reply: Reply,
    state: _State,
    out: AskResult[A],
) -> Reply:
    said = hooks.after_batch(calls)
    if said.halt is not None:
        out.tools_ended, out.halted = True, said.halt
        state.wrap_up = True
    if said.note is None:
        return reply
    content = reply.content
    if isinstance(content, str):
        return replace(reply, content=f"{content}\n{said.note}")
    return replace(reply, content=(*content, TextPart(said.note)))


def _charge_failed(
    charge: Callable[[Usage], float], request: ChatRequest, failure: ClientError
) -> float:
    """`failure` charged through the role's `charge_failed`. A `spend` line the session log
    refused (a full disk) costs the line, never the request's own ending: the ledger counted
    the charge before it wrote the line, so the cap still holds it in this process, and the
    role still ends on the client's error, which is what it reports."""
    try:
        return charge_unanswered(charge, lambda: request_tokens(request), failure)
    except OSError:
        _log.warning("a failed call's estimated charge was not written", exc_info=True)
        return 0.0


def _read[A](completion: Completion, role: Role[A]) -> A | ParseError | None:
    """What one turn says, by its `stop_reason` and `tool_calls`: a cut answer, `None`
    for a turn that asks for tools, else its answer parsed."""
    if completion.stop_reason is StopReason.MAX_TOKENS:
        # Its tool calls may be cut too, so none is dispatched and the turn is not kept as one.
        return role.cut()
    if completion.tool_calls:
        return None
    return role.parse(completion.text)


def kept[A](asked: AskResult[A]) -> Message | None:
    """The answer as its conversation keeps it: the model's words, as its reply. None when it
    did not answer. The loop never adds it (the module docstring); a role that continues its
    conversation adds this, and only this."""
    if asked.end != "answered":
        return None
    return Message(role="assistant", content=asked.text)


def _turn[A](
    completion: Completion,
    said: A | ParseError | None,
    conversation: RequestTranscript,
    role: Role[A],
    state: _State,
    out: AskResult[A],
    reasks: int,
    cancel: threading.Event | None,
    hooks: LoopHooks | None = None,
    hold: threading.Event | None = None,
) -> None:
    """Act on what one turn said (`_read`)."""
    if said is None:
        conversation.add(
            Message(role="assistant", content=completion.text, tool_calls=completion.tool_calls)
        )
        if state.wrap_up:
            _after_the_end(completion.tool_calls, conversation, role, state, out)
            return
        _dispatch(completion.tool_calls, conversation, role, state, out, cancel, hooks, hold)
        return
    if isinstance(said, ParseError):
        cut = completion.stop_reason is StopReason.MAX_TOKENS
        text = (completion.text or CUT) if cut else completion.text
        _reask(text, said, conversation, role, state, out, reasks)
        return
    out.answer, out.end, state.done = said, "answered", True
    out.text = completion.text


def _reask[A](
    text: str,
    error: ParseError,
    conversation: RequestTranscript,
    role: Role[A],
    state: _State,
    out: AskResult[A],
    reasks: int,
) -> None:
    out.error = error
    if state.reasks >= reasks:
        out.end, state.done = "unparsed", True
        return
    state.reasks += 1
    state.wrap_up = True
    conversation.add(
        Message(role="assistant", content=text),
        Message(role="user", content=role.reask(error)),
    )


def _after_the_end[A](
    calls: tuple[ToolCall, ...],
    conversation: RequestTranscript,
    role: Role[A],
    state: _State,
    out: AskResult[A],
) -> None:
    """Calls after the end are answered, never dispatched — once. A model that asks again is
    not going to answer."""
    if state.answered_after_end:
        out.end, state.done = "kept_calling", True
        return
    state.answered_after_end = True
    text = role.after_end()
    conversation.add(*(Message(role="tool", content=text, tool_call_id=c.id) for c in calls))


def _dispatch[A](
    calls: tuple[ToolCall, ...],
    conversation: RequestTranscript,
    role: Role[A],
    state: _State,
    out: AskResult[A],
    cancel: threading.Event | None,
    hooks: LoopHooks | None = None,
    hold: threading.Event | None = None,
) -> None:
    for call in calls:
        if cancel is not None and cancel.is_set():
            out.end, state.done = "cancelled", True
            return
        if hold is not None and hold.is_set():
            # Every call of the turn is answered, so the round goes on from here once the
            # message is read; the next model call is not made.
            for rest in calls[calls.index(call) :]:
                conversation.add(
                    Message(role="tool", content=HELD_NOT_RUN, tool_call_id=rest.id)
                )
            out.end, state.done = "held", True
            return
        reply = role.call(call, cancel)
        if reply.ends:
            out.tools_ended = True
            state.wrap_up = True
        if hooks is not None and call is calls[-1] and reply.final is None:
            # Before the batch's last answer is added, so what the hooks say rides on it
            # and is journaled with it.
            reply = _after_batch(hooks, calls, reply, state, out)
        conversation.add(Message(role="tool", content=reply.content, tool_call_id=call.id))
        if reply.final is not None:
            said = role.parse(reply.final)
            if not isinstance(said, ParseError):
                # Every call of the turn has an answer, so the conversation stays well formed;
                # one after the answer is never run.
                for rest in calls[calls.index(call) + 1 :]:
                    conversation.add(
                        Message(role="tool", content=_AFTER_FINAL, tool_call_id=rest.id)
                    )
                out.answer, out.end, state.done = said, "answered", True
                out.text = reply.final
                return
