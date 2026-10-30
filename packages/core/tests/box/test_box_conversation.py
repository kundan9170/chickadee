"""A box role's journaled conversation.

What these pin: every message is a `box_message` line before it joins; the head is journaled
first; `rebuilt` gives back exactly what was sent, a call and its result a pair, with the
compaction applied; `fit` evicts the oldest read or command first, never an edit tool's
answer nor the newest `keep_recent` exchanges, records one `box_compaction` line, and is a
`ContextFull` when even that does not fit.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chickadee.boxctx.conversation import (
    BoxConversation,
    ContextFull,
    JournaledTranscript,
    rebuilt,
)
from chickadee.client.request import Message
from chickadee.context.request import request_tokens
from chickadee.journal.writer import Journal
from chickadee.schema import RetryParams
from chickadee.schema.adapter import ToolCall
from chickadee.schema.journal import BoxCompactionP, BoxMessageP
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-10-15-0a1b2c3d"
PARAMS = RetryParams(temperature=0.2, max_tokens=1000)
BIG = "x = 1\n" * 400


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    return Journal.create(resolve_roots(tmp_path), RUN, {})


def _conversation() -> BoxConversation:
    return BoxConversation(
        role="worker",
        head=(
            Message(role="system", content="You are the worker."),
            Message(role="user", content="Task T1: use it in stats."),
        ),
        messages=(),
        params=PARAMS,
    )


def _exchange(cid: str, tool: str, result: str, **args: object) -> tuple[Message, Message]:
    call = ToolCall(id=cid, name=tool, arguments=dict(args))
    return (
        Message(role="assistant", content="", tool_calls=(call,)),
        Message(role="tool", content=result, tool_call_id=cid),
    )


def _transcript(
    journal: Journal, conversation: BoxConversation, *, budget: int = 100_000, keep: int = 1
) -> JournaledTranscript:
    return JournaledTranscript(
        conversation,
        journal,
        budget=budget,
        keep_recent=keep,
        keep_tools=frozenset({"edit", "create"}),
        task_id="T1",
        round=1,
    )


def test_each_message_is_journaled_before_it_joins_and_the_head_first(journal: Journal) -> None:
    t = _transcript(journal, _conversation())
    t.begin()
    t.add(*_exchange("c1", "read", "a = 1", path="a.py"))
    lines = journal.projection().lines(BoxMessageP)
    assert [p.payload.side for p in lines] == ["system", "user", "assistant", "tool"]
    assert all(p.task_id == "T1" and p.round == 1 for p in lines)
    assert len(t.conversation.messages) == 2


def test_rebuilt_gives_back_exactly_what_was_sent(journal: Journal) -> None:
    t = _transcript(journal, _conversation())
    t.begin()
    t.add(*_exchange("c1", "read", "a = 1", path="a.py"))
    t.add(Message(role="assistant", content="done"))
    t.add(Message(role="user", content="feedback: add a test"))
    back = rebuilt(journal.projection(), "worker", "T1", journal.get_blob, params=PARAMS)
    assert back.head == t.conversation.head
    assert back.messages == t.conversation.messages
    assert back.request() == t.conversation.request()


def test_fit_evicts_the_oldest_read_first_and_never_an_edit_or_the_newest(
    journal: Journal,
) -> None:
    conv = _conversation().appended(
        *_exchange("c1", "edit", BIG, path="a.py"),
        *_exchange("c2", "read", BIG, path="b.py"),
        *_exchange("c3", "shell", f"exit 0 · output h1\n{BIG}", command="pytest -q"),
        *_exchange("c4", "read", BIG, path="c.py"),
    )
    budget = request_tokens(conv.request()) - 100
    t = _transcript(journal, conv, budget=budget, keep=1)
    sent = t.request()
    assert t.conversation.evicted == frozenset({"c2"})
    receipt = next(m for m in sent.messages if m.tool_call_id == "c2")
    assert "for the current text" in str(receipt.content)
    [line] = journal.projection().lines(BoxCompactionP)
    assert line.payload.evict == ("c2",) and line.task_id == "T1"
    back = rebuilt(journal.projection(), "worker", "T1", journal.get_blob, params=PARAMS)
    assert back.evicted == frozenset({"c2"})


def test_a_command_receipt_keeps_its_exit_and_output_handle(journal: Journal) -> None:
    conv = replace(
        _conversation().appended(
            *_exchange("c1", "shell", f"exit 1 · output h9\n{BIG}", command="pytest -q"),
            *_exchange("c2", "read", "small", path="a.py"),
        ),
        commands=frozenset({"shell"}),
    )
    t = _transcript(journal, conv, budget=request_tokens(conv.request()) - 100, keep=1)
    sent = t.request()
    receipt = str(next(m for m in sent.messages if m.tool_call_id == "c1").content)
    assert "exit 1 · output h9" in receipt and "shell pytest -q" in receipt


def test_a_conversation_that_cannot_fit_is_context_full(journal: Journal) -> None:
    conv = _conversation().appended(*_exchange("c1", "edit", BIG, path="a.py"))
    t = _transcript(journal, conv, budget=50, keep=0)
    with pytest.raises(ContextFull) as raised:
        t.request()
    assert raised.value.exit.kind == "worker_overflow_mid"
