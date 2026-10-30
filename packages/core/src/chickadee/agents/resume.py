"""An agent's history, folded from the session log (T-3.188
Design §5).

Every round of an agent is three kinds of line: `agent_start` before anything of it runs, each
message as `agent_message` before it joins the conversation, and `agent_report` when it ends.
This module is the one reader of them. A follow-up reads its agent here whether the session
was resumed or not, so the conversation it continues is always the one on the log, never a
second copy kept in memory beside it.

**A round a crash cut** has its `agent_start` and no `agent_report`. Zeus's call that ran it
was answered `INTERRUPTED` by the loop's tail repair (`zeus/loop._repair`), never re-run. The
agent can still be followed up: its conversation is rebuilt as far as the log holds it, and
every tool call the log holds no result for is answered `INTERRUPTED`, right after the results
it does hold, since a conversation with a call and no result is one no endpoint accepts. A
crash leaves such calls, and so does a round that ended mid-turn: the person's stop between
calls, or a model that kept calling after its tools ended. The answers are made the same way
each time the conversation is read, so none is journaled.

**Ids count every agent the log ever held**, an undone turn's included: an undo hides its
`agent` lines (`context/rewind.py`), and an id given twice would make two agents one to every
reader of the whole log.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from chickadee.boxctx.conversation import message_of
from chickadee.client.request import Message
from chickadee.context.citations import repository_path
from chickadee.context.kept_lines import AgentRecord
from chickadee.context.results import Interrupted
from chickadee.context.session import SessionState
from chickadee.schema.adapter import ToolCall
from chickadee.schema.journal import AgentMessageP, AgentReportP, AgentStartP
from chickadee.tools.builtin import BUILTINS

__all__ = [
    "AgentHistory",
    "conversation",
    "fetched_url",
    "histories",
    "next_agent_id",
    "pages",
    "read_path",
    "reads",
]


@dataclass
class AgentHistory:
    """One agent as its lines say it is."""

    agent: str
    start: AgentStartP
    """Round 0's: its skill, profile and rung."""
    rounds: int = 1
    """Rounds started, its spawn's included: `rounds - 1` follow-ups."""
    report: AgentReportP | None = None
    """Its latest round's report, or None while that round has none: it is running here, or
    a crash cut it."""
    shas: list[str] = field(default_factory=list[str])
    """Its messages' blobs, in the order they joined, every round's."""


def histories(state: SessionState, *, with_hidden: bool = False) -> dict[str, AgentHistory]:
    """Every agent of the session, by id, in the order they were spawned: those of turns
    still in the conversation, or with `with_hidden` those an undo hid as well."""
    out: dict[str, AgentHistory] = {}
    for _, record in state.lines(with_hidden=with_hidden):
        if not isinstance(record, AgentRecord) or record.line is None:
            continue
        payload = record.line.payload
        if isinstance(payload, AgentStartP):
            if payload.round == 0:
                out[payload.agent] = AgentHistory(payload.agent, payload)
                continue
            known = out.get(payload.agent)
            if known is not None:
                known.rounds = payload.round + 1
                known.report = None
        elif isinstance(payload, AgentMessageP):
            known = out.get(payload.agent)
            if known is not None:
                known.shas.append(payload.sha)
        elif isinstance(payload, AgentReportP):
            known = out.get(payload.agent)
            if known is not None and payload.round == known.rounds - 1:
                known.report = payload
    return out


def next_agent_id(state: SessionState) -> str:
    """`A1`, `A2`,...: one past every agent the log holds, hidden ones included, so a resumed
    session goes on and an undone agent's id is never given again."""
    return f"A{len(histories(state, with_hidden=True)) + 1}"


def conversation(history: AgentHistory, blob: Callable[[str], bytes]) -> list[Message]:
    """Its conversation, read back from its blobs, every tool call answered: one the log holds
    no result for is answered `INTERRUPTED` after the results its turn does hold. A result
    answers a call of its own turn only: some endpoints number calls the same way every turn
    (`call_0`), so an id may come back in a later one."""
    out: list[Message] = []
    owed: list[ToolCall] = []
    for message in (*(message_of(blob(sha)) for sha in history.shas), None):
        if message is None or message.role != "tool":
            out += [_interrupted(c) for c in owed]
            owed = []
        if message is None:
            break
        out.append(message)
        if message.role == "assistant":
            owed = list(message.tool_calls)
        elif message.role == "tool":
            owed = [c for c in owed if c.id != message.tool_call_id]
    return out


def _interrupted(call: ToolCall) -> Message:
    why = "no result was recorded for this call: it did not run, or did not finish"
    return Message(
        role="tool", content=Interrupted(call.name, why).render(), tool_call_id=call.id
    )


def reads(messages: Sequence[Message]) -> dict[str, list[str]]:
    """By path, what each of a conversation's `read` calls showed, in order: what the
    citation check counts as seen (`citations.seen_in_reads`). A follow-up's report may cite
    what an earlier round read, since that is in its conversation still. A result is matched
    to a call of its own turn, as `conversation` matches them."""
    return _shown(messages, read_path)


def pages(messages: Sequence[Message]) -> dict[str, list[str]]:
    """By URL, what each of a conversation's page fetches showed, in order: what a
    follow-up's web citation of an earlier round's page is checked against,
    as `reads` is for a file. Matched to its call as `reads` matches."""
    return _shown(messages, fetched_url)


def _shown(
    messages: Sequence[Message], key: Callable[[str, Mapping[str, object]], str | None]
) -> dict[str, list[str]]:
    """By `key` of each call, the results its calls got, in order, each matched to a call
    of its own turn."""
    out: dict[str, list[str]] = {}
    turn: dict[str, str | None] = {}
    for m in messages:
        if m.role == "assistant":
            turn = {c.id: key(c.name, c.arguments) for c in m.tool_calls}
        elif m.role == "tool" and isinstance(m.content, str):
            found = turn.get(m.tool_call_id or "")
            if found is not None:
                out.setdefault(found, []).append(m.content)
    return out


def read_path(tool: str, arguments: Mapping[str, object]) -> str | None:
    """The file a call shows the lines of, by its spec, never its name: a read-family
    tool given a `path`, in the form a citation of it is checked in
    (`citations.repository_path`), so `./src/a.py` read is `src/a.py` cited. What the citation
    check counts as seen, live (`spawn._AgentSink`) and in a conversation read back
    (`reads`)."""
    found = BUILTINS.get(tool)
    path = arguments.get("path")
    if found is None or found.spec.family != "read" or not isinstance(path, str):
        return None
    return repository_path(path)


def fetched_url(tool: str, arguments: Mapping[str, object]) -> str | None:
    """The page a call shows the text of, by its spec, never its name: a read-family
    tool on the `web` port given a `url`, as it was asked."""
    found = BUILTINS.get(tool)
    url = arguments.get("url")
    if found is None or found.spec.family != "read" or "web" not in found.needs:
        return None
    return url if isinstance(url, str) else None
