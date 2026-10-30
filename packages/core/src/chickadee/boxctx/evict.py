"""Layer 1 for every box role: a tool exchange becomes a receipt, planned before it is applied.


A box role's tool results are reads it can repeat. Inside one packet the files do not move
until the packet's patch lands, and rollback restores them exactly, so calling the tool again
returns the same bytes: an evicted result is recoverable by the role itself, and its receipt
says how. A box receipt carries facts only — which tool, on what, how large, which result —
because a box role declares no finding for a call, and a
receipt must not invent the claims Zeus's carry.

Planning is separate from applying because every edit to the middle of a conversation is a
cache rebuild from that point: a caller accumulates its whole plan, then rebuilds once.

**On a live worktree the premise ends**: the task loop's files move
between calls, by the role's own edits and its commands. So `live_receipt` words a read's
receipt as "read it again for the current text", never "the same bytes", and a command's
keeps its result's first line, which names its exit code and its output's handle (`read`
reopens it). An edit tool's answer is never evicted there, nor are the newest exchanges: the
caller's plan (`boxctx/conversation.py`) leaves them out.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from chickadee.client.request import Message, text_of
from chickadee.context.request import est_text

__all__ = [
    "BoxExchange",
    "apply_eviction",
    "exchanges",
    "live_receipt",
    "plan_eviction",
    "receipt",
]

_SUBJECT_KEYS: Final = ("path", "symbol", "query", "pattern", "name")
"""The argument that says what a box call was about, in the order they are looked for."""
_SUBJECT_MAX: Final = 80


@dataclass(frozen=True)
class BoxExchange:
    """One tool call and the message that answered it."""

    call_id: str
    tool: str
    args: Mapping[str, object]
    result_index: int
    """Where the answering `tool` message sits in the sequence it was found in."""
    tokens: int
    """The result's size as it stands, by the local estimate."""
    sha: str
    """sha256 of the result's text: `CoderToolP.result_sha` for a worker read, so the
    receipt names the journal row that holds it."""

    @property
    def receipt(self) -> str:
        return receipt(self)

    @property
    def saving(self) -> int:
        return max(0, self.tokens - est_text(self.receipt))


def exchanges(
    messages: Sequence[Message], *, skip: Collection[str] = ()
) -> tuple[BoxExchange, ...]:
    """Every answered call in `messages`, oldest first. `skip` names calls whose results are
    receipts already: evicting one again would only rewrite it. A call with no answer yet is
    not an exchange — its result has not arrived to be evicted."""
    calls: dict[str, tuple[str, Mapping[str, object]]] = {}
    out: list[BoxExchange] = []
    for i, m in enumerate(messages):
        if m.role == "assistant":
            calls.update({c.id: (c.name, c.arguments) for c in m.tool_calls})
            continue
        call_id = m.tool_call_id
        if call_id is not None and call_id in calls and call_id not in skip:
            tool, args = calls[call_id]
            text = text_of(m)
            out.append(
                BoxExchange(
                    call_id=call_id,
                    tool=tool,
                    args=args,
                    result_index=i,
                    tokens=est_text(text),
                    sha=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
    return tuple(out)


def receipt(ex: BoxExchange) -> str:
    subject = _subject(ex.args)
    what = f"{ex.tool} {subject}" if subject else ex.tool
    return (
        f"[evicted · {what} · {ex.tokens:,} tok · result {ex.sha[:8]}"
        f" — call {ex.tool} again to see it]"
    )


def live_receipt(ex: BoxExchange, result: str, *, command: bool) -> str:
    """A box exchange's receipt on a live worktree, `result` the text it replaces: a command
    (`command`, its tool's spec mutating the environment) keeps its result's first line, its
    exit code and output handle; anything else is to be read again, since the file may have
    moved since."""
    if command:
        line = ex.args.get("command")
        what = f"{ex.tool} {_flat(line)}" if isinstance(line, str) else ex.tool
        head = result.split("\n", 1)[0]
        return (
            f"[evicted · {what} · {ex.tokens:,} tok · {head} — read its output by that handle "
            "to see it again]"
        )
    subject = _subject(ex.args)
    what = f"{ex.tool} {subject}" if subject else ex.tool
    return (
        f"[evicted · {what} · {ex.tokens:,} tok · result {ex.sha[:8]}"
        f" — call {ex.tool} again for the current text]"
    )


def plan_eviction(
    ordered: Sequence[BoxExchange], tokens: int, target: int
) -> tuple[tuple[str, ...], int]:
    """Receipts in `ordered`'s order until the projection reaches `target`: the call ids to
    evict and the tokens the conversation would then hold. Nothing is changed; an empty plan
    means nothing needed doing, and costs no rebuild."""
    chosen: list[str] = []
    projected = tokens
    for ex in ordered:
        if projected <= target:
            break
        if ex.saving <= 0:
            continue
        chosen.append(ex.call_id)
        projected -= ex.saving
    return tuple(chosen), projected


def apply_eviction(
    messages: Sequence[Message],
    evict: Collection[str],
    *,
    render: Callable[[BoxExchange, str], str] | None = None,
) -> tuple[Message, ...]:
    """The same messages with each named call's result replaced by its receipt (`render`,
    given the exchange and the text it replaces; the packet loop's `receipt` by default). The
    call stays, and its result keeps its `tool_call_id`: a provider refuses a call with no
    answer, and the receipt is the answer."""
    by_id = {ex.call_id: ex for ex in exchanges(messages) if ex.call_id in evict}
    out: list[Message] = []
    for m in messages:
        ex = by_id.get(m.tool_call_id) if m.tool_call_id is not None else None
        if ex is None:
            out.append(m)
            continue
        said = ex.receipt if render is None else render(ex, text_of(m))
        out.append(replace(m, content=said))
    return tuple(out)


def _subject(args: Mapping[str, object]) -> str:
    for key in _SUBJECT_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _flat(value)
    return ""


def _flat(value: str) -> str:
    flat = " ".join(value.split())
    return flat if len(flat) <= _SUBJECT_MAX else flat[: _SUBJECT_MAX - 1] + "…"
