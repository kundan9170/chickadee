"""Layer 1 and the dropped duplicates.

A tool exchange inside a cut becomes a receipt of its finding and how to recover it (no
intent); an older duplicate of a call made again is dropped. An exchange with no finding is
refused, never collapsed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from chickadee.context.handle import Shape
from chickadee.context.recovery import Recovery
from chickadee.context.request import args_text, est_call, est_text
from chickadee.context.session import ToolLine, Turn
from chickadee.edits.lines import split_lines
from chickadee.errors import HarnessBug

if TYPE_CHECKING:
    from chickadee.context.handle import Handle
    from chickadee.context.lines import Call
    from chickadee.context.session import SessionState
    from chickadee.context.spans import Span

__all__ = [
    "CollapseRefused",
    "Receipt",
    "ReceiptClaims",
    "ReceiptFacts",
    "ToolExchange",
    "collapse",
    "dropped_text",
    "exchanges_in",
    "flat",
    "kept_text",
    "recovery_line",
    "supersede",
    "superseded_text",
]


_SUBJECT_MAX: Final = 120
"""Characters of a call's argument shown on a receipt's first line; the digest names it
exactly."""


_SUBJECT_KEYS: Final = (
    "target",
    "path",
    "handle",
    "command",
    "pattern",
    "query",
    "name",
    "symbol",
)
"""The argument that says what a call was about, in the order they are looked for."""


@dataclass(frozen=True)
class ToolExchange:
    """A tool call and its result, as ONE unit: they evict together or not at all."""

    call_id: str
    tool: str
    args: Mapping[str, object]
    finding: str | None
    """Written AFTER the result, in the turn that consumed it."""
    handle: str | None
    shape: Shape
    """The handle's shape, or the result text's own when it came back without one."""
    tokens: int
    """The exchange's weight when it was inserted: the call and its result."""
    rehydrated: int
    """Later results that returned the same handle."""
    seq: int
    """The assistant turn that made the call."""
    result_seq: int
    view: tuple[Span, ...]
    image: bool


@dataclass(frozen=True)
class ReceiptFacts:
    """Measured. Every field is produced by the harness, none by a model."""

    tool: str
    subject: str
    """What the call was about — its own argument, verbatim and clipped."""
    args_digest: str
    tokens: int
    rehydrated: int
    shape: Shape
    recovery: Recovery
    recovery_line: str
    """How to get it back, or that it cannot be had back."""


@dataclass(frozen=True)
class ReceiptClaims:
    """Asserted by Zeus and checkable by nothing. Under `recovery = NONE` the finding is the
    only surviving record of what the call returned."""

    finding: str


@dataclass(frozen=True)
class Receipt:
    """What a collapsed exchange becomes. There is no `collapse()` for a Receipt, so running a
    cut over its own output collapses nothing: idempotence by type, not by a flag."""

    facts: ReceiptFacts
    claims: ReceiptClaims

    def render(self) -> str:
        """Claims quoted and labelled, facts bare — rendered apart (invariant 16), because a
        model's assertion set in the typography of a token count becomes ground truth six
        turns later."""
        f, c = self.facts, self.claims
        return "\n".join(
            [
                f"[evicted · {_label(f.tool, f.subject)}",
                f'   finding  "{flat(c.finding)}"',
                f"   {f.tokens:,} tok · {_shape_text(f.shape)} · rehydrated {f.rehydrated}x",
                f"   {f.recovery_line} ]",
            ]
        )


@dataclass(frozen=True)
class CollapseRefused:
    """§5.8 FAILS: collapsing without a finding destroys evidence and puts nothing in its
    place, which is strictly worse than spending the tokens."""

    call_id: str


def flat(text: str) -> str:
    return " ".join(text.split())


def _clip(text: str, limit_chars: int) -> str:
    return text if len(text) <= limit_chars else text[: limit_chars - 1] + "…"


def _label(tool: str, subject: str) -> str:
    return f"{tool} {subject}" if subject else tool


def _subject(args: Mapping[str, object]) -> str:
    for key in _SUBJECT_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value:
            return _clip(flat(value), _SUBJECT_MAX)
    text = args_text(args)
    return "" if text == "{}" else _clip(text, _SUBJECT_MAX)


def _shape_text(shape: Shape) -> str:
    if shape.is_image and shape.width is not None and shape.height is not None:
        return f"image {shape.width}x{shape.height}"
    if shape.lines is not None:
        return f"{shape.lines:,} lines"
    return f"{shape.bytes:,} bytes"


def _text_shape(text: str) -> Shape:
    return Shape(len(text.encode("utf-8")), len(split_lines(text)), None, "text/plain")


def exchanges_in(state: SessionState, lo: int, hi: int) -> tuple[ToolExchange, ...]:
    """Every call made in `[lo, hi]`, each with its result. A result past `hi` would put the
    cut between a `tool_use` and its `tool_result`: an assert."""
    found: list[ToolExchange] = []
    for _event, record in state.lines(lo, hi):
        if not isinstance(record, Turn):
            continue
        for call in record.calls:
            result = state.result(call.call_id)
            if result is None or result.seq > hi:
                raise HarnessBug(f"call {call.call_id} has no result inside seq {lo}-{hi}")
            found.append(_exchange(state, call, result))
    return tuple(found)


def _exchange(state: SessionState, call: Call, result: ToolLine) -> ToolExchange:
    handle = result.handle
    tokens = est_call(call.tool, call.args) + est_text(result.text)
    if result.image is not None and handle is not None:
        tokens += handle.shape.est_tokens or 0
    return ToolExchange(
        call_id=call.call_id,
        tool=call.tool,
        args=call.args,
        finding=state.finding(call.call_id),
        handle=handle.id if handle is not None else None,
        shape=handle.shape if handle is not None else _text_shape(result.text),
        tokens=tokens,
        rehydrated=state.rereads(handle.id, result.seq) if handle is not None else 0,
        seq=call.seq,
        result_seq=result.seq,
        view=result.view,
        image=result.image is not None,
    )


def supersede(
    exchanges: Sequence[ToolExchange],
) -> tuple[tuple[ToolExchange, ...], Mapping[str, ToolExchange]]:
    """`(kept, dropped)`, where `dropped` maps each older duplicate's call id to the
    newest exchange making the same call. The older one loses its result and keeps its
    finding — a different question may have been asked of the same call.

    Safe only because the journal is the authority, and here it always is: every exchange is
    a durable line of the session log, so "failed at turn 4, passed at turn 40" stays
    recoverable after the cut. An `ask_user` answer is never superseded: it is what a person
    said, not an observation a newer one makes obsolete."""
    groups: dict[tuple[str, str], list[ToolExchange]] = {}
    for ex in exchanges:
        if ex.tool != "ask_user":
            groups.setdefault((ex.tool, args_text(ex.args)), []).append(ex)
    dropped: dict[str, ToolExchange] = {}
    for group in groups.values():
        newest = max(group, key=lambda e: (e.seq, e.result_seq))
        dropped.update((ex.call_id, newest) for ex in group if ex is not newest)
    return tuple(ex for ex in exchanges if ex.call_id not in dropped), dropped


def recovery_line(ex: ToolExchange, recovery: Recovery, handle: Handle | None) -> str:
    """`read(<handle>, <span>) -> exact`, or why it cannot be had back. A stored output — a
    command's log, which existed nowhere else — is named even when the call cannot be re-run
    to the same answer, or it would be real, on disk and unnameable."""
    if recovery is Recovery.EXACT and handle is not None:
        if ex.image:
            return f'read(handle={handle.id}, depth="full") -> exact'
        if ex.view:
            reads = " · ".join(f"read(handle={handle.id}, lines={span})" for span in ex.view)
            return f"{reads} -> exact"
        return f"read(handle={handle.id}) -> exact"
    line = "not recoverable - finding is the record"
    if handle is not None and handle.stored:
        # `stored`, not the kind: an image read from a file has its original in the blob
        # store too, so the bytes are still nameable after the file is gone.
        what = "the image" if handle.shape.is_image else "recorded output"
        line += f" · {what}: read(handle={handle.id})"
    return line


def collapse(
    ex: ToolExchange, recovery: Recovery, handle: Handle | None = None
) -> Receipt | CollapseRefused:
    """§5.8. `recovery` is `actual_recovery` evaluated at the cut — after the freshness pass,
    so a stale handle has already been degraded to NONE and no receipt promises an exact read
    of bytes that moved."""
    # The finding alone is the claim a receipt needs: its facts name the call already.
    finding = ex.finding
    if not finding:
        return CollapseRefused(ex.call_id)
    facts = ReceiptFacts(
        tool=ex.tool,
        subject=_subject(ex.args),
        args_digest=hashlib.sha256(args_text(ex.args).encode("utf-8")).hexdigest()[:16],
        tokens=ex.tokens,
        rehydrated=ex.rehydrated,
        shape=ex.shape,
        recovery=recovery,
        recovery_line=recovery_line(ex, recovery, handle),
    )
    return Receipt(facts, ReceiptClaims(finding))


_NO_FINDING: Final = "no finding was written"
"""Why a collapse is refused: a finding is all a receipt needs, so all it can lack."""


def _claim_rows(ex: ToolExchange) -> list[str]:
    rows: list[str] = []
    if ex.finding:
        rows.append(f'   finding  "{flat(ex.finding)}"')
    return rows


def kept_text(ex: ToolExchange, shown: str) -> str:
    """A refused collapse kept verbatim: the call as sent and the result as read."""
    return "\n".join(
        [
            f"[kept verbatim · {_label(ex.tool, _subject(ex.args))} · {_NO_FINDING}",
            f"   call     {ex.tool} {args_text(ex.args)}",
            *_claim_rows(ex),
            "   result",
            shown,
            "]",
        ]
    )


def dropped_text(ex: ToolExchange, recovery: str) -> str:
    """A refused image, dropped whole: pixels cannot sit in a text block, and elision is
    visible (invariant 16), so what went and how to get it back are said."""
    head = f"[dropped · {_label(ex.tool, _subject(ex.args))} · {_shape_text(ex.shape)}"
    return "\n".join([f"{head} · {_NO_FINDING}", *_claim_rows(ex), f"   {recovery} ]"])


def superseded_text(ex: ToolExchange, newer: ToolExchange) -> str:
    head = f"[superseded · {_label(ex.tool, _subject(ex.args))}"
    head += f" · the same call at seq {newer.seq} is newer"
    rows = _claim_rows(ex)
    if not rows:
        return head + "]"
    return "\n".join([head, *rows[:-1], rows[-1] + " ]"])
