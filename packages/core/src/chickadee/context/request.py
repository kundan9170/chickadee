"""The per-turn request: what is sent, and whether it fits.

**The log is the context.** `assemble` builds `messages[]` from the session log alone: each
cut's block and standing snapshot in order, then the pinned bodies the newest cut re-emits,
then every line from the newest boundary on, verbatim. `Ctx` and `fits` say whether a request
fits its window; `est_text` and `est_call` are a weight inside one, the estimate every
part of compaction measures by. Compaction builds on these and this module imports none of it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Literal

from chickadee.client.request import (
    PDF_MEDIA_TYPE,
    ChatRequest,
    DocumentPart,
    ImagePart,
    Message,
    Part,
    TextPart,
    ToolSchema,
    text_of,
)
from chickadee.context.document import document_est
from chickadee.context.fit import render_lines
from chickadee.context.lines import Attachment, CutLine, FrameLine, canonical_json
from chickadee.context.outline import image_est
from chickadee.context.results import SUPERSEDED_STUB
from chickadee.context.session import ToolLine, Turn
from chickadee.context.spans import Span
from chickadee.context.standing import escape_frames
from chickadee.edits.lines import split_lines
from chickadee.errors import HarnessBug
from chickadee.limits import tokens_for_chars, window

if TYPE_CHECKING:
    from chickadee.config import RungEndpoint
    from chickadee.context.session import SessionState
    from chickadee.context.terms import Vision

__all__ = [
    "FIT_MARGIN",
    "Assembly",
    "Ctx",
    "Piece",
    "WindowBasis",
    "args_text",
    "assemble",
    "attached_line",
    "est_call",
    "est_text",
    "fits",
    "pieces_of",
    "tool_line",
    "tool_piece",
    "turn_piece",
    "within_sight",
]


def turn_header(seq: int) -> str:
    """`[turn N]`, `N` the line's seq. Part of the turn's bytes from the first request
    that carries it, so it never edits a cached prefix; the model never numbers anything,
    it only cites."""
    return f"[turn {seq}]"


FIT_MARGIN: Final = 0.05
"""Undecided: started at 0.05, to be measured. Slack for the local
estimate disagreeing with the provider's tokenizer, applied to the estimated half only."""


WindowBasis = Literal["input_limit", "context_limit"]


@dataclass(frozen=True)
class Ctx:
    """The assembled context, as the thing §5.4 measures."""

    tokens: int
    """Local estimate of everything assembled — the prefix and the session's messages. A
    count of bytes held, never a projection."""
    window: int
    """The limit `tokens` is measured against, per `window_basis`."""
    window_basis: WindowBasis
    """Which limit the provider declared. `context_limit` means no separate input limit was
    published, so `window` is the total and the response reserve does more of the work."""
    max_output: int
    """`rung.max_output_tokens`: the output cap every request asks for, which is
    `RESPONSE_RESERVE`."""

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("a context window is positive")

    @staticmethod
    def of(rung: RungEndpoint, tokens: int) -> Ctx:
        basis: WindowBasis = "input_limit" if rung.input_limit is not None else "context_limit"
        return Ctx(tokens, window(rung), basis, rung.max_output_tokens)

    @property
    def headroom(self) -> int:
        """`window - RESPONSE_RESERVE`. The answer needs room too, and the reserve is a cap
        we set, so the subtraction is exact rather than an allowance."""
        return self.window - self.max_output


def fits(ctx: Ctx, pending_tokens: int) -> bool:
    """Whether the request goes out. Pure, and asked on both sides of a cut from this
    one definition, so the question that gates a request and the one that validates the cut
    cannot drift apart.

    The margin is on the estimated half only: `ctx.tokens` is already held, and inflating it
    would be margin on a measurement. Err on the side of cutting — a false "no" costs one
    compaction nobody sees; a false "yes" costs the request, and nothing is salvaged from a
    request the provider rejects."""
    return ctx.tokens + pending_tokens * (1 + FIT_MARGIN) < ctx.headroom


def est_text(text: str) -> int:
    """P-2's local estimate, `ceil(chars / 3.5)`. An estimate, never reported as a count."""
    return tokens_for_chars(len(text))


def args_text(args: Mapping[str, object]) -> str:
    return canonical_json(dict(args)).decode("utf-8")


def est_call(tool: str, args: Mapping[str, object]) -> int:
    """A tool call's weight inside the assistant message that makes it."""
    return tokens_for_chars(len(tool) + len(args_text(args)))


def est_tools(tools: Sequence[ToolSchema]) -> int:
    """A request's `tools` array: each tool's name and schema, weighed as a call is, its
    description, and its grammar's definition when it is sent decoded under one.
    The one measure of it: the per-call estimate, P-9's prefix term and the load's fit all
    take this, so what a request is planned to carry and what it is counted to carry agree."""
    return sum(
        est_call(t.name, t.schema)
        + est_text(t.description)
        + (est_text(t.grammar.definition) if t.grammar is not None else 0)
        for t in tools
    )


def request_tokens(request: ChatRequest) -> int:
    """The local estimate of everything a request sends: every message, its images by the one
    image rule (`image_est`, as `assemble` counts them), every call it records, and the tool
    schemas. What a box request is fitted by, and what a call that failed after it was sent is
    charged for its input (`client/charge.py`)."""
    total = 0
    for m in request.messages:
        total += est_text(text_of(m))
        if not isinstance(m.content, str):
            total += sum(image_est(p.data) for p in m.content if isinstance(p, ImagePart))
            total += sum(document_est(p.data) for p in m.content if isinstance(p, DocumentPart))
        total += sum(est_call(c.name, c.arguments) for c in m.tool_calls)
    return total + est_tools(request.tools)


@dataclass(frozen=True)
class Assembly:
    """`messages[]` after the system prefix, and its weight."""

    messages: tuple[Message, ...]
    tokens: int
    """A local estimate, images by the vendor formula on the downscaled size: a count of what
    is held, never the provider's measurement."""


@dataclass(frozen=True)
class Piece:
    message: Message
    tokens: int


def _user(text: str) -> Piece:
    return Piece(Message("user", text), est_text(text))


def assemble(state: SessionState, blob: Callable[[str], bytes]) -> Assembly:
    """The session as the model reads it:

        per cut: its block, then its standing snapshot written once, never edited
        then: the pinned bodies the newest cut lifted rebuilt at every cut
        then: every line from the newest boundary on verbatim

    Earlier blocks are never merged: merging would move them and rebuild the cache from
    there, where appending the next block edits only what the next cut replaces anyway.
    The pinned bodies follow the newest snapshot rather than preceding it, so a change to
    the pinned set moves no earlier block or notice — `unload` is an omission at the next
    rebuild, and costs nothing on its own."""
    pieces = pieces_of(state, state.cuts(), blob)
    return Assembly(tuple(p.message for p in pieces), sum(p.tokens for p in pieces))


IMAGE_OUT_OF_SIGHT: Final = "[an image, not sent: the model now on this rung cannot see images]"
PDF_OUT_OF_SIGHT: Final = (
    "[a PDF, not sent: the model now on this rung cannot read PDFs; its text layer reads by "
    "its handle]"
)


def within_sight(messages: tuple[Message, ...], vision: Vision) -> tuple[Message, ...]:
    """The messages with each image or PDF the rung's model cannot take put as a line saying
    so. The log keeps what was sent when it was sent; a `/rung` switch
    can since have moved the rung to a model that takes less, and a part it cannot take is a
    refused request on every turn after. The line stands where the part stood, and the
    handle that reads it on is already in the text beside it."""
    if vision.supports_images and vision.supports_documents:
        return messages
    return tuple(_in_sight(m, vision) for m in messages)


def _in_sight(message: Message, vision: Vision) -> Message:
    if isinstance(message.content, str):
        return message
    parts: list[Part] = []
    for p in message.content:
        if isinstance(p, ImagePart) and not vision.supports_images:
            parts.append(TextPart(IMAGE_OUT_OF_SIGHT))
        elif isinstance(p, DocumentPart) and not vision.supports_documents:
            parts.append(TextPart(PDF_OUT_OF_SIGHT))
        else:
            parts.append(p)
    content = tuple(parts)
    return message if content == message.content else replace(message, content=content)


def pieces_of(
    state: SessionState, cuts: Sequence[CutLine], blob: Callable[[str], bytes]
) -> list[Piece]:
    pieces: list[Piece] = []
    for cut in cuts:
        pieces.append(_user(cut.block))
        pieces.append(_user(cut.standing))
    if cuts:
        pieces.extend(_lifted_piece(state, seq, blob) for seq in cuts[-1].lifted)
    pieces.extend(_tail(state, cuts[-1].boundary if cuts else 1, blob))
    return pieces


def _tail(state: SessionState, start: int, blob: Callable[[str], bytes]) -> list[Piece]:
    """Every line from `start` on. A notice that landed while calls waited is held until the
    results are in: nothing may come between a `tool_use` and its `tool_result`."""
    pieces: list[Piece] = []
    waiting: set[str] = set()
    held: list[Piece] = []
    for _event, record in state.lines(start):
        if isinstance(record, Turn):
            piece = turn_piece(record, blob)
            if record.role == "harness" and waiting:
                held.append(piece)
                continue
            pieces.append(piece)
            if record.role == "assistant":
                waiting = {c.call_id for c in record.calls}
        elif isinstance(record, ToolLine):
            pieces.append(tool_piece(state, record, blob))
            waiting.discard(record.call_id)
            if not waiting:
                pieces.extend(held)
                held.clear()
        elif isinstance(record, FrameLine):
            # The fold refuses a frame while calls wait, so it never splits a pair.
            pieces.append(_user(record.framed()))
    pieces.extend(held)
    return pieces


def turn_piece(turn: Turn, blob: Callable[[str], bytes]) -> Piece:
    if turn.role == "user":
        # `[turn N]` is part of the turn's bytes from the first request that carries it, so
        # it never edits a cached prefix.
        words = f"{turn_header(turn.seq)}\n{turn.text}"
        if not turn.attachments:
            return _user(words)
        return _with_attachments(words, turn.attachments, blob)
    if turn.role == "assistant":
        calls = tuple(c.tool_call() for c in turn.calls)
        tokens = est_text(turn.text) + sum(est_call(c.tool, c.args) for c in turn.calls)
        return Piece(Message("assistant", turn.text, tool_calls=calls), tokens)
    if turn.notice is None:
        raise HarnessBug(f"harness turn {turn.seq} carries no notice")
    return _user(turn.notice.framed())


def attached_line(a: Attachment) -> str:
    """`[attached A-1 shot.png · image · 1440x900 · h_3f2a]` and, when the model was not sent
    the content itself, what it was sent instead. Never a caption."""
    head = f"[attached {a.label} · {a.handle.id}]"
    if a.delivery == "none":
        if a.kind == "image":
            return f"{head} (your model cannot see images)"
        return f"{head} (no text layer, and your model cannot read PDFs)"
    if a.kind == "pdf" and a.delivery in ("text", "outline") and a.text_handle is not None:
        return f"{head} text layer only: no images, no layout · {a.text_handle.id}"
    return head


def _with_attachments(
    words: str, attachments: Sequence[Attachment], blob: Callable[[str], bytes]
) -> Piece:
    """The person's words, then each attachment's line and what it was sent as: pixels and
    PDFs as parts, text numbered as `read` numbers it, or the outline with the handle to read
    on by. Text from a file is escaped, since only the person's own words may look like a
    frame."""
    parts: list[Part] = []
    text = [words]
    tokens = est_text(words)
    for a in attachments:
        line = attached_line(a)
        if a.delivery == "native":
            text.append(line)
            parts.append(TextPart("\n".join(text)))
            text = []
            data = blob(a.handle.content_sha)
            if a.kind == "image":
                parts.append(ImagePart(a.media_type, a.handle.content_sha, data))
            else:
                parts.append(DocumentPart(a.media_type, a.handle.content_sha, a.name, data))
            tokens += est_text(line) + a.est_tokens
            continue
        body = _attached_body(a, blob)
        shown = f"{line}\n{escape_frames(body)}" if body else line
        text.append(shown)
        tokens += est_text(shown)
    if not parts:
        return _user("\n".join(text))
    if text:
        parts.append(TextPart("\n".join(text)))
    return Piece(Message("user", tuple(parts)), tokens)


def _attached_body(a: Attachment, blob: Callable[[str], bytes]) -> str:
    """What a text delivery shows: every line, numbered, or the outline and how to read on."""
    content = a.text_handle if a.text_handle is not None else a.handle
    if a.delivery == "outline":
        return f'{content.preview}\n(read(handle="{content.id}", lines=…) reads the rest)'
    if a.delivery != "text":
        return ""
    lines = split_lines(blob(content.content_sha).decode("utf-8"))
    return render_lines(lines, (Span(1, len(lines)),)) if lines else "(empty file)"


def _tool_text(state: SessionState, line: ToolLine) -> str:
    newer = state.superseded_by(line.call_id)
    if newer is not None:
        return SUPERSEDED_STUB.format(call_id=newer)
    if state.is_user_words(line.seq):
        return f"{turn_header(line.seq)}\n{line.text}"
    return line.text


def tool_piece(state: SessionState, line: ToolLine, blob: Callable[[str], bytes]) -> Piece:
    text = _tool_text(state, line)
    if line.image is None or state.superseded_by(line.call_id) is not None:
        return Piece(Message("tool", text, tool_call_id=line.call_id), est_text(text))
    parts, tokens = _with_image(text, line, blob)
    return Piece(Message("tool", parts, tool_call_id=line.call_id), tokens)


def _with_image(
    text: str, line: ToolLine, blob: Callable[[str], bytes]
) -> tuple[tuple[Part, ...], int]:
    """Text and pixels as parts: the downscaled bytes the live session sent, from the
    session's own blob store, named by the original's sha."""
    ref = line.image
    if ref is None:
        raise HarnessBug(f"tool line {line.seq} carries no image")
    data = blob(ref.sha)
    sha = line.handle.content_sha if line.handle is not None else ref.sha
    image: ImagePart | DocumentPart
    if ref.media_type == PDF_MEDIA_TYPE:
        name = line.handle.label if line.handle is not None else "document.pdf"
        image = DocumentPart(ref.media_type, sha, PurePosixPath(name).name, data)
    else:
        image = ImagePart(ref.media_type, sha, data)
    parts: tuple[Part, ...] = (TextPart(text), image) if text else (image,)
    return parts, est_text(text) + _image_est(line, data)


def _image_est(line: ToolLine, data: bytes) -> int:
    known = line.handle.shape.est_tokens if line.handle is not None else None
    if line.image is not None and line.image.media_type == PDF_MEDIA_TYPE:
        return document_est(data, known)
    return image_est(data, known)


def tool_line(state: SessionState, seq: int) -> ToolLine:
    line = state.line(seq)
    if line is None or not isinstance(line[1], ToolLine):
        raise HarnessBug(f"seq {seq} is not a tool result")
    return line[1]


def _lifted_piece(state: SessionState, seq: int, blob: Callable[[str], bytes]) -> Piece:
    """A pinned body re-emitted intact: framed, because its call went with the block and a
    body standing alone in a user message must not read as the user's words."""
    line = tool_line(state, seq)
    header = f"[pinned · from seq {seq} · re-emitted intact]"
    text = f"<harness_context>\n{header}\n{escape_frames(line.text)}\n</harness_context>"
    if line.image is None:
        return _user(text)
    parts, tokens = _with_image(text, line, blob)
    return Piece(Message("user", parts), tokens)
