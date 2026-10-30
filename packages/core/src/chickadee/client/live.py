"""A reply as it streams: the pieces a client hands the caller's `live`, and
the one fold that turns them into what a frontend draws.

**The pieces** are what one streamed event said, typed: text of the reply, reasoning on the
model's separate channel, a tool call's arguments as they grow, and a new attempt beginning.
Each transport reads its own events
into them (`openai_client._pieces`, `anthropic_client._pieces`), once: what an attempt streamed,
which the charge of a failed call reads, is counted from the same pieces (`streamed_chars`), so
the reading of an event has one home per transport.

**`Live` is called on the caller's thread,** between the stream's events (`stream.read_stream`),
never on the stream's reader thread: what it touches is the caller's.

**The fold, `LiveReply`,** is one model call's live state: the reply's text so far and its
offset, how long the model has been thinking, and each tool call's name, path and size. It
emits `Update`s, batched (`FLUSH_S`) so a reply of thousands of tokens is tens of updates, not
thousands: the progress channel keeps a short queue per client and drops past it. Each text
update carries its offset, so a client that lost one sees the gap. Nothing here
is journaled: the finished reply is recorded as it always was.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "FLUSH_S",
    "ContextUpdate",
    "Live",
    "LiveEvent",
    "LiveReply",
    "LiveSink",
    "NewAttempt",
    "Piece",
    "Restarted",
    "Speaker",
    "Text",
    "TextUpdate",
    "Thought",
    "ThoughtUpdate",
    "ToolArgs",
    "ToolUpdate",
    "Update",
    "hand",
    "speaking",
    "streamed_chars",
]

_log = logging.getLogger(__name__)

FLUSH_S: Final = 0.1
"""The longest a piece waits before an update carries it: ten updates a second at most, each
kind, however fast the model writes."""


@dataclass(frozen=True)
class Text:
    """Text of the reply."""

    text: str


@dataclass(frozen=True)
class Thought:
    """Reasoning, on the model's separate channel (`reasoning_content`, a thinking block)."""

    text: str


@dataclass(frozen=True)
class ToolArgs:
    """A native tool call's arguments as they arrive. `index` names the call in the reply;
    `name` is set on the piece where the provider first sends it."""

    index: int
    text: str
    name: str | None = None


@dataclass(frozen=True)
class NewAttempt:
    """Attempt `n` (from 2) began: what the earlier attempt streamed is void."""

    n: int


Piece = Text | Thought | ToolArgs | NewAttempt
Live = Callable[[Piece], None]
"""What a caller passes with a request (`ChatRequest.live`) to be handed each piece."""


def hand(live: Live, pieces: Iterable[Piece]) -> Live | None:
    """`pieces` handed to `live`; None when it raised, so its caller stops calling it. What it
    draws is a hint: its failure costs the drawing, never the reply."""
    try:
        for piece in pieces:
            live(piece)
    except Exception:
        _log.warning("a live view failed; the reply goes on without it", exc_info=True)
        return None
    return live


def streamed_chars(pieces: Iterable[Piece]) -> int:
    """The characters `pieces` streamed, all of them output: text, reasoning and arguments."""
    return sum(len(p.text) for p in pieces if not isinstance(p, NewAttempt))


# -- the fold ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class TextUpdate:
    """`piece` is the reply's text from `offset`: a client holding less sees a gap."""

    offset: int
    piece: str


@dataclass(frozen=True)
class ThoughtUpdate:
    """The model has been thinking `elapsed_s`; `piece` is its reasoning from `offset`."""

    elapsed_s: float
    offset: int
    piece: str


@dataclass(frozen=True)
class ToolUpdate:
    """Tool call `index` of the reply: its tool, the path it names once one can be read, and
    the bytes of its arguments so far."""

    index: int
    tool: str
    path: str | None
    size: int


@dataclass(frozen=True)
class Restarted:
    """A new attempt began: everything shown of this call so far is void."""


@dataclass(frozen=True)
class ContextUpdate:
    """Zeus's context, for the status line: `used` tokens of `window` on
    `model`, the catalog model that answered, after his last answer; `compacting` while a cut
    runs, `used` then what it is cutting from."""

    used: int
    window: int
    model: str
    compacting: bool = False


Update = TextUpdate | ThoughtUpdate | ToolUpdate | Restarted | ContextUpdate

_PATH: Final = re.compile(r'"path"\s*:\s*("(?:[^"\\]|\\.)*")')
"""A `path` argument whose string has closed, read from arguments still streaming."""


def _path(arguments: str) -> str | None:
    found = _PATH.search(arguments)
    if found is None:
        return None
    try:
        value = json.loads(found.group(1))
    except ValueError:
        return None
    return value if isinstance(value, str) and value else None


@dataclass
class _Growing:
    tool: str = ""
    arguments: str = ""
    shown: tuple[str, str | None, int] | None = None


@dataclass
class LiveReply:
    """One model call's live state, fed its pieces (`feed`, a `Live`) and emitting `Update`s
    through `emit`. `close` flushes what is held. A structured answer (a plan, a review, a
    task's report) is a tool call in every role, so it shows as its tool line, never as text.


    Used on the caller's thread only, as `Live` is."""

    emit: Callable[[Update], None]
    clock: Callable[[], float] = time.monotonic
    _text: str = ""
    _sent: int = 0
    _thought: str = ""
    _thought_sent: int = 0
    _thinking_since: float | None = None
    _calls: dict[int, _Growing] = field(default_factory=dict[int, _Growing])
    _flushed: float = float("-inf")
    """When an update last went out: never, so the first piece goes at once."""

    def feed(self, piece: Piece) -> None:
        match piece:
            case NewAttempt():
                self._restart()
                return
            case Text(text=text):
                self._text += text
            case Thought(text=text):
                if self._thinking_since is None:
                    self._thinking_since = self.clock()
                self._thought += text
            case ToolArgs(index=index, text=text, name=name):
                call = self._calls.setdefault(index, _Growing())
                if name:
                    call.tool = name
                call.arguments += text
        if self.clock() - self._flushed >= FLUSH_S:
            self._flush()

    def close(self) -> None:
        """The call answered, or failed: emit what is held."""
        self._flush()

    def _restart(self) -> None:
        self._text, self._sent = "", 0
        self._thought, self._thought_sent, self._thinking_since = "", 0, None
        self._calls.clear()
        self.emit(Restarted())

    def _flush(self) -> None:
        self._flushed = self.clock()
        if self._thinking_since is not None and (
            self._thought_sent < len(self._thought) or not self._text
        ):
            piece = self._thought[self._thought_sent :]
            self.emit(
                ThoughtUpdate(
                    round(self._flushed - self._thinking_since, 1), self._thought_sent, piece
                )
            )
            self._thought_sent = len(self._thought)
        if self._sent < len(self._text):
            self.emit(TextUpdate(self._sent, self._text[self._sent :]))
            self._sent = len(self._text)
        for index, call in self._calls.items():
            if not call.tool:
                continue
            now = (call.tool, _path(call.arguments), len(call.arguments.encode()))
            if now != call.shown:
                call.shown = now
                self.emit(ToolUpdate(index, *now))


# -- a role's voice ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveEvent:
    """One update of one model call, labelled with who is speaking: `role` (`zeus`, `athena`,
    `worker`), the task a worker holds, and `reply`, the call's id, unique in the process, which
    every update of one call shares."""

    role: str
    task: str | None
    reply: str
    update: Update
    calls: Literal["native", "python"] = "native"
    """How the role calls its tools: a worker's `python` replies carry their
    calls as Python blocks in their text, which a viewer shows as calls from the first piece."""


LiveSink = Callable[[LiveEvent], None]
"""Where a role's live events go: the session's running turn (`zeus/conversation.py`). A
hint, droppable everywhere after this."""


@dataclass(frozen=True)
class Speaker:
    """One role's live voice: each model call it makes gets a fresh `LiveReply`, labelled."""

    sink: LiveSink
    role: str
    task: Callable[[], str | None] = lambda: None
    """The task the role holds as the call starts: a worker's, or the one Athena reviews."""
    calls: Literal["native", "python"] = "native"
    """How the role calls its tools, said on every event (`LiveEvent.calls`)."""

    def reply(self) -> LiveReply:
        task = self.task()
        reply = f"{self.role}-{uuid.uuid4().hex[:12]}"

        def emit(update: Update) -> None:
            try:
                self.sink(LiveEvent(self.role, task, reply, update, self.calls))
            except Exception:  # a hint: its failure costs the hint, never the call
                _log.warning("a live event was not delivered", exc_info=True)

        return LiveReply(emit)


@contextmanager
def speaking(speaker: Speaker | None) -> Generator[Live | None]:
    """One model call's live view: the `live` to send with it (None without a speaker), and
    what it holds flushed when the call ends, however it ends."""
    if speaker is None:
        yield None
        return
    reply = speaker.reply()
    try:
        yield reply.feed
    finally:
        reply.close()
