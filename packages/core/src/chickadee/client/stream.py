"""Reading one streamed reply: the one home of the two silences and
of the person's stop, for both transports.

The vendor's SDK opens the stream and accumulates it; this module only decides when to
stop waiting for it. The stream is read on a thread of its own, and the caller's thread waits
for each event with a deadline, polling the cancel token as it waits:

- **The first-token limit** runs from sending the request to its first streamed event: a server
  that never starts (the model not loaded, the request queued, a hung server).
- **The token-gap limit** is the longest wait between two events once the reply has started: a
  server that stalls or dies mid-reply. A reply that keeps arriving is never cut off, however
  long it is.
- **A set cancel token** ends the wait within `POLL_S`, even before the first byte, because the
  wait is the caller's and never the socket read's.

Both limits are measured on `WAITED`, a clock that counts the machine's sleep, and no single
wait is longer than `WAKE_CHECK_S`: a request in flight when a laptop closed is past its limit
as soon as the machine wakes, never after another limit's time awake (`client/sleep.py`).

Whichever ends the wait closes the stream from outside the read, so the server stops
generating the reply. The reader thread is left to finish on its own: its socket read is
bounded by the SDK's own read timeout (`backstop_s`), so no abandoned thread outlives it.

Each event is read into its pieces on the reader thread (`pieces`, `client/live.py`): what they
streamed is counted there, so an attempt that does not finish says how much output it had
streamed, which the charge of a failed call reads (decision 5); and they are handed to the
caller's `live` on the caller's thread, as its wait takes each event. What the stream
accumulated is read on the reader thread too (`final`), after its last event: the snapshot is
never touched from two threads.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from chickadee.client.live import Live, Piece, hand, streamed_chars
from chickadee.client.sleep import WAITED
from chickadee.limits import tokens_for_chars

__all__ = [
    "POLL_S",
    "Broke",
    "Silences",
    "Silent",
    "StreamEnded",
    "StreamStopped",
    "Streamed",
    "read_stream",
]


POLL_S: Final = 0.05
"""How often a wait looks at the cancel token: Esc ends a request within this, at most."""

WAKE_CHECK_S: Final = 1.0
"""The longest single wait with no cancel token: a queue's wait runs on a clock that stops in
sleep, so the loop looks at `WAITED` this often to see a limit the sleep used up."""


@dataclass(frozen=True)
class Silences:
    """A model's two limits (`RungEndpoint.first_token_s`, `token_gap_s`), in seconds."""

    first_token_s: float
    token_gap_s: float

    @property
    def backstop_s(self) -> float:
        """The SDK's own read timeout: never shorter than either limit, or the SDK would end a
        wait this module still allows (a queued request's headers wait past the token gap).
        It bounds only the reader thread a stopped wait leaves behind."""
        return max(self.first_token_s, self.token_gap_s)


@dataclass(frozen=True)
class Streamed:
    """A stream read to its end: what `final` read of it, None when it carried no event, and
    the characters its events streamed."""

    final: object
    chars: int

    @property
    def streamed_tokens(self) -> int:
        """The output streamed, at P-2's local rate: what an attempt whose stream ended and
        whose reply was not a completion is billed with."""
        return tokens_for_chars(self.chars)


class StreamEnded(Exception):
    """The stream ended before its reply did. `opened`: the provider had answered the request
    with its stream's headers, so it had taken the request on; `started`: an event had arrived;
    `chars`: what the events streamed, counted from their pieces."""

    def __init__(self, detail: str, *, opened: bool, started: bool, chars: int) -> None:
        self.opened = opened
        self.started = started
        self.chars = chars
        super().__init__(detail)

    @property
    def streamed_tokens(self) -> int:
        """The output streamed before the end, at P-2's local rate: an estimate, as the charge
        of a failed call says it is."""
        return tokens_for_chars(self.chars)


class Silent(StreamEnded):
    """No event within a limit: `first_token` before the first, `token_gap` after it."""

    def __init__(
        self,
        kind: Literal["first_token", "token_gap"],
        waited_s: float,
        *,
        opened: bool,
        chars: int,
    ) -> None:
        self.kind: Literal["first_token", "token_gap"] = kind
        what = "no first token" if kind == "first_token" else "no token"
        super().__init__(
            f"{what} within {waited_s:g}s",
            opened=opened,
            started=kind == "token_gap",
            chars=chars,
        )


class StreamStopped(StreamEnded):
    """The cancel token was set while the reply was awaited."""

    def __init__(self, *, opened: bool, started: bool, chars: int) -> None:
        super().__init__("cancelled in flight", opened=opened, started=started, chars=chars)


class Broke(StreamEnded):
    """The SDK raised while opening or reading the stream: `cause` is what it raised, which the
    transport classifies (a status, a connect error, a body that was not a stream)."""

    def __init__(
        self, cause: BaseException, *, opened: bool, started: bool, chars: int
    ) -> None:
        self.cause = cause
        super().__init__(str(cause), opened=opened, started=started, chars=chars)


class SdkStream(Protocol):
    """An SDK stream helper's stream: its events, and a close that ends the request."""

    def __iter__(self) -> Iterator[object]: ...
    def close(self) -> None: ...


class _Done:
    """The reader's last item: what `final` read of the stream, None if no event arrived."""

    def __init__(self, result: object, chars: int) -> None:
        self.result = result
        self.chars = chars


class _Event:
    def __init__(self, chars: int, pieces: tuple[Piece, ...]) -> None:
        self.chars = chars
        self.pieces = pieces


class _Reader[S: SdkStream]:
    """The reader thread's side. `stream` is published once the SDK has opened it, under
    `lock`, which also guards `abandoned`: a stream opened after the caller stopped waiting is
    closed by the reader at once, one opened before it is closed by the caller."""

    def __init__(self) -> None:
        self.items: queue.Queue[_Event | _Done | BaseException] = queue.Queue()
        self.lock = threading.Lock()
        self.stream: S | None = None
        self.abandoned = False

    def run(
        self,
        open_stream: Callable[[], AbstractContextManager[S]],
        pieces: Callable[[object], tuple[Piece, ...]],
        final: Callable[[S], object],
    ) -> None:
        try:
            with open_stream() as stream:
                with self.lock:
                    self.stream = stream
                    late = self.abandoned
                if late:
                    return
                chars = 0
                seen = False
                for event in stream:
                    seen = True
                    said = pieces(event)
                    chars += streamed_chars(said)
                    self.items.put(_Event(chars, said))
                self.items.put(_Done(final(stream) if seen else None, chars))
        except BaseException as exc:  # carried to the caller's thread, raised there
            self.items.put(exc)

    @property
    def opened(self) -> bool:
        """Whether the SDK has opened the stream: the provider answered with its headers."""
        with self.lock:
            return self.stream is not None

    def abandon(self) -> None:
        """The caller stopped waiting: close the stream if it is open, else mark it so the
        reader closes it when it opens. A close that fails has nothing left to stop."""
        with self.lock:
            self.abandoned = True
            stream = self.stream
        if stream is not None:
            # A close racing the read's own end has nothing left to stop.
            with contextlib.suppress(Exception):
                stream.close()


def read_stream[S: SdkStream](
    open_stream: Callable[[], AbstractContextManager[S]],
    *,
    pieces: Callable[[object], tuple[Piece, ...]],
    final: Callable[[S], object],
    silences: Silences,
    cancel: threading.Event | None,
    live: Live | None = None,
    clock: Callable[[], float] = WAITED,
) -> Streamed:
    """Open the stream (`open_stream`, the SDK's manager: entering it sends the request), wait
    for each of its events under the two silences and `cancel`, and return what `final` read of
    it after its last event (None when the stream carried no event at all) with what it
    streamed. `pieces` reads one event into what it said (`client/live.py`); each is handed to
    `live`, when given, on this thread, in order.

    Raises `Silent`, `StreamStopped` or `Broke`, each saying whether the stream had opened,
    whether an event had arrived, and how many characters the events streamed."""
    reader: _Reader[S] = _Reader()
    thread = threading.Thread(
        target=reader.run, args=(open_stream, pieces, final), name="model-stream", daemon=True
    )
    thread.start()
    started = False
    chars = 0
    last = clock()
    while True:
        limit = silences.token_gap_s if started else silences.first_token_s
        left = last + limit - clock()
        if left <= 0:
            reader.abandon()
            raise Silent(
                "token_gap" if started else "first_token",
                limit,
                opened=reader.opened,
                chars=chars,
            )
        try:
            item = reader.items.get(
                timeout=min(left, POLL_S if cancel is not None else WAKE_CHECK_S)
            )
        except queue.Empty:
            item = None
        # The token is read after the wait, so an ending already waiting (the reply done, or
        # the SDK's error) is taken as what happened, and between events, so a reply that
        # keeps arriving is stopped too.
        if item is None or isinstance(item, _Event):
            if item is not None:
                started, chars, last = True, item.chars, clock()
                if live is not None:
                    live = hand(live, item.pieces)
            if cancel is not None and cancel.is_set():
                reader.abandon()
                raise StreamStopped(opened=reader.opened, started=started, chars=chars)
            continue
        if isinstance(item, _Done):
            return Streamed(item.result, item.chars)
        if isinstance(item, Exception):
            raise Broke(item, opened=reader.opened, started=started, chars=chars) from item
        raise item  # not an Exception: KeyboardInterrupt and the like, never classified
