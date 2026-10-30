"""The ephemeral channel: progress frames, dropped rather than buffered.

**A progress frame is a hint a correct client may discard**, so this channel is allowed to
lose them and is built to: each subscriber has a queue `PROGRESS_QUEUE_MAX` deep, and a frame
offered to a full queue is dropped, counted, and forgotten. Nothing waits on a slow client and
nothing grows for one. If dropping progress ever hurts, the admission test was applied
wrongly, and a shallow queue surfaces that in development instead of hiding it behind memory.

**Publishers are threads; subscribers are the server's event loop.** A turn's events arrive on
the engine's worker thread; a stream is an `async` generator. `publish` hands the frame to the
loop with `call_soon_threadsafe`, so the queues are only ever touched on the loop's thread and
need no lock of their own. The subscriber table is guarded by `_lock`, which both sides take.

Progress is per session: a client watching a session sees the progress of every turn on it,
whichever client started the turn. A client that connects late gets progress from now.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Final

from pydantic import JsonValue

from chickadee.client.live import (
    ContextUpdate,
    LiveEvent,
    Restarted,
    TextUpdate,
    ThoughtUpdate,
    ToolUpdate,
)
from chickadee.context.session import ToolLine
from chickadee.engine import TurnHandle
from chickadee.limits import tokens_for_chars
from chickadee.schema import ToolCall
from chickadee.zeus.loop import Event
from chickadee_serve.frames import KEEPALIVE_S, ProgressFrame

__all__ = [
    "KEEPALIVE_S",
    "PROGRESS_QUEUE_MAX",
    "ProgressHub",
    "Subscription",
    "TurnPumps",
    "describe",
]

log = logging.getLogger(__name__)

PROGRESS_QUEUE_MAX: Final = 64
"""Frames per subscriber before the next is dropped."""


@dataclass(eq=False)
class Subscription:
    """One stream's queue. Compared by identity: two subscribers with equal fields are two."""

    scope: str
    queue: asyncio.Queue[ProgressFrame]
    dropped: int = 0
    """Frames this subscriber lost to a full queue: a diagnostic, never read by a decision."""


@dataclass
class ProgressHub:
    depth: int = PROGRESS_QUEUE_MAX
    _loop: asyncio.AbstractEventLoop | None = None
    _subs: dict[str, set[Subscription]] = field(default_factory=dict[str, set[Subscription]])
    _lock: threading.Lock = field(default_factory=threading.Lock)
    """Guards `_subs` and `_loop`: taken by publisher threads and by the loop alike."""

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """The loop the subscribers live on; set as the server starts."""
        with self._lock:
            self._loop = loop

    def unbind(self) -> None:
        with self._lock:
            self._loop = None

    def publish(self, frame: ProgressFrame) -> None:
        """Offer `frame` to every subscriber of its scope, from any thread. Without a loop or
        a subscriber it is dropped."""
        with self._lock:
            loop = self._loop
            subs = tuple(self._subs.get(frame.scope, ()))
        if loop is None or not subs or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # the loop closed between the check and here
            loop.call_soon_threadsafe(_offer, subs, frame)

    @contextlib.asynccontextmanager
    async def subscribe(self, scope: str) -> AsyncIterator[Subscription]:
        """A queue of `scope`'s progress from now until the block ends."""
        sub = Subscription(scope, asyncio.Queue(maxsize=self.depth))
        with self._lock:
            self._subs.setdefault(scope, set()).add(sub)
        try:
            yield sub
        finally:
            with self._lock:
                peers = self._subs.get(scope)
                if peers is not None:
                    peers.discard(sub)
                    if not peers:
                        del self._subs[scope]

    def subscribers(self, scope: str) -> int:
        with self._lock:
            return len(self._subs.get(scope, ()))


def _offer(subs: tuple[Subscription, ...], frame: ProgressFrame) -> None:
    """On the loop's thread: put, or drop when full. Never waits."""
    for sub in subs:
        try:
            sub.queue.put_nowait(frame)
        except asyncio.QueueFull:
            sub.dropped += 1


# -- a turn's progress ------------------------------------------------------------------


class TurnPumps:
    """One pump per session: a thread that turns a running turn's events (`TurnHandle.events`)
    into progress frames on the hub. A turn queued behind a running one is delivered by the
    running turn's pump, so a second is never started for a session that has one.

    Guarded by `_lock`, which `start` takes: request threads call it as turns are posted."""

    def __init__(self, hub: ProgressHub) -> None:
        self._hub = hub
        self._lock = threading.Lock()
        self._running: dict[str, threading.Thread] = {}

    def start(self, scope: str, handle: TurnHandle) -> None:
        """`Served.on_turn`: pump `handle`'s events for `scope`, unless a pump already runs."""
        with self._lock:
            running = self._running.get(scope)
            if running is not None and running.is_alive():
                return
            thread = threading.Thread(
                target=self._pump, args=(scope, handle), name=f"progress-{scope}", daemon=True
            )
            self._running[scope] = thread
            thread.start()

    def _pump(self, scope: str, handle: TurnHandle) -> None:
        try:
            for event in handle.events():
                self._hub.publish(describe(scope, event))
        except Exception:  # progress is droppable: a failed pump loses hints, never records
            log.debug("a progress pump for %s stopped", scope, exc_info=True)


def describe(scope: str, event: Event) -> ProgressFrame:
    """A turn event as a progress frame: its kind and a short description for display, never a
    result's body (the record carries that). A live event carries the piece of the reply it
    streamed, which is all it is."""
    detail = event.payload
    if isinstance(detail, LiveEvent):
        return ProgressFrame(scope=scope, kind=event.kind, payload=_live(detail))
    payload: dict[str, JsonValue] = {"seq": event.seq}
    if isinstance(detail, ToolCall):
        payload["tool"] = detail.name
        payload["call_id"] = detail.id
    elif isinstance(detail, ToolLine):
        payload["call_id"] = detail.call_id
    elif event.kind == "error" and isinstance(detail, str):
        payload["text"] = detail  # why the turn ended: a reason, never a result's body
    return ProgressFrame(scope=scope, kind=event.kind, payload=payload)


def _live(event: LiveEvent) -> dict[str, JsonValue]:
    """`frames.ProgressKind`'s three live payloads, by the update they carry."""
    payload: dict[str, JsonValue] = {
        "role": event.role,
        "task": event.task,
        "reply": event.reply,
    }
    if event.calls == "python":
        # Said on every piece, so a viewer shows the reply's blocks as calls from its first.
        payload["calls"] = "python"
    match event.update:
        case TextUpdate(offset=offset, piece=piece):
            payload |= {"offset": offset, "piece": piece}
            payload["tokens"] = tokens_for_chars(offset + len(piece))
        case Restarted():
            payload["restart"] = True
        case ThoughtUpdate(elapsed_s=elapsed, offset=offset, piece=piece):
            payload |= {"elapsed_s": elapsed, "offset": offset, "piece": piece}
            payload["tokens"] = tokens_for_chars(offset + len(piece))
        case ToolUpdate(index=index, tool=tool, path=path, size=size):
            payload |= {"index": index, "tool": tool, "path": path, "size": size}
            payload["tokens"] = tokens_for_chars(size)
        case ContextUpdate(used=used, window=window, model=model, compacting=compacting):
            return {"used": used, "window": window, "model": model, "compacting": compacting}
    return payload
