"""An episode's language servers (Amendment 1).

A language's server starts at the episode's first index pass that needs it and is kept until
the episode ends: each later pass tells it what changed (`conversation.LspConversation`) and
asks its questions, with no restart. It runs under the same wrap as a per-pass session. Nothing
it starts outlives the episode that owns it: `close` is called before the episode gives
its write lease back (`box/episode.Box`), and again, harmlessly, by the episode's exit stack.

- **Restart once, then per-pass (decision 5).** A server that exits, errs or stops answering is
  started again once; failing again, the language has no server for the rest of the episode and
  the caller asks a per-pass session instead (`ask` answers None).
- **At most so many, harness-wide (decision 7).** `SLOTS` counts the servers alive in this
  process across every episode. At the cap a language has no server, never waits.
- **Idle servers stop.** One not asked anything for `idle_s` is closed and started again on its
  next use, which is no failure: its files' verdicts stand.
- **The re-check (decision 4, A2).** The files a change marked stale are asked their verdicts on
  a worker thread, nearest first, each written on the worker's own SQLite connection, unless a
  later change marked the file again. A step never waits on it.
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from chickadee.errors import InfraError
from chickadee.graph import diagnostics, store
from chickadee.graph.conversation import LspConversation, Request, Supervisor
from chickadee.graph.diagnostics import Verdict
from chickadee.limits import limit
from chickadee.process.supervisor import Cancelled, scrub_env
from chickadee.profiles.model import LanguageProfile

__all__ = [
    "ANSWER_TIMEOUT_S",
    "RECHECK_MAX",
    "SLOTS",
    "EpisodeServers",
    "Recheck",
    "Server",
    "ServerSlots",
]

_log = logging.getLogger(__name__)

RECHECK_MAX: Final = 50
"""Dependents re-checked per change, nearest first (decision 4); the rest stay marked."""
ANSWER_TIMEOUT_S: Final = 120.0
"""How long one answer is waited on: a server's first verdict includes its start's analysis.
Past it the server is taken to have stopped answering (decision 5)."""
LIFETIME_S: Final = 24 * 3600
"""A server session's own deadline: an episode longer than this has its server restarted."""


class ServerSlots:
    """How many servers are alive in this process, across every episode (decision 7). The
    count is guarded by `_lock`; each server takes a slot when it starts and gives it back when
    it closes, whatever closed it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._alive = 0

    def take(self, most: int) -> bool:
        with self._lock:
            if self._alive >= most:
                return False
            self._alive += 1
            return True

    def give(self) -> None:
        with self._lock:
            self._alive = max(0, self._alive - 1)

    @property
    def alive(self) -> int:
        with self._lock:
            return self._alive


SLOTS: Final = ServerSlots()
"""The process's one count: every episode's servers take their slot here."""


Build = Callable[[LspConversation], Sequence[Request]]
"""What to ask, given the conversation as it stands: the questions are built under the server's
lock, so the documents they sync are the ones the server holds when it answers."""


class _Live:
    """One language's running server: its process, its conversation, and `lock`, which every
    use holds, so one question set is asked at a time and the conversation is never shared."""

    def __init__(
        self, conversation: LspConversation, stack: ExitStack, slots: ServerSlots, now: float
    ) -> None:
        self.lock = threading.Lock()
        self._conversation = conversation
        self._stack = stack
        self._slots = slots
        self.last_used = now
        """When it last answered: read by the idle stop, under the pool's lock."""
        self._closed = False

    def ask(self, build: Build, now: float) -> bool:
        """`build`'s questions, under the lock. False when the server failed answering them."""
        with self.lock:
            if self._closed:
                return False
            try:
                ok = self._conversation.ask(build(self._conversation))
            except (TimeoutError, OSError, ValueError, InfraError, Cancelled):
                ok = False
            self.last_used = now
            return ok

    def close(self) -> None:
        """`shutdown`, then the process reaped and its slot given back; once."""
        with self.lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conversation.stop()
            except (TimeoutError, OSError, ValueError, InfraError, Cancelled):
                pass
            finally:
                self._stack.close()
                self._slots.give()


@dataclass
class _Language:
    live: _Live | None = None
    restarts: int = 0
    given_up: bool = False


@dataclass(frozen=True)
class Server:
    """A language's server as the episode runs it: its profile, the argv under the harness's
    wrap, and its own name, the verdicts' source (the wrap's argv starts with the wrapper)."""

    profile: LanguageProfile
    argv: tuple[str, ...]
    name: str
    local: bool = False
    """A linter: its verdicts are the file's own, and no dependency re-checks them."""

    def settings(self, section: str) -> object:
        """Its configuration for `section` (`[lsp].settings`); a linter reads its own config."""
        return None if self.local else self.profile.lsp.configuration(section)

    @property
    def key(self) -> str:
        """Its state's key in an episode: a language has its server and its linter's."""
        return f"{self.profile.name}/{self.name}"


@dataclass(frozen=True)
class Recheck:
    """One stale file to ask about (decision 4): its server and the pass's generation the
    question belongs to, so a verdict from before a later change does not clear that change's
    mark (A2)."""

    path: str
    server: Server
    asked_at: int


@dataclass
class EpisodeServers:
    """Each language's server for one episode, and the re-check worker over them."""

    root: Path
    supervisor: Supervisor
    servers_max: int = field(default_factory=lambda: int(limit("lsp.servers_max").value))
    """The limits register's `lsp.servers_max` (decision 7), as the person's config sets it."""
    idle_s: float = field(default_factory=lambda: float(limit("lsp.idle_s").value))
    """The register's `lsp.idle_s` (decision 7)."""
    slots: ServerSlots = SLOTS
    answer_timeout_s: float = ANSWER_TIMEOUT_S
    clock: Callable[[], float] = time.monotonic
    _languages: dict[str, _Language] = field(default_factory=dict[str, _Language])
    _lock: threading.Lock = field(default_factory=threading.Lock)
    """Guards `_languages`, `_closed` and the worker's start: each server's state, by
    `Server.key` (a language's server and its linter's)."""
    _cancel: threading.Event = field(default_factory=threading.Event)
    """Set at close: a read in flight raises `Cancelled` within a poll, not at its timeout."""
    _queue: queue.Queue[Recheck | None] = field(default_factory=queue.Queue[Recheck | None])
    _queued: dict[tuple[str, str], int] = field(default_factory=dict[tuple[str, str], int])
    """Each queued path and server, and the newest generation it was asked at: a path marked
    again while it waits is asked with the later one, so its verdict can clear the later mark
    (A2). By server too: a file of a language with a linter is two questions."""
    _worker: threading.Thread | None = None
    _idle: threading.Thread | None = None
    _closed: bool = False

    # -- asking -----------------------------------------------------------------------------

    def ask(self, server: Server, build: Build) -> bool:
        """`build`'s questions to the language's server, started if it is not running. True
        when they were answered; False when the language has no server here (the cap, a failed
        start, or a server that failed twice), and the caller asks a per-pass session. A server
        that fails is restarted once and asked again."""
        for _attempt in range(2):
            live = self._live(server)
            if live is None:
                return False
            if live.ask(build, self.clock()):
                return True
            self._failed(server.key, live)
        return False

    def _live(self, server: Server) -> _Live | None:
        with self._lock:
            if self._closed:
                return None
            state = self._languages.setdefault(server.key, _Language())
            if state.given_up:
                return None
            if state.live is not None:
                return state.live
            if not self.slots.take(self.servers_max):
                # At the cap: no server this time, which is no failure (decision 7).
                return None
            started = self._start(server)
            if started is None:
                self.slots.give()
                self._count_failure(state)
                return None
            state.live = started
            if self._idle is None:
                self._idle = threading.Thread(
                    target=self._watch_idle, name="lsp-idle", daemon=True
                )
                self._idle.start()
            return started

    def _start(self, server: Server) -> _Live | None:
        """A new server under the wrap's argv, initialized, in the slot the caller took; None
        when it would not start, and the caller gives the slot back. Called under `_lock`."""
        profile = server.profile
        stack = ExitStack()
        try:
            session = stack.enter_context(
                self.supervisor.session(
                    server.argv,
                    cwd=self.root,
                    timeout_s=LIFETIME_S,
                    env=scrub_env(),
                    cancel=self._cancel,
                )
            )
            conversation = LspConversation(
                session,
                self.root,
                lambda rel: profile.language_id_for(rel) or profile.name,
                server.name,
                timeout_s=self.answer_timeout_s,
                settings=server.settings,
            )
            if conversation.start():
                return _Live(conversation, stack, self.slots, self.clock())
        except (TimeoutError, OSError, ValueError, InfraError, Cancelled):
            _log.info("the %s server did not start", profile.name, exc_info=True)
        stack.close()
        return None

    def _failed(self, name: str, live: _Live) -> None:
        live.close()
        with self._lock:
            state = self._languages[name]
            if state.live is live:
                state.live = None
                self._count_failure(state)

    @staticmethod
    def _count_failure(state: _Language) -> None:
        """Decision 5: once restarted, a second failure leaves the language per-pass."""
        state.restarts += 1
        if state.restarts > 1:
            state.given_up = True

    def stop_idle(self) -> None:
        """Close every server asked nothing for `idle_s` (decision 7). Its next use starts it
        again, which is not a failure."""
        now = self.clock()
        with self._lock:
            idle: list[_Live] = []
            for state in self._languages.values():
                live = state.live
                if live is not None and now - live.last_used >= self.idle_s:
                    idle.append(live)
                    state.live = None
        for live in idle:
            live.close()

    def _watch_idle(self) -> None:
        """Look for idle servers a few times per `idle_s`, until the episode ends."""
        interval = max(1.0, min(self.idle_s / 4, 30.0))
        while not self._cancel.wait(interval):
            self.stop_idle()

    # -- the re-check -----------------------------------------------------------------------

    def recheck(self, files: Sequence[Recheck]) -> None:
        """Queue each stale file for its verdict, in the order given (nearest first), at most
        `RECHECK_MAX` of them; a file already queued is not queued twice, and is asked at the
        newer generation. The worker starts on the first."""
        with self._lock:
            if self._closed:
                return
            fresh: list[Recheck] = []
            for item in files[:RECHECK_MAX]:
                key = (item.path, item.server.name)
                if key in self._queued:
                    self._queued[key] = max(self._queued[key], item.asked_at)
                else:
                    self._queued[key] = item.asked_at
                    fresh.append(item)
            if fresh and self._worker is None:
                self._worker = threading.Thread(
                    target=self._recheck_all, name="lsp-recheck", daemon=True
                )
                self._worker.start()
        for item in fresh:
            self._queue.put(item)

    def wait_rechecked(self, timeout_s: float | None = None) -> bool:
        """Until every queued file has been asked, or `timeout_s` passes: for a caller that
        must see the verdicts, never a step. Whether the queue
        emptied: a file still queued is unchecked, never clean."""
        done = self._queue.all_tasks_done
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with done:
            while self._queue.unfinished_tasks:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                done.wait(left)
        return True

    def _recheck_all(self) -> None:
        """The worker: each queued file's verdict asked, then written on this thread's own
        connection (SQLite's is per thread)."""
        conn = store.connect(self.root)
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        return
                    with self._lock:
                        asked_at = self._queued.pop(
                            (item.path, item.server.name), item.asked_at
                        )
                    found = self._verdict(item)
                    if found is not None:
                        diagnostics.record_rechecked(conn, found, asked_at)
                except sqlite3.Error:
                    # The index's writer held the database past the timeout: the file stays
                    # marked, and says STALE, until a later pass or re-check judges it.
                    _log.info("a re-check of %s was not written", item, exc_info=True)
                finally:
                    self._queue.task_done()
        finally:
            conn.close()

    def _verdict(self, item: Recheck) -> Verdict | None:
        into: list[Verdict] = []

        def build(conversation: LspConversation) -> Sequence[Request]:
            sha = conversation.sync(item.path)
            return [] if sha is None else [conversation.verdict(item.path, sha, into)]

        self.ask(item.server, build)
        return into[0] if into else None

    # -- the end ----------------------------------------------------------------------------

    def close(self) -> None:
        """Stop the re-check and every server, before the episode gives its lease back; once,
        and harmless again."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker, idle = self._worker, self._idle
            lives = [s.live for s in self._languages.values() if s.live is not None]
            for state in self._languages.values():
                state.live = None
        self._cancel.set()
        if worker is not None:
            self._queue.put(None)
            worker.join()
        if idle is not None:
            idle.join()
        for live in lives:
            live.close()
