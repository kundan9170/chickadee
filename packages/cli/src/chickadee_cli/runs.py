"""Following streams: `pump`, the one loop that reads a session's or a
run's stream, and `RunFollower`, which follows each coder episode's journal beside the
session's own, for the REPL and the batch commands alike.

**A run is followed from the session's `episode` record**, which names it. Its records come
from `/run/{id}/events` on a thread of their own and go to the caller's `emit`, which renders
them (`render/stream.py`), until the run's `run_end`.

**An episode is recorded before its journal is written** (nothing is journaled until the
episode starts, and the planned path journals only after Athena has decomposed the design), so
a run the server does not know yet is asked again, quietly and with backoff, for as long as its
episode is live. A run stopped before it started (`BLOCKED{tier_above_approved}`) never writes
a journal: the follower stops asking once the episode **settles** (the session records the
coder call's result for that run, a newer episode starts, or the follower closes), after one
last ask. Only a stream that had started and then broke is reported lost; a run that
never started is not, since the coder's own result already says why.

**What settles a run is its end being said**: the coder call's result naming it with
its outcome (the run ended, or never started and says why), or a trigger-4 notice naming it (a
run handed over in the background ended). A result naming it with no outcome does not: the
`RUNNING` a background hand-over is answered with is written within seconds, long before a
planned run's journal, which waits for Athena's plan, and settling on it gave the run up before
there was anything to read. A `PAUSED`, or the session ending under the run, is no end either;
such a run is still settled by a newer episode or the follower's close.

**Every run, or the latest only.** The REPL prints each run it follows. A screen that shows
only the latest episode (the TUI) asks for `latest_only`: a newer run stops the stream of every
earlier one, so a session read from its start does not replay each past run's whole journal.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from typing import Final

from pydantic import ValidationError

from chickadee_cli.render.stream import RUN_ENDED
from chickadee_serve.frames import ErrorCode, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.events import Frame
from chickadee_wire.models import EpisodeRecord, HarnessTurnP, ToolRecord
from chickadee_wire.records import payload

__all__ = ["RETRY_MAX_S", "RETRY_S", "RunFollower", "pump"]

RETRY_S: Final = 0.2
"""The first wait before asking again for a run whose journal is not there yet."""
RETRY_MAX_S: Final = 2.0
"""The backoff's ceiling: a planned run can take minutes to write its first line."""


def pump(
    events: Iterable[Frame],
    on_frame: Callable[[Frame], None],
    stop: threading.Event,
    until: Callable[[Frame], bool] = lambda _frame: False,
) -> Exception | None:
    """Hand each frame of `events` to `on_frame` until `stop` is set or a frame meets `until`.
    Returns the exception that ended the stream (a lost connection, a route's refusal), or
    None when it was stopped or done. A stream closed under its read by a stop ends in an
    exception too; that one is not returned, since the reader asked for it."""
    try:
        for frame in events:
            if stop.is_set():
                return None
            on_frame(frame)
            if until(frame):
                return None
    except Exception as exc:
        return None if stop.is_set() else exc
    return None


def _run_end(frame: Frame) -> bool:
    return isinstance(frame, RecordFrame) and frame.kind == "run_end"


def _ended(frame: RecordFrame, record: object) -> str | None:
    """The run whose end `record` says, if it says one (see the module): the coder call's
    result naming it with its outcome, or a trigger-4 notice naming it."""
    if isinstance(record, ToolRecord) and record.outcome is not None:
        return frame.run_id
    if isinstance(record, HarnessTurnP) and record.trigger == RUN_ENDED:
        return record.handle
    return None


class RunFollower:
    """The runs one session opens, each followed to its end. `feed` takes the session's
    records; `wait` returns when every run it started following has ended."""

    def __init__(
        self,
        client: Client,
        emit: Callable[[Frame], None],
        lost: Callable[[str, str], None] = lambda _run, _why: None,
        *,
        latest_only: bool = False,
    ) -> None:
        self._client = client
        self._latest_only = latest_only
        self._emit = emit
        self._lost = lost
        self._lock = threading.Lock()
        self._running: dict[str, threading.Thread] = {}
        """Guarded by `_lock`: each run being followed, by id."""
        self._settled: dict[str, threading.Event] = {}
        """Guarded by `_lock`: per run, set once its episode settled (see the module)."""
        self._stops: dict[str, threading.Event] = {}
        """Guarded by `_lock`: per run, set to stop reading it: on `close`, and with
        `latest_only` once a newer run is followed."""
        self._closed = threading.Event()

    def feed(self, frame: RecordFrame) -> None:
        """A session record: an `episode` starts following its run, once, and settles every
        earlier one; a run's end said (`_ended`) settles it."""
        if frame.kind not in ("episode", "tool", "turn"):
            return
        try:
            record = payload(frame)
        except ValidationError:
            return
        if isinstance(record, EpisodeRecord):
            self.follow(record.run_id)
            return
        ended = _ended(frame, record)
        if ended is None:
            return
        with self._lock:
            settled = self._settled.get(ended)
        if settled is not None:
            settled.set()

    def follow(self, run_id: str, after: int = 0) -> None:
        """Follow run `run_id` to its end from its record `after`, once, settling every earlier
        one: a run an `episode` record names, or one resumed, which writes none and whose
        records up to `after` the person has seen. Once while it is read: a run a newer
        one stopped (`latest_only`) is read again when asked, since a resumed run goes on under
        its first id."""
        with self._lock:
            if self._closed.is_set():
                return
            stop = self._stops.get(run_id)
            if run_id in self._running and (stop is None or not stop.is_set()):
                return
            for earlier in self._settled.values():
                earlier.set()
            if self._latest_only:
                for stop in self._stops.values():
                    stop.set()
            settled = self._settled[run_id] = threading.Event()
            stop = self._stops[run_id] = threading.Event()
            thread = threading.Thread(
                target=self._follow,
                args=(run_id, settled, stop, after),
                name=f"run-{run_id}",
                daemon=True,
            )
            self._running[run_id] = thread
        thread.start()

    def wait(self, timeout_s: float) -> bool:
        """Whether every run followed so far ended within `timeout_s`."""
        deadline = time.monotonic() + timeout_s
        with self._lock:
            threads = list(self._running.values())
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(t.is_alive() for t in threads)

    @property
    def started(self) -> bool:
        """Whether any run has been followed: a turn that opened none has no terminal of its
        own to report."""
        with self._lock:
            return bool(self._running)

    def close(self) -> None:
        """Stop following. A thread mid-read ends when the caller closes the client."""
        self._closed.set()
        with self._lock:
            for settled in self._settled.values():
                settled.set()
            for stop in self._stops.values():
                stop.set()

    def _follow(
        self, run_id: str, settled: threading.Event, stop: threading.Event, after: int
    ) -> None:
        seen = threading.Event()

        def emit(frame: Frame) -> None:
            seen.set()
            self._emit(frame)

        delay = RETRY_S
        while True:
            last = settled.is_set()
            ended = pump(self._client.run_events(run_id, after), emit, stop, _run_end)
            if ended is None:
                return
            not_yet = (
                isinstance(ended, WireError)
                and ended.code is ErrorCode.NOT_FOUND
                and not seen.is_set()
            )
            if not not_yet:
                self._lost(
                    run_id, ended.message if isinstance(ended, WireError) else str(ended)
                )
                return
            if last or stop.is_set():
                return  # it never started; its coder result says why
            settled.wait(delay)
            delay = min(delay * 2, RETRY_MAX_S)
