"""What the screen reads from a server: the
session's stream from its first record, each run it opens, and the routes a record makes stale.

**The REPL's readers, not a second copy**. The session's stream is read by
`runs.pump`, and each `episode`'s run by `runs.RunFollower`, which already follows a run whose
journal is not written yet, one that never started and one whose stream broke. They block, so
they run on threads; **nothing here touches the screen's state**: every frame and every route
answer is handed to the caller's callbacks, which the app makes safe from any thread
(`App.call_from_thread`), so the feed is folded on the event loop alone.

**From seq 0.** The REPL follows from the session's last seq, since it prints what is new; the
panes are a fold of the whole session (the board, the spend, the stage), so the screen reads
every record.

**A route the server does not build** (`501 NOT_BUILT`, as a replay's `/budget`) leaves its
answer unknown, never zero ("nothing the facade lacks is pretended"), and is
handed on as `NotServed`, so a screen that needs it can say so. Any other failure of a route
or a stream is reported, never swallowed.

**An overlay's one-off reads** (`ask`): a route an overlay shows while it is open and the
screen does not hold (`/undoable`, the session list, another session's spend), read once on a
thread of its own, its answer or its failure handed back the same way.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Final

from chickadee_cli.runs import RunFollower, pump
from chickadee_cli.tui.feed import Jobs, NotServed, Recovered, Snapshot, Work
from chickadee_cli.tui.model import StaleRoute
from chickadee_serve.frames import ErrorCode, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.errors import WireError
from chickadee_wire.events import Frame, Link
from chickadee_wire.models import Liveness
from chickadee_wire.records import is_run_record

__all__ = ["NOT_SERVED_HERE", "START", "SessionReader"]

type Read = Callable[[Client, str], Snapshot]

_READS: Final[dict[str, Read]] = {
    StaleRoute.INFO: lambda c, sid: c.session_info(sid),
    StaleRoute.HEADER: lambda c, sid: c.header(sid),
    StaleRoute.BUDGET: lambda c, sid: c.budget(sid),
    StaleRoute.LIVENESS: lambda c, sid: c.liveness(sid),
    StaleRoute.HIDDEN: lambda c, sid: c.hidden(sid),
    StaleRoute.RUNG: lambda c, sid: c.models(sid),
    StaleRoute.HANDOFF: lambda c, sid: c.handoff(sid),
    StaleRoute.JOBS: lambda c, sid: Jobs(c.jobs(sid)),
    StaleRoute.WORK: lambda c, sid: Work(c.work(sid)),
}
"""Each route the screen holds an answer of, by its name or its `StaleRoute`."""

START: Final = tuple(
    name for name in _READS if name not in (StaleRoute.HANDOFF, StaleRoute.JOBS)
)
"""Read once when the screen opens; afterwards only as a record makes one stale. Not
`/handoff`: it runs git on the server, so it is read only while the diff screen shows. Not
`/jobs`: the session's records, read from the start, mark it stale at a job's first effect."""


NOT_SERVED_HERE: Final = "not served here"
"""What an overlay's read says when the server does not build its route (a replay's)."""


class SessionReader:
    """One session's frames and route answers, delivered to `on_frame`, `on_snapshot` and
    `on_lost(what, why, stream)`, each called from a reader thread."""

    def __init__(
        self,
        client: Client,
        session_id: str,
        *,
        on_frame: Callable[[Frame], None],
        on_snapshot: Callable[[Snapshot], None],
        on_lost: Callable[[str, str, bool], None],
        on_link: Callable[[Link], None] | None = None,
    ) -> None:
        """`on_link` hears the session stream reconnecting and coming back."""
        self._client = client
        self._sid = session_id
        self._on_frame = on_frame
        self._on_snapshot = on_snapshot
        self._on_lost = on_lost
        self._on_link = on_link
        self._stop = threading.Event()
        # A run's frames come from its follower's own thread: through `_deliver`, as every
        # other delivery, so one landing after `close` reaches no screen. The latest run only:
        # the screen draws the latest episode, and the session is read from its start.
        self._runs = RunFollower(
            client,
            lambda frame: self._deliver(on_frame, frame),
            lambda run, why: self._deliver(on_lost, f"run {run}'s stream", why, True),
            latest_only=True,
        )
        self._lock = threading.Lock()
        self._stale: set[str] = set()
        """Guarded by `_lock`: the routes to read again, each once however often named."""
        self._failed: set[str] = set()
        """Guarded by `_lock`: the routes whose last read failed, said lost until one
        answers."""
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self.refresh(START)
        for name, target in (("tui-session", self._follow), ("tui-routes", self._read)):
            thread = threading.Thread(target=target, name=name, daemon=True)
            self._threads.append(thread)
            thread.start()

    def refresh(self, routes: Iterable[str]) -> None:
        """Read `routes` again, soon. A route already waiting is read once."""
        with self._lock:
            self._stale.update(routes)
        self._wake.set()

    def follow(self, run_id: str, after: int = 0) -> None:
        """Follow run `run_id` past its record `after`: a run no `episode` record names, as one
        `/resume` continued. The REPL's own follower, so both follow it alike."""
        self._runs.follow(run_id, after)

    def ask[T](
        self,
        read: Callable[[Client], T],
        deliver: Callable[[T], None],
        failed: Callable[[str], None],
    ) -> None:
        """`deliver(read(client))` once, from a thread of its own; `failed(why)` if the route
        fails (`NOT_SERVED_HERE` when it is not built). Never touches the screen's state."""

        def run() -> None:
            try:
                value = read(self._client)
            except WireError as exc:
                why = NOT_SERVED_HERE if exc.code is ErrorCode.NOT_BUILT else exc.message
                self._deliver(failed, why)
                return
            except Exception as exc:  # a dropped connection: said, never swallowed
                self._deliver(failed, str(exc))
                return
            self._deliver(deliver, value)

        threading.Thread(target=run, name="tui-ask", daemon=True).start()

    def close(self) -> None:
        """Stop reading. A read in flight ends when the caller closes the client."""
        self._stop.set()
        self._wake.set()
        self._runs.close()

    # -- the reader threads ----------------------------------------------------------------

    def _follow(self) -> None:
        on_link = self._on_link
        watch = None if on_link is None else (lambda link: self._deliver(on_link, link))
        events = self._client.session_events(self._sid, 0, watch=watch)
        lost = pump(events, self._take, self._stop)
        if lost is not None:
            self._deliver(self._on_lost, "the session's stream", str(lost), True)

    def _take(self, frame: Frame) -> None:
        self._deliver(self._on_frame, frame)
        if isinstance(frame, RecordFrame) and not is_run_record(frame):
            self._runs.feed(frame)

    def _read(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                names, self._stale = self._stale, set()
            for name in sorted(names):
                if self._stop.is_set():
                    return
                self._read_one(name)

    def _read_one(self, name: str) -> None:
        try:
            value = _READS[name](self._client, self._sid)
        except WireError as exc:
            if exc.code is ErrorCode.NOT_BUILT:
                self._deliver(
                    self._on_snapshot, NotServed(name)
                )  # unknown, and said where needed
            else:
                self._failed_route(name, str(exc))
            return
        except Exception as exc:  # a dropped connection: said, never swallowed
            if not self._stop.is_set():
                self._failed_route(name, str(exc))
            return
        self._deliver(self._on_snapshot, value)
        with self._lock:
            was_lost = name in self._failed
            self._failed.discard(name)
        if was_lost:
            # It answers again: the screen stops saying it is lost.
            self._deliver(self._on_snapshot, Recovered(f"/{name}"))
        if isinstance(value, Liveness) and value.team_run is not None:
            # The run the team works on now, followed whether or not an `episode`
            # named it: a run Zeus's `coder` resumed goes on under its first id, and that
            # call's own `episode` names a run that never starts. Delivered after
            # the liveness, so the screen shows the run before its records reach it.
            self._runs.follow(value.team_run)

    def _failed_route(self, name: str, why: str) -> None:
        with self._lock:
            self._failed.add(name)
        self._deliver(self._on_lost, f"/{name}", why, False)

    def _deliver(self, call: Callable[..., None], *args: object) -> None:
        """`call(*args)`, unless the screen has gone: a read that ends after `close` has no
        one to tell, as `pump` treats a stream closed under its read. Not called at all once
        closed (the app's loop may have stopped); one racing `close` fails quietly."""
        if self._stop.is_set():
            return
        try:
            call(*args)
        except Exception:
            if not self._stop.is_set():
                raise
