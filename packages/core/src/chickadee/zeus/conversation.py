"""One open session's conversation: its loop, the thread that runs it, and its phase.

A session open in an engine is a `Conversation`. It holds the Zeus loop's word queue and stop
(`Interrupts`), the episode a stop also reaches, the turn streaming now,
and one worker thread that builds the loop and drives it. The thread matters: what the loop
holds that belongs to one thread (the graph's SQLite connection) is made on it, used
only there, and released there when the conversation closes.

**The phases are a table, D158 decision 2,** with the rows its Amendment 1 changed (a
turn does not wait for the build, so a word can reach `BUILDING`, `BUILDING_STOPPED` or
`FAILED`). Every method below makes the transitions that table lists for it, and nothing else.
A pair the table does not list raises `HarnessBug`
rather than guessing, with one exception the table's reasoning missed: an `interrupt()` can
fetch a conversation from the registry an instant before its build fails, so a stop that
reaches `FAILED` does nothing, as it did before the lifecycle had names (Notes).

**One lock per conversation** guards the phase, the live turn, the episode and the build's
result. Under it, only leaf locks are taken: the `Interrupts` lock, a `LiveTurn`'s condition
(to finish it, never to push to it), and the worker's `submit`, kept under the lock so that
"not closed" and "the drive started" are one step. Nothing that can block or call back runs
under it: the build, a turn's step, the episode's interrupt, the worker's shutdown, releasing
resources. `Conversations`' guard is never held while a conversation's lock is taken.

`Conversations` is the engine's one registry of open sessions. It forgets a session when it
closes or its build fails, so a later `open` builds it again.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from chickadee.client.live import LiveEvent
from chickadee.context.attachment import CheckedFile
from chickadee.errors import HarnessBug, SessionError
from chickadee.hooks.fire import Hooks
from chickadee.isolation.types import remove_session_tmp, sweep_session_tmp
from chickadee.process.supervisor import bind_session
from chickadee.schema.isolation import WorkspaceMode
from chickadee.zeus.interrupt import Interrupts
from chickadee.zeus.loop import Event, live_event

if TYPE_CHECKING:
    from chickadee.agents.spawn import Agents
    from chickadee.budget.manager import CostLedger
    from chickadee.context.session import Session
    from chickadee.work_progress import WorkProgress
    from chickadee.zeus.jobs import JobTable

__all__ = [
    "Build",
    "Built",
    "Conversation",
    "Conversations",
    "Interruptible",
    "Lasting",
    "LiveTurn",
    "Phase",
]


log = logging.getLogger(__name__)


class Phase(StrEnum):
    BUILDING = "building"
    """The loop is being built on the worker."""
    BUILDING_STOPPED = "building_stopped"
    """A stop arrived during the build."""
    HELD = "held"
    """Built after such a stop: the next `turn()` queues its word and starts nothing."""
    IDLE = "idle"
    """Built, and no turn running."""
    RUNNING = "running"
    """The worker is delivering queued words."""
    STOPPING = "stopping"
    """Stopped mid-turn: when the turn ends, the conversation pauses with its words queued."""
    CONTINUING = "continuing"
    """Stopped, then a word typed: when the stopped turn ends, the conversation goes on."""
    CLOSED = "closed"
    """`close()` ran. Terminal."""
    FAILED = "failed"
    """The build raised. Terminal; the registry has forgotten it."""


_UNBUILT = frozenset({Phase.BUILDING, Phase.BUILDING_STOPPED, Phase.FAILED})
"""The phases with no loop: no episode is attached to one, and no turn is driven in one (a
word typed while building waits for the build)."""

_TURNING = frozenset({Phase.RUNNING, Phase.STOPPING, Phase.CONTINUING})
"""The phases in which a turn is streaming."""

_BUILDING = frozenset({Phase.BUILDING, Phase.BUILDING_STOPPED})
"""The phases in which the loop is being built."""


class _Next(StrEnum):
    """What a turn's boundary does (`_boundary`)."""

    WORD = "word"
    """Deliver the oldest queued word."""
    WAKE = "wake"
    """Open a turn a job's exit asked for."""
    AGAIN = "again"
    """Read the gate again: an exit was told after this boundary's read of the log."""
    END = "end"


class Interruptible(Protocol):
    """The episode a stop reaches: the engine's `CoderHandle`, without importing the engine."""

    def interrupt(self, hard: bool = False) -> None: ...


@dataclass(frozen=True)
class Built:
    """What a build returns: the parts of the conversation the engine reads back."""

    step: Callable[[], Iterator[Event]]
    """Deliver the oldest queued word as one loop turn: the ledger's refresh, then the turn.
    Called only on the worker, never under the lock."""
    session: Session
    """The session's writer: the one this engine holds while the session is open here, which
    the facade writes the person's own lines through."""
    ledger: CostLedger
    """The session's ledger, restored before every turn: the one Zeus's rung view is
    cut from, and the one the session's episodes charge."""
    refresh: Callable[[], None]
    """Restore `ledger` from the family's records: before every turn, and before an
    episode a host opens between turns."""
    stale: Callable[[], bool] = lambda: False
    """Whether the session's wiring has moved since this build: a `/rung` switch
    written since. The worker rebuilds the loop before the next word when it has, so every
    part composed from a rung's model follows the switch from that word on."""
    wake: Callable[[], Iterator[Event]] | None = None
    """A turn a job's exit opens: the ledger's refresh, then the loop's
    turn with no user line. `None` where nothing can wake the loop."""
    wake_ready: Callable[[], bool] = lambda: False
    """Whether a job's exit notice is on the log after Zeus's last answer, and the zeus ledger
    may start: read on the worker, never under the conversation's lock."""
    max_wakes: int = 0
    """`zeus.max_wakes`: wake turns since the person last spoke."""
    jobs: JobTable | None = None
    """The session's jobs: the episode's steps and serve's `/jobs` read the same."""
    root: Callable[[], Path] | None = None
    """Where Zeus's view is now, his worktree once it moved: what `/files` lists."""
    agents: Agents | None = None
    """The session's agents, the ones Zeus's `spawn_agent` runs: what a host reaches
    to run one round of a profile itself (`PacketEngine.spawn_agent`, the researcher's
    evaluation)."""
    work: Callable[[], WorkProgress | None] | None = None
    """Where the session's open work stands, as Zeus's status line reads it:
    each task across the work's runs, what `/session/{sid}/work` answers."""


@dataclass(frozen=True)
class Lasting:
    """What a rebuild does not replace: the session's writer, its jobs and
    its hooks. A `/rung` switch composes a new loop over the same three, so
    a job runs on across it and its exit is recorded through the writer that recorded its
    start, and the session's hooks are bound, and `session_start` fired, once."""

    session: Session
    jobs: JobTable
    hooks: Hooks


Build = Callable[["Conversation", ExitStack], Built]
"""The engine's composition, run on the conversation's worker. Each resource it opens is
registered on the stack as it is opened, so a build that raises part way releases exactly what
it opened, and `close()` releases the same list, last opened first."""


class LiveTurn:
    """One turn's events as they happen, and how it ended. Written by the worker, read by any
    number of hosts; the worker never waits for a reader, so a reader that stops reading costs
    the turn nothing."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._events: list[Event] = []
        self._done = False
        self._error: BaseException | None = None

    def push(self, event: Event) -> None:
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def finish(self, error: BaseException | None = None) -> None:
        with self._cond:
            self._done = True
            self._error = error
            self._cond.notify_all()

    def follow(self) -> Iterator[Event]:
        seen = 0
        while True:
            with self._cond:
                while seen >= len(self._events) and not self._done:
                    self._cond.wait()
                batch = self._events[seen:]
                seen = len(self._events)
            if not batch:
                return
            yield from batch

    def wait(self, timeout: float | None) -> None:
        with self._cond:
            if not self._cond.wait_for(lambda: self._done, timeout):
                raise TimeoutError(f"the turn did not finish within {timeout}s")
            error = self._error
        if error is not None:
            raise SessionError(f"the turn stopped on an error: {error}") from error


class Conversation:
    """One open session's loop, its worker, and the phase D158 decision 2 names.

    The person's words wait in the loop's own queue, `stops`, in the order they were typed.
    The worker hands the loop one at a time (a driver's turn, `ZeusLoop.turn()` with no text,
    through the build's `step`), and the step restores the ledger before each. The stop is held
    there too, and cleared only when the person starts a turn, so a stop that lands
    between two queued words holds."""

    def __init__(
        self, session_id: str, build: Build, forget: Callable[[Conversation], None]
    ) -> None:
        self.session_id = session_id
        self.stops = Interrupts()
        """The loop's word queue and stop, shared with the loop the build makes."""
        self._build = build
        self._forget = forget
        """Takes this conversation out of the registry; called before a failed build's
        waiters are woken, so a turn that arrives after the failure builds anew."""
        # The worker is bound to its session before anything runs on it, so every process
        # the session spawns names it; the binding is the thread's own.
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"zeus-{session_id}",
            initializer=bind_session,
            initargs=(session_id,),
        )
        self._resources = ExitStack()
        """What the current build opened. Read and replaced on the worker only (`_current`)."""
        self._lasting = ExitStack()
        """What `lasting` opened, once: closed after `_resources`, by `_release`."""
        self._kept: Lasting | None = None
        """`lasting`'s, on the worker only."""
        self._built = threading.Event()
        """Set once the build has ended, whichever way."""
        self._lock = threading.Lock()
        """Guards `_phase`, `_live`, `_episode`, `_opened`, `_failure` and `_turn_failed`.
        Leaf locks only beneath it; see the module docstring."""
        self._phase = Phase.BUILDING
        self._live: LiveTurn | None = None
        self._episode: Interruptible | None = None
        self._opened: Built | None = None
        self._failure: BaseException | None = None
        self._turn_failed: str | None = None
        """Why the last word's turn ended on an error: an `error` event's text, or what its
        step raised. Cleared as each word's turn starts."""
        self._wakes = 0
        """Wake turns opened since the person last spoke, under `_lock`."""
        self._wake_told = False
        """An exit was told while a turn ran, since the boundary last read the gate: the
        boundary reads it again rather than end (`_Next.AGAIN`). Under `_lock`."""

    @property
    def phase(self) -> Phase:
        with self._lock:
            return self._phase

    def show(self, event: LiveEvent) -> None:
        """A reply of any role as it streams, among the running turn's events: a hint,
        dropped when no turn runs (a run resumed with nobody's turn open), never journaled.
        Called from whichever thread makes the model call: Zeus's worker or a box role's."""
        with self._lock:
            live = self._live
        if live is not None:
            live.push(live_event(event))

    @property
    def running(self) -> bool:
        """Whether a turn streams or the loop is being built: what a switch waits for."""
        with self._lock:
            return self._phase in _TURNING or self._phase in _BUILDING

    @property
    def opening(self) -> bool:
        """Whether a turn waits on the build: the session is being opened *for* someone, as
        distinct from a host opening it to read. Liveness reads it."""
        with self._lock:
            return self._phase in _BUILDING and self._live is not None

    @property
    def turn_failed(self) -> str | None:
        """Why the last turn ended on an error, while nothing runs; None while a turn runs or
        when the last one ended cleanly. Kept only here, in memory: liveness reads it,
        and nothing writes it to the session's log."""
        with self._lock:
            return self._turn_failed if self._phase is Phase.IDLE else None

    # -- the build: the registry's opener -------------------------------------------------

    def start(self) -> None:
        """Start the build on the worker and return: a turn need not wait for it. A
        word typed meanwhile is queued, and the same worker job drives it once the build ends
        (`_build_then_turn`)."""
        self._worker.submit(self._build_then_turn)

    def build(self) -> None:
        """`start`, and wait for the build (`_build_then_turn` makes its transitions). A
        build that raises leaves `FAILED`: the registry forgets the session and every resource
        is released **before** any waiter is woken, and the opener gets the build's own
        exception, with a note if releasing failed too."""
        self.start()
        self._built.wait()
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def _build_then_turn(self) -> None:
        """The build, on the worker, and then what was asked of it while it ran.
        `BUILDING` becomes `RUNNING` with a turn pending, driven here on this thread, or `IDLE`
        with none. `BUILDING_STOPPED` becomes `HELD`, or `IDLE` with a turn pending, which is
        finished unrun and its words kept, as a word typed into `HELD` leaves it. A build
        that raises tells the pending turn why, on its stream."""
        try:
            opened = self._build(self, self._resources)
        except BaseException as exc:
            with self._lock:
                self._phase = Phase.FAILED
                self._failure = exc
                waiting, self._live = self._live, None
            try:
                try:
                    self._forget(self)
                    # On the worker, which opened them: see `close`.
                    self._release()
                finally:
                    self._worker.shutdown(wait=False)
            except BaseException as released:
                # Nothing waits on this job's result, so a release that fails is said here:
                # logged, and noted on the failure every waiter and the turn are given.
                log.error(
                    "session %s: releasing a failed build", self.session_id, exc_info=released
                )
                exc.add_note(f"releasing what the build opened failed too: {released!r}")
            finally:
                self._built.set()
                if waiting is not None:
                    _say_failed(waiting, exc)
            return
        with self._lock:
            self._opened = opened
            waiting = self._live
            if self._phase is Phase.BUILDING_STOPPED:
                # A stop during the build holds: its words wait for the next turn. A word
                # typed during the build was the turn the hold stopped, as one typed into
                # `HELD` is, so the conversation is idle again and the next turn delivers it.
                self._phase = Phase.HELD if waiting is None else Phase.IDLE
                self._live = None
            elif waiting is not None:
                self._phase = Phase.RUNNING
            else:
                self._phase = Phase.IDLE
            drive = self._phase is Phase.RUNNING
        self._built.set()
        if waiting is not None and not drive:
            waiting.finish()
        if drive and waiting is not None:
            self._drive(waiting)

    def wait_open(self) -> None:
        """Wait for a build another caller started; `SessionError` if it failed."""
        self._built.wait()
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise SessionError(
                f"session {self.session_id} could not be opened: {failure}"
            ) from failure

    def opened(self) -> Built | None:
        """The build's parts once it has ended; `None` when it failed or the conversation is
        closed. Never called from the build itself, which it would wait for."""
        self._built.wait()
        with self._lock:
            return None if self._phase in (Phase.FAILED, Phase.CLOSED) else self._opened

    # -- the person's side: any thread ----------------------------------------------------

    def turn(
        self,
        text: str,
        workspace: WorkspaceMode | None = None,
        attachments: tuple[CheckedFile, ...] = (),
    ) -> tuple[LiveTurn, bool]:
        """The person's words: queued, and delivered at the next boundary. Returns the
        turn that will deliver them and whether it was already running (`queued`)."""
        with self._lock:
            if self._phase is not Phase.FAILED:
                return self._turn_locked(text, workspace, attachments)
            failure = self._failure
        # The build failed before this word reached it (a fast failure, or a fetch an instant
        # before): the turn says why, as the turn pending on it does, so the caller sees one
        # answer however the race fell. The word is not queued. Built outside the lock, where
        # a turn is pushed to; `FAILED` is final, so the phase read above still holds.
        return _failed_turn(failure), False

    def _turn_locked(
        self, text: str, workspace: WorkspaceMode | None, attachments: tuple[CheckedFile, ...]
    ) -> tuple[LiveTurn, bool]:
        """`turn` in every phase but `FAILED`, under the lock."""
        phase = self._phase
        if phase is Phase.CLOSED:
            raise SessionError(f"the engine is closed; {self.session_id} is not open")
        if phase in _BUILDING:
            # Typed while the session builds: queued, and driven by the build's
            # own worker job once it ends; every word typed meanwhile joins that turn.
            self.stops.type(text, workspace, attachments)
            if self._live is not None:
                return self._live, True
            self._live = LiveTurn()
            if phase is Phase.BUILDING_STOPPED:
                # Typed after the stop: the build will hold, as `HELD` does, so the word
                # waits for the next turn and is said to be queued.
                return self._live, True
            self.stops.begin_turn()
            return self._live, False
        self.stops.type(text, workspace, attachments)
        if phase is Phase.HELD:
            # Stopped before it could start: the stop holds, and the word waits for the
            # next turn, which delivers it first.
            self._phase = Phase.IDLE
            stopped = LiveTurn()
            stopped.finish()
            return stopped, True
        if phase in _TURNING:
            live = self._live
            if live is None:
                raise HarnessBug(f"session {self.session_id} is {phase} with no turn")
            if phase is not Phase.RUNNING:
                # Typed after the stop and before the stopped turn ended: the person has
                # moved on, so the conversation goes on once it ends.
                self._phase = Phase.CONTINUING
            return live, True
        if phase is not Phase.IDLE:
            raise HarnessBug(f"a turn reached session {self.session_id} in {phase}")
        # IDLE. A stop asked for while nothing ran stopped nothing; this turn starts
        # unstopped.
        self.stops.begin_turn()
        live = LiveTurn()
        self._live = live
        self._phase = Phase.RUNNING
        try:
            self._worker.submit(self._drive, live)
        except RuntimeError as exc:
            # The worker was shut down under a race no lock order allows; the word stays
            # queued and nothing runs it.
            self._live = None
            self._phase = Phase.IDLE
            raise SessionError(f"the engine is closed; {self.session_id} is not open") from exc
        return live, False

    def interrupt(self, hard: bool = False) -> None:
        """The stop: soft at the loop's next tool boundary, hard with the token set now.
        Queued words are kept. The episode in flight gets it too, outside the lock."""
        with self._lock:
            phase = self._phase
            if phase is Phase.FAILED:
                # Fetched from the registry an instant before the build failed: there is
                # nothing to stop, as there was not before the phases had names.
                return
            if phase in (Phase.BUILDING, Phase.BUILDING_STOPPED):
                self._phase = Phase.BUILDING_STOPPED
            elif phase in _TURNING:
                self._phase = Phase.STOPPING
            self.stops.interrupt(hard)
            episode = self._episode
        if episode is not None:
            episode.interrupt(hard)

    def close(self) -> None:
        """Stop hard, interrupt the episode, join the worker and release what the build
        opened. Waits for a build in progress. Idempotent. Never from the worker itself, which
        it joins. A turn blocked in `InteractionPort.ask` is handed the hard stop's token,
        so a port that honours it returns and the call is answered cancelled; a port
        that ignores it is waited for."""
        self._built.wait()
        with self._lock:
            if self._phase in (Phase.FAILED, Phase.CLOSED):
                return
            self._phase = Phase.CLOSED
            self.stops.interrupt(hard=True)
            episode = self._episode
        if episode is not None:
            episode.interrupt(hard=True)
        # Released on the worker, after whatever it is running: what belongs to one thread
        # (the graph's SQLite connection) is closed on the thread that opened it.
        # Read on the worker, not here: a rebuild before this ran (`_current`) replaced it.
        released = self._worker.submit(self._release)
        self._worker.shutdown(wait=True)
        released.result()
        # The session's `/tmp` goes with it, once nothing it ran can write there.
        remove_session_tmp(self.session_id)

    # -- the worker's side ----------------------------------------------------------------

    def attach_episode(self, episode: Interruptible) -> None:
        """The box's `open`: hold the episode a stop must reach. A stop that landed while the
        run was being named reaches it now, soft or hard as it was asked."""
        with self._lock:
            if self._phase in _UNBUILT:
                raise HarnessBug(f"an episode opened on session {self.session_id} unbuilt")
            self._episode = episode
            stopping = self.stops.stopping()
            hard = self.stops.cancel.is_set()
        if stopping:
            episode.interrupt(hard)

    def lasting(self, open_: Callable[[ExitStack], Lasting]) -> Lasting:
        """The session's `Lasting`: opened by the first build into the conversation's own
        stack, and the same one handed to every rebuild (`_current`); released by `close`, or
        by a first build that fails, after the build's own resources. On the worker only."""
        if self._kept is None:
            self._kept = open_(self._lasting)
        return self._kept

    def wake_due(self) -> None:
        """A job's exit notice is on the log, from its waiter thread. It
        decides nothing: idle, it starts the worker on the gate; in a turn, the gate runs at
        the turn's boundary, which reads it again if this landed after its read; held,
        building, closed or failed, nothing opens."""
        with self._lock:
            if self._phase in _TURNING:
                self._wake_told = True
                return
            if self._phase is not Phase.IDLE:
                return
            live = LiveTurn()
            self._live = live
            self._phase = Phase.RUNNING
            try:
                self._worker.submit(self._drive, live)
            except RuntimeError:
                # Shut down under a race: the notice stays on the log for the next words.
                self._live = None
                self._phase = Phase.IDLE

    def detach_episode(self) -> None:
        """The run's report or abandon: no stop reaches it any more."""
        with self._lock:
            self._episode = None

    def _drive(self, live: LiveTurn) -> None:
        """Every queued word, oldest first, until none is left or a stop pauses the
        conversation. The boundary's decision and the end of the turn are one step under the
        lock `turn()` queues under, so a word typed as the turn ends either runs in it or
        starts the next."""
        try:
            while True:
                with self._lock:
                    if self._phase is Phase.CONTINUING:
                        self.stops.begin_turn()
                        self._phase = Phase.RUNNING
                # Words first; with none waiting, the one gate.
                at = self._boundary(self._may_wake())
                if at is _Next.AGAIN:
                    continue
                if at is _Next.END:
                    live.finish()
                    return
                built = self._current()
                # The oldest queued word, as one turn, or a turn the job's exit opens. Only
                # this thread takes from the queue, so the word seen above is the one taken.
                turn = built.wake if at is _Next.WAKE and built.wake is not None else built.step
                for event in turn():
                    if event.kind == "error" and isinstance(event.payload, str):
                        with self._lock:
                            self._turn_failed = event.payload
                    live.push(event)
        except BaseException as exc:
            with self._lock:
                if self._phase in _TURNING:
                    self._phase = Phase.IDLE
                self._live = None
                self._turn_failed = str(exc) or type(exc).__name__
            live.finish(exc)

    def _boundary(self, ready: bool) -> _Next:
        """What the boundary does, under the lock, as one step with the end of the turn (so a
        word typed as it ends either runs in it or starts the next, and an exit told as it
        ends is read again rather than lost). `ready` is the gate's half that reads the log
        (`_may_wake`), read before the lock."""
        with self._lock:
            running = self._phase is Phase.RUNNING
            if running and self.stops.has_input():
                self._turn_failed = None
                self._wakes = 0
                return _Next.WORD
            limit = self._opened.max_wakes if self._opened is not None else 0
            if ready and running and not self.stops.stopping() and self._wakes < limit:
                self._wakes += 1
                self._turn_failed = None
                return _Next.WAKE
            if running and self._wake_told:
                return _Next.AGAIN
            if self._phase in _TURNING:
                self._phase = Phase.IDLE
            self._live = None
            return _Next.END

    def _may_wake(self) -> bool:
        """The gate's half that reads the log: on the worker, outside
        the lock, since it takes the session's. Its other half, the phase and the stop, is
        read under the lock after it."""
        with self._lock:
            # Cleared before the read: an exit told after it sets it again (`_boundary`).
            self._wake_told = False
            opened = self._opened
            stopped = self.stops.stopping() or self.stops.has_input()
            spent = opened is None or self._wakes >= opened.max_wakes
        if stopped or spent or opened is None or opened.wake is None:
            return False
        return opened.wake_ready()

    def _current(self) -> Built:
        """The loop to deliver the next word with: the one built, or a fresh build when the
        session's wiring has moved since, or when the last rebuild raised. On the worker
        only. The old build's resources are released first, on this thread, which opened them;
        a rebuild that raises fails this word as a step does, and the next word tries again."""
        opened = self._opened
        if opened is not None and not opened.stale():
            return opened
        with self._lock:
            self._opened = None
        self._resources.close()
        self._resources = ExitStack()
        built = self._build(self, self._resources)
        with self._lock:
            self._opened = built
        return built

    def _release(self) -> None:
        """What the build opened, then what outlived its rebuilds (`Lasting`), which goes
        however the first went: the session's jobs before its writer. On the worker."""
        with self._lasting:
            self._resources.close()


def _say_failed(live: LiveTurn, failure: BaseException | None) -> None:
    """End a turn whose session could not be opened: one `error` event, then the cause."""
    why = (str(failure) or type(failure).__name__) if failure is not None else "unknown"
    live.push(Event("error", 0, f"the session could not be opened: {why}"))
    live.finish(failure if failure is not None else SessionError(why))


def _failed_turn(failure: BaseException | None) -> LiveTurn:
    live = LiveTurn()
    _say_failed(live, failure)
    return live


class Conversations:
    """The sessions this engine has open: the one registry."""

    def __init__(self, build: Build) -> None:
        self._build = build
        self._guard = threading.Lock()
        """Guards `_open` and `_closed`. Never held while a conversation's lock is taken, nor
        while a session is built: one session opening must not hold up another's turn."""
        self._open: dict[str, Conversation] = {}
        self._closed = False
        # What crashed sessions left in their `/tmp`, once per engine.
        sweep_session_tmp()

    def start(self, session_id: str) -> Conversation:
        """The session's conversation, its build started on its own worker if it was not
        open, without waiting for it: what a turn asks for, since the word is queued
        until the build ends. A failed build is reported on the turn's stream."""
        conversation, new = self._registered(session_id)
        if new:
            conversation.start()
        return conversation

    def open(self, session_id: str) -> Conversation:
        """The session's conversation, built on first use on its own worker. Every caller
        waits for the one build; a failed build is the opener's exception and every waiter's
        `SessionError`."""
        conversation, new = self._registered(session_id)
        if new:
            conversation.build()
        else:
            conversation.wait_open()
        return conversation

    def _registered(self, session_id: str) -> tuple[Conversation, bool]:
        """The session's conversation, and whether it was made here, unbuilt: whoever made it
        starts its build, outside the guard."""
        with self._guard:
            if self._closed:
                raise SessionError(f"the engine is closed; session {session_id} is not open")
            conversation = self._open.get(session_id)
            if conversation is not None:
                return conversation, False
            conversation = Conversation(session_id, self._build, self._forget)
            self._open[session_id] = conversation
            return conversation, True

    def get(self, session_id: str) -> Conversation | None:
        """The session's conversation if it is open here; never builds one."""
        with self._guard:
            return self._open.get(session_id)

    def close(self, session_id: str) -> None:
        """Close one session and forget it; a later `open` builds it again."""
        with self._guard:
            conversation = self._open.pop(session_id, None)
        if conversation is not None:
            conversation.close()

    def close_all(self) -> None:
        """Close every session; `open` refuses from now on."""
        with self._guard:
            self._closed = True
            conversations = list(self._open.values())
            self._open.clear()
        for conversation in conversations:
            conversation.close()

    def _forget(self, conversation: Conversation) -> None:
        with self._guard:
            if self._open.get(conversation.session_id) is conversation:
                del self._open[conversation.session_id]
