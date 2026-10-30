"""Sessions, for the engine: create, open, list, find the latest, close.

A facade over the context package's session log (`context/store.py`, `context/session.py`).
The engine's session methods are this module plus composition. It holds the sessions this
process has open for writing, one `Session` per id, because a log has one writer and two
handles on it in one process would be two.

**Internal failures never escape**: a session that is missing, corrupt, open in another
process, or unwritable is `SessionError`, and its message names the path and the last
durable seq — *session broken* without them leaves the person nothing to inspect.

**A session's title is its latest own `title` line**: the `title` role's name for it,
written once after its first turn (`naming.py`), or the person's (`/rename`). With none, it is
D100's rule: the first line of the session's first user turn, cut at `SESSION_TITLE_WIDTH`
with a visible `…`. A fork's title is its own: its parent's lines are read through, never its
parent's name.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import TypeAdapter, with_config

from chickadee.context.questions import open_reclaimed
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session, SessionState, Turn
from chickadee.context.store import (
    LOCK_NAME,
    LOG_NAME,
    SESSION_ID_RE,
    SessionCorrupt,
    SessionEvent,
    SessionExists,
    SessionLocked,
    SessionLog,
    SessionMissing,
    SessionStoreError,
    list_sessions,
    session_dir,
    sessions_root,
)
from chickadee.context.vocabulary import LINE, PayloadError, SessionKind
from chickadee.durable import lock_is_held
from chickadee.errors import SessionError
from chickadee.ids import new_session_id

if TYPE_CHECKING:
    from chickadee.engine import SessionInfo
    from chickadee.state.roots import Roots

__all__ = [
    "SESSION_TITLE_WIDTH",
    "TITLE",
    "SessionTitle",
    "Sessions",
    "named",
    "new_session_id",
    "record_title",
    "title_of",
]

SESSION_TITLE_WIDTH: Final = 60
"""A title is at most this many characters, the last of them `…` when it was cut."""


def title_of(text: str) -> str | None:
    """D100 rule 1: the first line, cut at the width with a visible `…`. None for a turn
    with no words, which renders as *untitled*, never as an empty string."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not line:
        return None
    if len(line) <= SESSION_TITLE_WIDTH:
        return line
    return f"{line[: SESSION_TITLE_WIDTH - 1]}…"


TITLE: Final[SessionKind] = "title"


@with_config(LINE)
@dataclass(frozen=True)
class SessionTitle:
    """The session's name: the `title` role's, or the person's (`/rename`)."""

    KIND: ClassVar[SessionKind] = TITLE
    text: str
    """Cut as D100 cuts a first line (`title_of`)."""
    by: Literal["model", "person"]


_TITLE: Final = TypeAdapter(SessionTitle)


def record_title(session: Session, title: SessionTitle) -> SessionEvent:
    """Write `title`, refused when it has no words: a name is never an empty string."""
    if title_of(title.text) != title.text:
        raise PayloadError("a title is one line of words, at most SESSION_TITLE_WIDTH long")
    return session.append(title)


def named(
    state: SessionState, session_id: str, payload: Callable[[SessionEvent], bytes]
) -> SessionTitle | None:
    """The session's latest own `title` line, or None when it has none."""
    found: SessionTitle | None = None
    for event, _record in state.lines():
        if event.kind == TITLE and event.session_id == session_id:
            found = _TITLE.validate_json(payload(event))
    return found


def _title(
    state: SessionState, session_id: str, payload: Callable[[SessionEvent], bytes]
) -> str | None:
    """Its name; with none, D100's first line of its own first user turn, cut. A fork's
    read-through turns and names are its parent's, so a fork is titled by its own."""
    name = named(state, session_id, payload)
    if name is not None:
        return name.text
    for event, record in state.lines():
        own = event.session_id == session_id
        if own and isinstance(record, Turn) and record.role == "user":
            return title_of(record.text)
    return None


class Sessions:
    """Every session under one root, and the ones this process holds open."""

    def __init__(self, root: Path, folds: ReplayCache | None = None) -> None:
        """`folds` is the engine's one cache of session folds, which the
        engine passes; without it, one is built here."""
        self.root = root
        self._held: dict[str, Session] = {}
        self._guard = threading.Lock()
        # `list()` and `latest()` describe every session; replaying each unchanged log on
        # every call would read the whole history to learn one timestamp.
        self._folds = folds if folds is not None else ReplayCache(root)

    @staticmethod
    def at(roots: Roots) -> Sessions:
        """The repository's sessions, repo-local beside `runs/`."""
        return Sessions(sessions_root(roots))

    # -- writing --------------------------------------------------------------------------

    def create(self, session_id: str | None = None) -> Session:
        """A new session, open for writing. With no id, one is minted."""
        sid = session_id if session_id is not None else new_session_id()
        with self._guard, self.described(sid):
            session = Session.create(self.root, sid)
            self._held[sid] = session
            return session

    def open(self, session_id: str) -> Session:
        """The session open for writing: the one this process already holds, or the log
        reopened under its lock, a torn tail repaired and every line checked."""
        with self._guard:
            held = self._held.get(session_id)
            if held is not None:
                return held
            with self.described(session_id):
                session = open_reclaimed(self.root, session_id)
            self._held[session_id] = session
            return session

    def held(self, session_id: str) -> Session | None:
        with self._guard:
            return self._held.get(session_id)

    def close(self, session_id: str) -> None:
        """Flush and release one session. Closing one that is not held does nothing: the
        history outlives the process either way."""
        # Under the guard throughout: released between the pop and the close, a concurrent
        # `open` would find nothing held and try the file lock this close still holds.
        with self._guard:
            session = self._held.pop(session_id, None)
            if session is not None:
                with self.described(session_id):
                    session.close()

    def close_all(self) -> None:
        with self._guard:
            held = list(self._held)
        for sid in held:
            self.close(sid)

    # -- reading --------------------------------------------------------------------------

    def ids(self) -> tuple[str, ...]:
        return tuple(list_sessions(self.root))

    def require(self, session_id: str) -> None:
        """Refuse an id no session has, as a read here would, without reading its log: the
        store's own listing, a directory scan (for a turn that must answer at once)."""
        with self.described(session_id):
            if session_id not in self.ids():
                raise SessionMissing(session_dir(self.root, session_id) / LOG_NAME)

    def info(self, session_id: str) -> SessionInfo:
        """The stored facts: read from the log without its lock, so a session another
        process holds open is still describable."""
        from chickadee.engine import SessionInfo

        state, own, log = self._read(session_id)
        episodes = state.episodes()
        return SessionInfo(
            session_id=session_id,
            last_seq=state.last_seq,
            last_run_id=episodes[-1].run_id if episodes else None,
            created_at=own[0] if own else "",
            updated_at=own[-1] if own else "",
            title=_title(state, session_id, log.payload),
        )

    def title(self, session_id: str) -> str | None:
        """Its name, or D100's first line of its own first user turn, cut."""
        state, _, log = self._read(session_id)
        return _title(state, session_id, log.payload)

    def list(self) -> tuple[SessionInfo, ...]:
        """Every session, the most recently written first. A line's `at` has second
        precision, so two sessions written within one second are ordered by their logs'
        modification times, which are finer; `latest()` must not be a coin toss."""
        written = [(self.info(sid), self._written_ns(sid)) for sid in self.ids()]
        written.sort(key=lambda w: (w[0].updated_at, w[1], w[0].session_id), reverse=True)
        return tuple(info for info, _ns in written)

    def states(self) -> Iterator[tuple[str, SessionState]]:
        """Every session's id and fold, read without its lock (`open_works`). A session
        that cannot be read is skipped: one damaged log never hides every other's work."""
        for sid in self.ids():
            try:
                state, _, _ = self._read(sid)
            except SessionError:
                continue
            yield sid, state

    def live(self, session_id: str) -> bool:
        """Whether a process holds the session open, this one included: its work is being
        worked on, so no new session takes it over."""
        if self.held(session_id) is not None:
            return True
        return lock_is_held(session_dir(self.root, session_id) / LOCK_NAME)

    def latest(self) -> SessionInfo | None:
        """The session written to most recently. One with no line yet has nothing to
        continue, so it is latest only when there is no other."""
        listed = self.list()
        return listed[0] if listed else None

    # -- internals ------------------------------------------------------------------------

    def _read(self, session_id: str) -> tuple[SessionState, tuple[str, ...], SessionLog]:
        """The fold, the `at` of each line the session wrote itself (a fork's read-through
        lines are its parent's), and the log, whose payloads a name is read from."""
        with self.described(session_id):
            log, state = self._folds.get(session_id)
        own = tuple(e.at for e in log.events() if e.session_id == session_id)
        return state, own, log

    @contextmanager
    def described(self, session_id: str) -> Generator[None]:
        """Every internal failure as `SessionError`, naming the path and the last seq that
        is durable. Public for the engine, which opens a session's writer on the
        conversation's build and creates a log, each through this one description."""
        if not SESSION_ID_RE.match(session_id):
            raise SessionError(
                f"{session_id!r} is not a usable session id: letters, digits, `_`, `.` "
                f"and `-`, starting with a letter or digit"
            )
        try:
            yield
        except (SessionStoreError, OSError) as exc:
            raise SessionError(self._describe(session_id, exc)) from exc

    def _describe(self, session_id: str, exc: SessionStoreError | OSError) -> str:
        path = session_dir(self.root, session_id)
        if isinstance(exc, SessionMissing):
            return f"session {session_id} does not exist: no log at {exc.path}"
        if isinstance(exc, SessionCorrupt):
            where = f" at seq {exc.seq}" if exc.seq is not None else ""
            durable = f"last durable seq {exc.seq - 1}" if exc.seq else "no durable seq known"
            return (
                f"session {session_id} is corrupt{where}: {exc.detail} ({exc.path}; {durable})"
            )
        if isinstance(exc, SessionExists):
            return (
                f"session {session_id} already exists "
                f"({path}; last durable seq {self._last_durable(session_id)})"
            )
        if isinstance(exc, SessionLocked):
            return (
                f"session {session_id} is open for writing in pid {exc.pid} on {exc.host} "
                f"({path}; last durable seq {self._last_durable(session_id)})"
            )
        return (
            f"session {session_id} could not be written: {exc} "
            f"({path}; last durable seq {self._last_durable(session_id)})"
        )

    def _written_ns(self, session_id: str) -> int:
        try:
            return (session_dir(self.root, session_id) / LOG_NAME).stat().st_mtime_ns
        except OSError:
            return 0

    def _last_durable(self, session_id: str) -> str:
        try:
            return str(SessionLog.read(self.root, session_id).last_seq)
        except (SessionStoreError, OSError):
            # Called while describing a failure — often this same unreadable log. Nothing
            # raised here may escape as anything but the SessionError being built.
            return "unknown"
