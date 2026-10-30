"""The session log: one append-only file per session, and its blobs.

A session outlives every episode inside it, so it cannot live in a run journal: the
constraint a user stated before the first `coder` call must still bind the fifth. This file
is that record, and the in-memory `SessionContext` (`session.py`) is a fold of it —
invariant 3 holds here exactly as it does for the run journal.

Layout, under the repo-local root (`31-state.md` §2.1 — *a session outlives an episode and
must survive worktree removal*):

    <repo_local>/sessions/<session_id>/session.jsonl
    <repo_local>/sessions/<session_id>/blobs/<sha256>
    <repo_local>/sessions/<session_id>/LOCK

**A body never sits inline.** Each line names its payload by sha, so the log stays scannable
and a payload is written once however often it is read. The blob is durable before the line
that names it, because a line pointing at nothing is corruption and a blob nothing points at
is only a few wasted bytes.

**The asymmetry of repair is the journal's.** A torn trailing line is
the one thing a crash produces, so truncating it is the one legal repair. A seq gap, an
unparseable line mid-file, or a payload that does not match its sha refuses to open, because
a session that silently skipped a constraint is worse than one that will not open.

**A fork reads through, never copies.** A child's first line is `fork`, naming
its parent and the seq it branched at; reading the child walks into the parent's lines up to
that seq and stops. The parent's log below the fork point is append-only and therefore
immutable, so a bound is all a fork needs — and the parent is never written.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from chickadee import clock
from chickadee.context.lines import (
    SESSION_KINDS,
    ForkOrigin,
    ForkP,
    PayloadError,
    SessionKind,
    parse,
    render,
)
from chickadee.durable import (
    AppendLog,
    BlobStore,
    BlobWriteFailed,
    FileLock,
    LockHeld,
    LockUnavailable,
    complete_lines,
    sync_dir,
)
from chickadee.schema import sha256_hex
from chickadee.state.roots import Roots

__all__ = [
    "BLOBS_DIR",
    "FSYNC_KINDS",
    "LOCK_NAME",
    "LOG_NAME",
    "SESSIONS_DIR",
    "SESSION_ID_RE",
    "SessionCorrupt",
    "SessionEvent",
    "SessionExists",
    "SessionLocked",
    "SessionLog",
    "SessionMissing",
    "SessionStore",
    "SessionStoreError",
    "SessionWriteError",
    "list_sessions",
    "session_dir",
    "sessions_root",
]

SESSIONS_DIR: Final = "sessions"
LOG_NAME: Final = "session.jsonl"
BLOBS_DIR: Final = "blobs"
LOCK_NAME: Final = "LOCK"

_SETTINGS: Final = ("cap", "model_switched", "pinned", "handoff_intended", "handoff_completed")
FSYNC_KINDS: Final[frozenset[SessionKind]] = frozenset({"turn", "spend", *_SETTINGS})
"""Each append flushes; turn boundaries fsync. A turn holds the person's words; a lost
`spend` is spend the cap stops counting; a `cap`, `model_switched` or `pinned` line is a
setting no restart may forget; a handoff's lines bracket its git change."""

SESSION_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
"""A session id is a directory name. Anything that could climb out of `sessions/` — a slash,
a leading dot — is refused rather than normalised."""

# -- errors -----------------------------------------------------------------------------


class SessionStoreError(Exception):
    """Base for every session-log failure. `sessions.py` maps these to the public
    `SessionError`, which carries the path and last durable seq."""


class SessionMissing(SessionStoreError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"no session log at {path}")


class SessionCorrupt(SessionStoreError):
    """Mid-file damage, a seq gap, or a payload that does not match its sha. Never repaired:
    a log edited until it parses is no longer the record."""

    def __init__(self, path: Path, seq: int | None, detail: str) -> None:
        self.path = path
        self.seq = seq
        self.detail = detail
        super().__init__(f"session log corrupt at seq={seq} in {path}: {detail}")


class SessionWriteError(SessionStoreError):
    """An append or blob write failed. Nothing was folded and no seq was handed out. After a
    failed append, fsync or payload blob the store writes nothing more until the session is
    reopened."""


class SessionExists(SessionWriteError):
    """`create` was given an id that already names a session. Nothing was written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"session {path.name} already exists at {path}")


class SessionLocked(SessionStoreError):
    """Another process holds this session open for writing. A log has one writer."""

    def __init__(self, path: Path, pid: int, host: str) -> None:
        self.path = path
        self.pid = pid
        self.host = host
        super().__init__(f"session {path.parent.name} is open in pid {pid} on {host}")


# -- the line ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionEvent:
    """One durable line of the session log.

    `seq` orders; `at` is for people. `session_id` is the log the line physically lives in —
    not written to the line, since the file already says it, and needed because a fork reads
    lines that live in its parent, whose blobs are there too.
    """

    seq: int
    at: str
    kind: SessionKind
    payload_sha: str
    run_id: str | None
    session_id: str

    def line(self) -> str:
        """The bytes on disk, without the newline. Key order is fixed so a line is stable."""
        return json.dumps(
            {
                "seq": self.seq,
                "at": self.at,
                "kind": self.kind,
                "payload_sha": self.payload_sha,
                "run_id": self.run_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


_now = clock.stamp
"""The session log's clock (D151's one clock), named here so a test can hold a session's time
still. For humans: ordering is `seq`, never this."""


def sessions_root(roots: Roots) -> Path:
    """Repo-local, beside `runs/`. `28-context.md` §5.1 still spells it
    `.chickadee/sessions/`, which predates ST-2's move of history into the git directory."""
    return roots.repo_local / SESSIONS_DIR


def session_dir(root: Path, session_id: str) -> Path:
    if not SESSION_ID_RE.match(session_id):
        raise ValueError(f"{session_id!r} is not a usable session id")
    return root / session_id


# -- reading ----------------------------------------------------------------------------


def _parse_line(path: Path, raw: str, expected: int, session_id: str) -> SessionEvent:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SessionCorrupt(path, expected, f"unparseable line: {exc}") from exc
    if not isinstance(data, dict):
        raise SessionCorrupt(path, expected, "a line is a JSON object")
    fields = cast("dict[str, object]", data)
    seq, at, kind = fields.get("seq"), fields.get("at"), fields.get("kind")
    sha, run_id = fields.get("payload_sha"), fields.get("run_id")
    if set(fields) != {"seq", "at", "kind", "payload_sha", "run_id"}:
        raise SessionCorrupt(path, expected, f"unexpected fields {sorted(fields)}")
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise SessionCorrupt(path, expected, "seq is not an integer")
    if seq != expected:
        raise SessionCorrupt(path, expected, f"seq gap: expected {expected}, got {seq}")
    if not isinstance(at, str) or not isinstance(sha, str) or len(sha) != 64:
        raise SessionCorrupt(path, seq, "at or payload_sha is malformed")
    if kind not in SESSION_KINDS:
        raise SessionCorrupt(path, seq, f"unknown kind {kind!r}")
    if run_id is not None and not isinstance(run_id, str):
        raise SessionCorrupt(path, seq, "run_id is not a string")
    return SessionEvent(seq, at, kind, sha, run_id, session_id)


def _read_lines(
    path: Path, first_seq: int, session_id: str, *, stop_at: int | None = None, offset: int = 0
) -> tuple[list[SessionEvent], int]:
    """Every complete line past byte `offset`, validated in order, and the byte just past the
    last one read. A torn tail is skipped here and removed from disk only by the writer, since
    a reader may be looking at someone else's live session."""
    with path.open("rb") as fh:
        fh.seek(offset)
        chunks = complete_lines(fh.read())
    events: list[SessionEvent] = []
    expected, end = first_seq, offset
    for chunk in chunks:
        if stop_at is not None and expected > stop_at:
            break
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SessionCorrupt(path, expected, f"line is not UTF-8: {exc}") from exc
        if not text.strip():
            raise SessionCorrupt(path, expected, "blank line")
        events.append(_parse_line(path, text, expected, session_id))
        expected += 1
        end += len(chunk) + 1
    return events, end


def _verified(log: Path, event: SessionEvent, raw: bytes) -> bytes:
    """A line's payload bytes, once they are the bytes its line names. Parsing them is the
    line's type's (`context.lines`)."""
    if sha256_hex(raw) != event.payload_sha:
        raise SessionCorrupt(log, event.seq, "payload sha mismatch")
    return raw


# -- the store --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ancestor:
    """A parent log this one reads through, and the last seq it may read."""

    session_id: str
    directory: Path
    upto: int


class SessionStore:
    """One session's log, open for appending. One writer; not thread-safe by design."""

    def __init__(
        self,
        root: Path,
        session_id: str,
        *,
        _lock: FileLock,
        _log: AppendLog,
        _events: list[SessionEvent],
        _ancestors: tuple[_Ancestor, ...],
        _created: bool = False,
    ) -> None:
        self.root = root
        self.session_id = session_id
        self.directory = session_dir(root, session_id)
        self._lock: FileLock | None = _lock
        self._log = _log
        """Unbuffered, and stopped by a failed write or fsync: its `stopped` is why
        this store writes nothing more."""
        self._events = _events
        self._ancestors = _ancestors
        self._created = _created
        """Made by `create`, so `discard` may remove it. A reopened session never is."""
        self._blobs = BlobStore(self.directory / BLOBS_DIR)

    @property
    def repaired_bytes(self) -> int:
        """Bytes of a torn tail removed on open; 0 when the log was clean."""
        return self._log.repaired_bytes

    # -- lifecycle ------------------------------------------------------------------------

    @staticmethod
    def create(
        root: Path,
        session_id: str,
        *,
        fork: tuple[str, int, str] | None = None,
    ) -> SessionStore:
        """A new, empty session — or, with `fork=(parent_id, at_seq, reason)`, a child whose
        first line is the `fork` line. The caller checks the safe point; the
        store checks only that the parent has the seq it is asked to branch at.

        A create that fails leaves nothing behind. A child without its `fork` line would
        open as an empty root, and a directory left over would keep the id taken, so the
        failed attempt could not simply be made again."""
        directory = session_dir(root, session_id)
        ancestors: tuple[_Ancestor, ...] = ()
        prior: list[SessionEvent] = []
        if fork is not None:
            parent_id, at_seq, _reason = fork
            ancestors, prior, _ = _read_ancestry(root, parent_id, at_seq)
        try:
            directory.mkdir(parents=True)
        except FileExistsError:
            raise SessionExists(directory) from None
        try:
            lock = _acquire(directory / LOCK_NAME)
        except BaseException:
            # Nothing else can hold this lock: `open` needs a log, and there is none yet.
            shutil.rmtree(directory, ignore_errors=True)
            raise
        store: SessionStore | None = None
        try:
            (directory / BLOBS_DIR).mkdir()
            (directory / LOG_NAME).touch()
            # The session's own entries, and its entry in `sessions/`, are durable before
            # any line or blob that relies on them.
            sync_dir(directory)
            sync_dir(directory.parent)
            store = SessionStore(
                root,
                session_id,
                _lock=lock,
                _log=AppendLog.open(directory / LOG_NAME),
                _events=prior,
                _ancestors=ancestors,
                _created=True,
            )
            if fork is not None:
                parent_id, at_seq, reason = fork
                store.append(
                    "fork", render(ForkP(ForkOrigin(parent_id, at_seq), reason, _now()))
                )
        except BaseException:
            if store is None:
                _remove(directory, lock)
            else:
                store.discard()
            raise
        return store

    @staticmethod
    def open(root: Path, session_id: str) -> SessionStore:
        """Reopen for appending: take the lock, repair a torn tail, validate every line."""
        directory = session_dir(root, session_id)
        log = directory / LOG_NAME
        if not log.is_file():
            raise SessionMissing(log)
        lock = _acquire(directory / LOCK_NAME)
        appending: AppendLog | None = None
        try:
            # Opening repairs a torn tail, before anything reads or appends.
            appending = AppendLog.open(log)
            ancestors, events, _ = _read_ancestry(root, session_id)
            return SessionStore(
                root,
                session_id,
                _lock=lock,
                _log=appending,
                _events=events,
                _ancestors=ancestors,
            )
        except BaseException:
            # Best effort: the error already propagating is the caller's, and a second one
            # raised here would replace it and skip the release.
            if appending is not None:
                with contextlib.suppress(OSError):
                    appending.close()
            lock.release()
            raise

    def close(self) -> None:
        if self._lock is None:
            return
        lock, self._lock = self._lock, None
        try:
            # Quiet on a store that already failed: it said so when it failed, and closing it
            # is how its caller gets to reopen.
            self._log.close()
        except OSError as exc:
            raise SessionWriteError(f"fsync failed on close: {exc}") from exc
        finally:
            lock.release()

    def discard(self) -> None:
        """Remove a session whose making did not finish: a fork without its `fork` line, or
        without the arrival notice §5.1.1 step 5 owes it, is not a session, and leaving one
        would keep its id taken. Only a store `create` made may be discarded, and only while
        it still holds the lock — a session that was reopened is history, never deleted."""
        if not self._created or self._lock is None:
            raise SessionWriteError(
                f"session {self.session_id} is not one being created, so it is not removed"
            )
        lock, self._lock = self._lock, None
        with contextlib.suppress(OSError):
            self._log.close()
        _remove(self.directory, lock)

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- reading --------------------------------------------------------------------------

    @property
    def last_seq(self) -> int:
        return self._events[-1].seq if self._events else 0

    @property
    def next_seq(self) -> int:
        """The seq the next append will take. For ids that embed it (`SUM-<seq>`), and valid
        only until that append — the id is handed out after it lands, never before."""
        return self.last_seq + 1

    @property
    def forked_from(self) -> tuple[str, int] | None:
        """The immediate parent and fork seq, or None for a root session — read from this
        session's own `fork` line. Not from the ancestry: a fork at or before its parent's
        own fork point reads straight through to the grandparent, so the last ancestor is
        then not the parent."""
        for event in self._events:
            if event.session_id == self.session_id:
                if event.kind != "fork":
                    return None
                log = self.directory / LOG_NAME
                return _fork_line(log, event, self.payload(event))
        return None

    def events(self) -> tuple[SessionEvent, ...]:
        """Every line this session reads, in seq order: the ancestry up to each fork point,
        then its own."""
        return tuple(self._events)

    def event(self, seq: int) -> SessionEvent | None:
        if 1 <= seq <= len(self._events):
            return self._events[seq - 1]
        return None

    def payload(self, event: SessionEvent) -> bytes:
        """One line's payload bytes, from the store of the log the line lives in — an
        ancestor's, for a line a fork reads through. Damage names that log and the line's
        seq. Its type parses them (`context.lines`)."""
        log = session_dir(self.root, event.session_id) / LOG_NAME
        try:
            raw = self.get_blob(event.payload_sha, owner=event.session_id)
        except SessionCorrupt:
            raise SessionCorrupt(
                log, event.seq, f"blob {event.payload_sha} is missing"
            ) from None
        return _verified(log, event, raw)

    def iter_payloads(self) -> Iterator[tuple[SessionEvent, bytes]]:
        for event in self._events:
            yield event, self.payload(event)

    # -- blobs ----------------------------------------------------------------------------

    def put_blob(self, content: bytes) -> str:
        """Content-addressed and write-once: this session's `durable.BlobStore`."""
        try:
            return self._blobs.put(content)
        except BlobWriteFailed as exc:
            raise SessionWriteError(str(exc)) from exc

    def get_blob(self, sha: str, *, owner: str | None = None) -> bytes:
        """A blob by sha, from `owner`'s store when given, else this session's and then each
        ancestor's. An ancestor's store is only ever read. A session blob is never pruned
        (`gc` prunes runs), so a miss is damage."""
        for directory in self._blob_dirs(owner):
            store = BlobStore(directory / BLOBS_DIR)
            if store.has(sha):
                return store.read(sha)
        raise SessionCorrupt(self.directory / LOG_NAME, None, f"blob {sha} is missing")

    def has_blob(self, sha: str) -> bool:
        return any(BlobStore(d / BLOBS_DIR).has(sha) for d in self._blob_dirs(None))

    def _blob_dirs(self, owner: str | None) -> list[Path]:
        own = [self.directory] + [a.directory for a in reversed(self._ancestors)]
        if owner is None or owner == self.session_id:
            return own
        return [a.directory for a in self._ancestors if a.session_id == owner] or own

    # -- writing --------------------------------------------------------------------------

    def append(
        self, kind: SessionKind, payload: bytes, *, run_id: str | None = None
    ) -> SessionEvent:
        """Write one line, its payload already rendered by its type (`context.lines.render`);
        fsync at turn boundaries. Nothing is buffered in this process, so
        what reached the file is all there is, and no later append can carry an earlier
        one's remainder.

        A failure raises and hands out no id. The log is cut back to the last line folded,
        so the record the caller was told failed is not in it, and the store stops writing:
        after a failed write or fsync what is durable is uncertain, and only a reopen —
        which repairs a torn tail and replays what is on disk — may write again. The reopen
        reuses the seq, legitimately, since nothing was handed out for it.
        """
        self._check_writable()
        if kind not in SESSION_KINDS:
            raise ValueError(f"unknown session line kind {kind!r}")
        try:
            sha = self.put_blob(payload)
        except SessionWriteError as exc:
            # Nothing reached the log, but the caller's step may already have: a constraint
            # lands before the notice that announces it. Writing on would leave a sequence
            # no caller asked for, so a failed payload stops the store as a failed line does.
            raise self._stop(str(exc), cut=False) from exc
        event = SessionEvent(self.next_seq, _now(), kind, sha, run_id, self.session_id)
        line = (event.line() + "\n").encode("utf-8")
        try:
            self._log.append(line, sync=kind in FSYNC_KINDS)
        except OSError as exc:
            raise self._stop(f"append failed at seq={event.seq}: {exc}", cut=True) from exc
        self._events.append(event)
        return event

    def sync(self) -> None:
        """Force what has been appended to disk, for a caller about to do something that must
        not outlive an unsynced record. A failed fsync stops the store as a failed append
        does, without the cut: every line it would remove was folded and handed out."""
        self._check_writable()
        try:
            self._log.sync()
        except OSError as exc:
            raise self._stop(
                f"fsync failed after seq={self.last_seq}: {exc}", cut=False
            ) from exc

    def _check_writable(self) -> None:
        if self._lock is None:
            raise SessionWriteError("session is closed")
        if self._log.stopped is not None:
            raise SessionWriteError(self._log.stopped)

    def _stop(self, what: str, *, cut: bool) -> SessionWriteError:
        """Stop writing, with the store's account of why, which the log reports on every
        later write. `cut` says a failed append preceded this, which the log has already put
        back to the last line folded, or tried to."""
        after = ""
        if cut and self._log.cut:
            after = f"; the log is cut back to seq {self.last_seq}"
        elif cut:
            after = "; the cut failed too: the log may hold that line, whole or torn"
        why = f"{what}{after}. Reopen the session to write again"
        self._log.stop(why)
        return SessionWriteError(why)


@dataclass(frozen=True)
class SessionLog:
    """A session's lines as a reader sees them, taken without its lock.

    Reading needs no permission from the writer: the log below any seq already written is
    append-only and therefore immutable, and a torn tail is skipped rather than repaired.
    This is how a fork checks its parent's fork point while the parent may still be
    open in another process — and why the parent is never touched by it."""

    root: Path
    session_id: str
    _events: tuple[SessionEvent, ...]
    consumed: int = 0
    """The bytes of its own log a whole read took its lines from: where `extend` reads on."""

    @staticmethod
    def read(root: Path, session_id: str, *, upto: int | None = None) -> SessionLog:
        """Every line `session_id` reads, or only those up to `upto`. Asking for a seq the
        log does not have is `SessionCorrupt`, as it is for a fork."""
        _, events, consumed = _read_ancestry(root, session_id, upto)
        return SessionLog(root, session_id, tuple(events), consumed)

    def extend(self) -> SessionLog:
        """This log with the complete lines its own file gained since it was read, read from
        `consumed` on. The lines it reads through from its ancestry never change."""
        log = session_dir(self.root, self.session_id) / LOG_NAME
        new, end = _read_lines(log, self.last_seq + 1, self.session_id, offset=self.consumed)
        return SessionLog(self.root, self.session_id, (*self._events, *new), end)

    @property
    def last_seq(self) -> int:
        return self._events[-1].seq if self._events else 0

    def events(self) -> tuple[SessionEvent, ...]:
        return self._events

    def payload(self, event: SessionEvent) -> bytes:
        """The payload bytes, from the store of the log the line lives in — an ancestor's,
        for a line a fork reads through."""
        directory = session_dir(self.root, event.session_id)
        try:
            raw = BlobStore(directory / BLOBS_DIR).read(event.payload_sha)
        except FileNotFoundError:
            raise SessionCorrupt(
                directory / LOG_NAME, event.seq, f"blob {event.payload_sha} is missing"
            ) from None
        return _verified(directory / LOG_NAME, event, raw)


# -- ancestry ---------------------------------------------------------------------------


def _fork_line(path: Path, event: SessionEvent, raw: bytes) -> tuple[str, int]:
    """The parent and seq a fork line names, read through the `fork` line's type."""
    try:
        fork = parse("fork", raw)
    except PayloadError as exc:
        raise SessionCorrupt(path, event.seq, f"fork line is malformed: {exc}") from None
    assert isinstance(fork, ForkP)
    return fork.forked_from.session_id, fork.forked_from.at_seq


def _first_line(directory: Path, session_id: str) -> tuple[SessionEvent | None, bytes | None]:
    """The first line of a log and its payload — enough to tell a fork from a root."""
    log = directory / LOG_NAME
    with log.open("rb") as fh:
        head = fh.readline()
    if not head.endswith(b"\n"):
        return None, None
    try:
        text = head.decode("utf-8")
        first = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionCorrupt(log, None, f"first line unreadable: {exc}") from exc
    if not isinstance(first, dict) or cast("dict[str, object]", first).get("kind") != "fork":
        return None, None
    seq = cast("dict[str, object]", first).get("seq")
    if not isinstance(seq, int):
        raise SessionCorrupt(log, None, "first line has no seq")
    event = _parse_line(log, text.rstrip("\n"), seq, session_id)
    blobs = BlobStore(directory / BLOBS_DIR)
    if not blobs.has(event.payload_sha):
        raise SessionCorrupt(log, event.seq, "fork payload blob is missing")
    # Checked as every other line's payload is: its sha here, then its type. A damaged fork
    # line is damage like any other, never a raw decode error past the store.
    return event, _verified(log, event, blobs.read(event.payload_sha))


def _read_ancestry(
    root: Path, session_id: str, upto: int | None = None, depth: int = 0
) -> tuple[tuple[_Ancestor, ...], list[SessionEvent], int]:
    """The lines `session_id` reads, oldest ancestor first, the ancestors it reads through, and
    the bytes of its own log read. All of them, or those a child at `upto` reads, `session_id`
    then the last ancestor. A chain, not a special case: each link is one fork line,
    and the read is a fold over it."""
    if depth > 256:
        raise SessionCorrupt(
            session_dir(root, session_id) / LOG_NAME, None, "fork chain too deep"
        )
    directory = session_dir(root, session_id)
    log = directory / LOG_NAME
    if not log.is_file():
        raise SessionMissing(log)
    event, payload = _first_line(directory, session_id)
    ancestors: tuple[_Ancestor, ...] = ()
    inherited: list[SessionEvent] = []
    first_seq = 1
    if event is not None and payload is not None:
        grand_id, grand_at = _fork_line(log, event, payload)
        if upto is not None and upto <= grand_at:
            # Everything up to `upto` is the grandparent's: this log's own lines, its fork
            # line first, all come after it. Reading them would hand the child lines past
            # the seq it forked at.
            return _read_ancestry(root, grand_id, upto, depth + 1)
        ancestors, inherited, _ = _read_ancestry(root, grand_id, grand_at, depth + 1)
        first_seq = grand_at + 1
    own, end = _read_lines(log, first_seq, session_id, stop_at=upto)
    lines = inherited + own
    if upto is None:
        return ancestors, lines, end
    if (lines[-1].seq if lines else 0) < upto:
        raise SessionCorrupt(log, upto, f"parent {session_id} has no seq {upto} to fork at")
    return (*ancestors, _Ancestor(session_id, directory, upto)), lines, end


# -- the lock ---------------------------------------------------------------------------


def _acquire(lock: Path) -> FileLock:
    """The session's `durable.FileLock`, for the journal's reason: the kernel drops it
    when the holder dies, so a crashed session needs no cleanup and a live one cannot be
    evicted. Its errors in the store's terms."""
    try:
        return FileLock.acquire(lock)
    except LockHeld as held:
        raise SessionLocked(lock, held.pid, held.host) from None
    except LockUnavailable as exc:
        raise SessionWriteError(str(exc)) from exc


def _remove(directory: Path, lock: FileLock) -> None:
    """Delete a session directory this process made, then let its lock go — in that order,
    so nothing can open what is being removed."""
    try:
        shutil.rmtree(directory, ignore_errors=True)
    finally:
        lock.release()


def list_sessions(root: Path) -> Sequence[str]:
    """Session ids under `root`, sorted. Finding a session's children is a directory scan,
    never a log fact: discovery is not provenance."""
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            p.name
            for p in root.iterdir()
            if (p / LOG_NAME).is_file() and SESSION_ID_RE.match(p.name)
        )
    )
