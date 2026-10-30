"""The single writer.

One `Journal` per run, not thread-safe by design. Everything else in the harness is a
projection of what lands here, so an event that was not durably written did not happen —
which is why the fsync policy is a spec detail rather than a tuning knob.
"""

from __future__ import annotations

import contextlib
import json
from types import TracebackType
from typing import Any, Final, TypedDict, cast

from chickadee.clock import stamp
from chickadee.durable import (
    AppendLog,
    BlobStore,
    BlobWriteFailed,
    FileLock,
    LockHeld,
    LockUnavailable,
    make_dir,
    write_atomic,
)
from chickadee.journal import paths
from chickadee.journal.effects import EFFECT_EVENTS
from chickadee.journal.errors import (
    JournalCorrupt,
    JournalWriteError,
    ResumeRefused,
)
from chickadee.journal.projection import Fold, RunProjection
from chickadee.schema import EventKind
from chickadee.schema.journal import ENVELOPE_VERSION
from chickadee.state.roots import Roots

# Fsync on a task's boundaries: its delegation, each round's report, Athena's review
# of it, a plan, and the run's end. The crash window is then bounded by one round, which is
# exactly what resume redoes anyway.
FSYNC_KINDS: frozenset[EventKind] = (
    frozenset(
        {
            EventKind.TASK_DELEGATED,
            EventKind.TASK_REPORTED,
            EventKind.TASK_REVIEWED,
            EventKind.PLAN_WRITTEN,
            EventKind.PLAN_OUTGROWN,
            EventKind.RUN_END,
        }
    )
    # Every effect phase is durable before the effect happens.
    # `effect_intended` reaching disk is what makes an interrupted effect recoverable rather
    # than a mystery. The phases are stated once, by the recorder.
    | EFFECT_EVENTS
)

FSYNC_EVERY = 10


class _Dump(TypedDict):
    by_alias: bool
    exclude_none: bool


AS_STORED: Final[_Dump] = {"by_alias": True, "exclude_none": True}
"""How a journal line is serialized: field aliases (`pass`), and no `null` (an unset field is
absent). The one rule for the stored form. A reader that hands a line out as stored
(`scopes.run_page`) dumps with it too, so a client sees the line's own field names."""


class Journal:
    """Append-only writer plus blob store for one run."""

    def __init__(
        self,
        roots: Roots,
        run_id: str,
        *,
        _log: AppendLog,
        _fold: Fold | None = None,
        _lock: RunLock | None = None,
    ) -> None:
        self.roots = roots
        self.run_id = run_id
        self._lock = _lock
        # The run as folded so far: every line read back on resume, then every line
        # appended, each once it is durable. Its last seq is the journal's.
        self._fold = _fold or Fold()
        self._seq = self._fold.projection().last_seq
        self._since_fsync = 0
        self._closed = False
        self._blobs = BlobStore(paths.blobs_dir(roots, run_id))
        self._path = paths.journal_path(roots, run_id)
        # Unbuffered, and stopped by a failed write or fsync: a line the caller was told
        # failed cannot be flushed later by the next append or by `close()` .
        self._log = _log

    # -- lifecycle ----------------------------------------------------------

    @staticmethod
    def create(roots: Roots, run_id: str, config_snapshot: dict[str, Any]) -> Journal:
        """Start a fresh run. The config snapshot is archive discipline: a
        result nobody can reproduce the configuration of is not a result.

        A run id names one run, started once. The LOCK is taken before anything is written,
        so a `create` refused for a live run has touched nothing of it, and a run that
        already has a journal, finished or not, is refused: continuing one is
        `open_for_resume`'s job. Written over, its config would describe a run that never
        started, and a seq counter starting at 0 would append seq 1 after its last line,
        leaving a journal no reader accepts."""
        # Synced as they are made, as the session store's are: the run's entries are
        # durable before any line or blob that relies on them. Making a directory that
        # exists writes nothing, so a live run's is safe to ask for before its lock.
        make_dir(paths.blobs_dir(roots, run_id))
        make_dir(paths.operations_dir(roots, run_id))
        lock = RunLock.acquire(roots, run_id)
        try:
            journal = paths.journal_path(roots, run_id)
            if journal.exists():
                raise JournalWriteError(
                    f"run {run_id} already exists at {journal.parent}: continue it with "
                    "Journal.open_for_resume, or start a new run id"
                )
            write_atomic(
                paths.run_dir(roots, run_id) / paths.CONFIG_NAME,
                json.dumps(config_snapshot, indent=2, sort_keys=True).encode(),
            )
            log = AppendLog.open(journal, create=True)
            return Journal(roots, run_id, _log=log, _lock=lock)
        except BaseException:
            lock.release()
            raise

    @staticmethod
    def config_snapshot(roots: Roots, run_id: str) -> dict[str, Any]:
        """The config snapshot `create` wrote for the run, read back: what a caller that
        continues the run needs to know about how it was started."""
        path = paths.run_dir(roots, run_id) / paths.CONFIG_NAME
        doc: object = json.loads(path.read_bytes())
        if not isinstance(doc, dict):
            raise JournalCorrupt(None, str(path), "the config snapshot is not an object")
        return cast("dict[str, Any]", doc)

    @staticmethod
    def open_for_resume(roots: Roots, run_id: str) -> Journal:
        """Reopen an interrupted run. Its projection is the journal read back, and a
        journal the fold finds inconsistent is refused (`JournalCorrupt`)."""
        from chickadee.journal.reader import read_events

        lock = RunLock.acquire(roots, run_id)
        # Everything past this point releases the lock on the way out. A corrupt journal
        # is exactly when the caller wants to retry, and a lock leaked here would refuse
        # every later attempt in this process.
        log: AppendLog | None = None
        try:
            path = paths.journal_path(roots, run_id)
            # Opening repairs a torn tail, before anything reads or appends.
            log = AppendLog.open(path, create=True)
            if log.repaired_bytes:
                print(
                    f"chickadee: truncated {log.repaired_bytes} torn byte(s) from the tail "
                    f"of {path}",
                    flush=True,
                )
            fold = Fold()
            for event in read_events(path):
                fold.add(event)
            fold.projection().require_consistent()
            return Journal(roots, run_id, _log=log, _fold=fold, _lock=lock)
        except BaseException:
            # Best effort: the error already propagating is the caller's, and a second one
            # raised here would replace it and skip the release.
            if log is not None:
                with contextlib.suppress(OSError):
                    log.close()
            lock.release()
            raise

    def close(self) -> None:
        if self._closed:
            return
        # Marked closed before anything can fail: the lock is released on every path
        # below, so a second close must not try again. A failing `close()` that skipped
        # the release would hold the kernel lock for the life of the process and refuse
        # every later attempt on this run — the same leak the resume path had.
        self._closed = True
        try:
            # Quiet on a journal that already failed: it said so when it failed, and closing
            # it is how its caller gets to resume.
            self._log.close()
        finally:
            if self._lock is not None:
                self._lock.release()

    def __enter__(self) -> Journal:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- append -------------------------------------------------------------

    def append(
        self,
        kind: EventKind,
        payload: Any,
        *,
        task_id: str | None = None,
        round: int | None = None,
    ) -> int:
        """Validate, write one line, and return the assigned seq.

        Validation happens before the write, so a malformed payload is a caller bug that
        raises rather than a corrupt line someone has to explain later .
        """
        from chickadee.schema import JournalEvent

        if self._closed:
            raise JournalWriteError("journal is closed")
        if self._log.stopped is not None:
            raise JournalWriteError(self._log.stopped)

        # The counter advances only once the record is durably on disk. Bumping it first
        # means a rejected payload silently burns a sequence number, and the next good
        # append lands with a gap that makes the whole journal unreadable — the
        # failure mode is total, and it is caused by the event we successfully refused.
        next_seq = self._seq + 1
        event = JournalEvent.model_validate(
            {
                "v": ENVELOPE_VERSION,
                "run_id": self.run_id,
                "seq": next_seq,
                "ts": stamp(),
                "kind": kind.value,
                "payload": payload.model_dump(by_alias=True)
                if hasattr(payload, "model_dump")
                else payload,
                "task_id": task_id,
                "round": round,
            }
        )
        line = (event.model_dump_json(**AS_STORED) + "\n").encode()
        sync = kind in FSYNC_KINDS or self._since_fsync + 1 >= FSYNC_EVERY
        try:
            self._log.append(line, sync=sync)
        except OSError as exc:  # disk full, IO error
            raise self._stop(f"append failed at seq={next_seq}: {exc}") from exc
        self._since_fsync = 0 if sync else self._since_fsync + 1
        self._seq = next_seq
        # Folded after the write, so a reader never sees a line that is not durable, and
        # folded as the line reads back: the line drops `None` fields, so the event in hand
        # and the line on disk need not validate alike, and the projection is the file's.
        self._fold.add(JournalEvent.model_validate_json(line))
        return self._seq

    def _stop(self, what: str) -> JournalWriteError:
        """The journal's account of a failed append, which the log reports on every later
        one. After a failed write or fsync what is durable is uncertain, so only a resume,
        which repairs the tail and replays what is on disk, writes again; `next_seq` is left
        unclaimed, so the resume legitimately reuses it."""
        if self._log.cut:
            after = f"; the journal is cut back to seq {self._seq}"
        else:
            after = "; the cut failed too: the journal may hold that line, whole or torn"
        why = f"{what}{after}. Resume the run to write again"
        self._log.stop(why)
        return JournalWriteError(why)

    # -- blobs --------------------------------------------------------------

    def put_blob(self, content: bytes) -> str:
        """Content-addressed, write-once: the run's `durable.BlobStore`."""
        try:
            return self._blobs.put(content)
        except BlobWriteFailed as exc:
            raise JournalWriteError(str(exc)) from exc

    def get_blob(self, sha: str) -> bytes:
        try:
            return self._blobs.read(sha)
        except FileNotFoundError:
            # A miss says which kind — pruned by a logged gc, or absent. The lookup
            # is on the miss; nothing is stored beside a blob that may not exist.
            from chickadee.journal.gc import blob_missing

            raise blob_missing(paths.runs_root(self.roots), self.run_id, sha) from None

    def events(self) -> Any:
        from chickadee.journal.reader import read_events

        return read_events(self._path)

    def projection(self) -> RunProjection:
        """The run as of the last durable append: handed out immutable."""
        return self._fold.projection()


# -- LOCK -------------------------------------------------------------


class RunLock:
    """Exclusion for one run's journal: a `durable.FileLock` on the run's LOCK file,
    with its errors in the journal's terms. A busy run is `ResumeRefused`, naming the holder
    the file records; a lock that cannot be evaluated is `JournalWriteError`, because naming
    an owner that does not exist would send the person hunting a process."""

    def __init__(self, lock: FileLock) -> None:
        self._lock = lock
        self.path = lock.path

    @staticmethod
    def acquire(roots: Roots, run_id: str) -> RunLock:
        path = paths.lock_path(roots, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return RunLock(FileLock.acquire(path))
        except LockHeld as held:
            raise ResumeRefused(held.pid, held.host) from None
        except LockUnavailable as exc:
            raise JournalWriteError(str(exc)) from exc

    def release(self) -> None:
        self._lock.release()
