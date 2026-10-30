"""Durable files: the one home for how state reaches disk and survives a crash.

A leaf package: it imports nothing from `chickadee`, and raises only `OSError` and its own
`OSError` subclasses, which each caller wraps in its own typed error. Calling it grants no
permission to write: invariant 1's check counts a call into it as a write by the caller.

"""

from chickadee.durable.blobs import BlobPut, BlobStore, BlobWriteFailed
from chickadee.durable.files import (
    NEW_FILE_MODE,
    Fault,
    FaultHook,
    make_dir,
    rename_durably,
    sync_dir,
    unlink_durably,
    write_all,
    write_atomic,
    write_new,
)
from chickadee.durable.lock import (
    FileLock,
    LockHeld,
    LockUnavailable,
    lock_is_held,
    read_owner,
)
from chickadee.durable.log import (
    AppendLog,
    LogStopped,
    complete_lines,
    complete_lines_from,
    repair_tail,
)

__all__ = [
    "NEW_FILE_MODE",
    "AppendLog",
    "BlobPut",
    "BlobStore",
    "BlobWriteFailed",
    "Fault",
    "FaultHook",
    "FileLock",
    "LockHeld",
    "LockUnavailable",
    "LogStopped",
    "complete_lines",
    "complete_lines_from",
    "lock_is_held",
    "make_dir",
    "read_owner",
    "rename_durably",
    "repair_tail",
    "sync_dir",
    "unlink_durably",
    "write_all",
    "write_atomic",
    "write_new",
]
