"""A record's payload as its kind's type: the one way a client reads
what a `RecordFrame` carries.

A frame's `payload` crosses as JSON. Its type depends on what the record is: a run's record is
a journal event, whose payload per kind is core's frozen journal golden (`RUN_RECORDS`); a
session's record is a session-log line, whose payload per kind is the wire golden's
(`SESSION_RECORDS`). The two are told apart by the frame's scope, never by its kind, since
some kinds (`question_asked`) are both: a run's own record is numbered in that run.
"""

from __future__ import annotations

from chickadee_serve.frames import RecordFrame
from chickadee_wire._base import WireModel
from chickadee_wire.models import RUN_RECORDS, SESSION_RECORDS

__all__ = ["is_run_record", "payload"]


def is_run_record(frame: RecordFrame) -> bool:
    """A run's journal event, numbered in that run, rather than a session's line."""
    return frame.run_id is not None and frame.scope == frame.run_id


def payload(frame: RecordFrame) -> WireModel | None:
    """`frame.payload` as its kind's type; None for a session kind the golden marks opaque,
    which only its writer reads. A payload that is not its kind's type raises
    `ValidationError`: the server and this client disagree about the contract."""
    table = RUN_RECORDS if is_run_record(frame) else SESSION_RECORDS
    adapter = table.get(frame.kind)
    return None if adapter is None else adapter.validate_python(frame.payload)
