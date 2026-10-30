"""A write to one file does not invalidate handles on another.

Freshness is a per-file content hash — never an mtime, never a generation. A commit to one
file must not invalidate a handle on another, or every episode would invalidate the whole
context and the economy would collapse to re-reading.
"""

from __future__ import annotations

import os
from pathlib import Path

from context_helpers import Caller, reader_for, write_files

from chickadee.context.freshness import CHANGED_NOTE, dereference, is_stale, stale_check, sweep
from chickadee.context.handle import Handle
from chickadee.context.recovery import Recovery, actual_recovery
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import ToolResult
from chickadee.context.session import Session

A = "def a():\n    return 1\n"
B = "def b():\n    return 2\n"


def opened(session: Session, ws: Path, *names: str) -> dict[str, Handle]:
    reader = reader_for(session, ws)
    handles: dict[str, Handle] = {}
    for name in names:
        result = reader.read(name)
        assert isinstance(result, ToolResult)
        handles[name] = result.handle
    return handles


def test_a_write_to_one_file_leaves_a_handle_on_another_fresh(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"a.py": A, "b.py": B})
    h = opened(session, ws, "a.py", "b.py")
    resolver = HandleResolver(ws, session.blob)
    write_files(ws, {"a.py": A.replace("1", "10")})
    assert is_stale(h["a.py"], resolver)
    assert not is_stale(h["b.py"], resolver)
    now = dereference(h["b.py"], resolver)
    assert not now.stale
    assert now.current == B.encode()


def test_freshness_is_the_content_not_the_clock(session: Session, ws: Path) -> None:
    """An mtime that moves with the same bytes is fresh; bytes that change and change
    back are fresh again. Neither holds for a timestamp or a generation counter."""
    write_files(ws, {"a.py": A, "b.py": B})
    h = opened(session, ws, "a.py", "b.py")
    resolver = HandleResolver(ws, session.blob)
    later = (ws / "b.py").stat().st_mtime + 3_600
    os.utime(ws / "b.py", (later, later))
    write_files(ws, {"b.py": B})
    assert not is_stale(h["b.py"], resolver)
    write_files(ws, {"a.py": A.replace("1", "10")})
    assert is_stale(h["a.py"], resolver)
    write_files(ws, {"a.py": A})
    assert not is_stale(h["a.py"], resolver)


def test_a_deleted_file_is_stale_and_its_neighbour_is_not(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": A, "b.py": B})
    h = opened(session, ws, "a.py", "b.py")
    resolver = HandleResolver(ws, session.blob)
    (ws / "a.py").unlink()
    assert is_stale(h["a.py"], resolver)
    assert dereference(h["a.py"], resolver).current is None
    assert not is_stale(h["b.py"], resolver)


def test_stored_output_is_content_addressed_and_never_stale(session: Session, ws: Path) -> None:
    reader = reader_for(session, ws)
    handle = reader.store("output", b"3 passed\n")
    write_files(ws, {"a.py": A})
    assert not is_stale(handle, HandleResolver(ws, session.blob))


def test_the_sweep_notices_only_the_view_whose_file_moved(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": A, "b.py": B})
    call = Caller(session, reader_for(session, ws))
    session.record_user("look at both")
    a = call("read", path="a.py", lines="1-2")
    b = call("read", path="b.py", lines="1-2")
    assert a.handle is not None
    assert b.handle is not None
    write_files(ws, {"a.py": A.replace("1", "10")})
    seqs = sweep(session, HandleResolver(ws, session.blob))
    assert len(seqs) == 1
    turn = session.state.turn(seqs[0])
    assert turn is not None
    assert turn.notice is not None
    assert turn.notice.handle == a.handle.id
    assert session.state.noticed_stale == {a.handle.id}


def test_recovery_degrades_only_for_the_file_that_moved(session: Session, ws: Path) -> None:
    write_files(ws, {"a.py": A, "b.py": B})
    h = opened(session, ws, "a.py", "b.py")
    check = stale_check(session.state.handles.get, HandleResolver(ws, session.blob))
    write_files(ws, {"a.py": A.replace("1", "10")})
    assert actual_recovery("read", h["a.py"].id, is_stale=check) is Recovery.NONE
    assert actual_recovery("read", h["b.py"].id, is_stale=check) is Recovery.EXACT


def test_reading_through_a_handle_marks_only_the_changed_file(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"a.py": A, "b.py": B})
    h = opened(session, ws, "a.py", "b.py")
    write_files(ws, {"a.py": A.replace("1", "10")})
    reader = reader_for(session, ws)
    a = reader.read(h["a.py"], lines="1-2")
    b = reader.read(h["b.py"], lines="1-2")
    assert isinstance(a, ToolResult)
    assert isinstance(b, ToolResult)
    assert any(CHANGED_NOTE in note for note in a.notes)
    assert a.handle.id != h["a.py"].id
    assert "return 10" in (a.content or "")
    assert not any(CHANGED_NOTE in note for note in b.notes)
    assert b.handle == h["b.py"]
