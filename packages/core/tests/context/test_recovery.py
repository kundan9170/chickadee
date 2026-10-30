"""A STALE handle degrades EXACT to NONE; no receipt ever promises an exact re-read of
content that cannot be returned.

Recovery is a property of this result at this moment, never of the tool. A tool declares a
ceiling, and staleness degrades it per result at the cut, so the only receipts that say
`-> exact` are ones whose re-read returns the bytes the model saw.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from context_helpers import (
    ROW,
    Caller,
    at_length,
    cut,
    exchange,
    numbered,
    png,
    reader_for,
    write_files,
)
from tool_args import checked, zeus_answer

from chickadee.context.handle import Handle
from chickadee.context.recovery import TOOL_REGISTRY, Recovery, actual_recovery, ceiling
from chickadee.context.results import ToolResult
from chickadee.context.session import Session, ToolLine
from chickadee.context.terms import Vision
from chickadee.schema import ToolCall

EXACT_READ = re.compile(r"read\(handle=(h_[0-9a-f]+), lines=(\d+)-(\d+)\) -> exact")
NOT_RECOVERABLE = "not recoverable - finding is the record"


@pytest.mark.parametrize("tool", ["read", "slice", "outline", "grep_in", "journal", "diff"])
def test_handle_backed_tools_declare_an_exact_ceiling(tool: str) -> None:
    assert ceiling(tool) is Recovery.EXACT


@pytest.mark.parametrize(
    "tool", ["shell", "exec", "recall", "locate", "atlas", "impact", "coder", "ask_user"]
)
def test_live_queries_and_undeclared_tools_have_no_exact_ceiling(tool: str) -> None:
    """A ceiling nobody declared is not exact: `coder` and `ask_user` are absent from the
    table on purpose."""
    assert ceiling(tool) is Recovery.NONE


def test_the_ceiling_table_has_no_third_class() -> None:
    assert set(TOOL_REGISTRY.values()) == {Recovery.EXACT, Recovery.NONE}


def test_staleness_degrades_an_exact_ceiling_and_nothing_upgrades_a_none() -> None:
    def stale(handle: str) -> bool:
        return handle == "h_old"

    assert actual_recovery("read", "h_new", is_stale=stale) is Recovery.EXACT
    assert actual_recovery("read", "h_old", is_stale=stale) is Recovery.NONE
    assert actual_recovery("read", None, is_stale=stale) is Recovery.NONE
    assert actual_recovery("shell", "h_new", is_stale=stale) is Recovery.NONE


def record_output(session: Session, ws: Path, call_id: str, command: str, out: str) -> Handle:
    """A `shell` exchange whose output the session's blob store keeps."""
    handle = reader_for(session, ws).store("output", out.encode())
    session.record_assistant(
        "",
        [ToolCall(id=call_id, name="shell", arguments={"command": command})],
    )
    session.record_tool(call_id, ToolResult(handle, header=f"$ {command}"))
    session.record_assistant("ran", findings={call_id: "it printed its output"})
    return handle


def region(session: Session, ws: Path) -> dict[str, ToolLine]:
    """Reads of two files, the second of which will stay fresh, and one command's output;
    then enough turns that all of it is old enough to cut."""
    write_files(ws, {"a.txt": numbered(40, "alpha"), "b.txt": numbered(40, "beta")})
    call = Caller(session, reader_for(session, ws))
    session.record_user(at_length("look around"))
    lines: dict[str, ToolLine] = {}
    for name, cid in (("a.txt", "c1"), ("b.txt", "c2")):
        lines[name] = call("read", path=name, lines="5-9")
        session.record_assistant("noted", findings={cid: f"{name} lines 5-9 look fine"})
    record_output(session, ws, "c3", "make", numbered(30, "built"))
    session.record_user("now the next thing")
    session.record_assistant("on it")
    return lines


def test_a_receipt_for_a_file_that_moved_promises_nothing(session: Session, ws: Path) -> None:
    lines = region(session, ws)
    write_files(ws, {"a.txt": numbered(40, "changed")})
    done = cut(session, ws)
    assert done.cut is not None
    a, b = lines["a.txt"].handle, lines["b.txt"].handle
    assert a is not None
    assert b is not None
    block = done.cut.block
    assert f"read(handle={a.id}" not in block
    assert f"read(handle={b.id}, lines=5-9) -> exact" in block
    assert done.cut.facts["stale"] == [a.id]
    receipt = block[block.index("[evicted · read a.txt") :].split("]", 1)[0]
    assert NOT_RECOVERABLE in receipt


def test_a_receipt_for_a_deleted_image_still_names_the_blob_that_holds_it(
    session: Session, ws: Path
) -> None:
    """The blob holds the original, so the bytes are nameable after the file is gone.
    A receipt that said only "not recoverable" would orphan an image that still reads."""
    write_files(ws, {"shot.png": png(64, 48)})
    reader = reader_for(session, ws, vision=Vision("zeus", supports_images=True))
    session.record_user(at_length("look at the screenshot"))
    session.record_assistant(
        "",
        [ToolCall(id="i1", name="read", arguments={"path": "shot.png", "depth": "full"})],
    )
    line = zeus_answer(
        session, reader, "i1", "read", checked("read", {"path": "shot.png", "depth": "full"})
    )
    assert line.handle is not None
    session.record_assistant("I see it", findings={"i1": "an error dialog"})
    session.record_user("next")
    session.record_assistant("ok")
    (ws / "shot.png").unlink()
    done = cut(session, ws)
    assert done.cut is not None
    assert NOT_RECOVERABLE in done.cut.block
    assert f"the image: read(handle={line.handle.id})" in done.cut.block


def test_every_exact_receipt_rereads_to_the_lines_the_model_saw(
    session: Session, ws: Path
) -> None:
    """The property itself: parse every `-> exact` promise out of the block and keep it."""
    lines = region(session, ws)
    write_files(ws, {"a.txt": numbered(40, "changed")})
    done = cut(session, ws)
    assert done.cut is not None
    seen = {line.handle.id: line.text for line in lines.values() if line.handle is not None}
    promises = EXACT_READ.findall(done.cut.block)
    assert promises
    reader = reader_for(session, ws)
    for hid, start, end in promises:
        again = reader.read(hid, lines=f"{start}-{end}")
        assert isinstance(again, ToolResult)
        assert again.content is not None
        rows = {r for r in seen[hid].splitlines() if ROW.match(r)}
        assert set(again.content.splitlines()) == rows


def test_recorded_output_is_named_and_reads_back_exactly(session: Session, ws: Path) -> None:
    """`shell`'s answer now is not recoverable, but the output it printed then is in a blob:
    the receipt says both, and the handle it names returns those bytes."""
    region(session, ws)
    done = cut(session, ws)
    assert done.cut is not None
    handle = session.state.result("c3")
    assert handle is not None
    assert handle.handle is not None
    stored = handle.handle.id
    assert f"{NOT_RECOVERABLE} · recorded output: read(handle={stored})" in done.cut.block
    again = reader_for(session, ws).read(stored, lines="1-30")
    assert isinstance(again, ToolResult)
    assert again.content is not None
    assert [r.split("│", 1)[1] for r in again.content.splitlines()] == [
        f"built {i}" for i in range(1, 31)
    ]


def test_an_exchange_with_no_handle_is_never_exact(session: Session, ws: Path) -> None:
    session.record_user(at_length("check"))
    exchange(session, "x1", "read", {"path": "gone.txt"}, "INVALID_ARGUMENTS{...}")
    session.record_user("next")
    session.record_assistant("ok")
    done = cut(session, ws)
    assert done.cut is not None
    assert "-> exact" not in done.cut.block
    assert NOT_RECOVERABLE in done.cut.block
