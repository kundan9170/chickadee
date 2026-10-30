"""Elided regions are marked as omissions, never reconstructed (invariant 16; architecture
§7.4.1-§7.4.2).

Whatever a result leaves out, it says so where the hole is: a gap between spans is one
marker naming its lines, a cut row counts what it lost, an outline too long for its budget
counts the entries it did not show. And nothing fills a hole with text the file does not
hold now: a gap is bridged only with the file's own lines, and a view is never stitched
across an edit.
"""

from __future__ import annotations

from pathlib import Path

from context_helpers import (
    Caller,
    Diag,
    DiagSet,
    FakeDiagnostics,
    FakeIndex,
    at_length,
    cut,
    exchange,
    marked,
    numbered,
    python_module,
    reader_for,
    shown,
    write_files,
)

from chickadee.context.diagnostics import DIAG_MAX_RENDERED
from chickadee.context.fit import MERGE_GAP
from chickadee.context.handle import content_sha
from chickadee.context.results import ToolResult
from chickadee.context.session import Session
from chickadee.context.spans import Span
from chickadee.context.terms import Caps
from chickadee.edits.lines import elided, number_lines


def test_a_gap_between_spans_is_one_marker_naming_exactly_its_lines() -> None:
    lines = [f"l{i}" for i in range(1, 11)]
    rows = number_lines(lines, (Span(1, 2), Span(4, 4), Span(8, 10)))
    assert rows == [
        " 1│l1",
        " 2│l2",
        elided(3, 3),
        " 4│l4",
        elided(5, 7),
        " 8│l8",
        " 9│l9",
        "10│l10",
    ]
    assert elided(3, 3) == "[elided: line 3]"
    assert elided(5, 7) == "[elided: lines 5-7]"


def test_a_cut_marks_the_conversation_turns_it_leaves_out(session: Session, ws: Path) -> None:
    """Invariant 16: turns cut with no residue to stand for them are named as elided, never
    left to vanish from the block without a trace."""
    session.record_user(at_length("keep staging untouched"))
    exchange(session, "c1", "shell", {"command": "ls"}, "a\n", finding="one file")
    session.record_assistant("understood")
    session.record_user("next")
    session.record_assistant("ok")
    done = cut(session, ws)
    assert done.cut is not None
    assert "[1 user turn (seq 1) and 2 assistant turns" in done.cut.block
    assert "elided: not shown here, kept verbatim in the session log]" in done.cut.block
    assert done.cut.facts["turns_elided"] == [1, 4, 5]


def test_a_preview_marks_the_hole_between_its_head_and_tail(session: Session, ws: Path) -> None:
    write_files(ws, {"notes.txt": numbered(100)})
    result = reader_for(session, ws).read("notes.txt")
    assert isinstance(result, ToolResult)
    text = result.render()
    assert shown(text) | marked(text) == set(range(1, 101))
    assert not shown(text) & marked(text)
    assert "line 50" not in text


def test_signatures_elide_every_body_and_account_for_every_line(
    session: Session, ws: Path
) -> None:
    text, syms = python_module(5, body=6)
    write_files(ws, {"mod.py": text})
    reader = reader_for(session, ws, index=FakeIndex({"mod.py": syms}, ws))
    result = reader.read("mod.py", depth="signatures")
    assert isinstance(result, ToolResult)
    assert result.content is not None
    total = len(text.splitlines())
    assert shown(result.content) | marked(result.content) == set(range(1, total + 1))
    assert not shown(result.content) & marked(result.content)
    for sym in syms:
        assert sym.line_start in shown(result.content)
    assert "y5 = x" not in result.content
    assert "return 3" not in result.content


def test_a_merged_view_bridges_a_small_gap_with_the_files_own_lines(
    session: Session, ws: Path
) -> None:
    """A gap of at most `MERGE_GAP` lines is cheaper shown than marked, and what shows is
    read from the file, never inferred from its neighbours."""
    write_files(ws, {"a.txt": numbered(100)})
    call = Caller(session, reader_for(session, ws))
    session.record_user("read it")
    call("read", path="a.txt", lines="1-10")
    line = call("read", path="a.txt", lines=f"{11 + MERGE_GAP}-30")
    assert line.supersedes == ("c1",)
    assert shown(line.text) == set(range(1, 31))
    assert not marked(line.text)
    assert "13│line 13" in line.text


def test_a_merged_view_never_bridges_a_gap_the_cap_cannot_pay_for(
    session: Session, ws: Path
) -> None:
    """A bridged line is sent whole and unmarked. One 50,000-byte line in a small gap is not
    cheaper shown than marked, so the gap stays a gap and the read stays within the cap."""
    body = "".join("Z" * 50_000 + "\n" if i == 11 else f"line {i}\n" for i in range(1, 30))
    write_files(ws, {"a.txt": body})
    call = Caller(session, reader_for(session, ws, caps=Caps(inline_bytes=4000)))
    session.record_user("read it")
    call("read", path="a.txt", lines="1-10")
    line = call("read", path="a.txt", lines="12-15")
    assert line.ran
    assert "Z" * 100 not in line.text
    assert 11 in marked(line.text)
    assert len(line.text.encode("utf-8")) < 4000


def test_a_merged_view_is_capped_on_what_it_newly_sends(session: Session, ws: Path) -> None:
    """The earlier view's lines are re-sent in place of the result that carried them, so
    they are not charged again: a long line asked for next to a view that filled the cap is
    shown cut, never refused on a `max_bytes` the call did not pass."""
    body = "".join(
        "Y" * 5000 + "\n" if i == 120 else f"line {i:04d} padding\n" for i in range(1, 400)
    )
    write_files(ws, {"a.txt": body})
    call = Caller(session, reader_for(session, ws, caps=Caps(inline_bytes=2000)))
    session.record_user("read it")
    call("read", path="a.txt", lines="1-100")
    line = call("read", path="a.txt", lines="120-120")
    assert line.ran, line.text
    assert "max_bytes" not in line.text
    assert "…[+" in line.text, "line 120 is shown cut"


def test_a_merged_view_marks_a_large_gap_rather_than_filling_it(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"a.txt": numbered(100)})
    call = Caller(session, reader_for(session, ws))
    session.record_user("read it")
    call("read", path="a.txt", lines="1-10")
    line = call("read", path="a.txt", lines="60-70")
    assert line.supersedes == ("c1",)
    assert shown(line.text) == set(range(1, 11)) | set(range(60, 71))
    assert marked(line.text) == set(range(11, 60))
    assert "line 30" not in line.text


def test_a_view_is_never_stitched_across_an_edit(session: Session, ws: Path) -> None:
    """A view merged from before and after an edit would be a file that never existed:
    only the same bytes merge, and the new lines come under the new content's handle."""
    write_files(ws, {"a.txt": numbered(40)})
    call = Caller(session, reader_for(session, ws))
    session.record_user("read it")
    before = call("read", path="a.txt", lines="1-10")
    edited = numbered(40).replace("line 12\n", "line 12 edited\n")
    write_files(ws, {"a.txt": edited})
    after = call("read", path="a.txt", lines="11-20")
    assert after.supersedes == ()
    assert before.handle is not None
    assert after.handle is not None
    assert after.handle.id != before.handle.id
    assert after.handle.content_sha == content_sha(edited.encode())
    assert shown(after.text) == set(range(11, 21))
    assert "12│line 12 edited" in after.text


def test_a_csv_preview_counts_the_rows_it_does_not_show(session: Session, ws: Path) -> None:
    write_files(ws, {"t.csv": "id,name\n" + "".join(f"{i},n{i}\n" for i in range(1, 101))})
    result = reader_for(session, ws).read("t.csv")
    assert isinstance(result, ToolResult)
    text = result.render()
    assert shown(text) == set(range(1, 7))
    assert marked(text) == set(range(7, 102))
    assert "100 data rows by line count" in text


def test_an_outline_past_its_budget_counts_the_entries_it_cut(
    session: Session, ws: Path
) -> None:
    text, syms = python_module(2_000, body=1)
    write_files(ws, {"big.py": text})
    result = reader_for(session, ws, index=FakeIndex({"big.py": syms}, ws)).read("big.py")
    assert isinstance(result, ToolResult)
    tail = [row for row in result.render().splitlines() if "outline continues" in row]
    assert len(tail) == 1
    assert tail[0].endswith("more symbols not shown]")


def test_a_long_preview_row_is_cut_with_its_remainder_counted(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"wide.txt": "short\n" + "w" * 1_000 + "\nshort again\n"})
    result = reader_for(session, ws).read("wide.txt")
    assert isinstance(result, ToolResult)
    assert "…[+760 chars]" in result.render()
    assert result.view == (Span(1, 1), Span(3, 3)), "a cut row is not a view of its line"


def test_diagnostics_past_the_render_cap_are_counted_never_dropped(
    session: Session, ws: Path
) -> None:
    body = numbered(40)
    write_files(ws, {"mod.txt": body})
    rows = tuple(Diag(line=i, message=f"problem {i}") for i in range(1, 9))
    index = FakeDiagnostics(
        {"mod.txt": DiagSet(rows, coverage="total", analyzed_sha=content_sha(body.encode()))}
    )
    result = reader_for(session, ws, diagnostics=index).read("mod.txt")
    assert isinstance(result, ToolResult)
    listed = [row for row in result.diagnostics if "problem" in row]
    assert len(listed) == DIAG_MAX_RENDERED
    assert f"  [{len(rows) - DIAG_MAX_RENDERED} more not listed]" in result.diagnostics
