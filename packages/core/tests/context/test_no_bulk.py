"""No tool returns a whole file implicitly, at any size.

`read(path)` returns the skeleton and a handle, never the file. Lines arrive only for a
region the caller named, bounded once by the tool that produced them, with a `Truncation`
value saying where to continue. the one bounded exception, a file whose preview already
shows every line, is said aloud rather than left to be noticed.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from context_helpers import (
    Caller,
    FakeIndex,
    Sym,
    numbered,
    png,
    python_module,
    reader_for,
    shown,
    write_files,
)

from chickadee.context.parts import PREVIEW_HEAD, PREVIEW_TAIL
from chickadee.context.reader import Reader
from chickadee.context.results import InvalidArgument, NotText, ToolResult
from chickadee.context.session import Session
from chickadee.context.spans import Span, line_count
from chickadee.context.terms import INLINE_BYTES, MAX_LINES, Caps

PREVIEW_LINES = PREVIEW_HEAD + PREVIEW_TAIL


def shown_lines(result: ToolResult) -> set[int]:
    return shown(result.render())


def whole_note(n: int) -> str:
    return f"[the whole file: all {n} lines fit the preview]"


def read_ok(
    session: Session,
    ws: Path,
    target: str,
    *,
    lines: str | None = None,
    depth: str | None = None,
) -> ToolResult:
    result = reader_for(session, ws).read(target, lines=lines, depth=depth)
    assert isinstance(result, ToolResult), result
    return result


@pytest.mark.parametrize("n", [PREVIEW_LINES + 1, 100, 5_000])
def test_a_bare_read_of_a_file_past_the_preview_size_never_shows_every_line(
    session: Session, ws: Path, n: int
) -> None:
    write_files(ws, {"notes.txt": numbered(n)})
    result = read_ok(session, ws, "notes.txt")
    assert result.content is None
    assert len(shown_lines(result)) <= PREVIEW_LINES < n
    assert re.search(rf"\[elided: lines? {PREVIEW_HEAD + 1}\b", result.render())
    assert whole_note(n) not in result.render()


@pytest.mark.parametrize("n", [1, 12, PREVIEW_LINES])
def test_a_file_whose_preview_is_the_whole_file_says_so(
    session: Session, ws: Path, n: int
) -> None:
    """The first bounded exception: below the preview's size the skeleton would cost what
    the file costs. Allowed, and never silent."""
    write_files(ws, {"notes.txt": numbered(n)})
    result = read_ok(session, ws, "notes.txt")
    assert shown_lines(result) == set(range(1, n + 1))
    assert whole_note(n) in result.notes


def test_the_whole_file_exception_is_bounded_by_the_byte_cap_as_well(
    session: Session, ws: Path
) -> None:
    """Thirty lines is under the preview's line count, but thirty wide lines are not under
    its byte cap: the preview shrinks and makes no whole-file claim."""
    write_files(ws, {"wide.txt": "".join(f"{i:03} {'w' * 196}\n" for i in range(1, 31))})
    result = read_ok(session, ws, "wide.txt")
    assert len(shown_lines(result)) < 30
    assert "[elided: lines" in result.render()
    assert whole_note(30) not in result.render()


def test_an_empty_file_previews_as_empty_and_claims_nothing(session: Session, ws: Path) -> None:
    write_files(ws, {"empty.txt": ""})
    result = read_ok(session, ws, "empty.txt")
    assert "(empty file)" in result.render()
    assert result.view == ()
    assert "whole file" not in result.render()


def test_indexed_code_previews_its_symbols_and_none_of_its_bodies(
    session: Session, ws: Path
) -> None:
    text, syms = python_module(60)
    write_files(ws, {"mod.py": text})
    result = reader_for(session, ws, index=FakeIndex({"mod.py": syms}, ws)).read("mod.py")
    assert isinstance(result, ToolResult)
    assert result.content is None
    assert result.view == ()
    assert "def f1(x)" in result.render()
    assert "def f60(x)" in result.render()
    assert "y1 = x" not in result.render()
    assert not shown_lines(result)


def test_unindexed_code_previews_head_and_tail_and_says_it_has_no_outline(
    session: Session, ws: Path
) -> None:
    text, _ = python_module(60)
    write_files(ws, {"mod.py": text})
    result = read_ok(session, ws, "mod.py")
    assert "no index for this file" in result.render()
    assert len(shown_lines(result)) <= PREVIEW_LINES < len(text.splitlines())


@pytest.mark.parametrize(
    ("name", "body", "hidden"),
    [
        pytest.param(
            "guide.md",
            "".join(f"# Part {i}\n\nbody text {i} is here\n\n" for i in range(1, 30)),
            "body text",
            id="markdown-headings-only",
        ),
        pytest.param(
            "settings.json",
            json.dumps({"db": {"host": "value-host", "pool": [{"size": 5}]}, "debug": True}),
            "value-host",
            id="json-keys-and-types-only",
        ),
        pytest.param(
            "settings.toml",
            '[db]\nhost = "value-host"\nport = 5432\n',
            "value-host",
            id="toml-keys-and-types-only",
        ),
        pytest.param(
            "table.csv",
            "id,name\n" + "".join(f"{i},row-value-{i}\n" for i in range(1, 500)),
            "row-value-250",
            id="csv-header-and-first-rows",
        ),
        pytest.param(
            "bundle.min.js",
            "var a=" + "1+" * 4_000 + "1;\n",
            "1+1+1",
            id="minified-shape-only",
        ),
        pytest.param(
            "gen.py",
            "# @generated by a tool, do not edit\n" + numbered(400, "gen_value"),
            "gen_value 200",
            id="generated-shape-only",
        ),
    ],
)
def test_every_format_previews_its_skeleton_never_its_contents(
    session: Session, ws: Path, name: str, body: str, hidden: str
) -> None:
    write_files(ws, {name: body})
    result = read_ok(session, ws, name)
    assert result.content is None
    assert hidden not in result.render()


def test_a_binary_file_returns_its_shape_and_refuses_a_region(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"blob.bin": b"\x00\x01payload\x00" * 100})
    reader = reader_for(session, ws)
    bare = reader.read("blob.bin")
    assert isinstance(bare, ToolResult)
    assert "payload" not in bare.render()
    assert isinstance(reader.read("blob.bin", lines="1-2"), NotText)
    assert isinstance(reader.read("blob.bin", depth="full"), NotText)


def test_a_bare_read_of_an_image_sends_no_pixels(session: Session, ws: Path) -> None:
    write_files(ws, {"shot.png": png(64, 48)})
    result = read_ok(session, ws, "shot.png")
    assert result.image is None
    assert "tokens to view" in result.render()


def test_depth_full_with_no_region_is_bounded_by_the_caps(session: Session, ws: Path) -> None:
    """An explicit depth is not a licence past the caps: the tool bounds what it sends, and
    the `Truncation` value names where to continue."""
    write_files(ws, {"big.txt": numbered(5_000)})
    result = read_ok(session, ws, "big.txt", depth="full")
    assert result.content is not None
    assert len(result.content.encode()) <= INLINE_BYTES
    assert len(shown_lines(result)) <= MAX_LINES
    cut = result.truncation
    assert cut is not None
    assert cut.total == 5_000
    assert cut.reason in ("byte_cap", "line_cap")
    resume = cut.next_span()
    assert resume is not None
    assert resume.start == cut.kept[-1].end + 1


@pytest.mark.parametrize("width", [8, 80, 400])
def test_a_named_region_past_the_caps_is_cut_with_a_truncation_value(
    session: Session, ws: Path, width: int
) -> None:
    write_files(ws, {"big.txt": "".join(f"{i} {'z' * width}\n" for i in range(1, 3_001))})
    result = read_ok(session, ws, "big.txt", lines="1-3000")
    assert result.content is not None
    assert len(result.content.encode()) <= INLINE_BYTES
    assert len(shown_lines(result)) <= MAX_LINES
    cut = result.truncation
    assert cut is not None
    assert cut.dropped == 3_000 - len(shown_lines(result))
    assert 'next: read {"path": "big.txt", "lines": ' in result.render()


def test_max_bytes_lowers_the_cap_and_never_raises_it(session: Session, ws: Path) -> None:
    write_files(ws, {"big.txt": numbered(3_000, "x" * 60)})
    reader = reader_for(session, ws)
    raised = reader.read("big.txt", lines="1-3000", max_bytes=10 * INLINE_BYTES)
    lowered = reader.read("big.txt", lines="1-3000", max_bytes=512)
    assert isinstance(raised, ToolResult)
    assert isinstance(lowered, ToolResult)
    assert raised.content is not None
    assert lowered.content is not None
    assert len(raised.content.encode()) <= INLINE_BYTES
    assert len(lowered.content.encode()) <= 512
    assert isinstance(reader.read("big.txt", lines="1-3", max_bytes=0), InvalidArgument)


def test_grep_in_matching_every_line_is_bounded_by_the_result_cap(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"big.txt": numbered(5_000)})
    reader = reader_for(session, ws)
    opened = reader.read("big.txt")
    assert isinstance(opened, ToolResult)
    found = reader.grep_in(opened.handle.id, "line")
    assert isinstance(found, ToolResult)
    assert len(shown_lines(found)) <= MAX_LINES
    cut = found.truncation
    assert cut is not None
    assert cut.reason == "result_cap"
    assert cut.total == 5_000


def test_grep_in_is_bounded_by_the_byte_cap_as_well(session: Session, ws: Path) -> None:
    """Long matching lines stop a search at the inline byte cap, as they stop a read."""
    write_files(ws, {"app.log": "".join(f"ERROR {i} " + "x" * 480 + "\n" for i in range(1000))})
    reader = reader_for(session, ws)
    opened = reader.read("app.log")
    assert isinstance(opened, ToolResult)
    found = reader.grep_in(opened.handle.id, "ERROR", context=0)
    assert isinstance(found, ToolResult)
    assert found.content is not None
    assert len(found.content.encode("utf-8")) <= INLINE_BYTES
    assert found.truncation is not None
    assert found.truncation.reason == "result_cap"


def test_a_csv_preview_over_budget_keeps_the_rows_that_fit(session: Session, ws: Path) -> None:
    """The fallback shrinks the table's preview rows, not to its header alone."""
    rows = "".join(f"{i}," + "v" * 150 + "\n" for i in range(1, 50))
    write_files(ws, {"wide.csv": "id,value\n" + rows})
    found = reader_for(session, ws).read("wide.csv", max_bytes=600)
    assert isinstance(found, ToolResult)
    assert "1," + "v" * 150 in found.handle.preview


@pytest.mark.parametrize("cap", [40, 120])
def test_a_lowered_cap_bounds_a_line_over_it_and_says_the_line_was_cut(
    session: Session, ws: Path, cap: int
) -> None:
    """`max_bytes` holds even for a single line over it, and the result reports the line
    as not shown whole."""
    write_files(ws, {"wide.txt": "x" * 1000 + "\n"})
    found = reader_for(session, ws).read("wide.txt", lines="1-1", max_bytes=cap)
    assert isinstance(found, ToolResult)
    assert found.content is not None
    assert len(found.content.encode("utf-8")) <= cap
    assert found.truncation is not None
    assert found.truncation.reason == "byte_cap"
    assert found.view == ()


@dataclass(frozen=True)
class FakeChanges:
    """`ChangedLines` over a fixed table, for `at="@changed"`."""

    spans: dict[str, tuple[Span, ...]]

    def changed_lines(self, path: str) -> Sequence[Span] | None:
        return self.spans.get(path)


def test_a_cut_read_of_split_spans_names_only_the_lines_it_sent(
    session: Session, ws: Path
) -> None:
    """Invariant 16: two far-apart spans cut by the cap are reported as the two spans, never
    as one range that swallows the hundreds of lines between them."""
    write_files(ws, {"wide.txt": numbered(900, "x" * 200)})
    changes = FakeChanges({"wide.txt": (Span(1, 40), Span(880, 900))})
    found = reader_for(session, ws, changes=changes).read("wide.txt", at="@changed")
    assert isinstance(found, ToolResult)
    cut = found.truncation
    assert cut is not None
    assert cut.reason in ("byte_cap", "line_cap")
    sent: set[int] = set()
    for span in cut.kept:
        sent |= set(range(span.start, span.end + 1))
    assert sent == shown_lines(found), "kept names exactly the lines sent"
    assert 500 not in sent, "the gap between the spans was never sent"
    assert line_count(cut.kept) + cut.dropped == 61


def test_a_cut_signatures_read_counts_signature_lines_not_region_lines(
    session: Session, ws: Path
) -> None:
    """The bodies between signatures were never going to be shown, so they are not lines the
    request asked for and did not get."""
    text, syms = python_module(30, body=8)
    write_files(ws, {"mod.py": text})
    total = len(text.splitlines())
    reader = reader_for(
        session, ws, index=FakeIndex({"mod.py": syms}, ws), caps=Caps(max_lines=8)
    )
    found = reader.read("mod.py", lines=f"1-{total}", depth="signatures")
    assert isinstance(found, ToolResult)
    cut = found.truncation
    assert cut is not None
    assert line_count(cut.kept) + cut.dropped == 2 * len(syms), "one signature, one body line"
    assert cut.dropped < total - line_count(cut.kept)


@pytest.mark.parametrize("depth", [None, "signatures"])
def test_a_cap_with_no_room_for_the_line_is_refused_not_answered_empty(
    session: Session, ws: Path, depth: str | None
) -> None:
    """An empty answer carrying a continuation would move past a line never shown, so a
    cap too small for any of the line asked for is refused by name."""
    reader = wide_function(session, ws)
    found = reader.read("wide.py", lines="1-1", depth=depth, max_bytes=1)
    assert isinstance(found, InvalidArgument)
    assert found.field == "max_bytes"
    assert "no room for any of line 1" in found.why


def wide_function(session: Session, ws: Path) -> Reader:
    """An indexed module whose first function's signature line is 1,000 bytes wide, among
    enough ordinary lines that it reads as code rather than as minified."""
    text, syms = python_module(10)
    wide = "def f0(x='" + "x" * 1000 + "'):\n    return x\n\n"
    sym = Sym("f0", "wide.f0", "function", 1, 2, "def f0(x)")
    shifted = tuple(
        replace(s, line_start=s.line_start + 3, line_end=s.line_end + 3) for s in syms
    )
    write_files(ws, {"wide.py": wide + text})
    return reader_for(session, ws, index=FakeIndex({"wide.py": (sym, *shifted)}, ws))


@pytest.mark.parametrize("cap", [40, 120])
def test_signatures_hold_a_lowered_cap_as_rendered(
    session: Session, ws: Path, cap: int
) -> None:
    found = wide_function(session, ws).read("wide.py", depth="signatures", max_bytes=cap)
    assert isinstance(found, ToolResult)
    assert found.content is not None
    assert len(found.content.encode("utf-8")) <= cap


def test_grep_in_counts_the_cut_markers_it_renders(session: Session, ws: Path) -> None:
    """Long matching lines are cut with a `…[+N bytes]` marker; the marker is part of what
    the byte cap bounds."""
    write_files(ws, {"wide.log": "".join(f"hit {i} " + "y" * 1000 + "\n" for i in range(8))})
    reader = reader_for(session, ws)
    opened = reader.read("wide.log")
    assert isinstance(opened, ToolResult)
    found = reader.grep_in(opened.handle.id, "hit", context=0)
    assert isinstance(found, ToolResult)
    assert found.content is not None
    assert len(found.content.encode("utf-8")) <= INLINE_BYTES


def test_one_line_over_the_cap_alone_is_shown_cut_and_marked(
    session: Session, ws: Path
) -> None:
    """A read always progresses, and the cut is visible: the line's size is named and the
    shown part ends in a count of what was left out."""
    write_files(ws, {"long.txt": "a" * 50_000 + "\n"})
    result = read_ok(session, ws, "long.txt", lines="1-1")
    assert result.content is not None
    assert len(result.content.encode()) <= INLINE_BYTES
    assert "bytes]" in result.content
    assert any("over the inline cap alone" in note for note in result.notes)
    assert result.view == ()


def test_stored_output_is_previewed_and_read_back_by_region_only(
    session: Session, ws: Path
) -> None:
    reader = reader_for(session, ws)
    handle = reader.store("output", numbered(1_000, "out").encode())
    assert handle.stored
    assert "out 500" not in handle.preview
    bare = reader.read(handle)
    assert isinstance(bare, ToolResult)
    assert bare.content is None
    region = reader.read(handle, lines="500-501")
    assert isinstance(region, ToolResult)
    assert region.content is not None
    assert "out 500" in region.content


# -- the caps as a property: any sequence of reads and searches -------------------------

ROW = re.compile(r"^\s*(\d+)│(.*)$")
KEPT = re.compile(r"\[showing lines (\d+(?:-\d+)?(?:, \d+(?:-\d+)?)*|none)\b")


def _numbered(text: str) -> list[tuple[int, str, str]]:
    """Every numbered row of a result: its line, its body, and the row as sent."""
    rows: list[tuple[int, str, str]] = []
    for row in text.split("\n"):
        m = ROW.match(row)
        if m:
            rows.append((int(m.group(1)), m.group(2), row))
    return rows


def _true_to(body: str, real: str) -> bool:
    """A row is the file's line, whole or cut and marked; never anything else."""
    if body == real:
        return True
    head, marker, _ = body.partition("…[+")
    return bool(marker) and real.startswith(head)


def _sent(rows: Sequence[str]) -> int:
    """Rows are joined, not terminated."""
    return sum(len(r.encode("utf-8")) + 1 for r in rows) - (1 if rows else 0)


def _lines_in(text: str) -> set[int]:
    out: set[int] = set()
    if text != "none":
        for part in text.split(", "):
            a, _, b = part.partition("-")
            out |= set(range(int(a), int(b or a) + 1))
    return out


def _random_file(rng: random.Random) -> list[str]:
    """Short lines, some empty, and now and then one far past any cap."""
    out: list[str] = []
    for i in range(1, rng.randint(1, 300) + 1):
        width = rng.randint(500, 8000) if rng.random() < 0.05 else rng.randint(0, 60)
        out.append(f"{i}:" + "abcdefgh"[i % 8] * width)
    return out


@pytest.mark.parametrize("seed", range(200))
def test_no_sequence_of_reads_breaks_a_cap_or_claims_a_line_it_did_not_send(
    seed: int, session: Session, ws: Path
) -> None:
    """D78 and invariant 16 as properties rather than cases. For a random file and caps and a
    random run of reads — named spans and bare previews, lowered `max_bytes`, merges into
    the view before, a user turn or an edit between them — and searches:

    - what each result newly sends is within its cap: `max_bytes` when passed, else the
      inline cap. The earlier view a merge re-sends replaces the result that carried it;
    - every numbered row is the file's own line, whole or cut and marked;
    - every line its view or its truncation claims was sent whole;
    - a refusal on `max_bytes` only ever answers a call that passed one.

    Most of the review findings on this read path broke one of these for some input a
    hand-written case had not tried."""
    rng = random.Random(seed)
    lines = _random_file(rng)
    write_files(ws, {"f.txt": "\n".join(lines) + "\n"})
    caps = Caps(inline_bytes=rng.randint(200, 4000), max_lines=rng.randint(5, 200))
    call = Caller(session, reader_for(session, ws, caps=caps))
    session.record_user("look")
    views: dict[str, tuple[Span, ...]] = {}
    handle: str | None = None
    for step in range(rng.randint(3, 8)):
        if rng.random() < 0.15:
            session.record_user("and now")
        if rng.random() < 0.1:
            lines = _random_file(rng)
            write_files(ws, {"f.txt": "\n".join(lines) + "\n"})
        at = f"seed {seed} step {step}"
        if handle is not None and rng.random() < 0.2:
            pattern = rng.choice(["a", "b+", "^1", "e{3}"])
            found = call("grep_in", handle=handle, pattern=pattern, context=rng.randint(0, 4))
            if found.ran:
                rows = _numbered(found.text)
                assert _sent([r for _, _, r in rows]) <= caps.inline_bytes, at
                for n, body, _ in rows:
                    assert _true_to(body, lines[n - 1]), f"{at}: row {n}"
            continue
        args: dict[str, Any] = {"path": "f.txt"}
        if rng.random() < 0.8:
            start = rng.randint(1, len(lines))
            args["lines"] = f"{start}-{min(len(lines), start + rng.randint(0, 80))}"
        cap = caps.inline_bytes
        if rng.random() < 0.4:
            max_bytes = rng.choice([1, 20, rng.randint(1, caps.inline_bytes)])
            args["max_bytes"] = max_bytes
            cap = min(cap, max_bytes)
        line = call("read", **args)
        at = f"{at} read {args}"
        views[line.call_id] = line.view
        if line.handle is not None:
            handle = line.handle.id
        if not line.ran:
            assert "max_bytes" in args or "max_bytes" not in line.text, at
            continue
        prior: set[int] = set()
        for absorbed in line.supersedes:
            for span in views.get(absorbed, ()):
                prior |= set(range(span.start, span.end + 1))
        rows = _numbered(line.text)
        whole = {n for n, body, _ in rows if body == lines[n - 1]}
        for n, body, _ in rows:
            assert _true_to(body, lines[n - 1]), f"{at}: row {n}"
        assert _sent([r for n, _, r in rows if n not in prior]) <= cap, at
        claimed = {n for span in line.view for n in range(span.start, span.end + 1)}
        assert claimed <= whole, f"{at}: view claims {sorted(claimed - whole)[:8]}"
        kept = KEPT.search(line.text)
        if kept is not None:
            assert _lines_in(kept.group(1)) <= whole, f"{at}: truncation keeps unsent lines"
