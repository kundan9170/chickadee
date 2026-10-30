"""A partial search returns UNRESOLVED with its uncovered set, never an empty list
(invariant 20).

An empty list is a claim about everything, and a search that could not read everything is
not entitled to make one. So recall's empty answer is `UNRESOLVED`, naming what it searched
and what it could not; an answer with hits found under partial coverage says it is partial;
and a search whose coverage was total, `grep_in` over a whole file, says so when it finds
nothing. The same rule keeps an unanalyzed file from ever reading as a clean one.
"""

from __future__ import annotations

from pathlib import Path

from context_helpers import (
    Caller,
    DiagSet,
    FakeDiagnostics,
    new_session,
    numbered,
    python_module,
    reader_for,
    recall_of,
    sessions_dir,
    write_files,
)

from chickadee.context.handle import content_sha
from chickadee.context.recall import Recalled
from chickadee.context.results import IndexUnavailable, ToolResult, UnresolvedSearch
from chickadee.context.session import Session
from chickadee.context.store import LOG_NAME


def healthy(tmp_path: Path, sid: str, *said: str) -> None:
    with new_session(tmp_path, sid) as s:
        for text in said:
            s.record_user(text)
            s.record_assistant("noted")


def corrupt(tmp_path: Path, sid: str) -> None:
    """A session whose second line is damage: it will not open, and a reader says where."""
    healthy(tmp_path, sid, "first words", "second words")
    log = sessions_dir(tmp_path) / sid / LOG_NAME
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = "not a line of any log\n"
    log.write_text("".join(lines), encoding="utf-8")


def test_recall_that_finds_nothing_is_unresolved_and_names_what_it_searched(
    tmp_path: Path,
) -> None:
    healthy(tmp_path, "S-one", "rename the parser")
    healthy(tmp_path, "S-two", "fix the lexer")
    found = recall_of(sessions_dir(tmp_path))("zebra")
    assert isinstance(found, UnresolvedSearch)
    assert found.searched == ("S-one", "S-two")
    assert found.uncovered == ()


def test_a_session_that_will_not_read_is_named_in_uncovered_with_why(tmp_path: Path) -> None:
    healthy(tmp_path, "S-good", "rename the parser")
    corrupt(tmp_path, "S-bad")
    found = recall_of(sessions_dir(tmp_path))("zebra")
    assert isinstance(found, UnresolvedSearch)
    assert found.searched == ("S-good",)
    assert len(found.uncovered) == 1
    assert found.uncovered[0].startswith("S-bad (corrupt at seq 2: ")
    assert "uncovered: [S-bad (corrupt at seq 2" in found.render()


def test_hits_found_under_partial_coverage_say_they_are_partial(tmp_path: Path) -> None:
    healthy(tmp_path, "S-good", "the zebra parser is slow")
    corrupt(tmp_path, "S-bad")
    with new_session(tmp_path, "S-now") as now:
        found = recall_of(sessions_dir(tmp_path))("zebra", current=now.state)
        assert isinstance(found, Recalled)
        assert [h.pointer for h in found.hits] == ["S-good#1"]
        partial = found.notes()
        assert len(partial) == 1
        assert partial[0].startswith("[recall: partial — 1 session(s) could not be read")
        assert "S-bad (corrupt at seq 2" in partial[0]
        reader = reader_for(now, tmp_path)
        assert partial[0] in found.result(reader.store, 4096).notes


def test_nothing_to_search_is_unresolved_with_nothing_searched(tmp_path: Path) -> None:
    found = recall_of(tmp_path / "no-sessions-here")("anything")
    assert found == UnresolvedSearch("anything", (), ())


def test_the_recall_tool_records_unresolved_as_the_model_reads_it(
    tmp_path: Path, session: Session, ws: Path
) -> None:
    corrupt(tmp_path, "S-bad")
    session.record_user("have we seen a zebra before?")
    line = Caller(session, reader_for(session, ws))("recall", query="zebra")
    assert line.text.startswith("UNRESOLVED{query: 'zebra', searched: 1, uncovered: [S-bad")
    assert line.ran, "#571: a search that found nothing ran"


def test_a_whole_file_search_that_finds_nothing_states_its_total_coverage(
    session: Session, ws: Path
) -> None:
    """`grep_in` reads every line of the file, so its empty answer IS a fact — said with
    the coverage that makes it one."""
    write_files(ws, {"a.txt": numbered(300)})
    reader = reader_for(session, ws)
    opened = reader.read("a.txt")
    assert isinstance(opened, ToolResult)
    found = reader.grep_in(opened.handle, "zebra")
    assert isinstance(found, ToolResult)
    assert found.content == ""
    assert "no match for /zebra/ in all 300 lines" in found.header


def test_an_unanalyzed_file_never_renders_as_clean(session: Session, ws: Path) -> None:
    body = numbered(10)
    sha = content_sha(body.encode())
    write_files(ws, {"a.txt": body, "b.txt": body, "c.txt": body + "more\n"})
    index = FakeDiagnostics(
        {
            "b.txt": DiagSet(coverage="total", analyzed_sha=sha),
            "c.txt": DiagSet(coverage="total", analyzed_sha=sha),
        }
    )
    reader = reader_for(session, ws, diagnostics=index)
    verdicts: dict[str, tuple[str, ...]] = {}
    for name in ("a.txt", "b.txt", "c.txt"):
        result = reader.read(name)
        assert isinstance(result, ToolResult)
        verdicts[name] = result.diagnostics
    assert verdicts["a.txt"] == ("[diagnostics: unavailable — no analyzer verdict for .txt]",)
    assert verdicts["b.txt"] == ("[diagnostics: none]",)
    assert verdicts["c.txt"][0].startswith(f"[diagnostics: STALE — judged at {sha[:6]}")
    bare = reader_for(session, ws).read("a.txt")
    assert isinstance(bare, ToolResult)
    assert bare.diagnostics == (
        "[diagnostics: unavailable — no diagnostics index is attached to this read]",
    )


def test_a_verdict_a_dependency_outdated_names_it_and_is_never_clean(
    session: Session, ws: Path
) -> None:
    """The file's bytes are the ones judged, but a file it depends on changed
    since. Its own edit, when it has one too, is the age said."""
    body = numbered(10)
    sha = content_sha(body.encode())
    write_files(ws, {"a.txt": body, "b.txt": body + "edited\n"})
    marked = DiagSet(coverage="stale", analyzed_sha=sha, stale_by="util.py")
    index = FakeDiagnostics({"a.txt": marked, "b.txt": marked})
    reader = reader_for(session, ws, diagnostics=index)
    said = {}
    for name in ("a.txt", "b.txt"):
        result = reader.read(name)
        assert isinstance(result, ToolResult)
        said[name] = result.diagnostics
    assert said["a.txt"] == (
        "[diagnostics: STALE — util.py changed since this was judged · none]",
    )
    assert said["b.txt"][0].startswith(f"[diagnostics: STALE — judged at {sha[:6]}")


def test_a_verdict_on_these_very_bytes_reads_as_current_when_only_the_index_lags(
    session: Session, ws: Path
) -> None:
    """Review of #612: the index's copy is behind the disk (coverage `stale`, no dependency
    named), and the bytes this read returned are the ones judged. It said "judged at X, file is
    now X"; the verdict is on what the reader sees."""
    body = numbered(10)
    sha = content_sha(body.encode())
    write_files(ws, {"a.txt": body})
    index = FakeDiagnostics({"a.txt": DiagSet(coverage="stale", analyzed_sha=sha)})
    result = reader_for(session, ws, diagnostics=index).read("a.txt")
    assert isinstance(result, ToolResult)
    assert result.diagnostics == ("[diagnostics: none]",)


def test_an_outline_request_with_no_index_says_what_coverage_there_was(
    session: Session, ws: Path
) -> None:
    text, _ = python_module(3)
    write_files(ws, {"mod.py": text})
    refused = reader_for(session, ws).read("mod.py", at="f1")
    assert refused == IndexUnavailable("mod.py", "none: no index is attached to this read")
