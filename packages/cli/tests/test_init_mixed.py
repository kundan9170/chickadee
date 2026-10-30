"""Several profiles coexist; unsupported files are gaps."""

from __future__ import annotations

import pathlib

from chickadee.init import detect, init


def test_python_and_javascript_coexist(repo: pathlib.Path) -> None:
    """No repo-wide winner and no majority language: forcing one would send half the
    repository through gates that cannot read it."""
    (repo / "app.js").write_text("export const x = 1;\n")
    names = [e.name for e in init(repo).profiles]
    assert "python" in names and "javascript" in names


def test_evidence_records_counts_not_guesses(repo: pathlib.Path) -> None:
    (repo / "b.py").write_text("x = 1\n")
    (repo / "app.js").write_text("export const x = 1;\n")
    report = init(repo)
    by_name = {e.name: e for e in report.profiles}
    assert by_name["python"].file_count == 2
    assert by_name["javascript"].file_count == 1
    assert by_name["python"].evidence


def test_unsupported_files_are_gaps_not_defaults(repo: pathlib.Path) -> None:
    """A file the harness cannot parse must be visible as a hole: a graph answering "no
    callers" for an unparsed file answers the same as one with no callers at all."""
    (repo / "notes.txt").write_text("hello\n")
    (repo / "main.zig").write_text("pub fn main() void {}\n")
    report = init(repo)
    assert "main.zig" in report.gaps
    assert "notes.txt" in report.gaps
    assert all(e.name in {"python", "javascript"} for e in report.profiles)


def test_profile_order_is_deterministic(repo: pathlib.Path) -> None:
    """A report whose rows move between runs cannot be diffed."""
    (repo / "app.js").write_text("export const x = 1;\n")
    assert [e.name for e in init(repo).profiles] == [e.name for e in init(repo).profiles]


def test_vendored_directories_are_not_scanned(repo: pathlib.Path) -> None:
    """node_modules is not the user's code, and counting it would make every JavaScript
    repository look like it is mostly other people's."""
    (repo / "node_modules" / "dep").mkdir(parents=True)
    (repo / "node_modules" / "dep" / "index.js").write_text("module.exports = 1;\n")
    report = init(repo)
    assert not any("node_modules" in g for g in report.gaps)
    js = next((e for e in report.profiles if e.name == "javascript"), None)
    assert js is None or js.file_count == 0


def test_detection_needs_no_initialised_repo(repo: pathlib.Path) -> None:
    evidence, gaps = detect.detect_profiles(repo)
    assert [e.name for e in evidence] == ["python"]
    assert gaps == ()
