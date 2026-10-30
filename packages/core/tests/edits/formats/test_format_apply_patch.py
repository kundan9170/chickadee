"""`apply_patch`: Codex's V4A envelope to exact-text hunks, each keeping the scope
its `@@` lines name; `*** Add File` creates; deletes and moves read."""

from __future__ import annotations

from formats_helpers import FILE, C, E, applied, create, edits, parse, targets

from chickadee.edits.formats import apply_patch, parse_patch
from chickadee.edits.formats.base import FileEdits, Hunks, Move, Targets
from chickadee.edits.hunk import Hunk
from chickadee.schema.adapter import ParseError

PATCH = (
    "*** Begin Patch\n*** Update File: app.py\n@@ def go\n def go() -> int:\n"
    "-    return 1\n+    return 2\n*** End Patch"
)


def _lines(old: str, new: str) -> Hunk:
    return Hunk(old=old, new=new, lines=True)


def test_an_update_section_is_one_hunk_keeping_its_scope_and_applies() -> None:
    result = parse("apply_patch", f"Patch:\n{PATCH}\n")
    one = _lines("def go() -> int:\n    return 1\n", "def go() -> int:\n    return 2\n")
    assert result == edits(("app.py", Hunks((one,), (("def go",),))))
    assert applied(result) == FILE.replace("return 1", "return 2")


def test_at_lines_split_a_section_into_hunks_and_sections_into_files() -> None:
    given = targets(("a.py", E), ("new.py", C))
    reply = (
        "*** Begin Patch\n*** Add File: new.py\n+X = 1\n+\n+Y = 2\n"
        "*** Update File: a.py\n@@\n x\n-y\n+Y\n@@ class K\n z\n-w\n+W\n*** End of File\n"
        "*** End Patch\n"
    )
    # The packet rules ordered the set by its targets (a.py, new.py); the
    # parser keeps the reply's order.
    a = Hunks((_lines("x\ny\n", "x\nY\n"), _lines("z\nw\n", "z\nW\n")), ((), ("class K",)))
    assert parse("apply_patch", reply, given) == edits(
        create("new.py", "X = 1\n\nY = 2\n"), ("a.py", a)
    )


def test_refusals() -> None:
    def hint(body: str, *, closed: bool = True) -> str:
        end = "\n*** End Patch" if closed else ""
        result = parse("apply_patch", f"*** Begin Patch\n{body}{end}")
        assert isinstance(result, ParseError) and result.kind == "bad_format", result
        return result.hint

    assert hint("*** Update File: app.py\n-x\n+y", closed=False) == (
        "close the patch with *** End Patch"
    )
    assert (
        hint("-x\n+y") == "open a file with *** Update File: or *** Add File: before its lines"
    )
    assert hint("*** Update File: app.py\nx\n+y") == (
        "app.py: each hunk line starts with a space, - or +"
    )
    assert hint("*** Add File: app.py\nx") == "app.py: each *** Add File: line starts with +"
    assert hint("*** Update File: app.py\n-x\n+y\n*** Update File: app.py\n-z\n+w") == (
        "app.py: one section per file"
    )


def test_no_envelope_is_nothing_written() -> None:
    """The one-shot adapter read an empty parse as a tool request
    (`ToolRequest(read)`) or as `no_block`; the parser's own answer is that the reply wrote
    nothing, and that adapter retired."""
    assert parse("apply_patch", "TOOL read path=app.py") == FileEdits()
    assert parse("apply_patch", "-x\n+y") == FileEdits()


def test_an_empty_section_and_a_hunk_that_changes_nothing_are_refused() -> None:
    """Review: an empty section became a patch that could not be built, and the
    reply was told to emit fenced files."""
    for body, hint in (
        ("*** Update File: app.py\n@@", "app.py: an *** Update File: section needs a hunk"),
        ("*** Update File: app.py\n x", "app.py: a hunk changes nothing; mark lines - or +"),
    ):
        reply = f"*** Begin Patch\n{body}\n*** End Patch"
        assert parse("apply_patch", reply) == ParseError(kind="bad_format", hint=hint)


def test_an_indented_marker_is_a_context_line_quoting_v4a() -> None:
    """Review: markers were matched on stripped lines, so a context line holding
    `*** End Patch` closed the envelope early."""
    text = "Write:\n *** End Patch\n *** Add File: x\nend\n"
    reply = (
        "*** Begin Patch\n*** Update File: app.py\n  *** End Patch\n  *** Add File: x\n"
        "-end\n+done\n*** End Patch"
    )
    assert applied(parse("apply_patch", reply), text) == (
        "Write:\n *** End Patch\n *** Add File: x\ndone\n"
    )


def test_each_hunk_keeps_the_scope_its_at_lines_name_and_nested_ones_nest() -> None:
    """One-shot ignores a scope; the worker's draft honours it."""
    reply = (
        "*** Begin Patch\n*** Update File: app.py\n"
        "@@ class A\n@@ def go\n-    return 1\n+    return 2\n"
        "@@\n-x = 1\n+x = 2\n"
        "@@ def stop\n-    return 3\n+    return 4\n"
        "*** End Patch"
    )
    edits = apply_patch.parse(reply, Targets(writable=("app.py",)))
    assert isinstance(edits, FileEdits)
    (_, found) = edits.edits[0]
    assert isinstance(found, Hunks)
    assert found.scopes == (("class A", "def go"), (), ("def stop",))
    assert len(found.hunks) == 3


def test_parse_patch_reads_v4a_or_else_a_unified_diff() -> None:
    targets = Targets(writable=("app.py",))
    v4a = parse_patch(PATCH, targets)
    diff = parse_patch("--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n", targets)
    assert isinstance(v4a, FileEdits) and isinstance(diff, FileEdits)
    assert [p for p, _ in v4a.edits] == [p for p, _ in diff.edits] == ["app.py"]
    assert parse_patch("just prose", targets) == FileEdits()


def test_a_unified_diff_header_in_a_v4a_patch_names_no_scope() -> None:
    reply = (
        "*** Begin Patch\n*** Update File: app.py\n@@ -1,2 +1,2 @@\n"
        "-x = 1\n+x = 2\n*** End Patch"
    )
    edits = apply_patch.parse(reply, Targets(writable=("app.py",)))
    assert isinstance(edits, FileEdits)
    (_, found) = edits.edits[0]
    assert isinstance(found, Hunks) and found.scopes == ((),)


def test_a_delete_and_a_move_are_read_for_the_box() -> None:
    """The parser reads both, and the box's edit tool performs them."""
    text = (
        "*** Begin Patch\n*** Delete File: old.py\n"
        "*** Update File: a.py\n*** Move to: pkg/a.py\n-x\n+y\n"
        "*** Update File: b.py\n*** Move to: pkg/b.py\n*** End Patch"
    )
    parsed = apply_patch.parse(text, Targets(writable=()))
    assert isinstance(parsed, FileEdits)
    assert [(p, type(e).__name__) for p, e in parsed.edits] == [
        ("old.py", "Delete"),
        ("a.py", "Move"),
        ("b.py", "Move"),
    ]
    _, moved = parsed.edits[1]
    assert isinstance(moved, Move) and moved.target == "pkg/a.py"
    assert moved.hunks is not None and moved.hunks.hunks[0].new == "y\n"
    _, renamed = parsed.edits[2]
    assert renamed == Move("pkg/b.py")
