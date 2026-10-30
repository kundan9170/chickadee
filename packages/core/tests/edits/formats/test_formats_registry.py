"""`FORMATS`: one registry, the default `whole_file`, and its parser's
answers on several files.

The templates that framed each format's prompt, and the one-shot adapters that read a reply
through the packet rules, retired with the packet loop."""

from __future__ import annotations

from formats_helpers import C, E, edits, parse, targets, whole

from chickadee.edits.formats import DEFAULT, FORMATS
from chickadee.schema.adapter import ParseError


def test_the_registry_holds_the_four_formats_and_whole_file_is_the_default() -> None:
    assert sorted(FORMATS) == ["apply_patch", "search_replace", "unified_diff", "whole_file"]
    assert all(name == fmt.name for name, fmt in FORMATS.items())
    assert DEFAULT is FORMATS["whole_file"]


def test_a_whole_file_per_path_is_each_paths_content() -> None:
    """The packet rules made `new.py`'s fence a create, as its target declared,
    and ended each file with a newline; the parser says what each fence holds, as written."""
    given = targets(("app.py", E), ("new.py", C))
    reply = "```python path=app.py\nA = 1\n```\n```python path=new.py\nB = 2\n```"
    assert parse("whole_file", reply, given) == edits(
        whole("app.py", "A = 1"), whole("new.py", "B = 2")
    )


def test_a_fence_holding_markers_is_refused() -> None:
    """Moved by T-3.128: the parser refuses markers before anything sees the missing
    target. Before: the missing target's hint, `emit one fence per TARGETS path, in that order,
    naming each path`. Both are `bad_format`."""
    given = targets(("a.py", E), ("b.py", E))
    reply = "```python path=a.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n```"
    assert parse("whole_file", reply, given) == ParseError(
        kind="bad_format",
        hint="emit the complete file; SEARCH/REPLACE hunks are not applied yet",
    )
