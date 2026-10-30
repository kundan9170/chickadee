"""The worker's edit tools: `write`, `edit` and `apply_patch`, each a
resolver writing a draft through the `edits` port, on the task worker's own dispatcher over
the box's port on a real git worktree (it was the packet's view)."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import init_repo

from .tools_helpers import EditWorker, refused, said, worktree_edits

APP = "def go() -> int:\n    return 1\n\n\ndef stop() -> None:\n    return None\n"
EDITS = ("write", "edit", "apply_patch")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    # `debug.log` is ignored, so never committed: a file undo could not bring back.
    files = {"app.py": APP, ".gitignore": "*.log\n", "debug.log": "x\n"}
    return init_repo(tmp_path / "repo", files)


@pytest.fixture
def run(root: Path) -> EditWorker:
    return EditWorker(worktree_edits(root), EDITS)


def _text(root: Path, path: str = "app.py") -> str:
    return (root / path).read_text(encoding="utf-8")


# -- edit -----------------------------------------------------------------------------------


def test_an_edit_lands_in_the_worktree_and_every_read_of_the_target_sees_it(
    run: EditWorker, root: Path
) -> None:
    answer = said(run.call("edit", path="app.py", old="return 1", new="return 2"))
    assert answer == "ok: app.py\n    return 2"
    assert _text(root) == APP.replace("1", "2")


def test_a_missed_anchor_names_the_nearest_lines_and_the_next_call_re_anchors(
    run: EditWorker,
) -> None:
    reason = refused(
        run.call("edit", path="app.py", old="return  1", new="return 2"), "NO_MATCH"
    )
    assert "nearest lines" in reason and "2│    return 1" in reason
    said(run.call("edit", path="app.py", old="return 1", new="return 2"))


def test_an_ambiguous_anchor_is_refused_and_replace_all_takes_each(
    run: EditWorker, root: Path
) -> None:
    refused(run.call("edit", path="app.py", old="return", new="yield"), "AMBIGUOUS")
    every = said(run.call("edit", path="app.py", old="return", new="yield", replace_all=True))
    assert every.count("ok: app.py") == 2
    assert _text(root).count("yield") == 2


def test_an_edits_list_is_one_call_every_edit_or_none(run: EditWorker, root: Path) -> None:
    failed = run.call(
        "edit",
        path="app.py",
        edits=[{"old": "return 1", "new": "return 2"}, {"old": "nowhere", "new": "x"}],
    )
    assert "edit 1: old not found" in refused(failed, "NO_MATCH")
    assert _text(root) == APP
    said(
        run.call(
            "edit",
            path="app.py",
            edits=[
                {"old": "return 1", "new": "return 2"},
                {"old": "return 2", "new": "return 3"},
            ],
        )
    )
    assert "return 3" in _text(root)


def test_an_edit_outside_the_worktree_is_denied(run: EditWorker, tmp_path: Path) -> None:
    """The packet's `not a TARGETS path` in older versions; the worktree's bound since."""
    (tmp_path / "helpers.py").write_text("a = 1\n", encoding="utf-8")
    reason = refused(run.call("edit", path="../helpers.py", old="a", new="b"), "DENIED")
    assert "escapes with '..'" in reason
    assert (tmp_path / "helpers.py").read_text(encoding="utf-8") == "a = 1\n"


# -- write ------------------------------------------------------------------------------------


def test_an_absent_file_is_absent_until_written_whole(run: EditWorker, root: Path) -> None:
    refused(run.call("edit", path="new.py", old="a", new="b"), "FILE_ABSENT")
    written = said(run.call("write", path="new.py", content="x = 1\n"))
    assert written == "ok: new.py written, 1 lines"
    assert _text(root, "new.py") == "x = 1\n"


def test_a_write_eliding_code_is_refused(run: EditWorker, root: Path) -> None:
    reason = refused(
        run.call(
            "write",
            path="app.py",
            content="def go() -> int:\n    return 2\n# ... rest of the file unchanged\n",
        ),
        "ELIDED",
    )
    assert "line 3" in reason
    assert _text(root) == APP


# -- apply_patch ------------------------------------------------------------------------------


def _v4a(*lines: str) -> str:
    return "\n".join(["*** Begin Patch", *lines, "*** End Patch"]) + "\n"


def test_a_v4a_hunk_is_matched_within_its_scope(run: EditWorker, root: Path) -> None:
    patch = _v4a(
        "*** Update File: app.py", "@@ def stop() -> None:", "-    return None", "+    pass"
    )
    said(run.call("apply_patch", patch=patch))
    assert _text(root).endswith("def stop() -> None:\n    pass\n")
    scoped = _v4a("*** Update File: app.py", "@@ def nowhere():", "-    pass", "+    x")
    assert "scope" in refused(run.call("apply_patch", patch=scoped), "NO_MATCH")


def test_a_unified_diff_applies_too(run: EditWorker, root: Path) -> None:
    diff = "--- a/app.py\n+++ b/app.py\n@@ -2 +2 @@\n-    return 1\n+    return 7\n"
    said(run.call("apply_patch", patch=diff))
    assert "return 7" in _text(root)


def test_a_patch_deleting_what_undo_cannot_restore_or_adding_an_existing_file_is_refused(
    run: EditWorker, root: Path
) -> None:
    # A packet's draft, given no `removable`, refused every delete ("the
    # planner declares them"); the box's port refuses only what undo could not bring back
    # and a tracked file goes.
    ignored = refused(
        run.call("apply_patch", patch=_v4a("*** Delete File: debug.log")), "DENIED"
    )
    assert "git-ignored" in ignored and (root / "debug.log").exists()
    adding = _v4a("*** Add File: app.py", "+x = 1")
    assert "exists" in refused(run.call("apply_patch", patch=adding), "DENIED")
    refused(run.call("apply_patch", patch="prose"), "BAD_FORMAT")
    assert said(run.call("apply_patch", patch=_v4a("*** Delete File: app.py"))) == (
        "ok: app.py deleted"
    )
    assert not (root / "app.py").exists()


def test_replace_all_beside_edits_is_each_edits_default(run: EditWorker, root: Path) -> None:
    said(
        run.call(
            "edit",
            path="app.py",
            replace_all=True,
            edits=[{"old": "return", "new": "yield"}],
        )
    )
    assert _text(root).count("yield") == 2
