"""A run's named blobs and its tasks' and steps' diffs (`run_reads`),
over a real episode: the worker edits `app.py` in one step, and the reads answer from the trees
and blobs the journal names, never from the worktree as it is now. A diff's text and its counts
name the same paths, a rename included."""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import APP, WORKER_RUN, EpisodeRig, episode_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import git, init_repo

from chickadee.context.project import JournalRecords
from chickadee.errors import RunNotStarted
from chickadee.handoff import diff_of
from chickadee.inspect import BLOB_LIMIT
from chickadee.journal.effects import EffectKind
from chickadee.journal.runs import RunProjections
from chickadee.run_reads import NotNamed, NotReadable, run_blob, run_diff
from chickadee.schema.journal import TaskDelegatedP
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import diff_stats, tree_of

PLAN = "# Plan: go returns 2\n\n1. **Return 2.** Change `go()`.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
EDIT = call("edit", "w1", path="app.py", old="return 1", new="return 2")


@pytest.fixture
def rig(tmp_path: Path) -> EpisodeRig:
    root = init_repo(tmp_path / "repo", {"app.py": APP})
    rig = episode_rig(root, [], [calls(EDIT), answer("Changed go to return 2.")])  # type: ignore[arg-type]
    assert rig.desk.plan(PLAN, call_id="a-plan").startswith("the person approved the plan")
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    return rig


def _records(rig: EpisodeRig) -> JournalRecords:
    roots = resolve_roots(rig.root)
    return JournalRecords(roots, RunProjections(roots))


def test_a_blob_a_record_names_is_served_as_text(rig: EpisodeRig) -> None:
    [line] = rig.journal.projection().lines(TaskDelegatedP)
    got = run_blob(_records(rig), WORKER_RUN, line.payload.brief_sha)
    assert (got.field, got.text, got.withheld) == ("brief_sha", BRIEF, None)
    assert got.size == len(BRIEF.encode())


def test_a_sha_no_record_names_is_refused(rig: EpisodeRig) -> None:
    stored = rig.journal.put_blob(b"stored, but named by nothing")
    with pytest.raises(NotNamed):
        run_blob(_records(rig), WORKER_RUN, stored)


def test_a_run_with_no_journal_has_not_started(rig: EpisodeRig) -> None:
    with pytest.raises(RunNotStarted):
        run_blob(_records(rig), "2026-10-15-ffffffff", "0" * 64)


def test_a_blob_past_the_inspectors_limit_is_named_not_inlined(
    rig: EpisodeRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chickadee.run_reads as run_reads

    monkeypatch.setattr(run_reads, "BLOB_LIMIT", 4)
    [line] = rig.journal.projection().lines(TaskDelegatedP)
    got = run_blob(_records(rig), WORKER_RUN, line.payload.brief_sha)
    assert (
        got.text is None and got.withheld is not None and "named, not inlined" in got.withheld
    )
    assert BLOB_LIMIT > 4  # the limit is the inspector's, patched here only


def test_a_tasks_diff_is_its_start_tree_to_its_last_step(rig: EpisodeRig) -> None:
    diff = run_diff(_records(rig), rig.root, WORKER_RUN, task_id="T1")
    assert [(f.path, f.kind, f.added, f.removed) for f in diff.files] == [
        ("app.py", "edited", 1, 1)
    ]
    assert "-    return 1" in diff.text and "+    return 2" in diff.text


def test_a_steps_diff_is_its_tree_before_to_its_tree_after(rig: EpisodeRig) -> None:
    steps = [
        op.operation_id
        for op in rig.journal.projection().operations.values()
        if op.kind == EffectKind.WORKER_STEP.value and op.task_id == "T1"
    ]
    assert steps, "the edit was a step"
    diff = run_diff(_records(rig), rig.root, WORKER_RUN, operation_id=steps[0])
    assert [f.path for f in diff.files] == ["app.py"] and diff.tip is not None


def test_a_diff_names_one_task_or_one_step_the_run_holds(rig: EpisodeRig) -> None:
    records = _records(rig)
    with pytest.raises(NotReadable):
        run_diff(records, rig.root, WORKER_RUN)
    with pytest.raises(NotReadable):
        run_diff(records, rig.root, WORKER_RUN, task_id="T9")
    with pytest.raises(NotReadable):
        run_diff(records, rig.root, WORKER_RUN, operation_id="no-such-step")


def test_a_rename_is_a_deletion_and_a_creation_in_the_counts_and_the_text(
    tmp_path: Path,
) -> None:
    """`diff_of` never follows renames, whatever the user's git config says, so the paths its
    text names are the paths `diff_stats` counts (the TUI draws one beside the other)."""
    root = init_repo(tmp_path / "repo", {"old.py": "".join(f"line {i}\n" for i in range(20))})
    git(root, "config", "diff.renames", "true")
    base = tree_of(root, "HEAD")
    git(root, "mv", "old.py", "new.py")
    git(root, "commit", "-qm", "move")
    tip = tree_of(root, "HEAD")
    assert [(f.path, f.kind) for f in diff_stats(root, base, tip)] == [
        ("new.py", "created"),
        ("old.py", "deleted"),
    ]
    text = diff_of(root, base, tip)
    assert "rename from" not in text
    assert "+++ b/new.py" in text and "--- a/old.py" in text
