"""D111 / T-2.31 — the record is kept, the bulk is pruned on request, and every prune is
a line in `gc.log` a later miss can name."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
from typing import Any

import pytest
from helpers import payload_for, scope_args

from chickadee.journal import BlobMissing, Journal, JournalError, paths, project, read_events
from chickadee.journal.gc import (
    BULK,
    BlobClass,
    apply,
    classify,
    gc_log_for,
    plan,
    pruned_record,
)
from chickadee.journal.writer import RunLock
from chickadee.schema import EventKind
from chickadee.state.roots import Roots

OLD = "2026-08-18-a1b2c3d4"
YOUNG = "2026-09-20-0000ffee"
OTHER_OLD = "2026-08-18-b2c3d4e5"

# A run's prompt is its box roles' conversation: `box_message`'s bare `sha`.
BYTES = {
    BlobClass.PROMPT: b"the whole prompt",
    BlobClass.OUTPUT: b"checkpoint stdout",
    BlobClass.PREIMAGE: b"pre-image bytes",
    BlobClass.RECORD: b"the digest",
    BlobClass.UNREFERENCED: b"nobody names me",
}


def write_run(
    repo: Roots,
    run_id: str,
    *,
    terminal: bool = True,
    shared: bytes | None = None,
    own: bool = False,
    handed_off: bool = False,
) -> dict[BlobClass, str]:
    """A run with one blob per class, each named by the field that gives it its class.

    `own` salts every blob with the run id, so two fixture runs do not share every sha by
    accident — identical bytes are one sha, which is exactly the dedupe rule under test.

    `handed_off` ends the run the way the orchestrator does: the handoff text is a
    blob, and `run_end.handoff_path` names it as `blobs/<sha>`.
    """
    shas: dict[BlobClass, str] = {}
    with Journal.create(repo, run_id, {}) as j:
        for cls, data in BYTES.items():
            shas[cls] = j.put_blob(data + run_id.encode() if own else data)
        if shared is not None:
            shas[BlobClass.COMPLETION] = j.put_blob(shared)  # stands in for a shared sha
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(
            EventKind.BOX_MESSAGE,
            payload_for(EventKind.BOX_MESSAGE, side="user", sha=shas[BlobClass.PROMPT]),
            **scope_args(EventKind.BOX_MESSAGE),
        )
        j.append(
            EventKind.TOOL_CALL,
            payload_for(
                EventKind.TOOL_CALL,
                result_sha=shas.get(BlobClass.COMPLETION, shas[BlobClass.OUTPUT]),
            ),
            **scope_args(EventKind.TOOL_CALL),
        )
        j.append(
            EventKind.EXEC,
            payload_for(
                EventKind.EXEC,
                stdout_sha=shas[BlobClass.OUTPUT],
                stderr_sha=shas[BlobClass.OUTPUT],
            ),
            **scope_args(EventKind.EXEC),
        )
        j.append(
            EventKind.EFFECT_INTENDED,
            payload_for(EventKind.EFFECT_INTENDED, preimages_sha=shas[BlobClass.PREIMAGE]),
            **scope_args(EventKind.EFFECT_INTENDED),
        )
        j.append(
            EventKind.TASK_DELEGATED,
            payload_for(EventKind.TASK_DELEGATED, brief_sha=shas[BlobClass.RECORD]),
            **scope_args(EventKind.TASK_DELEGATED),
        )
        if terminal:
            handoff = "x"
            if handed_off:
                handoff = f"{paths.BLOBS_DIR}/{j.put_blob(HANDOFF + run_id.encode())}"
            j.append(EventKind.RUN_END, payload_for(EventKind.RUN_END, handoff_path=handoff))
    return shas


HANDOFF = b"# handoff\n"


def past_everything() -> dt.datetime:
    """A cutoff after every event written so far: the run is old by definition."""
    return dt.datetime.now(dt.UTC) + dt.timedelta(seconds=1)


def blobs_on_disk(repo: Roots, run_id: str) -> set[str]:
    return {p.name for p in paths.blobs_dir(repo, run_id).iterdir()}


def state_of(repo: Roots, run_id: str):  # type: ignore[no-untyped-def]
    return project(read_events(paths.journal_path(repo, run_id)))


# --- 1. classify ------------------------------------------------------------------------


def test_classify_names_every_class_from_the_field_that_names_the_sha(repo: Roots) -> None:
    shas = write_run(repo, OLD)
    classes = classify(paths.run_dir(repo, OLD))
    for cls, sha in shas.items():
        assert classes[sha] is cls, cls
    assert set(classes) >= set(shas.values()), "every blob on disk is classified"


def test_a_sha_any_record_field_names_is_record(repo: Roots) -> None:
    """Named by `result_sha` and by `brief_sha`: record wins, whatever else names it."""
    with Journal.create(repo, OLD, {}) as j:
        sha = j.put_blob(b"both")
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(
            EventKind.TOOL_CALL,
            payload_for(EventKind.TOOL_CALL, result_sha=sha),
            **scope_args(EventKind.TOOL_CALL),
        )
        j.append(
            EventKind.TASK_DELEGATED,
            payload_for(EventKind.TASK_DELEGATED, brief_sha=sha),
            **scope_args(EventKind.TASK_DELEGATED),
        )
        j.append(EventKind.RUN_END, payload_for(EventKind.RUN_END))
    assert classify(paths.run_dir(repo, OLD))[sha] is BlobClass.RECORD


def test_an_unknown_sha_field_is_record_not_bulk(repo: Roots) -> None:
    """The safe direction for a field the table does not know is keep."""
    with Journal.create(repo, OLD, {}) as j:
        sha = j.put_blob(b"args")
        j.append(
            EventKind.SCOUT_TOOL,
            payload_for(EventKind.SCOUT_TOOL, args_sha=sha),
            **scope_args(EventKind.SCOUT_TOOL),
        )
    assert classify(paths.run_dir(repo, OLD))[sha] is BlobClass.RECORD


def test_a_bare_sha_is_classed_by_its_line(repo: Roots) -> None:
    """Contracts v2 names some blobs by a bare `sha`. A box message's is model I/O, its
    assistant side a completion; `plan_written`'s is the plan the person approved at stop 2,
    record. Missed as unreferenced, the default classes pruned the plan."""
    with Journal.create(repo, OLD, {}) as j:
        said = j.put_blob(b"the worker's reply")
        approved = j.put_blob(b"1. **Return 2.**")
        j.append(
            EventKind.BOX_MESSAGE,
            payload_for(EventKind.BOX_MESSAGE, side="assistant", sha=said),
            **scope_args(EventKind.BOX_MESSAGE),
        )
        j.append(
            EventKind.PLAN_WRITTEN,
            payload_for(EventKind.PLAN_WRITTEN, sha=approved),
            **scope_args(EventKind.PLAN_WRITTEN),
        )
    classes = classify(paths.run_dir(repo, OLD))
    assert (classes[said], classes[approved]) == (BlobClass.COMPLETION, BlobClass.RECORD)


def test_classify_is_pure(repo: Roots) -> None:
    write_run(repo, OLD)
    before = blobs_on_disk(repo, OLD)
    classify(paths.run_dir(repo, OLD))
    assert blobs_on_disk(repo, OLD) == before


# --- 2. plan ------------------------------------------------------------------------------


def test_plan_lists_everything_but_record_once_the_run_is_old_and_handed_off(
    repo: Roots,
) -> None:
    shas = write_run(repo, OLD, handed_off=True)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    (row,) = p.rows
    assert row.run_id == OLD
    assert set(row.shas) == {sha for cls, sha in shas.items() if cls is not BlobClass.RECORD}
    assert row.classes == {
        BlobClass.PROMPT,
        BlobClass.OUTPUT,
        BlobClass.PREIMAGE,
        BlobClass.UNREFERENCED,
    }
    assert row.bytes == sum(len(BYTES[c]) for c in row.classes)
    assert p.skipped == ()
    assert blobs_on_disk(repo, OLD) >= set(shas.values()), "plan unlinks nothing"


def test_preimages_are_kept_until_the_handoff_exists(repo: Roots) -> None:
    """Until then they are the rollback path."""
    shas = write_run(repo, OLD)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    (row,) = p.rows
    assert shas[BlobClass.PREIMAGE] not in row.shas
    assert BlobClass.PREIMAGE not in row.classes


def handoff_sha(repo: Roots, run_id: str) -> str:
    end = [
        e for e in read_events(paths.journal_path(repo, run_id)) if e.kind is EventKind.RUN_END
    ]
    return end[-1].payload.model_dump()["handoff_path"].removeprefix(f"{paths.BLOBS_DIR}/")


def test_preimages_go_once_run_end_names_a_handoff_the_run_holds(repo: Roots) -> None:
    """ "handed off" is the journal's word, not a file's. The orchestrator writes no
    `handoff.md`; it blobs the handoff and names it in `run_end.handoff_path`."""
    shas = write_run(repo, OLD, handed_off=True)
    (row,) = plan(
        paths.runs_root(repo), older_than=past_everything(), classes={BlobClass.PREIMAGE}
    ).rows
    assert row.shas == (shas[BlobClass.PREIMAGE],)


def test_a_handoff_md_file_does_not_hand_the_run_off(repo: Roots) -> None:
    """A stray `handoff.md` is not the handoff: the preimages stay the rollback path."""
    write_run(repo, OLD)
    (paths.run_dir(repo, OLD) / "handoff.md").write_text("done\n", encoding="utf-8")
    p = plan(paths.runs_root(repo), older_than=past_everything(), classes={BlobClass.PREIMAGE})
    assert p.rows == ()


def test_preimages_stay_when_the_handoff_blob_is_missing(repo: Roots) -> None:
    write_run(repo, OLD, handed_off=True)
    (paths.blobs_dir(repo, OLD) / handoff_sha(repo, OLD)).unlink()
    p = plan(paths.runs_root(repo), older_than=past_everything(), classes={BlobClass.PREIMAGE})
    assert p.rows == ()


def test_preimages_stay_in_a_run_without_run_end(repo: Roots) -> None:
    write_run(repo, OLD, terminal=False)
    p = plan(paths.runs_root(repo), older_than=past_everything(), classes={BlobClass.PREIMAGE})
    assert p.rows == ()


def test_the_handoff_blob_is_record_and_never_pruned(repo: Roots) -> None:
    """`handoff_path` is not a `*_sha` field, so the blob it names read as unreferenced, which
    the default classes prune: `gc` deleted the handoff, the record J-6 keeps, and with it
    the very proof that let preimages go."""
    write_run(repo, OLD, handed_off=True)
    sha = handoff_sha(repo, OLD)
    assert classify(paths.run_dir(repo, OLD))[sha] is BlobClass.RECORD
    apply(plan(paths.runs_root(repo), older_than=past_everything()))
    assert sha in blobs_on_disk(repo, OLD)


def test_a_young_run_is_skipped_and_named(repo: Roots) -> None:
    write_run(repo, OLD)
    p = plan(paths.runs_root(repo), older_than=dt.timedelta(days=30))
    assert p.rows == ()
    assert [r for r, _ in p.skipped] == [OLD]
    assert "younger than the cutoff" in p.skipped[0][1]


def test_a_run_without_a_terminal_is_skipped_and_named(repo: Roots) -> None:
    write_run(repo, OLD, terminal=False)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    assert p.rows == ()
    assert p.skipped == ((OLD, "no terminal — the run is in flight or resumable"),)


def test_a_locked_run_is_skipped_and_named(repo: Roots) -> None:
    """A held LOCK is a live writer, whatever the journal says."""
    write_run(repo, OLD)
    held = RunLock.acquire(repo, OLD)
    try:
        p = plan(paths.runs_root(repo), older_than=past_everything())
    finally:
        held.release()
    assert p.rows == ()
    assert p.skipped == ((OLD, "LOCK is held by a live process (J-4)"),)
    # Released, the leftover LOCK file is inert and the run qualifies again.
    assert plan(paths.runs_root(repo), older_than=past_everything()).rows != ()


def test_a_lock_that_cannot_be_opened_is_held(
    repo: Roots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A LOCK the planner cannot open is not evidence of a dead writer. "Cannot tell"
    reading as "not held" would prune a run whose writer may be alive; it must skip the
    run and name it, and the person can look."""
    write_run(repo, OLD)
    lock = paths.lock_path(repo, OLD)
    assert lock.is_file(), "a finished run leaves its LOCK file behind"
    real_open = os.open

    def refuse(path: Any, *args: Any, **kwargs: Any) -> int:
        if pathlib.Path(os.fsdecode(path)) == lock:
            raise PermissionError(13, "Permission denied", os.fsdecode(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", refuse)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    assert p.rows == ()
    assert p.skipped == ((OLD, "LOCK is held by a live process (J-4)"),)


def test_an_old_runs_blobs_go_even_when_a_young_run_holds_the_same_bytes(
    repo: Roots,
) -> None:
    """A run's blobs are its own: `put_blob` writes `runs/<run_id>/blobs/<sha>`, so two runs
    holding the same bytes hold two files and unlinking one cannot reach the other.

    D111's rule — a sha goes only when every run naming it qualifies — is written for a
    *shared* store. Applied here it protected nothing and cost most of the feature: model I/O
    blobs are byte-identical across runs by construction, so one recent run pinned that
    content in every old run. What must hold is the safety property underneath the rule,
    asserted below: the young run's own copy survives the prune.
    """
    old = write_run(repo, OLD, shared=b"shared prefix")
    young = write_run(repo, YOUNG, shared=b"shared prefix", own=True)
    shared = old[BlobClass.COMPLETION]
    assert shared == young[BlobClass.COMPLETION], "the fixture must actually share a sha"
    # Age the old run's journal by rewriting its timestamps — the record is the clock.
    _age(paths.journal_path(repo, OLD), days=40)

    p = plan(paths.runs_root(repo), older_than=dt.timedelta(days=30), run=OLD)
    (row,) = p.rows
    assert shared in row.shas, "the old run's copy is the old run's to prune"
    assert old[BlobClass.OUTPUT] in row.shas, "the old run's own blobs still go"

    apply(p)
    assert not (paths.blobs_dir(repo, OLD) / shared).is_file(), "the old copy is gone"
    assert (paths.blobs_dir(repo, YOUNG) / shared).is_file(), (
        "the young run keeps its own copy — that is the property the rule was protecting"
    )


def test_pruning_one_run_does_not_pin_the_same_bytes_in_another(repo: Roots) -> None:
    """The ratchet the cross-run rule created: once a run was pruned it still *named* the
    sha it no longer *had*, so every other copy of those bytes became unprunable and each
    gc made the next one less effective."""
    write_run(repo, OLD, shared=b"shared prefix")
    write_run(repo, OTHER_OLD, shared=b"shared prefix", own=True)
    for run_id in (OLD, OTHER_OLD):
        _age(paths.journal_path(repo, run_id), days=40)
    cutoff = dt.timedelta(days=30)

    apply(plan(paths.runs_root(repo), older_than=cutoff, run=OLD))

    after = plan(paths.runs_root(repo), older_than=cutoff, run=OTHER_OLD)
    assert after.rows, "the second run is still prunable after the first was pruned"


def test_classes_narrow_the_plan_and_record_is_refused(repo: Roots) -> None:
    shas = write_run(repo, OLD)
    p = plan(paths.runs_root(repo), older_than=past_everything(), classes=(BlobClass.PROMPT,))
    (row,) = p.rows
    assert row.shas == (shas[BlobClass.PROMPT],)
    with pytest.raises(JournalError, match="never pruned"):
        plan(paths.runs_root(repo), older_than=past_everything(), classes=(BlobClass.RECORD,))
    assert BlobClass.RECORD not in BULK


def test_a_naive_cutoff_is_refused(repo: Roots) -> None:
    with pytest.raises(JournalError, match="timezone-aware"):
        plan(paths.runs_root(repo), older_than=dt.datetime(2026, 1, 1))


# --- 3. apply -----------------------------------------------------------------------------


def test_apply_unlinks_exactly_the_plan_and_logs_one_line_per_run(repo: Roots) -> None:
    shas = write_run(repo, OLD, handed_off=True)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    before = blobs_on_disk(repo, OLD)

    result = apply(p)

    assert blobs_on_disk(repo, OLD) == before - set(p.rows[0].shas)
    assert shas[BlobClass.RECORD] in blobs_on_disk(repo, OLD), "the record is never pruned"
    assert result.unlinked == len(p.rows[0].shas)
    assert result.bytes == p.bytes
    assert result.log == paths.gc_log_path(repo) == gc_log_for(paths.runs_root(repo))
    lines = [json.loads(ln) for ln in result.log.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    (line,) = lines
    assert line["run_id"] == OLD
    assert line["count"] == result.unlinked and line["bytes"] == result.bytes
    assert set(line["classes"]) == {"prompt", "output", "preimage", "unreferenced"}
    assert line["cutoff"] == p.cutoff.isoformat(timespec="seconds")
    assert line["at"] == result.at and line["user"]
    assert set(line["shas"]) == set(p.rows[0].shas)


def test_the_journal_replays_identically_before_and_after_a_prune(repo: Roots) -> None:
    """`project()` folds the journal and never the blob."""
    write_run(repo, OLD, handed_off=True)
    journal_bytes = paths.journal_path(repo, OLD).read_bytes()
    before = state_of(repo, OLD)

    apply(plan(paths.runs_root(repo), older_than=past_everything()))

    assert paths.journal_path(repo, OLD).read_bytes() == journal_bytes, "never opened to write"
    assert state_of(repo, OLD) == before


def test_a_pruned_blob_is_a_typed_miss_naming_the_log_line(repo: Roots) -> None:
    shas = write_run(repo, OLD)
    result = apply(plan(paths.runs_root(repo), older_than=past_everything()))

    journal = Journal.open_for_resume(repo, OLD)
    try:
        with pytest.raises(BlobMissing) as pruned:
            journal.get_blob(shas[BlobClass.OUTPUT])
        with pytest.raises(BlobMissing) as absent:
            journal.get_blob("f" * 64)
        assert journal.get_blob(shas[BlobClass.RECORD]) == BYTES[BlobClass.RECORD]
    finally:
        journal.close()

    assert pruned.value.why == "pruned"
    assert pruned.value.run_id == OLD
    assert pruned.value.gc_at == result.at
    assert pruned.value.user and "gc.log" in str(pruned.value)
    assert absent.value.why == "absent"
    assert absent.value.run_id is None

    line = pruned_record(paths.runs_root(repo), OLD, shas[BlobClass.OUTPUT])
    assert line is not None and line.at == result.at
    assert pruned_record(paths.runs_root(repo), OLD, "f" * 64) is None


def test_applying_the_same_plan_twice_is_a_no_op_that_still_logs(repo: Roots) -> None:
    write_run(repo, OLD)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    first = apply(p)
    second = apply(p)
    assert first.unlinked > 0
    assert second.unlinked == 0 and second.bytes == 0
    lines = paths.gc_log_path(repo).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_a_dry_run_is_a_plan_without_an_apply(repo: Roots) -> None:
    """`--dry-run` prints the same rows; nothing here unlinks."""
    shas = write_run(repo, OLD)
    p = plan(paths.runs_root(repo), older_than=past_everything())
    assert p.shas > 0
    assert blobs_on_disk(repo, OLD) >= set(shas.values())
    assert not paths.gc_log_path(repo).exists()


# --- helpers -------------------------------------------------------------------------------


def _age(journal: pathlib.Path, *, days: int) -> None:
    """Rewrite every event's `ts` back by `days`. A test-only edit of a fixture journal — the
    code under test opens it read-only, and this is how "old" is arranged without a clock.
    Timestamps stay in the writer's own shape: UTC, second precision, a trailing `Z`
    (`clock.stamp`), or a zone-free local time for a journal written in older versions."""
    lines = journal.read_text(encoding="utf-8").splitlines()
    aged: list[str] = []
    for line in lines:
        ev = json.loads(line)
        then = dt.datetime.fromisoformat(ev["ts"]) - dt.timedelta(days=days)
        ev["ts"] = then.strftime("%Y-%m-%dT%H:%M:%S") + ("Z" if then.tzinfo else "")
        aged.append(json.dumps(ev, separators=(",", ":")))
    journal.write_text("\n".join(aged) + "\n", encoding="utf-8")
