"""An episode's hooks are bound when its run starts (`box/episode.BoxRun`).

What these pin: the bound set is written once, as a `hook` line of kind `bound` right after
`run_start`, which stays the run's first line; `run_start.config_sha` moves with the set; and a
resumed run binds from its line, not today's config, so it runs the hooks it started with.
"""

from __future__ import annotations

from pathlib import Path

from box_helpers import BoxRig, design, with_hooks, worker_call
from box_helpers import athena_call as acall
from engines import answer, calls

from chickadee.box.episode import Paused
from chickadee.box.report import CoderReport
from chickadee.durable.blobs import BlobStore
from chickadee.hooks.bind import BOUND, specs_of
from chickadee.hooks.points import HookPoint
from chickadee.hooks.spec import HookSpec
from chickadee.journal import paths
from chickadee.journal.projection import RunProjection, project
from chickadee.journal.reader import read_events
from chickadee.schema.journal import EventKind, HookP, RunStartP
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import Workspace

PLAN = "1. **Return 2.** `go()` in `app.py` returns 2.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
OFF = HookSpec(event=HookPoint.BEFORE_STOP, name="owed_review", enabled=False)


def _athena() -> list[object]:
    return [
        calls(acall("plan", "a1", text=PLAN)),
        calls(acall("delegate", "a2", title="Return 2", brief=BRIEF)),
        calls(acall("review", "a3", task="T1", answer="accept")),
        answer("One task: go() now returns 2."),
    ]


WORKER = [
    calls(worker_call("edit", "w1", path="app.py", old="return 1", new="return 2")),
    answer("T1: go() returns 2."),
]


def _journal(ws: Workspace, run_id: str) -> Path:
    return paths.journal_path(resolve_roots(ws.root), run_id)


def _run(ws: Workspace, run_id: str) -> RunProjection:
    return project(read_events(_journal(ws, run_id)))


def _bound_lines(ws: Workspace, run_id: str) -> list[HookP]:
    return [p.payload for p in _run(ws, run_id).lines(HookP) if p.payload.event == BOUND]


def test_the_set_is_one_line_right_after_run_start(rig: BoxRig) -> None:
    rig.athena_script, rig.worker_script = _athena(), WORKER
    report = rig.box().open(design()).report()
    assert isinstance(report, CoderReport) and report.run_id is not None

    events = list(read_events(_journal(rig.workspaces[0], report.run_id)))
    assert events[0].kind is EventKind.RUN_START
    assert events[1].kind is EventKind.HOOK and isinstance(events[1].payload, HookP)
    assert events[1].payload.event == BOUND
    assert len(_bound_lines(rig.workspaces[0], report.run_id)) == 1


def test_config_sha_moves_with_the_set(rig: BoxRig) -> None:
    rig.athena_script, rig.worker_script = _athena(), WORKER
    plain = rig.box().open(design()).report()
    assert isinstance(plain, CoderReport) and plain.run_id is not None

    with_hooks(rig, OFF)
    rig.athena_script, rig.worker_script = _athena(), WORKER
    off = rig.box().open(design()).report()
    assert isinstance(off, CoderReport) and off.run_id is not None

    first = _run(rig.workspaces[0], plain.run_id).first(RunStartP)
    second = _run(rig.workspaces[1], off.run_id).first(RunStartP)
    assert first is not None and second is not None
    assert first.config_sha != second.config_sha


def test_a_resumed_run_binds_from_its_line_not_todays_config(rig: BoxRig) -> None:
    rig.athena_script, rig.worker_script = _athena(), WORKER
    rig.cancel.set()
    paused = rig.box().open(design()).report()
    assert isinstance(paused, Paused)

    with_hooks(rig, OFF)
    rig.cancel.clear()
    report = rig.box().resume(paused.run_id).report()
    assert isinstance(report, CoderReport)
    assert report.outcome == "COMPLETED", report.why

    ws = rig.workspaces[-1]
    assert len(_bound_lines(ws, paused.run_id)) == 1, "written once, never again on resume"
    blobs = BlobStore(paths.blobs_dir(resolve_roots(ws.root), paused.run_id))
    stored = specs_of(_run(ws, paused.run_id).lines(HookP), blobs.read)
    assert stored is not None and stored[0] == (), "it bound the defaults, as it started"
