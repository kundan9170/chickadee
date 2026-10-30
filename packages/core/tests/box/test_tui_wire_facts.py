"""The journal facts D234 adds for the TUI (T-5.23 Part 8): a plan's tasks, the model that
served each box call, and an approval's command as fields.

Each is read from what the harness already computes: `work.plan_tasks`, the rung's catalog
model, the shell claim. A line written before the field reads as it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from box_helpers import APP, FakeRunJournal, episode_rig
from repos import init_repo

from chickadee.box.sink import BoxSink
from chickadee.journal.port import JournalPort
from chickadee.schema.adapter import Completion, StopReason, Usage
from chickadee.schema.authorization import Tier
from chickadee.schema.journal import ModelCallP, PlanTaskP, PlanWrittenP
from chickadee.shell import ShellClaim
from chickadee.zeus.tools.talk import approval_of

LISTED = """# Plan: a retry

## Tasks
1. [T1: Retry the call](tasks/T1-retry.md): the loop, with tests. Depends on: none
2. [T2: Log each retry](tasks/T2-log.md): one line per attempt. Depends on: T1

## Final check
The full suite.
"""


def _plans(root: Path, text: str) -> list[PlanWrittenP]:
    rig = episode_rig(root, [], [])  # type: ignore[arg-type]
    assert rig.desk.plan(text, call_id="a-plan").startswith("the person approved the plan")
    return [p.payload for p in rig.journal.projection().lines(PlanWrittenP)]


def test_a_plan_records_the_tasks_its_text_lists(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"app.py": APP})
    [plan] = _plans(root, LISTED)
    assert plan.tasks == (
        PlanTaskP(id="T1", title="Retry the call"),
        PlanTaskP(id="T2", title="Log each retry"),
    )


def test_a_plan_whose_text_lists_no_tasks_records_none(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"app.py": APP})
    [plan] = _plans(root, "# Plan: go returns 2\n\n1. **Return 2.** Change `go()`.\n")
    assert plan.tasks == ()


def test_a_plan_line_written_before_it_reads_with_no_tasks() -> None:
    old = PlanWrittenP.model_validate(
        {"sha": "0" * 64, "by": "athena", "version": 1, "asked": True}
    )
    assert old.tasks == ()


def test_each_box_call_names_the_catalog_model_that_served_it() -> None:
    journal = FakeRunJournal()
    sink = BoxSink("worker", cast(JournalPort, journal), "low-param", "small-coder")
    done = Completion(
        text="ok",
        usage=Usage(prompt_tokens=10, completion_tokens=2),
        latency_ms=1,
        stop_reason=StopReason.END_TURN,
    )
    sink.model_call("task", done, 0.001, None)
    ((_, line, _, _),) = journal.lines
    assert isinstance(line, ModelCallP)
    assert (line.rung, line.model) == ("low-param", "small-coder")


def test_an_approval_carries_its_command_and_where_it_runs_and_says_it_leaves_the_sandbox() -> (
    None
):
    claim = ShellClaim(
        "pip install pygame",
        "src",
        Tier.T1,
        "lockfile_install",
        True,
        operation_id="op-1",
        idempotent=False,
    )
    asked = approval_of(claim)
    assert (asked.command, asked.cwd, asked.sandboxed) == ("pip install pygame", "src", False)
    # `display` still composes them for a person; the fields are what a frontend reads.
    assert asked.display.startswith("pip install pygame  (in src · outside the sandbox")
