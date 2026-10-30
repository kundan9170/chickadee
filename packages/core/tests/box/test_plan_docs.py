"""Athena's `doc` and the work folder's fence.

Her `doc` writes the open work's `plan/` through an edit port of its own, as a recorded step,
never gated by the held task as her code edits are (unchanged); a path that leaves
`plan/` is refused, and with no work open she has no folder. Her edit tools, like the
worker's, refuse the work folder, each naming why.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import APP, athena_call, episode_rig
from engines import answer, calls
from repos import init_repo

from chickadee.box.edits import (
    ATHENA_FENCE,
    WORKER_FENCE,
    WorkFence,
    WorkRule,
    WorktreeEdits,
)
from chickadee.box.syntax import SyntaxChecker
from chickadee.boxctx.athena import athena_head
from chickadee.context.work_line import WorkLine
from chickadee.edits.match import EMPTY_OLD_DO
from chickadee.journal.effects import EffectKind
from chickadee.schema.journal import BoxMessageP
from chickadee.secrets.denylist import Denylist

WORK = WorkLine(name="2026-10-18-retry", dir="docs/chickadee", state="open")
PLAN = "docs/chickadee/2026-10-18-retry/plan"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _run(root: Path, *script: object, work: WorkLine | None = WORK) -> object:
    rig = episode_rig(root, [*script, answer("done")], [], work=work)
    rig.athena.start(athena_head("You are Athena.", "P", "B", design="# Design"))
    return rig


def _answers(rig: object) -> list[str]:
    journal = rig.journal  # type: ignore[attr-defined]
    return [
        journal.get_blob(p.payload.sha).decode()
        for p in journal.projection().lines(BoxMessageP)
        if p.payload.side == "tool"
    ]


def test_her_doc_writes_plan_with_no_task_held(root: Path) -> None:
    rig = _run(
        root,
        calls(athena_call("doc", "a1", op="write", file="README.md", content="# Plan\n")),
        calls(athena_call("doc", "a2", op="write", file="tasks/T1-retry.md", content="# T1\n")),
    )
    assert (root / PLAN / "README.md").read_text() == "# Plan\n"
    assert (root / PLAN / "tasks" / "T1-retry.md").read_text() == "# T1\n"
    steps = [
        op.intended
        for op in rig.journal.projection().operations.values()  # type: ignore[attr-defined]
        if op.kind == EffectKind.WORKER_STEP.value and op.intended is not None
    ]
    assert [s.declared_surface for s in steps] == [("doc",), ("doc",)], "each one recorded step"


@pytest.mark.parametrize("file", ["../design/README.md", "/tmp/x.md", "../../../app.py"])
def test_her_doc_cannot_leave_plan(root: Path, file: str) -> None:
    rig = _run(root, calls(athena_call("doc", "a1", op="write", file=file, content="x\n")))
    [shown] = _answers(rig)
    assert "DENIED" in shown
    assert (root / "app.py").read_text() == APP
    assert not (root / "docs").exists()


def test_with_no_work_open_she_has_no_folder(root: Path) -> None:
    rig = _run(
        root,
        calls(athena_call("doc", "a1", op="write", file="README.md", content="# Plan\n")),
        work=None,
    )
    [shown] = _answers(rig)
    assert "DENIED" in shown and "no work folder is open" in shown


def test_her_title_is_left_unread_and_the_plan_is_written(root: Path) -> None:
    """The work is open and Zeus named it, so a title from her names nothing, and is left
    unread as his is on a later call. It was refused once, with an instruction never to write
    the path again: her README was never written, and `plan` failed for it (2026-10-23)."""
    rig = _run(
        root,
        calls(athena_call("doc", "a1", op="write", file="README.md", content="#\n", title="x")),
    )
    [shown] = _answers(rig)
    assert "plan/README.md written" in shown
    assert "DENIED" not in shown


def test_an_empty_old_is_told_how_to_add_text(root: Path) -> None:
    """She sent `edit` with an empty `old` to add a section, nine times, and was told to add
    the lines around it; then she rewrote every file whole (2026-10-23)."""
    rig = _run(
        root,
        calls(athena_call("doc", "a1", op="write", file="README.md", content="# Plan\n")),
        calls(athena_call("doc", "a2", op="edit", file="README.md", old="", new="## Report\n")),
    )
    shown = _answers(rig)[-1]
    assert "AMBIGUOUS" in shown
    assert EMPTY_OLD_DO in shown


@pytest.mark.parametrize(
    ("fence", "said"), [(ATHENA_FENCE, "doc"), (WORKER_FENCE, "read-only")]
)
def test_an_edit_port_with_the_fence_refuses_the_work_folder(
    root: Path, fence: WorkRule, said: str
) -> None:
    """Athena's and the worker's edit tools, each told why; `doc`'s own port
    has no fence. A path beside the folder that only shares its prefix is not in it."""
    port = WorktreeEdits(
        root,
        Denylist(),
        SyntaxChecker(root, ()),
        untracked_max_bytes=1 << 20,
        work=WorkFence("docs/chickadee", fence),
    )
    refused = port.refusal(f"{PLAN}/README.md")
    assert refused is not None
    why = refused.render()
    assert "is in the work folder docs/chickadee" in why and said in why
    assert port.refusal("docs/chickadee-notes.md") is None
    assert port.refusal("app.py") is None


@pytest.mark.writes_worktree("docs", "docs/*", "notes")
def test_a_link_into_the_work_folder_is_the_work_folder(root: Path) -> None:
    """The review of #607: the fence judges where a path lands, so a link made outside
    `work.dir` never writes the read-only plan through it."""
    (root / PLAN).mkdir(parents=True)
    (root / "notes").symlink_to(root / PLAN, target_is_directory=True)
    port = WorktreeEdits(
        root,
        Denylist(),
        SyntaxChecker(root, ()),
        untracked_max_bytes=1 << 20,
        work=WorkFence("docs/chickadee", WORKER_FENCE),
    )
    refused = port.refusal("notes/tasks/T2-x.md")
    assert refused is not None and "is in the work folder docs/chickadee" in refused.reason


def test_her_doc_writes_markdown_only(root: Path) -> None:
    """Pinned in older versions, when `doc` wrote `plan/pose.py` as it writes Markdown.
    Moved with D214 decision 1: a name that is not `.md` is refused, teaching why, and nothing
    is written; a `.md` one is written."""
    rig = _run(
        root,
        calls(athena_call("doc", "a1", op="write", file="pose.py", content="x = 1\n")),
        calls(athena_call("doc", "a2", op="write", file="pose.md", content="# Pose\n")),
    )
    refused, written = _answers(rig)[:2]
    assert (
        "DENIED" in refused
        and "pose.py is not a .md file. Do: name a .md file: design files are Markdown"
        in (refused)
    )
    assert "the worker writes the code" in refused
    assert not (root / PLAN / "pose.py").exists()
    assert (root / PLAN / "pose.md").read_text() == "# Pose\n"
    assert "DENIED" not in written
