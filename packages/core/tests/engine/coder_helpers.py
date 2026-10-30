"""An engine over a real, initialised repository, for the suites that delegate through
`PacketEngine.coder()`: the real box (the task loop, Athena and the worker
on their rungs), graph and workspace, with only the models scripted and the sandbox passed
through, as the g2 suite runs them.

One scripted rung answers every role, in the order the episode calls them. A one-task plan
(`one_task`) runs the worker's first round before Athena is called, so `solved`
scripts it in that order: the worker's edit and its reply, then Athena's accept and her report.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import pytest
from engines import answer, call, calls, with_prices
from repos import git, init_repo
from zeus_turn_helpers import ScriptedRung, Step, config_for, with_sessions

import chickadee.delegation as delegation_module
import chickadee.engine as engine_module
from chickadee.engine import ApprovedDesign, CoderRequest, OneTask, PacketEngine
from chickadee.isolation.none import Passthrough
from chickadee.schema import ToolCall, sha256_hex
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    InteractionPort,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee.state.roots import roots as resolve_roots

SID = "S-coder"
APP = "def go() -> int:\n    return 1\n"
REPORT = "T1 accepted: go() returns 2."


def make_repo(root: Path) -> Path:
    """A committed Python repository, initialised (its graph built), with session `SID`."""
    repo = init_repo(
        root,
        {"app.py": APP, "tests/test_app.py": "def test_go():\n    assert True\n"},
    )
    engine_module.init_repo(repo)
    return with_sessions(repo, SID)


def make_ignoring_repo(root: Path) -> Path:
    """`make_repo`, ignoring what running its checks leaves behind, as real repositories do: an
    unignored `__pycache__` would be the harness's to commit, undo and land."""
    repo = make_repo(root)
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore caches")
    return repo


def worktree_count(repo: Path) -> int:
    """The repository's worktrees, its main one included, as git lists them."""
    return git(repo, "worktree", "list", "--porcelain").count("worktree ")


class CoderModel(ScriptedRung):
    """`ScriptedRung`, whose script a test may extend between runs: the episodes of one engine
    call the same rung, so a later run's steps follow the earlier one's."""

    def then(self, *steps: Step) -> None:
        self._script.extend(steps)


def build(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    script: Sequence[Step] = (),
    *,
    interaction: InteractionPort | None = None,
) -> tuple[PacketEngine, CoderModel]:
    """The engine, with the scripted rung as every model (Zeus's and the box roles', which the
    episode makes in `delegation`) and no sandbox, and every rung a role runs on priced, so a
    session can open and an episode can charge. Its trust is the loader's, as a host constructs
    it."""
    model = CoderModel(script)
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: model)
    monkeypatch.setattr(delegation_module, "client_for", lambda _endpoint: model)
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    # Kept beside what the test set: a test that lets the shell run unsandboxed says so.
    raw["sandbox"] = {**raw.get("sandbox", {}), "require_sandbox": False}
    priced(raw)
    config = config_for(raw, priced=False)
    engine = PacketEngine(
        repo,
        config.engine,
        interaction,
        trust=config.trust(resolve_roots(repo).repo_local),
    )
    return engine, model


def priced(raw: dict[str, Any]) -> dict[str, Any]:
    """`engines.with_prices`, and the worker's rung too: an unpriced charge is refused,
    and the worker runs on `low-param`."""
    with_prices(raw)
    raw["prices"][raw["rungs"]["low-param"]] = {"price_in": 0.5, "price_out": 1.0}
    return raw


def one_task(
    title: str = "Return 2",
    brief: str = "## Goal\n`go()` returns 2.\n",
    *,
    workspace: Literal["worktree", "in_place"] = "worktree",
) -> CoderRequest:
    """A one-task plan as approved at stop 2: what a host hands `coder()`."""
    return CoderRequest(
        task=OneTask(title=title, brief=brief, sha=sha256_hex(brief), template_sha="0" * 64),
        workspace=workspace,
    )


DESIGN = "# Return 2\n\n## What changes\n- `go()` returns 2.\n"
PLAN = "# Plan: return 2\n\n1. **Return 2.** Edit `app.py`.\n\nFinal check: none.\n"


def designed(
    text: str = DESIGN, *, workspace: Literal["worktree", "in_place"] = "worktree"
) -> CoderRequest:
    """A design as approved at stop 1: Athena plans it, asks stop 2, and delegates its tasks."""
    return CoderRequest(
        design=ApprovedDesign(text=text, sha=sha256_hex(text), template_sha="0" * 64),
        workspace=workspace,
    )


def athena_delegates(
    title: str = "Return 2", *, n: int = 1, plan: str | None = PLAN
) -> list[Step]:
    """Athena's plan (stop 2, approved unattended), then her delegation of one task."""
    steps: list[Step] = []
    if plan is not None:
        steps.append(calls(box_call("plan", f"p{n}", text=plan)))
    steps.append(calls(box_call("delegate", f"d{n}", title=title, brief=f"## Task\n{title}\n")))
    return steps


def box_call(tool: str, cid: str, /, **args: object) -> ToolCall:
    """A call as a box role sends it: its arguments, no purpose line."""
    return ToolCall(id=cid, name=tool, arguments=dict(args))


def worker_edit(old: str, new: str, file: str = "app.py", *, n: int = 1) -> list[Step]:
    """The worker's round of a one-task plan: one edit, then its reply."""
    return [
        calls(box_call("edit", f"w{n}", path=file, old=old, new=new)),
        answer(f"T{n}: edited {file}."),
    ]


def athena_accepts(task: str = "T1", *, n: int = 1, report: str = REPORT) -> list[Step]:
    """Athena's review of `task`, accepted, then her report: the episode's last answer."""
    return [calls(box_call("review", f"a{n}", task=task, answer="accept")), answer(report)]


def solved(old: str, new: str, file: str = "app.py") -> list[Step]:
    """A one-task episode that edits `file` and is accepted: committed, `COMPLETED`."""
    return [*worker_edit(old, new, file), *athena_accepts()]


def zeus_delegates(
    old: str = "return 1",
    new: str = "return 2",
    *,
    cid: str = "c1",
    workspace: Literal["worktree", "in_place"] = "worktree",
) -> list[Step]:
    """Zeus's `coder` call, a one-task plan (approved at stop 2 unattended), and the episode it
    runs, `solved`: what a turn that changes `old` to `new` in `app.py` scripts before Zeus's
    reply."""
    work = {
        "title": "Return 2",
        "brief": "## Goal\n`go()` returns 2.\n",
        "workspace": workspace,
    }
    return [calls(call("coder", cid, **work)), *solved(old, new)]


class ProcessDied(BaseException):
    """The process died: a `BaseException`, so nothing on the way out records it."""


# -- a design in its work folder ---------------------------------------

DESIGN_README = """# {title}

## Goal
{goal}

## What changes
- `app.py`: {goal}.

## What must not change
- `go()`'s signature.

## Behaviour
- {goal}.

## Outside the repository
None.

## Files
None.
"""
"""A `design/README.md` with the design template's minimum sections filled; the harness adds
Request and Constraints at `coder(work)`."""

PLAN_README = """# Plan: {title}

## Tasks
1. [T1: {title}](tasks/T1-task.md): the edit, with its test. Depends on: none

## Final check
The full suite.
"""

TASK_FILE = """# T1: {title}

## Goal
{goal}

## Where to look
- `app.py`: `go()`.

## Approach
Edit the return.

## Don't
- Edit a test to pass.

## Done when
`go()` returns what the goal says.

## Report
What changed.
"""


def work_name(title: str) -> str:
    """The folder Zeus's first `doc` of `title` opens today: `<date>-<slug>`."""
    from datetime import date

    from chickadee.workspace import slug

    return f"{date.today().isoformat()}-{slug(title, 'work')}"


def zeus_designs(
    title: str = "Return 2", goal: str = "`go()` returns 2", *, cid: str = "c1"
) -> list[Step]:
    """Zeus writes `design/README.md` with `doc`, opening the work, and delegates it by name."""
    readme = DESIGN_README.format(title=title, goal=goal)
    doc = call("doc", f"{cid}-doc", op="write", file="README.md", content=readme, title=title)
    return [calls(doc), calls(call("coder", cid, work=work_name(title)))]


def athena_plans_folder(
    title: str = "Return 2", goal: str = "`go()` returns 2", *, n: int = 1
) -> list[Step]:
    """Athena writes `plan/` with `doc` (its README and T1's file), calls `plan()` (stop 2,
    approved unattended), then delegates T1 by its id."""
    return [
        calls(
            box_call(
                "doc",
                f"pd{n}",
                op="write",
                file="README.md",
                content=PLAN_README.format(title=title),
            )
        ),
        calls(
            box_call(
                "doc",
                f"pt{n}",
                op="write",
                file="tasks/T1-task.md",
                content=TASK_FILE.format(title=title, goal=goal),
            )
        ),
        calls(box_call("plan", f"p{n}")),
        calls(box_call("delegate", f"d{n}", task="T1")),
    ]


class Approving:
    """A person who approves every confirmation and takes every question's default."""

    def __init__(self) -> None:
        self.shown: list[ActionApproval] = []

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        return Answer(question_id=q.id, choice=q.default, source=AnswerSource.DEFAULT)

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self.shown.append(approval)
        return ApprovalDecision(action_id=approval.action_id, verdict="approve", source="human")


class LeavesAsked(Approving):
    """Asked about a command, the person leaves without answering: what leaves a run suspended
    at a question, for a test that continues it."""

    def __init__(self) -> None:
        super().__init__()
        self.asked = threading.Event()

    def confirm_action(self, approval: Any, *, cancel: threading.Event | None = None) -> Any:
        self.asked.set()
        assert cancel is not None
        cancel.wait(30)
        from chickadee.schema.interaction import ApprovalDecision

        return ApprovalDecision(action_id=approval.action_id, verdict="deny", source="default")
