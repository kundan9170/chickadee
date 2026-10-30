"""The recorded run a frontend's task views are built against, and how it is made.

The demo session's coder call (`demo_session.RUN_ID`) as its run journal holds it, in contracts
v2: Athena plans two tasks, the worker's first round of T1 fails its tests, her feedback
gets a second round that passes, T1 is accepted and committed, T2 ends its round at its limit
and she abandons it, and the run ends `PARTIAL`. Each line holds what the box writes at that
point, as `packages/core/tests/box/test_task_loop.py`'s episodes write it.

`build(dest)` writes it through core's own journal writer, which validates each payload against
its kind, with the clock held, so the bytes are the same on every machine and
`test_serve_fixture.py` can hold the committed copy to them: the recording is regenerated from
the code, never edited by hand, and a contract that moves fails there first.

    uv run python packages/serve/tests/recorded_run.py # rewrites tests/fixtures/run
"""

from __future__ import annotations

import contextlib
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from demo_session import RUN_ID

import chickadee.journal.writer as writer
from chickadee.ids import task_commit_id
from chickadee.journal.effects import EffectKind
from chickadee.journal.paths import LOCK_NAME, run_dir
from chickadee.journal.writer import Journal
from chickadee.schema import SCHEMA_VERSION
from chickadee.schema.journal import (
    CommandFactP,
    EffectCompletedP,
    EffectIntendedP,
    EffectStartedP,
    EventKind,
    ExecP,
    FileFactP,
    ModelCallP,
    PlanWrittenP,
    RunCapsP,
    RunEndP,
    RunStartP,
    TaskDelegatedP,
    TaskFactsP,
    TaskReportedP,
    TaskReviewedP,
    ToolCallP,
)
from chickadee.state.roots import Roots

FIXTURE: Final = Path(__file__).resolve().parent / "fixtures" / "run"
START: Final = datetime(2026, 10, 10, 10, 0, 20, tzinfo=UTC)
STEP: Final = timedelta(seconds=1)
CAP_USD: Final = 5.0

TREES: Final = {n: f"{n:x}" * 40 for n in range(1, 7)}
"""The worktree's tree after each step, by step: a recording names trees, never makes them."""
BASE: Final = "a0" * 20
"""The commit the run's worktree started from."""


@contextlib.contextmanager
def _held_clock() -> Iterator[None]:
    ticks = (START + n * STEP for n in range(10_000))
    # The writer's own name for the clock: the one place a line's `ts` comes from.
    held = writer.stamp  # pyright: ignore[reportPrivateImportUsage]
    writer.stamp = lambda: next(ticks).strftime("%Y-%m-%dT%H:%M:%SZ")  # pyright: ignore[reportPrivateImportUsage]
    try:
        yield
    finally:
        writer.stamp = held  # pyright: ignore[reportPrivateImportUsage]


class _RecordedRun:
    """The box's writes, one helper per kind of moment, each as the box makes it."""

    def __init__(self, journal: Journal) -> None:
        self.j = journal
        self.tokens = 0
        self.cost = 0.0

    def blob(self, text: str) -> str:
        return self.j.put_blob(text.encode())

    def athena(self, tools: tuple[str, ...], text: str | None = None) -> None:
        self._model("athena", "episode", "high-param", 2_400, 180, 0.0123, tools, text)

    def worker(self, task: str, round_: int, tools: tuple[str, ...], text: str | None) -> None:
        self._model("worker", "task", "low-param", 1_600, 90, 0.0004, tools, text, task, round_)

    def _model(
        self,
        role: str,
        request: str,
        rung: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        tools: tuple[str, ...],
        text: str | None,
        task: str | None = None,
        round_: int | None = None,
    ) -> None:
        self.tokens += tokens_in + tokens_out
        self.cost += cost
        self.j.append(
            EventKind.MODEL_CALL,
            ModelCallP(
                role=role,
                request=request,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                stop_reason="tool_use" if tools else "end_turn",
                tools=tools,
                text_preview=text,
                rung=rung,
            ),
            task_id=task,
            round=round_,
        )

    def tool(
        self,
        role: str,
        tool: str,
        args: str,
        result: str,
        task: str | None = None,
        round_: int | None = None,
    ) -> None:
        self.j.append(
            EventKind.TOOL_CALL,
            ToolCallP(
                role=role,
                tool=tool,
                args_sha=self.blob(args),
                outcome="served",
                result_tokens=len(result.split()),
                result_sha=self.blob(result),
                args_preview=args,
                result_preview=result,
            ),
            task_id=task,
            round=round_,
        )

    def step(
        self,
        op: str,
        task: str,
        round_: int,
        tool: str,
        before: str,
        after: str,
        changed: tuple[str, ...],
        ran: tuple[str, int, str] | None = None,
    ) -> None:
        """A worker's writing call: its effect's lifecycle, and its `exec` line
        between started and completed when it ran a command."""
        self.j.append(
            EventKind.EFFECT_INTENDED,
            EffectIntendedP(
                operation_id=op,
                effect_kind=EffectKind.WORKER_STEP.value,
                approved_sha=self.blob(f"{op} {tool}"),
                preconditions=(f"before={before}",),
                idempotent=False,
                declared_surface=(tool,),
                preimages_sha=before,
            ),
            task_id=task,
            round=round_,
        )
        self.j.append(
            EventKind.EFFECT_STARTED,
            EffectStartedP(operation_id=op),
            task_id=task,
            round=round_,
        )
        if ran is not None:
            label, exit_code, output = ran
            self.j.append(
                EventKind.EXEC,
                ExecP(
                    operation_id=op,
                    policy="worktree",
                    label=label,
                    argv_sha=self.blob(label),
                    cwd=f".git/chickadee/worktrees/{RUN_ID}",
                    exit_code=exit_code,
                    wall_ms=840,
                    timed_out=False,
                    sandbox="native",
                    network=False,
                    stdout_sha=self.blob(output),
                    stderr_sha="",
                    violations=(),
                ),
                task_id=task,
            )
        self.j.append(
            EventKind.EFFECT_COMPLETED,
            EffectCompletedP(
                operation_id=op,
                outcome="succeeded",
                observed_surface=changed,
                post_state_sha=after,
                result_sha=self.blob(f"{op} done"),
            ),
            task_id=task,
            round=round_,
        )

    def commit(self, task: str, files: tuple[str, ...], head: str, after: str) -> None:
        """An accepted task's commit, found by its trailer on resume."""
        op = task_commit_id(task)
        self.j.append(
            EventKind.EFFECT_INTENDED,
            EffectIntendedP(
                operation_id=op,
                effect_kind=EffectKind.TASK_COMMIT.value,
                approved_sha=self.blob(f"{op} {' '.join(files)}"),
                preconditions=(f"head={head}",),
                idempotent=False,
                declared_surface=files,
                preimages_sha=head,
            ),
            task_id=task,
        )
        self.j.append(EventKind.EFFECT_STARTED, EffectStartedP(operation_id=op), task_id=task)
        self.j.append(
            EventKind.EFFECT_COMPLETED,
            EffectCompletedP(
                operation_id=op,
                outcome="succeeded",
                observed_surface=files,
                post_state_sha=after,
            ),
            task_id=task,
        )


def _record(run: _RecordedRun) -> None:
    j = run.j
    plan = "# Plan\n\n1. T1 — raise the retry limit to 5\n2. T2 — lower the timeout\n"
    j.append(
        EventKind.RUN_START,
        RunStartP(
            goal="Raise the retry limit and lower the timeout",
            work_sha=run.blob("# Raise the retry limit and lower the timeout\n"),
            template_shas={"design": "d" * 64, "plan": "e" * 64},
            config_sha="c" * 64,
            repo_head=BASE,
            engine_version="0.0.0",
            schema_version=SCHEMA_VERSION,
            isolation="worktree",
            hard_cost_cap_usd=CAP_USD,
            caps=RunCapsP(total=CAP_USD),
        ),
    )
    run.athena(("plan",))
    j.append(
        EventKind.PLAN_WRITTEN,
        PlanWrittenP(call_id="a1", sha=run.blob(plan), by="athena", version=1, asked=True),
    )
    run.tool("athena", "plan", '{"text": "# Plan …"}', "approved at stop 2")
    run.athena(("delegate",))
    j.append(
        EventKind.TASK_DELEGATED,
        TaskDelegatedP(
            task_id="T1",
            call_id="a2",
            title="Raise the retry limit to 5",
            brief_sha=run.blob("Set max_retries to 5 in src/client.py and test it."),
            by="athena",
            start_tree=TREES[1],
            start_commit=BASE,
        ),
        task_id="T1",
    )
    # T1, round 1: the edit lands and its test fails.
    run.worker("T1", 1, ("edit",), None)
    run.step("step:T1:1:w1:0a0a0a0a", "T1", 1, "edit", TREES[1], TREES[2], ("src/client.py",))
    run.tool(
        "worker",
        "edit",
        '{"new": "max_retries = 5", "old": "max_retries = 3", "path": "src/client.py"}',
        "ok: src/client.py\nsyntax ok: src/client.py",
        "T1",
        1,
    )
    run.worker("T1", 1, ("shell",), None)
    failed = "FAILED tests/test_client.py::test_retries - assert 3 == 5\n1 failed"
    run.step(
        "step:T1:1:w2:0b0b0b0b",
        "T1",
        1,
        "shell",
        TREES[2],
        TREES[2],
        (),
        ("run the tests", 1, failed),
    )
    run.tool("worker", "shell", '{"command": "pytest -q"}', f"exit 1\n{failed}", "T1", 1)
    run.worker("T1", 1, (), "T1: max_retries is 5; tests/test_client.py still expects 3.")
    j.append(
        EventKind.TASK_REPORTED,
        TaskReportedP(
            task_id="T1",
            round=1,
            stopped="replied",
            report_sha=run.blob("max_retries is 5; the test still expects 3."),
            facts=TaskFactsP(
                start_tree=TREES[1],
                end_tree=TREES[2],
                files=(FileFactP(path="src/client.py", added=1, removed=1, kind="edited"),),
                commands=(
                    CommandFactP(
                        operation_id="step:T1:1:w2:0b0b0b0b",
                        command="pytest -q",
                        exit_code=1,
                        timed_out=False,
                        tail="1 failed",
                        test_outcome="failed",
                        failing=("tests/test_client.py::test_retries",),
                    ),
                ),
            ),
        ),
        task_id="T1",
    )
    run.tool("athena", "delegate", '{"title": "Raise the retry limit to 5"}', "T1 replied")
    run.athena(("review",))
    j.append(
        EventKind.TASK_REVIEWED,
        TaskReviewedP(
            task_id="T1",
            call_id="a3",
            answer="feedback",
            text_sha=run.blob("Update the test to expect 5."),
        ),
        task_id="T1",
    )
    # T1, round 2: the test follows, and passes.
    run.worker("T1", 2, ("edit",), None)
    run.step(
        "step:T1:2:w3:0c0c0c0c", "T1", 2, "edit", TREES[2], TREES[3], ("tests/test_client.py",)
    )
    run.tool(
        "worker",
        "edit",
        '{"new": "== 5", "old": "== 3", "path": "tests/test_client.py"}',
        "ok: tests/test_client.py\nsyntax ok: tests/test_client.py",
        "T1",
        2,
    )
    run.worker("T1", 2, ("shell",), None)
    run.step(
        "step:T1:2:w4:0d0d0d0d",
        "T1",
        2,
        "shell",
        TREES[3],
        TREES[3],
        (),
        ("run the tests", 0, "1 passed"),
    )
    run.tool("worker", "shell", '{"command": "pytest -q"}', "exit 0\n1 passed", "T1", 2)
    run.worker("T1", 2, (), "T1: the test expects 5 now; 1 passed.")
    j.append(
        EventKind.TASK_REPORTED,
        TaskReportedP(
            task_id="T1",
            round=2,
            stopped="replied",
            report_sha=run.blob("The test expects 5; 1 passed."),
            facts=TaskFactsP(
                start_tree=TREES[1],
                end_tree=TREES[3],
                files=(
                    FileFactP(path="src/client.py", added=1, removed=1, kind="edited"),
                    FileFactP(path="tests/test_client.py", added=1, removed=1, kind="edited"),
                ),
                tests_edited=("tests/test_client.py",),
                commands=(
                    CommandFactP(
                        operation_id="step:T1:2:w4:0d0d0d0d",
                        command="pytest -q",
                        exit_code=0,
                        timed_out=False,
                        tail="1 passed",
                        test_outcome="passed",
                    ),
                ),
            ),
        ),
        task_id="T1",
    )
    run.tool("athena", "review", '{"answer": "feedback"}', "T1 replied")
    run.athena(("review",))
    j.append(
        EventKind.TASK_REVIEWED,
        TaskReviewedP(task_id="T1", call_id="a4", answer="accept"),
        task_id="T1",
    )
    run.commit("T1", ("src/client.py", "tests/test_client.py"), BASE, "b1" * 20)
    run.tool("athena", "review", '{"answer": "accept"}', "T1 committed")
    run.athena(("delegate",))
    j.append(
        EventKind.TASK_DELEGATED,
        TaskDelegatedP(
            task_id="T2",
            call_id="a5",
            title="Lower the timeout",
            brief_sha=run.blob("Lower the request timeout in src/config.py."),
            by="athena",
            start_tree=TREES[3],
            start_commit="b1" * 20,
        ),
        task_id="T2",
    )
    # T2, round 1: an edit, then the round's limit.
    run.worker("T2", 1, ("edit",), None)
    run.step("step:T2:1:w5:0e0e0e0e", "T2", 1, "edit", TREES[3], TREES[4], ("src/config.py",))
    run.tool(
        "worker",
        "edit",
        '{"new": "timeout = 10", "old": "timeout = 30", "path": "src/config.py"}',
        "ok: src/config.py\nsyntax ok: src/config.py",
        "T2",
        1,
    )
    j.append(
        EventKind.TASK_REPORTED,
        TaskReportedP(
            task_id="T2",
            round=1,
            stopped="limit",
            facts=TaskFactsP(
                start_tree=TREES[3],
                end_tree=TREES[4],
                files=(FileFactP(path="src/config.py", added=1, removed=1, kind="edited"),),
            ),
        ),
        task_id="T2",
    )
    run.tool("athena", "delegate", '{"title": "Lower the timeout"}', "T2 stopped: limit")
    run.athena(("review",))
    j.append(
        EventKind.TASK_REVIEWED,
        TaskReviewedP(
            task_id="T2",
            call_id="a6",
            answer="abandon",
            text_sha=run.blob("Which timeout is meant is not settled; asking the person."),
        ),
        task_id="T2",
    )
    run.tool("athena", "review", '{"answer": "abandon"}', "T2 abandoned")
    run.athena((), "T1 is committed; T2 was abandoned: which timeout to lower is not settled.")
    j.append(
        EventKind.RUN_END,
        RunEndP(
            outcome="PARTIAL",
            why="T2 abandoned: which timeout to lower is not settled",
            tasks=2,
            committed=1,
            tokens=run.tokens,
            cost_usd=round(run.cost, 6),
            wall_ms=41_000,
        ),
    )


def build(dest: Path) -> Path:
    """Write the run's directory (its journal, blobs and config snapshot) at `dest` and return
    `dest`. Anything already there is replaced."""
    shutil.rmtree(dest, ignore_errors=True)
    scratch = dest.parent / f".{dest.name}-building"
    shutil.rmtree(scratch, ignore_errors=True)
    roots = Roots(project=scratch, repo_local=scratch, checkout_local=scratch, user=scratch)
    with _held_clock():
        journal = Journal.create(roots, RUN_ID, {"spend": {"hard_cost_usd": CAP_USD}})
        try:
            _record(_RecordedRun(journal))
        finally:
            journal.close()
    made = run_dir(roots, RUN_ID)
    (made / LOCK_NAME).unlink(missing_ok=True)
    shutil.copytree(made, dest)
    shutil.rmtree(scratch)
    return dest


if __name__ == "__main__":
    print(build(FIXTURE))
