"""The E1 Chickadee arm: the complete system, through Zeus.

Per problem the arm does what a person would: opens a session on the problem's directory,
says what to build, and lets Zeus choose the route — no delegation, the exact path, a one-task
plan or a design for Athena. Nothing in the engine knows it is being measured.

**Unattended, like a person who said yes to everything safe** (`NonInteractive`): every
question takes its declared default, a stop is approved, and an action above T0 is denied.

**The work comes back before it is scored.** Delegated work runs in the session's leased
worktree, so after each turn the arm hands it off with `apply`: the run's commits replay
onto the problem's branch, the tests see them, and the next turn's worktree starts from them.

**Retry parity is a person's turn**. A failing run's test output, redacted by
the naive control's own `redact_oracle`, is the next thing the person says, up to `max_turns`.
The tests are on disk only while `score` runs them: this arm's model can read
files, so a test file left between turns would be a route to the oracle no prompt scan sees.
The turns are the arm's `prompts`, the only path from the oracle to a model, and `run_problem`
scans every one with `leaks()`.

**What a problem records** (`ChickadeeStats`): turns, spend from the session's budget, and
from its journals the route, tasks, rounds and how they ended, and the refused calls. A run
that changes the harness is judged by these as much as by the solve rate.

**What a problem keeps** (`keep`), because the workspace is deleted when the problem ends and
a failure is read afterwards: each turn's test output, redacted as the retry turn is, so the
archive never holds the oracle; the diff from the problem as given; and the session's and
runs' journals, to see what each role did.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chickadee.engine import ChickadeeError, PacketEngine, load_host_config
from chickadee_eval.arms.e1_naive import redact_oracle
from chickadee_eval.problems.polyglot import LoadedProblem, PromptView
from chickadee_eval.runner import Attempt

__all__ = [
    "ChickadeeArm",
    "ChickadeeStats",
    "commit_all",
    "first_turn",
    "open_host_engine",
    "retry_turn",
]

ARM_NAME = "e1-chickadee"

FIRST_TURN = """\
{instructions}

####

Use the above instructions to modify the supplied files: {stub}
Don't change the names of existing functions or classes, as they may be referenced from other
code like unit tests, etc. Only use standard libraries, don't suggest installing any packages.
"""
"""Aider's Polyglot prompt (`benchmark.py`'s `instructions_addendum`), so a solve rate here is
read against published Polyglot numbers rather than against a prompt of our own."""

RETRY_TURN = """\
{output}

####

See the testing errors above. The tests are correct, don't try and change them.
Fix the code in {stub} to resolve the errors.
"""
"""Aider's retry (`test_failures`): the redacted test output, then the instruction."""

TESTS_NAME = "tests.txt"
DIFF_NAME = "solution.diff"
JOURNALS_NAME = "journals"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "chickadee-eval",
    "GIT_AUTHOR_EMAIL": "eval@chickadee.invalid",
    "GIT_COMMITTER_NAME": "chickadee-eval",
    "GIT_COMMITTER_EMAIL": "eval@chickadee.invalid",
}


def first_turn(view: PromptView) -> str:
    return FIRST_TURN.format(instructions=view.instructions.strip(), stub=view.stub_path)


def retry_turn(output: str, view: PromptView) -> str:
    return RETRY_TURN.format(output=output.strip(), stub=view.stub_path)


def open_host_engine(workspace: Path) -> PacketEngine:
    """The engine a host builds for `workspace`, with the person's own config and trust: what
    `chickadee serve` builds for a repository (`load_host_config`)."""
    host = load_host_config(workspace)
    return PacketEngine(host.repo, host.engine, trust=host.trust)


@dataclass
class ChickadeeStats:
    """One problem's record beyond pass and fail: what the system did on the way."""

    problem_id: str
    turns: int = 0
    passed: bool = False
    spent_usd: float = 0.0
    route: str = "none"
    """`none` (Zeus delegated nothing), or each run's path: `exact`, `planned`."""
    runs: int = 0
    tasks: int = 0
    rounds: int = 0
    round_ends: dict[str, int] = field(default_factory=dict[str, int])
    """How each worker round ended (`replied`, `limit`, `looped`, `harness_error`,...)."""
    reviews: dict[str, int] = field(default_factory=dict[str, int])
    """Athena's answers (`accept`, `feedback`, `take_over`, `re_cut`, `drop`)."""
    refused_calls: int = 0
    """Tool calls refused, every role's: the shape of trouble a harness change moves."""
    model_calls: int = 0
    not_applied: list[str] = field(default_factory=list[str])
    """Turns whose handoff did not apply, and why: a session with no worktree, or a refusal."""
    error: str = ""
    """What stopped the attempt early, when something did."""

    def line(self) -> str:
        """One line for `ProblemRecord.detail`: the counts a reader scans a table for."""
        ends = ",".join(f"{k}={v}" for k, v in sorted(self.round_ends.items())) or "-"
        parts = [
            f"turns={self.turns}",
            f"route={self.route}",
            f"tasks={self.tasks}",
            f"rounds={self.rounds}",
            f"ends={ends}",
            f"refused={self.refused_calls}",
        ]
        if self.error:
            parts.append(f"error={self.error}")
        return " ".join(parts)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChickadeeArm:
    """The complete system as an `ArmDriver`."""

    open_engine: Callable[[Path], PacketEngine]
    """The engine for a workspace: `open_host_engine` in a real run, a scripted one in tests."""
    oracles: Mapping[str, LoadedProblem]
    """Every problem by id, for `score` and for redacting its failure: the arm never renders
    anything from these, as the naive control never does."""
    score: Callable[[LoadedProblem, Path], tuple[bool, str]]
    """Runs the hidden tests against the workspace and takes them away again
    (`runner.score_hidden`): `(passed, output)`."""
    max_turns: int = 3
    """The naive control's `max_attempts`, so the two retry alike."""
    turn_timeout_s: float | None = 3600.0
    keep: Path | None = None
    """Where each problem's files are kept, under its id (`TESTS_NAME`, `DIFF_NAME`,
    `JOURNALS_NAME`); None keeps nothing."""
    name: str = ARM_NAME
    stats: list[ChickadeeStats] = field(default_factory=list[ChickadeeStats])
    """Every problem's record, in order, for the command to write beside the results."""

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")

    def attempt(self, problem: PromptView, workspace: Path) -> Attempt:
        oracle = self.oracles.get(problem.id)
        if oracle is None:
            raise KeyError(
                f"{problem.id!r} has no oracle registered with this arm; without it the "
                "retry turn cannot be redacted (D7, §5.1)"
            )
        stats = ChickadeeStats(problem.id)
        self.stats.append(stats)
        prompts: list[str] = []
        outputs: list[str] = []
        base = commit_all(workspace, "the problem as given")
        engine = self.open_engine(workspace)
        try:
            engine.init_repo()
            sid = engine.create_session(pinned=True).session_id
            text = first_turn(problem)
            for n in range(1, self.max_turns + 1):
                prompts.append(text)
                stats.turns = n
                engine.turn(sid, text).wait(self.turn_timeout_s)
                _apply(engine, sid, n, stats)
                stats.passed, output = self.score(oracle, workspace)
                redacted, _removed = redact_oracle(output, oracle)
                verdict = "pass" if stats.passed else "fail"
                outputs.append(f"== turn {n}: {verdict} ==\n{redacted.strip()}\n")
                if stats.passed:
                    break
                text = retry_turn(redacted, problem)
            stats.spent_usd = engine.budget(sid).spent_usd
            _read_journals(engine, sid, stats)
        except (ChickadeeError, TimeoutError, OSError) as exc:
            # An attempt the harness could not finish is still a row: it is scored by the
            # tests the runner runs next, and says why it stopped.
            stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            engine.close()
            if self.keep is not None:
                _keep(self.keep / problem.id, workspace, base, outputs, stats)
        return Attempt(
            solved_claim=stats.passed,
            cost_usd=stats.spent_usd,
            attempts=stats.turns,
            prompts=tuple(prompts),
            detail=stats.line(),
        )


def _git(workspace: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=workspace,
        env={**os.environ, **_GIT_ENV},
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


def commit_all(workspace: Path, message: str) -> str:
    """`workspace` as a git repository with everything in one commit: what the engine needs to
    lease, branch a worktree from, and hand back to. Returns the commit."""
    if not (workspace / ".git").exists():
        _git(workspace, "init", "-q", "-b", "main")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "--allow-empty", "-m", message)
    return _git(workspace, "rev-parse", "HEAD").strip()


def _keep(
    dest: Path, workspace: Path, base: str, outputs: list[str], stats: ChickadeeStats
) -> None:
    """The problem's files, kept before its workspace is deleted. A file that cannot be kept
    is said in `stats.error` and does not end the run: the score does not depend on it."""
    try:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / TESTS_NAME).write_text("\n".join(outputs), encoding="utf-8")
        # Intent-to-add, so a file the work created is in the diff; the engine's own
        # `.chickadee/` is not the solution.
        _git(workspace, "add", "-A", "-N")
        diff = _git(workspace, "diff", "--no-color", base, "--", ".", ":(exclude).chickadee")
        (dest / DIFF_NAME).write_text(diff, encoding="utf-8")
        state = workspace / ".git" / "chickadee"
        for part in ("sessions", "runs"):
            if (state / part).is_dir():
                shutil.copytree(
                    state / part,
                    dest / JOURNALS_NAME / part,
                    ignore=shutil.ignore_patterns("LOCK"),
                    dirs_exist_ok=True,
                )
    except (OSError, subprocess.CalledProcessError) as exc:
        stats.error = (stats.error + "; " if stats.error else "") + f"not kept: {exc}"


def _apply(engine: PacketEngine, sid: str, turn: int, stats: ChickadeeStats) -> None:
    """The turn's work onto the problem's branch."""
    try:
        engine.finalize(sid, "apply")
    except ChickadeeError as exc:
        # No worktree (Zeus delegated nothing, or worked in place) is the common case, and
        # nothing is lost: said, not raised.
        stats.not_applied.append(f"turn {turn}: {exc}")


def _records(engine: PacketEngine, scope: str) -> Iterator[Any]:
    """Every record of a session or a run, page by page."""
    page = engine.journal(scope)
    while page.records:
        yield from page.records
        page = engine.journal(scope, page.cursor)


def _read_journals(engine: PacketEngine, sid: str, stats: ChickadeeStats) -> None:
    """The route and the counts, from the session's log and each run it opened."""
    paths: list[str] = []
    runs: list[str] = []
    for record in _records(engine, sid):
        if record.kind == "episode":
            runs.append(str(record.payload.get("run_id")))
            paths.append(str(record.payload.get("path")))
        elif record.kind == "tool" and record.payload.get("ran") is False:
            stats.refused_calls += 1
    stats.runs = len(runs)
    stats.route = ",".join(paths) if paths else "none"
    for run in runs:
        try:
            records = list(_records(engine, run))
        except ChickadeeError:
            continue  # stopped before it started: its episode line already says so
        for record in records:
            payload = record.payload
            if record.kind == "task_delegated":
                stats.tasks += 1
            elif record.kind == "task_reported":
                stats.rounds += 1
                end = str(payload.get("stopped"))
                stats.round_ends[end] = stats.round_ends.get(end, 0) + 1
            elif record.kind == "task_reviewed":
                answer = str(payload.get("answer"))
                stats.reviews[answer] = stats.reviews.get(answer, 0) + 1
            elif record.kind == "tool_call" and payload.get("denied_reason"):
                stats.refused_calls += 1
            elif record.kind == "model_call":
                stats.model_calls += 1
