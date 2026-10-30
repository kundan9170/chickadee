"""A test run read by what it printed (D236 part 1 and Amendment 1).

What these pin: each shipped runner's output, captured from the real runner
(`fixtures/runs/`), reads to its outcome, counts and test names; printed failures beat exit 0;
every summary in one output is summed; a summary that counts nothing is `not_collected`; a test
command whose exit code is another command's, with no summary shown, is `partial`; a line that
only reads files is never a run; a stopped run is neither passed nor failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.executor.tests import TestOutcome, TestReading, read_test_run, reads_only
from chickadee.executor.types import ExecResult, SandboxLevel, Violation
from chickadee.profiles import load_profile, shipped_runners

RUNS = Path(__file__).parent / "fixtures" / "runs"
PROFILES = tuple(load_profile(n) for n in ("python", "go", "rust", "javascript", "java"))

P, F, E, N = (
    TestOutcome.PASSED,
    TestOutcome.FAILED,
    TestOutcome.ERRORED,
    TestOutcome.NOT_COLLECTED,
)


def _read(
    command: str,
    stdout: str,
    exit_code: int = 0,
    *,
    cancelled: bool = False,
    timed_out: bool = False,
    violations: tuple[Violation, ...] = (),
) -> TestReading | None:
    result = ExecResult(
        exit_code=exit_code,
        stdout_sha=None,
        stderr_sha=None,
        wall_ms=1,
        timed_out=timed_out,
        sandbox=SandboxLevel.NONE,
        stdout=stdout,
        cancelled=cancelled,
        violations=violations,
    )
    return read_test_run(command, result, PROFILES)


def _captured(name: str, command: str, exit_code: int) -> TestReading:
    read = _read(command, (RUNS / f"{name}.txt").read_text(encoding="utf-8"), exit_code)
    assert read is not None
    return read


@pytest.mark.parametrize(
    ("name", "command", "exit_code", "outcome", "counts"),
    [
        ("pytest_default", "pytest", 1, F, (3, 2, 1)),
        ("pytest_q", "pytest -q", 1, F, (3, 2, 1)),
        ("pytest_pass_q", "pytest tests/test_b.py -q", 0, P, (1, 0, 0)),
        ("pytest_none", "pytest -q -k nomatch", 5, N, (0, 0, 0)),
        ("pytest_collect", "pytest -q", 2, E, (0, 0, 1)),
        ("ut_fail", "python3 -m unittest", 1, F, (1, 1, 1)),
        ("ut_ok", "python3 -m unittest", 0, P, (1, 0, 0)),
        ("ut_none", "python3 -m unittest", 5, N, (0, 0, 0)),
        ("go_fail", "go test ./...", 1, F, (1, 1, 0)),
        ("go_ok", "go test ./b/", 0, P, (1, 0, 0)),
        ("go_build", "go test ./d/", 1, E, (0, 0, 1)),
        ("go_none", "go test -run Nope ./b/", 0, N, (0, 0, 0)),
        ("cargo_fail", "cargo test", 101, F, (1, 1, 0)),
        ("cargo_ok", "cargo test", 0, P, (4, 0, 0)),
        ("cargo_none", "cargo test nomatch", 0, N, (0, 0, 0)),
        ("jest_fail", "npx jest", 1, F, (2, 1, 2)),
        ("jest_ok", "npx jest t/b.test.js", 0, P, (1, 0, 0)),
        ("jest_none", "npx jest nomatch", 1, N, (0, 0, 0)),
        ("vitest_fail", "npx vitest run", 1, F, (2, 1, 2)),
        ("vitest_ok", "npx vitest run t/b.test.js", 0, P, (1, 0, 0)),
        ("vitest_none", "npx vitest run nomatch", 1, N, (0, 0, 0)),
        ("mvn_fail", "mvn -B test", 1, F, (2, 2, 0)),
        ("mvn_q_fail", "mvn -B -q test", 1, F, (2, 2, 0)),
        ("mvn_ok", "mvn -B test", 0, P, (4, 0, 0)),
        ("gradle_fail", "gradle test --console=plain", 1, F, (2, 2, 0)),
        ("ctest_fail", "ctest", 8, F, (2, 1, 0)),
        ("ctest_ok", "ctest -R good", 0, P, (2, 0, 0)),
        ("ctest_none", "ctest -R nomatch", 0, N, (0, 0, 0)),
    ],
)
def test_each_runner_s_captured_output_reads_by_its_summary(
    name: str, command: str, exit_code: int, outcome: TestOutcome, counts: tuple[int, int, int]
) -> None:
    read = _captured(name, command, exit_code)
    assert (read.outcome, read.how) == (outcome, "summary")
    assert (read.passed, read.failed, read.errors) == counts


@pytest.mark.parametrize(
    ("name", "command", "exit_code", "failing"),
    [
        (
            "pytest_q",
            "pytest -q",
            1,
            (
                "tests/test_a.py::test_bad",
                "tests/test_a.py::test_p[2]",
                "tests/test_a.py::test_err",
            ),
        ),
        ("ut_fail", "python3 -m unittest", 1, ("test_u.T.test_err", "test_u.T.test_bad")),
        ("go_fail", "go test ./...", 1, ("TestBad", "TestSub", "TestSub/one")),
        ("cargo_fail", "cargo test", 101, ("tests::bad",)),
        ("jest_fail", "npx jest", 1, ("t/a.test.js > grp \u203a bad",)),
        ("vitest_fail", "npx vitest run", 1, ("t/a.test.js > grp > bad",)),
        ("mvn_fail", "mvn -B test", 1, ("app.AppTest.bad", "app.AppTest.err")),
        ("gradle_fail", "gradle test", 1, ("AppTest > bad()", "AppTest > err()")),
        ("ctest_fail", "ctest", 8, ("broken",)),
    ],
)
def test_the_failing_tests_are_named_each_once(
    name: str, command: str, exit_code: int, failing: tuple[str, ...]
) -> None:
    assert _captured(name, command, exit_code).failing == failing


@pytest.mark.parametrize(
    ("name", "command", "exit_code", "passing"),
    [
        (
            "pytest_v",
            "pytest -v",
            1,
            (
                "tests/test_a.py::test_ok",
                "tests/test_a.py::test_p[1]",
                "tests/test_b.py::test_fine",
            ),
        ),
        ("ut_v", "python3 -m unittest -v", 1, ("test_u.T.test_ok",)),
        ("go_v", "go test -v ./...", 1, ("TestOk", "TestSub/two", "TestFine")),
        (
            "cargo_ok",
            "cargo test",
            0,
            ("tests::bad", "tests::ok", "integ", "src/lib.rs - one (line 1)"),
        ),
        (
            "vitest_v",
            "npx vitest run --reporter=verbose",
            1,
            ("t/b.test.js > fine", "t/a.test.js > grp > ok"),
        ),
        ("ctest_fail", "ctest", 8, ("good", "also_good")),
    ],
)
def test_the_passing_tests_are_named_where_the_runner_prints_them(
    name: str, command: str, exit_code: int, passing: tuple[str, ...]
) -> None:
    assert _captured(name, command, exit_code).passing == passing


def test_a_go_summary_counts_packages() -> None:
    assert _captured("go_fail", "go test ./...", 1).unit == "package"
    assert _captured("cargo_ok", "cargo test", 0).unit == "test"


def test_every_shipped_runner_is_read_from_a_captured_output() -> None:
    """A runner whose patterns no captured output exercises is a guess."""
    read: set[str] = set()
    for path in RUNS.glob("*.txt"):
        found = _read("make test", path.read_text(encoding="utf-8"), 1)
        if found is not None:
            read.update(found.runners)
    assert {r.name for r in shipped_runners()} <= read


def test_a_compile_error_with_no_summary_is_errored() -> None:
    for name, command, code in (
        ("cargo_build", "cargo test", 101),
        ("mvn_build", "mvn -B test", 1),
    ):
        read = _captured(name, command, code)
        assert (read.outcome, read.how) == (E, "summary")


def test_printed_failures_beat_exit_zero() -> None:
    read = _read("pytest -q 2>&1 | tail -5", "FAILED t.py::a - x\n2 failed, 1 passed in 0.1s\n")
    assert read is not None
    assert (read.outcome, read.how, read.failed) == (F, "summary", 2)


def test_every_summary_in_one_output_is_summed() -> None:
    out = "3 passed in 0.1s\n...\n1 failed, 4 passed in 0.2s\n"
    read = _read("make test", out)
    assert read is not None
    assert (read.outcome, read.passed, read.failed) == (F, 7, 1)


@pytest.mark.parametrize(
    "command",
    ["pytest -q | grep FAIL", "pytest -q | head -3", "pytest; echo done", "pytest || true"],
)
def test_an_exit_code_that_is_another_command_s_is_partial(command: str) -> None:
    read = _read(command, "FAILED tests/test_a.py::test_bad - assert\n", 0)
    assert read is not None
    assert (read.outcome, read.how) == (TestOutcome.PARTIAL, "partial")
    assert read.failing == ("tests/test_a.py::test_bad",)
    assert not read.behavioral


@pytest.mark.parametrize(
    ("command", "exit_code", "outcome"),
    [
        ("cd pkg && pytest -q", 0, P),
        ("timeout 60 pytest", 1, F),
        ("pytest > out.txt 2>&1", 0, P),
        ("mvn -B -q test", 0, P),
    ],
)
def test_a_test_command_whose_status_is_its_own_is_read_by_its_exit_code(
    command: str, exit_code: int, outcome: TestOutcome
) -> None:
    read = _read(command, "", exit_code)
    assert read is not None
    assert (read.outcome, read.how) == (outcome, "exit_code")


def test_a_runner_that_would_not_start_is_errored() -> None:
    read = _read("python -m pytest", "/usr/bin/python: No module named pytest\n", 1)
    assert read is not None
    assert (read.outcome, read.how) == (E, "exit_code")


@pytest.mark.parametrize(
    "command",
    ["cat log.txt", "git show HEAD", "grep -n passed out.txt", "tail -5 run.log", "echo hi"],
)
def test_a_line_that_only_reads_is_never_a_run(command: str) -> None:
    assert reads_only(command)
    assert _read(command, "FAILED t.py::a\n1 failed, 3 passed in 0.2s\n", 0) is None


def test_another_command_is_no_test_run_unless_it_printed_a_summary() -> None:
    assert _read("ls -la", "total 0\n") is None
    assert not reads_only("./scripts/check.sh")
    read = _read("./scripts/check.sh", "2 passed in 0.1s\n")
    assert read is not None and (read.outcome, read.runners) == (P, ("pytest",))


def test_colour_codes_do_not_hide_a_summary() -> None:
    read = _read("npx vitest run", "\x1b[32m      Tests  \x1b[39m1 failed | 2 passed (3)\n", 1)
    assert read is not None and (read.outcome, read.failed, read.passed) == (F, 1, 2)


def test_a_stopped_run_is_neither_passed_nor_failed() -> None:
    """The printed failure here is a reaped run's partial output: trusting it would report a
    test failing that never finished."""
    printed = "FAILED tests/test_a.py::test_b\n1 failed in 0.1s\n"
    refused = (Violation(path="/etc", reason="outside the worktree"),)
    for read, outcome in (
        (_read("pytest -q", printed, -15, cancelled=True), TestOutcome.CANCELLED),
        (_read("pytest -q", printed, -9, timed_out=True), TestOutcome.TIMED_OUT),
        (_read("pytest -q", printed, 126, violations=refused), E),
    ):
        assert read is not None
        assert (read.outcome, read.how, read.behavioral) == (outcome, "stopped", False)
