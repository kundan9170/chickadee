"""One record per test (D236 part 2 and Amendment 1).

What these pin: the cautious selection reads pytest's paths and ids, jest's and vitest's paths
and cargo's name filter, and anything else as unknown; a test is fixed when the output names it
passing, or a run covering it (the same command, the whole suite, or its selection) does not
name it failing; a narrower run that cannot be shown to cover it leaves its row; a test failing
that an earlier covering run passed is broken; one failing at first sight is marked so; a
parametrised case is its function; only a complete run moves a row.
"""

from __future__ import annotations

import pytest

from chickadee.box.tests_record import (
    RunSeen,
    RunSelection,
    TestsRecord,
    Where,
    file_of,
    row_id,
    selection_of,
)
from chickadee.executor.tests import TestReading, read_test_run
from chickadee.executor.types import ExecResult, SandboxLevel
from chickadee.profiles import load_profile, shipped_runners

RUNNERS = shipped_runners()
PROFILES = tuple(load_profile(n) for n in ("python", "rust", "javascript"))


def _reading(command: str, stdout: str, exit_code: int = 0) -> TestReading:
    result = ExecResult(
        exit_code=exit_code,
        stdout_sha=None,
        stderr_sha=None,
        wall_ms=1,
        timed_out=False,
        sandbox=SandboxLevel.NONE,
        stdout=stdout,
    )
    read = read_test_run(command, result, PROFILES)
    assert read is not None
    return read


class _Task:
    """A task's runs, one a call."""

    def __init__(self) -> None:
        self.record = TestsRecord(RUNNERS)
        self.call = 0

    def run(self, command: str, stdout: str, exit_code: int = 0) -> RunSeen:
        self.call += 1
        return self.record.see(
            command, _reading(command, stdout, exit_code), Where(1, self.call)
        )

    def status(self, test: str) -> str | None:
        return next((r.status for r in self.record.rows if r.id == test), None)


ROUTE = "tests/test_api.py::test_route"
FAILS = f"FAILED {ROUTE} - assert 1 == 2\n1 failed, 3 passed in 0.1s\n"


@pytest.mark.parametrize(
    ("command", "selection"),
    [
        ("pytest -q", RunSelection("whole", "pytest", "paths")),
        ("uv run pytest -q -x --tb=short", RunSelection("whole", "pytest", "paths")),
        (
            "python -m pytest tests/test_api.py -x",
            RunSelection("some", "pytest", "paths", ("tests/test_api.py",)),
        ),
        (f"pytest {ROUTE}", RunSelection("some", "pytest", "paths", (ROUTE,))),
        (
            "pytest --maxfail 3 ./tests/",
            RunSelection("some", "pytest", "paths", ("tests/",)),
        ),
        ("npx jest t/a --silent", RunSelection("some", "jest", "paths", ("t/a",))),
        (
            "npx vitest run t/b.test.js --reporter=verbose",
            RunSelection("some", "vitest", "paths", ("t/b.test.js",)),
        ),
        ("cargo test -q route", RunSelection("some", "cargo", "names", ("route",))),
        ("cargo test", RunSelection("whole", "cargo", "names")),
        ("pytest -k route", RunSelection("unknown", "pytest")),
        ("pytest 'tests/test_a.py::test_p[1]'", RunSelection("unknown", "pytest")),
        ("npx jest -t 'grp bad'", RunSelection("unknown", "jest")),
        ("cargo test --lib", RunSelection("unknown", "cargo")),
        ("cd pkg && pytest", RunSelection("unknown")),
        ("make test", RunSelection("unknown")),
        ("go test ./...", RunSelection("unknown")),
        ("pytest a.py; cargo test", RunSelection("unknown")),
    ],
)
def test_the_selection_is_read_cautiously(command: str, selection: RunSelection) -> None:
    assert selection_of(command, RUNNERS) == selection


def test_a_selection_covers_only_what_it_certainly_ran() -> None:
    tests = RunSelection("some", "pytest", "paths", ("tests/",))
    assert tests.covers("pytest", ROUTE)
    assert not tests.covers("pytest", "other/test_x.py::test_y")
    assert not tests.covers("cargo", ROUTE)
    one = RunSelection("some", "pytest", "paths", ("tests/test_api.py::test_route",))
    assert one.covers("pytest", ROUTE) and not one.covers("pytest", f"{ROUTE}_two")
    assert RunSelection("some", "cargo", "names", ("rout",)).covers("cargo", "api::route")
    assert RunSelection("some", "jest", "paths", ("t/a",)).covers(
        "jest", "t/a.test.js > grp > b"
    )
    assert not RunSelection("unknown", "pytest").covers("pytest", ROUTE)


def test_a_test_s_file_and_id() -> None:
    assert file_of(ROUTE) == "tests/test_api.py"
    assert file_of("t/a.test.js > grp > bad") == "t/a.test.js"
    assert file_of("src/lib.rs - one (line 1)") == "src/lib.rs"
    assert file_of("TestSub/one") is None and file_of("api::tests::route") is None
    assert row_id("tests/test_a.py::test_p[2-3]") == "tests/test_a.py::test_p"


def test_the_same_command_passing_fixes_a_test() -> None:
    task = _Task()
    assert task.run("pytest -q", FAILS, 1).failing_first == (ROUTE,)
    seen = task.run("pytest -q", "4 passed in 0.1s\n")
    assert (seen.fixed, task.status(ROUTE)) == ((ROUTE,), "passed")


def test_a_narrower_run_that_cannot_be_shown_to_cover_it_leaves_its_row() -> None:
    """The one loss of reading no command beyond the easy forms: a fix waits for a covering
    run, never faked."""
    task = _Task()
    task.run("pytest -q", FAILS, 1)
    assert task.run("pytest tests/test_other.py", "2 passed in 0.1s\n").fixed == ()
    assert task.run("pytest -k route", "1 passed, 3 deselected in 0.1s\n").fixed == ()
    assert task.status(ROUTE) == "failed"
    assert task.run("pytest tests/test_api.py", "1 passed in 0.1s\n").fixed == (ROUTE,)


def test_a_test_named_passing_is_passed_whatever_the_command() -> None:
    task = _Task()
    task.run("pytest -q", FAILS, 1)
    seen = task.run("pytest -k route -v", f"{ROUTE} PASSED [100%]\n1 passed in 0.1s\n")
    assert seen.fixed == (ROUTE,)


def test_a_test_a_covering_run_passed_is_broken_when_it_fails() -> None:
    task = _Task()
    task.run("pytest -q", "4 passed in 0.1s\n")
    seen = task.run("pytest -q", FAILS, 1)
    assert (seen.broken, seen.failing_first) == ((ROUTE,), ())
    row = next(r for r in task.record.rows if r.id == ROUTE)
    assert (row.failing_first, row.passed_at) == (False, Where(1, 1)), "the covering run's"


def test_a_test_failing_after_a_pass_is_broken_and_keeps_where_it_passed() -> None:
    task = _Task()
    task.run("pytest -q", FAILS, 1)
    task.run("pytest -q", "4 passed in 0.1s\n")
    seen = task.run("pytest -q", FAILS, 1)
    row = next(r for r in task.record.rows if r.id == ROUTE)
    assert seen.broken == (ROUTE,)
    assert (row.status, row.passed_at, row.seen_at) == ("failed", Where(1, 2), Where(1, 3))


def test_a_parametrised_case_is_its_function_failed_when_any_case_failed() -> None:
    task = _Task()
    out = (
        "tests/test_a.py::test_p[1] PASSED\n"
        "tests/test_a.py::test_p[2] FAILED\n"
        "FAILED tests/test_a.py::test_p[2] - assert\n"
        "1 failed, 1 passed in 0.1s\n"
    )
    task.run("pytest -v", out, 1)
    assert task.status("tests/test_a.py::test_p") == "failed"


def test_cargo_s_name_filter_covers_the_tests_it_names() -> None:
    task = _Task()
    task.run(
        "cargo test",
        "test tests::bad ... FAILED\ntest result: FAILED. 1 passed; 1 failed; 0 ignored;\n",
        101,
    )
    seen = task.run("cargo test bad", "test result: ok. 1 passed; 0 failed; 0 ignored;\n")
    assert seen.fixed == ("tests::bad",)


def test_jest_s_path_covers_the_tests_under_its_file() -> None:
    task = _Task()
    task.run(
        "npx jest",
        "FAIL t/a.test.js\n  \u25cf grp \u203a bad\n\n"
        "Tests:       1 failed, 2 passed, 3 total\n",
        1,
    )
    assert task.status("t/a.test.js > grp \u203a bad") == "failed"
    assert task.run("npx jest t/b", "Tests:       1 passed, 1 total\n").fixed == ()
    seen = task.run("npx jest t/a", "Tests:       2 passed, 2 total\n")
    assert seen.fixed == ("t/a.test.js > grp \u203a bad",)


def test_only_a_complete_run_moves_a_row() -> None:
    task = _Task()
    task.run("pytest -q", FAILS, 1)
    assert task.run("pytest -q | head -3", "...\n").fixed == ()
    assert task.run("pytest -q", "").fixed == ()
    assert task.status(ROUTE) == "failed"


def test_a_break_still_failing_after_more_changes_is_confirmed_once_a_round() -> None:
    """Broken at one run, failing again at a later one with files changed in
    between: a regression, named with where it last passed and what changed since."""
    record = TestsRecord(RUNNERS)
    edits: list[tuple[Where, tuple[str, ...]]] = []

    def changes(after: Where) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for at, ps in edits if at > after for p in ps))

    def run(at: int, out: str, code: int = 0) -> RunSeen:
        return record.see("pytest -q", _reading("pytest -q", out, code), Where(1, at), changes)

    run(1, "4 passed in 0.1s\n")
    edits.append((Where(1, 2), ("api.py",)))
    assert run(3, FAILS, 1).broken == (ROUTE,)
    assert run(4, FAILS, 1).regressed == (), "no change since it broke"
    edits.append((Where(1, 5), ("app.py",)))
    [regressed] = run(6, FAILS, 1).regressed
    assert (regressed.test, regressed.passed_at, regressed.changed) == (
        ROUTE,
        "call 1 of round 1",
        ("api.py", "app.py"),
    )
    edits.append((Where(1, 7), ("app.py",)))
    assert run(8, FAILS, 1).regressed == (), "once a test a round"


def test_the_task_s_tests_are_told_apart_from_those_already_failing() -> None:
    """A failing test is the task's when its brief names it, the task changed
    its file, or its file's stem names a changed file; any other failing at first sight was
    already failing."""
    task = _Task()
    out = (
        "FAILED tests/test_seg.py::test_cut - x\n"
        "FAILED tests/test_db.py::test_pool - x\n"
        "FAILED tests/test_misc.py::test_route - x\n"
        "3 failed, 1 passed in 0.1s\n"
    )
    task.run("pytest -q", out, 1)
    seen = task.record.seen("Make test_route pass.", {"src/seg.py"})
    assert seen is not None
    assert seen.failing == ("tests/test_misc.py::test_route", "tests/test_seg.py::test_cut")
    assert seen.already_failing == ("tests/test_db.py::test_pool",)
    assert TestsRecord(RUNNERS).seen("", set()) is None
