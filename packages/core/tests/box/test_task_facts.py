"""A task's facts: `box/facts.measure`.

What these pin: the facts are measured from the trees and the journal, never the report: each
file created, edited or deleted with its counts; a test file added apart from an existing one
edited; the symbols added and removed against the table taken at delegation, unknown without
it; every command with its exit, and a run of the profile's own test runner read by
`classify`, so a run that collected nothing never reads as a pass and a failing run names its
tests; a command the sandbox refused is named, with no exit; each changed file's syntax, by
its profile's check over the file the task left, and a file none judges left out.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from box_helpers import delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.box.facts import render_facts
from chickadee.boxctx.records import symbol_table
from chickadee.executor.tests import is_test_command
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.profiles import load_profile
from chickadee.schema.journal import CommandFactP, SyntaxFactP, TaskFactsP

APP = "def go() -> int:\n    return 1\n"
TEST = "from app import go\n\n\ndef test_go() -> None:\n    assert go() == 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {"app.py": APP, "old.py": "OLD = 1\n", "tests/test_app.py": TEST},
    )


def _run(root: Path, *script: object, stdout: bytes = b"", exit_code: int = 0) -> TaskFactsP:
    worker, journal, _ = worker_rig(
        root, [*script, answer("All tests pass.")], stdout=stdout, exit_code=exit_code
    )
    return worker.start("T1", "Do it.", delegate(journal, root)).facts


def _command(root: Path, command: str, stdout: bytes, exit_code: int) -> CommandFactP:
    facts = _run(
        root,
        calls(call("shell", "c1", command=command)),
        stdout=stdout,
        exit_code=exit_code,
    )
    [fact] = facts.commands
    return fact


def test_files_with_their_counts_and_tests_added_apart_from_tests_edited(root: Path) -> None:
    facts = _run(
        root,
        calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
        calls(call("edit", "c2", path="tests/test_app.py", old="== 1", new="== 2")),
        calls(
            call("create", "c3", path="tests/test_more.py", content="def test_x():\n    pass\n")
        ),
        calls(call("delete", "c4", path="old.py")),
    )
    assert {(f.path, f.kind, f.added, f.removed) for f in facts.files} == {
        ("app.py", "edited", 1, 1),
        ("tests/test_app.py", "edited", 1, 1),
        ("tests/test_more.py", "created", 2, 0),
        ("old.py", "deleted", 0, 1),
    }
    assert facts.tests_added == ("tests/test_more.py",)
    assert facts.tests_edited == ("tests/test_app.py",)


def test_each_changed_file_s_syntax_is_a_fact_athena_reads(root: Path) -> None:
    """D196 §6 reports whether the syntax broke: the worker is told by its edit, and Athena by
    this fact. A deleted file and a file no profile owns are not judged."""
    facts = _run(
        root,
        calls(call("edit", "c1", path="app.py", old="return 1", new="return (")),
        calls(call("create", "c2", path="tests/test_more.py", content="X = 1\n")),
        calls(call("create", "c3", path="NOTES.md", content="notes\n")),
        calls(call("delete", "c4", path="old.py")),
    )
    by_path = {f.path: f for f in facts.syntax}
    assert set(by_path) == {"app.py", "tests/test_more.py"}
    assert by_path["tests/test_more.py"] == SyntaxFactP(
        path="tests/test_more.py", ok=True, depth="compiled"
    )
    broken = by_path["app.py"]
    assert not broken.ok and broken.detail is not None and "app.py" in broken.detail
    row = next(r for r in render_facts(facts, "reported").splitlines() if "syntax:" in r)
    assert row.startswith("  syntax: broken in app.py (")
    assert row.endswith("; ok in tests/test_more.py")


def test_a_task_that_changed_nothing_says_no_file_was_checked(root: Path) -> None:
    rows = render_facts(_run(root), "reported").splitlines()
    assert "  syntax: no changed file was checked" in rows


def test_no_step_measures_nothing_and_the_report_is_not_read(root: Path) -> None:
    facts = _run(root)
    assert (facts.files, facts.commands, facts.end_tree) == ((), (), facts.start_tree)


def test_a_test_run_that_collected_nothing_never_reads_as_a_pass(root: Path) -> None:
    fact = _command(root, "pytest -q tests/nothing.py", b"no tests ran in 0.01s\n", 5)
    assert (fact.exit_code, fact.test_outcome) == (5, "not_collected")


def test_a_failing_run_names_its_failing_tests(root: Path) -> None:
    out = b"FAILED tests/test_app.py::test_go - assert 2 == 1\n1 failed in 0.02s\n"
    fact = _command(root, "pytest -q", out, 1)
    assert (fact.test_outcome, fact.failing) == ("failed", ("tests/test_app.py::test_go",))
    assert fact.tail.endswith("1 failed in 0.02s") and fact.output_sha is not None


def test_a_passing_run_reads_as_passed_through_a_path_and_an_assignment(root: Path) -> None:
    fact = _command(root, "PYTHONPATH=. .venv/bin/pytest -q", b"1 passed in 0.01s\n", 0)
    assert fact.test_outcome == "passed"


def test_printed_failures_beat_an_exit_code_a_pipe_hid(root: Path) -> None:
    """`pytest | tail` exits with tail's 0, and read so it was a pass."""
    out = b"FAILED tests/test_app.py::test_go - assert 2 == 1\n1 failed in 0.02s\n"
    fact = _command(root, "pytest -q 2>&1 | tail -5", out, 0)
    assert (fact.test_outcome, fact.failing) == ("failed", ("tests/test_app.py::test_go",))


def test_a_cut_run_with_no_summary_is_partial_never_a_pass(root: Path) -> None:
    fact = _command(root, "pytest -q | head -3", b"..F\n", 0)
    assert fact.test_outcome == "partial"


def test_a_log_shown_by_cat_is_not_a_test_run(root: Path) -> None:
    fact = _command(root, "cat last_run.log", b"1 failed, 3 passed in 0.2s\n", 0)
    assert fact.test_outcome is None


def test_another_command_is_not_read_as_a_test_run(root: Path) -> None:
    fact = _command(root, "ls -la", b"total 0\n", 0)
    assert (fact.command, fact.exit_code, fact.test_outcome) == ("ls -la", 0, None)


@pytest.fixture
def indexed(root: Path) -> Iterator[tuple[Indexer, Graph]]:
    indexer = Indexer(root)
    indexer.full_index()
    yield indexer, Graph(root, indexer.conn)
    indexer.close()


def test_symbols_added_and_removed_against_the_table_at_delegation(
    root: Path, indexed: tuple[Indexer, Graph]
) -> None:
    indexer, graph = indexed
    before = symbol_table(graph, ["app.py", "old.py"])
    worker, journal, _ = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="def go()", new="def run()")),
            calls(
                call("create", "c2", path="new.py", content="def fresh() -> None:\n    pass\n")
            ),
            answer("done"),
        ],
        graph=graph,
        indexer=indexer,
    )
    facts = worker.start("T1", "Rename.", delegate(journal, root), before=before).facts
    assert facts.symbols_known
    assert (facts.symbols_added, facts.symbols_removed) == (
        ("app.run", "new.fresh"),
        ("app.go",),
    )


def test_without_the_table_at_delegation_the_symbols_are_unknown(root: Path) -> None:
    facts = _run(root)
    assert (facts.symbols_known, facts.symbols_added, facts.symbols_removed) == (False, (), ())


@pytest.mark.parametrize(
    ("command", "runs_tests"),
    [
        ("pytest -q", True),
        ("cd pkg && pytest -x", True),
        ("uv run pytest", True),
        ("python -m pytest tests/", True),
        ("timeout 60 pytest", True),
        ("go test ./...", True),
        ("cargo test -q", True),
        ("npx jest src", True),
        ("mvn -q test", True),
        ("cargo +nightly test", True),
        ("npm test", True),
        ("yarn test", True),
        ("make test", True),
        ("tox -e py312", True),
        ("./gradlew test", True),
        ("python -m unittest", True),
        ("go build -o test ./...", False),
        ("yarn install", False),
        ("make build", False),
        ("pip install pytest", False),
        ("go fmt ./...", False),
        ("go generate ./...", False),
        ("cargo fmt", False),
        ("npx prettier --write src", False),
        ("echo pytest", False),
    ],
)
def test_a_test_run_is_the_runner_and_its_command_wherever_the_line_runs_it(
    command: str, runs_tests: bool
) -> None:
    """The one reading of a test run: the runner's program and the words that pick its
    test command, looking through launchers and the line's other commands."""
    profiles = tuple(load_profile(n) for n in ("python", "go", "rust", "javascript", "java"))
    assert is_test_command(command, profiles) is runs_tests
