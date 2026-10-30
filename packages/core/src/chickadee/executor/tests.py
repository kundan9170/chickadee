"""Read a test run (D236 part 1 and Amendment 1): the one reading of what a command's tests did,
for the facts Athena reads and the progress watchdog alike.

**A run is read by what it printed.** Each runner's summary lines are data
(`profiles/runners/`); a summary found in the output makes the command a test run whatever ran
it (`make test`, a script), and is read whatever the exit code: `2 failed` beats exit 0, and
every summary is summed, since cargo prints one per test binary and go one per package. A
summary that counts no test is `not_collected`, never a pass.

**An exit code is the test's only when the line's own status is the test command's**
(`shell_line.status_command`). A recognised test command with no summary whose status is
another command's (`pytest | grep`, `pytest; echo`, `pytest || true`) is `partial`: nothing
can be said. Otherwise it is read by its exit code, and says so.

**A line that only reads files is never a test run** (`reads_only`), whatever it prints: a log
shown by `cat` looks like a run.

`is_test_command` is the one reading of "this command runs tests": the facts,
the watchdog and this reading all ask it.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final, Literal

from chickadee.executor.types import ExecResult
from chickadee.profiles.loader import shipped_runners
from chickadee.profiles.model import LanguageProfile, SummarySpec, TestRunner
from chickadee.shell_line import PRINTERS, join, read_line, status_command

__all__ = [
    "DENIED_EXIT",
    "Reading",
    "TestOutcome",
    "TestReading",
    "is_test_command",
    "launched",
    "read_test_run",
    "reads_only",
]


class TestOutcome(StrEnum):
    # The module map names this file `tests.py`, so pytest tries to collect anything here
    # whose name starts with "Test". These are value types, not test cases.
    __test__ = False

    PASSED = "passed"
    FAILED = "failed"
    """A check ran and its assertion did not hold."""
    ERRORED = "errored"
    """The suite could not run: import error, collection crash, a compile error."""
    NOT_COLLECTED = "not_collected"
    """The run selected no checks at all."""
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    """The user stopped the run and the group was reaped. Nothing finished, so this is
    neither `passed` nor `errored`, and never `failed`: the nonzero exit a reap leaves is
    the signal's, not an assertion's."""
    PARTIAL = "partial"
    """A test command whose output was cut or whose exit code is another command's, with no
    summary in what was shown: nothing can be said of it, so never a pass."""


Reading = Literal["summary", "exit_code", "partial", "stopped"]
"""How a run was read: by its printed summary (or a line saying it could not run), by its exit
code, not at all (`partial`), or by the supervisor (`stopped`: cancelled, timed out,
refused)."""

DENIED_EXIT = 126
"""`Executor.run` returns this with a violation when policy refused to spawn."""

_CANNOT_RUN = re.compile(
    r"ERROR collecting|ImportError|ModuleNotFoundError|No module named|Cannot find module|"
    r"SyntaxError|INTERNALERROR|errors? during collection|command not found",
    re.I,
)
"""With no summary and a failing exit code that is the test's: the suite did not run."""

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class TestReading:
    """What one run's tests did, and how that was read."""

    __test__ = False

    outcome: TestOutcome
    how: Reading
    exec: ExecResult
    """The raw run, kept on every reading: a reading that discards its evidence cannot be
    argued with when it is wrong."""
    failing: tuple[str, ...] = ()
    """The failing tests the output names, in order, each once."""
    passing: tuple[str, ...] = ()
    """The passing tests the output names, where the runner prints them (`-v`, cargo, CTest)."""
    passed: int = 0
    failed: int = 0
    errors: int = 0
    runners: tuple[str, ...] = ()
    """The runners whose summary was read."""
    unit: Literal["test", "package"] = "test"
    """What the counts count: go's summaries are one a package."""

    @property
    def behavioral(self) -> bool:
        """Whether this says anything about the code under test: `errored`, `not_collected`,
        `partial` and a stopped run do not, and reporting one as a result is the overstatement
        this reading exists to prevent."""
        return self.outcome in (TestOutcome.PASSED, TestOutcome.FAILED)


def read_test_run(
    command: str,
    result: ExecResult,
    profiles: Sequence[LanguageProfile],
    runners: Sequence[TestRunner] | None = None,
) -> TestReading | None:
    """`command`'s run read as a test run, or None when it is not one: no runner's summary in
    its output and not a test command, or a line that only reads files."""
    if reads_only(command):
        return None
    runners = shipped_runners() if runners is None else runners
    text = _ANSI.sub("", f"{result.stdout}\n{result.stderr}")
    found = [(r, _summed(r, text)) for r in runners]
    read = [(r, counts) for r, counts in found if counts is not None]
    is_test = is_test_command(command, profiles)
    if not read and not is_test:
        return None
    stopped = _stopped(result)
    if stopped is not None:
        return TestReading(stopped, "stopped", result)
    named = [r for r, _ in read] or list(runners)
    failing = _names(text, named, "failing")
    passing = _names(text, named, "passing")
    if read:
        passed, failed, errors = (sum(c[i] for _, c in read) for i in range(3))
        return TestReading(
            _outcome(passed, failed, errors),
            "summary",
            result,
            failing,
            passing,
            passed,
            failed,
            errors,
            tuple(r.name for r, _ in read),
            "package" if any(r.unit == "package" for r, _ in read) else "test",
        )
    if any(_rx(p).search(text) for r in runners for p in r.names.broken):
        return TestReading(TestOutcome.ERRORED, "summary", result, failing, passing)
    status = status_command(command)
    if status is None or not is_test_command(join(status), profiles):
        return TestReading(TestOutcome.PARTIAL, "partial", result, failing, passing)
    if result.exit_code == 0:
        return TestReading(TestOutcome.PASSED, "exit_code", result, (), passing)
    if _CANNOT_RUN.search(text):
        return TestReading(TestOutcome.ERRORED, "exit_code", result)
    return TestReading(TestOutcome.FAILED, "exit_code", result, failing, passing)


def _stopped(result: ExecResult) -> TestOutcome | None:
    if result.cancelled:
        # First: a reaped run's partial output can read as anything, a half-printed `FAILED`
        # line or an import error the suite had not finished recovering from.
        return TestOutcome.CANCELLED
    if result.timed_out:
        return TestOutcome.TIMED_OUT
    if result.exit_code == DENIED_EXIT and result.violations:
        # The policy refused before anything spawned: no test ran, so never `failed`.
        return TestOutcome.ERRORED
    return None


def _outcome(passed: int, failed: int, errors: int) -> TestOutcome:
    """Printed failures win; errors with nothing else run are a suite that could not run."""
    if failed:
        return TestOutcome.FAILED
    if errors:
        return TestOutcome.FAILED if passed else TestOutcome.ERRORED
    return TestOutcome.PASSED if passed else TestOutcome.NOT_COLLECTED


def _summed(runner: TestRunner, text: str) -> tuple[int, int, int] | None:
    """`runner`'s summaries in `text`, every one summed as (passed, failed, errors); None when
    it printed none."""
    total = [0, 0, 0]
    seen = False
    for spec in runner.summary:
        for match in _rx(spec.line).finditer(text):
            seen = True
            for i, n in enumerate(_counts(spec, match.group(0))):
                total[i] += n
    return (total[0], total[1], total[2]) if seen else None


def _counts(spec: SummarySpec, line: str) -> tuple[int, int, int]:
    failed = _count(spec.failed, line)
    errors = _count(spec.errors, line)
    if spec.passed:
        passed = _count(spec.passed, line)
    else:
        rest = failed + errors + _count(spec.skipped, line)
        passed = max(_count(spec.total, line) - rest, 0)
    return passed, failed, errors


def _count(patterns: Sequence[str], line: str) -> int:
    """Each pattern's numbers summed, or its matches counted when it has no group."""
    n = 0
    for pattern in patterns:
        rx = _rx(pattern)
        for match in rx.finditer(line):
            n += int(match.group(1)) if rx.groups else 1
    return n


def _names(
    text: str, runners: Sequence[TestRunner], kind: Literal["failing", "passing"]
) -> tuple[str, ...]:
    """The test names `runners` print as `kind`, in order of their place in `text`, each once.
    A runner that prints a name under its file's line (jest) has it read as `file > name`."""
    found: list[tuple[int, str]] = []
    for runner in runners:
        files = (
            [(m.start(), m.group(1)) for m in _rx(runner.names.file).finditer(text)]
            if runner.names.file is not None
            else []
        )
        patterns = runner.names.failing if kind == "failing" else runner.names.passing
        for pattern in patterns:
            for match in _rx(pattern).finditer(text):
                name = match.group(1).strip()
                if not name:
                    continue
                under = [f for at, f in files if at < match.start()]
                found.append((match.start(), f"{under[-1]} > {name}" if under else name))
    return tuple(dict.fromkeys(name for _, name in sorted(found, key=lambda f: f[0])))


@functools.cache
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.M)


# -- what only reads ------------------------------------------------------------------------

_READERS: Final = frozenset(
    {
        *PRINTERS,
        "cat",
        "less",
        "more",
        "head",
        "tail",
        "grep",
        "egrep",
        "fgrep",
        "rg",
        "sed",
        "awk",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "nl",
        "column",
        "ls",
    }
)
"""Programs that show files or their own words and run nothing: what they print is never a
run, however much it looks like one."""

_GIT_READS: Final = frozenset({"log", "show", "diff", "blame", "grep", "status"})


def reads_only(command: str) -> bool:
    """Whether every command `command` runs only shows files or words (`cat log.txt`, `git
    show`, `grep -n FAILED out.txt`), so that what it printed is never read as a test run."""
    commands = read_line(command).commands
    return bool(commands) and all(_reads(words) for words in commands)


def _reads(words: Sequence[str]) -> bool:
    if not words:
        return True
    head = PurePosixPath(words[0]).name
    if head == "git":
        rest = [w for w in words[1:] if not w.startswith("-")]
        return bool(rest) and rest[0] in _GIT_READS
    return head in _READERS


# -- what runs tests ------------------------------------------------------------------------

_LAUNCHERS: Final = (
    ("uv", "run"),
    ("poetry", "run"),
    ("pipenv", "run"),
    ("pdm", "run"),
    ("hatch", "run"),
    ("python", "-m"),
    ("python3", "-m"),
    ("bundle", "exec"),
    ("pnpm", "exec"),
    ("npm", "exec"),
    ("npx",),
    ("yarn",),
)
"""Commands that run another tool by its name, which is what runs: `uv run pytest` is pytest,
`npx jest` is jest."""


_ANY_PROJECT: Final = (
    ("npm", "test"),
    ("npm", "run", "test"),
    ("yarn", "test"),
    ("pnpm", "test"),
    ("make", "test"),
    ("make", "check"),
    ("tox",),
    ("nox",),
    ("gradlew", "test"),
    ("unittest",),
)
"""Commands that run a project's tests whatever runner its profile names: a package's test
script, a make target, an environment runner, the wrapper a build ships, the standard library's
runner."""


def is_test_command(command: str, profiles: Sequence[LanguageProfile]) -> bool:
    """Whether `command` runs tests, in any of its simple commands: the one reading of "a test
    run", which the facts read and the progress watchdog never counts as a change. A profile's
    runner is its program and the words that pick its test command
    (`go test`, `cargo test`, `pytest`), so `go fmt` is no test run; options between them are
    skipped (`mvn -q test`, `cargo +nightly test`). A launcher is looked through (`uv run
    pytest`, `python -m pytest`), and so are a line's other commands and wrappers (`cd pkg &&
    pytest`, `timeout 60 pytest`, `shell_line.read_line`). A project's own test command counts
    whatever the profiles say (`_ANY_PROJECT`)."""
    keys = [
        *(_runner(p.tests.runner.argv) for p in profiles if p.tests.runner is not None),
        *_ANY_PROJECT,
    ]
    for words in read_line(command).commands:
        for ran in _forms(words):
            if any(k and ran[: len(k)] == k for k in keys):
                return True
    return False


def _forms(words: Sequence[str]) -> list[tuple[str, ...]]:
    """A simple command as each program it runs (`launched`), each with its options set
    aside: `mvn -q test` is `mvn test`."""
    return [
        (ran[0], *(w for w in ran[1:] if not w.startswith(("-", "+"))))
        for ran in launched(words)
    ]


def launched(words: Sequence[str]) -> list[tuple[str, ...]]:
    """A simple command as each program it runs: itself, then the tool each launcher in front
    runs (`yarn test` is yarn's test script, `yarn jest` is jest), each program by its file's
    name."""
    out: list[tuple[str, ...]] = []
    ran = list(words)
    while ran:
        ran[0] = PurePosixPath(ran[0]).name
        out.append(tuple(ran))
        prefix = next(
            (p for p in _LAUNCHERS if tuple(ran[: len(p)]) == p and len(ran) > len(p)), None
        )
        if prefix is None:
            break
        ran = ran[len(prefix) :]
    return out


def _runner(argv: Sequence[str]) -> tuple[str, ...]:
    """A runner's program and the words that pick its test command, up to its first option,
    placeholder or path: `go test./...` is `go test`, `npx jest --silent` is `jest`."""
    runs = launched(argv)
    if not runs:
        return ()
    ran = runs[-1]
    key = [ran[0]]
    for word in ran[1:]:
        if word.startswith(("-", "{")) or "/" in word or "." in word:
            break
        key.append(word)
    return tuple(key)
