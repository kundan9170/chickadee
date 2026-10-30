"""One record per test : each test's last status, where it was
seen, where it last passed, and which runs could have seen it. What the watchdog and hook 2
read to tell a test the task broke from one already failing.

**A status comes from the output, never from the command.** A test the output names failing is
failed; one it names passing is passed. A test failing earlier and not named is passed only when
the run **covers** it: the run is the same command as one that saw it, or the whole suite of its
runner, or its selection, read by the cautious parser (`selection_of`), includes it. Any other
run leaves its row as it was. So the record can miss progress, never fake it, and a row says
"last seen failing", never "still failing".

**Only a complete run moves a row:** one read by its summary (`executor/tests.read_test_run`).
A partial, exit-code or stopped reading changes nothing.

**The cautious parser reads the easy forms only:** for pytest, file paths and `path::id`s; for
jest and vitest, substrings of a file's path; for cargo, a substring of a test's name. Each
runner's flags that select nothing are data (`profiles/runners/`). Anything else is unknown:
a flag it does not know, a filter expression (`-k`, `-run`, `-t`), a wrapper (`make test`), a
`cd` before the runner, any other runner.

A parametrised case is read as its function (`test_x[1-2]` is `test_x`), failed when any case
failed.

**Concurrency.** A record belongs to one task's progress record, fed on its worker's thread.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final, Literal

from chickadee.executor.tests import TestReading, launched
from chickadee.hooks.payloads import Regression, TestsSeen
from chickadee.profiles.model import RunnerCommand, TestRunner
from chickadee.shell_line import read_line

__all__ = [
    "RunSeen",
    "RunSelection",
    "TestRow",
    "TestsRecord",
    "Where",
    "file_of",
    "relevant",
    "row_id",
    "selection_of",
]


@dataclass(frozen=True, order=True)
class Where:
    """A call of a task: its round and its place in that round's calls, from 1."""

    round: int
    call: int

    def __str__(self) -> str:
        return f"call {self.call} of round {self.round}"


# -- the selection -------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSelection:
    """What a run selected, as far as the command says: `whole` (the runner with nothing that
    selects), `some` (the `paths` or `names` it names) or `unknown`."""

    kind: Literal["whole", "some", "unknown"]
    runner: str | None = None
    selects: Literal["paths", "names"] | None = None
    items: tuple[str, ...] = ()

    def covers(self, runner: str, test: str) -> bool:
        """Whether this run certainly ran `test` of `runner`; False when it cannot be told."""
        if self.runner != runner or self.kind == "unknown":
            return False
        if self.kind == "whole":
            return True
        if self.selects == "names":
            return any(item in test for item in self.items)
        file = file_of(test)
        if file is None:
            return False
        return any(_under(test, file, item, runner) for item in self.items)


UNKNOWN: Final = RunSelection("unknown")

_MOVES: Final = frozenset({"cd", "pushd", "popd"})
_REGEX: Final = re.compile(r"[\\^$*+?()\[\]{}|]")


def selection_of(command: str, runners: Sequence[TestRunner]) -> RunSelection:
    """`command`'s selection, read cautiously: exact, or `unknown`. The line
    must run one runner the data describes, and nothing that moves its directory."""
    commands = read_line(command).commands
    if any(words and PurePosixPath(words[0]).name in _MOVES for words in commands):
        return UNKNOWN
    found: list[tuple[TestRunner, RunnerCommand, tuple[str, ...]]] = []
    for words in commands:
        for runner in runners:
            spec = runner.command
            if spec is None:
                continue
            for ran in launched(words):
                if ran[: len(spec.words)] == spec.words:
                    found.append((runner, spec, ran[len(spec.words) :]))
                    break
    if len(found) != 1:
        return UNKNOWN
    runner, spec, args = found[0]
    items = _arguments(spec, args)
    if items is None:
        return RunSelection("unknown", runner.name)
    if not items:
        return RunSelection("whole", runner.name, spec.selects)
    return RunSelection("some", runner.name, spec.selects, items)


def _arguments(spec: RunnerCommand, args: Sequence[str]) -> tuple[str, ...] | None:
    """The words that select, or None for any word the data does not know."""
    items: list[str] = []
    i = 0
    while i < len(args):
        word = args[i]
        if word.startswith("-"):
            name = word.split("=", 1)[0]
            if word in spec.flags:
                i += 1
            elif name in spec.values:
                i += 1 if "=" in word else 2
            else:
                return None
            continue
        if spec.selects == "paths" and _REGEX.search(word) is not None:
            return None
        items.append(word.removeprefix("./"))
        i += 1
    return tuple(items)


def _under(test: str, file: str, item: str, runner: str) -> bool:
    """Whether a `paths` selection's `item` takes in `test`, whose file is `file`."""
    if runner == "pytest":
        if "::" in item:
            return test == item or test.startswith(f"{item}::") or test.startswith(f"{item}[")
        item = item.rstrip("/")
        return file == item or file.startswith(f"{item}/")
    # jest and vitest match their arguments against the file's path.
    return item in file


# -- a test's id and file -----------------------------------------------------------------

_CASE: Final = re.compile(r"\[[^\]]*\]$")


def row_id(name: str) -> str:
    """A test's id as the record keys it: a parametrised case is its function."""
    return _CASE.sub("", name)


def file_of(test: str) -> str | None:
    """The file a test's id names, where it names one: pytest's `path::name`, jest's and
    vitest's `path > name`, a cargo doc test's `path - item (line n)`."""
    for sep in ("::", " > ", " - "):
        head, found, _ = test.partition(sep)
        if found and "." in PurePosixPath(head).name:
            return head
    return None


# -- the record ----------------------------------------------------------------------------


@dataclass
class TestRow:
    """One test as the task's runs saw it."""

    __test__ = False

    runner: str
    id: str
    status: Literal["passed", "failed"]
    seen_at: Where
    """The last run that saw it, named or covering it."""
    passed_at: Where | None = None
    """The last run it passed in."""
    failing_first: bool = False
    """Failing the first time any run could see it: there before the task's runs, or new and
    failing."""
    failed_at: Where | None = None
    """The last run it failed in."""
    commands: set[str] = field(default_factory=set[str])
    """Every command whose run saw it."""

    @property
    def key(self) -> tuple[str, str]:
        return (self.runner, self.id)


@dataclass(frozen=True)
class RunSeen:
    """What one complete run changed: the tests it fixed (failed to passed) and broke (passed
    to failed), and those it saw failing that were never seen passing."""

    fixed: tuple[str, ...] = ()
    broken: tuple[str, ...] = ()
    failing_first: tuple[str, ...] = ()
    regressed: tuple[Regression, ...] = ()
    """Tests broken earlier and failing again in this run after more changes, each once a
    round."""


Changes = Callable[[Where], tuple[str, ...]]
"""The files the task changed after a call, in the order it changed them."""


@dataclass(frozen=True)
class _SeenRun:
    at: Where
    command: str
    runners: tuple[str, ...]
    selection: RunSelection
    failing: frozenset[tuple[str, str]]
    passing: frozenset[tuple[str, str]]


class TestsRecord:
    """A task's tests, fed each test run in order (`see`), the same live and rebuilt."""

    __test__ = False

    def __init__(self, runners: Sequence[TestRunner]) -> None:
        self._runners = tuple(runners)
        self._rows: dict[tuple[str, str], TestRow] = {}
        self._runs: list[_SeenRun] = []
        self._broken: dict[tuple[str, str], Where] = {}
        """Each test broken and not fixed since, and where it broke."""
        self._noted: set[tuple[int, tuple[str, str]]] = set()
        """Each (round, test) a regression note was given for: once a test a round."""

    @property
    def rows(self) -> tuple[TestRow, ...]:
        return tuple(self._rows.values())

    def failing(self) -> tuple[TestRow, ...]:
        """The tests last seen failing."""
        return tuple(r for r in self._rows.values() if r.status == "failed")

    def see(
        self, command: str, reading: TestReading, at: Where, changes: Changes | None = None
    ) -> RunSeen:
        """A test run at `at`: its rows moved as its output and coverage say. `changes` names
        the files changed after a call, for the regression note; without it none is given."""
        if reading.how != "summary" or not reading.runners:
            return RunSeen()
        runner = reading.runners[0] if len(reading.runners) == 1 else None
        selection = selection_of(command, self._runners) if runner is not None else UNKNOWN
        named_failing = _keyed(reading.runners, reading.failing)
        named_passing = _keyed(reading.runners, reading.passing) - named_failing
        run = _SeenRun(
            at,
            _spelled(command),
            reading.runners,
            selection,
            frozenset(named_failing),
            frozenset(named_passing),
        )
        fixed: list[str] = []
        broken: list[str] = []
        first: list[str] = []
        for key in sorted(named_failing):
            row = self._rows.get(key)
            if row is None:
                passed = self._last_passed(key)
                if passed is not None:
                    broken.append(key[1])
                    row = TestRow(key[0], key[1], "failed", at, passed_at=passed)
                else:
                    first.append(key[1])
                    row = TestRow(key[0], key[1], "failed", at, failing_first=True)
                self._rows[key] = row
            elif row.status == "passed":
                broken.append(key[1])
            row.status, row.seen_at, row.failed_at = "failed", at, at
            row.commands.add(run.command)
        for key in sorted(named_passing):
            row = self._rows.setdefault(key, TestRow(key[0], key[1], "passed", at))
            if row.status == "failed":
                fixed.append(key[1])
            row.status, row.seen_at, row.passed_at = "passed", at, at
            row.commands.add(run.command)
        for row in self._rows.values():
            if row.key in named_failing or row.key in named_passing:
                continue
            if row.runner in run.runners and self._covers(run, row):
                if row.status == "failed":
                    fixed.append(row.id)
                row.status, row.seen_at, row.passed_at = "passed", at, at
                row.commands.add(run.command)
        self._runs.append(run)
        regressed = self._regressed(named_failing, set(broken), at, changes)
        for test in broken:
            self._broken[(run.runners[0], test)] = at
        for key in [k for k in self._broken if self._rows[k].status == "passed"]:
            del self._broken[key]
        return RunSeen(tuple(fixed), tuple(broken), tuple(first), regressed)

    def _regressed(
        self,
        failing: set[tuple[str, str]],
        broken_now: set[str],
        at: Where,
        changes: Changes | None,
    ) -> tuple[Regression, ...]:
        """The tests this run confirms broken: broken by an earlier run, failing in this one,
        with the files changed in between."""
        if changes is None:
            return ()
        out: list[Regression] = []
        for key in sorted(failing):
            broke = self._broken.get(key)
            row = self._rows[key]
            if broke is None or key[1] in broken_now or (at.round, key) in self._noted:
                continue
            if not changes(broke) or row.passed_at is None:
                continue
            self._noted.add((at.round, key))
            out.append(
                Regression(
                    test=key[1], passed_at=str(row.passed_at), changed=changes(row.passed_at)
                )
            )
        return tuple(out)

    def seen(self, brief: str, changed: Collection[str]) -> TestsSeen | None:
        """The task's tests for the reader of its report; None before any
        complete run. `changed` is every file the task changed."""
        if not self._runs:
            return None
        rows = list(self._rows.values())
        fixed = [r for r in rows if r.status == "passed" and r.failed_at is not None]
        broken = [r for r in rows if r.status == "failed" and r.passed_at is not None]
        mine = [
            r
            for r in rows
            if r.status == "failed" and r.passed_at is None and relevant(r, brief, changed)
        ]
        theirs = [
            r
            for r in rows
            if r.status == "failed" and r.passed_at is None and not relevant(r, brief, changed)
        ]
        return TestsSeen(
            fixed=tuple(r.id for r in fixed),
            broken=tuple(r.id for r in broken),
            failing=tuple(r.id for r in mine),
            already_failing=tuple(r.id for r in theirs),
            edited=tuple(r.id for r in fixed if file_of(r.id) in changed),
        )

    def _covers(self, run: _SeenRun, row: TestRow) -> bool:
        return run.command in row.commands or run.selection.covers(row.runner, row.id)

    def _last_passed(self, key: tuple[str, str]) -> Where | None:
        """Where an earlier complete run last passed this test, unnamed or named: it named it
        passing, or covered it by its selection and named it nowhere failing."""
        runner, test = key
        passed = [
            run.at
            for run in self._runs
            if runner in run.runners
            and key not in run.failing
            and (key in run.passing or run.selection.covers(runner, test))
        ]
        return passed[-1] if passed else None


def relevant(row: TestRow, brief: str, changed: Collection[str]) -> bool:
    """Whether a test is the task's: its brief names it or its file, the
    task changed its file, it passed earlier in the task, or its file's stem (its name's,
    where it names no file) names a file the task changed (`test_seg.py` and `seg.py`): the
    build's stand-in for `tests_for`."""
    file = file_of(row.id)
    name = _name(row.id)
    if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", brief) or (file and file in brief):
        return True
    if (file is not None and file in changed) or row.passed_at is not None:
        return True
    stem = _stem(PurePosixPath(file).name if file is not None else name)
    return bool(stem) and any(_stem(PurePosixPath(p).name) == stem for p in changed)


def _name(test: str) -> str:
    """A test's own name, the last part of its id: `test_x` of `a.py::T::test_x`, `bad` of
    `tests::bad` and of jest's `t/a.test.js > grp \u203a bad`, `TestSub` of go's
    `TestSub/one`, `test_bad` of unittest's `mod.T.test_bad`, `one` of a cargo doc test."""
    if " - " in test:
        # A cargo doc test: `src/lib.rs - one (line 1)` is `one`.
        return test.split(" - ", 1)[1].split(" (", 1)[0]
    for sep in ("::", " > ", " \u203a "):
        if sep in test:
            test = test.rsplit(sep, 1)[-1]
    if "/" in test:
        return test.split("/", 1)[0]
    return test.rsplit(".", 1)[-1] if " " not in test else test


_MARKS: Final = re.compile(r"^tests?_|_tests?$|^[Tt]est(?=[A-Z])|(?<=[a-z])Tests?$")


def _stem(name: str) -> str:
    """A file's or test's name with its extensions and test marks off: `test_seg.py`,
    `seg.test.js`, `seg_test.go`, `TestSeg` and `SegTest` are all `seg`."""
    return _MARKS.sub("", name.split(".")[0]).lower()


def _keyed(runners: Sequence[str], names: Iterable[str]) -> set[tuple[str, str]]:
    """Each name as its row's key: its runner (the run's one, or the first when it summed
    several) and its id."""
    runner = runners[0]
    return {(runner, row_id(n)) for n in names}


def _spelled(command: str) -> str:
    """A command as runs compare it: its words, spaced once."""
    return " ".join(command.split())
