"""The worker's progress, from recorded facts: what the progress watchdog reads, never the
model's words,
never a model.

Every worker call is tagged, the highest that applies:

- **changed**: an edit, or a shell command that is not a test run
  (`executor/tests.is_test_command`), left the task's files in a state the task never had;
- **learned**: it showed something the task had not seen: a result never shown before in the
  task, compared after masking what varies by itself (`mask`);
- **nothing**: it was refused; its result was shown before; a read was answered with no lines,
  since they are held (`rereads.is_pointer`); or it put the files back in a state the task had
  (an undo).

What a call is comes from its tool's spec, never its name: a tool that acts on the
environment with a `command` may be a test run; a `read`-family tool names the file it read in
its spec's `paths`.

**The file state** is the task's change from its start tree, over the paths that count: each
path that differs, with its blob (`workspace.tree_change`), leaving out what lies under a cache
or build directory (`counts`) unless git tracked it when the task began. Paths git ignores are
in no step's tree. A change is credited to a call only for the paths its own step changed: a
test run's by-products, what Athena wrote between rounds, and what a step beside a running job
recorded move the state but are never the call's change, so they cannot hide a loop.

**One record per task, fed one way.** The worker tells it each model response (`respond`) and
each call as the model reads its answer (`observe`). In a process that did not run the task's
rounds so far, it is `rebuilt` by feeding it the same from the journal: the worker's messages as
journaled, each call's `tool_call` line (its refusal), each step's completed tree. So a resumed
round has the meters it had. A result is compared as the model read it, which on the journal
carries a hook's note on a batch's last answer: such a result may count as new once, which
makes the watchdog later, never earlier.

**Concurrency.** A record belongs to one task, fed on its worker's thread, one call at a time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from chickadee.box.facts import ran_of
from chickadee.box.steps import beside_jobs
from chickadee.box.tests_record import TestsRecord, Where
from chickadee.boxctx.conversation import message_of
from chickadee.client.request import Part, TextPart
from chickadee.context.rereads import is_pointer
from chickadee.edits.lines import split_lines
from chickadee.executor.tests import is_test_command, read_test_run
from chickadee.executor.types import ExecResult
from chickadee.hooks.payloads import (
    DiagnosticsSeen,
    ProgressCall,
    ProgressTag,
    Regression,
    TestRunSeen,
    WorkerProgress,
)
from chickadee.hooks.verdicts import LABEL_HEAD
from chickadee.hooks.watchdog import HEAD as WATCHDOG_HEAD
from chickadee.journal.effects import EffectKind
from chickadee.journal.metrics import python_note
from chickadee.journal.projection import RunProjection
from chickadee.profiles.loader import shipped_runners
from chickadee.profiles.model import LanguageProfile, TestRunner
from chickadee.schema.adapter import ToolCall
from chickadee.schema.journal import (
    BoxMessageP,
    EffectCompletedP,
    ExecP,
    TaskDelegatedP,
    TaskReportedP,
    ToolCallP,
)
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.spec import ToolSpec
from chickadee.workspace import DELETED, pinned_paths, tree_change

__all__ = [
    "CACHE_DIRS",
    "FAILURE_LINES",
    "StepSeen",
    "TaskProgress",
    "counts",
    "fingerprint",
    "mask",
    "named_files",
]

CACHE_DIRS: Final = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".coverage",
        ".hypothesis",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        ".cache",
        ".gradle",
        ".nyc_output",
        "htmlcov",
        "node_modules",
        "target",
        "build",
        "dist",
    }
)
"""Directories whose files a run, an install or an environment leaves behind: never a change,
unless git tracked the file when the task began."""

_CACHE_SUFFIXES: Final = (".egg-info",)
"""Directory names that end so are a build's by-products (`pip install -e.`)."""

FAILURE_LINES: Final = 12
"""How much of the last failing output is kept: its end, where the error usually is."""

_FAILURE_CHARS: Final = 1500
_NAMED_MAX: Final = 6
"""At most this many files named from a failure: a traceback can name dozens."""

_BRIEF_MAX: Final = 10
"""At most this many files counted from a brief: past 10, wandering's threshold is at its
ceiling (`WatchdogParams.wandering_max`)."""

_NOISE: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\x1b\[[0-9;?]*[A-Za-z]"), ""),
    (
        re.compile(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
        ),
        "<time>",
    ),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<time>"),
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<id>",
    ),
    # A step's operation id ends a shell answer's header (`ids.step_id`), and a call id is the
    # provider's: both are new every call.
    (re.compile(r"\bstep:\S+"), "<step>"),
    (re.compile(r"\b(?:call|toolu)_[A-Za-z0-9_-]+"), "<call>"),
    # The repetition rule's note counts the calls (`catalogue._count_note`): the same answer
    # sent again is not new because its count moved.
    (re.compile(r"(?m)^repetition\{[^}\n]*\}.*$"), "<repetition>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    (re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b"), "<hex>"),
    (re.compile(r"(?:/private)?/(?:tmp|var/folders)/\S+"), "<tmp>"),
    (re.compile(r"\bpid[ =:]*\d+", re.IGNORECASE), "pid <n>"),
    (
        re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|µs|us|ns|s|sec|secs|seconds|min|minutes)\b"),
        "<dur>",
    ),
)
"""What varies by itself between two runs of one thing, each with what it is masked to."""

_EXIT: Final = re.compile(r"(?m)^exit (-?\d+) ·")
"""A shell answer's header (`shell.py`): the command's exit code."""

_PATHLIKE: Final = re.compile(r"[A-Za-z0-9_./-]*[A-Za-z0-9_-]\.[A-Za-z0-9]+")

_ABOUT_ARGS: Final = ("path", "handle", "command", "pattern", "query", "symbol", "name", "glob")
"""The argument that says what a call is about, the first a call has."""

_HANDLE: Final = "handle"
"""How a `read` names a file it opened before: its handle stands for the file."""


@dataclass(frozen=True)
class StepSeen:
    """What a call's step left: its tree, the paths it changed, and whether a job ran beside
    it, so those paths may be the job's."""

    tree: str
    changed: tuple[str, ...]
    beside: bool = False


def counts(path: str) -> bool:
    """Whether a change to `path` is a change: not under a cache or build directory. A path
    git tracked at the task's start counts wherever it is (`TaskProgress`)."""
    return not any(
        part in CACHE_DIRS or part.endswith(_CACHE_SUFFIXES)
        for part in PurePosixPath(path).parts
    )


def mask(text: str) -> str:
    """`text` with what varies by itself masked: durations, times, ids, addresses, temporary
    paths, colour codes. Two runs of one failing command mask alike."""
    for pattern, said in _NOISE:
        text = pattern.sub(said, text)
    return text


def fingerprint(text: str) -> str:
    return hashlib.sha256(mask(text).encode("utf-8")).hexdigest()


def named_files(
    text: str, files: Iterable[str], root: Path | None = None, *, at_most: int = _NAMED_MAX
) -> tuple[str, ...]:
    """The repository files `text` names, in the order it names them, at most a few: only
    those in `files`, so a library's own files in a traceback are never named."""
    known = set(files)
    prefix = f"{root.as_posix().rstrip('/')}/" if root is not None else None
    found: dict[str, None] = {}
    for token in _PATHLIKE.findall(text):
        path = token.removeprefix("./")
        if prefix is not None and path.startswith(prefix):
            path = path[len(prefix) :]
        if path in known:
            found[path] = None
            if len(found) >= at_most:
                break
    return tuple(found)


def _text(content: str | Sequence[Part]) -> str:
    """An answer's text as its tool gave it: a hook's note on a batch's last answer
    (`hooks/loop.py`) is the harness's, not the tool's, and is left out, so a result is the
    same read live and from the journal, and the watchdog never quotes its own words back."""
    text = (
        content
        if isinstance(content, str)
        else "\n".join(p.text if isinstance(p, TextPart) else "<image>" for p in content)
    )
    lines = split_lines(text)
    noted = next(
        (i for i, line in enumerate(lines) if line.startswith((WATCHDOG_HEAD, LABEL_HEAD))),
        None,
    )
    return text if noted is None else "\n".join(lines[:noted])


def _tail(text: str) -> str:
    tail = "\n".join(split_lines(text.rstrip())[-FAILURE_LINES:])
    return tail if len(tail) <= _FAILURE_CHARS else tail[-_FAILURE_CHARS:]


def _target(spec: ToolSpec | None, args: Mapping[str, object]) -> str | None:
    """The file a read-family call read: the first its spec's `paths` name, or its handle."""
    if spec is None or spec.family != "read":
        return None
    for key in (*spec.paths, _HANDLE):
        found = args.get(key)
        if isinstance(found, str) and found:
            return found
    return None


def _test_run(
    spec: ToolSpec | None, args: Mapping[str, object], profiles: Sequence[LanguageProfile]
) -> bool:
    """Whether the call runs a profile's test runner: a command acting on the environment."""
    command = args.get("command")
    return (
        spec is not None
        and spec.mutates == "environment"
        and isinstance(command, str)
        and is_test_command(command, profiles)
    )


def _about(tool: str, args: Mapping[str, object], refused: bool) -> str:
    word = next((v for k in _ABOUT_ARGS if isinstance(v := args.get(k), str) and v.strip()), "")
    word = " ".join(word.split())
    if len(word) > 60:
        word = f"{word[:57]}..."
    lines = args.get("lines")
    said = " ".join(
        part
        for part in (tool, f"`{word}`" if word else "", lines if isinstance(lines, str) else "")
        if part
    )
    return f"{said} (refused)" if refused else said


def _key(tool: str, args: Mapping[str, object]) -> str:
    raw = json.dumps([tool, dict(args)], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


class TaskProgress:
    """One task's progress record. `registry_name` reads a call's name as the dispatcher does;
    `profiles` say which command is a test run and which language has a runner;
    `runners` read a run by what it printed; `can_run`, whether the worker may run
    commands (it has a shell)."""

    def __init__(
        self,
        root: Path,
        start_tree: str,
        *,
        brief: str = "",
        profiles: Sequence[LanguageProfile] = (),
        runners: Sequence[TestRunner] | None = None,
        can_run: bool = True,
        registry_name: Callable[[str], str | None] = lambda name: name,
    ) -> None:
        self._root = root
        self._start = start_tree
        self._profiles = tuple(profiles)
        self._brief = brief
        self._can_run = can_run
        self._tests = TestsRecord(shipped_runners() if runners is None else runners)
        self._changes: list[tuple[Where, tuple[str, ...]]] = []
        """Where each change was made in the task, and the files it changed."""
        self._diagnostics: DiagnosticsSeen | None = None
        self._name = registry_name
        self._calls_left: int | None = None
        self._files = frozenset(pinned_paths(root, start_tree))
        """The repository's files at the task's start: what a text may name."""
        self._brief_files = named_files(brief, self._files, root, at_most=_BRIEF_MAX)
        self._tree = start_tree
        """The tree the last call that stepped left."""
        self._state: frozenset[tuple[str, str]] = frozenset()
        self._states: set[frozenset[tuple[str, str]]] = {self._state}
        self._results: set[str] = set()
        self._opened: dict[str, None] = {}
        self._failure: str | None = None
        self._failure_call: int | None = None
        """Where in this round's calls the failure was, from 1; None for an earlier round's."""
        self._round: int | None = None
        self._response = 0
        self._calls: list[ProgressCall] = []

    @property
    def round(self) -> int | None:
        return self._round

    def begin_round(self, round_: int, calls_left: int | None, *, held: bool = False) -> None:
        """A new round, with the task's calls it may still make: its calls start from none, as
        the meters do. What the task has seen is kept. The round already running is not begun
        again. After a round `held` for a message the calls go on as one
        stretch: a tell never resets a threshold."""
        if round_ == self._round:
            return
        if held:
            self._round = round_
            return
        self._round, self._response, self._calls = round_, 0, []
        self._calls_left = calls_left
        self._failure_call = None

    def diagnosed(self, seen: DiagnosticsSeen) -> None:
        """The task's files against their start, as its stop read them: told to the
        hooks from now on."""
        self._diagnostics = seen

    def respond(self) -> None:
        """A model response: the calls observed next are its calls."""
        self._response += 1

    def observe(
        self,
        call: ToolCall,
        content: str | Sequence[Part],
        *,
        refused: bool,
        step: StepSeen | None,
        weight: int = 1,
        ran: ExecResult | None = None,
    ) -> ProgressCall:
        """`call`, answered with `content`, tagged. `step` is what the call's step left, None
        when it made no step; `weight` the tool calls it made (a Python block's several);
        `ran` the command its step ran, its whole output, which a test run is read from (the
        answer keeps a long output's head, where no summary is)."""
        tool = self._name(call.name) or call.name
        found = BUILTINS.get(tool)
        spec = found.spec if found is not None else None
        args = call.arguments
        text = _text(content)
        where = Where(self._round or 0, len(self._calls) + 1)
        run = self._run(spec, args, ran, where)
        moved = self._moved(
            step,
            test_run=run is not None or _test_run(spec, args, self._profiles),
            command=spec is not None and spec.mutates == "environment",
        )
        result = fingerprint(text)
        seen = result in self._results
        self._results.add(result)
        file = _target(spec, args)
        if file is not None:
            self._opened[file] = None
        tag: ProgressTag
        if moved is not None:
            tag = moved
        elif refused or (file is not None and is_pointer(text)) or seen:
            tag = "nothing"
        else:
            tag = "learned"
        if refused or _failed(text):
            self._failure, self._failure_call = _tail(text), len(self._calls) + 1
        changed = ()
        if tag == "changed" and step is not None:
            changed = tuple(p for p in step.changed if self._counts(p))
            self._changes.append((where, changed))
        made = ProgressCall(
            response=self._response,
            tool=tool,
            tag=tag,
            key=_key(tool, args),
            about=_about(tool, args, refused),
            file=file,
            weight=max(weight, 1),
            changed=changed,
            run=run[0] if run is not None else None,
            regressed=run[1] if run is not None else (),
        )
        self._calls.append(made)
        return made

    def _run(
        self,
        spec: ToolSpec | None,
        args: Mapping[str, object],
        ran: ExecResult | None,
        where: Where,
    ) -> tuple[TestRunSeen, tuple[Regression, ...]] | None:
        """The call's test run, read by what it printed and seen by the task's tests record;
        None when it ran none."""
        command = args.get("command")
        if spec is None or spec.mutates != "environment" or not isinstance(command, str):
            return None
        if ran is None:
            # Its output was not handed over: a test command is still a test run, of which
            # nothing can be said, so the unchecked meter is never fooled into a nudge.
            if not is_test_command(command, self._profiles):
                return None
            return TestRunSeen(outcome="partial", how="partial"), ()
        reading = read_test_run(command, ran, self._profiles)
        if reading is None:
            return None
        seen = self._tests.see(command, reading, where, self._changed_after)
        return (
            TestRunSeen(
                outcome=reading.outcome.value,
                how=reading.how,
                passed=reading.passed,
                failed=reading.failed,
                errors=reading.errors,
                unit=reading.unit,
                failing=reading.failing,
                fixed=seen.fixed,
                broken=seen.broken,
            ),
            seen.regressed,
        )

    def _changed_after(self, where: Where) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(p for at, paths in self._changes if at > where for p in paths)
        )

    def _runnable(self, written: Iterable[str]) -> bool:
        """Whether a test runner exists for a changed file's language, and the worker may run
        commands: whether going on without a test run means anything."""
        if not self._can_run:
            return False
        runs = {e for p in self._profiles if p.tests.runner is not None for e in p.extensions}
        return any(PurePosixPath(path).suffix in runs for path in written)

    def _counts(self, path: str) -> bool:
        return counts(path) or path in self._files

    def _moved(
        self, step: StepSeen | None, *, test_run: bool, command: bool
    ) -> ProgressTag | None:
        """What the call's own move of the files says, if it made one: a state never had is a
        change, one had before is nothing. A test run's, or a command's beside a running job
        (its changes may be the job's), is judged by its answer instead, as is a move the call
        did not make. An edit knows what it wrote, so a job beside it changes nothing here."""
        if step is None:
            return None
        if step.tree != self._tree:
            self._tree = step.tree
            self._state_now(step.tree)
        new = self._state not in self._states
        self._states.add(self._state)
        own = any(self._counts(p) for p in step.changed)
        if test_run or (command and step.beside) or not own:
            return None
        return "changed" if new else "nothing"

    def _state_now(self, tree: str) -> None:
        self._state = frozenset(
            (path, blob)
            for path, blob in tree_change(self._root, self._start, tree)
            if self._counts(path)
        )

    def now(self) -> WorkerProgress:
        """The round so far, as a hook is told it."""
        written = {p for p, blob in self._state if blob != DELETED}
        failure = self._failure
        return WorkerProgress(
            calls=tuple(self._calls),
            brief_files=self._brief_files,
            opened=tuple(self._opened),
            last_failure=failure,
            failure_call=self._failure_call,
            failure_files=()
            if failure is None
            else named_files(failure, self._files | written, self._root),
            calls_left=self._calls_left,
            runnable=self._runnable(written),
            tests=self._tests.seen(self._brief, {p for p, _ in self._state}),
            diagnostics=self._diagnostics,
        )

    @classmethod
    def rebuilt(
        cls,
        run: RunProjection,
        task: str,
        blob: Callable[[str], bytes],
        root: Path,
        *,
        profiles: Sequence[LanguageProfile] = (),
        call_cap: int | None = None,
        registry_name: Callable[[str], str | None] = lambda name: name,
        on_round: Callable[[int, WorkerProgress], None] | None = None,
    ) -> TaskProgress:
        """`task`'s record from its lines: fed its responses and calls as the worker fed it
        live, so a round resumed in another process keeps its meters. `call_cap` is the
        task's, each round told what was left of it when it began. `on_round` is told each
        round as it stood at its end, the last one included (`box/progress_replay.py`)."""
        lines = run.tasks.get(task, ())
        delegated = next(
            (p.payload for p in lines if isinstance(p.payload, TaskDelegatedP)), None
        )
        if delegated is None:
            raise ValueError(f"{task} was never delegated; it has no progress")
        made = cls(
            root,
            delegated.start_tree,
            brief=blob(delegated.brief_sha).decode("utf-8", "replace"),
            profiles=profiles,
            registry_name=registry_name,
        )
        asked: dict[str, ToolCall] = {}
        told: list[ToolCallP] = []
        step: StepSeen | None = None
        ran: ExecResult | None = None
        execs = {str(p.payload.operation_id): p.payload for p in run.lines(ExecP)}
        calls = 0
        """The task's calls before this line, as its call cap counts them (`task_counters`)."""
        was_held = False
        """Whether the last round ended held for a message: the next goes on from it."""
        for placed in lines:
            line = placed.payload
            if isinstance(line, TaskReportedP):
                was_held = line.stopped == "held"
            elif isinstance(line, ToolCallP):
                if line.role == "worker" and not python_note(line):
                    told.append(line)
                    calls += 1
            elif isinstance(line, EffectCompletedP):
                op = run.operations.get(line.operation_id)
                if op is not None and op.kind == EffectKind.WORKER_STEP.value:
                    held = op.intended.preconditions if op.intended is not None else ()
                    step = _step_of(line, held) or step
                    found = execs.get(str(line.operation_id))
                    ran = ran_of(found, blob) if found is not None else ran
            elif isinstance(line, BoxMessageP) and line.role == "worker":
                if placed.round is not None and placed.round != made.round:
                    if on_round is not None and made.round is not None:
                        on_round(made.round, made.now())
                    left = None if call_cap is None else max(call_cap - calls, 0)
                    made.begin_round(placed.round, left, held=was_held)
                message = message_of(blob(line.sha))
                call = asked.get(message.tool_call_id or "")
                if message.role == "tool" and call is not None and told:
                    # A call the worker made: its `tool_call` line came first. An answer with
                    # none (a call after the end, one a crash left unanswered) was never run.
                    made.observe(
                        call,
                        message.content,
                        refused=any(t.denied_reason is not None for t in told),
                        step=step,
                        weight=len(told),
                        ran=ran,
                    )
                elif message.role == "assistant":
                    made.respond()
                    asked = {c.id: c for c in message.tool_calls}
                told, step, ran = [], None, None
        if on_round is not None and made.round is not None:
            on_round(made.round, made.now())
        return made


def _step_of(line: EffectCompletedP, preconditions: Sequence[str]) -> StepSeen | None:
    """A completed step as the record reads it, None for one that left no tree. A call with
    several steps (a Python block's) is read by its last, as the worker reads it live
    (`StepRecord.last`)."""
    if not line.post_state_sha:
        return None
    return StepSeen(
        line.post_state_sha, tuple(line.observed_surface), bool(beside_jobs(preconditions))
    )


def _failed(text: str) -> bool:
    """Whether a shell answer says its command failed: its header's exit code is not 0."""
    found = _EXIT.search(text)
    return found is not None and found.group(1) != "0"
