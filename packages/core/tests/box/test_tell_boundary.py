"""A worker's round held at its next tool boundary for a message from Zeus: each row of §2.7.1's
table, through `box/worker.WorkerRun` with
the run's `box/tells.Tells`, and `continue`'s next round.

What these pin: a tell written while the model answers holds every call of that answer; a read
in flight finishes and the calls after it are answered `not run`; a sealed shell command is cut
and its step put back; a networked command finishes; a permission question is withdrawn, a
networked command's too, since nothing ran; no next model call is made and the round reports
`held` with no report; `continue` makes the next call on the conversation as it stands, with
the task's calls left; and the progress watchdog reads a held boundary as one stretch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from box_helpers import WORKER_RUN, delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo
from sandboxes import Scripted

from chickadee.box.progress import TaskProgress
from chickadee.box.tells import CUT, RoundToken, Tells, Told
from chickadee.boxctx.conversation import message_of
from chickadee.context.reader import Reader
from chickadee.journal.effects import EffectKind
from chickadee.process import Completed
from chickadee.schema import ToolCall
from chickadee.schema.journal import BoxMessageP, TaskReportedP
from chickadee.shell import ShellClaim
from chickadee.tools.loop import HELD_NOT_RUN
from chickadee.workspace import current_tree

APP = "def go() -> int:\n    return 1\n"
WAIT_S = 10.0


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


class _Told:
    """The session's `tell` lines for the run, as `Tells` reads them; `tell` writes one and
    wakes the run, as Zeus's `tell` does."""

    def __init__(self) -> None:
        self.lines: list[Told] = []
        self.tells: Tells | None = None

    def __call__(self, journal: Any) -> Tells:
        self.tells = Tells(WORKER_RUN, journal, lambda: tuple(self.lines))
        return self.tells

    def tell(self, text: str = "use argon2") -> None:
        assert self.tells is not None
        self.lines.append(Told(f"tell-{len(self.lines) + 1}", text, (3,), WORKER_RUN))
        self.tells.wake()


def _tool_answers(journal: Any) -> list[str]:
    """The worker's tool messages, in order."""
    out: list[str] = []
    for p in journal.projection().lines(BoxMessageP):
        if p.payload.side == "tool":
            content = message_of(journal.get_blob(p.payload.sha)).content
            out.append(content if isinstance(content, str) else str(content))
    return out


def _until(what: str, done: Any) -> None:
    deadline = time.monotonic() + WAIT_S
    while not done():
        if time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        time.sleep(0.01)


def _held(journal: Any, report: Any) -> None:
    """The round ended held: no report, and its `task_reported` line says so."""
    assert (report.stopped, report.report) == ("held", None)
    [line] = journal.projection().lines(TaskReportedP)
    assert (line.payload.stopped, line.payload.report_sha) == ("held", None)


def test_a_tell_while_the_model_answers_holds_every_call_of_the_answer(root: Path) -> None:
    told = _Told()

    def answers_while_told(request: object) -> Any:
        told.tell()
        return calls(
            call("edit", "c1", path="app.py", old="return 1", new="return 2"),
            call("read", "c2", path="app.py"),
        )

    worker, journal, rung = worker_rig(root, [answers_while_told], tells=told)
    report = worker.start("T1", "Make go return 2.", delegate(journal, root))

    _held(journal, report)
    assert _tool_answers(journal) == [HELD_NOT_RUN, HELD_NOT_RUN]
    assert (root / "app.py").read_text(encoding="utf-8") == APP
    assert len(rung.requests) == 1, "no next model call is made"
    assert told.tells is not None and told.tells.token.is_set(), "set until Athena reads it"


def test_a_read_in_flight_finishes_and_the_calls_after_it_are_not_run(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    told = _Told()
    read = Reader.read

    def told_while_reading(self: Reader, *args: Any, **kwargs: Any) -> Any:
        told.tell()
        return read(self, *args, **kwargs)

    monkeypatch.setattr(Reader, "read", told_while_reading)
    worker, journal, rung = worker_rig(
        root,
        [
            calls(
                call("read", "c1", path="app.py"),
                call("edit", "c2", path="app.py", old="return 1", new="return 2"),
            )
        ],
        tells=told,
    )
    report = worker.start("T1", "Make go return 2.", delegate(journal, root))

    _held(journal, report)
    first, second = _tool_answers(journal)
    assert "return 1" in first
    assert second == HELD_NOT_RUN
    assert (root / "app.py").read_text(encoding="utf-8") == APP
    assert len(rung.requests) == 1


class _Waits(Scripted):
    """A command that writes the worktree, a tell written while it runs, and then runs until
    its cancel is set or `WAIT_S` passes: a long test suite."""

    def __init__(self, root: Path, told: _Told) -> None:
        super().__init__(b"1 passed\n")
        self._root = root
        self._told = told
        self.cancelled = False

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        self.spawned += 1
        (self._root / "app.py").write_text("half written\n", encoding="utf-8")
        self._told.tell()
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                self.cancelled = True
                return Completed(b"", b"killed", -9, False)
            time.sleep(0.01)
        return Completed(self.stdout, b"", 0, False)


class _Grants:
    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        return True


@pytest.mark.writes_worktree("app.py")
def test_a_shell_command_in_flight_is_cut_and_its_step_put_back(root: Path) -> None:
    told = _Told()
    supervisor = _Waits(root, told)
    worker, journal, rung = worker_rig(
        root,
        [calls(call("shell", "c1", command="pytest -q"), call("read", "c2", path="app.py"))],
        tells=told,
        supervisor=supervisor,
    )
    report = worker.start("T1", "Run the tests.", delegate(journal, root))

    _held(journal, report)
    assert supervisor.cancelled, "reaped through its cancel"
    cut, after = _tool_answers(journal)
    assert CUT in cut and "the files are as before this call" in cut
    assert after == HELD_NOT_RUN
    assert (root / "app.py").read_text(encoding="utf-8") == APP, "the step is put back"
    [step] = [
        op
        for op in journal.projection().operations.values()
        if op.kind == EffectKind.WORKER_STEP.value
    ]
    assert step.completed is not None and step.completed.outcome == "failed"
    assert len(rung.requests) == 1


@pytest.mark.writes_worktree("app.py")
def test_a_networked_command_finishes(root: Path) -> None:
    """Its outcome would be unknown if cut: it runs to its end."""
    told = _Told()
    supervisor = _Waits(root, told)
    worker, journal, _ = worker_rig(
        root,
        [
            calls(
                call("shell", "c1", command="pip download six", network=True),
                call("read", "c2", path="app.py"),
            )
        ],
        tells=told,
        supervisor=supervisor,
        authority=_Grants(),
    )
    report = worker.start("T1", "Fetch it.", delegate(journal, root))

    _held(journal, report)
    assert not supervisor.cancelled
    ran, after = _tool_answers(journal)
    assert CUT not in ran
    assert after == HELD_NOT_RUN
    assert (root / "app.py").read_text(encoding="utf-8") == "half written\n", "kept: it ran"


class _Asks:
    """The person asked about a command: the tell written while the question waits, which
    waits until it is withdrawn."""

    def __init__(self, told: _Told) -> None:
        self._told = told
        self.withdrawn = False

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self._told.tell()
        assert cancel is not None
        _until("the question withdrawn", cancel.is_set)
        self.withdrawn = True
        return False


@pytest.mark.parametrize("network", [False, True])
def test_a_permission_question_is_withdrawn_and_the_call_answered_interrupted(
    root: Path, network: bool
) -> None:
    """A networked command's question too: nothing ran, so nothing is unknown."""
    told = _Told()
    authority = _Asks(told)
    supervisor = Scripted()
    shell = call("shell", "c1", command="pip install six", network=True)
    if not network:
        shell = call("shell", "c1", command="pip install six", sandbox=False)
    worker, journal, _ = worker_rig(
        root,
        [calls(shell)],
        tells=told,
        supervisor=supervisor,
        authority=authority,
    )
    report = worker.start("T1", "Install it.", delegate(journal, root))

    _held(journal, report)
    assert authority.withdrawn and supervisor.spawned == 0
    [cut] = _tool_answers(journal)
    assert CUT in cut


def test_continue_makes_the_next_call_on_the_conversation_as_it_stands(root: Path) -> None:
    told = _Told()

    def answers_while_told(request: object) -> Any:
        told.tell()
        return calls(call("edit", "c1", path="app.py", old="return 1", new="return 2"))

    worker, journal, rung = worker_rig(
        root,
        [
            answers_while_told,
            calls(call("edit", "c2", path="app.py", old="return 1", new="return 2")),
            answer("Done: go returns 2."),
        ],
        tells=told,
    )
    worker.start("T1", "Make go return 2.", delegate(journal, root))
    assert told.tells is not None
    told.tells.deliver(task="T1", round_=1)
    calls_left = worker.calls_left("T1")

    report = worker.cont("T1", None, 1)

    assert (report.stopped, report.round) == ("replied", 2)
    resumed = rung.requests[1].messages
    assert resumed[-1].role == "tool" and resumed[-1].content == HELD_NOT_RUN, (
        "no message of hers: the held answer is the last thing the worker read"
    )
    assert "return 2" in (root / "app.py").read_text(encoding="utf-8")
    assert worker.calls_left("T1") == calls_left - 1, "the held round's calls count"


def test_the_watchdog_reads_a_held_boundary_as_one_stretch(root: Path) -> None:
    """A tell never resets a threshold: the round `continue` opens goes on from the
    held one's calls; any other round starts from none."""
    tree, _ = current_tree(root, untracked_max_bytes=None)
    record = TaskProgress(root, tree)
    record.begin_round(1, 40)
    record.respond()
    record.observe(
        ToolCall(id="c1", name="read", arguments={"path": "app.py"}),
        APP,
        refused=False,
        step=None,
    )
    record.begin_round(2, 39, held=True)
    assert (len(record.now().calls), record.now().calls_left) == (1, 40)
    record.begin_round(3, 39)
    assert (len(record.now().calls), record.now().calls_left) == (0, 39)


def test_a_cut_is_never_read_as_a_later_call_s_on_the_same_token() -> None:
    """Athena's token spans her request: a command a tell cut, then, after the message, a
    networked one that runs whole with no question (no cuttable span) is not "cut"."""
    tells = threading.Event()
    token = RoundToken(None, tells)
    token.begin_call()
    with token.cuttable():
        tells.set()
        assert token.is_set() and token.tripped
    tells.clear()  # delivered
    token.begin_call()  # the next call, networked: it enters no cuttable span
    assert not token.tripped and not token.is_set()
