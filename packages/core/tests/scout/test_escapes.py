"""`request_exec` is a typed request a deterministic policy grants or
refuses; the command runs as written or not at all; its intent is durable before dispatch;
and recovery reconciles a started escape before anything re-runs it."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from scout_helpers import (
    FileReader,
    GraphNavigate,
    ListSink,
    ScriptedClient,
    answer,
)

from chickadee.approval.commands import parse_row, shipped
from chickadee.config import RungEndpoint
from chickadee.errors import HarnessBug, InfraError
from chickadee.executor.executor import Executor
from chickadee.executor.policy import prefilter
from chickadee.executor.reconcile import Landed, may_retry
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.graph.queries import Graph
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.journal import Journal, paths, project, read_events
from chickadee.journal.effects import EffectKind
from chickadee.process.supervisor import Completed, ProcessSupervisor
from chickadee.schema.journal import (
    EffectCompletedP,
    EffectIntendedP,
    EffectStartedP,
    EventKind,
    ExecP,
    ScoutEscapeP,
)
from chickadee.scout.escapes import (
    REQUEST_EXEC,
    EscapeRequest,
    Escapes,
    action_in_disguise,
    reconcile,
)
from chickadee.scout.findings import Checker
from chickadee.scout.loop import Scout
from chickadee.scout.rung import ScoutRung
from chickadee.scout.search import Search
from chickadee.scout.sink import JournalSink
from chickadee.scout.snapshot import build_snapshot
from chickadee.scout.tools import LocateCaps, ScoutTools
from chickadee.secrets.denylist import DEFAULT_PATTERNS, Denylist
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-09-30-5c0e7001"


class _Recording(ProcessSupervisor):
    """Records the argv it was handed and what the sink held at that moment."""

    def __init__(
        self, sink: ListSink | None = None, stdout: bytes = b"targets: help\n"
    ) -> None:
        self.sink = sink
        self.stdout = stdout
        self.argv: list[tuple[str, ...]] = []
        self.kinds_at_spawn: list[list[EventKind]] = []

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
        self.argv.append(argv)
        if self.sink is not None:
            self.kinds_at_spawn.append(self.sink.kinds())
        return Completed(self.stdout, b"", 0, False)


class _PowerCut(ProcessSupervisor):
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
        raise KeyboardInterrupt("the process died mid-escape")


def _escapes(
    repo: Path,
    supervisor: ProcessSupervisor,
    *,
    memo: dict[str, str] | None = None,
    budget: int = 3,
    output_bytes: int = 4096,
) -> Escapes:
    executor = Executor(backend=Passthrough(), supervisor=supervisor)
    return Escapes(
        executor=executor,
        cwd=repo,
        memo=memo if memo is not None else {},
        budget=budget,
        output_bytes=output_bytes,
        allow_unsandboxed=True,
    )


def _ask(command: str, reason: str = "the tools cannot see it") -> EscapeRequest:
    return EscapeRequest(command=command, reason=reason)


# -- the typed request -------------------------------------------------------------------


def test_the_request_is_validated_against_its_schema(tools: ScoutTools, sink: ListSink) -> None:
    """Through the scout's one path: the dispatcher's `Valid` checks the schema, and
    `EscapeRequest.of` what the schema cannot say (a command of whitespace). Nothing runs."""
    assert EscapeRequest.of("make help", "task names") == EscapeRequest(
        "make help", "task names"
    )
    for bad in (
        {"command": "make help"},
        {"reason": "no command"},
        {"command": "", "reason": "r"},
        {"command": "   ", "reason": "r"},
        {"command": "ls", "reason": "r", "cwd": "/"},
        {"command": ["ls"], "reason": "r"},
    ):
        answer = tools.call(REQUEST_EXEC, bad, sink, offer_exec=True)
        assert answer.text.startswith("INVALID_ARGUMENTS: request_exec"), bad
    assert sink.of(EventKind.SCOUT_ESCAPE) == [], "nothing ran"


# -- the grant/refuse matrix -------------------------------------------------------------


def test_already_known_answers_from_memory_and_spends_nothing(
    repo: Path, sink: ListSink
) -> None:
    sup = _Recording()
    esc = _escapes(repo, sup, memo={"make help": "targets: help test"})
    out = esc.request(_ask("make help"), sink)
    assert out.granted and out.known and out.output == "targets: help test"
    assert esc.used == 0 and sup.argv == []
    assert sink.kinds() == [EventKind.SCOUT_ESCAPE]
    (rec,) = sink.of(EventKind.SCOUT_ESCAPE)
    assert isinstance(rec, ScoutEscapeP) and rec.granted and rec.exec_seq == 0


def test_a_command_run_once_is_known_the_second_time(repo: Path, sink: ListSink) -> None:
    sup = _Recording()
    esc = _escapes(repo, sup)
    esc.request(_ask("make help"), sink)
    again = esc.request(_ask("make help"), sink)
    assert again.known and esc.used == 1 and len(sup.argv) == 1


def test_the_budget_is_per_locate_call(repo: Path, sink: ListSink) -> None:
    esc = _escapes(repo, _Recording(), budget=1)
    assert esc.request(_ask("make help"), sink).granted
    refused = esc.request(_ask("git log --oneline -3"), sink)
    assert not refused.granted and refused.refusal == "budget"


def test_a_call_sees_what_the_snapshot_promoted_after_it_was_made(
    repo: Path, sink: ListSink
) -> None:
    """The promoted commands are the session's, shared by every call's escapes: a rebuild of
    the snapshot that promotes a command is seen by a call made before it, so that call
    counts it toward promotion rather than a stale list."""
    session = _escapes(repo, _Recording(), budget=1)
    call = session.for_call()
    session.run_promoted(["make help"], sink)
    assert [command for command, _ in call.asked()] == ["make help"]


def test_a_rebuild_keeps_the_promoted_set_until_the_new_one_is_known(
    repo: Path, sink: ListSink
) -> None:
    """`run_promoted` runs the commands first and swaps the set last: while a command runs,
    a call reading `promoted` still sees the last set, never an emptied one."""
    seen: list[dict[str, str | None]] = []
    escapes: list[Escapes] = []

    class _Watching(_Recording):
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
            if escapes:
                seen.append(dict(escapes[0].promoted))
            return super().run(
                argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin, cancel=cancel
            )

    session = _escapes(repo, _Watching(), budget=1)
    escapes.append(session)
    session.run_promoted(["make help"], sink)
    before = dict(session.promoted)
    assert before
    seen.clear()
    session.run_promoted(["make help"], sink)
    assert seen == [before], "the set was emptied while the command ran"


def test_a_stopped_rebuild_changes_nothing(repo: Path, sink: ListSink) -> None:
    """The session's stop ends a rebuild between commands. A stopped rebuild is `None`
    and leaves the promoted set as it was: a command cut short did not fail, so it does not
    demote, and nothing after the stop runs."""
    stop = threading.Event()

    class _Stopping(_Recording):
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
            done = super().run(
                argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin, cancel=cancel
            )
            if len(self.argv) == 2:
                stop.set()  # the person stops while the rebuild's first command runs
            return done

    sup = _Stopping()
    session = _escapes(repo, sup)
    assert session.run_promoted(["make help"], sink) is not None
    before = dict(session.promoted)

    assert session.run_promoted(["make help", "make test"], sink, stop) is None
    assert session.promoted == before, "a stopped rebuild demoted what it cut short"
    assert [a[-1] for a in sup.argv] == ["make help"] * 2, "a command ran after the stop"
    assert session.run_promoted(["make test"], sink, stop) is None
    assert len(sup.argv) == 2, "a rebuild already stopped started a command"


def test_each_call_owns_its_budget_and_shares_the_session_s_memo(
    repo: Path, sink: ListSink
) -> None:
    """`for_call`: two `locate` calls, even interleaved, each spend their own budget
    and keep their own history; what one ran, the other recalls from the session's memo."""
    session = _escapes(repo, _Recording(), budget=1)
    first, second = session.for_call(), session.for_call()
    assert first.request(_ask("make help"), sink).granted
    assert second.request(_ask("make help"), sink).known, "recalled from the shared memo"
    assert second.request(_ask("git log --oneline -3"), sink).granted
    assert first.request(_ask("git log --oneline -1"), sink).refusal == "budget"
    assert [o.request.command for o in first.history] == ["make help", "git log --oneline -1"]
    assert session.used == 0 and session.history == [], "the session's own state is untouched"
    assert first.memo is session.memo is second.memo


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "uv pip install requests",
        "npm install left-pad",
        "make -n test && make install",
        "git commit -m x",
        "git checkout main",
        "curl https://example.com",
        "wget http://example.com",
        "sed -i s/a/b/ f.py",
        "touch x",
        "rm -rf build",
        "sudo ls",
        "echo x > out.txt",
        "ls; git push origin",
        "git -C sub push",
    ],
)
def test_a_mutation_in_disguise_is_refused(repo: Path, sink: ListSink, command: str) -> None:
    sup = _Recording()
    out = _escapes(repo, sup).request(_ask(command), sink)
    assert not out.granted and out.refusal == "mutation", out
    assert sup.argv == [], "nothing ran"


@pytest.mark.parametrize(
    "command",
    [
        "make help",
        "make -n test",
        "just --list",
        "git log --oneline -5",
        "ls -la",
        "cat Makefile",
        # A build runs sealed in a read-only repository: what it would write, the sandbox
        # refuses, so it is a question the scout may ask.
        "make",
        "cargo build",
        "docker build .",
        "python -c 'import x; print(x.__version__)'",
        "pytest -q 2>&1",
    ],
)
def test_read_only_questions_are_not_actions(command: str) -> None:
    assert action_in_disguise(command) is None


def test_the_scout_and_a_discovery_run_ask_the_same_question(
    repo: Path, sink: ListSink
) -> None:
    """Step 3 and the executor's DISCOVERY check are one rule, the table's `action`:
    what step 3 refuses as a mutation the executor would refuse too, so the scout's own
    refusal is the one it sees."""
    out = _escapes(repo, _Recording()).request(_ask("echo x > out.txt"), sink)
    assert out.refusal == "mutation" and "writes files" in out.why
    request = ExecRequest(
        command="echo x > out.txt", cwd=repo, policy=PolicyClass.DISCOVERY, timeout_s=5
    )
    assert (prefilter(request) or "").startswith("DISCOVERY is read-only")


def test_a_command_the_runs_table_classes_as_an_action_is_refused(
    repo: Path, sink: ListSink
) -> None:
    """Step 3 and the DISCOVERY pre-filter read the run's `[[commands]]` table, which
    the engine gives the executor, not the shipped one. `deploy.sh`, unknown to the shipped
    table, is a push once a row says so, and is refused before it runs."""
    row = parse_row({"prefix": ["deploy.sh"], "class": "push"})
    table = shipped().layered([("user", (row,))])
    sup = _Recording()
    executor = Executor(backend=Passthrough(), supervisor=sup, commands=table)
    escapes = Escapes(executor=executor, cwd=repo, allow_unsandboxed=True)
    out = escapes.request(_ask("deploy.sh --dry-run"), sink)
    assert out.refusal == "mutation" and "deploy.sh" in out.why
    assert sup.argv == [], "nothing ran"

    request = ExecRequest(
        command="deploy.sh --dry-run", cwd=repo, policy=PolicyClass.DISCOVERY, timeout_s=5
    )
    result = executor.run(request, operation_id="op-1")
    assert result.exit_code == 126 and result.stderr.startswith("DISCOVERY is read-only")
    assert sup.argv == [], "the executor's pre-filter refused it too"
    assert _escapes(repo, _Recording()).request(_ask("deploy.sh --dry-run"), sink).granted, (
        "the shipped table alone does not know it"
    )


def test_a_deny_listed_path_is_refused_by_name(repo: Path, sink: ListSink) -> None:
    out = _escapes(repo, _Recording()).request(_ask("cat .env"), sink)
    assert out.refusal == "deny_list" and out.why.startswith("SECRET_PATH")


def test_the_deny_list_is_the_executor_s_the_one_the_run_resolved(
    repo: Path, sink: ListSink
) -> None:
    """An entry the config added is refused by the policy, by name, before anything
    runs — never granted and then refused by the executor, which would journal a denied read
    as a granted escape, spend the budget, and memoize the refusal as the answer."""
    listed = Denylist(patterns=(*DEFAULT_PATTERNS, "config/*.yaml"))
    sup = _Recording()
    esc = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=sup, denylist=listed),
        cwd=repo,
        allow_unsandboxed=True,
    )
    out = esc.request(_ask("cat config/app.yaml"), sink)
    assert out.refusal == "deny_list" and not out.granted
    assert sup.argv == [] and esc.used == 0 and esc.memo == {}
    assert esc.asked() == ()


def test_a_run_cut_short_is_not_remembered_and_may_be_asked_again(
    repo: Path, sink: ListSink
) -> None:
    class _Slow(_Recording):
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
            super().run(argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin, cancel=cancel)
            return Completed(b"partial", b"", -9, True)

    esc = _escapes(repo, _Slow())
    first = esc.request(_ask("git log"), sink)
    assert first.timed_out and "git log" not in esc.memo
    second = esc.request(_ask("git log"), sink)
    assert not second.known and esc.used == 2


def test_a_non_zero_exit_is_an_answer_and_is_remembered(repo: Path, sink: ListSink) -> None:
    esc = Escapes(executor=Executor(backend=Passthrough()), cwd=repo, allow_unsandboxed=True)
    out = esc.request(_ask("grep -r no-such-text-anywhere pkg"), sink)
    assert out.granted and out.exit_code == 1
    assert esc.request(_ask("grep -r no-such-text-anywhere pkg"), sink).known


def test_a_refusal_carries_no_output_and_says_nothing_ran(repo: Path, sink: ListSink) -> None:
    out = _escapes(repo, _Recording()).request(_ask("pip install x"), sink)
    assert out.output == "" and out.operation_id is None
    assert out.render().startswith("REFUSED{reason: mutation")
    assert "did not run" in out.render()
    assert EventKind.EFFECT_INTENDED not in sink.kinds()
    (rec,) = sink.of(EventKind.SCOUT_ESCAPE)
    assert isinstance(rec, ScoutEscapeP) and not rec.granted and rec.exec_seq == 0


def test_a_granted_command_runs_exactly_as_written(repo: Path, sink: ListSink) -> None:
    command = "git  log   --format='%s'  -1"
    sup = _Recording()
    _escapes(repo, sup).request(_ask(command), sink)
    assert sup.argv == [("/bin/sh", "-c", command)], "never rewritten"


def test_the_intent_is_durable_before_dispatch(repo: Path, sink: ListSink) -> None:
    sup = _Recording(sink=sink)
    out = _escapes(repo, sup).request(_ask("make help"), sink)
    (at_spawn,) = sup.kinds_at_spawn
    assert at_spawn == [EventKind.EFFECT_INTENDED, EventKind.EFFECT_STARTED]
    assert sink.kinds() == [
        EventKind.EFFECT_INTENDED,
        EventKind.EFFECT_STARTED,
        EventKind.EXEC,
        EventKind.EFFECT_COMPLETED,
        EventKind.SCOUT_ESCAPE,
    ]
    intent = sink.of(EventKind.EFFECT_INTENDED)[0]
    assert isinstance(intent, EffectIntendedP)
    assert intent.effect_kind == EffectKind.SCOUT_ESCAPE and intent.idempotent
    assert intent.operation_id == out.operation_id
    assert "policy=discovery" in intent.preconditions


def test_a_real_discovery_command_runs_and_its_output_comes_back(
    repo: Path, sink: ListSink
) -> None:
    esc = Escapes(executor=Executor(backend=Passthrough()), cwd=repo, allow_unsandboxed=True)
    out = esc.request(_ask("git log --format=%s -1"), sink)
    assert out.granted and out.exit_code == 0 and "first commit" in out.output
    (exec_rec,) = sink.of(EventKind.EXEC)
    assert (
        isinstance(exec_rec, ExecP) and exec_rec.policy == "discovery" and not exec_rec.network
    )


def test_output_is_capped_at_the_scout_surface_s_byte_cap(repo: Path, sink: ListSink) -> None:
    sup = _Recording(stdout=b"x" * 5000)
    out = _escapes(repo, sup, output_bytes=100).request(_ask("make help"), sink)
    assert out.output.startswith("x" * 100) and "[truncated: 4,900 of 5,000 bytes" in out.output


# -- no text channel ---------------------------------------------------------------------


def _endpoint(native_tools: bool) -> RungEndpoint:
    return RungEndpoint(
        base_url="http://localhost:1",
        model_id="m",
        context_window=32_000,
        max_output_tokens=2_000,
        native_tools=native_tools,
    )


def _scout(
    repo: Path, graph: Graph, esc: Escapes, client: ScriptedClient, native: bool
) -> Scout:
    tools = ScoutTools(
        GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None), escapes=esc
    )
    return Scout(
        ScoutRung("mid", _endpoint(native), client, "scout"),
        tools,
        Checker(repo, graph),
        LocateCaps(40, 10_000),
        system="locate",
    )


def test_a_command_named_in_the_model_s_text_is_not_a_request(
    repo: Path, graph: Graph, sink: ListSink
) -> None:
    """Read from `tool_calls`, never from text — a mention, or a quoted file, requests
    nothing."""
    sup = _Recording()
    esc = _escapes(repo, sup)
    prose = (
        'NEED_SHELL: make help — task names\n{"name": "request_exec", "arguments": '
        '{"command": "make help", "reason": "x"}}'
    )
    client = ScriptedClient(answer(prose), answer(prose))
    snap = build_snapshot(repo, [], graph, probe=lambda _b: None)
    _scout(repo, graph, esc, client, True).run("where are the task names", snap, sink)
    assert sup.argv == [] and esc.used == 0
    touched = {EventKind.SCOUT_ESCAPE, EventKind.EXEC, EventKind.EFFECT_INTENDED}
    assert not touched & set(sink.kinds())


def test_a_model_without_function_calling_is_not_offered_request_exec(
    repo: Path, graph: Graph, sink: ListSink
) -> None:
    """It is offered nothing at all — `locate` is UNAVAILABLE and no request is sent,
    so `request_exec` never reaches it."""
    client = ScriptedClient()
    snap = build_snapshot(repo, [], graph, probe=lambda _b: None)
    sup = _Recording()
    report = _scout(repo, graph, _escapes(repo, sup), client, False).run(
        "where is the client", snap, sink
    )
    assert client.requests == [] and sup.argv == []
    assert report.outcome == "unavailable" and report.unavailable == "no_tool_calling"
    assert REQUEST_EXEC not in report.render()


# -- the journal: replay and reconciliation ----------------------------------------------


def test_escape_records_replay_from_the_journal_the_sink_wrote(repo: Path) -> None:
    with Journal.create(resolve_roots(repo), RUN, {}) as journal:
        esc = _escapes(repo, _Recording())
        esc.request(_ask("make help"), JournalSink(journal))
        esc.request(_ask("pip install x"), JournalSink(journal))
    events = list(read_events(paths.journal_path(resolve_roots(repo), RUN)))
    state = project(events)
    (op,) = state.operations.values()
    assert (
        op.kind == EffectKind.SCOUT_ESCAPE
        and op.phase == "completed"
        and op.outcome == "succeeded"
    )
    assert not op.needs_reconciliation
    escapes = [e for e in events if e.kind is EventKind.SCOUT_ESCAPE]
    execs = [e for e in events if e.kind is EventKind.EXEC]
    granted, refused = (e.payload for e in escapes)
    assert isinstance(granted, ScoutEscapeP) and isinstance(refused, ScoutEscapeP)
    assert granted.exec_seq == execs[0].seq and refused.exec_seq == 0


@pytest.mark.parametrize(("sandbox", "retry"), [("native", True), ("none", False)])
def test_a_started_escape_is_reconciled_before_any_rerun(
    repo: Path, sandbox: str, retry: bool
) -> None:
    """A crash between `effect_started` and `effect_completed` leaves an operation
    recovery must reconcile. Sealed, the escape landed nothing and may re-run; at
    `sandbox=none` nothing enforced the seal, so it may not."""
    with Journal.create(resolve_roots(repo), RUN, {}) as journal:
        esc = _escapes(repo, _PowerCut())
        with pytest.raises(KeyboardInterrupt):
            esc.request(_ask("make help"), JournalSink(journal))
    events = list(read_events(paths.journal_path(resolve_roots(repo), RUN)))
    state = project(events)
    (op,) = state.operations.values()
    assert op.phase == "started" and op.needs_reconciliation
    (intent,) = (e.payload for e in events if e.kind is EventKind.EFFECT_INTENDED)
    assert isinstance(intent, EffectIntendedP) and intent.operation_id == op.operation_id
    assert "sandbox=none" in intent.preconditions, "the suite has no backend"
    # The level the escape was dispatched at rides in the intent; the test varies it.
    at = intent.model_copy(
        update={
            "preconditions": tuple(
                f"sandbox={sandbox}" if p.startswith("sandbox=") else p
                for p in intent.preconditions
            )
        }
    )
    verdict = reconcile(at)
    assert verdict.landed is (Landed.NO if retry else Landed.UNKNOWN)
    allowed, _why = may_retry(at.idempotent, verdict)
    assert allowed is retry


class _Broken(ProcessSupervisor):
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
        raise RuntimeError("the supervisor broke mid-escape")


def test_an_exception_after_the_start_closes_the_escape_ambiguous(
    repo: Path, sink: ListSink
) -> None:
    """An exception other than `InfraError` may come after the
    command ran, so the escape is closed `ambiguous`, never left started, and the exception
    reaches the caller unchanged. An `InfraError` stays a `failed` refusal, and a kill still
    writes nothing (the power-cut test above)."""
    with pytest.raises(RuntimeError, match="broke mid-escape"):
        _escapes(repo, _Broken()).request(_ask("make help"), sink)
    assert sink.kinds() == [
        EventKind.EFFECT_INTENDED,
        EventKind.EFFECT_STARTED,
        EventKind.EFFECT_COMPLETED,
    ]
    (intent,) = sink.of(EventKind.EFFECT_INTENDED)
    (completed,) = sink.of(EventKind.EFFECT_COMPLETED)
    assert isinstance(intent, EffectIntendedP) and isinstance(completed, EffectCompletedP)
    assert completed.operation_id == intent.operation_id
    assert completed.outcome == "ambiguous" and completed.observed_surface == ()


def test_reconcile_refuses_an_operation_that_is_not_an_escape() -> None:
    other = EffectIntendedP(
        operation_id="bootstrap:x",
        effect_kind="bootstrap",
        approved_sha="0" * 64,
        preconditions=(),
        idempotent=True,
        declared_surface=(),
    )
    with pytest.raises(HarnessBug):
        reconcile(other)


# -- no sandbox: never unprotected unless the person allowed it --------------------------


class _Sealed:
    """A backend that reports an enforcing sandbox and wraps nothing — the suite has no real
    one, and what is under test is the scout's decision, not the kernel's."""

    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        return isolation.argv


def test_without_a_sandbox_an_escape_is_refused_and_nothing_runs(
    repo: Path, sink: ListSink
) -> None:
    """The command is the model's; with nothing enforcing the seal, it does not run."""
    sup = _Recording()
    esc = Escapes(executor=Executor(backend=Passthrough(), supervisor=sup), cwd=repo)
    assert not esc.sealed() and not esc.runnable()
    out = esc.request(_ask("git log -1"), sink)
    assert out.refusal == "no_sandbox" and not out.granted
    assert sup.argv == [] and esc.used == 0
    assert EventKind.EFFECT_INTENDED not in sink.kinds()


def test_with_a_sandbox_an_escape_runs_without_any_opt_in(repo: Path, sink: ListSink) -> None:
    sup = _Recording()
    esc = Escapes(executor=Executor(backend=_Sealed(), supervisor=sup), cwd=repo)
    assert esc.sealed() and esc.runnable()
    out = esc.request(_ask("make help"), sink)
    assert out.granted and sup.argv
    intent = sink.of(EventKind.EFFECT_INTENDED)[0]
    assert isinstance(intent, EffectIntendedP) and "sandbox=native" in intent.preconditions


def test_the_person_s_opt_in_runs_it_unsandboxed_and_the_record_says_so(
    repo: Path, sink: ListSink
) -> None:
    sup = _Recording()
    esc = Escapes(
        executor=Executor(backend=Passthrough(), supervisor=sup),
        cwd=repo,
        allow_unsandboxed=True,
    )
    out = esc.request(_ask("make help"), sink)
    assert out.granted and sup.argv
    intent = sink.of(EventKind.EFFECT_INTENDED)[0]
    assert isinstance(intent, EffectIntendedP) and "sandbox=none" in intent.preconditions
    assert reconcile(intent).landed is Landed.UNKNOWN, "recovery never re-runs it"


def test_without_a_sandbox_request_exec_is_not_offered(
    repo: Path, graph: Graph, sink: ListSink
) -> None:
    """Not offered at all: a tool that could only refuse costs its schema and the model's
    turns, and tells it nothing it can use."""
    esc = Escapes(executor=Executor(backend=Passthrough(), supervisor=_Recording()), cwd=repo)
    client = ScriptedClient(answer({"targets": []}))
    snap = build_snapshot(repo, [], graph, probe=lambda _b: None)
    _scout(repo, graph, esc, client, True).run("where is the client", snap, sink)
    (request,) = client.requests
    assert REQUEST_EXEC not in {t.name for t in request.tools}
    assert {"grep", "glob"} <= {t.name for t in request.tools}


def test_without_a_sandbox_a_promoted_command_is_not_run(repo: Path, sink: ListSink) -> None:
    sup = _Recording()
    esc = Escapes(executor=Executor(backend=Passthrough(), supervisor=sup), cwd=repo)
    assert esc.run_promoted(["make help"], sink) == {}
    assert sup.argv == [] and esc.asked() == ()


# -- D153 decision 8: every field of an escape's effect lines, pinned -----------


class _Exits(_Recording):
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
        return Completed(b"", b"no rule to make target\n", 2, False)


class _Unavailable(_Recording):
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
        raise InfraError("the spawn failed")


_ESCAPE_SUPERVISORS: dict[str, Callable[[], ProcessSupervisor]] = {
    "granted": _Recording,
    "exits": _Exits,
    "unavailable": _Unavailable,
    "raises": _Broken,
}


def _effect_fields(kinds_and_payloads: Sequence[tuple[EventKind, object]]) -> list[object]:
    """The escape's effect lines as their fields; the operation id is random per escape, so
    it is pinned by being the same on every line."""
    ops: dict[str, str] = {}
    out: list[object] = []
    for kind, p in kinds_and_payloads:
        if not isinstance(p, EffectIntendedP | EffectStartedP | EffectCompletedP):
            continue
        dumped = p.model_dump()
        op = ops.setdefault(str(dumped.pop("operation_id")), f"<op{len(ops) + 1}>")
        for key, value in dumped.items():
            if isinstance(value, str) and len(value) in (40, 64):
                dumped[key] = "<sha>"
        out.append((kind.value, op, dumped))
    return out


ESCAPE_PINS: dict[str, list[object]] = {
    "exits": [
        (
            "effect_intended",
            "<op1>",
            {
                "effect_kind": "scout_escape",
                "approved_sha": "<sha>",
                "preconditions": (
                    "policy=discovery",
                    "repository read-only",
                    "network off",
                    "sandbox=none",
                ),
                "idempotent": True,
                "declared_surface": (),
                "preimages_sha": None,
            },
        ),
        ("effect_started", "<op1>", {}),
        (
            "effect_completed",
            "<op1>",
            {
                "outcome": "failed",
                "observed_surface": (),
                "post_state_sha": None,
                "result_sha": None,
            },
        ),
    ],
    "granted": [
        (
            "effect_intended",
            "<op1>",
            {
                "effect_kind": "scout_escape",
                "approved_sha": "<sha>",
                "preconditions": (
                    "policy=discovery",
                    "repository read-only",
                    "network off",
                    "sandbox=none",
                ),
                "idempotent": True,
                "declared_surface": (),
                "preimages_sha": None,
            },
        ),
        ("effect_started", "<op1>", {}),
        (
            "effect_completed",
            "<op1>",
            {
                "outcome": "succeeded",
                "observed_surface": (),
                "post_state_sha": None,
                "result_sha": "<sha>",
            },
        ),
    ],
    "raises": [
        (
            "effect_intended",
            "<op1>",
            {
                "effect_kind": "scout_escape",
                "approved_sha": "<sha>",
                "preconditions": (
                    "policy=discovery",
                    "repository read-only",
                    "network off",
                    "sandbox=none",
                ),
                "idempotent": True,
                "declared_surface": (),
                "preimages_sha": None,
            },
        ),
        ("effect_started", "<op1>", {}),
        (
            "effect_completed",
            "<op1>",
            {
                "outcome": "ambiguous",
                "observed_surface": (),
                "post_state_sha": None,
                "result_sha": None,
            },
        ),
    ],
    "unavailable": [
        (
            "effect_intended",
            "<op1>",
            {
                "effect_kind": "scout_escape",
                "approved_sha": "<sha>",
                "preconditions": (
                    "policy=discovery",
                    "repository read-only",
                    "network off",
                    "sandbox=none",
                ),
                "idempotent": True,
                "declared_surface": (),
                "preimages_sha": None,
            },
        ),
        ("effect_started", "<op1>", {}),
        (
            "effect_completed",
            "<op1>",
            {
                "outcome": "failed",
                "observed_surface": (),
                "post_state_sha": None,
                "result_sha": None,
            },
        ),
    ],
}


@pytest.mark.parametrize("name", sorted(_ESCAPE_SUPERVISORS))
def test_an_escapes_effect_lines_are_pinned_through_both_sinks(
    repo: Path, sink: ListSink, name: str
) -> None:
    """The same payloads through the scout's list sink and through the run journal's sink:
    where the lines land never changes what they say."""
    with contextlib.suppress(RuntimeError):
        _escapes(repo, _ESCAPE_SUPERVISORS[name]()).request(_ask("make help"), sink)
    listed = _effect_fields([(k, p) for k, p in sink.records])
    assert listed == ESCAPE_PINS[name]
    with Journal.create(resolve_roots(repo), RUN, {}) as journal:
        with contextlib.suppress(RuntimeError):
            _escapes(repo, _ESCAPE_SUPERVISORS[name]()).request(
                _ask("make help"), JournalSink(journal)
            )
        journaled = _effect_fields([(ev.kind, ev.payload) for ev in journal.events()])
    assert journaled == listed
