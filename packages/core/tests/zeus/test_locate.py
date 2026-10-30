"""`locate` from Zeus: the scout wired to what Zeus already has.

The session's `Locator` builds the scout's `Escapes` from the person's config — no escape
runs unprotected unless they said so — reads through Zeus's own graph and a reader of its own
for each `locate`, and records the scout's frozen payloads to the session log as `scout`
lines. The repository and its graph are real; the scout's model is scripted.
"""

from __future__ import annotations

import io
import json
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image
from repos import init_repo
from tool_args import checked, zeus_answer

from chickadee.client.request import ChatRequest, ImagePart, text_of
from chickadee.config import EngineConfig, RungEndpoint
from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import CallCancelled
from chickadee.context.session import Session
from chickadee.context.store import SessionLog
from chickadee.context.terms import Vision
from chickadee.errors import HarnessBug
from chickadee.executor.executor import Executor
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.isolation.none import Passthrough
from chickadee.process.supervisor import Completed, ProcessSupervisor
from chickadee.profiles.loader import load_profile
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema.adapter import Completion, StopReason, ToolCall, Usage
from chickadee.schema.journal import EffectIntendedP, EventKind, ScoutStartP, TurnP
from chickadee.scout.hints import Hints
from chickadee.scout.loop import Scout
from chickadee.scout.snapshot import Snapshot
from chickadee.zeus.tools.navigate import (
    ScoutLocator,
    SessionScoutSink,
    scout_locator,
)

from .zeus_helpers import ScriptedGraph

CLIENT = '''\
"""A courier."""


class Courier:
    """Carries messages."""

    def request(self, url: str) -> str:
        """Send one request."""
        return url
'''

QUESTION = "where are outgoing HTTP requests sent from"
TARGET = {"path": "pkg/client.py", "line_start": 4, "line_end": 9, "symbol": "Courier"}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "client.py").write_text(CLIENT, encoding="utf-8")
    return init_repo(root, commit="first")


@pytest.fixture
def graph(repo: Path) -> Iterator[Graph]:
    ix = Indexer(repo)
    ix.full_index()
    yield Graph(repo, ix.conn)
    ix.close()


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    opened = Session.create(tmp_path / "sessions", "S-locate")
    yield opened
    opened.close()


@pytest.fixture
def reader(session: Session, repo: Path) -> Reader:
    return Reader(
        SessionHome(session),
        HandleResolver(workspace=repo, session_blob=session.blob),
        vision=Vision("zeus"),
    )


def _config(
    *,
    allow_unsandboxed: bool = False,
    roles: dict[str, str] | None = None,
    blind: tuple[str, ...] = (),
) -> EngineConfig:
    """`blind` names the catalog models that cannot see (`supports_images = false`)."""
    model = {
        "base_url": "http://localhost:8000/v1",
        "model_id": "m",
        "context_window": 65_536,
        "max_output_tokens": 16_384,
    }
    models = {
        name: model | ({"supports_images": False} if name in blind else {})
        for name in ("frontier", "local-mid")
    }
    return EngineConfig.with_defaults(
        {
            "models": models,
            "rungs": {name: "frontier" for name in DEFAULT_RUNGS} | {"mid-param": "local-mid"},
            "spend": {"hard_cost_usd": 5.0},
            "sandbox": {"allow_unsandboxed_escapes": allow_unsandboxed},
            "roles": roles or {},
        }
    )


class ScriptedScout:
    """The scout rung's client, scripted: each `send` takes the next reply."""

    def __init__(self, *replies: Completion) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return self.replies.pop(0)

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing to release."""


def _usage() -> Usage:
    return Usage(prompt_tokens=100, completion_tokens=20)


def _calls(name: str, args: dict[str, Any]) -> Completion:
    return Completion(
        text="",
        usage=_usage(),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=(ToolCall(id="c0", name=name, arguments=args),),
    )


def _answer(findings: dict[str, Any]) -> Completion:
    return Completion(
        text=json.dumps(findings), usage=_usage(), latency_ms=1, stop_reason=StopReason.END_TURN
    )


def _locator(
    config: EngineConfig,
    session: Session,
    reader: Reader,
    repo: Path,
    graph: Graph,
    client: ScriptedScout,
    navigator: ScriptedGraph,
) -> ScoutLocator:
    def client_for(endpoint: RungEndpoint) -> ScriptedScout:
        assert endpoint.model_id == "m"
        return client

    return scout_locator(
        config,
        session,
        workspace=repo,
        graph=graph,
        navigator=navigator,
        reader=reader,
        executor=Executor(backend=Passthrough()),
        client_for=client_for,
    )


def test_the_escape_hatch_is_shut_without_a_sandbox_unless_the_person_opened_it(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """D134 items 1-2: the key is read where the scout's `Escapes` is built, and nowhere
    else. With no enforcing backend (`Passthrough`) and the default config, `request_exec`
    cannot run — so the loop does not offer it — and the person's opt-in opens it."""
    shut = _locator(_config(), session, reader, repo, graph, ScriptedScout(), ScriptedGraph())
    assert shut.escapes is not None
    assert shut.escapes.allow_unsandboxed is False
    assert not shut.escapes.runnable()

    opened = _locator(
        _config(allow_unsandboxed=True),
        session,
        reader,
        repo,
        graph,
        ScriptedScout(),
        ScriptedGraph(),
    )
    assert opened.escapes is not None
    assert opened.escapes.allow_unsandboxed is True
    assert opened.escapes.runnable()


def test_a_locate_reads_through_zeus_s_own_tools_and_answers_in_findings(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """The scout forks none of Zeus's tools. Its `symbol` reaches the navigator Zeus's
    own `symbol` reaches, and what Zeus reads is the scout's report — checked pointers."""
    navigator = ScriptedGraph()
    client = ScriptedScout(
        _calls("symbol", {"name": "Courier"}), _answer({"targets": [TARGET]})
    )
    locator = _locator(_config(), session, reader, repo, graph, client, navigator)
    session.record_assistant(
        "", [ToolCall(id="l1", name="locate", arguments={"question": QUESTION})]
    )
    reply = zeus_answer(
        session,
        reader,
        "l1",
        "locate",
        checked("locate", {"question": QUESTION}),
        locator=locator,
    )

    assert ("symbol", "Courier") in navigator.asked
    text = reply.text
    assert "## FINDINGS" in text
    assert "pkg/client.py" in text
    assert reply.ran
    # The question reached the scout's model as Zeus wrote it.
    assert QUESTION in "\n".join(text_of(m) for m in client.requests[0].messages)


def test_the_scout_reads_lines_zeus_already_holds(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """D140, the review's repro. Zeus has read `pkg/client.py`; its own read of the same lines
    is pointed at them. The scout is a separate conversation that has never seen them, so its
    `read` of those lines reaches the scout's model with the lines, not `already hold`."""
    session.record_user("look at the client")
    session.record_assistant(
        "",
        [ToolCall(id="z1", name="read", arguments={"path": "pkg/client.py", "lines": "1-10"})],
    )
    zeus_answer(
        session,
        reader,
        "z1",
        "read",
        checked("read", {"path": "pkg/client.py", "lines": "1-10"}),
    )
    session.record_assistant(
        "",
        [ToolCall(id="z2", name="read", arguments={"path": "pkg/client.py", "lines": "1-10"})],
    )
    again = zeus_answer(
        session,
        reader,
        "z2",
        "read",
        checked("read", {"path": "pkg/client.py", "lines": "1-10"}),
    )
    assert "already hold" in again.text, "Zeus holds the lines"

    client = ScriptedScout(
        _calls("read", {"path": "pkg/client.py", "lines": "1-10"}),
        _answer({"targets": [TARGET]}),
    )
    _locator(_config(), session, reader, repo, graph, client, ScriptedGraph()).locate(
        QUESTION, cancel=None
    )

    seen = "\n".join(text_of(m) for m in client.requests[1].messages)
    assert "def request(self, url: str) -> str:" in seen
    assert "already hold" not in seen


def test_what_a_locate_reads_is_gone_when_it_ends(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """Each `locate` has a reader of its own. What the scout opens never joins Zeus's
    session, and the next `locate` cannot reach it: the handle the first one's read showed the
    scout's model is expired for the second."""
    client = ScriptedScout(
        _calls("read", {"path": "pkg/client.py", "lines": "1-10"}),
        _answer({"targets": [TARGET]}),
    )
    locator = _locator(_config(), session, reader, repo, graph, client, ScriptedGraph())
    locator.locate(QUESTION, cancel=None)
    seen = "\n".join(text_of(m) for m in client.requests[1].messages)
    assert "def request(self, url: str) -> str:" in seen
    assert len(session.state.handles) == 0, "the scout's handles are not Zeus's"

    shown = re.findall(r"\[(h_[0-9a-f]{4,64})\]", seen)
    assert shown, "the first locate's read named its handle to the scout"
    client.replies.extend(
        [_calls("read", {"handle": shown[-1], "lines": "1-10"}), _answer({"targets": [TARGET]})]
    )
    locator.locate(QUESTION, cancel=None)
    said = "\n".join(text_of(m) for m in client.requests[-1].messages)
    assert f"HANDLE_EXPIRED{{handle: {shown[-1]}" in said
    assert len(session.state.handles) == 0


def _shot(repo: Path) -> None:
    image = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(image, format="PNG")
    (repo / "shot.png").write_bytes(image.getvalue())


def _images(request: ChatRequest) -> int:
    return sum(
        1
        for m in request.messages
        if not isinstance(m.content, str)
        for part in m.content
        if isinstance(part, ImagePart)
    )


@pytest.mark.writes_worktree("shot.png")
@pytest.mark.parametrize("sees", [True, False])
def test_the_scout_sees_as_its_own_rungs_model(
    session: Session, reader: Reader, repo: Path, graph: Graph, sees: bool
) -> None:
    """The scout's reader has the `scout` rung's vision, built from the
    model that rung names: a seeing model's reader returns the image, and a blind one's
    refuses it naming the scout's rung. Zeus's model (`frontier`) sees in both cases."""
    _shot(repo)
    client = ScriptedScout(
        _calls("read", {"path": "shot.png", "depth": "full"}),
        _answer({"targets": [TARGET]}),
    )
    config = _config(blind=() if sees else ("local-mid",))
    _locator(config, session, reader, repo, graph, client, ScriptedGraph()).locate(
        QUESTION, cancel=None
    )

    said = "\n".join(text_of(m) for m in client.requests[1].messages)
    refused = "NOT_SUPPORTED{capability: images, rung: mid-param}" in said
    assert refused is not sees
    assert _images(client.requests[1]) == (1 if sees else 0)


class Nesting(ScriptedScout):
    """A scout model that, on one of its calls, makes another `locate` happen first: the two
    calls interleave, as calls that run together would."""

    def __init__(self, *replies: Completion, at: int, inner: Callable[[], object]) -> None:
        super().__init__(*replies)
        self._at, self._inner, self._sends = at, inner, 0

    def send(self, request: ChatRequest) -> Completion:
        self._sends += 1
        if self._sends == self._at:
            self._inner()
        return super().send(request)


def test_a_locate_that_runs_inside_another_leaves_its_escapes_alone(
    tmp_path: Path, session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """Each call owns its escape budget and history. An inner `locate` that runs
    while the outer one is mid-way never resets the outer's: each call's own command is
    counted toward promotion once, and the other's is not counted for it."""
    outer, inner = "git log --oneline -1", "git log --oneline -2"
    hints = Hints(tmp_path / "hints.toml")
    holder: list[ScoutLocator] = []
    client = Nesting(
        _calls("request_exec", {"command": outer, "reason": "the graph has no history"}),
        _calls("request_exec", {"command": inner, "reason": "the graph has no history"}),
        _answer({"targets": [TARGET]}),
        _answer({"targets": [TARGET]}),
        at=2,
        inner=lambda: holder[0].locate("and the one before?", cancel=None),
    )
    locator = scout_locator(
        _config(allow_unsandboxed=True),
        session,
        workspace=repo,
        graph=graph,
        navigator=ScriptedGraph(),
        reader=reader,
        executor=Executor(backend=Passthrough()),
        client_for=lambda _endpoint: client,
        hints=hints,
    )
    holder.append(locator)
    locator.locate(QUESTION, cancel=None)

    counted = {cmd: hint.hits for cmd in (outer, inner) if (hint := hints.get(cmd))}
    assert counted == {outer: 1, inner: 1}


@pytest.mark.writes_worktree("shot.png")
def test_the_scout_sees_as_the_rung_its_role_was_moved_to(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """One lookup decides the scout's rung, for its model and its sight alike: moved onto
    Athena's rung (`roles = {locate = "high-param"}`), it sees as that rung's model does, here a
    blind one, while its own rung's model could see; the refusal names `high-param`."""
    _shot(repo)
    client = ScriptedScout(
        _calls("read", {"path": "shot.png", "depth": "full"}),
        _answer({"targets": [TARGET]}),
    )
    _locator(
        _config(roles={"locate": "high-param"}, blind=("frontier",)),
        session,
        reader,
        repo,
        graph,
        client,
        ScriptedGraph(),
    ).locate(QUESTION, cancel=None)

    said = "\n".join(text_of(m) for m in client.requests[1].messages)
    assert "NOT_SUPPORTED{capability: images, rung: high-param}" in said


class _Alone:
    """Notes whether anyone else was inside when a caller came in: each `inside()` stays long
    enough for another thread to arrive, if it can. The count is kept under a lock of its
    own, so two callers arriving together are both counted and the overlap is seen."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._inside = 0
        self.overlapped = threading.Event()
        self.entered: list[str] = []

    def inside(self, what: str) -> None:
        with self._guard:
            self._inside += 1
            if self._inside > 1:
                self.overlapped.set()
            self.entered.append(what)
        self.overlapped.wait(0.1)
        with self._guard:
            self._inside -= 1


def _together(call: Callable[[], object], times: int) -> list[object]:
    """`call` from `times` threads at once; what each returned, or raised."""
    out: list[object] = []

    def one() -> None:
        try:
            out.append(call())
        except BaseException as exc:
            out.append(exc)

    threads = [threading.Thread(target=one) for _ in range(times)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_calls_that_run_together_share_the_snapshot_and_hints_one_at_a_time(
    session: Session,
) -> None:
    """`locate` is `capped`, so calls may run together. The snapshot rebuild (which runs the
    promoted commands) and the hints are the session's, and are held one call at a time: no
    build, count or save starts while another is inside, so the promoted commands never run
    twice at once and no call's counts are lost."""
    alone = _Alone()

    def snapshot(cancel: threading.Event | None) -> Snapshot:
        alone.inside("build")
        return cast(Snapshot, object())

    hints = SimpleNamespace(
        record_run=lambda _asked: alone.inside("count"), save=lambda: alone.inside("save")
    )
    report = SimpleNamespace(render=lambda: "FINDINGS")
    escapes = SimpleNamespace(asked=lambda: ())
    scout = SimpleNamespace(run=lambda *_: report, tools=SimpleNamespace(escapes=escapes))
    locator = ScoutLocator(
        lambda: cast(Scout, scout),
        SessionScoutSink(session),
        snapshot,
        escapes=None,
        hints=cast(Hints, hints),
    )

    answers = _together(lambda: locator.locate(QUESTION, cancel=None), 2)

    assert answers == ["FINDINGS", "FINDINGS"]
    assert sorted(alone.entered) == ["build"] * 2 + ["count"] * 2 + ["save"] * 2
    assert not alone.overlapped.is_set(), "two calls were inside the session's share at once"


def test_a_locate_stopped_before_its_scout_starts_is_cancelled(
    session: Session, reader: Reader
) -> None:
    """Stopped before its search began, `locate` is `CANCELLED{tool: locate}` like any call the
    stop reached first: no snapshot is built and no scout starts."""
    stop = threading.Event()
    stop.set()

    def never() -> Any:
        raise AssertionError("a stopped locate started work")

    locator = ScoutLocator(never, SessionScoutSink(session), lambda _cancel: never())
    session.record_assistant(
        "", [ToolCall(id="l1", name="locate", arguments={"question": QUESTION})]
    )
    reply = zeus_answer(
        session,
        reader,
        "l1",
        "locate",
        checked("locate", {"question": QUESTION}),
        locator=locator,
        cancel=stop,
    )

    assert (reply.ran, reply.text) == (False, CallCancelled("locate").render())


def test_a_stopped_rebuild_answers_cancelled_and_keeps_no_snapshot(
    tmp_path: Path, session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """The stop reaches the build: stopped while a promoted command runs, the rebuild
    ends, `locate` is `CANCELLED{tool: locate}` with no scout started, and nothing is kept, so
    the next call builds afresh."""
    promoted = "git log --oneline -1"
    hints = Hints(tmp_path / "hints.toml")
    hints.record_run([(promoted, None)])
    assert hints.promoted() == (promoted,)
    stop = threading.Event()
    ran: list[str] = []

    class _StopsWhileRunning(ProcessSupervisor):
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
            ran.append(argv[-1])
            if len(ran) == 1:
                stop.set()  # the person stops while the first build runs the command
            return Completed(b"abc123 the last commit\n", b"", 0, False)

    client = ScriptedScout(_answer({"targets": [TARGET]}))
    locator = scout_locator(
        _config(allow_unsandboxed=True),
        session,
        workspace=repo,
        graph=graph,
        navigator=ScriptedGraph(),
        reader=reader,
        executor=Executor(backend=Passthrough(), supervisor=_StopsWhileRunning()),
        client_for=lambda _endpoint: client,
        hints=hints,
    )

    assert locator.locate(QUESTION, cancel=stop) == CallCancelled("locate")
    assert client.requests == [], "a scout started after the stop"

    stop.clear()  # the next turn
    assert "FINDINGS" in str(locator.locate(QUESTION, cancel=stop))
    assert ran == [promoted, promoted], "the stopped build was kept, so nothing was rebuilt"


def test_a_build_stopped_during_its_probes_is_not_kept(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """The stop reaches the version probes too: stopped while one runs, the build runs no
    further probe and is not kept, `locate` is `CANCELLED{tool: locate}`, and the next call
    builds afresh."""
    stop = threading.Event()
    asked: list[str] = []

    def probe(binary: str) -> str:
        asked.append(binary)
        if len(asked) == 1:
            stop.set()  # the person stops while the first build's first probe runs
        return f"{binary} 1.0"

    client = ScriptedScout(_answer({"targets": [TARGET]}))
    locator = scout_locator(
        _config(),
        session,
        workspace=repo,
        graph=graph,
        navigator=ScriptedGraph(),
        reader=reader,
        executor=Executor(backend=Passthrough()),
        client_for=lambda _endpoint: client,
        profiles=[load_profile("python")],
        probe=probe,
    )

    assert locator.locate(QUESTION, cancel=stop) == CallCancelled("locate")
    assert len(asked) == 1 and client.requests == []

    stop.clear()  # the next turn
    assert "FINDINGS" in str(locator.locate(QUESTION, cancel=stop))
    assert len(asked) > 2, "the stopped build was kept, so nothing was rebuilt"


def test_the_scouts_records_are_session_log_lines_that_replay(
    session: Session, reader: Reader, repo: Path, graph: Graph
) -> None:
    """For a `locate` from Zeus the sink is the session log. Each record is one
    `scout` line carrying the frozen journal payload as the scout emitted it, in order, and
    the log replays with them in it."""
    client = ScriptedScout(
        _calls("symbol", {"name": "Courier"}), _answer({"targets": [TARGET]})
    )
    locator = _locator(_config(), session, reader, repo, graph, client, ScriptedGraph())
    locator.locate(QUESTION, cancel=None)

    log = SessionLog.read(session.root, session.session_id)
    lines = [json.loads(log.payload(e)) for e in log.events() if e.kind == "scout"]
    events = [line["event"] for line in lines]
    assert events == ["scout_start", "scout_tool", "scout_report"]
    start = lines[0]["payload"]
    assert isinstance(start, dict)
    ScoutStartP.model_validate(start)

    root, sid = session.root, session.session_id
    session.close()
    with Session.open(root, sid) as again:
        assert len(again.state.others("scout")) == 3


def test_an_escape_s_intent_is_on_disk_before_its_command_runs(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sink returns an effect phase only once it is synced, so a crash between the
    intent and the dispatch leaves a started effect recovery can see. A tool record is
    flushed like any session line, as the run journal does the same records."""
    synced: list[int] = []
    monkeypatch.setattr(session, "sync", lambda: synced.append(1))
    sink = SessionScoutSink(session)

    sink.record(EventKind.SCOUT_START, ScoutStartP(rung="m", goal_sha="g", snapshot_sha="s"))
    assert synced == []
    sink.record(
        EventKind.EFFECT_INTENDED,
        EffectIntendedP(
            operation_id="scout:1",
            effect_kind="scout_escape",
            approved_sha="a",
            preconditions=(),
            idempotent=False,
            declared_surface=(),
        ),
    )
    assert synced == [1]


def test_the_sink_writes_only_what_the_scout_records(session: Session) -> None:
    """`check_record` guards the port: a kind outside the scout's closed set is a harness
    bug, raised before anything reaches the log."""
    sink = SessionScoutSink(session)
    with pytest.raises(HarnessBug, match="does not record"):
        sink.record(EventKind.TURN, TurnP(input_sha="x", mode="m", artifact_id="a"))
    assert not session.state.others("scout")
