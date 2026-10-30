"""An episode's language servers (Amendment 1): one
server per language kept between passes and told what changed, restarted once and then given up
for per-pass sessions, at most so many alive harness-wide, an idle one stopped, and the files a
change marked stale asked again in the background, never clearing a later mark."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest
from test_lsp import _languages, _unwrapped
from test_lsp_wired import PYRIGHT, _unsealed

from chickadee.box.spawn import index_resolver
from chickadee.graph import diagnostics
from chickadee.graph.diagnostics import DiagCoverage, Diagnostic, Verdict
from chickadee.graph.indexer import Indexer
from chickadee.graph.lsp import CheckerResolver
from chickadee.graph.queries import Graph
from chickadee.graph.servers import EpisodeServers, Recheck, Server, ServerSlots
from chickadee.process.supervisor import ProcessSupervisor
from chickadee.profiles import ProfileSet

FAKE = r"""
import json, os, sys

STARTS = "STARTS_PATH"
DIE_AFTER = DIE_AFTER_N
PUSH = PUSH_N
CHATTY = CHATTY_FLAG
if STARTS:
    with open(STARTS, "a") as f:
        f.write("start\n")
docs = {}
asked = 0


def send(payload):
    body = json.dumps(payload).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
    sys.stdout.buffer.flush()


def frames():
    buf = b""
    while True:
        end = buf.find(b"\r\n\r\n")
        if end >= 0:
            n = int(buf[:end].split(b":")[1])
            if len(buf) >= end + 4 + n:
                yield json.loads(buf[end + 4 : end + 4 + n])
                buf = buf[end + 4 + n :]
                continue
        chunk = os.read(0, 65536)
        if not chunk:
            return
        buf += chunk


for msg in frames():
    method, ident, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if ident is not None and method not in ("initialize", "shutdown"):
        asked += 1
        if DIE_AFTER and asked > DIE_AFTER:
            sys.exit(3)
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": ident, "result": {"capabilities": {}}})
    elif method == "textDocument/didOpen":
        docs[params["textDocument"]["uri"]] = params["textDocument"]["text"]
        if PUSH:
            # A blocking write: a client that stops reading leaves this server stuck here.
            uri = params["textDocument"]["uri"]
            notice = {"uri": uri, "diagnostics": [], "pad": "x" * PUSH}
            pushed = "textDocument/publishDiagnostics"
            send({"jsonrpc": "2.0", "method": pushed, "params": notice})
    elif method == "textDocument/didChange":
        docs[params["textDocument"]["uri"]] = params["contentChanges"][-1]["text"]
    elif method == "textDocument/didClose":
        docs.pop(params["textDocument"]["uri"], None)
    elif method == "textDocument/diagnostic":
        if CHATTY:
            import time
            while True:
                log = {"message": "."}
                send({"jsonrpc": "2.0", "method": "window/logMessage", "params": log})
                time.sleep(0.2)
        text = docs.get(params["textDocument"]["uri"], "")
        items = []
        if any("POISON" in held for held in docs.values()):
            pos = {"line": 0, "character": 0}
            where = {"start": pos, "end": pos}
            items.append({"range": where, "message": "poisoned", "severity": 1})
        if "BAD" in text:
            line = text[: text.index("BAD")].count("\n")
            pos = {"line": line, "character": 0}
            items.append({"range": {"start": pos, "end": pos}, "message": "bad", "severity": 1})
        send({"jsonrpc": "2.0", "id": ident, "result": {"kind": "full", "items": items}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": ident, "result": None})
    elif method == "exit":
        break
    elif ident is not None:
        send({"jsonrpc": "2.0", "id": ident, "result": None})
"""

UTIL = "def helper(x):\n    return x\n"
APP = "from util import helper\n\n\ndef go():\n    return helper(1)\n"


def _install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    die_after: int = 0,
    *,
    push: int = 0,
    chatty: bool = False,
) -> None:
    """The fake server on `PATH`, outside the repository, counting its starts in `starts`. Its
    settings are written into it: a server's environment is scrubbed (`scrub_env`)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    server = bin_dir / "fake-ls"
    body = (
        FAKE.replace("STARTS_PATH", str(tmp_path / "starts"))
        .replace("DIE_AFTER_N", str(die_after))
        .replace("PUSH_N", str(push))
        .replace("CHATTY_FLAG", str(chatty))
    )
    server.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    server.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(tmp_path, monkeypatch)


def _starts(tmp_path: Path) -> int:
    path = tmp_path / "starts"
    return len(path.read_text().splitlines()) if path.exists() else 0


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "r"
    root.mkdir(parents=True)
    (root / "util.py").write_text(UTIL, encoding="utf-8")
    (root / "app.py").write_text(APP, encoding="utf-8")
    return root


def _episode(
    root: Path, *, slots: ServerSlots | None = None, servers_max: int = 4
) -> tuple[Indexer, Graph, EpisodeServers]:
    servers = EpisodeServers(
        root.resolve(),
        ProcessSupervisor(),
        servers_max=servers_max,
        slots=slots or ServerSlots(),
    )
    resolver = CheckerResolver(
        root,
        ProcessSupervisor(),
        _languages("fake-ls --stdio"),
        wrap=_unwrapped,
        servers=servers,
    )
    ix = Indexer(root, resolver)
    ix.full_index()
    return ix, Graph(root, ix.conn), servers


def test_one_server_serves_every_pass_and_is_told_what_changed(
    tmp_path: Path, fake: None
) -> None:
    root = _repo(tmp_path)
    ix, graph, servers = _episode(root)
    try:
        assert graph.diagnostics_in("app.py").coverage is DiagCoverage.TOTAL
        (root / "app.py").write_text(APP + "BAD = 1\n", encoding="utf-8")
        ix.reindex([root / "app.py"])
        [problem] = graph.diagnostics_in("app.py").in_span
        assert problem.line == 6, "the server judged the new text it was sent"
        (root / "app.py").write_text(APP, encoding="utf-8")
        ix.reindex([root / "app.py"])
        assert not graph.diagnostics_in("app.py").in_span
        assert _starts(tmp_path) == 1, "no restart between passes"
    finally:
        servers.close()


def test_a_server_that_dies_is_restarted_once_then_given_up_for_per_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every start dies after one question: the episode's server fails, its restart fails,
    and the pass is asked of a session of its own, as every later pass is."""
    _install(tmp_path, monkeypatch, die_after=1)
    root = _repo(tmp_path)
    ix, _, servers = _episode(root)
    try:
        assert _starts(tmp_path) == 3, "the server, its one restart, then a per-pass session"
        ix.reindex([root / "app.py"])
        assert _starts(tmp_path) == 4, "per-pass from then on"
    finally:
        servers.close()


def test_at_the_cap_an_episode_uses_per_pass_sessions(tmp_path: Path, fake: None) -> None:
    slots = ServerSlots()
    first_root = _repo(tmp_path / "a")
    second_root = _repo(tmp_path / "b")
    _, _, first = _episode(first_root, slots=slots, servers_max=1)
    try:
        assert slots.alive == 1
        _, graph, second = _episode(second_root, slots=slots, servers_max=1)
        assert slots.alive == 1, "nothing waits, nothing fails"
        assert graph.diagnostics_in("app.py").coverage is DiagCoverage.TOTAL, "judged per pass"
        second.close()
    finally:
        first.close()
    assert slots.alive == 0


def test_an_idle_server_stops_and_starts_again_on_its_next_use(
    tmp_path: Path, fake: None
) -> None:
    root = _repo(tmp_path)
    now = [0.0]
    slots = ServerSlots()
    servers = EpisodeServers(
        root.resolve(), ProcessSupervisor(), slots=slots, clock=lambda: now[0]
    )
    resolver = CheckerResolver(
        root,
        ProcessSupervisor(),
        _languages("fake-ls --stdio"),
        wrap=_unwrapped,
        servers=servers,
    )
    ix = Indexer(root, resolver)
    try:
        ix.full_index()
        now[0] = 599.0
        servers.stop_idle()
        assert slots.alive == 1, "not idle yet"
        now[0] = 600.0
        servers.stop_idle()
        assert slots.alive == 0
        ix.reindex([root / "app.py"])
        assert slots.alive == 1 and _starts(tmp_path) == 2, "a stop is not a failure"
    finally:
        servers.close()


def test_a_change_s_dependents_are_rechecked_in_the_background(
    tmp_path: Path, fake: None
) -> None:
    root = _repo(tmp_path)
    ix, graph, servers = _episode(root)
    try:
        (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
        ix.reindex([root / "util.py"])
        servers.wait_rechecked()
        app = graph.diagnostics_in("app.py")
        assert app.coverage is DiagCoverage.TOTAL and app.stale_by is None
    finally:
        servers.close()


def test_closing_stops_the_servers_and_gives_their_slots_back(
    tmp_path: Path, fake: None
) -> None:
    slots = ServerSlots()
    _, _, servers = _episode(_repo(tmp_path), slots=slots)
    assert slots.alive == 1
    servers.close()
    servers.close()
    assert slots.alive == 0


def test_a_verdict_asked_before_a_later_change_does_not_clear_it(tmp_path: Path) -> None:
    """A2: a re-check asked at generation 2 lands after a change at generation 3 marked the file
    again; the mark stands."""
    root = _repo(tmp_path)
    ix = Indexer(root)
    ix.full_index()
    sha = Verdict("app.py", "a" * 64, "fake")
    diagnostics.record(ix.conn, (sha,))
    diagnostics.mark_stale(ix.conn, {"app.py": "util.py"}, 3)
    ix.conn.commit()
    clean = Verdict("app.py", "a" * 64, "fake", (Diagnostic(1, 1, "error", None, "x", "fake"),))
    assert not diagnostics.record_rechecked(ix.conn, clean, asked_at=2)
    assert Graph(root, ix.conn).diagnostics_in("app.py").stale_by == "util.py"
    assert diagnostics.record_rechecked(ix.conn, clean, asked_at=3)
    assert Graph(root, ix.conn).diagnostics_in("app.py").stale_by is None


@pytest.mark.skipif(not PYRIGHT, reason="pyright cannot run here")
def test_pyright_kept_for_the_episode_judges_each_change(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "app.py").write_text(APP.replace("helper(1)", "helper()"), encoding="utf-8")
    profiles = ProfileSet.load(repo=root)
    servers = EpisodeServers(root.resolve(), ProcessSupervisor(), slots=ServerSlots())
    resolver = index_resolver(root, _unsealed(require_sandbox=False), profiles, servers)
    ix = Indexer(root, resolver, profiles=profiles)
    graph = Graph(root, ix.conn)
    try:
        ix.full_index()
        assert [d.severity for d in graph.diagnostics_in("app.py").in_span] == ["error"]
        (root / "app.py").write_text(APP, encoding="utf-8")
        ix.reindex([root / "app.py"])
        fixed = graph.diagnostics_in("app.py")
        assert fixed.coverage is DiagCoverage.TOTAL and not fixed.in_span
    finally:
        servers.close()


# -- the review of #612 ----------------------------------------------------------------------


def test_reaching_the_cap_is_no_failure(tmp_path: Path, fake: None) -> None:
    """Two passes at the cap, then a free slot: the episode starts its server, not given up."""
    slots = ServerSlots()
    assert slots.take(1)
    root = _repo(tmp_path)
    ix, _, servers = _episode(root, slots=slots, servers_max=1)
    try:
        ix.reindex([root / "app.py"])
        assert _starts(tmp_path) == 2, "both passes per-pass, at the cap"
        slots.give()
        ix.reindex([root / "app.py"])
        ix.reindex([root / "app.py"])
        assert _starts(tmp_path) == 3, "one server for the episode once a slot is free"
    finally:
        servers.close()


def test_a_file_marked_again_while_queued_is_asked_at_the_newer_generation(
    tmp_path: Path,
) -> None:
    servers = EpisodeServers(tmp_path, ProcessSupervisor(), slots=ServerSlots())
    # A worker that is already there and never runs: what is queued stays queued.
    servers._worker = threading.Thread(target=lambda: None)  # pyright: ignore[reportPrivateUsage]
    profile = _languages("fake-ls --stdio").all()[0]
    server = Server(profile, ("fake-ls",), "fake-ls")
    servers.recheck([Recheck("b.py", server, 2)])
    servers.recheck([Recheck("b.py", server, 3)])
    assert servers._queued == {("b.py", "fake-ls"): 3}  # pyright: ignore[reportPrivateUsage]


def test_a_held_file_changed_outside_the_pass_is_sent_before_any_question(
    tmp_path: Path, fake: None
) -> None:
    """An open document overrides the disk for the server: `a.py` changed, and only `b.py` is
    reindexed; the server must judge `b.py` against the new `a.py`."""
    root = _repo(tmp_path)
    (root / "b.py").write_text("from app import go\n", encoding="utf-8")
    ix, graph, servers = _episode(root)
    try:
        assert not graph.diagnostics_in("b.py").in_span
        (root / "app.py").write_text(APP + "POISON = 1\n", encoding="utf-8")
        ix.reindex([root / "b.py"])
        assert [d.message for d in graph.diagnostics_in("b.py").in_span] == ["poisoned"]
    finally:
        servers.close()


def test_a_server_that_only_logs_is_out_of_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each answer's deadline is the whole wait, not each read: a server that logs every 0.2 s
    and never answers is given up at it, never kept to the session's 24 hours."""
    _install(tmp_path, monkeypatch, chatty=True)
    root = _repo(tmp_path)
    servers = EpisodeServers(
        root.resolve(), ProcessSupervisor(), slots=ServerSlots(), answer_timeout_s=1.0
    )
    resolver = CheckerResolver(
        root,
        ProcessSupervisor(),
        _languages("fake-ls --stdio"),
        wrap=_unwrapped,
        servers=servers,
        timeout_s=3,
    )
    started = time.monotonic()
    try:
        Indexer(root, resolver).full_index()
    finally:
        servers.close()
    assert time.monotonic() - started < 30


def test_a_server_that_pushes_while_it_is_sent_files_does_not_stall_the_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each `didOpen` makes the server push 200 kB with a blocking write; the client reads what
    it has written before each write of its own, so neither side waits on the other."""
    _install(tmp_path, monkeypatch, push=200_000)
    root = _repo(tmp_path)
    for n in range(12):
        (root / f"m{n}.py").write_text(f"X{n} = {n}\n", encoding="utf-8")
    resolver = CheckerResolver(
        root, ProcessSupervisor(), _languages("fake-ls --stdio"), wrap=_unwrapped, timeout_s=20
    )
    ix = Indexer(root, resolver)
    started = time.monotonic()
    ix.full_index()
    assert time.monotonic() - started < 15
    graph = Graph(root, ix.conn)
    assert all(
        graph.diagnostics_in(f"m{n}.py").coverage is DiagCoverage.TOTAL for n in range(12)
    )
