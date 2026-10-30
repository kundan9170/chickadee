"""Supervisor-session Resolver: confirmed locations are `resolved`; failure is not. Each
language's server and `languageId` are its profile's, run through the harness's wrap.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from chickadee.errors import InfraError
from chickadee.graph import conversation as conversation_mod
from chickadee.graph import lsp as lsp_mod
from chickadee.graph.edges import FALLBACK, RESOLVED
from chickadee.graph.indexer import Indexer
from chickadee.graph.lsp import (
    CheckerResolver,
    DefQuery,
    encode_dump,
    frame_message,
    parse_frames,
)
from chickadee.process.supervisor import ProcessSupervisor, scrub_env
from chickadee.profiles import LanguageProfile, ProfileSet, load_profile
from chickadee.profiles.model import LintSpec, LspSpec, Template
from chickadee.profiles.tools import find_tool

FAKE_LSP = r"""
import json
import os
import select
import sys

TARGET = "TARGET_URI_PLACEHOLDER"
KNOWN = {"initialize", "textDocument/definition", "shutdown", "exit"}


def write_frame(payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
    sys.stdout.buffer.flush()


def cut_frame(data):
    header_end = data.find(b"\r\n\r\n")
    sep = 4
    if header_end < 0:
        header_end = data.find(b"\n\n")
        sep = 2
    if header_end < 0:
        return None, data
    header = data[:header_end].decode("ascii", "replace")
    length = 0
    for line in header.splitlines():
        if line.lower().startswith("content-length:"):
            try:
                length = int(line.split(":", 1)[1].strip())
            except ValueError:
                length = 0
    start = header_end + sep
    end = start + length
    if end > len(data) or length <= 0:
        return None, data
    try:
        obj = json.loads(data[start:end])
    except json.JSONDecodeError:
        return None, data[end:]
    if not isinstance(obj, dict):
        return None, data[end:]
    return obj, data[end:]


def discard_pending():
    fd = sys.stdin.fileno()
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            return
        chunk = os.read(fd, 65536)
        if not chunk:
            return


def main():
    buf = bytearray()
    initialized = False
    fd = sys.stdin.fileno()
    while True:
        obj = None
        while obj is None:
            obj, rest = cut_frame(bytes(buf))
            if obj is not None:
                buf = bytearray(rest)
                break
            chunk = os.read(fd, 4096)
            if not chunk:
                return
            buf.extend(chunk)
        method = obj.get("method")
        ident = obj.get("id")
        if method == "initialize":
            write_frame({"jsonrpc": "2.0", "id": ident, "result": {"capabilities": {}}})
            initialized = True
            # Dump-ahead: remaining frames already sit in `buf` (and maybe the pipe).
            buf = bytearray()
            discard_pending()
            continue
        if method == "textDocument/definition" and initialized:
            write_frame(
                {
                    "jsonrpc": "2.0",
                    "id": ident,
                    "result": {
                        "uri": TARGET,
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 1},
                        },
                    },
                }
            )
        if method == "shutdown":
            write_frame({"jsonrpc": "2.0", "id": ident, "result": None})
        if method == "exit":
            return
        if method not in KNOWN and ident is not None:
            # A conforming server answers what it does not serve with an error, never silence.
            write_frame(
                {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": method}}
            )


if __name__ == "__main__":
    main()
"""


class ScriptedSession:
    def __init__(self, payload: bytes, *, timed_out: bool = False) -> None:
        self._data = payload
        self._i = 0
        self.timed_out = timed_out
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def read(self, n: int) -> bytes:
        if self.timed_out:
            raise TimeoutError
        chunk = self._data[self._i : self._i + n]
        self._i += len(chunk)
        return chunk

    def __enter__(self) -> ScriptedSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class ScriptedSupervisor:
    """Canned session bytes. Never spawns — the graph must not subprocess."""

    def __init__(self, payload: bytes, *, timed_out: bool = False) -> None:
        self.payload = payload
        self.timed_out = timed_out
        self.sessions: list[tuple[str, ...]] = []
        self.opened: list[ScriptedSession] = []

    def session(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_s: int,
        env: object = None,
        cancel: object = None,
    ) -> ScriptedSession:
        self.sessions.append(argv)
        session = ScriptedSession(self.payload, timed_out=self.timed_out)
        self.opened.append(session)
        return session

    def sent(self) -> list[dict[str, object]]:
        """Every frame the resolver wrote, across its sessions."""
        return [f for s in self.opened for f in parse_frames(b"".join(s.writes))]


def _with_server(
    profile: LanguageProfile, server: str | None, lint: str | None = None
) -> LanguageProfile:
    """`profile` with `server` as its language server and `lint` as its linter's, none by
    default: a test asks the servers it names, never the profile's own."""
    lsp = LspSpec(
        server=Template.parse(server) if server else None, language_id=profile.lsp.language_id
    )
    linter = LintSpec(server=Template.parse(lint) if lint else None)
    return profile.model_copy(update={"lsp": lsp, "lint": linter})


def _languages(server: str | None = "true --stdio") -> ProfileSet:
    """Python and JavaScript, each with `server` as its language server."""
    return ProfileSet.of(
        _with_server(load_profile(name), server) for name in ("python", "javascript")
    )


def _unwrapped(argv: tuple[str, ...], _cwd: Path) -> tuple[str, ...]:
    return argv


def _resolver(
    root: Path, supervisor: object, *, server: str | None = "true --stdio"
) -> CheckerResolver:
    return CheckerResolver(root, supervisor, _languages(server), wrap=_unwrapped)  # pyright: ignore[reportArgumentType]


def _init_and_definition(root: Path, rel: str, line_0: int, request_id: int = 1) -> bytes:
    uri = root.resolve().joinpath(rel).as_uri()
    init = frame_message({"jsonrpc": "2.0", "id": 0, "result": {"capabilities": {}}})
    definition = frame_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "uri": uri,
                "range": {
                    "start": {"line": line_0, "character": 0},
                    "end": {"line": line_0, "character": 1},
                },
            },
        }
    )
    return init + definition


def _pair(tmp_path: Path) -> Path:
    root = tmp_path / "r"
    root.mkdir()
    (root / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "app.py").write_text("def go():\n    return helper()\n", encoding="utf-8")
    return root


def _install_fake(root: Path, target_rel: str) -> Path:
    uri = root.resolve().joinpath(target_rel).as_uri()
    path = root / "fake-langserver"
    path.write_text(
        "#!/usr/bin/env python3\n" + FAKE_LSP.replace("TARGET_URI_PLACEHOLDER", uri),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _call_edge(ix: Indexer) -> tuple[str, str]:
    row = ix.conn.execute(
        "SELECT dst, edge_class FROM edges WHERE file = 'app.py' AND type = 'calls'"
    ).fetchone()
    return row["dst"], row["edge_class"]


def test_parse_frames_reads_content_length() -> None:
    framed = frame_message({"jsonrpc": "2.0", "id": 1, "result": None})
    frames = parse_frames(framed)
    assert frames[0]["id"] == 1


def test_a_confirmed_location_is_resolved(tmp_path: Path) -> None:
    root = _pair(tmp_path)
    supervisor = ScriptedSupervisor(_init_and_definition(root, "util.py", line_0=0))
    ix = Indexer(root, resolver=_resolver(root, supervisor))
    ix.full_index()
    assert _call_edge(ix) == ("util.helper", RESOLVED)
    assert supervisor.sessions, "the Resolver must go through the supervisor"


def test_the_handshake_skips_notifications_and_answers_the_servers_requests(
    tmp_path: Path,
) -> None:
    """pyright logs before it replies to `initialize`: taking the first frame as the reply
    resolved nothing, ever. A request the server makes of its client is answered."""
    root = _pair(tmp_path)
    log = frame_message(
        {
            "jsonrpc": "2.0",
            "method": "window/logMessage",
            "params": {"type": 3, "message": "hi"},
        }
    )
    ask = frame_message(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "workspace/configuration",
            "params": {"items": [{"section": "python"}, {"section": "python.analysis"}]},
        }
    )
    supervisor = ScriptedSupervisor(log + ask + _init_and_definition(root, "util.py", line_0=0))
    ix = Indexer(root, resolver=_resolver(root, supervisor))
    ix.full_index()
    assert _call_edge(ix) == ("util.helper", RESOLVED)
    answered = [f for f in supervisor.sent() if f.get("id") == 99]
    assert answered == [{"jsonrpc": "2.0", "id": 99, "result": [None, None]}]


def test_timeout_falls_back_to_name_match(tmp_path: Path) -> None:
    root = _pair(tmp_path)
    resolver = _resolver(root, ScriptedSupervisor(b"", timed_out=True))
    ix = Indexer(root, resolver=resolver)
    report = ix.full_index()
    assert report.files_indexed == 2
    assert _call_edge(ix)[1] == FALLBACK


def test_a_location_we_did_not_box_is_not_resolved(tmp_path: Path) -> None:
    """LSP pointing at a line with no indexed symbol is `None`, not a guessed `resolved`."""
    root = _pair(tmp_path)
    supervisor = ScriptedSupervisor(_init_and_definition(root, "app.py", line_0=99))
    ix = Indexer(root, resolver=_resolver(root, supervisor))
    ix.full_index()
    assert _call_edge(ix)[1] == FALLBACK


def test_a_local_is_not_an_edge_to_the_function_holding_it(tmp_path: Path) -> None:
    """A definition in a symbol's body, not on its declaration line, is a local: an edge to
    the enclosing function labelled `resolved` was wrong."""
    root = _pair(tmp_path)
    supervisor = ScriptedSupervisor(_init_and_definition(root, "app.py", line_0=1))
    ix = Indexer(root, resolver=_resolver(root, supervisor))
    ix.full_index()
    assert _call_edge(ix)[1] == FALLBACK


def test_no_server_named_or_installed_is_a_silent_none(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    for server in (None, "no-such-language-server --stdio"):
        supervisor = ScriptedSupervisor(b"")
        resolver = _resolver(root, supervisor, server=server)
        resolver.prefetch([DefQuery("a.py", 1, "x")], lookup=lambda _f, _l: None)
        assert resolver.definition("x", "a.py", 1) is None
        assert supervisor.sessions == []


def test_the_session_runs_the_profiles_server_through_the_wrap(tmp_path: Path) -> None:
    root = _pair(tmp_path)
    supervisor = ScriptedSupervisor(_init_and_definition(root, "util.py", line_0=0))
    wrapped: list[tuple[tuple[str, ...], Path]] = []

    def wrap(argv: tuple[str, ...], cwd: Path) -> tuple[str, ...]:
        wrapped.append((argv, cwd))
        return ("sandbox", *argv)

    resolver = CheckerResolver(root, supervisor, _languages(), wrap=wrap)  # pyright: ignore[reportArgumentType]
    Indexer(root, resolver=resolver).full_index()
    # By the path it was found at (`find_tool`), so the server's environment need not hold it.
    true = str(find_tool("true"))
    assert wrapped == [((true, "--stdio"), root.resolve())]
    assert supervisor.sessions == [("sandbox", true, "--stdio")]


def test_a_wrap_that_refuses_starts_no_server(tmp_path: Path) -> None:
    root = _pair(tmp_path)
    supervisor = ScriptedSupervisor(_init_and_definition(root, "util.py", line_0=0))

    def refuse(_argv: tuple[str, ...], _cwd: Path) -> tuple[str, ...]:
        raise InfraError("nothing seals here")

    resolver = CheckerResolver(root, supervisor, _languages(), wrap=refuse)  # pyright: ignore[reportArgumentType]
    ix = Indexer(root, resolver=resolver)
    ix.full_index()
    assert supervisor.sessions == []
    assert _call_edge(ix)[1] == FALLBACK


def test_one_session_per_language_asks_each_question_once(tmp_path: Path) -> None:
    supervisor = ScriptedSupervisor(
        frame_message({"jsonrpc": "2.0", "id": 0, "result": {"capabilities": {}}})
    )
    for rel in ("a.py", "b.py", "c.js"):
        (tmp_path / rel).write_text("x\n" * 3, encoding="utf-8")
    resolver = _resolver(tmp_path, supervisor)
    q = DefQuery("a.py", 1, "x")
    resolver.prefetch(
        [q, q, DefQuery("b.py", 2, "y"), DefQuery("c.js", 1, "z")], lookup=lambda _f, _l: None
    )
    assert len(supervisor.sessions) == 2, "one for Python, one for JavaScript"
    asked = [f for f in supervisor.sent() if f.get("method") == "textDocument/definition"]
    assert len(asked) == 3


def test_at_most_max_queries_are_asked(tmp_path: Path) -> None:
    supervisor = ScriptedSupervisor(
        frame_message({"jsonrpc": "2.0", "id": 0, "result": {"capabilities": {}}})
    )
    (tmp_path / "a.py").write_text("x\n" * 5, encoding="utf-8")
    resolver = _resolver(tmp_path, supervisor)
    resolver.max_queries = 2
    resolver.prefetch([DefQuery("a.py", i, "x") for i in range(1, 6)], lambda _f, _l: None)
    asked = [f for f in supervisor.sent() if f.get("method") == "textDocument/definition"]
    assert len(asked) == 2


def test_the_language_id_is_the_profiles(tmp_path: Path) -> None:
    (tmp_path / "view.jsx").write_text("go();\n")
    (tmp_path / "a.js").write_text("go();\n")
    supervisor = ScriptedSupervisor(
        frame_message({"jsonrpc": "2.0", "id": 0, "result": {"capabilities": {}}})
    )
    _resolver(tmp_path, supervisor).prefetch(
        [DefQuery("view.jsx", 1, "go"), DefQuery("a.js", 1, "go")], lambda _f, _l: None
    )
    opened = [f for f in supervisor.sent() if f.get("method") == "textDocument/didOpen"]
    ids: dict[str, object] = {}
    for frame in opened:
        doc = frame["params"]["textDocument"]  # pyright: ignore[reportIndexIssue, reportUnknownVariableType]
        ids[str(doc["uri"]).rsplit("/", 1)[-1]] = doc["languageId"]  # pyright: ignore[reportUnknownArgumentType]
    assert ids == {"view.jsx": "javascriptreact", "a.js": "javascript"}


def test_the_question_points_at_the_reference_not_its_first_spelling(tmp_path: Path) -> None:
    """`M(public_contracts=graph.public_contracts(k))`: the keyword spells the name first, and
    asking about it found a dataclass field. The parse's column names the call."""
    line = "    return M(public_contracts=graph.public_contracts(k))\n"
    (tmp_path / "m.py").write_text("def build(graph, k):\n" + line)
    at = line.index("graph.public_contracts") + len("graph.")
    assert lsp_mod._column(tmp_path, "m.py", 2, "public_contracts", at) == at  # pyright: ignore[reportPrivateUsage]
    assert lsp_mod._column(tmp_path, "m.py", 2, "public_contracts") == line.index("public")  # pyright: ignore[reportPrivateUsage]
    # LSP counts UTF-16 code units: a character outside the BMP before the name counts twice.
    (tmp_path / "u.py").write_text("s = '\U0001f600'; f()\n")
    col = "s = '\U0001f600'; ".encode().__len__()
    assert lsp_mod._column(tmp_path, "u.py", 1, "f", col) == len("s = '") + 2 + len("'; ")  # pyright: ignore[reportPrivateUsage]


def test_failed_prefetch_does_not_keep_a_stale_definition(tmp_path: Path) -> None:
    root = _pair(tmp_path)
    hit = ScriptedSupervisor(_init_and_definition(root, "util.py", line_0=0))
    resolver = _resolver(root, hit)
    q = DefQuery("app.py", 2, "helper")
    resolver.prefetch([q], lookup=lambda _f, _l: "util.helper")
    assert resolver.definition("helper", "app.py", 2) == "util.helper"
    resolver.supervisor = ScriptedSupervisor(b"", timed_out=True)
    resolver.prefetch([q], lookup=lambda _f, _l: "util.helper")
    assert resolver.definition("helper", "app.py", 2) is None


@pytest.mark.parametrize("module", [lsp_mod, conversation_mod], ids=["lsp", "conversation"])
def test_graph_lsp_does_not_import_subprocess_or_execrequest(module: ModuleType) -> None:
    """Neither the questions nor the conversation spawns: the session they are given does."""
    source = Path(str(module.__file__)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "subprocess" not in imported
    assert "chickadee.executor.types" not in imported
    assert "ExecRequest" not in source


def _python_id(_rel: str) -> str:
    return "python"


def test_live_session_resolves_when_dumped_stdin_cannot(tmp_path: Path) -> None:
    """The fake server ignores dump-ahead definition frames (T-2.17 letter protocol).

    That claim is what needs a real process, and it is deterministic. The indexing half
    below deliberately does **not** assert `RESOLVED`: `prefetch` swallows a slow session
    and records `fallback` by design, so pinning the resolved outcome
    behind a real spawn asserts that a timeout was not exceeded on the runner. It failed
    exactly that way on CI while the same commit passed the identical job twice concurrently.
    `test_a_confirmed_location_is_resolved` carries the resolved outcome against a
    `ScriptedSupervisor`, where it is a statement about behaviour.
    """
    root = _pair(tmp_path)
    fake = _install_fake(root, "util.py")
    queries: Sequence[DefQuery] = [DefQuery("app.py", 2, "helper")]
    blob = encode_dump(root, queries, _python_id)
    dumped = ProcessSupervisor().run(
        (str(fake), "--stdio"),
        cwd=root,
        env=scrub_env(),
        timeout_s=5,
        stdin=blob,
    )
    assert 1 not in [frame.get("id") for frame in parse_frames(dumped.stdout)]

    resolver = CheckerResolver(
        root, ProcessSupervisor(), _languages(f"{fake} --stdio"), wrap=_unwrapped
    )
    ix = Indexer(root, resolver=resolver)
    ix.full_index()
    dst, edge_class = _call_edge(ix)
    # Either verdict is correct; what must hold is that the edge exists and names the
    # right symbol whichever way the race went. `fallback` is the honest degradation, not
    # a wrong answer.
    assert edge_class in (RESOLVED, FALLBACK)
    assert dst == "util.helper"
