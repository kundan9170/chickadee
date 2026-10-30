"""Speaking LSP to one language server (D205 and its Amendment 1):
the JSON-RPC framing, the documents the server has been sent, the questions asked of it a
window at a time, and the verdict a server gives a file when asked.

One home for the conversation, whoever owns the process: an index pass's own session, started
and shut down within the pass (`graph/lsp.CheckerResolver`), and an episode's server, kept alive
between passes (`graph/servers.py`). It never spawns anything; the session it is given does.

**A verdict is asked for, never taken from a push (Amendment 1).** A server publishes
diagnostics unasked, and a publication can be provisional: pyright under load published an
empty list for a file with a type error, then began analysing it. The answer to
`textDocument/diagnostic` is computed before it is sent, so only a `full` report is a verdict.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast, runtime_checkable
from urllib.parse import unquote, urlparse

from chickadee.graph.diagnostics import Diagnostic, Severity, Verdict
from chickadee.paths import within
from chickadee.schema.ids import sha256_hex

__all__ = [
    "WINDOW",
    "ByteSession",
    "Channel",
    "LspConversation",
    "Request",
    "Settings",
    "Supervisor",
    "frame_message",
    "parse_frames",
]

_DRAIN_S: Final = 0.001
"""How long a drain waits for output that is not there yet: none to speak of."""

WINDOW: Final = 32
"""Questions sent ahead of their answers. Small, so the replies a window holds stay well inside
a pipe's buffer while the harness is writing the next question."""


class ByteSession(Protocol):
    def write(self, data: bytes) -> None: ...
    def read(self, n: int) -> bytes: ...


class Supervisor(Protocol):
    """How a server's process is had (`process.supervisor.ProcessSupervisor`): a session for its
    stdio, under `cancel` when one is given, so a read in flight ends at once when it is set."""

    def session(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout_s: int,
        env: Mapping[str, str] | None = None,
        cancel: threading.Event | None = None,
    ) -> AbstractContextManager[ByteSession]: ...


@runtime_checkable
class TimedSession(Protocol):
    """A session whose read can be bounded on its own: an episode's server lives long, and each
    answer still has a deadline (`process.supervisor.ProcessSession`)."""

    def read_within(self, n: int, timeout_s: float) -> bytes: ...


# -- framing ----------------------------------------------------------------------------------


def frame_message(payload: dict[str, object]) -> bytes:
    """One LSP JSON-RPC frame. Public so tests can script a server without a binary."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


def parse_frames(data: bytes) -> tuple[dict[str, object], ...]:
    """Split an LSP byte stream into JSON payloads. Truncated frames are dropped."""
    out: list[dict[str, object]] = []
    rest = data
    while True:
        obj, rest = _cut_frame(rest)
        if obj is None:
            break
        out.append(obj)
    return tuple(out)


def _cut_frame(data: bytes) -> tuple[dict[str, object] | None, bytes]:
    """One complete frame and the unconsumed tail. `(None, data)` if truncated."""
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
        raw: object = json.loads(data[start:end])
    except json.JSONDecodeError:
        return None, data[end:]
    obj = json_object(raw)
    return obj, data[end:]


def json_object(value: object) -> dict[str, object] | None:
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return None


def uri_of(root: Path, rel: str) -> str:
    return root.resolve().joinpath(rel).as_uri()


def rel_of(root: Path, uri: str) -> str | None:
    """The repository path a `file:` URI names, or None outside the repository."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    landed = within(root, Path(unquote(parsed.path)))
    if landed is None:
        return None
    return landed.relative_to(os.path.realpath(root)).as_posix()


Settings = Callable[[str], object]
"""A section's configuration, as the client answers `workspace/configuration` for it."""


class Channel:
    """One JSON-RPC object at a time from a live session, answering what the server asks of its
    client on the way, so a server that waits on its client (`workspace/configuration`,
    `client/registerCapability`) is not left waiting. A notification is read past."""

    def __init__(self, session: ByteSession, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings
        """`workspace/configuration`'s answer for a section (`LspSpec.configuration`)."""
        self._buf = bytearray()

    def reply(
        self, wanted: int | set[int], *, timeout_s: float | None = None
    ) -> dict[str, object] | None:
        """The next reply to a request in `wanted`. None at EOF or the deadline: the session's,
        or `timeout_s` for the whole wait when the session can bound a read (`TimedSession`).
        The notifications and server requests read past on the way never extend it: a server
        that logs while it never answers is out of time all the same."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            frame = self._next(deadline)
            if frame is None:
                return None
            method, fid = frame.get("method"), frame.get("id")
            if isinstance(method, str):
                if fid is not None:
                    self._session.write(client_reply(fid, frame.get("params"), self._settings))
                continue
            if fid in (wanted if isinstance(wanted, set) else {wanted}):
                return frame

    def drain(self) -> None:
        """Whatever the server has written so far, read into the buffer without waiting: what
        keeps its output pipe from filling while the harness is writing to it, which, with a
        server that blocks on a full pipe and stops reading, would leave both sides waiting."""
        if not isinstance(self._session, TimedSession):
            return
        while True:
            try:
                chunk = self._session.read_within(65536, _DRAIN_S)
            except TimeoutError:
                return
            if not chunk:
                return
            self._buf.extend(chunk)

    def _next(self, deadline: float | None) -> dict[str, object] | None:
        while True:
            obj, rest = _cut_frame(bytes(self._buf))
            if obj is not None:
                self._buf = bytearray(rest)
                return obj
            try:
                if deadline is not None and isinstance(self._session, TimedSession):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return None
                    chunk = self._session.read_within(4096, left)
                else:
                    chunk = self._session.read(4096)
            except TimeoutError:
                return None
            if not chunk:
                return None
            self._buf.extend(chunk)


def client_reply(fid: object, params: object, settings: Settings | None = None) -> bytes:
    """The client's answer to a server's request. `workspace/configuration` is answered one
    item per item asked: the section's `settings`, or `null` for one not configured; every
    other request `null`."""
    items = json_object(params)
    asked = items.get("items") if items is not None else None
    result: object = None
    if isinstance(asked, list):
        result = [_section(item, settings) for item in cast("list[object]", asked)]
    return frame_message({"jsonrpc": "2.0", "id": fid, "result": result})


def _section(item: object, settings: Settings | None) -> object:
    found = json_object(item)
    section = found.get("section") if found is not None else None
    return settings(section) if settings is not None and isinstance(section, str) else None


def initialize(root: Path) -> bytes:
    """`initialize`, saying the client asks for diagnostics (`textDocument.diagnostic`, LSP
    3.17), so a server that answers them only when asked for knows it may, and answers
    `workspace/configuration`, so a server asks for the profile's settings rather than taking
    its own defaults (`LspSpec.settings`)."""
    return frame_message(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": root.as_uri(),
                "capabilities": {
                    "textDocument": {"diagnostic": {}},
                    "workspace": {"configuration": True},
                },
            },
        }
    )


INITIALIZED: Final = frame_message({"jsonrpc": "2.0", "method": "initialized", "params": {}})
EXIT: Final = frame_message({"jsonrpc": "2.0", "method": "exit"})


def shutdown(fid: int) -> bytes:
    return frame_message({"jsonrpc": "2.0", "id": fid, "method": "shutdown", "params": None})


def read_file(root: Path, rel: str) -> bytes | None:
    try:
        return (root / rel).read_bytes()
    except OSError:
        return None


def did_open(root: Path, rel: str, language_id: str, data: bytes, version: int = 1) -> bytes:
    return frame_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": uri_of(root, rel),
                    "languageId": language_id,
                    "version": version,
                    "text": data.decode("utf-8", "replace"),
                }
            },
        }
    )


def _did_change(root: Path, rel: str, data: bytes, version: int) -> bytes:
    """The whole new text (full sync): no range arithmetic to get wrong."""
    return frame_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didChange",
            "params": {
                "textDocument": {"uri": uri_of(root, rel), "version": version},
                "contentChanges": [{"text": data.decode("utf-8", "replace")}],
            },
        }
    )


def _did_close(root: Path, rel: str) -> bytes:
    return frame_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/didClose",
            "params": {"textDocument": {"uri": uri_of(root, rel)}},
        }
    )


# -- the conversation ------------------------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """One question: its frame for an id, and what to do with the reply to it."""

    frame: Callable[[int], bytes]
    answered: Callable[[dict[str, object]], None]


@dataclass
class _Document:
    version: int
    sha256: str


class LspConversation:
    """What one server has been told, and the questions asked of it. Each file is sent once and
    then only when its bytes change, so a server kept alive analyses only what changed. Not
    thread-safe: its owner serialises it (`graph/servers.LiveServer`)."""

    def __init__(
        self,
        session: ByteSession,
        root: Path,
        language_id: Callable[[str], str],
        source: str,
        *,
        timeout_s: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._session = session
        self._root = root
        self._language_id = language_id
        self.source = source
        """The server's name, the verdicts' `source` where an item names none."""
        self._timeout_s = timeout_s
        """Each answer's deadline when the session can bound a read; None, the session's."""
        self._channel = Channel(session, settings)
        self._documents: dict[str, _Document] = {}
        self._next_id = 1

    def start(self) -> bool:
        """`initialize` and its reply, then `initialized`. False when it never answered."""
        self._send(initialize(self._root))
        if self._channel.reply(0, timeout_s=self._timeout_s) is None:
            return False
        self._send(INITIALIZED)
        return True

    def _send(self, data: bytes) -> None:
        """Every write, after what the server has written so far is read (`Channel.drain`)."""
        self._channel.drain()
        self._session.write(data)

    def sync(self, rel: str) -> str | None:
        """`rel` as it is on disk now, sent if the server has not seen these bytes: `didOpen`
        the first time, `didChange` with the whole text after. The sha256 of what the server
        holds, None when the file cannot be read (it is closed, if it was open)."""
        data = read_file(self._root, rel)
        if data is None:
            if rel in self._documents:
                self._send(_did_close(self._root, rel))
                del self._documents[rel]
            return None
        return self.send(rel, data)

    def send(self, rel: str, data: bytes) -> str:
        """`data` as `rel`'s text, sent if the server does not hold these bytes: the disk's
        (`sync`), or another's, such as the file at a task's start, which the next
        pass's `refresh` brings back to disk. The sha256 of what the server holds."""
        held = self._documents.get(rel)
        sha = sha256_hex(data)
        if held is None:
            self._send(did_open(self._root, rel, self._language_id(rel), data))
            self._documents[rel] = _Document(1, sha)
        elif held.sha256 != sha:
            held.version += 1
            held.sha256 = sha
            self._send(_did_change(self._root, rel, data, held.version))
        return sha

    def refresh(self) -> None:
        """Every file the server holds brought up to disk: an open document overrides the disk
        for the server, so one that changed outside a pass's files would otherwise be judged,
        and judge its dependents, as it was. Gone ones are closed."""
        for rel in list(self._documents):
            self.sync(rel)

    def ask(self, requests: Sequence[Request]) -> bool:
        """Each request asked, at most `WINDOW` unanswered at a time: a server answers in
        milliseconds once it has analysed a file, so one at a time spends the pass waiting on
        the pipe, and a bounded window keeps either side's pipe from filling while the other
        waits. Each reply is handled as it lands. False when an answer never came: the rest are
        left unanswered, and the caller treats the server as failed."""
        by_id: dict[int, Request] = {}
        queue = list(requests)
        while queue or by_id:
            while queue and len(by_id) < WINDOW:
                fid = self._allocate()
                request = queue.pop(0)
                self._send(request.frame(fid))
                by_id[fid] = request
            frame = self._channel.reply(set(by_id), timeout_s=self._timeout_s)
            if frame is None:
                return False
            by_id.pop(cast("int", frame["id"])).answered(frame)
        return True

    def verdict(self, rel: str, sha: str, into: list[Verdict]) -> Request:
        """`textDocument/diagnostic` for `rel` (Amendment 1): a `full` report is its verdict on
        the bytes the server holds (`sha`), appended to `into`; any other answer, none."""
        uri = uri_of(self._root, rel)

        def frame(fid: int) -> bytes:
            return frame_message(
                {
                    "jsonrpc": "2.0",
                    "id": fid,
                    "method": "textDocument/diagnostic",
                    "params": {"textDocument": {"uri": uri}},
                }
            )

        def answered(reply: dict[str, object]) -> None:
            report = json_object(reply.get("result"))
            items = report.get("items") if report is not None else None
            if report is None or report.get("kind") != "full" or not isinstance(items, list):
                return
            found = (diagnostic(i, self.source) for i in cast("list[object]", items))
            into.append(Verdict(rel, sha, self.source, tuple(d for d in found if d)))

        return Request(frame, answered)

    def stop(self) -> None:
        """`shutdown` and `exit`, unawaited: every answer that mattered has landed."""
        self._send(shutdown(self._allocate()))
        self._send(EXIT)

    def _allocate(self) -> int:
        fid, self._next_id = self._next_id, self._next_id + 1
        return fid


# -- a verdict's items ------------------------------------------------------------------------

_SEVERITY: Final[Mapping[int, Severity]] = {1: "error", 2: "warning", 3: "hint", 4: "hint"}
"""LSP's `DiagnosticSeverity` as the table spells it: information is below a warning, as a hint
is, and neither is listed in a read."""


def diagnostic(item: object, server: str) -> Diagnostic | None:
    """One LSP `Diagnostic`, 1-based; None when it names no position or no message. A severity
    left out is an error: a problem is surfaced rather than hidden below a warning."""
    obj = json_object(item)
    if obj is None:
        return None
    rng = json_object(obj.get("range"))
    start = json_object(rng.get("start")) if rng is not None else None
    line = start.get("line") if start is not None else None
    message = obj.get("message")
    if not isinstance(line, int) or not isinstance(message, str):
        return None
    character = start.get("character") if start is not None else None
    severity = obj.get("severity")
    code = obj.get("code")
    source = obj.get("source")
    return Diagnostic(
        line=line + 1,
        col=character + 1 if isinstance(character, int) else None,
        severity=_SEVERITY.get(severity, "error") if isinstance(severity, int) else "error",
        code=str(code) if isinstance(code, (str, int)) else None,
        message=message,
        source=source if isinstance(source, str) and source else server,
    )
