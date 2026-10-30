"""Plain helpers for the scout suite (a uniquely named module, never `conftest`).

The fakes stand where the scout takes a port: the sink, the model client, and the shared
tools `T-1.11` builds. The graph, the files and the git repository are real.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from repos import init_repo

from chickadee.client.request import ChatRequest
from chickadee.context.handle import Handle, Shape, content_sha, handle_id
from chickadee.context.results import ToolResult
from chickadee.graph.queries import Graph
from chickadee.paths import Place
from chickadee.schema.adapter import Completion, StopReason, ToolCall, Usage
from chickadee.schema.journal import EventKind, P

CLIENT = '''\
"""An HTTP client."""

DEFAULT_TIMEOUT = 30


class HttpClient:
    """Talks HTTP."""

    def request(self, url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
        """Send one request."""
        return url
'''

FETCHER = '''\
"""Fetches things."""
from pkg.net.client import HttpClient


def fetch(url: str) -> str:
    """Fetch one."""
    return HttpClient().request(url)
'''

TEST_CLIENT = """\
from pkg.net.client import HttpClient


def test_request():
    assert HttpClient().request("u") == "u"
"""

PYPROJECT = """\
[project]
name = "demo"
version = "0"
dependencies = ["httpx>=0.27", "pydantic"]
"""

MAKEFILE = "help:\n\t@echo targets: help test\n"


def make_repo(root: Path) -> Path:
    """A small repository with code the graph indexes and files it does not."""
    files = {
        "pkg/__init__.py": "",
        "pkg/net/__init__.py": "",
        "pkg/net/client.py": CLIENT,
        "pkg/api/__init__.py": "",
        "pkg/api/fetcher.py": FETCHER,
        "pkg/__main__.py": "from pkg.api.fetcher import fetch\n",
        "tests/test_client.py": TEST_CLIENT,
        "config/app.yaml": "http:\n  timeout: 30\n",
        "pyproject.toml": PYPROJECT,
        "Makefile": MAKEFILE,
        ".gitignore": "ignored_dir/\n",
        "ignored_dir/hidden.py": "SECRET_WORD = 1\n",
        ".env": "timeout_token=hunter2\n",
    }
    return init_repo(root, files, commit="first commit")


class ListSink:
    """Every record, in order, with a sequence number — durable the moment it is listed."""

    def __init__(self) -> None:
        self.records: list[tuple[EventKind, P]] = []

    def record(self, kind: EventKind, payload: P) -> int:
        self.records.append((kind, payload))
        return len(self.records)

    def kinds(self) -> list[EventKind]:
        return [k for k, _ in self.records]

    def of(self, kind: EventKind) -> list[P]:
        return [p for k, p in self.records if k is kind]


@dataclass
class GraphNavigate:
    """A stand-in for Zeus's navigator (the `navigator` port), over the real graph."""

    graph: Graph
    calls: list[str] = field(default_factory=list[str])

    def atlas(self) -> str:
        raise AssertionError("the scout is never offered atlas (14-scout.md §4)")

    def symbol(self, name: str) -> str:
        self.calls.append(f"symbol {name}")
        found = self.graph.where_defined(name)
        rows = [f"{p.path}:{p.line_start}-{p.line_end} {p.symbol}" for p in found]
        return "\n".join(rows) or f"UNRESOLVED{{query: {name!r}}}"

    def neighbors(self, target: str, depth: int = 1) -> str:
        self.calls.append(f"neighbors {target}")
        rows = [
            f"{e.file}:{e.line} → {e.dst} ({e.edge_class})" for e in self.graph.callers(target)
        ]
        return "\n".join(rows) or "no callers"

    def impact(self, symbols: Sequence[str]) -> str:
        self.calls.append(f"impact {','.join(symbols)}")
        return ", ".join(self.graph.blast_radius(symbols)) or "no impact"

    def tests_for(self, paths: Sequence[str]) -> str:
        self.calls.append(f"tests_for {','.join(paths)}")
        return "no test file"


def _result(path: str, text: str) -> ToolResult:
    """What the Reader returns (the `files` port's answer), with `text` as its content."""
    data = text.encode("utf-8")
    sha = content_sha(data)
    handle = Handle(
        id=handle_id("file", path=path, run_id=None, sha=sha),
        kind="file",
        shape=Shape(bytes=len(data), lines=None, symbols=None, kind="text"),
        preview="",
        ops=(),
        content_sha=sha,
        path=path,
    )
    return ToolResult(handle, content=text)


@dataclass
class FileReader:
    """A stand-in for `context.Reader` (the `files` port): a path's lines, numbered."""

    root: Path
    calls: list[str] = field(default_factory=list[str])

    def read(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
        max_bytes: int | None = None,
        call_id: str | None = None,
    ) -> ToolResult:
        self.calls.append(f"read {target}")
        text = (self.root / target).read_text(encoding="utf-8").splitlines()
        start, end = 1, len(text)
        if lines is not None:
            a, b = lines.split("-")
            start, end = int(a), int(b)
        return _result(
            target,
            "\n".join(f"{n}: {text[n - 1]}" for n in range(start, min(end, len(text)) + 1)),
        )

    def repeat_key(self, target: str, **_: str | None) -> None:
        # Its reads count by their words.
        return None

    def grep_in(self, handle: str, pattern: str, *, context: int = 2) -> ToolResult:
        self.calls.append(f"grep_in {handle} {pattern}")
        return _result(handle, "")

    def place(self, sent: str) -> Place | None:
        return Place(sent) if (self.root / sent).is_file() else None


def usage(prompt: int = 100, completion: int = 20) -> Usage:
    return Usage(prompt_tokens=prompt, completion_tokens=completion)


def calls(*items: tuple[str, Mapping[str, object]]) -> Completion:
    """A turn requesting tools."""
    return Completion(
        text="",
        usage=usage(),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=tuple(
            ToolCall(id=f"c{i}", name=name, arguments=dict(args))
            for i, (name, args) in enumerate(items)
        ),
    )


def answer(findings: dict[str, object] | str) -> Completion:
    """A terminal turn: the findings JSON (or any text)."""
    text = findings if isinstance(findings, str) else json.dumps(findings)
    return Completion(text=text, usage=usage(), latency_ms=1, stop_reason=StopReason.END_TURN)


class ScriptedClient:
    """Replies in order; records every request. An exception in the script is raised."""

    def __init__(self, *replies: Completion | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("the scout asked for more turns than the script has")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing to release."""


TARGET = {"path": "pkg/net/client.py", "line_start": 6, "line_end": 11, "symbol": "HttpClient"}
CALLER = {
    "path": "pkg/api/fetcher.py",
    "line_start": 7,
    "line_end": 7,
    "symbol": "HttpClient.request",
}
