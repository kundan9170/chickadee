"""Plain helpers for the agents suite (a uniquely named module, never `conftest`).

The fakes stand where an agent takes a port: the model client, the reader, the navigator and the
search. The session log, the skills on disk and the files cited are real.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chickadee.agents.profiles import Profile, ProfileLimits
from chickadee.agents.spawn import AgentPorts, Agents
from chickadee.budget.manager import Ledger
from chickadee.client.request import ChatRequest
from chickadee.config import RungEndpoint
from chickadee.context.citations import workspace_lines
from chickadee.context.handle import Handle, Shape, content_sha, handle_id
from chickadee.context.results import HandleExpired, ToolFailure, ToolResult
from chickadee.context.session import Session
from chickadee.context.spans import Span
from chickadee.edits.lines import number_lines
from chickadee.paths import Place
from chickadee.roles import ZEUS_CEILING, role
from chickadee.schema.adapter import Completion, StopReason, ToolCall, Usage
from chickadee.skills.discover import SkillCatalog, SkillRoots, discover
from chickadee.tools.policy import RolePolicy

ENDPOINT = RungEndpoint(
    base_url="http://x", model_id="m", context_window=32_000, max_output_tokens=2_000
)


def turn_calls(*items: tuple[str, Mapping[str, object]]) -> Completion:
    return Completion(
        text="",
        usage=Usage(prompt_tokens=100, completion_tokens=20),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=tuple(
            ToolCall(id=f"c{i}", name=name, arguments=dict(args))
            for i, (name, args) in enumerate(items)
        ),
    )


def turn_text(text: str) -> Completion:
    return Completion(
        text=text,
        usage=Usage(prompt_tokens=100, completion_tokens=20),
        latency_ms=1,
        stop_reason=StopReason.END_TURN,
    )


def report(findings: str = "- none", *, extra: str = "", not_done: str = "nothing") -> str:
    return f"## Summary\nIt is done.\n## Findings\n{findings}\n{extra}## Not done\n{not_done}\n"


class AgentClient:
    """Replies in order; keeps every request."""

    def __init__(self, *replies: Completion | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError("the agent asked for more turns than the script has")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing."""


@dataclass
class LineReader:
    """The `files` port: a path's lines, numbered as the real reader numbers them."""

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
            start, end = int(a), min(int(b), len(text))
        shown = "\n".join(number_lines(text, [Span(start, end)]))
        data = shown.encode("utf-8")
        sha = content_sha(data)
        handle = Handle(
            id=handle_id("file", path=target, run_id=None, sha=sha),
            kind="file",
            shape=Shape(bytes=len(data), lines=None, symbols=None, kind="text"),
            preview="",
            ops=(),
            content_sha=sha,
            path=target,
        )
        return ToolResult(handle, content=shown)

    def repeat_key(self, target: str, **_: str | None) -> Hashable | None:
        return None

    def grep_in(self, handle: str, pattern: str, *, context: int = 2) -> ToolResult:
        raise AssertionError("not used here")

    def place(self, sent: str) -> Place | None:
        return Place(sent) if (self.root / sent).is_file() else None


class NoGraph:
    """The `navigator` port, answering nothing."""

    def atlas(self) -> str:
        return "no graph"

    def symbol(self, name: str) -> str:
        return f"UNRESOLVED{{query: {name!r}}}"

    def neighbors(self, target: str, depth: int) -> str:
        return "none"

    def impact(self, symbols: Sequence[str]) -> str:
        return "none"

    def tests_for(self, paths: Sequence[str]) -> str:
        return "no test file"


def no_search(
    query: str, *, kind: str = "text", paths: Sequence[str] = (), cancel: object = None
) -> str:
    return "no match"


def write_repo(root: Path) -> Path:
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "auth.py").write_text(
        "def login(user):\n    check(user)\n    return token(user)\n", encoding="utf-8"
    )
    return root


def reader_profile(**limits: int) -> Profile:
    r = role("reader")
    assert r.tool_defaults is not None
    base = ProfileLimits(
        max_tool_calls=30,
        max_result_tokens=10_000,
        seconds=300,
        max_cost_cents=50,
        report_reasks=1,
    )
    return Profile(
        "reader", "reader", r.tool_defaults.enabled, "mid-param", base.lowered(limits)
    )


def agents_for(
    session: Session,
    repo: Path,
    client: AgentClient,
    *,
    roots: SkillRoots | None = None,
    profiles: Mapping[str, Profile] | None = None,
    handles: Mapping[str, str] | None = None,
    trusted: bool = True,
    third_party: bool = False,
    reader: LineReader | None = None,
    clients: Callable[[], AgentClient] | None = None,
    ports: Callable[[Profile], AgentPorts] | None = None,
    ledger: Ledger | None = None,
    max_follow_ups: int = 3,
) -> Agents:
    """`clients` makes a client per round, for rounds that run at once; else `client` serves
    every round. `ledger` is every rung's."""
    catalog = discover(roots) if roots is not None else SkillCatalog()
    given = dict(handles or {})
    files = reader if reader is not None else LineReader(repo)

    def policy(name: str) -> RolePolicy:
        r = role(name)
        assert r.tool_defaults is not None
        return r.tool_defaults

    def handle_text(handle: str) -> str | ToolFailure:
        text = given.get(handle)
        return text if text is not None else HandleExpired(handle, "not_in_session")

    return Agents(
        session,
        catalog,
        profiles if profiles is not None else {"reader": reader_profile()},
        ceiling=ZEUS_CEILING,
        ceilings=lambda name: role(name).tool_ceiling,
        policy=policy,
        endpoint=lambda rung: ("m", ENDPOINT),
        client_for=lambda endpoint: client if clients is None else clients(),
        ports=ports
        if ports is not None
        else lambda profile: AgentPorts(files, NoGraph(), no_search),
        handle_text=handle_text,
        read=workspace_lines(repo),
        ledger=None if ledger is None else lambda rung: ledger,
        trusted=trusted,
        third_party=third_party,
        max_follow_ups=max_follow_ups,
    )
