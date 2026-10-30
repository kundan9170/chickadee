"""Pins: the scout loop's exact transcripts and reports, recorded before the loop moved to
`tools/loop.py` (a refactor proves it preserves behaviour).

Each scenario runs the scout against a scripted model and records every request it sent —
role, content, tool-call id and the calls an assistant turn carried — and the report it
returned. The recording is `fixtures/loop_pins.json`, written once on the code before the move
(`CHICKADEE_WRITE_PINS=1`) and compared byte for byte after it. A pin that moves is a behaviour
change, never a refreshed fixture.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from scout_helpers import (
    TARGET,
    FileReader,
    GraphNavigate,
    ListSink,
    ScriptedClient,
    answer,
    calls,
)

from chickadee.client.errors import RequestCancelled, TransportExhausted
from chickadee.client.request import ChatRequest, Message, TextPart
from chickadee.config import RungEndpoint
from chickadee.graph.queries import Graph
from chickadee.schema.adapter import Completion, StopReason, Usage
from chickadee.scout.findings import Checker, Report
from chickadee.scout.loop import Scout
from chickadee.scout.rung import ScoutRung
from chickadee.scout.search import Search
from chickadee.scout.snapshot import Snapshot, build_snapshot
from chickadee.scout.tools import LocateCaps, ScoutTools

PINS = Path(__file__).parent / "fixtures" / "loop_pins.json"
QUESTION = "where is the timeout for outgoing requests set"


class _PinLedger:
    rung = "scout"

    def __init__(self, *, open_: bool = True) -> None:
        self.open = open_

    def charge_usage(
        self, scope_packet: str, usage: Usage, model: str, *, journaled: bool = False
    ) -> float:
        return 0.01

    def may_start(self) -> bool:
        return self.open

    def stopped_by(self) -> str | None:
        return None if self.open else "the scout rung's cap is reached"


def _endpoint() -> RungEndpoint:
    return RungEndpoint(
        base_url="http://localhost:1",
        model_id="m",
        context_window=32_000,
        max_output_tokens=2_000,
        native_tools=True,
    )


def _content(m: Message, repo: Path) -> str:
    if isinstance(m.content, str):
        text = m.content
    else:
        text = "".join(p.text if isinstance(p, TextPart) else "<image>" for p in m.content)
    # The fixture repository is committed afresh each run, so its head moves.
    return re.sub(r"@ [0-9a-f]{12}", "@ <head>", text.replace(str(repo), "<repo>"))


def _transcript(requests: list[ChatRequest], repo: Path) -> list[list[list[object]]]:
    return [
        [
            [
                m.role,
                _content(m, repo),
                m.tool_call_id,
                [c.name for c in m.tool_calls] if m.tool_calls else [],
            ]
            for m in r.messages
        ]
        for r in requests
    ]


def _report(r: Report) -> dict[str, object]:
    return {
        "outcome": r.outcome,
        "truncated": r.findings.truncated,
        "targets": len(r.findings.targets),
        "calls": r.calls,
        "tokens": r.tokens,
        "cost_usd": round(r.cost_usd, 6),
        "searched": list(r.searched),
        "notes": list(r.notes),
        "unavailable": r.unavailable,
        "detail": r.detail,
        "render": r.render(),
    }


def _cut(c: Completion) -> Completion:
    return c.model_copy(
        update={"stop_reason": StopReason.MAX_TOKENS, "text": '{"targets": []}'}
    )


_PROSE = Completion(
    text='{"targets": [',
    usage=Usage(prompt_tokens=1, completion_tokens=1),
    latency_ms=1,
    stop_reason=StopReason.MAX_TOKENS,
)

Scenario = tuple[list[Completion | Exception], LocateCaps, bool, bool]
"""Replies, caps, whether the ledger is open, whether cancel is set by the first find."""

SCENARIOS: dict[str, Scenario] = {
    "found": (
        [
            calls(("symbol", {"name": "HttpClient"}), ("grep", {"pattern": "timeout"})),
            answer({"targets": [TARGET]}),
        ],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "call_cap": (
        [
            calls(
                ("symbol", {"name": "a"}), ("symbol", {"name": "b"}), ("symbol", {"name": "c"})
            ),
            answer({"targets": [TARGET]}),
        ],
        LocateCaps(2, 20_000),
        True,
        False,
    ),
    "result_cap": (
        [calls(("read", {"path": "pkg/net/client.py"})), answer({"targets": [TARGET]})],
        LocateCaps(40, 5),
        True,
        False,
    ),
    "no_progress": (
        [
            calls(("symbol", {"name": "HttpClient"})),
            calls(("symbol", {"name": "HttpClient"})),
            calls(("symbol", {"name": "HttpClient"})),
            calls(("symbol", {"name": "HttpClient"})),
            answer({"targets": [TARGET]}),
        ],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "unreachable": (
        [TransportExhausted("connection refused", 3)],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "failed_mid_search": (
        [calls(("symbol", {"name": "HttpClient"})), TransportExhausted("reset by peer", 3)],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "cancelled_between_calls": (
        [calls(("symbol", {"name": "HttpClient"}), ("symbol", {"name": "HttpClient"}))],
        LocateCaps(40, 20_000),
        True,
        True,
    ),
    "cancelled_request": ([RequestCancelled(0)], LocateCaps(40, 20_000), True, False),
    "unparsed_twice": (
        [answer("the client is in pkg/net"), answer("still prose")],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "unparsed_once": (
        [answer("the client is in pkg/net"), answer({"targets": [TARGET]})],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "spend_cap": ([], LocateCaps(40, 20_000), False, False),
    "max_tokens_once": (
        [_cut(calls(("symbol", {"name": "HttpClient"}))), answer({"targets": [TARGET]})],
        LocateCaps(40, 20_000),
        True,
        False,
    ),
    "max_tokens_twice": ([_PROSE, _PROSE], LocateCaps(40, 20_000), True, False),
    "calls_after_the_end": (
        [
            calls(("symbol", {"name": "a"}), ("symbol", {"name": "b"})),
            calls(("symbol", {"name": "c"})),
            calls(("symbol", {"name": "d"})),
        ],
        LocateCaps(1, 20_000),
        True,
        False,
    ),
    "calls_after_the_end_then_report": (
        [
            calls(("symbol", {"name": "a"}), ("symbol", {"name": "b"})),
            calls(("symbol", {"name": "c"})),
            answer({"targets": [TARGET]}),
        ],
        LocateCaps(1, 20_000),
        True,
        False,
    ),
}


def _run(repo: Path, graph: Graph, snap: Snapshot, scenario: Scenario) -> dict[str, object]:
    replies, caps, open_, stop_on_find = scenario
    cancel = threading.Event()
    nav = GraphNavigate(graph)
    if stop_on_find:
        original: Callable[[str], str] = nav.symbol

        def symbol(name: str) -> str:
            cancel.set()
            return original(name)

        nav.symbol = symbol  # type: ignore[method-assign]
    client = ScriptedClient(*replies)
    tools = ScoutTools(nav, FileReader(repo), Search(root=repo, rg=None))
    scout = Scout(
        ScoutRung("local-mid", _endpoint(), client, "scout"),
        tools,
        Checker(repo, graph),
        caps,
        ledger=_PinLedger(open_=open_),
        system="locate",
    )
    sink = ListSink()
    report = scout.run(QUESTION, snap, sink, cancel)
    return {
        "requests": _transcript(client.requests, repo),
        "tools": [sorted(t.name for t in r.tools) for r in client.requests],
        "report": _report(report),
        "records": [k.value for k in sink.kinds()],
        "navigated": nav.calls,
        "unused_replies": len(client.replies),
    }


@pytest.fixture
def snap(repo: Path, graph: Graph) -> Snapshot:
    return build_snapshot(repo, [], graph, probe=lambda _b: None)


def test_the_scout_loop_s_transcripts_and_reports_are_pinned(
    repo: Path, graph: Graph, snap: Snapshot
) -> None:
    got = {name: _run(repo, graph, snap, s) for name, s in SCENARIOS.items()}
    text = json.dumps(got, indent=1, sort_keys=True) + "\n"
    if os.environ.get("CHICKADEE_WRITE_PINS") == "1":
        PINS.parent.mkdir(parents=True, exist_ok=True)
        PINS.write_text(text, encoding="utf-8")
    want = json.loads(PINS.read_text(encoding="utf-8"))
    for name in SCENARIOS:
        assert got[name] == want[name], f"scenario {name!r} moved"
    assert set(want) == set(SCENARIOS)
