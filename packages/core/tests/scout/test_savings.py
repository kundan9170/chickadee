"""The asymmetry, as a diagnostic guard: the
scout may read a great deal and must return little. What it read stays in its own discarded
context; only pointers cross into Zeus's.

Not the D41 headline gate — a fixture, with an estimating client that charges each request
by what it actually carries, so "spent" is measured rather than scripted.
"""

from __future__ import annotations

from pathlib import Path

from scout_helpers import FileReader, GraphNavigate, ListSink, answer, calls, make_repo

from chickadee.client.request import ChatRequest, text_of
from chickadee.config import RungEndpoint
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.limits import tokens_for_chars
from chickadee.schema.adapter import Completion
from chickadee.scout.findings import Checker
from chickadee.scout.loop import Scout
from chickadee.scout.rung import ScoutRung
from chickadee.scout.search import Search
from chickadee.scout.snapshot import build_snapshot
from chickadee.scout.tools import LocateCaps, ScoutTools

BIG = "\n\n".join(
    f'def handler_{n:03d}(request):\n    """Handle case {n}."""\n    return request.get({n})'
    for n in range(300)
)


class _Estimating:
    """Replies in order, and reports usage as the size of what was sent — P-2's estimate."""

    def __init__(self, *replies: Completion) -> None:
        self.replies = list(replies)

    def send(self, request: ChatRequest) -> Completion:
        reply = self.replies.pop(0)
        sent = tokens_for_chars(sum(len(text_of(m)) for m in request.messages))
        usage = reply.usage.model_copy(
            update={
                "prompt_tokens": sent,
                "completion_tokens": tokens_for_chars(len(reply.text)),
            }
        )
        return reply.model_copy(update={"usage": usage})

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing to release."""


def test_the_scout_returns_a_small_fraction_of_what_it_spent(tmp_path: Path) -> None:
    root = tmp_path / "svc"
    (root / "svc").mkdir(parents=True)
    (root / "svc" / "handlers.py").write_text(BIG + "\n", encoding="utf-8")
    make_repo(root)
    ix = Indexer(root)
    ix.full_index()
    graph = Graph(root, ix.conn)
    try:
        target = {
            "path": "svc/handlers.py",
            "line_start": 597,
            "line_end": 599,
            "symbol": "handler_149",
        }
        client = _Estimating(
            calls(("symbol", {"name": "handler"}), ("grep", {"pattern": "request.get"})),
            calls(("read", {"handle": "svc/handlers.py"})),
            answer(
                {"targets": [target], "evidence": [{"path": "svc/handlers.py", "line": 598}]}
            ),
        )
        endpoint = RungEndpoint(
            base_url="http://localhost:1",
            model_id="m",
            context_window=128_000,
            max_output_tokens=4_000,
        )
        scout = Scout(
            ScoutRung("local-mid", endpoint, client, "scout"),
            ScoutTools(GraphNavigate(graph), FileReader(root), Search(root=root, rg=None)),
            Checker(root, graph),
            LocateCaps(40, 60_000),
            system="locate",
        )
        snap = build_snapshot(root, [], graph, probe=lambda _b: None)
        report = scout.run("which handler deals with case 149", snap, ListSink())
    finally:
        ix.close()

    assert report.outcome == "found" and report.dropped_pointers == 0
    spent, returned = report.tokens, report.returned_tokens
    assert spent > 5_000, "the fixture has the scout read the whole module"
    assert returned * 20 < spent, (spent, returned)
    text = report.render()
    assert "Handle case 12." not in text, "a file body never crosses the boundary"
    assert "svc/handlers.py:597-599   handler_149" in text
