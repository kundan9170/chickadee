"""Each box model call names what its transport got. Every `model_call` line of an episode
carries `sent_sha`, and the blobs it names rebuild, in order, exactly the requests the rung's
transport received: messages, tool schemas, settings. A client a role wraps around its rung
(PR 641's Python calls is one) sits above the recorder, so the request it rewrote is the one
recorded."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from box_helpers import BoxRig, FakeRunJournal, design, worker_call
from box_helpers import athena_call as acall
from engines import ScriptedRung, answer, calls

from chickadee.box.sink import BoxSink
from chickadee.boxctx.conversation import message_bytes
from chickadee.boxctx.sent import SentRequests
from chickadee.client.request import ChatRequest, Message, ToolSchema
from chickadee.journal import paths
from chickadee.journal.reader import read_events
from chickadee.schema import RetryParams
from chickadee.schema.journal import ModelCallP
from chickadee.state.roots import roots as resolve_roots

BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"


def _rebuilt(blob: Any, sha: str) -> dict[str, Any]:
    index = json.loads(blob(sha))
    return {
        "messages": [blob(m) for m in index["messages"]],
        "tools": json.loads(blob(index["tools"])),
        "params": index["params"],
        "force_tool": index["force_tool"],
    }


def _expected(request: ChatRequest) -> dict[str, Any]:
    return {
        "messages": [message_bytes(m) for m in request.messages],
        "tools": [
            {"name": t.name, "description": t.description, "parameters": dict(t.schema)}
            for t in request.tools
        ],
        "params": request.params.model_dump(mode="json"),
        "force_tool": request.force_tool,
    }


def test_every_call_of_an_episode_names_exactly_what_its_transport_got(rig: BoxRig) -> None:
    rig.athena_script = [
        calls(acall("plan", "a1", text="1. **Return 2.**\n")),
        calls(acall("delegate", "a2", title="Return 2", brief=BRIEF)),
        calls(acall("review", "a3", task="T1", answer="accept")),
        answer("done"),
    ]
    rig.worker_script = [
        calls(worker_call("edit", "w1", path="app.py", old="return 1", new="return 2")),
        answer("T1: go() returns 2."),
    ]
    report = rig.box().open(design()).report()
    root = paths.run_dir(resolve_roots(rig.workspaces[0].root), report.run_id or "")
    blobs = root / "blobs"

    def blob(sha: str) -> bytes:
        return (blobs / sha).read_bytes()

    lines = [
        e.payload
        for e in read_events(root / "journal.jsonl")
        if isinstance(e.payload, ModelCallP)
    ]
    for role in ("athena", "worker"):
        mine = [p for p in lines if p.role == role]
        sent: list[ChatRequest] = rig.rungs[role].requests
        assert mine and len(mine) == len(sent)
        assert all(p.sent_sha is not None for p in mine)
        got = [_rebuilt(blob, p.sent_sha or "") for p in mine]
        assert got == [_expected(r) for r in sent]
        assert got[0]["tools"], f"{role}'s tool schemas are recorded"


class _Rewriting:
    """A client a role wraps around its rung, as PR 641's does: it sends other messages and no
    tools."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def send(self, request: ChatRequest) -> Any:
        system = Message("system", "REWRITTEN with python stubs")
        return self._inner.send(replace(request, messages=(system,), tools=()))

    def close(self) -> None:
        self._inner.close()


def test_a_wrapped_client_is_recorded_as_it_rewrote_the_request(tmp_path: Path) -> None:
    del tmp_path
    transport = ScriptedRung([answer("ok")])
    recorder = SentRequests(transport)
    journal = FakeRunJournal()
    sink = BoxSink("worker", journal, "low-param")  # type: ignore[arg-type]
    sink.sent = recorder
    request = ChatRequest(
        messages=(Message("system", "native"), Message("user", "go")),
        params=RetryParams(temperature=0.0, max_tokens=100),
        tools=(ToolSchema("read", "Read a file.", {"type": "object"}),),
    )
    completion = _Rewriting(recorder).send(request)
    sink.model_call("task", completion, 0.0, None)

    ((_, line, _, _),) = journal.lines
    assert isinstance(line, ModelCallP) and line.sent_sha is not None
    got = _rebuilt(lambda sha: journal.blobs[sha], line.sent_sha)
    assert got == _expected(transport.requests[0])
    assert got["tools"] == []
    assert json.loads(got["messages"][0])["content"] == "REWRITTEN with python stubs"
    # Taken once: a later line never names a request it did not send.
    sink.model_call("task", completion, 0.0, None)
    assert journal.lines[-1][1].sent_sha is None
