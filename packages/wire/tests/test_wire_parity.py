"""The wire changes nothing (from the client's side): one script run as a
frontend runs it, `host_in_process` and `Client`, writes the same session journal as the same
script run on `PacketEngine` in this process, record by record. Only the clock may differ."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from engines import Step, answer, call, calls, config_for, raw_config, start
from hosting import SID, answered, repository, scripted, short_root

from chickadee.scopes import Cursor
from chickadee_wire.host import host_in_process
from chickadee_wire.models import SESSION_RECORDS
from chickadee_wire.records import payload as typed

TURNS = ("read the readme", "and summarise it")


def _script() -> list[Step]:
    return [
        calls(call("read_file", "c1", path="README.md")),
        answer("it is a readme"),
        answer("a readme, briefly"),
    ]


type Line = tuple[int, str, str | None, object]


def _comparable(seq: int, kind: str, run_id: str | None, payload: object) -> Line:
    return (seq, kind, run_id, payload)


def _over_the_wire(
    monkeypatch: pytest.MonkeyPatch, root: Path, script: Sequence[Step]
) -> list[Line]:
    repo = repository(monkeypatch, root)
    scripted(monkeypatch, script)
    with host_in_process(repo) as host, host.client() as client:
        for n, text in enumerate(TURNS, start=1):
            client.turn(SID, text)
            answered(client, n)
        records = client.session_journal(SID)
        for record in records:  # every served record reads as its kind's type
            assert typed(record) is not None or record.kind not in SESSION_RECORDS, record.kind
        return [_comparable(r.seq, r.kind, r.run_id, r.payload) for r in records]


def _in_process(
    monkeypatch: pytest.MonkeyPatch, root: Path, script: Sequence[Step]
) -> list[Line]:
    repo = repository(monkeypatch, root)
    engine, _ = start(monkeypatch, repo, config_for(raw_config()), script)
    try:
        for text in TURNS:
            engine.turn(SID, text).wait(timeout=20)
        page = engine.journal(SID, Cursor())
    finally:
        engine.close()
    return [_comparable(r.seq, r.kind, r.run_id, dict(r.payload)) for r in page.records]


def test_a_session_over_the_wire_journals_as_it_does_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One path for both, rebuilt between them: the prompt prefix hashes the environment,
    # the repository's path included.
    root = short_root()
    wire = _over_the_wire(monkeypatch, root, _script())
    shutil.rmtree(root)
    local = _in_process(monkeypatch, root, _script())
    assert {"turn", "tool", "spend"} <= {kind for _, kind, _, _ in wire}
    assert wire == local
