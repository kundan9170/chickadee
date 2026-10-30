"""A run's record is its journal line as stored: served over the wire, every
payload equals the stored one, field names included. An aliased field (`lint_result`'s `pass`)
reached clients under its Python name (`passed`), which no client could read."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from hosting import hosted

RECORDED = [Path(__file__).resolve().parent / "fixtures" / "run"]
"""The recorded task run (`recorded_run.py`). The packet loop's recorded journals
are v1, which the server refuses to read since."""


@pytest.mark.parametrize("recorded", RECORDED, ids=lambda p: p.name)
def test_each_served_payload_is_the_stored_line(
    monkeypatch: pytest.MonkeyPatch, recorded: Path
) -> None:
    journal = recorded / "journal.jsonl"
    stored = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    run_id = stored[0]["run_id"]
    with hosted(monkeypatch) as host, host.client() as client:
        target = host.state_dir / "runs" / run_id
        target.mkdir(parents=True)
        shutil.copy(journal, target / "journal.jsonl")
        served = client.run_journal(run_id)
    assert [f.seq for f in served] == [e["seq"] for e in stored]
    assert [f.payload for f in served] == [e["payload"] for e in stored]
