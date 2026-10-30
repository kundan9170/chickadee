"""`chickadee inspect`'s reader over a session: Zeus's prompt rebuilt from the `prefix` line's
part blobs, and an id that is not a session, or climbs out of `sessions/`, not found."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.context.session import Session
from chickadee.engine import inspect_runs
from chickadee.zeus.prefix import record_prefix

from .zeus_helpers import zeus_prefix


def test_a_session_carries_zeus_prompt(tmp_path: Path) -> None:
    prefix = zeus_prefix(tmp_path / "ws")
    with Session.create(tmp_path / "sessions", "S-one") as session:
        record_prefix(session, prefix)
    inspection = inspect_runs(state=tmp_path)
    assert [s["id"] for s in inspection.index()["sessions"]] == ["S-one"]
    [line] = inspection.session("S-one")["events"]
    assert line["kind"] == "prefix"
    assert line["prompt"]["system"] == prefix.text
    assert [t["name"] for t in line["prompt"]["tools"]] == [t.name for t in prefix.tools]


@pytest.mark.parametrize("bad", ["../sessions", "S-none"])
def test_an_id_that_is_not_a_session_is_not_found(tmp_path: Path, bad: str) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_runs(state=tmp_path).session(bad)
