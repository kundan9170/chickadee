"""`/rename`: the person names the session, and is told the name; a
name with no words is not sent."""

from __future__ import annotations

import pytest
from repl_driver import Typed, repl


def test_rename_names_the_session_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    typed = ["/rename Retry the fetcher", "/rename", "/exit"]
    with repl(monkeypatch, [], [Typed(t) for t in typed]) as (source, _code):
        out = source.out
    assert "named Retry the fetcher" in out
    assert "/rename <name>" in out, "no words: the usage, nothing sent"
