"""`Session.fork` takes the person's choice and never chooses for them: `keep` delivers the
kept-code notice once on arrival, `restore` none."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from chickadee.context.session import Session, Turn

KEPT = "code changed in turns you no longer see: auth.py (run r7)"


def arrived(session: Session, code: Literal["restore", "keep"]) -> list[str]:
    session.record_user("change auth")
    at = session.record_assistant("done").seq
    root, sid = session.root, session.session_id
    child = Session.fork(
        root,
        sid,
        at,
        "again",
        "S-child",
        code=code,
        kept_paths=("auth.py",),
        runs=("r7",),
    )
    with child:
        return [
            r.text
            for e, r in child.state.lines()
            if e.session_id == "S-child" and isinstance(r, Turn) and r.role == "harness"
        ]


def test_keep_names_the_code_the_child_no_longer_sees_once(session: Session) -> None:
    assert arrived(session, "keep") == [KEPT]


def test_restore_delivers_no_kept_code_notice(session: Session) -> None:
    assert arrived(session, "restore") == []


def test_a_caller_must_choose() -> None:
    with pytest.raises(TypeError, match="code"):
        Session.fork(Path("."), "S", 1, "r", "C")  # type: ignore[call-arg]
