"""Concerns: advisories with an anchored, durable home (X-20-X-22).

An unanchored advisory is refused, not stored: naming no path, it could never surface. A
concern surfaces only where its anchor is in scope. Promotion stops it surfacing and says
nothing about whether it was right. And only an explicit call creates one: nothing reads the
conversation back and extracts concerns from it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import pytest
from context_helpers import cut, exchange, recall_of, sessions_dir, work_unit
from refusals import bare

import chickadee.context.concerns as concerns_module
from chickadee.context.concerns import (
    accrue,
    concerns,
    concerns_for_critic,
    dismiss,
    ingest_report_concerns,
    note_concern,
    promote,
)
from chickadee.context.lines import ConcernLine
from chickadee.context.recall import Recalled
from chickadee.context.results import InvalidArgument
from chickadee.context.session import Session
from chickadee.context.standing import NoticeTrigger, standing_snapshot
from chickadee.errors import HarnessBug

POOL = "db/pool.py"
VIEWS = "api/views.py"
RUN = "2026-09-26-0000beef"


def notices(session: Session) -> list[str]:
    return [
        t.text
        for t in session.state.turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.CONCERN
    ]


def raised(session: Session) -> str:
    session.record_user("look at the pool")
    cid = note_concern(session, "connections may leak on timeout", [POOL], "zeus")
    assert cid == "K-1"
    return "K-1"


# -- 1 · anchored, or it is not a concern --------------------------------------------------


@pytest.mark.parametrize(
    ("text", "anchor", "by", "field"),
    [
        ("leaks on timeout", [], "zeus", "anchor"),
        ("leaks on timeout", ["  ", ""], "zeus", "anchor"),
        ("   ", [POOL], "zeus", "text"),
        ("leaks on timeout", [POOL], "the linter", "by"),
    ],
)
def test_an_advisory_that_cannot_surface_is_refused_and_not_stored(
    session: Session, text: str, anchor: list[str], by: str, field: str
) -> None:
    session.record_user("look at the pool")
    refused = note_concern(session, text, anchor, by)
    assert isinstance(refused, InvalidArgument)
    assert refused.field == field
    assert session.state.last_seq == 1
    assert session.state.concerns.all() == ()


def test_a_concern_keeps_the_raisers_words_exactly(session: Session) -> None:
    session.record_user("look at the pool")
    words = "  Connections   may LEAK on timeout.\nSee line 40.  "
    assert note_concern(session, words, [POOL, POOL, f" {VIEWS} "], "critic") == "K-1"
    stored = session.state.concerns.get("K-1")
    assert stored is not None
    assert stored.text == words
    assert stored.anchor == (POOL, VIEWS)


# -- 2 · surfaces only where its anchor is in scope ----------------------------------------


def test_a_concern_surfaces_only_for_its_anchored_paths(session: Session) -> None:
    raised(session)
    assert [c.id for c in concerns(session.state, path=POOL)] == ["K-1"]
    assert concerns(session.state, path=VIEWS) == ()
    assert concerns_for_critic(session.state, "post_code", [VIEWS]) == ()
    assert [c.id for c in concerns_for_critic(session.state, "post_code", [POOL])] == ["K-1"]


def test_the_pre_code_critic_sees_only_the_users_concerns(session: Session) -> None:
    raised(session)
    assert note_concern(session, "the pool size is wrong", [POOL], "user") == "K-2"
    assert [c.id for c in concerns_for_critic(session.state, "pre_code")] == ["K-2"]


def test_a_notice_names_only_paths_no_live_concern_already_surfaces(session: Session) -> None:
    raised(session)
    note_concern(session, "and the retry is unbounded", [POOL], "zeus")
    note_concern(session, "views swallow errors", [POOL, VIEWS], "zeus")
    assert notices(session) == [f"concern K-1 raised · {POOL}", f"concern K-3 raised · {VIEWS}"]


def test_the_standing_snapshot_carries_paths_never_text(session: Session) -> None:
    raised(session)
    snapshot = standing_snapshot(session.state, session.state.last_seq + 1).text
    assert POOL in snapshot
    assert "concerns(path=…)" in snapshot
    assert "leak" not in snapshot


# -- 3 · promotion retires without resolving -----------------------------------------------


def test_promotion_stops_surfacing_and_asserts_nothing_about_resolution(
    session: Session, tmp_path: Path
) -> None:
    cid = raised(session)
    before = session.state.concerns.get(cid)
    assert isinstance(promote(session, cid, "tests/test_pool.py::test_timeout"), int)
    assert concerns(session.state, path=POOL) == ()
    assert [c.id for c in concerns(session.state, include_retired=True)] == [cid]
    assert session.state.concerns.get(cid) == before, "no status field was written"
    assert POOL not in standing_snapshot(session.state, session.state.last_seq + 1).text
    found = recall_of(sessions_dir(tmp_path))("concerns")
    assert isinstance(found, Recalled)
    (hit,) = found.hits
    assert hit.text.endswith("· watched by tests/test_pool.py::test_timeout")
    assert "resolved" not in hit.text


def test_promotion_names_a_check_and_happens_once(session: Session) -> None:
    cid = raised(session)
    assert bare(promote(session, cid, "  ")) == InvalidArgument(
        "check_id", "promotion names the check that watches it"
    )
    assert isinstance(promote(session, cid, "check-a"), int)
    assert bare(promote(session, cid, "check-b")) == InvalidArgument(
        "id", f"{cid} is already watched by a check"
    )


def test_only_the_user_dismisses_a_concern(session: Session) -> None:
    cid = raised(session)
    assert bare(dismiss(session, cid, by="zeus")) == InvalidArgument(
        "by", "only the user dismisses a concern"
    )
    assert isinstance(dismiss(session, cid, by="user"), int)
    assert concerns(session.state) == ()


def test_touching_an_anchored_path_records_it_and_decides_nothing(session: Session) -> None:
    cid = raised(session)
    note_concern(session, "views swallow errors", [VIEWS], "zeus")
    promote(session, "K-2", "check-views")
    written = accrue(session, [POOL, VIEWS, "README.md"], run_id=None)
    assert len(written) == 1
    assert [t.path for t in session.state.concerns.touched(cid)] == [POOL]
    assert session.state.concerns.is_live(cid)


# -- 4 · only an explicit call creates one ---------------------------------------------------


@dataclass(frozen=True)
class BoxConcern:
    text: str
    anchor: tuple[str, ...]


def test_the_boxs_concerns_are_ingested_and_an_unanchored_one_refused(session: Session) -> None:
    session.record_user("go")
    made = ingest_report_concerns(
        session,
        [BoxConcern("pool leaks", (POOL,)), BoxConcern("something feels off", ())],
        run_id=RUN,
    )
    assert made[0] == "K-1"
    assert isinstance(made[1], InvalidArgument)
    assert [(c.raised_by, c.run_id) for c in session.state.concerns.all()] == [("coder", RUN)]


def test_a_report_over_the_cap_is_the_boxs_bug_not_trimmed(session: Session) -> None:
    session.record_user("go")
    report = [BoxConcern(f"worry {i}", (POOL,)) for i in range(4)]
    with pytest.raises(HarnessBug):
        ingest_report_concerns(session, report, run_id=RUN)
    assert session.state.concerns.all() == ()


def test_reading_the_conversation_back_creates_no_concern(session: Session, ws: Path) -> None:
    """The re-extraction loop cannot form: a session full of worry, compacted and
    recalled, still holds exactly the concerns someone raised by calling for one."""
    for i in range(3):
        session.record_user(f"I am concerned that {POOL} leaks connections ({i})")
        exchange(session, f"c{i}", "shell", {"command": f"grep leak {POOL}"}, "leak()\n")
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("worried about the pool still")
    session.record_assistant("noted, the pool may leak")
    assert cut(session, ws).cut is not None
    recall_of(session.root)("pool leak", current=session.state)
    assert session.state.concerns.all() == ()


_CONCERN_LINES = frozenset(t.__name__ for t in get_args(get_args(ConcernLine)[0]))
"""Every type a `concern` line is: raised, touched, promoted, dismissed."""


def test_only_the_concerns_module_writes_a_concern_line() -> None:
    """Checked at the source, where a new extractor would have to appear: in the context
    package, a `concern` line is appended by `concerns.py` alone, and a concern is raised
    only through `note_concern`, whose one caller here ingests the box's own report."""
    package = Path(concerns_module.__file__).parent
    appenders: list[str] = []
    raisers: list[str] = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", getattr(node.func, "id", None))
                # A `concern` line is one of its four types; building one is writing
                # it, since `Session.append` is the only thing that takes one.
                if name in _CONCERN_LINES:
                    appenders.append(f"{path.name}:{fn.name}")
                if name == "note_concern":
                    raisers.append(f"{path.name}:{fn.name}")
    assert sorted(set(appenders)) == [
        "concerns.py:accrue",
        "concerns.py:dismiss",
        "concerns.py:note_concern",
        "concerns.py:promote",
    ]
    assert raisers == ["concerns.py:ingest_report_concerns"]
