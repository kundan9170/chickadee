"""SK-11, SK-17: a skill leaves the pinned set only by `unload_skill`, and the unload is a line
on the log: the person's act, kept by a rewind."""

from __future__ import annotations

from pathlib import Path

from skills_helpers import Tree, filler, load, must_fit, new_session, write_skill

from chickadee.context.rewind import REWIND
from chickadee.schema.adapter import ToolCall
from chickadee.skills.discover import discover
from chickadee.skills.load import Skills
from chickadee.skills.pin import SKILL_UNLOADED, pinned, unload_skill


def test_cuts_never_unpin_and_unload_does(tmp_path: Path) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "style")
    session = new_session(tmp_path)
    try:
        load(session, Skills(discover(tree.roots()), session), "style", "k1")
        filler(session)
        must_fit(session)
        assert [p.name for p in pinned(session)] == ["style"], "a cut never unpins"
        told = unload_skill(session, "style")
        assert told == "style unloaded · leaves context at the next compaction"
        assert pinned(session) == ()
        unloads = [e for e, _ in session.state.lines() if e.kind == SKILL_UNLOADED]
        assert len(unloads) == 1, "journaled, once"
    finally:
        session.close()


def test_unloading_what_is_not_loaded_writes_nothing(tmp_path: Path) -> None:
    session = new_session(tmp_path)
    try:
        before = session.state.last_seq
        assert unload_skill(session, "style") == "style is not loaded"
        assert session.state.last_seq == before
    finally:
        session.close()


def test_a_rewind_keeps_the_unload_and_hides_the_load_with_its_body() -> None:
    assert REWIND["skill_unloaded"] == "kept"
    assert REWIND["skill_loaded"] == REWIND["tool"] == "hidden"


def test_the_pinned_set_is_read_from_the_log_so_a_reopened_session_has_it(
    tmp_path: Path,
) -> None:
    tree = Tree(tmp_path)
    write_skill(tree.project, "style")
    session = new_session(tmp_path)
    load(session, Skills(discover(tree.roots()), session), "style", "k1")
    root, sid = session.root, session.session_id
    session.close()
    from chickadee.context.session import Session

    again = Session.open(root, sid)
    try:
        assert [(p.name, p.call_id) for p in pinned(again)] == [("style", "k1")]
        assert pinned(again)[0].tool_seq is not None
    finally:
        again.close()


def test_a_load_whose_call_did_not_run_pins_nothing(tmp_path: Path) -> None:
    """The load was recorded, then the session ended before its body was: the loop answers
    the call `INTERRUPTED` with `ran: false`. That is no pin, and a new load brings the body."""
    tree = Tree(tmp_path)
    write_skill(tree.project, "style", body="# Style\n\nsmall functions\n")
    session = new_session(tmp_path)
    try:
        skills = Skills(discover(tree.roots()), session)
        session.record_assistant(
            "", [ToolCall(id="k1", name="load_skill", arguments={"name": "style"})]
        )
        skills.load("style", "k1")
        assert [p.tool_seq for p in pinned(session)] == [None], "in flight: the slot is taken"
        session.record_tool("k1", "INTERRUPTED{tool: load_skill}", ran=False)
        assert pinned(session) == ()
        again = load(session, skills, "style", "k2")
        assert "small functions" in again
        assert [p.call_id for p in pinned(session)] == ["k2"]
    finally:
        session.close()
