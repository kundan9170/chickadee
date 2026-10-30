"""Pins on what a box role keeps of its final answer, before the shared loop's rule
for it is given one home (the owner's option 2): the loop adds no final answer to the
conversation, so each role that continues one adds it itself, with these exact words.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import delegate, episode_rig, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.boxctx.athena import athena_head
from chickadee.boxctx.conversation import message_of
from chickadee.schema.journal import BoxMessageP


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": "def go():\n    return 1\n"})


def test_pin_a_worker_report_given_by_finish_is_kept_as_its_reply(root: Path) -> None:
    """Held: the report `finish` handed over joins the conversation as the worker's reply,
    after the call's result, so the next round's feedback follows it."""
    worker, journal, rung = worker_rig(
        root, [calls(call("finish", "c1", report="Nothing to change.")), answer("added")]
    )
    first = worker.start("T1", "Look.", delegate(journal, root))
    worker.cont("T1", "Add a test.", first.round)
    tail = [(m.role, m.content, bool(m.tool_calls)) for m in rung.requests[-1].messages[-4:]]
    assert tail == [
        ("assistant", "", True),
        ("tool", "reported: the task ends here", False),
        ("assistant", "Nothing to change.", False),
        ("user", "Add a test.", False),
    ]


def test_pin_athenas_answer_is_the_last_line_of_her_conversation(root: Path) -> None:
    """Held: her reply, word for word, is journaled as her conversation's last message."""
    rig = episode_rig(root, [answer("Nothing to plan.")], [])
    head = athena_head("You are Athena.", "PLAN TEMPLATE", "BRIEF TEMPLATE", design="# Design")
    got = rig.athena.start(head)
    assert (got.report, got.end) == ("Nothing to plan.", "answered")
    last = rig.journal.projection().lines(BoxMessageP)[-1].payload
    said = message_of(rig.journal.get_blob(last.sha))
    assert (last.side, said.content, said.tool_calls) == ("assistant", "Nothing to plan.", ())
