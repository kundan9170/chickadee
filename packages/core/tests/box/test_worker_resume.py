"""Resume of an open worker round.

What these pin: a step the crash cut off is put back to its pre-image tree, completed
`failed`, and its call answered `interrupted`; a cut-off step whose command declared the
network is answered `outcome unknown` and never run again; a step completed whose answer was
lost says so, and a call that never ran is answered not run; the round then continues from its
conversation as journaled, the same messages the worker saw; a reply cut off after its report
is owed to the next round; with no round open, resume does nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from box_helpers import WORKER_RUN, CrashingCommand, KilledMidCall, delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.box.resume import INTERRUPTED, LOST, NOT_RUN, OUTCOME_UNKNOWN
from chickadee.boxctx.conversation import BoxConversation, JournaledTranscript
from chickadee.client.request import Message
from chickadee.ids import step_id
from chickadee.journal.effects import EffectKind, Intent, on_journal, perform
from chickadee.journal.writer import Journal
from chickadee.schema import RetryParams
from chickadee.schema.journal import EffectCompletedP, EventKind, TaskFactsP, TaskReportedP
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import current_tree

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


@pytest.mark.writes_worktree("app.py", "made.txt")
def test_a_cut_off_step_is_put_back_answered_and_the_round_continues(root: Path) -> None:
    worker, journal, first = worker_rig(
        root,
        [calls(call("shell", "c1", command="make gen"))],
        supervisor=CrashingCommand(root),
    )
    with pytest.raises(KilledMidCall):
        worker.start("T1", "Generate it.", delegate(journal, root))
    journal.close()

    resumed, journal, rung = worker_rig(root, [answer("generated")], resume=True)
    report = resumed.resume("T1")

    assert report is not None and (report.round, report.stopped) == (1, "replied")
    assert (root / "app.py").read_text(encoding="utf-8") == APP
    assert not (root / "made.txt").exists()
    [sent] = rung.requests
    # The conversation the worker saw, then the cut-off call's answer.
    assert sent.messages[:-1] == first.requests[0].messages + sent.messages[-2:-1]
    assert (sent.messages[-1].content, sent.messages[-1].tool_call_id) == (INTERRUPTED, "c1")
    [op] = journal.projection().operations.values()
    assert op.completed is not None
    assert (op.completed.outcome, op.completed.observed_surface) == (
        "failed",
        ("app.py", "made.txt"),
    )
    assert report.facts.files == ()


def _open_round(
    root: Path, journal: Journal, *tool_calls: Any, turn: bool = True
) -> BoxConversation:
    """A round left as a crash leaves it: the head and one turn asking for `tool_calls` (no
    turn, `turn` False)."""
    delegate(journal, root)
    conversation = BoxConversation(
        role="worker",
        head=(Message(role="system", content="S"), Message(role="user", content="B")),
        messages=(),
        params=RetryParams(temperature=0.2, max_tokens=100),
    )
    transcript = JournaledTranscript(
        conversation,
        journal,
        budget=100_000,
        keep_recent=4,
        keep_tools=frozenset(),
        task_id="T1",
        round=1,
    )
    transcript.begin()
    if turn:
        transcript.add(Message(role="assistant", content="", tool_calls=tool_calls))
    return transcript.conversation


def _step(
    journal: Journal, root: Path, cid: str, *, hosts: tuple[str, ...] = (), done: bool
) -> None:
    before, _ = current_tree(root, untracked_max_bytes=None)
    intent = Intent(
        operation_id=step_id("T1", 1, cid),
        kind=EffectKind.WORKER_STEP,
        approved_sha="0" * 64,
        preconditions=(f"before={before}", *(f"host={h}" for h in hosts)),
        idempotent=False,
        surface=("shell",),
        preimages=before,
    )
    log = on_journal(journal, task_id="T1", round=1)
    if not done:
        with pytest.raises(KilledMidCall), perform(log, intent):
            raise KilledMidCall
        return
    with perform(log, intent) as effect:
        effect.complete("succeeded", post_state=before)


def _journal(root: Path) -> Journal:
    return Journal.create(resolve_roots(root), WORKER_RUN, {})


def test_a_cut_off_networked_step_is_never_run_again_and_its_outcome_unknown(
    root: Path,
) -> None:
    journal = _journal(root)
    _open_round(root, journal, call("shell", "c1", command="pip download x", network=["pypi"]))
    _step(journal, root, "c1", hosts=("pypi",), done=False)
    journal.close()
    worker, journal, rung = worker_rig(root, [answer("checked")], resume=True)
    worker.resume("T1")
    tool = rung.requests[0].messages[-1]
    assert (tool.content, tool.tool_call_id) == (OUTCOME_UNKNOWN, "c1")
    [op] = journal.projection().operations.values()
    assert op.completed is not None and op.completed.outcome == "failed"


def test_a_lost_answer_and_a_call_never_run_are_each_said(root: Path) -> None:
    journal = _journal(root)
    _open_round(
        root,
        journal,
        call("shell", "c1", command="make"),
        call("read", "c2", path="app.py"),
    )
    _step(journal, root, "c1", done=True)
    journal.close()
    worker, _, rung = worker_rig(root, [answer("ok")], resume=True)
    worker.resume("T1")
    said = [(m.tool_call_id, m.content) for m in rung.requests[0].messages[-2:]]
    assert said == [("c1", LOST), ("c2", NOT_RUN)]


def test_a_reply_cut_off_after_its_report_is_owed_to_the_next_round(root: Path) -> None:
    """The report is written before the reply joins the conversation, so a round with a reply
    is never open; a crash between leaves the reply owed, and feedback follows it."""
    journal = _journal(root)
    _open_round(root, journal, turn=False)
    journal.append(
        EventKind.TASK_REPORTED,
        TaskReportedP(
            task_id="T1",
            round=1,
            stopped="replied",
            report_sha=journal.put_blob(b"all done"),
            facts=TaskFactsP(start_tree=_tree(root), end_tree=_tree(root)),
        ),
        task_id="T1",
    )
    journal.close()
    worker, _, rung = worker_rig(root, [answer("tested")], resume=True)
    assert worker.resume("T1") is None
    report = worker.cont("T1", "Add a test.", 1)
    assert report.report == "tested"
    said = [(m.role, m.content) for m in rung.requests[0].messages[-3:]]
    assert said[1:] == [("assistant", "all done"), ("user", "Add a test.")]


def _tree(root: Path) -> str:
    tree, _ = current_tree(root, untracked_max_bytes=None)
    return tree


def test_with_no_round_open_resume_does_nothing(root: Path) -> None:
    worker, journal, _ = worker_rig(root, [answer("done")])
    worker.start("T1", "Do it.", delegate(journal, root))
    assert worker.resume("T1") is None
    with pytest.raises(ValueError, match="never delegated"):
        worker.resume("T9")
    assert not any(
        isinstance(p.payload, EffectCompletedP) for p in journal.projection().tasks["T1"]
    )
