"""A hook's verdict across a crash: what the journal holds is what a
resumed round goes on with, and a bound counts what was taken before the crash.

What these pin: a round cut off between the worker's reply and a `Continue`'s words resumes
with those words, read from the stop's journaled line, and its hook is not asked again for that
stop; a `Continue` taken before the crash counts against `hooks.max_continues` after it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import KilledMidCall, delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.boxctx.conversation import JournaledTranscript
from chickadee.client.request import Message
from chickadee.hooks.fire import BoundHook, HookSet
from chickadee.hooks.payloads import HookEvent
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Continue, HookVerdict, Note
from chickadee.schema.journal import HookP

APP = "def go() -> int:\n    return 1\n"
GO_ON = "run the suite before you stop"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


@pytest.fixture
def asked() -> list[HookEvent]:
    return []


@pytest.fixture
def hooks(asked: list[HookEvent]) -> HookSet:
    def go_on(event: HookEvent) -> HookVerdict:
        asked.append(event)
        return Continue(GO_ON)

    spec = HookSpec.model_validate({"event": "before_stop", "name": "suite", "use": "suite"})
    return HookSet((BoundHook(spec, go_on, "suite"),))


@pytest.fixture
def cut_before_the_words(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process dying after the reply is journaled and before the hook's words are."""
    add = JournaledTranscript.add

    def dying(self: JournaledTranscript, *messages: Message) -> None:
        content = messages[1].content if len(messages) == 2 else None
        if isinstance(content, str) and content.startswith(GO_ON):
            add(self, messages[0])
            raise KilledMidCall
        add(self, *messages)

    monkeypatch.setattr(JournaledTranscript, "add", dying)


@pytest.mark.writes_worktree("app.py")
def test_a_round_cut_off_before_the_words_resumes_with_them(
    root: Path,
    hooks: HookSet,
    asked: list[HookEvent],
    cut_before_the_words: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit = calls(call("edit", "w1", path="app.py", old="return 1", new="return 2"))
    worker, journal, _ = worker_rig(root, [edit, answer("done")], hooks=hooks)
    with pytest.raises(KilledMidCall):
        worker.start("T1", "Return 2.", delegate(journal, root))
    journal.close()
    assert len(asked) == 1
    monkeypatch.undo()

    resumed, journal, rung = worker_rig(
        root, [answer("ran it: 1 passed")], resume=True, hooks=hooks
    )
    report = resumed.resume("T1")

    assert report is not None and report.report == "ran it: 1 passed"
    [sent] = rung.requests
    assert (sent.messages[-2].role, sent.messages[-2].content) == ("assistant", "done")
    assert (sent.messages[-1].role, sent.messages[-1].content) == ("user", GO_ON)
    lines = [p.payload for p in journal.projection().lines(HookP)]
    assert [(line.decision, line.bounded) for line in lines] == [
        ("continue", False),
        ("continue", True),
    ], "the stop before the crash is not asked again; the next one is past the bound"
    assert len(asked) == 2


@pytest.mark.writes_worktree("app.py")
def test_the_resumed_words_are_the_live_ones_with_the_stops_notes(
    root: Path,
    asked: list[HookEvent],
    hooks: HookSet,
    cut_before_the_words: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review finding: a note at the same stop rides with the `Continue`, live and on resume."""
    lint = HookSpec.model_validate({"event": "before_stop", "name": "lint", "use": "lint"})
    both = HookSet((*hooks.hooks, BoundHook(lint, lambda e: Note("lint failed in x.py"), "l")))
    edit = calls(call("edit", "w1", path="app.py", old="return 1", new="return 2"))
    worker, journal, _ = worker_rig(root, [edit, answer("done")], hooks=both)
    with pytest.raises(KilledMidCall):
        worker.start("T1", "Return 2.", delegate(journal, root))
    journal.close()
    monkeypatch.undo()

    resumed, journal, rung = worker_rig(root, [answer("ran it")], resume=True, hooks=both)
    resumed.resume("T1")
    [sent] = rung.requests
    assert sent.messages[-1].content == f"{GO_ON}\n\nlint failed in x.py"
