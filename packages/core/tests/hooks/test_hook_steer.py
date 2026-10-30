"""`round_ended` across a resume (`hooks/steer.py`).

What these pin: a round its hooks already answered is read back from its lines, never asked
again: its notes stay the ones written, and a retry whose review line a crash cut off is taken
from its words with `fresh` False, the hook not run a second time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.hooks.fire import BoundHook, Hooks, HookSet, RoleHooks
from chickadee.hooks.payloads import HookEvent, RoundEnded
from chickadee.hooks.sinks import RunHookSink
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.steer import hook_review, retry_line, round_ended
from chickadee.hooks.verdicts import HookVerdict, Note, RetryTask
from chickadee.journal.writer import Journal
from chickadee.schema.journal import EventKind, HookP, TaskFactsP
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-10-24-0e1f2a3b"


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    return Journal.create(resolve_roots(tmp_path), RUN, {})


def _ended() -> RoundEnded:
    facts = TaskFactsP(start_tree="a", end_tree="b")
    return RoundEnded(task="T1", round=1, stopped="looped", report="stuck", facts=facts)


def _hooks(journal: Journal, verdict: HookVerdict, asked: list[HookEvent]) -> RoleHooks:
    def run(event: HookEvent) -> HookVerdict:
        asked.append(event)
        return verdict

    spec = HookSpec.model_validate({"event": "round_ended", "name": "steer", "use": "steer"})
    bound = HookSet((BoundHook(spec, run, "steer"),))
    return RoleHooks(Hooks(bound, RunHookSink(journal)), "athena", run_id=RUN)


def test_a_round_already_answered_is_read_back_not_asked_again(journal: Journal) -> None:
    asked: list[HookEvent] = []
    hooks = _hooks(journal, Note("the loop was on read"), asked)
    assert round_ended(hooks, _ended(), room=True) is None
    # The desk acting again on the same round, here or in a resumed process.
    assert round_ended(hooks, _ended(), room=True) is None
    again = RoleHooks(Hooks(hooks.hooks.bound, RunHookSink(journal)), "athena", run_id=RUN)
    assert round_ended(again, _ended(), room=True) is None
    assert len(asked) == 1
    assert len(journal.projection().lines(HookP)) == 1


def test_a_retry_whose_review_was_cut_off_is_taken_from_its_line(journal: Journal) -> None:
    asked: list[HookEvent] = []
    hooks = _hooks(journal, RetryTask("again, smaller", fresh=True), asked)
    taken = round_ended(hooks, _ended(), room=True)
    assert taken is not None and taken[1] == RetryTask("again, smaller", fresh=True)
    # A crash before the desk wrote its review line: the resumed process reads the retry.
    again = RoleHooks(Hooks(hooks.hooks.bound, RunHookSink(journal)), "athena", run_id=RUN)
    assert round_ended(again, _ended(), room=True) == (
        "steer",
        RetryTask("again, smaller", fresh=False),
    )
    assert len(asked) == 1


def test_a_retry_line_is_found_by_its_whole_id_never_a_suffix(journal: Journal) -> None:
    """Review finding: the module that writes a retry's id is the one that reads it, whole."""
    retry = RetryTask("again")
    for task, round_ in (("T1", 12), ("T11", 2)):
        line = retry_line("steer", task, round_, journal.put_blob(b"again"), retry)
        journal.append(EventKind.TASK_REVIEWED, line, task_id=task)
    run = journal.projection()
    assert hook_review(run, "T1", 2) is None
    found = hook_review(run, "T1", 12)
    assert found is not None and found.call_id == "hook:steer:T1:12"
