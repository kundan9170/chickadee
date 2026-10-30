"""The fork notice lists what a fork cannot undo, and nothing it does (bug 2).

A fork undoes a run's worktree by never inheriting it, so the effects confined to that
worktree are not listed: the kinds `KINDS` marks `worktree_only`. Those must be kinds
the journal writes, read here from the writers rather than restated, or the column silently
matches nothing and every patch is reported as a change to the world. An in-place run has no
worktree: what it wrote is in the person's checkout, which a turn snapshot restores or the
kept-code notice names, so it is listed only for a turn with no
snapshot.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from context_helpers import FakeRecords, effect, run_end, run_id, run_start

from chickadee.context.standing import (
    _run_effects,  # pyright: ignore[reportPrivateUsage]
)
from chickadee.journal.effects import KINDS, EffectKind
from chickadee.schema.journal import JournalEvent

SRC = Path(__file__).resolve().parents[2] / "src" / "chickadee"


def _members(node: ast.expr) -> set[str]:
    """Every `EffectKind` member `node` can be: `EffectKind.X`, or either branch of a
    conditional."""
    if isinstance(node, ast.IfExp):
        return _members(node.body) | _members(node.orelse)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "EffectKind"
    ):
        return {EffectKind[node.attr].value}
    return set()


def written_kinds() -> set[str]:
    """Every effect kind the source writes: the kind of each `Intent(...)` handed to the
    recorder, its second argument or its `kind=`, and the `kind=` of each
    `commit_effect(...)`, which builds the commit's intent from it (`box/commit.py`)."""
    kinds: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name not in ("Intent", "commit_effect"):
                continue
            if name == "Intent" and len(node.args) >= 2:
                kinds |= _members(node.args[1])
            for kw in node.keywords:
                if kw.arg == "kind":
                    kinds |= _members(kw.value)
    return kinds


def test_every_worktree_kind_is_one_the_journal_writes() -> None:
    written = written_kinds()
    # The scan itself is checked: a reader that found nothing would pass anything. D166,
    # It found the packet loop's `patch_apply`, `patch_rollback`, `packet_commit` and
    # `action_exec`, which retired with it; the box's kinds are what it finds now.
    assert {"bootstrap", "worker_step", "task_commit", "task_known_good"} <= written
    worktree = {kind.value for kind, rule in KINDS.items() if rule.worktree_only}
    assert written >= worktree, worktree - written


RUN = run_id("fork")
WORKTREE_KINDS = {"bootstrap", "task_commit"}
"""The kinds `KINDS` marks `worktree_only`: what a fork of a worktree run never inherits."""


def _journal(isolation: Literal["worktree", "in_place"] = "worktree") -> list[JournalEvent]:
    """A task loop's run: its bootstrap, one worker step, and the task's commit. The
    recordings these read before were the packet loop's, which no reader takes now."""
    return [
        run_start(RUN, "build it", isolation=isolation),
        *effect(RUN, 2, "op-boot", EffectKind.BOOTSTRAP),
        *effect(RUN, 5, "op-step", EffectKind.WORKER_STEP),
        *effect(RUN, 8, "op-commit", EffectKind.TASK_COMMIT),
        run_end(RUN, 11),
    ]


def _kinds(evs: list[JournalEvent], *, accounted: bool = False) -> list[str]:
    records = FakeRecords(runs={RUN: tuple(evs)})
    return [e.kind for e in _run_effects(RUN, 1, records, accounted)]


def test_a_coder_runs_notice_lists_no_commit() -> None:
    assert not WORKTREE_KINDS & set(_kinds(_journal()))


def test_a_step_is_still_listed() -> None:
    """The recorded `ambiguous_exec` run listed its `action_exec`, an effect in
    the world; that kind retired, and a worker step, which no fork is assumed to undo, is what
    is listed now."""
    assert _kinds(_journal()) == ["worker_step"]


def test_an_in_place_runs_writes_in_a_turn_with_no_snapshot_are_listed() -> None:
    """In place, the run wrote the person's checkout; with no snapshot of its turn, nothing
    restores it or names it."""
    assert "task_commit" in _kinds(_journal("in_place"))


def test_an_in_place_runs_writes_in_a_snapshotted_turn_are_not_listed() -> None:
    """D162 decision 3, the answer that moved from T-3.99: the rewind restores that code or the
    kept-code notice names it, so the effects list would say it twice."""
    listed = _kinds(_journal("in_place"), accounted=True)
    assert not WORKTREE_KINDS & set(listed)
