"""The standing rule, checked over every key the screen has (T-5.23 notes, the owner's rule of
2026-10-19; "keys that do not apply in the stage are dimmed, not
hidden"): no built key is left dimmed everywhere, and no dimmed key is bound.

The tables: the base bar and the keys off it (`BAR`, `OFF_BAR`), the diff screen's (`DIFF_BAR`,
`DIFF_OFF_BAR`), the panes' own (`ATHENA_KEYS`, `WORKER_KEYS`), every overlay's foot
(`overlays/*`), and the modals' answers (`STOP_ANSWERS`, `CONFIRM_ANSWERS`). For each:
- a key not built names the part or the wire request that builds it, and nothing binds it;
- a built key is bound where its table says, and lights up in at least one recorded state
  (an empty session, a run under way, a run settled; the base view and the diff, with hunks
  and without), so none is built and never drawn lit.
"""

from __future__ import annotations

import itertools

import pytest
from recordings import run_frames
from tui_frames import SID, episode, stop_asked

from chickadee_cli.tui.app import OVERLAYS, ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.keys import (
    ATHENA_KEYS,
    BACKGROUND_KEYS,
    BAR,
    CODE_KEYS,
    CODE_OFF_KEYS,
    DIFF_BAR,
    DIFF_OFF_BAR,
    FRONT_KEYS,
    OFF_BAR,
    WORKER_KEYS,
    Key,
    ScreenView,
    lit,
)
from chickadee_cli.tui.modals import Answer, answer_bindings
from chickadee_cli.tui.modals.confirm import CONFIRM_ANSWERS
from chickadee_cli.tui.modals.stop import STOP_ANSWERS
from chickadee_cli.tui.model import AgentItem, TuiState
from chickadee_cli.tui.overlays import Overlay, bindings
from chickadee_cli.tui.overlays.command import CommandDetail
from chickadee_cli.tui.overlays.task import TaskDetail
from chickadee_cli.tui.panes.athena import AthenaFrame
from chickadee_cli.tui.panes.background import BackgroundFrame
from chickadee_cli.tui.panes.diff import DiffFrame
from chickadee_cli.tui.panes.worker import WorkerFrame
from chickadee_cli.tui.screens.diff import FullDiff


def _states() -> list[TuiState]:
    """An empty session, T1 under way (its commands run), the run settled, an agent at work in
    the background, a stop open, and one put back as a banner."""
    out = [TuiState()]
    working = TuiState()
    working.background.agents["A1"] = AgentItem("A1", "researcher", "c1")
    out.append(working)
    asked = Feed(SID)
    asked.apply(stop_asked("stop-1", "plan", 1))
    out.append(asked.state)
    waiting = Feed(SID)
    waiting.apply(stop_asked("stop-1", "plan", 1))
    waiting.state.stops_put_back.add("stop-1")
    out.append(waiting.state)
    for upto in (40, 10_000):
        feed = Feed(SID)
        feed.apply(episode(1))
        for frame in run_frames():
            if frame.seq <= upto:
                feed.apply(frame)
        out.append(feed.state)
    return out


STATES = _states()
TABLES: dict[str, tuple[Key, ...]] = {
    "BAR": tuple(k for group in BAR for k in group),
    "OFF_BAR": OFF_BAR,
    "DIFF_BAR": tuple(k for group in DIFF_BAR for k in group),
    "DIFF_OFF_BAR": DIFF_OFF_BAR,
    "ATHENA_KEYS": ATHENA_KEYS,
    "WORKER_KEYS": WORKER_KEYS,
    "CODE_KEYS": CODE_KEYS,
    "CODE_OFF_KEYS": CODE_OFF_KEYS,
    "FRONT_KEYS": FRONT_KEYS,
    "BACKGROUND_KEYS": BACKGROUND_KEYS,
}


def _bound(bindings_of: object) -> set[tuple[str, str]]:
    from textual.binding import Binding

    return {(b.key, b.action) for b in bindings_of if isinstance(b, Binding)}  # type: ignore[union-attr]


BOUND: dict[str, set[tuple[str, str]]] = {
    "app": _bound(ChickadeeApp.BINDINGS),
    "diff": _bound(FullDiff.BINDINGS),
    "athena": _bound(AthenaFrame.BINDINGS),
    "worker": _bound(WorkerFrame.BINDINGS),
    "code": _bound(DiffFrame.BINDINGS),
    "background": _bound(BackgroundFrame.BINDINGS),
}
HOME = {
    "BAR": ("app", "diff"), "OFF_BAR": ("app",), "DIFF_BAR": ("app", "diff"),
    "DIFF_OFF_BAR": ("diff",), "ATHENA_KEYS": ("athena",), "WORKER_KEYS": ("worker",),
    "CODE_KEYS": ("code",), "CODE_OFF_KEYS": ("code",), "FRONT_KEYS": ("code",),
    "BACKGROUND_KEYS": ("background",),
}  # fmt: skip
"""Where each table's built keys are bound: the diff screen binds its own, except the ones
the app binds already to the same action (`keys.diff_bound`)."""


@pytest.mark.parametrize(("table", "key"), [(t, k) for t, ks in TABLES.items() for k in ks])
def test_a_key_is_built_bound_and_lit_somewhere_or_dimmed_unbound_and_named(
    table: str, key: Key
) -> None:
    if key.action is None:
        assert key.builds_in, f"{key.key} names what builds it"
        for home in HOME[table]:
            assert key.key not in {k for k, _ in BOUND[home]}, f"{key.key} is bound in {home}"
        return
    assert any((key.key, key.action) in BOUND[home] for home in HOME[table]), (
        f"{table}'s {key.key} is built and bound"
    )
    views = tuple(ScreenView)
    assert any(
        lit(key, state, view=view, hunks=hunks)
        for state, view, hunks in itertools.product(STATES, views, (False, True))
    ), f"{table}'s {key.key} lights up in some recorded state"


OVERLAY_CLASSES: list[type[Overlay]] = [
    *(cls for cls in OVERLAYS.values() if isinstance(cls, type)),
    TaskDetail,
    CommandDetail,
]


@pytest.mark.parametrize("overlay", OVERLAY_CLASSES, ids=lambda c: c.__name__)
def test_an_overlay_s_foot_binds_its_built_keys_and_none_of_the_rest(
    overlay: type[Overlay],
) -> None:
    bound = _bound(overlay.BINDINGS)
    assert bound == _bound(bindings(overlay.KEYS)), "its bindings are its table's"
    for entry in overlay.KEYS:
        if not entry.keys:
            continue  # a word on how it is used (`Type to filter`), no key
        if entry.actions is None:
            assert entry.builds_in, f"{overlay.__name__}'s {entry.shown} names what builds it"
            assert not any(k in entry.keys for k, _ in bound), entry.shown
        else:
            for key, action in zip(entry.keys, entry.actions, strict=True):
                assert (key, action) in bound


@pytest.mark.parametrize("answers", [STOP_ANSWERS, CONFIRM_ANSWERS], ids=["stop", "confirm"])
def test_a_modal_binds_its_built_answers_and_names_the_rest(
    answers: tuple[Answer, ...],
) -> None:
    bound = {b.key for b in answer_bindings(answers) if hasattr(b, "key")}  # type: ignore[union-attr]
    for answer in answers:
        if answer.choice is None:
            assert answer.wire_request, f"{answer.key} names its wire request"
            assert answer.key not in bound
        else:
            assert answer.key in bound


def test_what_stays_dimmed_after_part_7_is_what_waits_on_the_wire() -> None:
    """The list the Part 7 plan ends on, as D234 leaves it: `u` (W14) and the confirm modal's
    edit, both deferred by D234; `w`; fork.
    Every other key is built."""
    unbuilt = {
        (t, k.key, k.builds_in) for t, ks in TABLES.items() for k in ks if k.action is None
    }
    assert unbuilt == {("DIFF_BAR", "u", "a later release")}
    foot = {
        (cls.__name__, e.shown, e.builds_in)
        for cls in OVERLAY_CLASSES
        for e in cls.KEYS
        if e.actions is None and e.keys
    }
    assert foot == {
        ("Budget", "w", "a later release"),
        ("SessionList", "f", "chickadee fork"),
    }
    assert {(a.key, a.wire_request) for a in CONFIRM_ANSWERS if a.choice is None} == {
        ("e", "a later release")
    }
