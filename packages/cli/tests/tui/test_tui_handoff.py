"""Hand off (`/handoff`, and on leaving), through the pilot.

What these pin: it reads `/handoff` again as it opens and draws the worktree against its base,
its commits and its committed diff's files and lines; the five choices in `CHOICES`' order,
in the engine's words, one
the engine refuses dimmed with why and doing nothing; `discard` asks again; a choice carries
out `finalize` and shows `/handoff <choice>`'s own lines, here and in Zeus's pane; `d` opens
the diff screen; `Ctrl+Q` with work not handed off opens it first, its `Esc` leaving the
worktree pending and printing so once the screen has gone, while work handed off or kept
leaves at once; and the REPL and the TUI ask the same rule (`asked_on_leaving`).
"""

from __future__ import annotations

import pytest
from overlay_fakes import Wire, handoff_summary, hosted
from recordings import run_frames
from tui_frames import SID, episode

from chickadee_cli.commands import LEAVE_PENDING
from chickadee_cli.render.diff import CHOICES, asked_on_leaving
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.keys import ScreenView
from chickadee_cli.tui.overlays.handoff import DISCARD_AGAIN, Handoff
from chickadee_wire.models import HandoffResult

pytestmark = pytest.mark.anyio

SIZE = (120, 40)
DIFF = "\n".join(
    [
        "diff --git a/app.py b/app.py", "--- a/app.py", "+++ b/app.py", "@@ -1,2 +1,3 @@",
        " keep", "-old", "+new", "+more",
    ]
)  # fmt: skip


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _app(summary_given: bool = True) -> ChickadeeApp:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        feed.apply(frame)
    if summary_given:
        feed.snapshot(handoff_summary(diff=DIFF, refused={"apply": "the source moved"}))
    return ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")


def _result(choice: str = "squash") -> HandoffResult:
    return HandoffResult.model_validate(
        {
            "branch": "ccr-1", "choice": choice, "commits_applied": ["d" * 40],
            "detail": "squashed onto main", "operation_id": "op-h", "source_branch": "main",
            "source_head": "a" * 40, "worktree_path": "/wt/s-1",
        }
    )  # fmt: skip


def _handoff(app: ChickadeeApp) -> Handoff:
    assert isinstance(app.overlay, Handoff)
    return app.overlay


def test_the_repl_and_the_screen_ask_on_leaving_by_one_rule() -> None:
    assert asked_on_leaving(handoff_summary())
    assert not asked_on_leaving(handoff_summary(commits=0)), "nothing to choose over"
    assert not asked_on_leaving(handoff_summary(handoff="keep")), "kept earlier: not yet"


async def test_it_draws_the_worktree_its_commits_its_diff_and_the_choices() -> None:
    app = _app()
    reader, _sent = hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("handoff")
        await pilot.pause()
        assert "handoff" in reader.refreshed, "read again as it opens"
        body = str(_handoff(app).content.render())
        assert "wt/s-1 against main   1 file +2 \u22121" in body, "the mockup's minus sign"
        assert "commits  0000000   1111111" in body
        drawn = [
            line[4:].split()[0] for line in body.splitlines() if line[:3] in ("(*)", "( )")
        ]
        assert drawn == list(CHOICES), "CHOICES' order"
        assert "refused: the source moved" in body
        assert "squash   put the 2 commit(s) on main as one commit" in body, "engine's words"
        assert "discard  delete the worktree. Asks again." in body
        assert "never pushes" in str(_handoff(app).query_one("#box").border_title)


async def test_a_refused_choice_does_nothing_and_discard_asks_again() -> None:
    app = _app()
    wire = Wire(result=_result("discard"))
    hosted(app, wire)
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("handoff")
        await pilot.pause()
        await pilot.press("down", "enter")  # `apply`, refused
        assert wire.finalized == []
        await pilot.press("down", "down", "down", "enter")
        assert DISCARD_AGAIN in str(_handoff(app).content.render()) and wire.finalized == []
        await pilot.press("enter")
        await pilot.pause()
        assert wire.finalized == ["discard"]


async def test_a_choice_shows_handoff_s_own_lines_here_and_in_zeus_s_pane() -> None:
    app = _app()
    wire = Wire(result=_result())
    hosted(app, wire)
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("handoff")
        await pilot.pause()
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        assert wire.finalized == ["squash"]
        body = str(_handoff(app).content.render())
        assert "squash: squashed onto main" in body and "nothing was pushed" in body
        said = [e.text or "" for e in app.state.zeus.entries]
        assert any("squash: squashed onto main" in line for line in said)
        await pilot.press("escape")
        assert app.overlay is None


async def test_d_opens_the_diff_screen() -> None:
    app = _app()
    hosted(app, Wire())
    async with app.run_test(size=SIZE) as pilot:
        app.open_overlay("handoff")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert app.overlay is None and app.view is ScreenView.DIFF


async def test_ctrl_q_with_work_not_handed_off_asks_first_and_esc_leaves_it_pending() -> None:
    app = _app(summary_given=False)
    hosted(app, Wire(summary=handoff_summary(diff=DIFF)))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert _handoff(app).leaving and app.is_running, "D7.8"
        await pilot.press("escape")
        await pilot.pause()
    assert not app.is_running
    assert any(LEAVE_PENDING in line for line in app.farewell), "said once the screen is gone"


async def test_ctrl_q_asks_again_after_the_leaving_overlay_went_to_the_diff() -> None:
    """`d` from the leaving overlay shows the diff and does not leave; the next `Ctrl+Q` asks
    again rather than leaving with the work not handed off."""
    app = _app(summary_given=False)
    hosted(app, Wire(summary=handoff_summary(diff=DIFF)))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert app.overlay is None and app.view is ScreenView.DIFF
        await pilot.press("ctrl+q")
        await pilot.pause()
        assert _handoff(app).leaving and app.is_running, "asked again"
        await pilot.press("escape")
        await pilot.pause()
    assert not app.is_running


async def test_ctrl_q_with_nothing_to_choose_over_leaves_at_once() -> None:
    app = _app(summary_given=False)
    hosted(app, Wire(summary=handoff_summary(handoff="keep")))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()
    assert not app.is_running and app.farewell == []
