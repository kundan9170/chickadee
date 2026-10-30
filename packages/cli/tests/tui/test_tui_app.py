"""The TUI's screen (layout E), driven by Textual's pilot at the
mockup's size, 104 by 33.

What these pin: the panes sit where layout E puts them, in the mockup's proportions; F1 to F3
and Tab move focus, and the focused pane alone is double-bordered; each frame's title and
right label come from the state, the worker's naming its rung and never a model; a pane with
nothing to show is faint and says what will appear; `show` redraws only what changed; the
pulse runs only while a turn is in flight; an unbuilt key does nothing; and a narrow
terminal still draws.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.content import Content
from tui_frames import SID, assistant, episode, progress, run_frames, run_record, user

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.keys import ScreenView
from chickadee_cli.tui.model import Pane, TuiState
from chickadee_cli.tui.panes import IDLE, RoleFrame, border_title
from chickadee_cli.tui.pulse import PULSE_DIM, in_flight
from chickadee_serve.frames import JsonValue
from chickadee_wire.models import CatalogModel, Liveness, ModelWiring

pytestmark = pytest.mark.anyio

SIZE = (104, 33)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _feed(upto: int | None = None) -> Feed:
    feed = Feed(SID)
    if upto is None:
        return feed
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


def _app(state: TuiState | None = None) -> ChickadeeApp:
    return ChickadeeApp(state, repo="project-chickadee", depth="truecolor", encoding="utf-8")


def _frame(app: ChickadeeApp, role: str) -> RoleFrame:
    return app.query_one(f"#{role}", RoleFrame)


def _title(app: ChickadeeApp, role: str) -> str:
    """The border title as drawn: Textual keeps it as markup."""
    title = _frame(app, role).border_title
    return Content.from_markup(str(title)).plain if title is not None else ""


def _body(app: ChickadeeApp, role: str) -> str:
    content = _frame(app, role).body.content
    return content.plain if isinstance(content, Content) else str(content)


# -- layout E --------------------------------------------------------------------------------


async def test_the_panes_sit_where_layout_e_puts_them() -> None:
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        regions = {r: _frame(app, r).region for r in ("zeus", "diff", "athena", "worker")}
    # ui/ SCREENS.base: Zeus 0..36, CODE 36..76, the right column 76..104; rows 6..32, under
    # the five-row header and its rule.
    assert (regions["zeus"].x, regions["zeus"].width) == (0, 36)
    assert (regions["diff"].x, regions["diff"].width) == (36, 40)
    assert (regions["athena"].x, regions["athena"].width) == (76, 28)
    assert (regions["athena"].y, regions["athena"].height) == (6, 10)
    assert (regions["worker"].y, regions["worker"].height) == (16, 16)


async def test_the_proportions_scale_with_the_terminal() -> None:
    app = _app()
    async with app.run_test(size=(208, 50)) as pilot:
        await pilot.pause()
        widths = [_frame(app, r).region.width for r in ("zeus", "diff", "athena")]
    assert widths == [72, 80, 56], "36 : 40 : 28, doubled"


async def test_a_narrow_terminal_still_draws() -> None:
    """80 columns is stacked (the owner's D6.4): Zeus over the CODE tab, full width, and
    Athena and the worker as tabs named on the tab line."""
    app = _app(_feed(30).state)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert all(_frame(app, r).region.width == 80 for r in ("zeus", "diff"))
        assert not _frame(app, "athena").display
        assert "PLANNER" in str(app.query_one("#tabs").render())


# -- focus -----------------------------------------------------------------------------------


def _borders(app: ChickadeeApp) -> dict[str, str]:
    return {
        r: _frame(app, r).styles.border_top[0] for r in ("zeus", "diff", "athena", "worker")
    }


async def test_zeus_is_focused_at_start_and_f1_to_f3_move_focus() -> None:
    app = _app(_feed(30).state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert _borders(app) == {
            "zeus": "double", "diff": "round", "athena": "round", "worker": "round",
        }  # fmt: skip
        assert app.focused is app.zeus.composer, "§2: his pane's focus is the box you type in"
        for key, role in (("f2", "athena"), ("f3", "worker"), ("f1", "zeus")):
            await pilot.press(key)
            await pilot.pause()
            focused = app.zeus.composer if role == "zeus" else _frame(app, role)
            assert app.focused is focused, key
            assert [r for r, b in _borders(app).items() if b == "double"] == [role]
        await pilot.press("f2", "tab")
        await pilot.pause()
        assert app.focused is _frame(app, "worker"), "Tab goes on in the screen's order"


async def test_the_focused_border_is_drawn_in_the_role_s_colour() -> None:
    app = _app(_feed(30).state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        colour = _frame(app, "athena").styles.border_top[1]
    assert colour.hex.lower() == "#3fd0c0"


# -- what each frame says --------------------------------------------------------------------


async def test_an_empty_session_draws_faint_panes_that_say_what_will_appear() -> None:
    # Before any words the CODE pane is the front page; once
    # the person has spoken and no task runs, it is 03-idle.png's.
    feed = Feed(SID)
    feed.apply(user("hello", 1))
    app = _app(feed.state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert [r for r in ("diff", "athena", "worker") if _frame(app, r).has_class(IDLE)] == [
            "diff", "athena", "worker",
        ]  # fmt: skip
        assert not _frame(app, "zeus").has_class(IDLE)
        assert _title(app, "diff").startswith("CODE · idle")
        assert _title(app, "athena").endswith("no plan")
        assert _body(app, "athena") == "idle\n\nThe plan shows here.", "as the worker's says it"
        assert "Starts once the plan is approved." in _body(app, "worker")
        assert _body(app, "diff").startswith("No changes yet.")


async def test_the_titles_come_from_the_state() -> None:
    state = _feed(30).state
    state.zeus.cost_usd = 0.31
    app = _app(state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert _title(app, "zeus").startswith("DESIGNER")
        assert _title(app, "zeus").endswith("$0.31")
        assert _title(app, "diff").startswith("CODE · T1 · live")
        assert _title(app, "diff").endswith("+1 \u22121"), "the mockup's minus sign"
        assert _title(app, "athena").endswith(f"${state.athena.cost_usd:.2f}")
        assert not _frame(app, "athena").has_class(IDLE)


async def test_a_spend_no_record_carries_is_not_drawn_as_zero() -> None:
    """Absent is unknown, never $0: with no `spend` line Zeus's title names no cost,
    and with no `model_call` of hers Athena's names none (found in the replay's video: the demo
    session has no spend line, and his title read `$0.00`)."""
    state = _feed(30).state
    state.athena.cost_usd = None
    assert state.zeus.cost_usd is None, "the recorded run carries no spend line of his"
    app = _app(state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert "$" not in _title(app, "zeus")
        assert "$" not in _title(app, "athena")


async def test_the_worker_s_label_is_its_spend_and_context_never_its_model() -> None:
    """The worker's title says what it spent and how full its prompt was, as the other
    panes' do; its model is the header's. Narrow, the context goes and the spend stays."""
    call: dict[str, JsonValue] = {
        "role": "worker", "rung": "low-param", "model": "small", "cost_usd": 0.25,
        "request": "r", "stop_reason": "end_turn", "tokens_in": 250, "tokens_out": 1,
    }  # fmt: skip
    feed = _feed(6)
    feed.snapshot(
        ModelWiring(
            catalog=["small"], rungs={"low-param": "small"}, switches=[],
            roles={"worker": "low-param"},
            models={"small": CatalogModel(
                context_window=1000, max_output_tokens=100, provider="p"
            )},
        )
    )  # fmt: skip
    feed.apply(run_record("model_call", call, 7, task_id="T1"))
    titles: list[str] = []
    for size in ((208, 50), SIZE):
        app = _app(feed.state)
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            titles.append(_title(app, "worker"))
    wide, narrow = titles
    assert wide.startswith("WORKER") and wide.endswith("$0.25 · ctx ██░░░░░░ 25%")
    assert narrow.startswith("WORKER") and narrow.endswith("$0.25"), "the spend, never cut"
    assert "small" not in wide and "low-param" not in wide, "the model is the header's"


def test_a_label_that_does_not_fit_is_cut_and_marked_never_the_title() -> None:
    left, right = Content("WORKER · coder"), Content("low-param")
    title = border_title(left, right, 28, double=False).plain
    assert title.startswith("WORKER · coder") and "low" in title and title.endswith("…")


async def test_after_the_last_task_the_diff_is_the_last_task_s_not_live() -> None:
    app = _app(_feed(56).state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert _title(app, "diff").startswith("CODE · T2 · last task")


# -- redraws ---------------------------------------------------------------------------------


async def test_show_redraws_only_the_panes_it_is_told_about() -> None:
    state = _feed(30).state
    app = _app(state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        before = _title(app, "athena")
        state.zeus.cost_usd = 9.99
        state.athena.cost_usd = 9.99
        app.show(state, {Pane.ZEUS})
        await pilot.pause()
        assert _title(app, "zeus").endswith("$9.99")
        assert _title(app, "athena") == before, "not named, not redrawn"
        app.show(state, {Pane.ATHENA})
        await pilot.pause()
        assert _title(app, "athena").endswith("$9.99")


# -- the pulse (never a spinner forever) -----------------------------------------------


async def test_the_pulse_runs_only_while_a_turn_is_in_flight() -> None:
    feed = _feed()
    app = _app(feed.state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        assert not app.pulse.running
        feed.apply(progress("thinking"))
        app.show(feed.state, {Pane.ZEUS})
        assert app.pulse.running
        await pilot.pause(1.2)
        assert app.pulse.ticks >= 1, "it pulsed while the turn was in flight"
        feed.apply(assistant("done", 1))  # the turn's record: the hint is over
        app.show(feed.state, {Pane.ZEUS})
        assert not app.pulse.running
        assert not app.screen.has_class(PULSE_DIM), "stopped lit, never stuck dim"
        ticks = app.pulse.ticks
        await pilot.pause(0.7)
        assert app.pulse.ticks == ticks, "nothing ticks with nothing in flight"


def test_a_session_that_is_not_live_never_pulses() -> None:
    state = _feed(30).state
    assert in_flight(state)
    for dead in (
        Liveness(kind="abandoned", last_seq=30),
        Liveness(kind="unknown", last_seq=30),
    ):
        assert not in_flight(replace(state, liveness=dead)), dead.kind
    assert not in_flight(replace(state, liveness=Liveness(kind="awaiting", last_seq=30)))


# -- keys ------------------------------------------------------------------------------------


async def test_an_unbuilt_key_does_nothing() -> None:
    app = _app(_feed(30).state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        # Every key of the base bar is built since Part 7; the diff screen's `u` (undo one
        # hunk) waits on wire request W14.
        await pilot.press("ctrl+d")
        await pilot.pause()
        focused, scope = app.focused, app.full_diff.scope
        await pilot.press("u")
        await pilot.pause()
        assert app.focused is focused and app.full_diff.scope is scope and app.is_running
        assert app.overlay is None and app.view is ScreenView.DIFF


async def test_zeus_s_pane_follows_the_conversation_until_scrolled_up() -> None:
    """Newest at the bottom and followed, however late a reply grows (a Markdown reply
    lays itself out after it mounts); scrolled up, it stays put; back at the end, it follows
    again (found in the live demo: his report landed below the fold)."""
    reply = "Done, in two tasks:\n\n" + "\n".join(f"{n}. a step of the work" for n in range(12))
    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        zeus = app.zeus.transcript
        app.take(assistant("Short.", 1))
        await pilot.pause(0.2)
        first = zeus.children[-1]
        assert zeus.scroll_y == 0 and first.region.y <= zeus.content_region.y + 1, (
            "a conversation that fits is drawn from the top, as the mockups draw it"
        )
        for seq in range(2, 6):
            app.take(assistant(reply, seq))
            await pilot.pause(0.1)
        assert zeus.max_scroll_y > 0, "the conversation is taller than the pane"
        assert zeus.scroll_y == zeus.max_scroll_y, "followed to the newest"
        zeus.scroll_home(animate=False)
        await pilot.pause(0.1)
        app.take(assistant(reply, 6))
        await pilot.pause(0.1)
        assert zeus.scroll_y == 0, "scrolled up, it stays where the person put it"
        zeus.scroll_end(animate=False)
        await pilot.pause(0.1)
        app.take(assistant(reply, 7))
        await pilot.pause(0.2)
        assert zeus.scroll_y == zeus.max_scroll_y, "back at the end, it follows again"
        newest = app.zeus.transcript.children[-1]
        assert newest.region.bottom <= app.zeus.query_one(".foot").region.y, (
            "the newest line is above the foot, never behind it"
        )


async def test_a_streamed_reply_redraws_the_status_line_not_the_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live `text` frame moves only Zeus's status line: the views' visibility is worked out
    again only when the transcript, what an undo hides, or `/clear` changed, so streaming a
    reply late in a long session does not walk every entry for each piece."""
    from chickadee_cli.tui.model import ZeusPane

    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    walked: list[int] = []
    visible = ZeusPane.visible

    def counted(self: ZeusPane) -> list[object]:
        walked.append(len(self.entries))
        return list(visible(self))

    monkeypatch.setattr(ZeusPane, "visible", counted)
    async with app.run_test(size=SIZE) as pilot:
        app.take(user("go", 1))
        app.take(assistant("Short.", 2))
        await pilot.pause(0.1)
        before = len(walked)
        offset = 0
        for piece in ("Reading", " segtree", ".py", " first."):
            app.take(
                progress("text", role="zeus", reply="r1", offset=offset, piece=piece, tokens=1)
            )
            offset += len(piece)
        await pilot.pause(0.1)
        assert "Reading segtree.py first." in str(app.zeus.status.content)
        assert len(walked) == before, "no piece walked the transcript"
        app.take(assistant("Reading segtree.py first.", 3))
        await pilot.pause(0.1)
        assert len(walked) > before, "a new entry places the views again"
