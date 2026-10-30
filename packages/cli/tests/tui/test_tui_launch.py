"""The screen as it first opens (the owner's D7.9), through the
pilot and from the plain functions.

What these pin: while nothing is said and no run is open the screen is its front page, and the
person's first words give way to the mockup, a run to the mockup, while a session with
turns opens on its state; the CODE pane is titled CHICKADEE with the client's version and says
where the session runs, the rungs' models, and the server's recent sessions (this one left
out, newest first, three at most), the tree's state and the sandbox's level unknown;
Zeus's pane greets the person, centred ("Hi, I'm Chickadee." and "What shall we build
today?"), never a `zeus` line, with no suggestions; `Tab` moves the focus on, as everywhere.
"""

from __future__ import annotations

import pytest
from overlay_fakes import Wire, hosted
from textual.content import Content
from tui_frames import SID, assistant, episode, user

from chickadee_cli import __version__
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.panes import RoleFrame
from chickadee_cli.tui.panes.diff import (
    FRONT_META,
    RECENT_SHOWN,
    front_sessions,
    launch_body,
)
from chickadee_cli.tui.panes.zeus import WELCOME
from chickadee_cli.tui.theme import ASCII, UNICODE, Glyphs, palette
from chickadee_wire.models import ModelWiring, SessionInfo

pytestmark = pytest.mark.anyio

SIZE = (104, 33)
COLOURS = palette("truecolor")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _info(sid: str, when: str, title: str | None = None) -> SessionInfo:
    return SessionInfo(session_id=sid, last_seq=1, updated_at=when, title=title)


RECENT = (
    _info("S-old", "2026-10-20T09:00:00Z", "old work"),
    _info(SID, "2026-10-23T12:00:00Z", "this one"),
    _info("S-new", "2026-10-22T09:00:00Z", "new work"),
    _info("S-mid", "2026-10-21T09:00:00Z", "mid work"),
    _info("S-older", "2026-10-19T09:00:00Z"),
)


def test_the_screen_lands_until_words_are_said_or_a_run_opens() -> None:
    feed = Feed(SID)
    assert feed.state.landing
    feed.sent("hello")
    assert not feed.state.landing, "words on their way count"
    said = Feed(SID)
    said.apply(user("hello", 1))
    assert not said.state.landing
    replied = Feed(SID)
    replied.apply(assistant("hi", 1))
    assert not replied.state.landing, "a session with turns opens on its state"
    ran = Feed(SID)
    ran.apply(episode(1))
    assert not ran.state.landing


def test_the_front_page_is_the_mark_and_the_recent_sessions() -> None:
    """Where the session runs, its tree, its sandbox and the models are the header's;
    the front page says them no more."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    feed.snapshot(ModelWiring(catalog=["big"], rungs={"high-param": "big"}, switches=[]))
    body = launch_body(feed.state, COLOURS, recent=RECENT, pictures=False, width=80)
    lines = [line.strip() for line in body.plain.splitlines()]
    assert lines[:3] == ["C H I C K A D E E", "design·plan·code", ""]
    assert lines[3] == "RECENT", "nothing between the mark and the sessions"
    assert not any(line.split()[:1] in (["repo"], ["tree"], ["sandbox"]) for line in lines)
    recent = lines[lines.index("RECENT") + 1 : lines.index("RECENT") + 1 + RECENT_SHOWN]
    assert [line.split()[0] for line in recent] == ["S-new", "S-mid", "S-old"]
    assert "mid work" in recent[1] and "2026-10-22 09:00" in recent[0]


def test_the_front_page_is_centred_where_it_fits_its_sessions_left_aligned_under_it() -> None:
    """Centred top to bottom; the sessions and the hint one block, centred side to side
    and left-aligned."""
    body = launch_body(
        Feed(SID).state, COLOURS, recent=RECENT, pictures=False, width=80, height=30
    )
    lines = body.plain.splitlines()
    drawn = [i for i, line in enumerate(lines) if line.strip()]
    assert abs(drawn[0] - (30 - 1 - drawn[-1])) <= 1, "centred top to bottom"
    block = lines[lines.index(next(ln for ln in lines if ln.strip() == "RECENT")) :]
    margin = len(block[0]) - len(block[0].lstrip())
    right = 80 - max(len(line) for line in block)
    assert margin > 0 and abs(margin - right) <= 1, "centred side to side"
    assert all(line.startswith(" " * margin) for line in block if line.strip())
    tall = launch_body(
        Feed(SID).state, COLOURS, recent=RECENT, pictures=False, width=80, height=3
    )
    assert tall.plain.splitlines()[0].strip(), "taller than the pane: from the top, to scroll"


@pytest.mark.parametrize("pictures", [True, False])
def test_the_mark_is_centred_side_to_side_whatever_the_sessions_width(pictures: bool) -> None:
    """T-5.38, the owner's report: a recent session wider than the mark set the page's left
    edge, and the mark sat left of centre. The mark is centred on the pane on its own."""
    wide = tuple(
        _info(f"S-{n}", f"2026-10-2{n}T09:00:00Z", "a session title wider than the mark " * 2)
        for n in range(3)
    )
    body = launch_body(
        Feed(SID).state, COLOURS, recent=wide, pictures=pictures, width=120, height=30
    )
    lines = body.plain.splitlines()
    mark = lines[: lines.index(next(ln for ln in lines if ln.strip() == "RECENT")) - 1]
    widest = max((ln for ln in mark if ln.strip()), key=lambda ln: len(ln.strip()))
    left = len(widest) - len(widest.lstrip())
    assert abs(left - (120 - len(widest))) <= 1, "the mark's widest row centred on the pane"
    tagline = next(ln for ln in mark if ln.strip() == "design·plan·code")
    centre = len(tagline) - len(tagline.lstrip()) + len("design·plan·code") / 2
    assert abs(centre - 60) <= 1


def test_before_the_routes_answer_it_says_so() -> None:
    lines = launch_body(Feed(SID).state, COLOURS, recent=None, pictures=False, width=80).plain
    assert "reading the sessions…" in lines


async def test_the_screen_opens_on_its_front_page_and_words_give_way_to_03() -> None:
    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, repo="project-chickadee", depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        code = app.query_one("#diff", RoleFrame)
        title = Content.from_markup(str(code.border_title)).plain
        assert title.startswith("CHICKADEE") and title.endswith(f"v{__version__}")
        assert "RECENT" in str(code.body.content)
        greeting = str(app.zeus.welcome.render())
        assert app.zeus.welcome.display and greeting.splitlines() == list(WELCOME)
        assert "try" not in greeting.lower(), "no suggestions (the owner's call, 2026-10-26)"
        assert not app.state.zeus.entries, "the greeting is the screen's, not a zeus line"
        app.take(user("fix the retries", 1))
        await pilot.pause()
        title = Content.from_markup(str(code.border_title)).plain
        assert title.startswith("CODE · idle"), "03-idle.png"
        assert not app.zeus.welcome.display


async def test_on_the_front_page_tab_moves_the_focus_and_never_fills_the_box() -> None:
    app = ChickadeeApp(feed=Feed(SID), depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        assert app.zeus.composer.text == "", "no line to try is put there"
        assert app.focused is not app.zeus.composer, "Tab moves the focus on"


async def test_past_the_front_page_tab_moves_the_focus_as_before() -> None:
    feed = Feed(SID)
    feed.apply(user("hello", 1))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("tab")
        await pilot.pause()
        assert app.zeus.composer.text == "" and app.focused is not app.zeus.composer


def test_the_chosen_session_is_marked_and_no_row_moves() -> None:
    """The row Up and Down chose is marked `▸`, every row two columns in, so a choice
    moves nothing; with sessions to choose the hint names the keys."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    assert [s.session_id for s in front_sessions(feed.state, RECENT)] == [
        "S-new", "S-mid", "S-old"
    ]  # fmt: skip

    def rows(chosen: int | None, glyphs: Glyphs = UNICODE) -> list[str]:
        body = launch_body(
            feed.state, COLOURS, recent=RECENT, pictures=False, width=80, chosen=chosen,
            glyphs=glyphs,
        )  # fmt: skip
        return body.plain.splitlines()

    none, mid = rows(None), rows(1)
    start = next(i for i, line in enumerate(none) if line.strip() == "RECENT") + 1
    assert [line.split()[0] for line in mid[start : start + 3]] == ["S-new", "▸", "S-old"]
    assert mid[start + 1].split()[1] == "S-mid"
    assert [len(a) for a in none] == [len(b) for b in mid], "a choice moves nothing"
    assert "↑↓ choose · Enter open" in none[-1]
    assert "> S-mid" in rows(1, ASCII)[start + 1] and "Up/Down choose" in rows(1, ASCII)[-1]
    alone = launch_body(feed.state, COLOURS, recent=(), pictures=False, width=80).plain
    assert "choose" not in alone, "nothing to choose, no keys named"


async def test_up_and_down_choose_a_recent_session_and_enter_opens_it() -> None:
    """T-5.45, the owner's ask: "from tui we can open by selecting up and down". Down from
    none chooses the newest, each end holds, Up from the newest chooses none again; Enter
    reopens the screen on the one chosen."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    app = ChickadeeApp(feed=feed, repo="project-chickadee", depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._recent(RECENT)  # pyright: ignore[reportPrivateUsage] # what on_mount reads
        code = app.code
        for key, at in (("down", 0), ("down", 1), ("down", 2), ("down", 2), ("up", 1)):
            await pilot.press(key)
            await pilot.pause()
            assert code.front_chosen == at, key
        assert "▸ S-mid" in str(code.body.content)
        await pilot.press("up", "up")
        await pilot.pause()
        assert code.front_chosen is None, "Up from the newest: back to the box"
        await pilot.press("down", "down", "enter")
        await pilot.pause()
    assert app.next_session == "S-mid"


async def test_words_typed_drop_the_choice_and_enter_sends_them() -> None:
    feed = Feed(SID)
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._recent(RECENT)  # pyright: ignore[reportPrivateUsage]
        await pilot.press("down")
        await pilot.pause()
        assert app.code.front_chosen == 0
        await pilot.press("h", "i")
        await pilot.pause()
        assert app.code.front_chosen is None, "words typed drop the choice"
        await pilot.press("enter")
        await pilot.pause()
        assert app.next_session is None, "Enter sends the words, it opens nothing"
        assert app.zeus.composer.text == ""


def test_a_session_with_nothing_said_is_not_listed() -> None:
    """Every screen makes a session as it opens, so earlier launches' empty ones filled
    the rows, and opening one drew the front page again; nothing in them, they are left out,
    the screen's own new one before its info arrives with them."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    empty = SessionInfo(session_id="S-empty", last_seq=0, updated_at="2026-10-29T09:00:00Z")
    listed = front_sessions(feed.state, (empty, *RECENT))
    assert [s.session_id for s in listed] == ["S-new", "S-mid", "S-old"]
    body = launch_body(feed.state, COLOURS, recent=(empty,), pictures=False, width=80).plain
    assert "S-empty" not in body and "none yet" in body
    fresh = SessionInfo(session_id="S-fresh", last_seq=0)
    before_its_info = front_sessions(Feed("S-fresh").state, (fresh, *RECENT))
    assert "S-fresh" not in [s.session_id for s in before_its_info], "the screen's own"


def test_a_session_that_only_continued_open_work_is_not_listed() -> None:
    """T-5.48, the owner's report: "every session is showing untitled". A screen opened where
    there is open work makes its session continuing it, and the adoption is written into it: a
    record, no turn, no title. The newest, they filled the rows; only a titled one is listed."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    launches = tuple(
        SessionInfo(session_id=f"S-launch{i}", last_seq=1, updated_at=f"2026-10-30T0{i}:00:00Z")
        for i in range(3)
    )
    listed = front_sessions(feed.state, (*launches, *RECENT))
    assert [s.session_id for s in listed] == ["S-new", "S-mid", "S-old"]
    body = launch_body(feed.state, COLOURS, recent=launches, pictures=False, width=80).plain
    assert "untitled" not in body and "none yet" in body


async def test_with_the_code_pane_focused_up_down_and_enter_still_open() -> None:
    """T-5.46, the owner's report: "i can't seem to select any session". A click on the CODE
    pane, or F4, gives it the focus, and there the keys did nothing."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._recent(RECENT)  # pyright: ignore[reportPrivateUsage]
        await pilot.press("f4")
        await pilot.pause()
        assert app.focused is app.code
        await pilot.press("down", "down")
        await pilot.pause()
        assert app.code.front_chosen == 1
        await pilot.press("enter")
        await pilot.pause()
    assert app.next_session == "S-mid"


async def test_a_click_on_a_row_opens_its_session() -> None:
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._recent(RECENT)  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        # Where the screen draws the row: the cell whose click opens `S-old`.
        width, height = SIZE
        cells = [(x, y) for y in range(height) for x in range(width)]
        at = next(
            xy for xy in cells if app.screen.get_style_at(*xy).meta.get(FRONT_META) == "S-old"
        )
        await pilot.click(offset=at)
        await pilot.pause()
    assert app.next_session == "S-old"


async def test_past_the_front_page_the_code_pane_takes_no_front_key() -> None:
    feed = Feed(SID)
    feed.apply(user("hello", 1))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._recent(RECENT)  # pyright: ignore[reportPrivateUsage]
        assert not app.code.check_action("front", (1,))
        assert not app.code.check_action("front_open", ())
        await pilot.press("f4", "down", "enter")
        await pilot.pause()
        assert app.code.front_chosen is None and app.next_session is None


async def test_with_the_code_pane_focused_the_rows_keep_their_colours() -> None:
    """T-5.47, the owner's report: with the pane focused, every row's id, title and date turned
    white and underlined, Textual's link style for a `@click` span. Only the chosen id is Zeus's
    colour; nothing is underlined."""
    feed = Feed(SID)
    feed.snapshot(_info(SID, "2026-10-23T12:00:00Z"))
    app = ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")
    hosted(app, Wire(listed=RECENT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        app._recent(RECENT)  # pyright: ignore[reportPrivateUsage]
        await pilot.press("f4", "down", "down")
        await pilot.pause()
        width, height = SIZE
        cells = {
            (x, y): app.screen.get_style_at(x, y) for y in range(height) for x in range(width)
        }
        rows = {xy: st for xy, st in cells.items() if st.meta.get(FRONT_META)}
        assert rows, "the rows carry their session"
        assert not any(st.underline for st in rows.values()), "no row is underlined"
        zeus = COLOURS["zeus"].lower()

        def hexes(sid: str) -> set[str]:
            return {
                st.color.triplet.hex
                for st in rows.values()
                if st.meta[FRONT_META] == sid and st.bold and st.color and st.color.triplet
            }

        assert hexes("S-mid") == {zeus}, "the chosen id in Zeus's colour"
        assert not hexes("S-new") and not hexes("S-old"), "the others' ids are not bold"
