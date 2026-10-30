"""The brand's pictures (`tui/brand.py`): the bird, the gold wordmark, and how the front page
gives way by width, centred and never wrapping its art.
"""

from __future__ import annotations

import pytest
from overlay_fakes import Wire, hosted
from tui_frames import SID

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.brand import (
    BIRD,
    SPACED,
    TAGLINE,
    WORDMARK,
    bird,
    front_mark,
    wordmark,
)
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.panes import RoleFrame
from chickadee_cli.tui.theme import palette

pytestmark = pytest.mark.anyio

COLOURS = palette("truecolor")
HALF_BLOCKS = {" ", "▀", "▄"}
WORD_WIDTH = len(WORDMARK[0])
BIRD_WIDTH = len(BIRD[0])
BIRD_ROWS = len(BIRD) // 2
STACKED = BIRD_ROWS + 1 + 3 + 1
"""The bird, a blank row, the wordmark's three, the tagline."""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_the_bird_is_five_rows_of_half_blocks() -> None:
    rows = bird()
    assert len(rows) == 5 and {len(r.plain) for r in rows} == {14}
    assert set("".join(r.plain for r in rows)) <= HALF_BLOCKS


def test_the_wordmark_is_three_rows_of_gold_block_letters() -> None:
    assert WORD_WIDTH == 36, "a 120-column screen's front page holds it under the bird"
    rows = wordmark(COLOURS["gold"])
    assert len(rows) == 3 and {r.cell_length for r in rows} == {WORD_WIDTH}
    assert set("".join(r.plain for r in rows)) <= {"░", "█", "▀", "▄"}


def _plain(width: int, *, pictures: bool = True) -> list[str]:
    return [row.plain for row in front_mark(COLOURS, width=width, pictures=pictures)]


@pytest.mark.parametrize(
    "width", [WORD_WIDTH, WORD_WIDTH - 1, BIRD_WIDTH, BIRD_WIDTH - 1, 0, 200]
)
def test_the_front_mark_never_outgrows_its_width(width: int) -> None:
    rows = _plain(width)
    assert sum(row.strip() == TAGLINE for row in rows) == 1
    assert all(len(row) <= max(width, len(SPACED)) for row in rows)


def _centre(row: str) -> float:
    text = row.strip()
    return row.index(text) + len(text) / 2


def test_the_bird_sits_centred_over_the_wordmark() -> None:
    rows = _plain(200)
    assert len(rows) == STACKED and SPACED not in "".join(rows)
    assert {len(r) for r in rows[BIRD_ROWS + 1 : BIRD_ROWS + 4]} == {WORD_WIDTH}
    indent = (WORD_WIDTH - BIRD_WIDTH) // 2
    assert all(
        r[:indent] == " " * indent and len(r) == indent + BIRD_WIDTH for r in rows[:BIRD_ROWS]
    )
    assert abs(_centre(rows[-1]) - WORD_WIDTH / 2) <= 1, "the tagline centred under it too"


def test_narrower_the_name_is_text_under_the_bird_then_alone() -> None:
    text = _plain(WORD_WIDTH - 1)
    assert [r.strip() for r in text[-2:]] == [SPACED, TAGLINE] and len(text) == BIRD_ROWS + 3
    assert [r.strip() for r in _plain(BIRD_WIDTH - 1)] == [SPACED, TAGLINE]


def test_without_half_blocks_the_name_is_text() -> None:
    assert [r.strip() for r in _plain(200, pictures=False)] == [SPACED, TAGLINE]


@pytest.mark.parametrize("size", [(190, 40), (120, 34)])
async def test_the_front_page_is_centred_in_its_pane(size: tuple[int, int]) -> None:
    app = ChickadeeApp(feed=Feed(SID), repo="r", depth="truecolor", encoding="utf-8")
    hosted(app, Wire())
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        code = app.query_one("#diff", RoleFrame)
        region = code.scrollable_content_region
        lines = str(code.body.content).splitlines()
        assert all(len(line) <= region.width for line in lines), "the art is never wrapped"
        drawn = [i for i, line in enumerate(lines) if line.strip()]
        above, below = drawn[0], region.height - 1 - drawn[-1]
        assert abs(above - below) <= 1, "centred top to bottom"
        word = lines[drawn[0] + BIRD_ROWS + 1]
        left, right = len(word) - len(word.lstrip()), region.width - len(word.rstrip())
        assert word.strip() == WORDMARK[0] and abs(left - right) <= 1


async def test_an_ascii_terminal_gets_the_name_as_text() -> None:
    app = ChickadeeApp(feed=Feed(SID), repo="r", depth="16", encoding="ascii")
    hosted(app, Wire())
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        lines = str(app.query_one("#diff", RoleFrame).body.content).splitlines()
        drawn = [line.strip() for line in lines if line.strip()]
        assert drawn[:2] == [SPACED, TAGLINE]
