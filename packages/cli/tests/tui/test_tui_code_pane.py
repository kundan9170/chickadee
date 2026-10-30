"""The CODE pane: every
task along the top, the open task's files as tabs, a search over the session's files, and a
file no task changed shown read only.

What these pin: the strip lists the board's tasks and the plan's waiting ones, the open one
whole; `Tab` opens the next started task and `←`/`→` the next file; `/` in the pane opens the
search, whose matches say a changed file's lines and task and any other `unchanged`; `Enter`
on a changed file opens its task on that tab, on any other asks the screen to read it; a file
read is drawn numbered and safe, and `Esc` closes it without interrupting Zeus.
"""

from __future__ import annotations

import pytest
from textual.content import Content
from tui_frames import SID, episode, run_frames

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import DiffFile, TuiState
from chickadee_cli.tui.panes.diff import (
    DiffFrame,
    file_lines,
    found_lines,
    strip_tasks,
    task_strip,
)
from chickadee_cli.tui.theme import UNICODE, palette
from chickadee_wire.models import FileText, PlanTaskP, RunDiff

pytestmark = pytest.mark.anyio

COLOURS = palette("truecolor")
SIZE = (120, 34)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _state(upto: int = 10_000) -> TuiState:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed.state


def _app(state: TuiState) -> ChickadeeApp:
    return ChickadeeApp(state, repo="project-chickadee", depth="truecolor", encoding="utf-8")


def _title(app: ChickadeeApp) -> str:
    title = app.code.border_title
    return Content.from_markup(str(title)).plain if title is not None else ""


# -- the strip and the tabs ---------------------------------------------------------


def test_the_strip_lists_every_task_the_open_one_whole() -> None:
    state = _state()
    state.athena.plan_tasks = (PlanTaskP(id="T9", title="Use it in stats everywhere"),)
    tasks = strip_tasks(state, UNICODE)
    assert [t.task_id for t in tasks] == [*state.worker.tasks, "T9"], "the plan's waiting last"
    assert [t.started for t in tasks][-1] is False, "a task not delegated has no diff to open"
    first = tasks[0]
    wide = task_strip(state, UNICODE, COLOURS, first.task_id, 200).plain
    assert f"{first.task_id} {first.title}" in wide, "the open task whole"
    narrow = task_strip(state, UNICODE, COLOURS, first.task_id, 30).plain
    assert all(t.task_id in narrow for t in tasks), "every id, however narrow"


async def test_tab_opens_the_next_task_and_arrows_the_next_file() -> None:
    state = _state()
    started = list(state.worker.tasks)
    assert len(started) >= 2, "the recording has two tasks"
    app = _app(state)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f4")
        assert app.focused is app.code
        assert _title(app).startswith(f"CODE · {state.worker.current}")
        await pilot.press("tab")
        opened = app.code.open_task(app.state)
        assert opened != state.worker.current and opened in started
        assert _title(app).startswith(f"CODE · {opened} · chosen")
        files = [f.path for f in app.state.diff_of(opened).files]
        if len(files) > 1:
            await pilot.press("right")
            assert app.code.path == files[1]
            await pilot.press("left")
            assert app.code.path == files[0]
        await pilot.press("shift+tab")
        assert app.code.open_task(app.state) == state.worker.current


# -- the search ---------------------------------------------------------------------


def test_a_match_says_a_changed_file_s_lines_and_task_and_any_other_unchanged() -> None:
    changed = {"src/client.py": ("T1", DiffFile("src/client.py", 18, 2))}
    found = found_lines(["src/client.py", "README.md"], 0, changed, UNICODE, COLOURS, 46)
    first, second, keys = found.plain.split("\n")
    assert "src/client.py" in first and first.rstrip().endswith(f"+18 {UNICODE.minus}2  T1")
    assert "README.md" in second and second.rstrip().endswith("unchanged")
    assert "Enter open" in keys


async def test_slash_in_the_pane_finds_and_enter_opens_a_changed_file_on_its_tab() -> None:
    state = _state()
    app = _app(state)
    async with app.run_test(size=SIZE) as pilot:
        code = app.code
        code.paths = lambda: ("README.md", "src/client.py", "docs/notes.md")
        await pilot.press("f4", "slash")
        assert code.finding and app.focused is code.find
        assert app.zeus.composer.text == "", "the pane's `/`, not the composer's slash"
        await pilot.press(*"client")
        assert code.found[0] == "src/client.py"
        await pilot.press("enter")
        assert not code.finding
        assert code.path == "src/client.py" and code.opened is None


class _Asking(ChickadeeApp):
    """The screen, keeping the files the CODE pane asks it to read."""

    chosen: list[str]

    def on_diff_frame_file_chosen(self, message: DiffFrame.FileChosen) -> None:
        self.chosen.append(message.path)


async def test_enter_on_an_unchanged_file_asks_for_it_and_esc_closes_the_search() -> None:
    app = _Asking(_state(), repo="project-chickadee", depth="truecolor", encoding="utf-8")
    app.chosen = []
    async with app.run_test(size=SIZE) as pilot:
        code = app.code
        code.paths = lambda: ("README.md",)
        await pilot.press("f4", "slash", *"READ", "enter")
        await pilot.pause()
        assert app.chosen == ["README.md"]
        await pilot.press("slash")
        assert code.finding
        await pilot.press("escape")
        assert not code.finding and app.focused is code


# -- a file read only ---------------------------------------------------------


def test_a_file_is_numbered_and_drawn_safe_or_says_why_it_has_no_text() -> None:
    read = FileText(path="a.py", size=12, text="x = 1\n\x1b[2Jy\n")
    lines = file_lines(read, COLOURS).plain.split("\n")
    assert lines == ["   1   x = 1", "   2   \\x1b[2Jy"], "an escape never reaches the terminal"
    binary = FileText(path="logo.png", size=9, text=None, withheld="not UTF-8 text")
    assert file_lines(binary, COLOURS).plain == "logo.png: not UTF-8 text"


async def test_a_file_shows_read_only_and_esc_goes_back_without_interrupting() -> None:
    app = _app(_state())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("f4")
        app.code.show_file(FileText(path="README.md", size=6, text="hello\n"))
        await pilot.pause()
        assert _title(app).startswith("CODE · README.md")
        assert app.code.takes_escape()
        assert not app.check_action("interrupt", ()), "Esc is the pane's while a file is open"
        await pilot.press("escape")
        assert app.code.opened is None and not app.code.takes_escape()
        assert app.check_action("interrupt", ()) is True, "and Zeus's again after"


def test_idle_until_a_task_or_a_file() -> None:
    frame = DiffFrame(COLOURS, UNICODE)
    assert frame.idle(TuiState())
    frame.opened = FileText(path="a", size=0, text="")
    assert not frame.idle(TuiState())


def test_a_new_run_does_not_show_an_older_run_s_diffs() -> None:
    """Task ids repeat across runs (`T1` in each): a new episode drops the diffs read."""
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        feed.apply(frame)
    task = next(iter(feed.state.worker.tasks))
    feed.state.diffs[task] = RunDiff(
        run_id=feed.state.athena.run_id or "", task_id=task, operation_id=None,
        base="b" * 40, tip=None, files=[], text="",
    )  # fmt: skip
    feed.apply(episode(10_000, run_id="2026-10-26-0000other"))
    assert feed.state.diffs == {} and feed.state.step_diffs == {}


async def test_slash_before_any_task_shows_the_search() -> None:
    app = _app(TuiState())
    async with app.run_test(size=SIZE) as pilot:
        app.code.paths = lambda: ("README.md",)
        await pilot.press("f4", "slash")
        assert app.code.finding and app.code.head.display, "never a hidden box in focus"
        await pilot.press(*"READ")
        assert app.code.found == ["README.md"]
