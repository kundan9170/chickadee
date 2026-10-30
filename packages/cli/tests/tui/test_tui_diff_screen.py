"""The diff screen: `^D` or a bare `/diff`.

What these pin: the committed diff is split by file and hunk, each line numbered from its
hunk's header and drawn safe (control characters shown, never sent); "this task" lists the
worker's task's files and says its lines are not on the wire; "all tasks" lists each
task's files by `/diff`'s own rule, with counts from its facts and an unknown total while any
is unknown, and draws the committed hunks; before `/handoff` answers it says it is reading,
and a server that does not build it is said not to serve it; `^D` waits for a run, opens the
screen in place of the panes and closes it, as `Esc` does; a bare `/diff` opens it and
`/diff <run>` stays the REPL's; `n`, `t` and `[ ]` move, `[ ]` only where hunks show, `u` does
nothing, and the bar's keys follow; the foot says committed, dropped or not yet, never
more; `/handoff` is read when the screen opens and when a commit lands while it shows, never
otherwise; and a question's modal opens over it.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from recordings import run_frames
from textual.content import Content
from tui_frames import SID, approval_asked, episode

from chickadee_cli.render.diff import NO_LINES
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.diffs import split_diff
from chickadee_cli.tui.feed import Feed, NotServed
from chickadee_cli.tui.keys import DIFF_BAR, DIFF_OFF_BAR, ScreenView, diff_bound, lit
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.modals.confirm import ConfirmModal
from chickadee_cli.tui.model import DiffScope, StaleRoute
from chickadee_cli.tui.screens.diff import NOT_READ, NOT_SERVED, FullDiff, changed_files
from chickadee_cli.tui.talk import SCREEN_COMMANDS
from chickadee_wire.models import HandoffSummary

pytestmark = pytest.mark.anyio

SIZE = (104, 33)
DIFF = (
    "diff --git a/src/client.py b/src/client.py\n"
    "index 1..2 100644\n--- a/src/client.py\n+++ b/src/client.py\n"
    "@@ -39,4 +39,4 @@ class Client:\n"
    "     def connect(self):\n"
    "-        max_retries = 3\n"
    "+        max_retries = 5\n"
    "         return self._open()\n"
    "@@ -60,2 +60,3 @@\n"
    "     x = 1\n"
    "+    y = '\x1b[2Kevil'\n"
    "     z = 3\n"
    "diff --git a/notes.md b/notes.md\nnew file mode 100644\n--- /dev/null\n+++ b/notes.md\n"
    "@@ -0,0 +1 @@\n+hello\n"
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def armed_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.0)


def _summary(diff: str = DIFF) -> HandoffSummary:
    spend = {"scope": "session", "tokens_in": 0, "tokens_out": 0, "cached_in": 0,
             "cost_usd": 0.0, "wall_ms": 0}  # fmt: skip
    return HandoffSummary.model_validate(
        {
            "session_id": SID,
            "git": None,
            "spend": spend,
            "episodes": [],
            "last_outcome": None,
            "unverified": [],
            "diff": diff,
        }
    )


def _feed(upto: int = 60) -> Feed:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    return feed


class _FakeReader:
    """A stand-in for the app's `SessionReader`: the routes asked for, in order."""

    def __init__(self) -> None:
        self.asked: list[set[str]] = []

        self.reads = 0
        """One-off reads (`ask`): the shown task's diff. Never answered here."""

    def refresh(self, routes: Iterable[str]) -> None:
        self.asked.append(set(routes))

    def ask(self, *_args: object) -> None:
        self.reads += 1

    def close(self) -> None:
        return


def _app(feed: Feed) -> ChickadeeApp:
    return ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")


def _diff(app: ChickadeeApp) -> FullDiff:
    return app.full_diff


def _text(widget: FullDiff, part: str) -> str:
    return str(widget.query_one(part).render())


def _title(title: object) -> str:
    """A border title as drawn: Textual keeps it as markup."""
    return Content.from_markup(str(title)).plain if title is not None else ""


# -- the committed diff, split ---------------------------------------------------------------


def test_the_committed_diff_is_split_by_file_and_hunk_and_numbered() -> None:
    files = split_diff(DIFF)
    assert list(files) == ["src/client.py", "notes.md"]
    client = files["src/client.py"]
    assert [h.header.split(" @@")[0] for h in client.hunks] == [
        "@@ -39,4 +39,4",
        "@@ -60,2 +60,3",
    ]
    first = client.hunks[0].lines
    assert [(ln.kind, ln.old, ln.new) for ln in first] == [
        ("context", 39, 39), ("removed", 40, None), ("added", None, 40), ("context", 41, 41),
    ]  # fmt: skip
    assert (client.added, client.removed) == (2, 1)
    assert files["notes.md"].notes == ("new file mode 100644",)


def test_a_diff_line_s_control_characters_are_shown_not_sent() -> None:
    added = split_diff(DIFF)["src/client.py"].hunks[1].lines[1]
    assert added.text == "    y = '\\x1b[2Kevil'", (
        "the escape is drawn, never sent to the terminal"
    )


# -- what each scope lists -------------------------------------------------------------------


def test_this_task_lists_the_worker_s_task_and_all_tasks_each_task_s_files() -> None:
    state = _feed().state
    task = changed_files(state, DiffScope.TASK)
    assert [(f.task_id, f.path) for f in task] == [("T2", "src/config.py")]
    every = changed_files(state, DiffScope.ALL)
    assert [(f.task_id, f.path, f.committed) for f in every] == [
        ("T1", "src/client.py", True), ("T1", "tests/test_client.py", True),
        ("T2", "src/config.py", False),
    ]  # fmt: skip
    assert all(f.added is not None for f in every), "counts from each task's facts"


async def test_a_count_no_facts_measured_leaves_the_total_unknown() -> None:
    app = _app(_feed(12))  # T1's steps, before its round reports
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        diff = _diff(app)
        assert changed_files(app.state, DiffScope.TASK)[0].added is None
        title = _title(diff.changed.border_title)
        assert "+" not in title, "no total while a count is unknown"


# -- opening and closing ---------------------------------------------------------------------


async def test_ctrl_d_waits_for_a_run() -> None:
    app = _app(Feed(SID))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        assert app.view is ScreenView.BASE and not _diff(app).display


async def test_ctrl_d_opens_it_in_place_of_the_panes_and_esc_and_ctrl_d_close_it() -> None:
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        assert app.view is ScreenView.DIFF and _diff(app).display
        assert not app.base.query_one("#panes").display
        assert app.focused is _diff(app)
        await pilot.press("escape")
        assert app.view is ScreenView.BASE and app.base.query_one("#panes").display
        assert app.focused is app.zeus.composer
        await pilot.press("ctrl+d", "ctrl+d")
        assert app.view is ScreenView.BASE, "^D again goes back"


def test_a_bare_slash_diff_opens_it_and_one_naming_a_run_stays_the_repl_s() -> None:
    opened: list[str] = []

    class Screen:
        def open_diff(self) -> None:
            opened.append("diff")

    SCREEN_COMMANDS["/diff"](Screen())  # type: ignore[arg-type]
    assert opened == ["diff"]
    assert "/diff 2026-10-10-0123abcd" not in SCREEN_COMMANDS, "the REPL's /diff [run]"


async def test_a_question_s_modal_opens_over_it() -> None:
    feed = _feed()
    app = _app(feed)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        app.take(approval_asked("shell-1", 900))
        await pilot.pause()
        assert isinstance(app.screen, ConfirmModal)


# -- what it draws ---------------------------------------------------------------------------


async def test_this_task_says_its_lines_are_not_on_the_wire() -> None:
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        assert NO_LINES in _text(_diff(app), "#body")
        assert "src/config.py · T2" in _title(_diff(app).hunks.border_title)


async def test_its_keys_in_the_bar_are_buttons_too() -> None:
    """A diff screen key's chip, clicked, does what the key does, though the diff screen
    binds it and not the app (`statusbar._click`, `action_on_diff`)."""
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        bar = app.base.query_one("StatusBar")
        drawn = str(bar.render())
        await pilot.click("StatusBar", offset=(len(drawn.split(" t ")[0]) + 2, 0))
        await pilot.pause()
        assert _diff(app).scope is DiffScope.ALL, "t, clicked"
        drawn = str(bar.render())
        await pilot.click("StatusBar", offset=(len(drawn.split(" Esc ")[0]) + 2, 0))
        await pilot.pause()
        assert app.view is ScreenView.BASE, "Esc, clicked"


async def test_all_tasks_reads_then_draws_the_committed_hunks_or_says_none_are_served() -> None:
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d", "t")
        diff = _diff(app)
        assert diff.scope is DiffScope.ALL
        assert NOT_READ in _text(diff, "#body"), "said while /handoff has not answered"
        app.take_snapshot(_summary())
        await pilot.pause()
        body = _text(diff, "#body")
        assert (
            "  40 -         max_retries = 3" in body
            and "  40 +         max_retries = 5" in body
        )
        assert "committed, from base" in _title(diff.hunks.border_title)
        app.take_snapshot(NotServed(StaleRoute.HANDOFF.value))
        app.state.handoff = None
        diff.show(app.state)
        assert NOT_SERVED in _text(diff, "#body")


async def test_n_t_and_the_hunk_keys_move_and_u_does_nothing() -> None:
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        app.take_snapshot(_summary())
        await pilot.press("ctrl+d", "t")
        diff = _diff(app)
        assert diff.chosen == 0 and diff.hunks_shown
        await pilot.press("right_square_bracket")
        assert diff.hunk == 1
        await pilot.press("right_square_bracket")
        assert diff.hunk == 0, "wraps"
        await pilot.press("left_square_bracket")
        assert diff.hunk == 1
        await pilot.press("n")
        assert diff.chosen == 1 and diff.hunk == 0
        assert not diff.hunks_shown, "tests/test_client.py has no committed hunk in this diff"
        assert "not committed" in _text(diff, "#body")
        await pilot.press("u")
        assert app.view is ScreenView.DIFF and diff.chosen == 1, "undo hunk is not built (W14)"
        await pilot.press("t")
        assert diff.scope is DiffScope.TASK and diff.chosen == 0


async def test_the_foot_says_committed_dropped_or_not_yet_never_more() -> None:
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        diff = _diff(app)
        assert "T2 was dropped: nothing of it is committed." in _text(diff, "#foot")
        await pilot.press("t")
        assert "T1 is committed." in _text(diff, "#foot")
    running = _app(_feed(19))
    async with running.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d")
        foot = _text(_diff(running), "#foot")
        assert "This is the live worktree. Nothing is committed for T1 yet." in foot


# -- its keys --------------------------------------------------------------------------------


def test_its_bar_binds_what_is_built_and_draws_the_rest_dimmed() -> None:
    state = _feed().state
    bindings = {b.key for b in FullDiff.BINDINGS if not isinstance(b, tuple)}
    assert bindings == {k.key for k in diff_bound()}
    assert {"n", "t", "escape", "left_square_bracket", "right_square_bracket"} <= bindings
    keys = {k.key: k for g in DIFF_BAR for k in g} | {k.key: k for k in DIFF_OFF_BAR}
    assert not lit(keys["u"], state, view=ScreenView.DIFF), "W14"
    assert not lit(keys["left_square_bracket"], state, view=ScreenView.DIFF, hunks=False)
    assert lit(keys["left_square_bracket"], state, view=ScreenView.DIFF, hunks=True)
    assert not lit(keys["n"], state), "a diff key does nothing on the panes"


async def test_the_panes_keys_do_nothing_on_it() -> None:
    app = _app(_feed())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("ctrl+d", "f2", "x")
        assert app.view is ScreenView.DIFF and app.focused is _diff(app)
        assert app.zeus.composer.text == "", "a letter here is not Zeus's"


# -- reading /handoff --------------------------------------------------------------------------


def _commit_seq() -> int:
    """The seq of the first record whose fold makes `/handoff` stale: T1's commit."""
    feed = _feed(30)
    for frame in run_frames():
        if frame.seq > 30 and StaleRoute.HANDOFF in feed.apply(frame).refresh:
            return frame.seq
    raise AssertionError("the recorded run commits a task")


async def test_handoff_is_read_when_it_opens_and_on_a_commit_only_while_it_shows() -> None:
    commit = _commit_seq()
    app = _app(_feed(30))
    reader = _FakeReader()
    async with app.run_test(size=SIZE) as pilot:
        app.live = reader  # type: ignore[assignment]
        for frame in run_frames():
            if 30 < frame.seq <= commit:
                app.take(frame)
        await pilot.pause()
        assert not any(StaleRoute.HANDOFF in r for r in reader.asked), "closed: never read"
        await pilot.press("ctrl+d")
        assert {StaleRoute.HANDOFF} in reader.asked, "read as it opens"
        reader.asked.clear()
        for frame in run_frames():
            if frame.seq > commit:
                app.take(frame)
        await pilot.pause()
        assert any(StaleRoute.HANDOFF in r for r in reader.asked), "the run's end, while open"
