"""A stop as a modal over the screen, driven by the pilot over
records built by hand; what it sends is recorded at the app's `answer`, the one way a modal
sends (`Talk.answer` takes it to the REPL's `Interaction.send`; `test_tui_confirm.py` runs that
path end to end).

What these pin: a stop opens over the screen with what the REPL shows of it (who asks, where
the work runs, why the last answer was not used, the markdown) and its title names the stop,
how often it was shown and the text's sha; an answer is chosen (its letter, `←`/`→`, `Tab`)
and only `Enter` submits it, nothing being highlighted when it opens; its keys do nothing
until it is armed; `a` then `Enter` approves and the record, not the send, closes it; a note
needs words, a cancel's reason is optional, an unchanged edit sends nothing, each said in the
modal; Enter and Esc alone answer nothing; a refused send lets the keys work again; how many
times a stop of a kind was shown counts across notes and edits and restarts once one is
settled;
stop 2 is a note to Athena and counts no tasks; the panes keep updating beneath; the next
question opens when one closes; a question Zeus asks stays in his pane, and a modal waits
behind it; the REPL's renderer does not print what the screen draws.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import pytest
from textual.pilot import Pilot
from textual.widgets import Markdown, Static
from tui_frames import SID, answered, approval_asked, session, stop_asked, until, user

from chickadee_cli.interaction import Interaction
from chickadee_cli.render.style import Style
from chickadee_cli.repl import handle_line
from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.modals.confirm import ConfirmModal
from chickadee_cli.tui.modals.stop import StopModal, stop_title
from chickadee_cli.tui.model import ZeusKind, drawn_as_modal
from chickadee_serve.frames import JsonValue, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.models import ApprovalStop
from chickadee_wire.questions import OpenQuestion

pytestmark = pytest.mark.anyio

SIZE = (104, 33)
DESIGN = "# Design: range sums\n\nIterative, over `[lo, hi)`.\n"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def armed_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modals armed as they open, so each test presses keys at once; the arming delay is
    `test_its_keys_do_nothing_until_it_is_armed`'s."""
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.0)


@dataclass
class StopSends:
    """What the app was asked to send, and whether the next send is refused."""

    answers: list[tuple[str, str | None, str | None]] = field(
        default_factory=list[tuple[str, str | None, str | None]]
    )
    refuse: bool = False


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> StopSends:
    record = StopSends()

    def answer(
        _self: ChickadeeApp,
        _asked: OpenQuestion,
        choice: str,
        note: str | None,
        text: str | None,
        failed: Callable[[], None],
    ) -> None:
        record.answers.append((choice, note, text))
        if record.refuse:
            failed()

    monkeypatch.setattr(ChickadeeApp, "answer", answer)
    return record


def _stop(kind: str = "design", seq: int = 1, **fields: JsonValue) -> RecordFrame:
    frame = stop_asked(f"stop-{kind}", kind, seq)
    stop = frame.payload["stop"]
    assert isinstance(stop, dict)
    stop.update({"text": DESIGN, "sha": "7c1e" + "0" * 60, **fields})
    return frame


def _app(*frames: RecordFrame, put_back: bool = False) -> ChickadeeApp:
    """The screen over `frames`; `put_back`, the person has put each stop back as its banner
    with `Esc` (a stop opens on the screen as it is asked)."""
    feed = Feed(SID)
    for frame in frames:
        feed.apply(frame)
    if put_back:
        feed.state.stops_put_back.update(q.id for q in feed.state.stops.open if q.stop)
    return ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")


def _modal(app: ChickadeeApp) -> StopModal:
    assert isinstance(app.screen, StopModal), type(app.screen).__name__
    return app.screen


def _why(app: ChickadeeApp) -> str:
    return str(_modal(app).why.render())


async def _type(pilot: Pilot[None], text: str) -> None:
    await pilot.press(*("space" if c == " " else c for c in text))


# -- what it shows ---------------------------------------------------------------------------


def test_the_title_names_the_stop_the_times_shown_and_the_sha() -> None:
    stop = ApprovalStop.model_validate(_stop().payload["stop"])
    assert stop_title(stop, 2) == ("STOP 1 · DESIGN · shown 2 times", "zeus · 7c1e…")
    once = ApprovalStop.model_validate(_stop("plan").payload["stop"])
    assert stop_title(once, 1) == ("STOP 2 · PLAN · shown 1 time", "athena · 7c1e…")


async def test_a_stop_opens_over_the_screen_with_what_the_repl_shows() -> None:
    app = _app(_stop(round=2, refused="the last edit changed nothing", source_ref="main"))
    async with app.run_test(size=SIZE):
        modal = _modal(app)
        said = [str(s.render()) for s in modal.query(Static)]
        assert "zeus asks you to approve the design (round 2)" in said
        assert "the work runs in a new worktree from main" in said
        assert "the last edit changed nothing" in said
        assert modal.query_one(Markdown).source == DESIGN, "the markdown, whole"
        title = modal.query_one("#box").border_title
        assert title is not None and "STOP 1 · DESIGN · shown 2 times" in str(title)
        assert "zeus · 7c1e…" in str(title)


# -- answering -------------------------------------------------------------------------------


async def test_a_approves_and_the_record_not_the_send_closes_it(sent: StopSends) -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a", "enter")
        assert sent.answers == [("approve", None, None)]
        await pilot.press("a", "enter")
        assert len(sent.answers) == 1, "sent: the keys wait for the record"
        assert isinstance(app.screen, StopModal), "still open until the record lands"
        app.take(answered("stop-design", 2))
        await pilot.pause()
        assert app.modal is None and not isinstance(app.screen, StopModal)


async def test_a_note_needs_words_and_is_sent_with_them(sent: StopSends) -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("n", "enter", "enter")  # chosen, confirmed, then no words
        assert sent.answers == []
        assert "a note needs words" in _why(app)
        assert _modal(app).words.has_class("-open"), "still open, for the words"
        await _type(pilot, "keep the api")
        await pilot.press("enter")
        assert sent.answers == [("note", "keep the api", None)]


async def test_letters_typed_in_a_note_are_the_note_not_answers(sent: StopSends) -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("n", "enter", "a", "c")
        assert sent.answers == [], "a and c are letters of the note here"
        assert _modal(app).words.value == "ac"
        await pilot.press("escape")
        assert not _modal(app).words.has_class("-open"), "Esc backs out of the note"
        assert sent.answers == []


async def test_a_cancel_s_reason_is_optional(sent: StopSends) -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("c", "enter", "enter")  # chosen, confirmed, no reason given
        assert sent.answers == [("cancel", None, None)]


@pytest.mark.parametrize(("saved", "sends"), [(DESIGN, False), (None, False), ("# v2\n", True)])
async def test_an_edit_sends_the_saved_text_and_an_unchanged_one_nothing(
    sent: StopSends, monkeypatch: pytest.MonkeyPatch, saved: str | None, sends: bool
) -> None:
    monkeypatch.setattr(ChickadeeApp, "edit", lambda _self, _text: saved)
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("e", "enter")
        if sends:
            assert sent.answers == [("edit", None, "# v2\n")]
        else:
            assert sent.answers == []
            assert "the text is unchanged; nothing was sent" in _why(app)


async def test_enter_and_esc_answer_nothing(sent: StopSends) -> None:
    """Esc puts an open stop back to its banner, answering nothing: with the person here a stop
    has no timeout, so hiding it approves nothing, and `^S` opens it again."""
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        assert _modal(app).selected is None, "nothing is highlighted when it opens"
        await pilot.press("enter", "y", "enter")
        assert isinstance(app.screen, StopModal), "Enter and a stray letter hide nothing"
        await pilot.press("escape")
        await pilot.pause()
        assert sent.answers == []
        assert app.modal is None and not isinstance(app.screen, StopModal)
        assert app.state.waiting_stop is not None, "it waits as a banner again"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, StopModal)


async def test_a_letter_only_chooses_an_answer_and_enter_submits_it(sent: StopSends) -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a")
        assert sent.answers == [], "a letter alone answers nothing"
        assert _modal(app).selected == "a"
        assert "▸a" in str(_modal(app).answers.render()), "the chosen answer is marked"
        await pilot.press("c")
        assert _modal(app).selected == "c", "another letter moves the choice"
        await pilot.press("a", "enter")
        assert sent.answers == [("approve", None, None)]


async def test_left_right_and_tab_move_the_choice_and_wrap(sent: StopSends) -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        modal = _modal(app)
        await pilot.press("right")
        assert modal.selected == "a", "from nothing, the first"
        await pilot.press("right", "tab")
        assert modal.selected == "n"
        await pilot.press("left", "left", "left")
        assert modal.selected == "c", "past the first, the last"
        await pilot.press("shift+tab")
        assert modal.selected == "n"
        assert sent.answers == [], "moving submits nothing"


async def test_its_keys_do_nothing_until_it_is_armed(
    sent: StopSends, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A modal opens over keys already on their way (a confirmation opens on its own while the
    person types to Zeus): until it is armed they reach nothing, and its answers are drawn
    dimmed. (A delay
    far longer than the test, then armed by hand: the test does not race the clock.)"""
    monkeypatch.setattr(QuestionModal, "ARM_S", 600.0)
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        modal = _modal(app)
        await pilot.press("a", "enter", "right", "enter")
        assert sent.answers == [] and modal.selected is None
        assert not modal.lit("a"), "drawn dimmed while it arms"
        modal.arm()
        assert modal.lit("a")
        await pilot.press("a", "enter")
        assert sent.answers == [("approve", None, None)]


async def test_it_arms_itself_after_its_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.2)
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await until(pilot, lambda: _modal(app).armed, "armed by its timer")


async def test_a_refused_send_lets_the_keys_work_again(sent: StopSends) -> None:
    sent.refuse = True
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press("a", "enter")
        assert not _modal(app).sent
        await pilot.press("enter")
        assert len(sent.answers) == 2, "the choice stays; Enter sends it again"


async def test_stop_2_is_a_note_to_athena_and_counts_no_tasks() -> None:
    app = _app(_stop("plan"))
    async with app.run_test(size=SIZE):
        modal = _modal(app)
        assert "note to Athena" in str(modal.answers.render())
        title = str(modal.query_one("#box").border_title)
        # W1: the stop carries the plan's text only; its tasks are in it, not counted.
        assert "task" not in title


# -- around it -------------------------------------------------------------------------------


async def test_the_panes_keep_updating_beneath_it() -> None:
    app = _app(_stop())
    async with app.run_test(size=SIZE) as pilot:
        before = len(app.zeus.transcript.children)
        app.take(user("also the changelog", 2))
        await pilot.pause()
        assert isinstance(app.screen, StopModal)
        assert app.state.zeus.entries[-1].kind is ZeusKind.YOU
        assert len(app.zeus.transcript.children) > before, "drawn beneath the modal"


async def test_the_next_question_opens_when_one_closes() -> None:
    app = _app(_stop(), approval_asked("shell-1", 2))
    async with app.run_test(size=SIZE) as pilot:
        assert isinstance(app.screen, StopModal), "oldest first"
        app.take(answered("stop-design", 3))
        await pilot.pause()
        assert isinstance(app.screen, ConfirmModal)
        assert app.screen.asked.id == "shell-1"


async def test_a_question_zeus_asks_stays_in_his_pane_and_a_modal_waits_behind_it() -> None:
    question: dict[str, JsonValue] = {
        "id": "q1", "kind": "clarify_goal", "text": "which branch?", "default": "main",
    }  # fmt: skip
    asked = session("question_asked", {"question": question}, 1)
    app = _app(asked, _stop(seq=2))
    async with app.run_test(size=SIZE) as pilot:
        assert app.modal is None, "his question is answered in his pane"
        assert app.zeus.composer.placeholder == "Answer Chickadee…"
        app.take(answered("q1", 3, choice="main"))
        await pilot.pause()
        assert isinstance(app.screen, StopModal)


async def test_a_stop_opens_on_the_screen_as_it_is_asked() -> None:
    """A design or a plan proposed is on the screen at once, the stop's modal, with no
    banner and no key to press first; a stop asked while the screen is up opens too."""
    app = _app()
    async with app.run_test(size=SIZE) as pilot:
        assert app.modal is None
        app.take(_stop("plan"))
        await pilot.pause()
        assert isinstance(app.screen, StopModal)
        assert not app.base.query_one("#banner", Static).display


async def test_a_stop_put_back_waits_as_a_banner_until_ctrl_s_opens_it() -> None:
    """D247, keeping D240 R-8's purpose: put back with `Esc`, a stop leaves the screen to the
    person talking to Zeus; the banner says which stop waits and how to open it, and the run
    waits on it."""
    app = _app(_stop("plan"), put_back=True)
    async with app.run_test(size=SIZE) as pilot:
        assert app.modal is None, "the screen stays the person's"
        banner = str(app.base.query_one("#banner", Static).render())
        assert "STOP 2 · PLAN" in banner and "^S opens it" in banner
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert isinstance(app.screen, StopModal)


async def test_a_confirmation_is_not_held_behind_a_waiting_stop() -> None:
    """A command's confirmation still opens over the screen at once: a stop waiting as a banner
    holds nothing behind it."""
    app = _app(_stop(), approval_asked("shell-1", 2), put_back=True)
    async with app.run_test(size=SIZE):
        assert isinstance(app.screen, ConfirmModal)
        assert app.screen.asked.id == "shell-1"


def test_a_line_typed_while_a_stop_waits_is_words_never_its_answer() -> None:
    """D240 R-8, D247: with the stop put back as a banner the composer is the person's, so an
    `a` meant for Zeus must not approve the plan; the REPL, which draws nothing itself, still
    takes the line as the stop's answer."""
    told: list[str] = []

    class Words:
        def tell(self, text: str) -> None:
            told.append(text)

    screen = Interaction(
        cast(Client, object()),
        SID,
        Style(tty=False),
        lambda _line: None,
        draws=drawn_as_modal,
        answers=drawn_as_modal,
    )
    screen.feed(_stop())
    assert screen.current is not None and screen.typed is None
    handle_line(Words(), screen, "a")  # type: ignore[arg-type]
    assert told == ["a"], "told to Zeus, and nothing answered"
    repl = Interaction(cast(Client, object()), SID, Style(tty=False), lambda _line: None)
    repl.feed(_stop())
    assert repl.typed is not None and repl.typed.id == "stop-design"


def test_the_repl_s_renderer_prints_only_what_the_screen_does_not_draw() -> None:
    printed: list[str] = []
    screen = Interaction(object(), SID, Style(tty=False), printed.append, draws=drawn_as_modal)  # type: ignore[arg-type]
    screen.feed(_stop())
    assert printed == [], "the modal draws the stop"
    assert screen.current is not None and screen.current.id == "stop-design"
    repl: list[str] = []
    Interaction(object(), SID, Style(tty=False), repl.append).feed(_stop())  # type: ignore[arg-type]
    assert any("zeus asks you to approve the design" in line for line in repl), "unchanged"


# -- how many times it was shown -------------------------------------------------------------


def _answer(qid: str, seq: int, choice: str) -> RecordFrame:
    return answered(qid, seq, choice=choice)


def test_a_stop_sent_back_counts_up_and_a_settled_one_starts_again() -> None:
    """A note or an edit sends the text back, and its author answers with a new stop (the
    engine's `round` restarts at 1): the same work shown again. Approval or a cancel settles
    it; the next stop of that kind is shown for the first time. Each kind counts alone."""
    feed = Feed(SID)
    steps = [
        stop_asked("plan-1", "plan", 1), _answer("plan-1", 2, "note"),
        stop_asked("plan-2", "plan", 3), _answer("plan-2", 4, "edit"),
        stop_asked("plan-3", "plan", 5), _answer("plan-3", 6, "approve"),
        stop_asked("plan-4", "plan", 7), _answer("plan-4", 8, "cancel"),
        stop_asked("plan-5", "plan", 9),
        stop_asked("design-1", "design", 10), _answer("design-1", 11, "note"),
        stop_asked("design-2", "design", 12),
    ]  # fmt: skip
    for frame in steps:
        feed.apply(frame)
    shown = feed.state.stop_shown
    assert [shown[f"plan-{n}"] for n in range(1, 6)] == [1, 2, 3, 1, 1]
    assert (shown["design-1"], shown["design-2"]) == (1, 2), "the design counts by itself"


async def test_a_plan_shown_again_after_a_note_is_titled_shown_2_times() -> None:
    """Found in the live demo: Athena's revision is a new stop with `round: 1`, and the title
    read "shown 1 time" for a plan shown the second time."""
    app = _app(_stop("plan"), _answer("stop-plan", 2, "note"), stop_asked("plan-b", "plan", 3))
    async with app.run_test(size=SIZE):
        title = str(_modal(app).query_one("#box").border_title)
        assert "STOP 2 · PLAN · shown 2 times" in title
