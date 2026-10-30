"""A command's confirmation as a modal, and
the keys under a modal (the dimmed-keys rule).

What these pin: each row is a field of the question, and one the record cannot fill is left
out; the risk says "above T0" only above it; the command is shown whole, never cut; the
answers are exactly the wire's (`approval_choices`); Deny is highlighted when it opens, a
letter or an arrow moves the choice, never to an answer not offered or not built, and
`Enter` submits it, so `Enter` alone and Esc deny; the mockup's `e` dimmed and doing nothing
; *always* says what it covers
and how far; under a modal the screen's keys are dimmed and do nothing, and light again when
it closes; every modal key the table builds is bound and no other; and, end to end, a hosted
engine's real T2 command opens the modal and its `y` lets the command run.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import hosting
import pytest
from engines import Step, answer, call, calls, raw_config
from hosting import SID as HOSTED_SID
from hosting import hosted
from textual.widgets import Static
from tui_frames import SID, answered, approval_asked, stop_asked, until

from chickadee_cli.render.question import grant_breadth
from chickadee_cli.tui.app import ChickadeeApp, connector
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.keys import BAR, OFF_BAR, lit
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.modals.confirm import (
    CONFIRM_ANSWERS,
    SANDBOX_OUTSIDE,
    ConfirmModal,
    answer_labels,
    confirm_rows,
    risk,
)
from chickadee_cli.tui.modals.stop import STOP_ANSWERS, StopModal
from chickadee_serve.frames import JsonValue, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.models import ActionApproval, QuestionAskedRecord
from chickadee_wire.questions import OpenQuestion

pytestmark = pytest.mark.anyio

SIZE = (104, 33)
GRANT: dict[str, JsonValue] = {
    "cls": "package install", "prefix": ["uv", "add"], "destination": "pypi.org",
    "effects": ["pyproject.toml", "uv.lock"],
}  # fmt: skip


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def armed_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modals armed as they open (the arming delay is `test_tui_stop.py`'s)."""
    monkeypatch.setattr(QuestionModal, "ARM_S", 0.0)


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    choices: list[str] = []

    def answer(
        _self: ChickadeeApp,
        _asked: OpenQuestion,
        choice: str,
        _note: str | None,
        _text: str | None,
        _failed: Callable[[], None],
    ) -> None:
        choices.append(choice)

    monkeypatch.setattr(ChickadeeApp, "answer", answer)
    return choices


def _approval(**fields: JsonValue) -> ActionApproval:
    asked = QuestionAskedRecord.model_validate(approval_asked("shell-1", 1, **fields).payload)
    assert asked.approval is not None
    return asked.approval


def _app(*frames: RecordFrame) -> ChickadeeApp:
    feed = Feed(SID)
    for frame in frames:
        feed.apply(frame)
    return ChickadeeApp(feed=feed, depth="truecolor", encoding="utf-8")


def _modal(app: ChickadeeApp) -> ConfirmModal:
    assert isinstance(app.screen, ConfirmModal), type(app.screen).__name__
    return app.screen


def _text(modal: ConfirmModal) -> str:
    return "\n".join(str(s.render()) for s in modal.query(Static))


# -- the rows --------------------------------------------------------------------------------


def test_each_row_is_a_field_and_one_the_record_cannot_fill_is_left_out() -> None:
    bare = _approval()
    assert confirm_rows(bare) == [("network", "no"), ("risk", "T2 · above T0, so you decide")]
    granted = _approval(network=True, grant=GRANT, credentials=["PYPI_TOKEN"])
    assert confirm_rows(granted) == [
        ("network", "yes · pypi.org"),
        ("writes", "pyproject.toml, uv.lock"),
        ("credentials", "PYPI_TOKEN"),
        ("risk", "T2 · above T0, so you decide"),
    ]
    assert confirm_rows(_approval(network=True))[0] == ("network", "yes"), "no host named"


def test_a_shells_question_says_where_it_runs_and_that_it_leaves_the_sandbox() -> None:
    """The command and its directory as fields; `sandboxed` false, since an
    approved command runs outside the sandbox."""
    asked = _approval(command="uv add numpy", cwd="src", sandboxed=False)
    rows = confirm_rows(asked)
    assert rows[0] == ("in", "src")
    assert ("sandbox", SANDBOX_OUTSIDE) in rows
    assert [n for n, _ in rows] == ["in", "network", "sandbox", "risk"]


def test_the_risk_says_above_t0_only_above_it() -> None:
    assert risk(_approval(tier="t1")) == "T1 · above T0, so you decide"
    assert risk(_approval(tier="t0")) == "T0", "with no sandbox a T0 command asks too"


def test_this_session_names_an_exact_command_or_the_grant_s_prefix() -> None:
    assert answer_labels(_approval(grant=GRANT))["s"] == "Always allow 'uv add' this session"
    exact: dict[str, JsonValue] = {"cls": "package install", "prefix": ["uv", "add", "numpy"]}
    assert answer_labels(_approval(exact=exact))["s"] == "Allow this exact command this session"


# -- the modal -------------------------------------------------------------------------------


async def test_the_command_is_shown_whole_and_who_asks_is_named() -> None:
    display = "uv add numpy " + "--extra-index-url https://example.org/simple " * 4
    app = _app(approval_asked("shell-1", 1, display=display))
    async with app.run_test(size=SIZE):
        modal = _modal(app)
        band = str(modal.query_one("#band").render())
        assert band == f"$ {display}", "whole: wrapped, never cut"
        assert "The worker wants to run" in _text(modal)
        title = str(modal.query_one("#box").border_title)
        assert "CONFIRM A COMMAND" in title and "worker · T2" in title


async def test_a_once_only_question_offers_once_deny_and_abort(sent: list[str]) -> None:
    app = _app(approval_asked("shell-1", 1))
    async with app.run_test(size=SIZE) as pilot:
        modal = _modal(app)
        drawn = str(modal.answers.render())
        assert "Run once" in drawn and "Deny" in drawn and "Stop the turn" in drawn
        assert "Always" not in drawn, "no grant to give: no always drawn"
        assert "Edit command" in drawn and not modal.lit("e"), "W12: drawn dimmed"
        assert "Always-allow" not in _text(modal)
        assert modal.selected == "n", "Deny, the default, highlighted when it opens"
        await pilot.press("a", "s", "e")
        assert sent == [] and modal.selected == "n", "not offered, or not built: no choice"
        await pilot.press("y")
        assert sent == [], "a letter only chooses"
        await pilot.press("enter")
        assert sent == ["approve"]


@pytest.mark.parametrize(
    ("keys", "choice"),
    [(("y", "enter"), "approve"), (("a", "enter"), "repo"), (("s", "enter"), "session"),
     (("n", "enter"), "deny"), (("x", "enter"), "abort"), (("enter",), "deny"),
     (("escape",), "deny")],
)  # fmt: skip
async def test_each_answer_chosen_and_confirmed_sends_its_choice(
    sent: list[str], keys: tuple[str, ...], choice: str
) -> None:
    app = _app(approval_asked("shell-1", 1, network=True, grant=GRANT))
    async with app.run_test(size=SIZE) as pilot:
        await pilot.press(*keys)
        assert sent == [choice]


async def test_the_arrows_step_through_what_is_offered_and_built(sent: list[str]) -> None:
    app = _app(approval_asked("shell-1", 1, network=True, grant=GRANT))
    async with app.run_test(size=SIZE) as pilot:
        modal = _modal(app)
        steps: list[str | None] = []
        for key in ("right", "right", "right", "right", "right", "down", "up", "left"):
            await pilot.press(key)
            steps.append(modal.selected)
        # Drawn order y a s e n x; `e` is not built and is stepped over.
        assert steps == ["x", "y", "a", "s", "n", "x", "n", "s"]
        assert sent == [], "moving submits nothing"


async def test_always_says_what_it_covers_and_how_far() -> None:
    app = _app(approval_asked("shell-1", 1, grant=GRANT, repo_any_arguments=True))
    async with app.run_test(size=SIZE):
        said = _text(_modal(app))
        assert "package install `uv add`" in said
        assert grant_breadth(_approval(repo_any_arguments=True)) in said
        assert "Always-allow saves to this repo." in said


# -- the keys under a modal (the dimmed-keys rule) -------------------------------------------


async def test_under_a_modal_the_screen_s_keys_are_dimmed_and_do_nothing() -> None:
    app = _app(approval_asked("shell-1", 1))
    async with app.run_test(size=SIZE) as pilot:
        # ^D also waits for a run, F5 for background work, and ^S for a stop
        # waiting as a banner, which this session has none of: not a modal's doing.
        built = [
            k for k in [*(k for g in BAR for k in g), *OFF_BAR]
            if k.action is not None and k.key not in ("ctrl+d", "f5", "ctrl+s")
        ]  # fmt: skip
        assert all(not lit(k, app.state) for k in built), "dimmed under the modal"
        await pilot.press("f2", "ctrl+o")
        assert isinstance(app.screen, ConfirmModal) and app.zeus.mode.name == "COLLAPSED"
        app.take(answered("shell-1", 2, choice="deny"))
        await pilot.pause()
        assert all(lit(k, app.state) for k in built), "lit again once it closes"
        await pilot.press("f2")
        assert app.focused is not None and app.focused.id == "athena"


def test_every_modal_key_the_table_builds_is_bound_and_no_other() -> None:
    for modal, table in ((StopModal, STOP_ANSWERS), (ConfirmModal, CONFIRM_ANSWERS)):
        bound = {
            b.key
            for b in modal.BINDINGS
            if not isinstance(b, tuple) and b.action == f"answer('{b.key}')"
        }
        assert bound == {a.key for a in table if a.choice is not None}, modal.__name__
        for a in table:
            assert a.choice is not None or a.wire_request, f"{a.key} names its wire request"


def test_a_stop_dims_the_screen_s_keys_while_open_and_not_once_put_back() -> None:
    """A stop opens as it is asked and has the screen as a modal does; put back as a
    banner, the screen is the person's again."""
    feed = Feed(SID)
    feed.apply(stop_asked("stop-1", "design", 1))
    assert not lit(BAR[0][0], feed.state), "open: the modal has the screen"
    feed.state.stops_put_back.add("stop-1")
    assert lit(BAR[0][0], feed.state), "put back: waiting as a banner"


# -- end to end ------------------------------------------------------------------------------


@pytest.fixture
def unsandboxed(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Client, Path]]:
    """A hosted engine with no sandbox and the floor on: every shell command Zeus runs
    is T2 and confirmed, as `serve/tests/test_tier_floor_unchanged.py` runs it."""
    import chickadee.engine as engine_module
    from chickadee.isolation.none import Passthrough

    def config() -> dict[str, object]:
        raw = raw_config()
        raw["sandbox"] = {"require_sandbox": False}
        return raw

    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    monkeypatch.setattr(hosting, "raw_config", config)
    script: list[Step] = [
        calls(call("shell", "s1", command="touch ran")),
        answer("marked"),
    ]
    with hosted(monkeypatch, script) as host, host.client() as client:
        yield client, host.repo


async def test_a_real_command_opens_the_modal_and_y_lets_it_run(
    unsandboxed: tuple[Client, Path],
) -> None:
    client, repo = unsandboxed
    app = ChickadeeApp(
        feed=Feed(HOSTED_SID),
        connect=connector(client, HOSTED_SID),
        depth="truecolor",
        encoding="utf-8",
    )
    async with app.run_test(size=SIZE) as pilot:
        app.zeus.composer.load_text("mark it")
        await pilot.press("enter")
        await until(pilot, lambda: isinstance(app.screen, ConfirmModal), "the modal")
        modal = _modal(app)
        assert modal.approval.tier == "t2" and "touch ran" in modal.approval.display
        assert not any("asks to run" in (e.text or "") for e in app.state.zeus.entries), (
            "drawn by the modal, not printed in Zeus's pane"
        )
        await pilot.press("y", "enter")
        await until(pilot, lambda: app.modal is None, "the answer's record closing it")
        deadline = time.monotonic() + 20
        while not (repo / "ran").exists() and time.monotonic() < deadline:
            await pilot.pause(0.05)
    assert (repo / "ran").exists(), "approved by the person's y, it ran"
