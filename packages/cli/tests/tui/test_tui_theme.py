"""The TUI's look: the pictures' colours with a 16-colour fallback, a
glyph and an ASCII fallback for each mark, and a word for every state, so colour is never the
only signal; and the header's and status bar's lines, built from the state alone.
"""

from __future__ import annotations

from dataclasses import fields

from tui_frames import SID, episode, run_frames, run_record, workspace

from chickadee_cli.render.board import STATES
from chickadee_cli.tui.feed import SETTLED, Feed
from chickadee_cli.tui.header import BAR_CELLS, spend_lines, stage_strip, where_line
from chickadee_cli.tui.keys import BAR, bound, lit
from chickadee_cli.tui.model import HeaderSpend, SpendLevel, Stage, TuiState
from chickadee_cli.tui.statusbar import bar_line
from chickadee_cli.tui.theme import (
    ASCII,
    PALETTE,
    PALETTE_16,
    STAGE_WORDS,
    TASK_GLYPH,
    UNICODE,
    Glyphs,
    glyphs_for,
    palette,
)
from chickadee_serve.frames import JsonValue
from chickadee_wire.models import SessionBudget

COLOURS = palette("truecolor")


# -- the theme -------------------------------------------------------------------------------


def test_the_role_colours_are_the_pictures_and_fall_back_to_16() -> None:
    assert (PALETTE["zeus"], PALETTE["athena"], PALETTE["worker"]) == (
        "#ffb000",
        "#3fd0c0",
        "#6be36b",
    )
    assert (PALETTE_16["zeus"], PALETTE_16["athena"], PALETTE_16["worker"]) == (
        "ansi_yellow",
        "ansi_cyan",
        "ansi_green",
    )
    assert set(PALETTE) == set(PALETTE_16), "every colour has its fallback"


def test_every_glyph_has_an_ascii_fallback_chosen_by_the_encoding() -> None:
    for f in fields(Glyphs):
        unicode, ascii_ = getattr(UNICODE, f.name), getattr(ASCII, f.name)
        assert unicode and ascii_.isascii(), f.name
    assert glyphs_for("utf-8") is UNICODE
    assert glyphs_for("ascii") is ASCII
    assert glyphs_for("latin-1") is ASCII, "it cannot print a box line"
    assert glyphs_for(None) is ASCII


def test_every_state_has_a_glyph_and_a_word() -> None:
    assert set(TASK_GLYPH) == STATES, "each board state is drawn, with the board's own word"
    assert set(STAGE_WORDS) == set(Stage)
    assert SETTLED <= STATES


# -- the header ------------------------------------------------------------------------------


def _fed(upto: int, *extra: tuple[str, dict[str, JsonValue]]) -> TuiState:
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= upto:
            feed.apply(frame)
    for seq, (kind, payload) in enumerate(extra, start=1000):
        feed.apply(run_record(kind, payload, seq))
    return feed.state


def test_the_stage_strip_marks_done_current_and_next_and_never_invents_the_plan_size() -> None:
    strip = stage_strip(_fed(30), UNICODE, COLOURS).plain
    assert strip == "● Design ━━ ● Plan ━━  ▶ WORK 0/?  ┈┈ ○ Check ┈┈ ○ Hand off"
    assert stage_strip(TuiState(), ASCII, COLOURS).plain == (
        " > DESIGN  .. o Plan .. o Work .. o Check .. o Hand off"
    )


def test_a_one_task_plan_shows_design_and_planning_skipped() -> None:
    plan: dict[str, JsonValue] = {
        "asked": True, "by": "zeus", "outcome": "approved", "sha": "p" * 64, "version": 1,
    }  # fmt: skip
    strip = stage_strip(_fed(0, ("plan_written", plan)), UNICODE, COLOURS).plain
    # A one-task plan is one task; in older versions its size read `?`.
    assert strip.startswith("⊘ Design ┈┈ ⊘ Plan ┈┈  ▶ WORK 0/1 ")


def test_an_approved_plan_s_listed_tasks_are_its_size_and_all_settled_is_check() -> None:
    tasks: list[JsonValue] = [{"id": "T1", "title": "One"}, {"id": "T2", "title": "Two"}]
    plan: dict[str, JsonValue] = {
        "asked": True, "by": "athena", "sha": "p" * 64, "version": 1, "tasks": tasks,
    }  # fmt: skip
    strip = stage_strip(_fed(0, ("plan_written", plan)), UNICODE, COLOURS).plain
    assert "WORK 0/2" in strip
    # Each listed task delegated and ended (dropped is an end): Athena's final check.
    state = _ended(plan, "T1", "T2")
    assert state.header.stage is Stage.CHECK
    # One still to go is WORK, whatever the delegated ones say.
    assert _ended(plan, "T1").header.stage is Stage.WORK


def _ended(plan: dict[str, JsonValue], *tasks: str) -> TuiState:
    feed = Feed(SID)
    feed.apply(episode(1))
    feed.apply(run_record("plan_written", plan, 1))
    seq = 2
    for task in tasks:
        delegated: dict[str, JsonValue] = {
            "task_id": task, "title": task, "brief_sha": "b" * 64, "by": "athena",
            "start_tree": "t" * 40, "plan_task": task,
        }  # fmt: skip
        reviewed: dict[str, JsonValue] = {
            "task_id": task,
            "call_id": f"r-{task}",
            "answer": "drop",
        }
        feed.apply(run_record("task_delegated", delegated, seq, task_id=task))
        feed.apply(run_record("task_reviewed", reviewed, seq + 1, task_id=task))
        seq += 2
    return feed.state


def _spend(budget: SessionBudget | None, level: SpendLevel = SpendLevel.OK) -> list[str]:
    return [line.plain for line in spend_lines(HeaderSpend(budget, level), UNICODE, COLOURS)]


def test_spend_is_a_dash_until_known_then_the_engine_s_against_its_cap() -> None:
    assert _spend(None) == ["SPEND", "—", "░" * BAR_CELLS]
    known = SessionBudget(spent_usd=0.72, cap_usd=5.0, cap_source="config", cap_seq=None)
    assert _spend(known) == ["SPEND", "$0.72 / $5.00  14%", "███" + "░" * (BAR_CELLS - 3)]
    lifted = SessionBudget(spent_usd=0.72, cap_usd=None, cap_source="session", cap_seq=2)
    assert _spend(lifted) == ["SPEND", "$0.72 / no cap", ""], "D144: lifted, not unknown"
    over = SessionBudget(spent_usd=6.0, cap_usd=5.0, cap_source="config", cap_seq=None)
    assert _spend(over, SpendLevel.CAP)[2] == "█" * BAR_CELLS


def test_the_spend_colour_is_amber_at_a_warning_and_red_at_the_cap() -> None:
    known = SessionBudget(spent_usd=4.0, cap_usd=5.0, cap_source="config", cap_seq=None)
    for level, colour in ((SpendLevel.WARNING, "#ffb000"), (SpendLevel.CAP, "#ff5f6d")):
        line = spend_lines(HeaderSpend(known, level), UNICODE, COLOURS)[1]
        assert str(line.spans[0].style) == colour, level


def test_the_where_line_says_only_what_is_known() -> None:
    feed = Feed(SID)
    assert where_line(feed.state, None) == ""
    feed.apply(workspace("pending", 1))
    shown = where_line(feed.state, "project-chickadee")
    assert shown == "project-chickadee  wt/s-1  branch ccr-1"


# -- the status bar and its keys (the standing rule: dimmed, never hidden) ------------------


def test_every_key_is_drawn_and_an_unbuilt_one_is_dimmed_and_unbound() -> None:
    state = TuiState()
    line = bar_line(state, COLOURS).plain
    for group in BAR:
        for key in group:
            # Every key but F5, drawn only while its BACKGROUND pane shows (the owner's call).
            drawn = f" {key.shown} " in line and key.label in line
            assert drawn is key.drawn(state), key.key
            assert lit(key, state) is (key.action is not None and key.when(state)), key.key
            if key.action is None:
                assert key.builds_in, f"{key.key} names the part that builds it"
    diff = next(k for g in BAR for k in g if k.key == "ctrl+d")
    assert diff.action is not None and not lit(diff, state), "^D waits for a run (D6.5)"
    # Each part that builds a key adds it here (the standing rule): Part 2 focus, Part 4 `/`,
    # Esc and Ctrl+O off the bar, Part 6 `^D` and the zoom, Ctrl+Up and Ctrl+Down, Part 7 the
    # palette `^P`, undo `^U` and help `?`; T-5.28 full screen `^F`; T-5.31 `F4` (CODE) and
    # T-5.30 `F5` (BACKGROUND); T-3.223 `^S` (a waiting stop) and `^G` (pause the team).
    assert {k.key for k in bound()} == {
        "f1", "f2", "f3", "f4", "f5", "slash", "escape", "ctrl+o", "ctrl+d", "ctrl+up",
        "ctrl+down", "ctrl+p", "ctrl+u", "ctrl+f", "question_mark", "ctrl+s", "ctrl+g",
    }  # fmt: skip


def test_a_built_key_that_does_not_apply_now_is_dimmed() -> None:
    from dataclasses import replace

    key = replace(BAR[0][0], when=lambda s: s.episode_open)
    assert not lit(key, TuiState())
    assert lit(key, _fed(1))
