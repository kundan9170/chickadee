"""Zeus's transcript as drawn (`ui/` `zeusRun`), from the records
alone, without a terminal.

What these pin: the person's words are on their band; his calls and their results are cards
(`test_tui_cards.py`), so a result of his call is never a line of its own, and one no call of
his made is drawn only expanded; a stop's answer is a rule across the pane; every other record
is the REPL's own line; a command's output keeps the REPL's styling; and the status line says
what he is doing, whom he waits on and what is not yet recorded.
"""

from __future__ import annotations

from tui_frames import SID, assistant, episode, progress, session, stop_asked, user

from chickadee_cli.render.stream import Verbosity
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import ZeusEntry, ZeusKind
from chickadee_cli.tui.panes.zeus import ToolMode, entry_text, status_text, tool_mode
from chickadee_cli.tui.theme import UNICODE, palette
from chickadee_serve.frames import JsonValue

COLOURS = palette("truecolor")


def _calling() -> Feed:
    feed = Feed(SID)
    # No call carries a purpose line: the arguments as sent are all a call line shows.
    call: dict[str, JsonValue] = {
        "id": "c1", "name": "read", "arguments": {"target": "segtree.py"},
    }  # fmt: skip
    turn: dict[str, JsonValue] = {
        "role": "assistant",
        "text": "**Reading** it.",
        "calls": [call],
    }
    feed.apply(session("turn", turn, 1))
    feed.apply(session("tool", {"call_id": "c1", "ran": True, "text": "1 line\nclass Tree"}, 2))
    return feed


def test_the_tool_mode_follows_verbose_and_ctrl_o() -> None:
    assert tool_mode(Verbosity.NORMAL, expanded=False) is ToolMode.COLLAPSED
    assert tool_mode(Verbosity.NORMAL, expanded=True) is ToolMode.EXPANDED
    assert tool_mode(Verbosity.VERBOSE, expanded=False) is ToolMode.EXPANDED
    assert tool_mode(Verbosity.QUIET, expanded=True) is ToolMode.QUIET


def test_a_result_of_his_call_is_in_its_card_never_a_line() -> None:
    feed = _calling()
    result = feed.state.zeus.entries[1]
    for mode in ToolMode:
        assert entry_text(result, mode, Verbosity.NORMAL, COLOURS, feed.state.zeus) is None


def test_a_result_no_call_of_his_made_is_drawn_only_expanded_in_the_repl_s_words() -> None:
    feed = Feed(SID)
    feed.apply(session("tool", {"call_id": "c9", "ran": True, "text": "1 line\nclass Tree"}, 1))
    result, zeus = feed.state.zeus.entries[0], feed.state.zeus
    assert entry_text(result, ToolMode.COLLAPSED, Verbosity.NORMAL, COLOURS, zeus) is None
    shown = entry_text(result, ToolMode.EXPANDED, Verbosity.NORMAL, COLOURS, zeus)
    assert shown is not None and shown.plain.splitlines()[0] == "  ← 1 line"


def test_the_person_s_words_are_drawn_as_said() -> None:
    feed = Feed(SID)
    feed.apply(user("add range sums", 1))
    shown = entry_text(
        feed.state.zeus.entries[0],
        ToolMode.COLLAPSED,
        Verbosity.NORMAL,
        COLOURS,
        feed.state.zeus,
    )
    assert shown is not None and shown.plain == "you  add range sums"


def test_a_stop_s_answer_is_a_rule_across_the_pane() -> None:
    feed = Feed(SID)
    feed.apply(stop_asked("stop-1", "design", 1))
    answered: dict[str, JsonValue] = {
        "id": "stop-1", "choice": "approve", "source": "human", "elapsed_ms": 3,
    }  # fmt: skip
    feed.apply(session("question_answered", answered, 2))
    marker = feed.state.zeus.entries[-1]
    shown = entry_text(marker, ToolMode.COLLAPSED, Verbosity.NORMAL, COLOURS, feed.state.zeus)
    assert shown is not None and shown.plain == "── design approved ──"


def test_every_other_record_is_the_repl_s_own_line() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    shown = entry_text(
        feed.state.zeus.entries[0],
        ToolMode.COLLAPSED,
        Verbosity.NORMAL,
        COLOURS,
        feed.state.zeus,
    )
    assert shown is not None and shown.plain == "  episode 2026-10-10-0123abcd"
    quiet = entry_text(
        feed.state.zeus.entries[0], ToolMode.QUIET, Verbosity.QUIET, COLOURS, feed.state.zeus
    )
    assert quiet is None, "/verbose q shows the person's and the assistant's words only"


def test_a_command_s_output_keeps_the_repl_s_styling() -> None:
    (line,) = Style(tty=True).lines("no command /nope; /help lists them", Tone.BAD)
    entry = ZeusEntry(None, ZeusKind.OUTPUT, text=line)
    shown = entry_text(
        entry, ToolMode.COLLAPSED, Verbosity.NORMAL, COLOURS, Feed(SID).state.zeus
    )
    assert shown is not None and shown.plain == "no command /nope; /help lists them"
    assert any("red" in str(span.style) for span in shown.spans), "Tone.BAD, as the REPL paints"


def test_the_status_says_what_he_does_whom_he_waits_on_and_what_is_queued() -> None:
    feed = Feed(SID)
    feed.apply(progress("thinking"))
    assert status_text(feed.state, UNICODE, COLOURS).plain == "thinking…"
    feed.apply(assistant("on it", 1))
    feed.apply(episode(2))
    assert (
        status_text(feed.state, UNICODE, COLOURS).plain == "holding until the planner reports"
    )
    feed.sent("also the changelog")
    assert status_text(feed.state, UNICODE, COLOURS).plain.splitlines()[-1] == (
        "you  also the changelog  (queued)"
    )
