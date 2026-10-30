"""A re-read answered by what the model can still use, and a read cap raised once it is
clearly needed: `context/rereads.py` through the reader, over
each home.

The box roles' home (`ConversationHome`) answers from the conversation it is told of (`Now`):
a view is held while its call is answered there and not evicted, and its age is what came after
it. Zeus's (`SessionHome`) answers from his session, with the same ladder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from context_helpers import Caller, numbered, write_files

from chickadee.client.request import Message
from chickadee.context.homes import ConversationHome, HeldView, Now, SessionHome
from chickadee.context.reader import Reader
from chickadee.context.rereads import (
    FileReads,
    next_unread,
    pointer_note,
    raise_to,
    reread,
    strikes,
)
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import ToolResult
from chickadee.context.session import Session
from chickadee.context.spans import Span
from chickadee.context.terms import Caps, Held, Raise, ReadRules, Vision
from chickadee.schema.adapter import ToolCall

RULES = ReadRules(
    Caps(inline_bytes=2048, max_lines=400, whole_bytes=1024),
    Held(turns=6, tokens=16_000, insist=1),
    Raise(ceiling=8192, share=25, strikes=2),
)


class _Talk:
    """A box role's conversation as its transcript would report it: each read asked by an
    assistant message and answered by a tool message."""

    def __init__(self, key: str = "T1") -> None:
        self.key = key
        self.messages: list[Message] = []
        self.evicted: set[str] = set()
        self._n = 0

    def now(self) -> Now:
        return Now(self.key, tuple(self.messages), frozenset(self.evicted), room=lambda: 90_000)

    def read(self, reader: Reader, path: str, **args: object) -> ToolResult:
        self._n += 1
        cid = f"c{self._n}"
        self.messages.append(
            Message(role="assistant", content="", tool_calls=(ToolCall(id=cid, name="read"),))
        )
        got = reader.read(path, call_id=cid, **args)  # type: ignore[arg-type]
        assert isinstance(got, ToolResult), got
        self.messages.append(Message(role="tool", content=got.render(), tool_call_id=cid))
        return got


def _box(ws: Path, talk: _Talk) -> Reader:
    home = ConversationHome(talk.now)
    return Reader(home, HandleResolver(ws, home.blob), vision=Vision("worker"), rules=RULES)


def _lines(result: ToolResult) -> list[int]:
    return [s for span in result.view for s in range(span.start, span.end + 1)]


# -- the ladder, through the box home ---------------------------------------------------------


def test_an_evicted_view_is_not_held_and_is_resent(ws: Path) -> None:
    write_files(ws, {"a.txt": numbered(60)})
    talk = _Talk()
    reader = _box(ws, talk)
    talk.read(reader, "a.txt", lines="1-20")
    talk.evicted.add("c1")
    again = talk.read(reader, "a.txt", lines="1-20")
    assert _lines(again) == list(range(1, 21))
    assert "you already hold" not in again.render()


def test_a_new_conversation_holds_nothing_of_the_last(ws: Path) -> None:
    write_files(ws, {"a.txt": numbered(60)})
    talk = _Talk("T1")
    reader = _box(ws, talk)
    talk.read(reader, "a.txt", lines="1-20")
    talk.key, talk.messages = "T2", []
    again = talk.read(reader, "a.txt", lines="1-20")
    assert _lines(again) == list(range(1, 21))


def test_a_raise_is_for_its_task_and_gone_at_the_next(ws: Path) -> None:
    write_files(ws, {"big.txt": numbered(400, "x" * 20)})
    talk = _Talk("T1")
    reader = _box(ws, talk)
    talk.read(reader, "big.txt", lines="1-50")
    talk.read(reader, "big.txt", lines="1-200", max_bytes=50_000)
    talk.read(reader, "big.txt", lines="1-200")
    pointer = talk.read(reader, "big.txt", lines="1-200")
    assert "limit raised for this file" in str(talk.messages[-3].content)
    assert "you already hold lines 1-200" in pointer.render(), "shown whole, so held whole"
    talk.key, talk.messages = "T2", []
    fresh = talk.read(reader, "big.txt", lines="1-200")
    assert len(_lines(fresh)) < 200, "the next task reads with the role's own cap again"


def test_a_view_older_than_its_turns_is_shown_again(ws: Path) -> None:
    write_files(ws, {"a.txt": numbered(60)})
    talk = _Talk()
    reader = _box(ws, talk)
    talk.read(reader, "a.txt", lines="1-20")
    talk.messages += [Message(role="assistant", content="thinking")] * 7
    again = talk.read(reader, "a.txt", lines="1-20")
    assert _lines(again) == list(range(1, 21))
    assert "[shown again: your view of these lines at call 1 is old]" in again.render()


def test_a_view_with_more_tokens_after_it_than_its_row_is_shown_again(ws: Path) -> None:
    write_files(ws, {"a.txt": numbered(60)})
    talk = _Talk()
    reader = _box(ws, talk)
    talk.read(reader, "a.txt", lines="1-20")
    talk.messages.append(Message(role="user", content="z" * 70_000))
    again = talk.read(reader, "a.txt", lines="1-20")
    assert "is old]" in again.render()


# -- Zeus's session home ----------------------------------------------------------------------


def test_zeuss_recent_re_read_is_pointed_at_then_shown(session: Session, ws: Path) -> None:
    write_files(ws, {"a.txt": numbered(60)})
    rules = ReadRules(Caps(), Held(turns=6, tokens=16_000, insist=1))
    home = SessionHome(session)
    reader = Reader(home, HandleResolver(ws, session.blob), vision=Vision("zeus"), rules=rules)
    call = Caller(session, reader)
    first = call("read", path="a.txt", lines="1-20")
    pointer = call("read", path="a.txt", lines="1-20")
    shown = call("read", path="a.txt", lines="1-20")
    assert "1│" in first.text
    assert "you already hold lines 1-20" in pointer.text and "from your call c1" in pointer.text
    assert "the next unread lines are 21-40" in pointer.text
    assert "[shown again, as you asked for them again]" in shown.text


# -- the rules, alone -------------------------------------------------------------------------


def _view(spans: Sequence[Span], *, turns: int = 0, tokens: int = 0) -> HeldView:
    return HeldView(tuple(spans), "call 1", 1, turns, tokens)


def test_reread_names_the_newest_view_a_span_falls_within() -> None:
    held = Held(turns=6, tokens=100, insist=1)
    old, new = _view([Span(1, 10)], turns=9), _view([Span(1, 10)])
    assert reread([old, new], [Span(2, 9)], held) == ("recent", new)
    assert reread([old], [Span(2, 9)], held) == ("again", old)
    assert reread([new], [Span(5, 20)], held) == ("show", None)


def test_a_span_held_by_several_views_is_as_old_as_the_oldest() -> None:
    """The review of T-3.189: the newest view alone decided, and pointed at lines the read
    never asked for while the old ones were never shown again."""
    held = Held(turns=6, tokens=100, insist=1)
    old, new = _view([Span(1, 200)], turns=10), _view([Span(201, 210)])
    assert reread([old, new], [Span(1, 210)], held) == (
        "again",
        HeldView((Span(1, 210),), "call 1", 1, 10, 0),
    )
    fresh = _view([Span(1, 200)], turns=1)
    answer, view = reread([fresh, new], [Span(1, 210)], held)
    assert answer == "recent" and view is not None and view.spans == (Span(1, 210),)
    assert reread([old, new], [Span(300, 310)], held) == ("show", None)


def test_the_next_unread_span_skips_what_is_held_and_ends_at_the_file() -> None:
    assert next_unread([Span(1, 10)], Span(1, 10), 100, 10) == Span(11, 20)
    assert next_unread([Span(1, 10), Span(11, 30)], Span(1, 10), 100, 10) == Span(31, 40)
    assert next_unread([Span(1, 100)], Span(1, 100), 100, 10) is None
    note = pointer_note("h_1", _view([Span(1, 100)]), None, 100)
    assert note.endswith("you hold the file to its end, line 100]")


def test_only_a_restart_past_the_cap_strikes() -> None:
    reads = FileReads()
    assert not strikes(reads, [Span(5, 200)], fits=False, held=()), "a first read"
    reads.shown.append((Span(5, 100),))
    assert not strikes(reads, [Span(101, 200)], fits=False, held=()), "a continuation"
    assert not strikes(reads, [Span(5, 50)], fits=False, held=(Span(5, 100),)), "held"
    assert not strikes(reads, [Span(5, 200)], fits=True, held=()), "it fits"
    assert strikes(reads, [Span(5, 200)], fits=False, held=(Span(5, 100),)), "a restart"
    assert strikes(reads, [Span(1, 200)], fits=False, held=(Span(5, 100),)), "from before it"


def test_a_raise_stays_within_the_ceiling_and_the_context_share() -> None:
    rule = Raise(ceiling=10_000, share=25, strikes=2)
    assert raise_to(5_000, rule, None) == 5_000
    assert raise_to(9_500, rule, None) is None, "past the ceiling less its headroom"
    assert raise_to(5_000, rule, 4_000) is None, "past a quarter of what is free"


def test_asking_what_a_read_would_ask_mints_nothing(ws: Path) -> None:
    """The review of T-3.189: `repeat_key` opened the file as a read does, minting a handle the
    model never saw. It looks only."""
    write_files(ws, {"a.txt": numbered(60)})
    talk = _Talk()
    home = ConversationHome(talk.now)
    reader = Reader(home, HandleResolver(ws, home.blob), vision=Vision("worker"), rules=RULES)
    key = reader.repeat_key("a.txt", lines="1-20")
    assert key is not None and key[0] == "a.txt" and key[2] == "1-20"
    assert home.handles().find("file", path="a.txt", run_id=None, sha=key[1]) is None
    talk.read(reader, "a.txt", lines="1-20")
    assert reader.repeat_key("a.txt", lines="2-19") == key, "within the held view: that view"
