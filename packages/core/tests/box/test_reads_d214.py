"""The worker's reads through its real dispatch (D214 Part 3).

Pinned first, as the 2026-10-20 Nemotron run met them: the worker read with Zeus's 4 KB
cap, so a 10 KB file showed about 96 lines a read; a cut said `(byte_cap) · continue:
read(handle=…, lines=…)`; reading past the end and `max_bytes` above the cap were said nothing
of; and the same read sent twice was resent whole, counted only when worded alike.

Moved with D214, listed in T-3.189's Log:
- the worker reads with its own rows: a 10 KB file shows whole in one read (16 KB cap), and a
  plain `read(path)` returns a file whole at or below 8 KB, its preview above;
- a cut names the next call in the worker's own form, and says what one read shows;
- past the end and `max_bytes` above the cap are each said;
- a re-read of lines held, recent and unchanged, is a pointer and the next unread span, then the
  lines once if asked again, then the repetition bound; a span within a held view, in other
  words, counts as that view asked again.

And the cap is raised only after two restarts past it, for that file, within the ceiling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from box_helpers import WORKER_LIMITS, delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.boxctx.conversation import message_of
from chickadee.schema.journal import BoxMessageP, TaskReportedP


def _code(n: int) -> str:
    """`n` lines of 43 bytes, within the width 2026-10-20's files were measured at."""
    return "".join(f"    value_{i:04d} = compute(arg_{i:04d}, flag=1)\n" for i in range(n))


POSE = _code(212)
"""About 9 KB: what `design/pose.py` was."""

SHORT = _code(93)

SMALL = _code(140)
"""6 KB: at or below the whole-file threshold."""

TEN = _code(240)
"""10 KB: above it."""

BIG = _code(950)
"""40 KB: more than one read shows, and more than the raise ceiling."""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    files = {
        "pose.py": POSE,
        "short.py": SHORT,
        "small.py": SMALL,
        "ten.py": TEN,
        "big.py": BIG,
    }
    return init_repo(tmp_path / "repo", files)


def _tool_texts(journal: object) -> list[str]:
    """What the worker was given back, call by call."""
    lines = journal.projection().lines(BoxMessageP)  # type: ignore[attr-defined]
    return [
        str(message_of(journal.get_blob(p.payload.sha)).content)  # type: ignore[attr-defined]
        for p in lines
        if p.payload.side == "tool"
    ]


def _run(root: Path, *reads: object, limits: dict[str, int] | None = None) -> list[str]:
    script = [*(calls(r) for r in reads), answer("done")]  # type: ignore[arg-type]
    worker, journal, _ = worker_rig(root, script, limits=limits)
    worker.start("T1", "Read it.", delegate(journal, root))
    return _tool_texts(journal)


def _shown(text: str) -> list[int]:
    """The file line numbers a result shows."""
    out: list[int] = []
    for row in text.splitlines():
        head, bar, _ = row.partition("│")
        if bar and head.strip().isdigit():
            out.append(int(head))
    return out


# -- the caps, the cut notes ------------------------------------------------------------------


def test_the_workers_read_shows_a_10_kb_file_in_one_view(root: Path) -> None:
    [got] = _run(root, call("read", "c1", path="pose.py", lines="5-208"))
    assert _shown(got) == list(range(5, 209))
    assert "showing lines" not in got and "byte_cap" not in got


def test_a_plain_read_is_the_whole_file_at_or_below_8_kb_and_its_preview_above(
    root: Path,
) -> None:
    small, ten, full = _run(
        root,
        call("read", "c1", path="small.py"),
        call("read", "c2", path="ten.py"),
        call("read", "c3", path="ten.py", depth="full"),
    )
    assert "[the whole file: all 140 lines fit the preview]" in small
    assert len(_shown(small)) == 140
    assert len(_shown(ten)) < 240
    assert _shown(full) == list(range(1, 241)), "depth full shows it in one view"


def test_a_cut_names_the_next_call_in_the_workers_own_form(root: Path) -> None:
    [got] = _run(root, call("read", "c1", path="big.py", lines="1-950"))
    shown = _shown(got)
    assert shown[0] == 1 and len(shown) < 950
    nxt = f"{shown[-1] + 1}-{min(950, 2 * shown[-1])}"
    want = f"next: read {json.dumps({'path': 'big.py', 'lines': nxt})}"
    assert (
        f"[showing lines 1-{shown[-1]} of 950; one read shows at most 16.0 KB; {want}]" in got
    )
    assert "byte_cap" not in got and "handle=" not in got


def test_past_the_end_and_max_bytes_above_the_cap_are_each_said(root: Path) -> None:
    past, plain, raised = _run(
        root,
        call("read", "c1", path="short.py", lines="80-120"),
        call("read", "c2", path="big.py", lines="5-500"),
        call("read", "c3", path="small.py", lines="1-20", max_bytes=50000),
    )
    assert _shown(past) == list(range(80, 94))
    assert "[end of file: 93 lines; you asked up to 120]" in past
    assert "max_bytes" not in plain
    assert "[max_bytes only lowers the limit; it stays 16.0 KB]" in raised
    assert "raise max_bytes" not in "".join((past, plain, raised))


# -- a re-read of lines held ------------------------------------------------------------------


def test_a_recent_re_read_is_a_pointer_then_the_lines_once_then_the_bound(root: Path) -> None:
    worker, journal, _ = worker_rig(
        root,
        [
            *(calls(call("read", f"c{i}", path="short.py", lines="1-40")) for i in range(1, 5)),
            answer("done"),
        ],
    )
    worker.start("T1", "Read it.", delegate(journal, root))
    first, pointer, insisted, *rest = _tool_texts(journal)
    assert _shown(first) == list(range(1, 41))
    assert _shown(pointer) == []
    assert (
        "you already hold lines 1-40 (" in pointer
        and "from call 1, just now; the next unread lines are 41-80]" in pointer
    )
    assert "repetition{count: 2}" in pointer
    assert _shown(insisted) == list(range(1, 41))
    assert "[shown again, as you asked for them again]" in insisted
    assert "BOUND_REACHED" in rest[0]
    [reported] = journal.projection().lines(TaskReportedP)
    assert reported.payload.stopped == "looped"


def test_a_span_within_a_held_view_is_that_view_asked_again(root: Path) -> None:
    first, within, full = _run(
        root,
        call("read", "c1", path="short.py", lines="1-93"),
        call("read", "c2", path="short.py", lines="2-39"),
        call("read", "c3", path="short.py", depth="full"),
    )
    assert len(_shown(first)) == 93
    assert _shown(within) == [] and "you already hold lines 1-93" in within
    assert "you hold the file to its end, line 93" in within
    assert "repetition{count: 2}" in within
    assert "[shown again, as you asked for them again]" in full
    assert "repetition{count: 3}" in full


def test_a_changed_file_is_resent_and_marked(root: Path) -> None:
    first, _, again = _run(
        root,
        call("read", "c1", path="short.py", lines="1-10"),
        call("edit", "c2", path="short.py", old="value_0000 =", new="value_zero ="),
        call("read", "c3", path="short.py", lines="1-10"),
    )
    assert len(_shown(first)) == len(_shown(again)) == 10
    assert "value_zero" in again
    assert "[changed since your view at call 1]" in again
    assert "repetition" not in again


def test_an_old_view_is_resent_and_counted(root: Path) -> None:
    first, _, _, again = _run(
        root,
        call("read", "c1", path="short.py", lines="1-10"),
        call("glob", "c2", pattern="*.py"),
        call("glob", "c3", pattern="*.md"),
        call("read", "c4", path="short.py", lines="1-10"),
        limits={**WORKER_LIMITS, "worker.held_turns": 1},
    )
    assert _shown(again) == _shown(first)
    assert "[shown again: your view of these lines at call 1 is old]" in again


# -- strikes and the raise --------------------------------------------------------------------


def test_the_2026_10_20_sequence_raises_at_call_6_exactly(root: Path) -> None:
    """The run's order of asks, replayed against a 40 KB file: the outline, a read cut at the
    cap, two spans within what it showed, then two restarts past the cap."""
    texts = _run(
        root,
        call("read", "c1", path="big.py"),
        call("read", "c2", path="big.py", lines="5-500"),
        call("read", "c3", path="big.py", lines="20-60"),
        call("read", "c4", path="big.py", lines="6-200"),
        call("read", "c5", path="big.py", lines="6-499"),
        call("read", "c6", path="big.py", lines="5-500"),
    )
    raised = [i + 1 for i, t in enumerate(texts) if "limit raised for this file" in t]
    assert raised == [6]
    assert "you already hold lines 5-" in texts[2], "within what is held: the pointer"
    assert _shown(texts[3]) == list(range(6, 201)), "asked again right after: the lines"
    assert _shown(texts[5]) == list(range(5, 501)), "the span asked, in one view"
    assert (
        "[limit raised for this file: you asked 2 times for more than one read shows]"
        in (texts[5])
    )


def test_paging_never_strikes(root: Path) -> None:
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-300"),
        call("read", "c2", path="big.py", lines="301-600"),
        call("read", "c3", path="big.py", lines="601-900"),
        call("read", "c4", path="big.py", lines="901-950"),
    )
    assert not any("limit raised" in t for t in texts)


def test_max_bytes_above_the_cap_is_a_strike(root: Path) -> None:
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-100"),
        call("read", "c2", path="big.py", lines="300-800", max_bytes=50000),
        call("read", "c3", path="big.py", lines="1-500"),
    )
    assert "limit raised" not in texts[1], "one strike: a habit, not yet a need"
    assert "limit raised for this file" in texts[2]
    assert _shown(texts[2]) == list(range(1, 501))


def test_one_read_earns_one_strike_however_it_asks(root: Path) -> None:
    """A restart past the cap that also sends `max_bytes` above it is one strike, not two
    : two asks, not two words in one, show the need."""
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-100"),
        call("read", "c2", path="big.py", lines="1-500", max_bytes=50000),
    )
    assert "limit raised" not in texts[1]
    assert len(_shown(texts[1])) < 500


def test_a_read_that_fits_earns_no_strike(root: Path) -> None:
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-100"),
        call("read", "c2", path="big.py", lines="1-50", max_bytes=50000),
        call("read", "c3", path="big.py", lines="60-90", max_bytes=50000),
        call("read", "c4", path="big.py", lines="1-500"),
    )
    assert not any("limit raised" in t for t in texts), "the fitting reads struck nothing"


def test_max_bytes_lowers_the_read_that_earns_the_raise(root: Path) -> None:
    """`max_bytes` only lowers, raised or not: the read that earns the
    raise still shows no more than it asked."""
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-100"),
        call("read", "c2", path="big.py", lines="1-500"),
        call("read", "c3", path="big.py", lines="1-500", max_bytes=2000),
    )
    assert "limit raised for this file" in texts[2]
    assert len("\n".join(r for r in texts[2].splitlines() if "│" in r).encode()) <= 2000


def test_a_span_past_the_ceiling_is_not_raised_and_says_what_to_do(root: Path) -> None:
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-100"),
        call("read", "c2", path="big.py", depth="full"),
        call("read", "c3", path="big.py", lines="1-950"),
    )
    assert not any("limit raised" in t for t in texts)
    assert "[this file is 39.9 KB; one read shows at most 16.0 KB;" in texts[2]
    assert 'read it in parts with "lines", or by symbol with "at"]' in texts[2]


def test_the_raise_is_for_that_file_only(root: Path) -> None:
    texts = _run(
        root,
        call("read", "c1", path="big.py", lines="1-100"),
        call("read", "c2", path="big.py", lines="1-500", max_bytes=50000),
        call("read", "c3", path="big.py", lines="1-500"),
        call("read", "c4", path="ten.py", lines="1-240"),
        call("read", "c5", path="pose.py", lines="1-212"),
    )
    assert "limit raised" in texts[2]
    assert "limit raised" not in texts[3] + texts[4]
