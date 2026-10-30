"""#543: a coder result's outcome and why travel typed onto its `tool` line, so a reader shows
the reason without parsing the text; every other tool's line is written as before. The report
here is one a stop ended before its run (`delegation.stop_at_dispatch`): a note at stop 1."""

from __future__ import annotations

from chickadee.box.report import CoderReport
from chickadee.context.lines import ToolP, parse, render
from chickadee.context.results import CoderOutcome
from chickadee.tools.tool import Said
from chickadee.zeus.delegate import render_report

REASON = "the person sent a note at stop 1, the design: keep the public API"


def _asked() -> CoderReport:
    return CoderReport.before_run("ASK", REASON)


def test_a_report_ended_at_its_stop_carries_its_outcome_and_why() -> None:
    said = render_report(_asked())
    assert isinstance(said, Said)
    assert said.text.startswith("ASK · run not started")
    assert said.outcome == CoderOutcome("ASK", (REASON,))


def test_the_tool_line_carries_it_and_other_lines_keep_their_bytes() -> None:
    said = render_report(_asked())
    assert isinstance(said, Said)
    coder = render(ToolP("c1", said.text, True, outcome=said.outcome))
    assert parse("tool", coder) == ToolP("c1", said.text, True, outcome=said.outcome)
    other = render(ToolP("r1", "def get(key):", True, outcome=None))
    assert b'"outcome"' in coder and b'"outcome"' not in other
    assert other == (
        b'{"call_id":"r1","handle":null,"image":null,"ran":true,"supersedes":[],'
        b'"text":"def get(key):","view":[]}'
    ), "a line without an outcome keeps the bytes it had before the field existed"


def test_each_outcome_says_what_it_means_for_the_reply() -> None:
    """T-3.159 (a REPL run): Zeus told the user "successfully implemented" on a `PARTIAL` run
    that committed nothing. The report now says what each outcome means, in plain words."""
    from chickadee.zeus import delegate

    meanings = delegate._MEANING  # pyright: ignore[reportPrivateUsage]

    said = render_report(_asked())
    assert isinstance(said, Said)
    assert said.text.splitlines()[1] == f"meaning: {meanings['ASK']}"
    partial = delegate._meaning("PARTIAL", 0, 1)  # pyright: ignore[reportPrivateUsage]
    assert partial.startswith("not finished: 0 of 1 task(s) committed")
    assert "do not call the work done" in partial
    for words in (*meanings.values(), partial):
        assert not [w for w in ("Zeus", "Athena", "worker", "stop 1", "stop 2") if w in words]
